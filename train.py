"""Minimal HaNDF training — the recipe that produced weights/handf_epoch_2000.pth.

One object-conditioned distance field over the 15 MANO finger joints. Each step:

  1. draw a clean grasp and its object's 64-d DeepSDF code
  2. corrupt it by rotating the joints about a random axis through a geodesic angle
     h ~ Exp(0.8) (see `corrupt`); half the batch is
     "partial" -- the noise is supported on 1-4 whole fingers, the rest stay exact
  3. label = exact geodesic distance to the nearest clean pose OF THE SAME OBJECT
     (the sample's own clean pose is a candidate), searched exhaustively
  4. loss = L1(predicted distance, label)

    python train.py                        # full recipe: 2000 epochs
    python train.py --max-epoch 2 --epoch-size 8192 --batch-size 512   # smoke test

Defaults mirror weights/handf_config.yaml. Data: data/poses_contact_v2 (train split)
and the object codes in weights/object_codes.
"""
import argparse
import glob
import os

import numpy as np
import torch

from handf.handf_deepsdf import DeepSDFCondHaNDF
from handf_infer import HERE, aa_to_quat

N_HAND_JOINTS = 15
FINGER_JOINTS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
CATEGORIES = ['Bottle', 'Bowl', 'Kettle', 'Knife', 'Mug', 'ToyCar']


def hamilton(a, b):
    """(..., 4) x (..., 4) -> (..., 4) quaternion product, (w, x, y, z)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def load_objects(pose_root, codes_dir):
    """[{key, poses (P,15,4), code (64,)}] for every training object with a code."""
    objects = []
    for cat in CATEGORIES:
        for f in sorted(glob.glob(os.path.join(
                pose_root, 'SAMPLED_POSES_MULTI_OBJECT', f'{cat}_train', 'N*.npz'))):
            oid = os.path.basename(f)[:-4]                        # 'N11'
            code_path = os.path.join(codes_dir, f'{cat}__{int(oid.lstrip("N")):03d}.pt')
            if not os.path.exists(code_path):
                raise FileNotFoundError(f'{cat} {oid} has poses but no object code at {code_path}')
            code = torch.load(code_path, map_location='cpu', weights_only=False)['latent']
            objects.append({'key': f'{cat}__{oid}',
                            'poses': aa_to_quat(np.load(f)['pose_body']).astype(np.float32),
                            'code': code.detach().view(-1).float().numpy()})
    if not objects:
        raise SystemExit(f'no training poses under {pose_root}')
    n = [len(o['poses']) for o in objects]
    print(f'{len(objects)} objects, {sum(n):,} clean poses ({min(n)}-{max(n)} each)')
    return objects


def corrupt(clean, sigma, rng, partial):
    """clean (15,4) -> noisy (15,4). `partial` restricts the noise to 1-4 whole fingers."""
    v = np.zeros((N_HAND_JOINTS, 3), dtype=np.float32)
    if partial:
        k = rng.integers(1, 5)                                    # 1..4 fingers, >=1 intact
        jidx = [j for p in rng.choice(5, size=k, replace=False) for j in FINGER_JOINTS[p]]
        v[jidx] = rng.standard_normal((len(jidx), 3)).astype(np.float32)
    else:
        v = rng.standard_normal((N_HAND_JOINTS, 3)).astype(np.float32)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return clean.copy()
    v = v / n
    theta = np.float32(rng.exponential(scale=sigma)) * np.linalg.norm(v, axis=-1, keepdims=True)
    vn = np.linalg.norm(v, axis=-1, keepdims=True)
    nq = np.zeros((N_HAND_JOINTS, 4), dtype=np.float32)
    nq[:, 0] = np.cos(theta).squeeze(-1)
    nq[:, 1:] = np.sin(theta) * np.where(vn > 1e-8, v / np.maximum(vn, 1e-12), 0.0)
    return hamilton(nq, clean).astype(np.float32)


def object_nn(q, pool, org):
    """Distance to the nearest clean pose of this object, under the mean per-joint geodesic.

    q (B,15,4), pool (P,15,4), org (B,15,4) the samples' own clean poses (extra candidate).
    -> distances (B,)
    """
    d = np.arccos(np.clip(np.abs(np.einsum('bjk,pjk->bpj', q, pool)), -1.0, 1.0)).mean(-1)
    best_d = d.min(1)
    do = np.arccos(np.clip(np.abs((q * org).sum(-1)), -1.0, 1.0)).mean(-1)
    return np.minimum(best_d, do).astype(np.float32)


def make_batch(objects, size, sigma, partial_frac, rng):
    """(noisy (B,15,4), code (B,64), label (B,)) — labels searched per object."""
    oi = rng.integers(0, len(objects), size=size)
    noisy = np.empty((size, N_HAND_JOINTS, 4), dtype=np.float32)
    org = np.empty_like(noisy)
    code = np.empty((size, 64), dtype=np.float32)
    for b, k in enumerate(oi):
        o = objects[k]
        org[b] = o['poses'][rng.integers(0, len(o['poses']))]
        noisy[b] = corrupt(org[b], sigma, rng, partial=rng.random() < partial_frac)
        code[b] = o['code']
    label = np.empty(size, dtype=np.float32)
    for k in np.unique(oi):                                        # one search per object
        m = oi == k
        label[m] = object_nn(noisy[m], objects[k]['poses'], org[m])
    return noisy, code, label



def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--poses', default=os.path.join(HERE, 'data', 'poses_contact_v2'))
    p.add_argument('--codes', default=os.path.join(HERE, 'weights', 'object_codes'))
    p.add_argument('--out', default=os.path.join(HERE, 'runs', 'handf'))
    p.add_argument('--noise-level', type=float, default=0.8)
    p.add_argument('--partial-frac', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--epoch-size', type=int, default=100000)
    p.add_argument('--max-epoch', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if not 0.0 <= a.partial_frac <= 1.0:
        raise ValueError(f'--partial-frac must be in [0, 1], got {a.partial_frac}')

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)

    objects = load_objects(a.poses, a.codes)
    model = DeepSDFCondHaNDF({'DeepSDFCondHaNDF': dict(
        latent_dim=64, dim=256, depth=4, heads=8, mlp_dim=1024,
        dropout=0.1, use_rope=True, softplus_beta=1)}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epoch, eta_min=1e-6)
    steps = max(a.epoch_size // a.batch_size, 1)
    print(f'{sum(p_.numel() for p_ in model.parameters()):,} params | '
          f'{a.max_epoch} epochs x {steps} steps x batch {a.batch_size} | seed {a.seed}')

    for ep in range(a.max_epoch):
        model.train()
        tot = 0.0
        for _ in range(steps):
            noisy, code, label = make_batch(
                objects, a.batch_size, a.noise_level, a.partial_frac, rng)
            th = torch.from_numpy(noisy).to(dev)
            ol = torch.from_numpy(code).to(dev)
            gt = torch.from_numpy(label).to(dev)

            pred = model(th, ol).squeeze(-1)
            loss = torch.nn.functional.l1_loss(pred, gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        print(f'epoch {ep:4d}  loss {tot / steps:.5f}  lr {sched.get_last_lr()[0]:.2e}', flush=True)

        if (ep + 1) % a.save_every == 0 or ep + 1 == a.max_epoch:
            path = os.path.join(a.out, f'epoch_{ep + 1}.pth')
            torch.save({'epoch': ep, 'state_dict': model.state_dict()}, path)
            print(f'  saved {path}', flush=True)


if __name__ == '__main__':
    main()
