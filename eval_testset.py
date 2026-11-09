"""Evaluate HaNDF on the denoising test set in testset/.

Reports, between the denoised pose and the ground-truth pose, averaged over the test set:

    dq    RMS-over-15-joints geodesic distance, in degrees
    m2m   distance between the two MANO hands, in mm

    python eval_testset.py
    python eval_testset.py --norm --step 0.02    # normalised-gradient setting

m2m needs MANO, which its authors do not permit us to redistribute. Install manopth and
put MANO_RIGHT.pkl where --mano-root points, or set MANO_PATH. Without it dq is still
reported and m2m is skipped.

Test set   testset/testset_meta.npz (clean_q, categories, object_ids)
           testset/testset_noise_<sigma>.npz (noisy_q)
Object     one 64-d code per test object, from weights/object_codes/<Cat>__<nnn>.pt
"""
import argparse
import os

import numpy as np
import torch

from handf_infer import HERE, load_handf, project


def rms_geo_15(q1, q2):
    """(N,15,4) x (N,15,4) -> (N,) RMS geodesic over the 15 joints, radians."""
    dot = np.clip(np.abs((q1 * q2).sum(-1)), -1.0, 1.0)
    return np.sqrt((np.arccos(dot) ** 2).mean(-1))


def quaternion_to_axis_angle(q):
    """(..., 4) unit quaternions (w, x, y, z) -> (..., 3) axis-angle."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    v, w = q[..., 1:], q[..., :1]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(s > 1e-8, v * (2 * np.arctan2(s, w) / np.maximum(s, 1e-12)), 0.0)


def load_mano(mano_root, device):
    """The MANO layer used for m2m, or None with the reason printed."""
    try:
        from manopth.manolayer import ManoLayer
    except ImportError:
        print('  m2m skipped: manopth is not installed')
        return None
    if not os.path.exists(os.path.join(mano_root, 'MANO_RIGHT.pkl')):
        print(f'  m2m skipped: no MANO_RIGHT.pkl under {mano_root}')
        return None
    return ManoLayer(mano_root=mano_root, use_pca=False, ncomps=45,
                     flat_hand_mean=True).to(device)


def m2m_mm(layer, q_a, q_b, device, batch_size=128):
    """(N,15,4) x (N,15,4) -> (N,) distance between the two MANO hands, mm.

    The wrist is held at zero, so only the finger articulation is compared.
    """
    aa_a = torch.from_numpy(quaternion_to_axis_angle(q_a).reshape(len(q_a), 45)).float()
    aa_b = torch.from_numpy(quaternion_to_axis_angle(q_b).reshape(len(q_b), 45)).float()
    out = np.zeros(len(q_a), dtype=np.float32)
    for s in range(0, len(q_a), batch_size):
        e = min(s + batch_size, len(q_a))
        wrist = torch.zeros(e - s, 3, device=device)
        with torch.no_grad():
            v_a = layer(torch.cat([wrist, aa_a[s:e].to(device)], -1))[0]
            v_b = layer(torch.cat([wrist, aa_b[s:e].to(device)], -1))[0]
            out[s:e] = (v_a - v_b).norm(dim=-1).mean(dim=-1).cpu().numpy()
    return out


def load_codes(meta, codes_dir):
    """One (N,64) array: each sample's object code, looked up by (category, object_id)."""
    cats, oids = meta['categories'], meta['object_ids']
    cache, out = {}, np.zeros((len(cats), 64), dtype=np.float32)
    for i, (c, o) in enumerate(zip(cats.tolist(), oids.tolist())):
        if (c, o) not in cache:
            p = os.path.join(codes_dir, f'{c}__{int(o.lstrip("N")):03d}.pt')
            if not os.path.exists(p):
                raise FileNotFoundError(f'no object code for {c} {o}: expected {p}')
            z = torch.load(p, map_location='cpu', weights_only=False)['latent']
            cache[(c, o)] = z.detach().view(-1).float().numpy()
        if cache[(c, o)].shape != (64,):
            raise ValueError(f'{c} {o}: object code must be 64-d, got {cache[(c, o)].shape}')
        out[i] = cache[(c, o)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--testset', default=os.path.join(HERE, 'testset'))
    p.add_argument('--sigma', type=float, default=0.8)
    p.add_argument('--codes', default=os.path.join(HERE, 'weights', 'object_codes'))
    p.add_argument('--ckpt', default=os.path.join(HERE, 'weights', 'handf.pth'))
    p.add_argument('--config', default=os.path.join(HERE, 'weights', 'handf_config.yaml'))
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--step', type=float, default=2.0)
    p.add_argument('--norm', dest='norm', action='store_true',
                   help='normalise the Riemannian gradient (pairs with --step 0.02)')
    p.add_argument('--mano-root', default=os.environ.get(
        'MANO_PATH', os.path.join(HERE, 'mano', 'models')),
        help='directory holding MANO_RIGHT.pkl, for m2m')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()

    meta = np.load(os.path.join(a.testset, 'testset_meta.npz'), allow_pickle=True)
    noisy = np.load(os.path.join(a.testset, f'testset_noise_{a.sigma:.3f}.npz'))['noisy_q']
    clean = meta['clean_q'].astype(np.float32)
    if noisy.shape != clean.shape:
        raise ValueError(f'noisy {noisy.shape} and clean {clean.shape} disagree')
    codes = load_codes(meta, a.codes)

    dev = torch.device(a.device)
    model = load_handf(dev, ckpt=a.ckpt, config=a.config)
    print(f'{len(clean)} poses | sigma {a.sigma} | {a.iterations} iters | step {a.step} | '
          f'norm_grad {a.norm}')

    out = []
    for s in range(0, len(clean), a.batch_size):
        e = min(s + a.batch_size, len(clean))
        out.append(project(model, noisy[s:e], codes[s:e], dev,
                           iterations=a.iterations, step_size=a.step,
                           norm_grad=a.norm).numpy())
        print(f'  {e}/{len(clean)}', end='\r', flush=True)
    out = np.concatenate(out).astype(np.float32)

    ok = np.isfinite(out).reshape(len(out), -1).all(-1)
    if not ok.all():
        raise SystemExit(f'{(~ok).sum()}/{len(out)} denoised poses are NaN/Inf')

    print()
    before = float(np.degrees(rms_geo_15(noisy, clean).mean()))
    after = float(np.degrees(rms_geo_15(out, clean).mean()))
    print(f'dq  : {before:.4f} -> {after:.4f} deg')

    layer = load_mano(a.mano_root, dev)
    if layer is not None:
        b = float(m2m_mm(layer, noisy, clean, dev).mean())
        c = float(m2m_mm(layer, out, clean, dev).mean())
        print(f'm2m : {b:.4f} -> {c:.4f} mm')


if __name__ == '__main__':
    main()
