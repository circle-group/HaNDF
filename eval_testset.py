"""Evaluate HaNDF on the denoising test set in testset/.

Reports dq: the RMS-over-15-joints geodesic distance between the denoised pose and
the ground-truth pose, averaged over the test set, in degrees.

    python eval_testset.py
    python eval_testset.py --norm --step 0.02    # normalised-gradient setting

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
    p.add_argument('--ckpt', default=os.path.join(HERE, 'weights', 'handf_epoch_2000.pth'))
    p.add_argument('--config', default=os.path.join(HERE, 'weights', 'handf_config.yaml'))
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--step', type=float, default=0.1)
    p.add_argument('--norm', dest='norm', action='store_true',
                   help='normalise the Riemannian gradient (pairs with --step 0.02)')
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

    before = float(np.degrees(rms_geo_15(noisy, clean).mean()))
    after = float(np.degrees(rms_geo_15(out, clean).mean()))
    print(f'\ndq_deg: {before:.4f} (noisy) -> {after:.4f} (denoised)')


if __name__ == '__main__':
    main()
