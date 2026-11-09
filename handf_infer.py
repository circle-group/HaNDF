"""HaNDF inference: distance query and projection of a hand pose onto the grasp manifold of an object.

Weights  weights/handf.pth + weights/handf_config.yaml
Object   one 64-d DeepSDF code: weights/object_codes/<name>.pt (HOI4D objects)
Pose     15 MANO finger joints as unit quaternions (w, x, y, z), shape (B, 15, 4), MANO
         flat_hand_mean=True convention. MANO axis-angle (B, 45) or (B, 48) -> aa_to_quat.

Projection runs 500 Riemannian steps at step size 2.0 with an un-normalized gradient.

    python handf_infer.py --code weights/object_codes/Mug__046.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.autograd import grad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handf.handf_deepsdf import DeepSDFCondHaNDF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_HAND_JOINTS = 15


# MANO axis-angle <-> unit quaternions: 15 finger joints, (w, x, y, z).
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def aa_to_quat(pose):
    """MANO axis-angle (N, 48) with wrist [dropped] or (N, 45) -> (N, 15, 4) float32 quaternions."""
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (45, 48):
        raise ValueError(f'expected MANO axis-angle (N, 45) or (N, 48), got {arr.shape}')
    if arr.shape[1] == 48:
        arr = arr.reshape(-1, 16, 3)[:, 1:, :]
    else:
        arr = arr.reshape(-1, 15, 3)
    return axis_angle_to_quaternion(torch.from_numpy(arr.copy())).numpy().astype(np.float32)


# Riemannian descent on (S^3)^15. A unit quaternion may only move tangentially, so the
# Euclidean gradient is projected onto the tangent space of each joint's sphere and applied
# by exponential map, keeping every joint a valid rotation without renormalising.
def quaternion_hamilton_product(a, b):

    ndim = len(a.shape)

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    res = torch.concat((
        (aw * bw - ax * bx - ay * by - az * bz).unsqueeze(-1),
        (aw * bx + ax * bw + ay * bz - az * by).unsqueeze(-1),
        (aw * by - ax * bz + ay * bw + az * bx).unsqueeze(-1),
        (aw * bz + ax * by - ay * bx + az * bw).unsqueeze(-1)
    ), dim=ndim-1)

    return res


def exp_map(q, rgrad, device='cuda'):
    alpha = torch.norm(rgrad[:, :, 1:], dim=-1)
    tmp = torch.sin(alpha) / alpha
    beta = torch.ones_like(alpha).to(device)
    mask = (alpha > 0.)
    beta[mask] = tmp[mask]

    exp_v = torch.ones_like(rgrad).to(device)
    exp_v[:, :, 0] = torch.cos(alpha)
    exp_v[:, :, 1:] = torch.einsum('...d, ...->...d', rgrad[:, :, 1:], beta)

    prod = quaternion_hamilton_product(q, exp_v)
    prod = prod / torch.norm(prod, dim=-1).unsqueeze(-1)
    mask = (prod[:, :, 0] < 0)
    prod[mask] *= -1
    return prod


def egrad2rgrad(egrad, q, device='cuda', norm=True):
    bs, n_joints, _ = q.shape

    Id = torch.eye(4).to(device)
    Id = Id.expand(bs, n_joints, 4, 4)
    P = Id - torch.einsum('...ij, ...jk -> ...ik', q.unsqueeze(-1), q.unsqueeze(2))  # (bs, nj, 4, 4)

    # project egrad to the tangent of q -> v
    v = torch.einsum('...ij, ...jk -> ...ik', P, egrad.unsqueeze(-1))  # (bs, nj, 4, 1)
    if norm:
        v = torch.nn.functional.normalize(v, dim=2)  # unit gradient length

    # unit quaternion constraint
    rmat = torch.eye(4)  # (bs, nj, 4, 4)
    rmat = rmat.expand(bs, n_joints, 4, 4).to(device)

    rmat[:, :, 0, 0] = 0.
    rmat[:, :, 1, 0] = -q[:, :, 1] / (1 + q[:, :, 0])
    rmat[:, :, 2, 0] = -q[:, :, 2] / (1 + q[:, :, 0])
    rmat[:, :, 3, 0] = -q[:, :, 3] / (1 + q[:, :, 0])

    rgrad = torch.einsum('...ij, ...jk -> ...ik', rmat, v)
    rgrad = rgrad.squeeze(-1)  # [bs, nj, 4]

    return rgrad


def rdfgrad(egrad, q, dist, step_size, device='cuda', norm=True):
    rgrad = egrad2rgrad(egrad, q, device=device, norm=norm)  # [bs, nj, 4]
    rgrad = -step_size * dist * rgrad
    res = exp_map(q, rgrad, device=device)
    return res


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


# Loading the trained field, an object's code, and validating what callers pass in.
def load_handf(device, ckpt=os.path.join(HERE, 'weights', 'handf.pth'),
               config=os.path.join(HERE, 'weights', 'handf_config.yaml')):
    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = DeepSDFCondHaNDF(cfg['model']).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    print(f'Loaded {os.path.basename(ckpt)}')
    return model


def load_code(path):
    """weights/object_codes/<name>.pt -> (64,) float32."""
    e = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(e, dict) or 'latent' not in e:
        raise ValueError(f'{path}: expected a dict with key "latent"')
    z = e['latent'].detach().view(-1).float()
    if z.shape != (64,) or not torch.isfinite(z).all():
        raise ValueError(f'{path}: code shape {tuple(z.shape)}, expected finite (64,)')
    return z


def _check(hand, code):
    if hand.ndim != 3 or hand.shape[1:] != (N_HAND_JOINTS, 4):
        raise ValueError(f'hand pose must be (B, 15, 4) quaternions, got {tuple(hand.shape)}')
    if code.ndim != 2 or code.shape != (hand.shape[0], 64):
        raise ValueError(f'object code must be (B, 64) with B = {hand.shape[0]}, got {tuple(code.shape)}')
    n = hand.norm(dim=-1)
    if (n - 1).abs().max() > 1e-3:
        raise ValueError(f'hand quaternions are not unit (norm range {n.min():.4f}-{n.max():.4f})')


# The two things this field is for: score a pose, or move it onto the object's grasp manifold.
@torch.no_grad()
def distance(model, hand, code, device):
    """(B, 15, 4) poses, (B, 64) codes -> (B,) predicted distance to the object's grasp manifold (rad)."""
    hand, code = torch.as_tensor(hand).float(), torch.as_tensor(code).float()
    _check(hand, code)
    return model(hand.to(device), code.to(device)).view(-1).cpu()


def project(model, hand, code, device, iterations=500, step_size=2.0, norm_grad=False, mask=None):
    """Riemannian descent of the predicted distance.

    hand (B, 15, 4), code (B, 64); mask (B, 15) bool = joints to update (default: all; frozen joints
    keep their input rotation, e.g. for completing a partially observed hand). Returns (B, 15, 4).
    """
    hand, code = torch.as_tensor(hand).float(), torch.as_tensor(code).float()
    _check(hand, code)
    if mask is None:
        mask = torch.ones(hand.shape[0], N_HAND_JOINTS, dtype=torch.bool)
    mask = torch.as_tensor(mask)
    if mask.shape != (hand.shape[0], N_HAND_JOINTS) or mask.dtype != torch.bool:
        raise ValueError(f'mask must be bool (B, 15), got {mask.dtype} {tuple(mask.shape)}')
    cur = hand.clone().float().to(device)
    obj = code.clone().float().to(device)
    m = mask.to(device).float().unsqueeze(-1)              # (B, 15, 1)

    for step in range(iterations):
        cur = cur.detach().requires_grad_(True)
        with torch.enable_grad():
            d = model(cur, obj)                             # (B, 1)
            g = gradient(cur, d).reshape(-1, N_HAND_JOINTS, 4)

        with torch.no_grad():
            updated = rdfgrad(g, cur,
                              d.view(-1, 1, 1).expand(-1, N_HAND_JOINTS, 1),
                              step_size, device=device, norm=norm_grad)
            cur = (m * updated + (1 - m) * cur).detach()
    return cur.detach().cpu()


def main():
    ap = argparse.ArgumentParser(description='demo: project noisy flat hands onto the grasp manifold of one object')
    ap.add_argument('--code', required=True, help='object code .pt (weights/object_codes/<name>.pt)')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--noise', type=float, default=0.3, help='per-joint random rotation angle (rad)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iterations', type=int, default=500, help='projection steps')
    ap.add_argument('--step', type=float, default=2.0, help='projection step size')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dev = torch.device(args.device)
    model = load_handf(dev)
    z = load_code(args.code)
    rng = np.random.default_rng(args.seed)
    axis = rng.normal(size=(args.n, 15, 3))
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    hand = aa_to_quat((axis * args.noise).reshape(args.n, 45))
    code = z.unsqueeze(0).expand(args.n, -1)
    d0 = distance(model, hand, code, dev)
    out = project(model, hand, code, dev, iterations=args.iterations, step_size=args.step)
    d1 = distance(model, out, code, dev)
    print(f'{os.path.basename(args.code)}: predicted distance before {d0.mean():.4f} -> after {d1.mean():.4f} rad '
          f'(mean of {args.n} poses)')


if __name__ == '__main__':
    main()
