"""
HaNDF: a hand distance field conditioned on a fixed DeepSDF object code.

Token layout (17 total):
    [DIST] + 15 hand tokens + 1 OBJ token

The object is a SINGLE token projected from its 64-d DeepSDF code, so the
transformer attends to a compact object embedding instead of a point cloud --
no permutation variance, and RoPE applies cleanly to all 17 tokens.

The code is fixed at both training and inference, so only one scalar is
predicted: the geodesic distance from the hand pose to that object's grasp
manifold. Training corrupts clean grasps by rotating their joints about random axes.
"""

import torch
import torch.nn as nn

# Rotary position embedding. The 17 tokens are an ordered sequence (1 [DIST] + 15 hand +
# 1 object), so each joint needs a position; RoPE encodes it by rotating q and k rather
# than adding a learned vector, which keeps attention a function of relative position.

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1).unsqueeze(0)


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, pos_emb):
    cos = pos_emb.cos()
    sin = pos_emb.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# A standard pre-norm transformer: self-attention over all tokens, then a position-wise MLP,
# each wrapped in a residual. Stacked `depth` times to form the trunk.

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            pos_emb = self.rope(x, N).unsqueeze(1)  # (1, 1, N, head_dim)
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_dim=1024, dropout=0.0, use_rope=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout, use_rope)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Turning raw inputs into tokens of the model's width: the hand becomes 15 joint tokens,
# one per quaternion, and the object becomes a single token projected from its 64-d DeepSDF code.

class HandTokenizer(nn.Module):
    """15 quaternions -> 15 tokens of dim D."""
    def __init__(self, joint_dim=4, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(joint_dim, output_dim)

    def forward(self, hand_pose):
        # accept (B, 15, 4) or (B, 60)
        if hand_pose.dim() == 2 and hand_pose.shape[-1] == 60:
            hand_pose = hand_pose.reshape(-1, 15, 4)
        return self.proj(hand_pose)


class ObjectTokenizer(nn.Module):
    """Project a single DeepSDF latent (B, latent_dim) -> 1 token (B, 1, D)."""
    def __init__(self, latent_dim=64, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(latent_dim, output_dim)

    def forward(self, obj_latent):
        # (B, latent_dim) -> (B, 1, output_dim)
        if obj_latent.dim() == 1:
            obj_latent = obj_latent.unsqueeze(0)
        return self.proj(obj_latent).unsqueeze(1)



# The distance field itself: (15 joint quaternions, object code) -> one non-negative scalar,
# the geodesic distance from this pose to the object's grasp manifold. The [DIST] token's
# output carries the prediction; softplus keeps it positive.

class DeepSDFCondHaNDF(nn.Module):
    """1 [DIST] token + 15 hand tokens + 1 object token -> one distance (softplus)."""
    def __init__(self, opt):
        super().__init__()
        net_opt = opt.get('DeepSDFCondHaNDF', opt)

        dim       = net_opt.get('dim', 256)
        depth     = net_opt.get('depth', 4)
        num_heads = net_opt.get('heads', 8)
        mlp_dim   = net_opt.get('mlp_dim', 1024)
        dropout   = net_opt.get('dropout', 0.1)
        use_rope  = net_opt.get('use_rope', True)
        latent_dim = net_opt.get('latent_dim', 64)
        model_dim = dim
        self.model_dim = model_dim
        # Positive bias on the head's final Linear so Softplus starts in its linear
        # regime. Without it the network can sit at "predict 0" early in training.
        # Set to 0.0 to disable.
        head_bias_init = float(net_opt.get('head_bias_init', 0.1))
        self.head_bias_init = head_bias_init

        self.dim = dim
        self.latent_dim = latent_dim

        # learnable tokens / type embeddings
        self.dist_token       = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dist_type_embed  = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.hand_type_embed  = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.hand_tokenizer = HandTokenizer(joint_dim=4, output_dim=dim)

        self.obj_type_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.obj_tokenizer = ObjectTokenizer(latent_dim=latent_dim, output_dim=dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(model_dim, num_heads, mlp_dim, dropout, use_rope)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(model_dim)

        # `softplus_beta` controls how sharp the positivity constraint is. A large beta
        # has a dead zone: a sample whose pre-activation starts negative gets almost no
        # gradient, and training can collapse to predicting 0. A smaller beta softens
        # that, which matters when the target distances are small. The released weights
        # use beta = 1.
        softplus_beta = float(net_opt.get('softplus_beta', 100))
        self.hand_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
            nn.Softplus(beta=softplus_beta),
        )
        if head_bias_init != 0.0:
            # Sequential[-2] is the final Linear (before Softplus).
            nn.init.constant_(self.hand_head[-2].bias, head_bias_init)

        self._print_info(net_opt)

    def _print_info(self, opt):
        n = sum(p.numel() for p in self.parameters())
        n_tokens = 17
        print(f"\n{'='*60}\nDeepSDFCondHaNDF\n{'='*60}")
        print(f"  Tokens:     1 [DIST] + 15 hand + 1 obj = {n_tokens}")
        print(f"  Dim:        {self.dim}, depth={opt.get('depth', 4)}, heads={opt.get('heads', 8)}")
        print(f"  Obj latent: dim={self.latent_dim}  [single token, projected]")
        print(f"  RoPE:       on all {n_tokens} tokens")
        print(f"  Output:     one distance, softplus(beta={float(opt.get('softplus_beta', 100)):g})")
        print(f"  Params:     {n:,}\n{'='*60}\n")

    def forward(self, hand_pose, obj_latent):
        """
        hand_pose:  (B, 15, 4) quaternions
        obj_latent: (B, latent_dim) DeepSDF code
        returns hand_dist (B, 1)
        """
        B = hand_pose.shape[0]
        dist_tok = self.dist_token.expand(B, -1, -1) + self.dist_type_embed  # (B, 1, D)
        hand_tok = self.hand_tokenizer(hand_pose) + self.hand_type_embed     # (B, 15, D)
        obj_tok = self.obj_tokenizer(obj_latent) + self.obj_type_embed       # (B, 1, D)

        x = torch.cat([dist_tok, hand_tok, obj_tok], dim=1)                  # (B, 17, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.hand_head(x[:, 0])                                       # (B, 1)
