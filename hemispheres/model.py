"""Decoder-only transformer in MLX.

This is the dense baseline: a GPT-style model with pre-norm RMSNorm, rotary
position embeddings, a GELU MLP, and tied input/output embeddings.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_ratio: int = 4
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# Named sizes used by the benchmark and the experiments.
SIZES = {
    "tiny": dict(n_layer=6, n_head=6, d_model=384),       # ~11M non-embedding
    "small": dict(n_layer=8, n_head=8, d_model=512),      # ~25M non-embedding
    "gpt2": dict(n_layer=12, n_head=12, d_model=768),     # ~85M non-embedding (124M w/ GPT-2 vocab)
    "medium": dict(n_layer=24, n_head=16, d_model=1024),  # ~302M non-embedding (350M w/ GPT-2 vocab)
    "large": dict(n_layer=36, n_head=20, d_model=1280),   # ~708M non-embedding (770M w/ GPT-2 vocab)
    # Capacity-curve ladder (HEM-F1): each about double the last, depth never decreasing. Total dense
    # params with the 576-token vocab; the latent arm adds 13-38% for its read layers and key encoder.
    "250k": dict(n_layer=4, n_head=2, d_model=64),        # 0.23M
    "500k": dict(n_layer=4, n_head=3, d_model=96),        # 0.50M
    "1m": dict(n_layer=5, n_head=4, d_model=128),         # 1.06M
    "2m": dict(n_layer=6, n_head=5, d_model=160),         # 1.94M
    "4m": dict(n_layer=6, n_head=7, d_model=224),         # 3.74M
    "8m": dict(n_layer=6, n_head=10, d_model=320),        # 7.56M
    "16m": dict(n_layer=9, n_head=12, d_model=384),       # 16.15M
}
LADDER = ("250k", "500k", "1m", "2m", "4m", "8m", "16m")
# Below the ladder, for finding where the latent core stops passing (HEM-4): the 250k core already passes.
# Latent parameters with the default key_dim 128: 125k 0.17M, 60k 0.07M.
SIZES["125k"] = dict(n_layer=3, n_head=3, d_model=48)  # 0.11M dense
SIZES["60k"] = dict(n_layer=2, n_head=2, d_model=32)   # 0.04M dense


def config_for(size: str, **overrides) -> GPTConfig:
    return GPTConfig(**{**SIZES[size], **overrides})


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, base=cfg.rope_base)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = (t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = self.rope(q), self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask="causal")
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.gelu_approx(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.norm = nn.RMSNorm(cfg.d_model)

    def __call__(self, idx: mx.array) -> mx.array:
        x = self.wte(idx)
        for block in self.blocks:
            x = block(x)
        return self.wte.as_linear(self.norm(x))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(v.size for _, v in tree_flatten(self.parameters()))
        return n - self.wte.weight.size if non_embedding else n


def lm_loss(model: GPT, inputs: mx.array, targets: mx.array) -> mx.array:
    # Upcast logits so the softmax is computed in float32 even when training in bf16.
    logits = model(inputs).astype(mx.float32)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def masked_lm_loss(model: GPT, inputs: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
    """Mean cross-entropy over the target positions where weights == 1."""
    logits = model(inputs).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")
    return (ce * weights).sum() / mx.maximum(weights.sum(), 1.0)


def train_flops_per_token(cfg: GPTConfig, seq_len: int) -> float:
    """Training FLOPs per token, PaLM-style: 6N for the matmuls + 12·L·d·T for attention.

    N counts the non-embedding weights plus the tied LM head (a real matmul);
    the input embedding lookup is free.
    """
    d, L = cfg.d_model, cfg.n_layer
    per_layer = 3 * d * d + d * d + 2 * cfg.mlp_ratio * d * d
    n_matmul = L * per_layer + cfg.vocab_size * d
    return 6 * n_matmul + 12 * L * d * seq_len
