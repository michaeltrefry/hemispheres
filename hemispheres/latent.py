"""Latent-store reasoner: lookups happen inside the forward pass.

The model is the dense GPT plus read layers, placed after evenly spaced blocks
(after blocks 2, 4 and 6 of 8 by default). At every position, read layer i:

  1. forms a query q = W_q · norm(h);
  2. retrieves the top-k store entries by q · key / sqrt(d_k). Keys come from a
     key encoder over the subject's name tokens and the relation; they depend
     only on surface names, so a store for any world can be encoded with a
     forward pass;
  3. cross-attends from h over the retrieved objects' name tokens (embedded
     with the shared token embedding plus a within-name position embedding)
     and a learned null slot. Each entry's retrieval score is added to its
     attention logits, so selection is learned end to end.

Read layer i is meant to perform hop i. Optional hop supervision (an InfoNCE
loss over the whole store) trains the query of read layer i, at the position
before a fact is needed, toward that fact's key. See research/latent-arm-design.md.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .model import Block, GPTConfig
from .store import MAX_NAME_LEN, RELATION_INDEX


@dataclass
class LatentConfig(GPTConfig):
    n_reads: int = 3
    top_k: int = 4
    key_dim: int = 128
    retrieval_chunk: int = 2048  # positions scored against the whole store at once
    retrieval_group: int = 256   # store entries per group in two-stage top-k

    @property
    def read_after(self) -> list[int]:
        """Number of blocks run before each read layer."""
        return [round(self.n_layer * (i + 1) / (self.n_reads + 1)) for i in range(self.n_reads)]


def top_k_indices(scores: mx.array, k: int, group: int) -> mx.array:
    """Column indices (P, k) of the k largest scores in each row of (P, N), in no particular order.

    Two stages, exact up to ties: split each row into groups of `group`, keep
    the k groups with the largest maxima, then take the top k of those k·group
    scores. Every top-k entry lies in one of those groups (a group holding one
    would otherwise be beaten by k groups each holding a larger score). This
    replaces a partition over N columns with a max-reduction plus two small
    partitions.
    """
    P, N = scores.shape
    n_groups = -(-N // group)
    if n_groups <= k:
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    if n_groups * group != N:
        scores = mx.pad(scores, [(0, 0), (0, n_groups * group - N)], constant_values=-float("inf"))
    grouped = scores.reshape(P, n_groups, group)
    top_groups = mx.argpartition(-grouped.max(axis=-1), kth=k - 1, axis=-1)[:, :k]            # (P, k)
    candidates = mx.take_along_axis(grouped, top_groups[:, :, None], axis=1).reshape(P, k * group)
    local = mx.argpartition(-candidates, kth=k - 1, axis=-1)[:, :k]                            # (P, k)
    return mx.take_along_axis(top_groups, local // group, axis=1) * group + local % group


class KeyEncoder(nn.Module):
    """key(entry) = MLP(mean(tok_emb(subject) + pos_emb) + rel_emb)."""

    def __init__(self, cfg: LatentConfig):
        super().__init__()
        self.pos = nn.Embedding(MAX_NAME_LEN, cfg.d_model)
        self.rel = nn.Embedding(len(RELATION_INDEX), cfg.d_model)
        self.norm = nn.RMSNorm(cfg.d_model)
        self.fc = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.key_dim, bias=False)

    def __call__(self, wte: nn.Embedding, store: dict) -> mx.array:
        e = wte(store["subj"]) + self.pos.weight
        m = store["subj_mask"][..., None]
        pooled = (e * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1.0)  # padding entries have no tokens
        return self.out(nn.gelu_approx(self.fc(self.norm(pooled + self.rel(store["rel"])))))


class ReadLayer(nn.Module):
    def __init__(self, cfg: LatentConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.norm = nn.RMSNorm(d)
        self.query = nn.Linear(d, cfg.key_dim, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.score_scale = mx.array(1.0)
        self.null_logit = mx.zeros((cfg.n_head,))
        self.null_value = mx.zeros((cfg.n_head, cfg.head_dim))

    def retrieve(self, q: mx.array, keys: mx.array, bias: mx.array | None = None) -> mx.array:
        """Indices (B, T, k) of the exact top-k entries for each position; no gradient.

        `bias` (N,) is added to every position's scores (-1e9 excludes padding entries).
        """
        B, T, dk = q.shape
        flat, kt = mx.stop_gradient(q).reshape(-1, dk), mx.stop_gradient(keys).T
        bias = 0.0 if bias is None else bias
        chunks = [top_k_indices(flat[c:c + self.cfg.retrieval_chunk] @ kt + bias, self.cfg.top_k,
                                self.cfg.retrieval_group)
                  for c in range(0, flat.shape[0], self.cfg.retrieval_chunk)]
        return mx.concatenate(chunks, axis=0).reshape(B, T, self.cfg.top_k)

    def __call__(self, h: mx.array, keys: mx.array, store: dict, wte: nn.Embedding,
                 val_pos: nn.Embedding) -> tuple[mx.array, mx.array]:
        """Cross-attention over the retrieved name tokens.

        A retrieved token's embedding is wte[id] + pos[l], and the key/value
        projections are linear. So instead of projecting every retrieved token
        at every position, project the vocabulary and position tables once,
        score queries against the whole projected vocabulary (V is small), and
        gather the retrieved entries' scores and values by token id.
        """
        B, T, D = h.shape
        H, dh, k, L = self.n_head, self.head_dim, self.cfg.top_k, MAX_NAME_LEN
        x = self.norm(h)
        query = self.query(x)                                            # (B, T, dk)
        idx = self.retrieve(query, keys, store.get("bias"))              # (B, T, k)
        score = (query[:, :, None, :] * keys[idx]).sum(-1) / math.sqrt(self.cfg.key_dim)  # (B, T, k)

        ids = store["val"][idx].reshape(B, T, k * L)                     # token ids of retrieved names
        valid = store["val_mask"][idx].reshape(B, T, k * L)
        bias = mx.repeat(self.score_scale * score, L, axis=-1)           # (B, T, k*L)
        bias = mx.where(valid > 0, bias, -1e9)

        qh = self.q(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)        # (B, H, T, dh)
        k_tok = self.k(wte.weight).reshape(-1, H, dh).transpose(1, 2, 0)  # (H, dh, V)
        k_pos = self.k(val_pos.weight).reshape(L, H, dh).transpose(1, 2, 0)  # (H, dh, L)
        qk_tok = (qh @ k_tok).transpose(0, 2, 1, 3)                      # (B, T, H, V)
        qk_pos = (qh @ k_pos).transpose(0, 2, 1, 3)                      # (B, T, H, L)
        logits = mx.take_along_axis(qk_tok, mx.broadcast_to(ids[:, :, None, :], (B, T, H, k * L)), axis=-1)
        logits = (logits + mx.tile(qk_pos, (1, 1, 1, k))) / math.sqrt(dh) + bias[:, :, None, :]  # (B, T, H, kL)

        null = mx.broadcast_to(self.null_logit.reshape(1, 1, H, 1), (B, T, H, 1))
        attn = mx.softmax(mx.concatenate([logits, null], axis=-1).astype(mx.float32), axis=-1).astype(h.dtype)
        v_tok, v_pos = self.v(wte.weight), self.v(val_pos.weight)        # (V, D), (L, D)
        vh = (v_tok[ids] + mx.tile(v_pos, (k, 1))).reshape(B, T, k * L, H, dh).transpose(0, 1, 3, 2, 4)
        out = (attn[..., None, :-1] @ vh)[..., 0, :]                     # (B, T, H, dh)
        out = out + attn[..., -1:] * self.null_value
        return h + self.o(out.reshape(B, T, D)), query


class LatentGPT(nn.Module):
    def __init__(self, cfg: LatentConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.norm = nn.RMSNorm(cfg.d_model)
        self.reads = [ReadLayer(cfg) for _ in range(cfg.n_reads)]
        self.key_encoder = KeyEncoder(cfg)
        self.val_pos = nn.Embedding(MAX_NAME_LEN, cfg.d_model)

    def encode_keys(self, store: dict) -> mx.array:
        return self.key_encoder(self.wte, store)

    def forward(self, idx: mx.array, store: dict) -> tuple[mx.array, list[mx.array], mx.array]:
        """(logits, per-read-layer queries, store keys). A store carrying "keys" (computed once with
        `encode_keys`, possibly compressed) uses them instead of re-encoding every entry."""
        keys = store["keys"] if "keys" in store else self.encode_keys(store)
        x, queries, reads = self.wte(idx), [], iter(zip(self.cfg.read_after, self.reads))
        after, read = next(reads, (None, None))
        for i, block in enumerate(self.blocks):
            x = block(x)
            while after == i + 1:
                x, q = read(x, keys, store, self.wte, self.val_pos)
                queries.append(q)
                after, read = next(reads, (None, None))
        return self.wte.as_linear(self.norm(x)), queries, keys

    def __call__(self, idx: mx.array, store: dict) -> mx.array:
        return self.forward(idx, store)[0]

    def num_params(self, non_embedding: bool = False) -> int:
        from mlx.utils import tree_flatten
        n = sum(v.size for _, v in tree_flatten(self.parameters()))
        return n - self.wte.weight.size if non_embedding else n


def latent_loss(model: LatentGPT, store: dict, inputs: mx.array, targets: mx.array, weights: mx.array,
                sup_idx: mx.array, sup_fact: mx.array, sup_valid: mx.array, hop_weight: float):
    """LM loss + hop_weight · InfoNCE(read-layer query → gold fact key).

    Returns (total, lm, nce, hop_hits, hop_counts): the last two count, per read
    layer, supervised positions and those where the query's top-1 entry over the
    whole store is the gold fact (retrieval accuracy; no gradient).
    """
    logits, queries, keys = model.forward(inputs, store)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    lm = (ce * weights).sum() / mx.maximum(weights.sum(), 1.0)
    q = mx.stack(queries)[sup_idx[:, 0], sup_idx[:, 1], sup_idx[:, 2]]      # (S, dk): [hop, row, pos]
    nce_logits = (q @ keys.T).astype(mx.float32) / math.sqrt(model.cfg.key_dim) + store["bias"]
    nce_ce = nn.losses.cross_entropy(nce_logits, sup_fact, reduction="none")
    nce = (nce_ce * sup_valid).sum() / mx.maximum(sup_valid.sum(), 1.0)
    hit = (mx.argmax(nce_logits, axis=-1) == sup_fact).astype(mx.float32) * sup_valid
    per_hop = (sup_idx[:, :1] == mx.arange(model.cfg.n_reads)).astype(mx.float32)          # (S, n_reads)
    hop_hits = mx.stop_gradient((hit[:, None] * per_hop).sum(axis=0))
    hop_counts = mx.stop_gradient((sup_valid[:, None] * per_hop).sum(axis=0))
    return lm + hop_weight * nce, lm, nce, hop_hits, hop_counts
