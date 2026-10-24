"""Forward FLOPs of one call: prefill of context tokens [a, b) and decode of o tokens after position b.

Two models, matmul FLOPs only (2 per multiply-add), attention counted per attended position:

- ``dsv4``: DeepSeek-V4-Pro's public config (49B active, 61 layers; 128 query heads over one 512-wide KV head; a 128-token
  sliding window in every layer; layers with compress ratio 128 attend to every compressed entry, t/128; layers with ratio 4
  score every compressed entry with a 64x128 indexer and attend to the top 1024 of t/4).
- ``dense7b``: a 7B dense transformer with full causal attention (32 layers, d = 4096, Llama-2-7B / Mistral-7B shape).

Positions are continuous (integrals, not sums), which is exact to within one token's attention per call.
"""
from __future__ import annotations

import numpy as np

DSV4_RATIOS = [128, 128] + [4, 128] * 29 + [4]  # config.json's compress_ratios for the 61 main layers (the 62nd is MTP)
PER_POS = 2 * 2 * 128 * 512  # QK^T and AV over one attended position, 128 heads x 512 dims


def _dsv4_attn_cum(t: np.ndarray) -> np.ndarray:
    """Attention FLOPs of all tokens at positions [0, t), DSV4-Pro."""
    t = np.asarray(t, dtype=np.float64)
    n_hca = DSV4_RATIOS.count(128)
    n_csa = DSV4_RATIOS.count(4)
    n_all = len(DSV4_RATIOS)
    w = 128.0
    window = np.where(t <= w, PER_POS * t * t / 2, PER_POS * (w * w / 2 + w * (t - w)))
    hca = (PER_POS / 128) * t * t / 2
    indexer = (2 * 64 * 128 / 4) * t * t / 2
    sel_knee = 4 * 1024.0
    selected = np.where(t <= sel_knee, (PER_POS / 4) * t * t / 2,
                        (PER_POS / 4) * sel_knee * sel_knee / 2 + PER_POS * 1024 * (t - sel_knee))
    return n_all * window + n_hca * hca + n_csa * (indexer + selected)


def _dense7b_attn_cum(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return 2 * 2 * 32 * 4096 * t * t / 2


MODELS = {
    "dsv4": {"label": "DeepSeek-V4-Pro-like (49B active)", "lin": 2 * 49e9, "attn_cum": _dsv4_attn_cum},
    "dense7b": {"label": "7B dense, full attention", "lin": 2 * 7e9, "attn_cum": _dense7b_attn_cum},
}


def call_flops(model: str, start, ctx, out) -> np.ndarray:
    """FLOPs to prefill positions [start, ctx) and decode ``out`` tokens after ``ctx``."""
    m = MODELS[model]
    start = np.asarray(start, dtype=np.float64)
    ctx = np.asarray(ctx, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    cum = m["attn_cum"]
    return m["lin"] * (ctx - start + out) + cum(ctx + out) - cum(start)
