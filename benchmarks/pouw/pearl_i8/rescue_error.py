"""Output error of the rescue for PROTO-DESIGN (Dither with U widened to ±96, so weights at |W| <= 31) against Dither (±63).

y = x·Wᵀ with heavy-tailed weights (Student t, ν = 4, per-row absmax RTN to ±q) and Gaussian activations quantized per row
to int8 as H1 serves them (X = floor(x/s) + b, b a fair coin bit). Prints the relative output error (Frobenius) and its
weight and activation shares, so the perplexity cost can be scaled from Dither's measured +0.80% (Llama-3.1-8B,
WikiText-2 2K) by error variance.
"""

from __future__ import annotations

import numpy as np


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    rng = np.random.default_rng(3)
    n, k, m = 512, 4096, 256
    w = rng.standard_t(4, size=(n, k))
    x = rng.standard_normal((m, k))
    y = x @ w.T
    sx = np.abs(x).max(axis=1, keepdims=True) / 126
    xq = (np.floor(x / sx) + rng.integers(0, 2, size=x.shape)) * sx
    for q in (63, 31):
        sw = np.abs(w).max(axis=1, keepdims=True) / q
        wq = np.round(w / sw) * sw
        e_w, e_x, e = rel(x @ wq.T, y), rel(xq @ w.T, y), rel(xq @ wq.T, y)
        print(f"|W| <= {q}: output rel err {e:.4f} (weights alone {e_w:.4f}, activations alone {e_x:.4f})")


if __name__ == "__main__":
    main()
