"""The input-side rotation R = D·H/√B of a K-column matrix: D the diagonal of ±1 signs drawn from seed K, and H
block-diagonal, m copies of the Sylvester Hadamard matrix of order B, where B = 2^p is the largest power of 2 dividing
K = m·B (m = 1 for every Llama-3.2-1B matrix; Llama-3.1-8B's down_proj has K = 7·2048). Each block is applied as
H_a ⊗ H_b (a·b = B, both powers of 2), so a row costs K·(a + b) multiply-adds instead of K·B. R is orthogonal, so
x·Wᵀ = (x·R)·(W·R)ᵀ exactly in real arithmetic: the served product rotates its activations and holds the codes of W·R.
An entry of W·R is a ±1/√B-weighted sum of a block of W's row, so it is near-Gaussian whatever W's outliers were
(QuIP's incoherence processing), and clipping it to a few σ costs little. Numpy only: `quant.py` (torch) and
`price.py` (the CPU nodes' python3) share it.
"""

from __future__ import annotations

import numpy as np


def factors(k: int) -> tuple[int, int, int]:
    """(m, a, b) with K = m·a·b and a·b the largest power of 2 dividing K."""
    assert k > 0, k
    blk = k & -k
    a = 1 << ((blk.bit_length() - 1) // 2)
    return k // blk, a, blk // a


def signs(k: int) -> np.ndarray:
    return np.random.default_rng(k).integers(0, 2, k).astype(np.float64) * 2 - 1


def hadamard(n: int) -> np.ndarray:
    """The Sylvester Hadamard matrix of order n over √n (orthogonal, symmetric)."""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def rotate(w: np.ndarray) -> np.ndarray:
    """W·R in float64, for a matrix of K columns."""
    n, k = w.shape
    m, a, b = factors(k)
    x = (w.astype(np.float64) * signs(k)).reshape(n, m, a, b)
    return (hadamard(a) @ x @ hadamard(b)).reshape(n, k)
