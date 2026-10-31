"""The input-side rotation R = D·H/√K of a K-column matrix: D the diagonal of ±1 signs drawn from seed K, H the Sylvester
Hadamard matrix of order K (a power of 2), applied as H_a ⊗ H_b (a·b = K, both powers of 2) so a row costs K·(a + b)
multiply-adds instead of K². R is orthogonal, so x·Wᵀ = (x·R)·(W·R)ᵀ exactly in real arithmetic: the served product
rotates its activations and holds the codes of W·R. A row of W·R is a ±1/√K-weighted sum of every entry of W's row, so
its entries are near-Gaussian with the row's norm whatever W's outliers were (QuIP's incoherence processing), and
clipping it to a few σ costs little. Numpy only: `quant.py` (torch) and `price.py` (the CPU nodes' python3) share it.
"""

from __future__ import annotations

import numpy as np


def factors(k: int) -> tuple[int, int]:
    assert k > 0 and k & (k - 1) == 0, f"K = {k} is not a power of 2"
    a = 1 << ((k.bit_length() - 1) // 2)
    return a, k // a


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
    a, b = factors(k)
    x = (w.astype(np.float64) * signs(k)).reshape(n, a, b)
    return (hadamard(a) @ x @ hadamard(b)).reshape(n, k)
