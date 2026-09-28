"""Poseidon2 over BabyBear (width 24, Plonky3 parameters): the Ligero side of the ``verity.ml.commitments`` reference.

The permutation, its constants (pinned by ``CONSTANTS_SHA256``), the known-answer vector, the scalar sponge and the
operand packing live in :mod:`verity.ml.commitments.poseidon2_babybear` (standard library only: the package boundary
test forbids numpy there) so the committer (``leaf/v2h``) and the in-relation gadget (``hashchain.py``) are literally the
same code; ``permute_ring`` evaluates the permutation over any ring (Python ints, numpy / torch arrays, or the
compiler's affine ``Expr`` with the S-box supplied as product rows).  This module re-exports it and adds the numpy
row-vectorised evaluations the committer and the tests use: :func:`permute_np`, :func:`pack_words_np`,
:func:`hash_rows_np`, :func:`row_digests`.
"""
from __future__ import annotations

import numpy as np
from verity.ml.commitments.poseidon2_babybear import *  # noqa: F401,F403
from verity.ml.commitments.poseidon2_babybear import __all__ as _ALL, _external_layer, _internal_layer  # noqa: F401
from verity.ml.commitments.poseidon2_babybear import CAPACITY, IV, LANE_BITS, OUT, P, RATE, WIDTH, permute_ring

__all__ = list(_ALL) + ["hash_rows_np", "pack_words_np", "permute_np", "row_digests"]


def _sbox_np(x: np.ndarray) -> np.ndarray:
    x2 = (x * x) % P
    x4 = (x2 * x2) % P
    x3 = (x2 * x) % P
    return (x4 * x3) % P


def permute_np(state: np.ndarray) -> np.ndarray:
    """(24, N) int64 states in [0, p) -> (24, N): N permutations at once (every product of two residues fits int64)."""
    st = np.asarray(state, dtype=np.int64) % P
    if st.shape[0] != WIDTH:
        raise ValueError("state must be (24, N)")
    out = permute_ring([st[i] for i in range(WIDTH)], _sbox_np, lambda x: x % P)
    return np.stack(out)


def pack_words_np(words, word_bits: int) -> np.ndarray:
    """(..., n_words) words of ``word_bits`` bits -> (..., n_words * word_bits / 16) elements of 16 bits each: consecutive
    words in little-endian order (``w0 + 2^word_bits w1 + ...``) -- ``poseidon2_babybear.pack_words`` on arrays."""
    w = np.asarray(words, dtype=np.int64)
    per = LANE_BITS // word_bits
    if LANE_BITS % word_bits or w.shape[-1] % per:
        raise ValueError(f"{word_bits}-bit words do not tile 16-bit lanes over {w.shape[-1]} words")
    if ((w < 0) | (w >= 2**word_bits)).any():
        raise ValueError("word out of range")
    g = w.reshape(*w.shape[:-1], w.shape[-1] // per, per)
    weights = (2 ** (word_bits * np.arange(per, dtype=np.int64)))
    return (g * weights).sum(-1)


def hash_rows_np(words: np.ndarray, word_bits: int, role: int) -> np.ndarray:
    """(N, n_words) rows -> (N, 8) digests, all rows in lockstep (the committer; equals ``hash_row`` row by row)."""
    w = np.asarray(words, dtype=np.int64)
    if w.ndim != 2:
        raise ValueError("rows must be (N, n_words)")
    el = pack_words_np(w, word_bits)                         # (N, E)
    N, E = el.shape
    if E % RATE:
        raise ValueError("row does not tile the rate")
    iv = IV(role, word_bits, w.shape[1])
    st = np.zeros((WIDTH, N), dtype=np.int64)
    for i in range(CAPACITY):
        st[RATE + i] = iv[RATE + i]
    for blk in range(0, E, RATE):
        st[:RATE] = el[:, blk:blk + RATE].T
        st = permute_np(st)
    return st[:OUT].T.copy()


def row_digests(rows, word_bits: int, role: int) -> np.ndarray:
    """``leaf/v2h`` (``verity.ml.commitments.rowleaf``) for many rows: (count, n_words) -> (count, 8) digests."""
    return hash_rows_np(np.asarray(rows, dtype=np.int64), word_bits, role)
