"""ro-sketch-blake3/v0 over the wide store, byte-exact: the sketch tag's sketch-then-hash tag (the sign matrix, the
sketch, the tag, the leaves and the root as `benchmarks/sketch_tag/sketch.py` at 1b21bc8f7 states them) on `wide`'s
records, so the fused GEMM sketches the encoded words it already holds.

~~~text
lane bytes  lane l's record of task q (`wide`'s store: its L·N u32 limbs in (word, limb) order), little-endian, zero-
            padded to 16 m bytes: m = ⌈L·N / 4⌉ steps, a step one mma.m16n8k32 A fragment (registers a0 … a3)
            (16 at p256 and p512, 17 at p127 and p521, 18 at p255)
tile        T = 4 consecutive row blocks at one 512-column chunk: one CTA of `wide.cu`'s fused GEMM (KW = 1), warp i
            holding task (T G + i, kc)
vector r    of tile (G, kc), index 16 (G · chunks + kc) + r: fragment row r of tasks (T G + i, kc), i = 0 … T − 1, byte
            (i, s, col) at position 32 (i m + s) + col, (r, col) = `codec.fragment_offset(lane, reg, b)` of lane byte
            16 s + 4 reg + b; n = 32 m T bytes, read as int8
S           k × n signs (k = 16): K_S = derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n}); row c is the first ⌈n/8⌉
            bytes of K_S.stream(("row", c)), bit p (byte p // 8, bit p % 8) 1 → −1, 0 → +1
sketch      S · e_v over the integers, k int32 little-endian words
tag y_v     BLAKE3-keyed(K, u64be (jbase + v) ‖ sketch), K = SHA512(prefix(TAG_DOMAIN, NAME, σ))[0:32]; 16 tags a leaf
            (one leaf a tile) and the RFC 6962 root, as the bandwidth audit's tree
~~~
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wide as W  # noqa: E402

NAME = "ro-sketch-blake3/v0"
SIGN_DOMAIN = "verity/bandwidth/sketch-signs/v0"
K = 16
T = 4
ROWS = 16


def steps(f: W.Field) -> int:
    return -(-f.lane_words() * f.limbs // 4)


def vector_bytes(f: W.Field) -> int:
    return 32 * steps(f) * T


def tiles(n_rows: int, kdim: int) -> int:
    return -(-(n_rows // ROWS) // T) * (kdim // W.TASK_COLS)


def sign_bits(seed: bytes, k: int, n: int) -> np.ndarray:
    """S's bits, k × n uint8 (1 is −1)."""
    from verity.ml.randomness import derive

    key = derive(seed, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n})
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(k)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def signs(seed: bytes, f: W.Field, k: int = K) -> np.ndarray:
    """S, k × n int8 in {−1, +1}."""
    return (1 - 2 * sign_bits(seed, k, vector_bytes(f)).astype(np.int8)).astype(np.int8)


def lane_steps(store: bytes, n_rows: int, kdim: int, f: W.Field) -> np.ndarray:
    """The store's bytes as (row block, chunk, step, lane, 16)."""
    nbs, nch, lw, m = n_rows // ROWS, kdim // W.TASK_COLS, f.lane_words(), steps(f)
    u = np.frombuffer(store, dtype="<u4")
    if n_rows % ROWS or kdim % W.TASK_COLS or u.size != nbs * nch * lw * f.limbs * W.LANES:
        raise ValueError("the store is L·N·32 limbs a task of 16 rows and 512 columns")
    u = u.reshape(nbs, nch, lw * f.limbs, W.LANES).transpose(0, 1, 3, 2)
    u = np.concatenate([u, np.zeros((nbs, nch, W.LANES, 4 * m - lw * f.limbs), "<u4")], axis=3)
    return np.ascontiguousarray(u).view(np.uint8).reshape(nbs, nch, W.LANES, m, 16).transpose(0, 1, 3, 2, 4)


def arrange(a: np.ndarray, t: int = T) -> np.ndarray:
    """(row block, chunk, step, lane, 16) → (vector, position), a short last tile zero-padded."""
    nbs, nchunks, m = a.shape[:3]
    groups = -(-nbs // t)
    a = np.concatenate([a, np.zeros((groups * t - nbs, *a.shape[1:]), a.dtype)])
    # lane = 4 g + tq, lane byte = 4 reg + b, reg = 2 h + e; fragment row g + 8 e, column 4 tq + 16 h + b
    a = a.reshape(groups, t, nchunks, m, 8, 4, 2, 2, 4)        # G, i, kc, s, g, tq, h, e, b
    a = a.transpose(0, 2, 7, 4, 1, 3, 6, 5, 8)                  # G, kc, e, g, i, s, h, tq, b
    return a.reshape(groups * nchunks * ROWS, t * m * 32)


def vectors(store: bytes, n_rows: int, kdim: int, f: W.Field) -> np.ndarray:
    """Every vector of the store, (tiles · 16) × n int8, vector 16 (G · chunks + kc) + r."""
    return arrange(lane_steps(store, n_rows, kdim, f)).view(np.int8)


def sketch(vs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """S · e_v for every vector, exactly: (vectors) × k int64 (|S · e_v| ≤ 128 n < 2^31)."""
    if vs.shape[1] != s.shape[1]:
        raise ValueError("a vector has n bytes")
    return vs.astype(np.int64) @ s.astype(np.int64).T


def tag_scheme():
    from verity.core.protocols.bandwidth import tags

    return tags.Blake3Tag(name=NAME)


def tags_of(seed: bytes, sk: np.ndarray, jbase: int = 0, which=None) -> list[bytes]:
    """y_v for v in `which` (default every vector)."""
    if np.abs(sk).max(initial=0) >= 1 << 31:
        raise ValueError("a sketch word is an int32")
    keyed, words = tag_scheme().keyed(seed), sk.astype("<i4")
    return [keyed(jbase + v, words[v].tobytes()) for v in (range(words.shape[0]) if which is None else which)]


def leaves(seed: bytes, ys: list[bytes]) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags

    return tags.leaves(tag_scheme(), seed, ys)


def root(seed: bytes, ys: list[bytes]) -> bytes:
    from verity.core.protocols.pous.verifier import MerkleTree

    return MerkleTree(leaves(seed, ys)).root


def sign_fragments(s: np.ndarray, m: int) -> np.ndarray:
    """S as the B operands, uint8: for warp i, step s and lane l = 4 g + tq, 16 bytes at ((i m + s) 32 + l) 16, b0 and
    b1 of sketch columns 0 … 7, then b0 and b1 of columns 8 … 15; byte q of b_h of half c8 is
    S[8 c8 + g][32 (i m + s) + 16 h + 4 tq + q]."""
    k, n = s.shape
    if k != K or n != 32 * m * T:
        raise ValueError("S is 16 × 32 m T")
    x = s.reshape(2, 8, T, m, 2, 4, 4)            # c8, g, i, s, h, tq, q
    x = x.transpose(2, 3, 1, 5, 0, 4, 6)           # i, s, g, tq, c8, h, q
    return np.ascontiguousarray(x).reshape(-1).view(np.uint8)


def sketch_from_fragments(store: bytes, n_rows: int, kdim: int, f: W.Field, frag: np.ndarray) -> np.ndarray:
    """The sketch as the fused GEMM computes it, one mma.m16n8k32.s8 at a time from `sign_fragments` and the lane
    bytes, the T warps' accumulators summed: (vectors) × 16 int64 (checks the packing against the MMA's layouts)."""
    a = lane_steps(store, n_rows, kdim, f).view(np.int8).astype(np.int64)
    nbs, nch, m = a.shape[:3]
    b = frag.view(np.int8).astype(np.int64).reshape(T, m, 8, 4, 2, 2, 4)   # i, s, g', tq, c8, h, q
    a = a.reshape(nbs, nch, m, 8, 4, 2, 2, 4)                               # nb, kc, s, g, tq, h, e, q
    out = np.zeros((-(-nbs // T), nch, 2, 8, 2, 8), np.int64)               # G, kc, e, g, c8, g'
    for nb in range(nbs):
        G, i = divmod(nb, T)
        # D[g + 8 e][8 c8 + g'] += Σ_{s, tq, h, q} A[g + 8 e][16 h + 4 tq + q] · B[16 h + 4 tq + q][8 c8 + g']
        out[G] += np.einsum("ksgtheq,sutchq->kegcu", a[nb], b[i])
    return out.reshape(-1, K)
