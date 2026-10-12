"""Sketch-then-hash, byte-exact: the bandwidth audit's tag over an exact INT8 sketch of each n-byte vector of the store.

~~~text
S          the window's k × n sign matrix: K_S = derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n}); row c is the
           first ⌈n/8⌉ bytes of K_S.stream(("row", c)), bit p of the row (byte p // 8, bit p % 8, LSB first) is S[c][p],
           1 → −1, 0 → +1
vector v   n bytes of the store read as int8 (`vectors`), every byte of the store in exactly one vector
sketch     σ_v = S · e_v over the integers (|σ_v[c]| ≤ 128 n < 2^31), as k int32 little-endian words
y_v        ro-sketch-blake3/v0: BLAKE3-keyed(K, u64be (jbase + v) ‖ σ_v), K = SHA512(prefix(TAG_DOMAIN, NAME, σ))[0:32]
           (`tags.Blake3Tag` under NAME); leaves (16 tags a leaf, one leaf a tile) and the RFC 6962 root as `tags`
~~~

The vectors follow the fused GEMM's mma.m16n8k32 fragments, so a warp sketches the bytes it already holds: the store's
16-row × 512-column tasks (`codec.fragment_stream`), lane l's bytes of a task split into m steps of 16 (a step is one
A fragment, registers a0 … a3; m = 16 plain, ⌈8 L / 16⌉ encoded: 19 at 56 payload bits, 18 at 60, the last step
zero-padded). A tile is T consecutive row blocks at one 512-column chunk; vector r of tile (G, kc), index
16 (G · chunks + kc) + r, is fragment row r of tasks (G T + i, kc), i = 0 … T − 1, in step order: byte (i, s, col) at
position 32 (i m + s) + col, (r, col) = `codec.fragment_offset(lane, reg, b)` of lane byte 16 s + 4 reg + b. So
n = 32 m T, and a short last tile is zero-padded. The GEMM puts the store's bytes in the int8 MMA's A operand and the
signs in B: 16 vectors share each sign fragment, and no two vectors' bytes are ever added under one sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "word_codec"))

import codec as C  # noqa: E402

from verity.core.protocols.bandwidth import tags  # noqa: E402
from verity.core.protocols.pous.verifier import MerkleTree  # noqa: E402
from verity.ml.randomness import Key, derive  # noqa: E402

NAME = "ro-sketch-blake3/v0"
SIGN_DOMAIN = "verity/bandwidth/sketch-signs/v0"
TAG = tags.Blake3Tag(name=NAME)
KS = (8, 16, 32)
ROWS = C.TILE_ROWS
LANES = C.LANES
PLAIN = 0


def sign_key(seed: bytes, k: int, n: int) -> Key:
    if k not in KS or n < 1:
        raise ValueError(f"k is one of {KS} and n positive")
    return derive(seed, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n})


def sign_bits(seed: bytes, k: int, n: int) -> np.ndarray:
    """S's bits, k × n uint8 (1 is −1)."""
    key = sign_key(seed, k, n)
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(k)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def signs(seed: bytes, k: int, n: int) -> np.ndarray:
    """S, k × n int8 in {−1, +1}."""
    return (1 - 2 * sign_bits(seed, k, n).astype(np.int8)).astype(np.int8)


def steps(bits: int) -> int:
    """m, a lane's 16-byte steps a task: 16 for the plain store (bits = PLAIN), ⌈8 L / 16⌉ for the encoded one."""
    return C.TASK_TILES if bits == PLAIN else -(-8 * C.lane_words(bits) // 16)


def vector_bytes(bits: int, t: int) -> int:
    return 32 * steps(bits) * t


def lane_steps(store: bytes, n_rows: int, kdim: int, bits: int) -> np.ndarray:
    """The store's bytes as (row block, chunk, step, lane, 16): plain `codec.plain_fragments` or encoded words
    (`codec.fragment_payloads` encoded, u64le, word w of lane l of task q at L·32·q + 32·w + l)."""
    C.check_shape(n_rows, kdim)
    nbs, nchunks, m = n_rows // ROWS, kdim // C.TASK_COLS, steps(bits)
    a = np.frombuffer(store, dtype=np.uint8)
    if bits == PLAIN:
        if a.size != n_rows * kdim:
            raise ValueError("a plain store is N × K bytes")
        return a.reshape(nbs, nchunks, m, LANES, 16)
    lw = C.lane_words(bits)
    if a.size != nbs * nchunks * lw * LANES * 8:
        raise ValueError("an encoded store is L·32 words a task")
    w = a.reshape(nbs, nchunks, lw, LANES, 8).transpose(0, 1, 3, 2, 4).reshape(nbs, nchunks, LANES, 8 * lw)
    w = np.concatenate([w, np.zeros((nbs, nchunks, LANES, 16 * m - 8 * lw), np.uint8)], axis=3)
    return w.reshape(nbs, nchunks, LANES, m, 16).transpose(0, 1, 3, 2, 4)


def vectors(store: bytes, n_rows: int, kdim: int, bits: int, t: int) -> np.ndarray:
    """Every vector of the store, (tiles · 16) × n int8, vector 16 (G · chunks + kc) + r."""
    return arrange(lane_steps(store, n_rows, kdim, bits), t).view(np.int8)


def arrange(a: np.ndarray, t: int) -> np.ndarray:
    """(row block, chunk, step, lane, 16) → (vector, position), any dtype, a short last tile zero-padded."""
    if t < 1:
        raise ValueError("a tile is at least one row block")
    nbs, nchunks, m = a.shape[:3]
    groups = -(-nbs // t)
    a = np.concatenate([a, np.zeros((groups * t - nbs, *a.shape[1:]), a.dtype)])
    # lane = 4 g + tq, lane byte = 4 reg + b, reg = 2 h + e; fragment row g + 8 e, column 4 tq + 16 h + b
    a = a.reshape(groups, t, nchunks, m, 8, 4, 2, 2, 4)        # G, i, kc, s, g, tq, h, e, b
    a = a.transpose(0, 2, 7, 4, 1, 3, 6, 5, 8)                  # G, kc, e, g, i, s, h, tq, b
    return a.reshape(groups * nchunks * ROWS, t * m * 32)


def tiles(n_rows: int, kdim: int, t: int) -> int:
    return -(-(n_rows // ROWS) // t) * (kdim // C.TASK_COLS)


def sketch(vs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """S · e_v for every vector, exactly in int64: (vectors) × k."""
    if vs.shape[1] != s.shape[1]:
        raise ValueError("a vector has n bytes")
    return vs.astype(np.int64) @ s.astype(np.int64).T


def sketch_words(sk: np.ndarray) -> np.ndarray:
    if np.abs(sk).max(initial=0) >= 1 << 31:
        raise ValueError("a sketch word is an int32")
    return sk.astype("<i4")


def tags_of(seed: bytes, sk: np.ndarray, jbase: int = 0) -> list[bytes]:
    keyed, words = TAG.keyed(seed), sketch_words(sk)
    return [keyed(jbase + v, words[v].tobytes()) for v in range(words.shape[0])]


def leaves(seed: bytes, ys: list[bytes]) -> list[bytes]:
    return tags.leaves(TAG, seed, ys)


def root(seed: bytes, ys: list[bytes]) -> bytes:
    return MerkleTree(leaves(seed, ys)).root


def sign_fragments(bits_kn: np.ndarray, m: int, t: int) -> np.ndarray:
    """The device's copy of S, packed for the B operand: for tile task i, step group u (128 / k steps of 16 bytes,
    steps padded to MP = ⌈m / spl⌉ spl) and lane l, 16 bytes at ((i · MP / spl + u) · 32 + l) · 16; byte
    (s mod spl) · k/8 + blk holds b0's four signs (bits 0 … 3) and b1's (bits 4 … 7) of rows 8 blk + g at step
    i m + s: b0 bit q is S[8 blk + g][32 (i m + s) + 4 tq + q], b1 bit q the same at + 16."""
    k, n = bits_kn.shape
    if k not in KS or n != 32 * m * t:
        raise ValueError("S is k × 32 m T")
    spl, kb = 128 // k, k // 8
    mp = -(-m // spl) * spl
    s = bits_kn.reshape(kb, 8, t, m, 2, 4, 4)                     # blk, g, i, s, h, tq, q
    s = np.concatenate([s, np.zeros((kb, 8, t, mp - m, 2, 4, 4), np.uint8)], axis=3)
    nib = (s << np.arange(4, dtype=np.uint8)).sum(axis=-1, dtype=np.uint8)   # blk, g, i, s, h, tq
    byte = nib[..., 0, :] | (nib[..., 1, :] << 4)                  # blk, g, i, s, tq
    byte = byte.reshape(kb, 8, t, mp // spl, spl, 4)               # blk, g, i, u, sl, tq
    return np.ascontiguousarray(byte.transpose(2, 3, 1, 5, 4, 0)).reshape(-1)   # i, u, g, tq, sl, blk


def sketch_from_fragments(store: bytes, n_rows: int, kdim: int, bits: int, t: int, frag: np.ndarray, k: int):
    """The sketch as the device computes it, mma by mma from the packed signs: a check of `sign_fragments`."""
    a = lane_steps(store, n_rows, kdim, bits).view(np.int8).astype(np.int64)
    nbs, nchunks, m = a.shape[:3]
    spl, kb = 128 // k, k // 8
    mp = -(-m // spl) * spl
    f = frag.reshape(t, mp // spl, 8, 4, spl, kb).transpose(0, 1, 4, 2, 3, 5).reshape(t, mp, 8, 4, kb)
    sg = np.stack([(f >> q) & 1 for q in range(8)], axis=-1)       # i, s, c, tq, blk, (h, q)
    b = (1 - 2 * sg.astype(np.int64))[:, :m].reshape(t, m, 8, 4, kb, 2, 4)
    a = a.reshape(nbs, nchunks, m, 8, 4, 2, 2, 4)                  # nb, kc, s, g, tq, h, e, q
    out = np.zeros((-(-nbs // t), nchunks, ROWS, k), np.int64)
    for nb in range(nbs):
        G, i = divmod(nb, t)
        # D[(e, g)][(blk, c)] += Σ_{s, tq, h, q} A[(e, g)][(s, h, tq, q)] · B[(s, h, tq, q)][(blk, c)]
        out[G] += np.einsum("ksgtheq,sctbhq->kegbc", a[nb], b[i]).reshape(nchunks, ROWS, k)
    return out.reshape(-1, k)
