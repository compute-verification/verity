"""ro-bound-sketch-blake3/v0, byte-exact: the read receipt bound to the matmul, on ra-codec-wide's p512 store and fused
GEMM (`benchmarks/word_codec/wide.cu`, `wide_sketch.py`). The sketch's ±1 matrix becomes B = R ⊕ M: M the window's
fresh sign masks, R bytes of the matmul's own values (z), one R byte for every B byte, so z can carry as much entropy
as the block.

~~~text
tile u      wide_sketch's tile: 4 warps i × one 512-column chunk kc, u = G · chunks + kc (the CTA), 16 vectors r of
            n = 2,048 bytes, positions p as wide_sketch places them; T = 4 warps, m = 16 steps of 16 bytes a lane
fragment    B register q (0 … 3) of warp i, step s, lane l = 4 g + tq is u32 word idx = 4 ((16 i + s) 32 + l) + q of
            a 32 KiB tile array; byte b of register q = 2 c8 + h is B[8 c8 + g][32 (16 i + s) + 16 h + 4 tq + b]
            (`wide_sketch.sign_fragments`' order)
M           masks: bit p of row c from K_M = derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": 16, "n": n}) as wide_sketch's
            sign bits, byte 0xFF where the bit is 1 and 0x00 where it is 0, in fragment order
z source    zprev, the previous linear's output, M_b × N_z FP32 (row-major, little-endian u32 words, Z4 = M_b N_z words)
tile share  S4 = 4 ⌈Z4 / (4 · tiles)⌉ words; z_u[t] = zprev[(u S4 + t) mod Z4], t < S4 (S4 ≤ 8,192)
acc         the warp's FP32 accumulators after k32 step tt of the chunk (tt = 0 … 15): C[row][col] = the
            mma.m16n8k32 E4M3 chain (sm_120's `BLACKWELL_SM120_E4M3_M16N8K32`) of W's row and x's column over chunk
            columns 0 … 32 tt + 31 from +0; J batch blocks of 8 columns, J = 1, 2, 4, 8 for M_b ≤ 8, 16, 32, 64, x's
            rows past M_b zero; lane (g, tq)'s acc_j[q] = C[16 i + g + 8 ⌊q/2⌋][8 j + 2 tq + (q mod 2)]
mode        tile:      R[idx] = z_u[idx mod S4] for every step
            slice w:   (w = 32, 64, 128; S_w = w / 32) steps 0 … 3 (word 0) as tile; step s = 4 a + e of word a ≥ 1
                       reads the snapshot after tt_end = 4 (a − 1) + S_w ⌊e / S_w⌋ + S_w − 1 (the end of word a − 1's
                       w-slice holding k32 step s − 4), e' = e mod S_w: R = ⊕_{j ≡ e' (mod S_w)} acc_j if J ≥ S_w,
                       else acc_{e' mod J}, as the lane's four u32 (bits of the FP32 words)
            unbound:   R = 0 and B = wide_sketch's ±1 signs: ro-sketch-blake3/v0 itself
B           R ⊕ M as int8 bytes
sketch      v_{u,r} = Σ_p e_{u,r}[p] · B[c][p] over the integers, 16 int32 (|v| ≤ 128 · 128 · n = 2^25)
receipt     BLAKE3-keyed(K, u64be (jbase + 16 u + r) ‖ v_{u,r} as int32 LE), K = SHA512(prefix(TAG_DOMAIN, NAME,
            σ))[0:32]; 16 receipts a leaf and the RFC 6962 root, as the bandwidth audit's tree
~~~
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "word_codec"))

import wide as W
import wide_sketch as WS

NAME = "ro-bound-sketch-blake3/v0"
K = WS.K
T = WS.T
STEPS = 16
TILE_WORDS = T * STEPS * 32 * 4
MODES = {"unbound": 0, "tile": 1, "slice32": 32, "slice64": 64, "slice128": 128}
FIELD = W.field(512)


def batch_blocks(m: int) -> int:
    """J, the batch blocks of 8 columns the kernel computes."""
    for j in (1, 2, 4, 8):
        if m <= 8 * j:
            return j
    raise ValueError("a batch of at most 64 rows")


def mask_bits(seed: bytes, n: int = WS.vector_bytes(FIELD)) -> np.ndarray:
    """M's bits, 16 × n uint8 (1 is the byte 0xFF)."""
    from verity.ml.randomness import derive

    key = derive(seed, WS.SIGN_DOMAIN, {"tag": NAME, "k": K, "n": n})
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(K)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def mask_fragments(seed: bytes) -> np.ndarray:
    """M in fragment order, uint8 0x00 / 0xFF (T · m · 512 bytes), the kernel's `masks`."""
    return WS.sign_fragments(-mask_bits(seed).astype(np.int8), STEPS)


def sign_fragments(seed: bytes) -> np.ndarray:
    """ro-sketch-blake3/v0's ±1 signs in fragment order: the unbound mode's `masks`."""
    return WS.sign_fragments(WS.signs(seed, FIELD), STEPS)


def from_fragments(frag: np.ndarray) -> np.ndarray:
    """A 32 KiB fragment-order tile array as the 16 × n int8 matrix it holds (inverse of `sign_fragments`)."""
    x = np.ascontiguousarray(frag).view(np.uint8).reshape(T, STEPS, 8, 4, 2, 2, 4)  # i, s, g, tq, c8, h, b
    return np.ascontiguousarray(x.transpose(4, 2, 0, 1, 5, 3, 6)).reshape(K, -1).view(np.int8)


def share_words(z_words: int, tiles: int) -> int:
    s4 = 4 * -(-z_words // (4 * tiles))
    if s4 > TILE_WORDS:
        raise ValueError("a tile's share is at most 32 KiB")
    return s4


def z_share(zprev: np.ndarray, u: int, tiles: int) -> np.ndarray:
    """z_u, S4 u32 words of zprev (any array of FP32 or u32 words, read row-major)."""
    z = np.ascontiguousarray(zprev).reshape(-1).view("<u4")
    s4 = share_words(z.size, tiles)
    return z[(u * s4 + np.arange(s4)) % z.size]


def fragment_index() -> np.ndarray:
    """idx of (warp, step, lane, register): (T, 16, 32, 4) int64."""
    return np.arange(TILE_WORDS, dtype=np.int64).reshape(T, STEPS, 32, 4)


def accumulators(w_rows: np.ndarray, x_chunk: np.ndarray) -> np.ndarray:
    """acc after each k32 step of the chunk: (16, 64, 8 J) u32 FP32 words. w_rows (64, 512) and x_chunk (8 J, 512) are
    E4M3 bytes, x's rows past M_b zero."""
    from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32 as P

    rows, cols = w_rows.shape[0], x_chunk.shape[0]
    wl = [[bytes(w_rows[r, 32 * t:32 * t + 32]) for t in range(STEPS)] for r in range(rows)]
    xl = [[bytes(x_chunk[c, 32 * t:32 * t + 32]) for t in range(STEPS)] for c in range(cols)]
    out = np.zeros((STEPS, rows, cols), np.uint32)
    for r in range(rows):
        for c in range(cols):
            acc = 0
            for t in range(STEPS):
                acc = P.step(acc, wl[r][t], xl[c][t])
                out[t, r, c] = acc
    return out


def lane_acc(acc: np.ndarray) -> np.ndarray:
    """acc (16, 64, 8 J) as each lane's registers: (16, T, 32, J, 4), [tt, i, l, j, q] = acc_j[q] of lane l."""
    tt, _rows, cols = acc.shape
    j = cols // 8
    a = acc.reshape(tt, T, 2, 8, j, 4, 2)                    # tt, i, h, g, j, tq, c   (q = 2 h + c)
    return np.ascontiguousarray(a.transpose(0, 1, 3, 5, 4, 2, 6)).reshape(tt, T, 32, j, 4)


def r_words(mode: str, z_u: np.ndarray | None = None, acc: np.ndarray | None = None) -> np.ndarray:
    """R in fragment order, (T, 16, 32, 4) u32."""
    if mode == "unbound":
        return np.zeros((T, STEPS, 32, 4), np.uint32)
    r = z_u[fragment_index() % z_u.size].astype(np.uint32)
    if mode == "tile":
        return r
    sw = MODES[mode] // 32
    la = lane_acc(acc)
    j = la.shape[3]
    for a in range(1, 4):
        for e in range(4):
            end, e1 = 4 * (a - 1) + sw * (e // sw) + sw - 1, e % sw
            if j >= sw:
                y = np.bitwise_xor.reduce(la[end][:, :, e1::sw, :], axis=2)
            else:
                y = la[end][:, :, e1 % j, :]
            r[:, 4 * a + e] = y
    return r


def b_matrix(r: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """B = R ⊕ M as the 16 × n int8 matrix."""
    return from_fragments(np.ascontiguousarray(r, dtype="<u4").view(np.uint8).reshape(-1) ^ masks)


def sketch(vectors: np.ndarray, b: np.ndarray) -> np.ndarray:
    """v for each of a tile's vectors: (16, 16) int64."""
    return WS.sketch(vectors, b)


def tile_sketch(mode: str, vectors: np.ndarray, masks: np.ndarray, zprev: np.ndarray | None = None, u: int = 0,
                tiles: int = 1, acc: np.ndarray | None = None) -> np.ndarray:
    """A tile's 16 sketches: its vectors (wide_sketch's 16 × n int8), the mode's `masks` (`mask_fragments`, or
    `sign_fragments` for unbound), the z source and tile index, and the chunk's `accumulators` for a slice mode."""
    z_u = None if mode == "unbound" else z_share(zprev, u, tiles)
    return sketch(vectors, b_matrix(r_words(mode, z_u, acc), masks))


def scheme():
    from verity.core.protocols.bandwidth import tags

    return tags.Blake3Tag(name=NAME)


def receipts_of(seed: bytes, sk: np.ndarray, jbase: int = 0, which=None, name: str = NAME) -> list[bytes]:
    """The receipts of the vectors in `which` (default every vector); `name` the hash's (ro-sketch-blake3/v0 for the
    unbound mode)."""
    from verity.core.protocols.bandwidth import tags

    if np.abs(sk).max(initial=0) >= 1 << 31:
        raise ValueError("a sketch word is an int32")
    keyed, words = tags.Blake3Tag(name=name).keyed(seed), sk.astype("<i4")
    return [keyed(jbase + v, words[v].tobytes()) for v in (range(words.shape[0]) if which is None else which)]


def leaves(seed: bytes, ys: list[bytes]) -> list[bytes]:
    from verity.core.protocols.bandwidth import tags

    return tags.leaves(scheme(), seed, ys)


def root(seed: bytes, ys: list[bytes]) -> bytes:
    from verity.core.protocols.pous.verifier import MerkleTree

    return MerkleTree(leaves(seed, ys)).root


def tile_rows(n_rows: int, kdim: int, u: int) -> tuple[int, int]:
    """Tile u's first weight row and chunk."""
    G, kc = divmod(u, kdim // W.TASK_COLS)
    return 16 * T * G, kc


def binding_ratio(mode: str, m: int, kdim: int, nz: int | None = None, n_rows: int | None = None) -> float:
    """Distinct z bytes over block bytes, per tile, for a batch of m (nz, n_rows: the z source's columns and the
    linear's rows; tile mode's share is 4 m nz / (n_rows kdim) of a block, 4 m / kdim when nz = n_rows)."""
    if mode == "unbound":
        return 0.0
    tile = 4 * m * (nz or 1) / ((n_rows or 1) * kdim)
    if mode == "tile":
        return min(1.0, tile)
    return 0.25 * min(1.0, tile) + 0.75 * min(1.0, 4 * m / MODES[mode])
