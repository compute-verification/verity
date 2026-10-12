"""Receipts inside attention (`kv.cu`'s `kv_attn` forms 10 and 11, on mma-ws): the sign table, the reference sketches
and the receipts.

~~~text
block      RT tiles of one (sequence, KV head) pair at tile offsets that are multiples of RT; a tile is MT = 32 tokens, and
           a decoder warp's lane l of warp i holds word l % 8 of token 4i + l // 8 of the head's share (K words 0-3,
           V words 4-7), 64 raw (encoded) bytes
vector     warp i's A row r = g + 8e over a block: byte b of the lane's register 4s + 2h + e, lane 4g + tq, tile τ, at
           position p = 128τ + 32s + 16h + 4tq + b, so n = 128 RT (ra-sketch-tag's fragment order); index
           ((pair · blocks + block) · MDW + i) · 16 + r, pair = sequence · KV_HEADS + head
S          16 × n signs: derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": 16, "n": n}), row c the first n/8 bytes of
           stream(("row", c)), bit p (LSB first) 1 → −1 (ra-sketch-tag's `sketch.sign_bits`)
B          unbound (ro-sketch-blake3/v0): S[c][p]. Bound (ro-bound-sketch-blake3/v0, ra-bound-receipt's §1): byte b of
           R ⊕ M as int8, register q = 2 blk + h of (warp i, step s, lane l = 4g + tq) for c = 8 blk + g, R = z word
           (512 i + 128 s + 4 l + q) mod ZW of the CTA's tile RLAG back (0 for its first RLAG tiles), M = 0xFF where
           S[c][p] = −1; z is the MMA warps' PV accumulators after that tile (`kv.cu`'s layout)
sketch     v[r][c] = Σ_p e_r[p] · B[p][c], 16 int32
receipt    BLAKE3-keyed(K, u64be(vector) ‖ sketch as 16 int32 LE), K = tags.Blake3Tag(NAME).key(σ)
~~~
"""

from __future__ import annotations

import numpy as np

from verity.core.protocols.bandwidth import tags
from verity.ml.randomness import derive

NAMES = {1: "ro-sketch-blake3/v0", 2: "ro-bound-sketch-blake3/v0"}
SIGN_DOMAIN = "verity/bandwidth/sketch-signs/v0"
KS = 16
LANES = 32
WORD_BYTES = 64


def sign_bits(seed: bytes, rc: int, n: int) -> np.ndarray:
    """S's bits, 16 × n uint8 (1 is −1)."""
    key = derive(seed, SIGN_DOMAIN, {"tag": NAMES[rc], "k": KS, "n": n})
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(KS)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def pack(bits: np.ndarray, rt: int) -> np.ndarray:
    """`kv_rc_set`'s table, (rt, 32 lanes, 4) uint32: register r = 4s + 2 blk + h of tile τ at bit
    16 ((r % 6) // 3) + 4b + r % 3 of word r // 6, set where byte b's sign is −1."""
    if bits.shape != (KS, 128 * rt):
        raise ValueError("a block's signs are 16 × 128 RT")
    out = np.zeros((rt, LANES, 4), dtype=np.uint32)
    lane = np.arange(LANES)
    g, tq = lane // 4, lane % 4
    for tau in range(rt):
        for r in range(16):
            s, blk, h = r // 4, (r % 4) // 2, r % 2
            for b in range(4):
                bit = bits[8 * blk + g, 128 * tau + 32 * s + 16 * h + 4 * tq + b].astype(np.uint32)
                out[tau, :, r // 6] |= bit << (16 * ((r % 6) // 3) + 4 * b + r % 3)
    return out


def key_words(seed: bytes, rc: int) -> np.ndarray:
    return np.frombuffer(tags.Blake3Tag(name=NAMES[rc]).key(seed), dtype="<u4").copy()


def vectors(lanes: np.ndarray, rt: int) -> np.ndarray:
    """A CTA's lanes (tiles, MDW, 32, 64) uint8, tiles a multiple of rt, as (blocks, MDW, 16, 128 rt) int8."""
    t, nw = lanes.shape[:2]
    a = lanes.reshape(t // rt, rt, nw, 8, 4, 4, 2, 2, 4)  # blk, τ, i, g, tq, s, h, e, b
    a = a.transpose(0, 2, 7, 3, 1, 5, 6, 4, 8)  # blk, i, e, g, τ, s, h, tq, b
    return np.ascontiguousarray(a).reshape(t // rt, nw, 16, rt * 128).view(np.int8)


def multipliers(bits: np.ndarray, rt: int, nw: int, z: np.ndarray | None, lag: int, zw: int) -> np.ndarray:
    """B for each block and warp of a CTA, (blocks, MDW, n, 16) int8; z (tiles, zw) uint32 the z after each tile (None:
    unbound)."""
    n = 128 * rt
    if z is None:
        s = (1 - 2 * bits.astype(np.int8)).T  # n × 16
        return np.broadcast_to(s, (1, nw, n, KS))
    t = z.shape[0]
    zt = np.concatenate([np.zeros((lag, zw), np.uint32), z[: t - lag]]) if t > lag else np.zeros((t, zw), np.uint32)
    i, s, lane, q = np.meshgrid(np.arange(nw), np.arange(4), np.arange(LANES), np.arange(4), indexing="ij")
    idx = (512 * i + 128 * s + 4 * lane + q) % zw
    rw = zt[:, idx]  # tiles, i, s, l, q
    rb = np.ascontiguousarray(rw, dtype="<u4").view(np.uint8).reshape(t, nw, 4, 8, 4, 2, 2, 4)  # T, i, s, g, tq, blk, h, b
    m = (0xFF * bits.astype(np.uint8)).reshape(2, 8, rt, 4, 2, 4, 4)  # blk, g, τ, s, h, tq, b
    m = m.transpose(2, 3, 1, 5, 0, 4, 6)  # τ, s, g, tq, blk, h, b
    rb = rb.reshape(t // rt, rt, nw, 4, 8, 4, 2, 2, 4) ^ m[None, :, None]
    bv = rb.view(np.int8).transpose(0, 2, 1, 3, 7, 5, 8, 6, 4)  # blk, i, τ, s, h, tq, b, blk_c, g
    return np.ascontiguousarray(bv).reshape(t // rt, nw, n, KS)


def sketch(lanes: np.ndarray, bits: np.ndarray, rt: int, z: np.ndarray | None = None, lag: int = 2,
           zw: int = 1024) -> np.ndarray:
    """A CTA's sketches, (blocks, MDW, 16, 16) int64: lanes (tiles, MDW, 32, 64) uint8, zero-padded to whole blocks."""
    v = vectors(lanes, rt).astype(np.float64)
    b = multipliers(bits, rt, lanes.shape[1], z, lag, zw).astype(np.float64)
    return np.rint(v @ b).astype(np.int64)


def receipt(seed: bytes, rc: int, vector: int, sk: np.ndarray) -> bytes:
    return tags.Blake3Tag(name=NAMES[rc]).keyed(seed)(vector, sk.astype("<i4").tobytes())
