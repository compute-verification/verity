"""`sketch-sha512`: the remote bandwidth driver's second receipt kind, a pass-keyed ±1 sketch of each block and SHA-512
over the sketch, in place of SHA-512 over the block (`receipts.Receipter`). Leaves, salts and the tree are the
driver's own (`hm96-sha512/row/v2`, frame-v3-sha512): a `SketchReceipter` drops in where a `Receipter` goes, and
`receipts.tree` and `receipts.verify_leaf` take its receipts unchanged.

~~~text
S_i      16 × 4096 signs of pass i: K = derive(seed_i, SIGN_DOMAIN, {"tag": KIND, "k": 16, "n": 4096}), row c the
         first 512 bytes of K.stream(("row", c)), bit u (byte u // 8, LSB first) 1 → −1, 0 → +1; re-keyed every pass
         by the chained seed, so no sketch exists before the previous pass's root
E_j      the store's block j, 4096 bytes. A weight GEMM's store (wt.cu's fused decode GEMM) is cut into blocks as the
         GEMM tiles it: block j is its sketch vector j (`gemm_blocks`: row r of tile t of a CTA, 32 bytes of each
         512-byte step of each of the tile's T = 8 tasks, 32 · 16 · 8 = 4096 at SB = 8192, p512's and p1024's), so the
         sketch warps' tensor-core sums over the stored bytes are σ_j, and a launch's blocks are j = j0 + v
e_j      E_j ⊕ spread(z_(i,j)), its 4096 bytes as int8
σ_j      S_i · e_j over the integers (|σ_j[c]| ≤ 2^19), 16 int32 little-endian: 64 bytes
y_(i,p)  SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be j ‖ σ_j),  j = plan_i[p]: one compression after the
         seed's block (72 bytes, 200 in all), SKETCH_RECEIPT_DOMAIN = "verity/remote-bandwidth/sketch-receipt/v0\0"
~~~

Its soundness rests on the sketch (`RateAuditSketch`), which the full-block receipt does not need: a block that
differs from E_j ⊕ spread(z) passes a row with probability at most 1/2 over S_i, all 16 with at most 2^−16.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity.ml.randomness import derive, frame

KIND = "sketch-sha512"
SIGN_DOMAIN = "verity/remote-bandwidth/sketch-signs/v0"
SKETCH_RECEIPT_DOMAIN = b"verity/remote-bandwidth/sketch-receipt/v0\0"
ROWS = 16
BLOCK = 4096


def sign_bits(seed: bytes) -> np.ndarray:
    """S_i's sign bits, ROWS × 512 bytes: bit u of row c (byte u // 8, LSB first) is 1 where S_i[c][u] = −1."""
    key = derive(rb._digest(seed, "a pass seed"), SIGN_DOMAIN, {"tag": KIND, "k": ROWS, "n": BLOCK})
    return np.stack([np.frombuffer(key.stream(("row", c), BLOCK // 8), np.uint8) for c in range(ROWS)])


def signs(seed: bytes) -> np.ndarray:
    """S_i, ROWS × 4096 int8 of ±1."""
    bits = np.unpackbits(sign_bits(seed), axis=1, bitorder="little")
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8)


def sketch(s: np.ndarray, block: bytes, z: bytes) -> bytes:
    """σ_j's 64 bytes: S_i (`signs`) times E_j ⊕ spread(z) as int8, 16 int32 little-endian."""
    if type(block) is not bytes or len(block) != BLOCK:
        raise ValueError(f"a block is {BLOCK} bytes")
    e = np.frombuffer(rb._xor(block, rb.spread(z)), np.int8).astype(np.int64)
    return (s.astype(np.int64) @ e).astype("<i4").tobytes()


def sketches(s: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """σ of many already-masked blocks at once (n × 4096 uint8 or int8): n × 16 int32."""
    e = np.asarray(blocks).view(np.int8).reshape(-1, BLOCK).astype(np.int64)
    return (e @ s.astype(np.int64).T).astype("<i4")


def gemm_blocks(store: np.ndarray, nbs: int, nchunks: int, sb: int, groups: int, t: int) -> np.ndarray:
    """A weight GEMM's store (`wt.cu`: nbs row blocks × nchunks 512-column chunks, SB = sb stored bytes a task) as its
    blocks, (blocks × 4096) uint8, block v the launch's sketch vector v (`sketching.vectors` at its groups and T = t),
    zeros where a CTA's last tile runs past its tasks."""
    import sketching

    vs = sketching.vectors(store, nbs, nchunks, sb, groups, t)
    if vs.shape[1] != BLOCK:
        raise ValueError(f"a GEMM's vector is {vs.shape[1]} bytes at SB = {sb} and T = {t}, not a {BLOCK}-byte block")
    return vs.view(np.uint8)


def prefix(seed: bytes) -> bytes:
    """`Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = SKETCH_RECEIPT_DOMAIN + frame(rb._digest(seed, "a pass seed"))
    return body + bytes(-len(body) % 128)


def receipt_of(head, j: int, sigma: bytes) -> bytes:
    """y from the seed block's hash state `head` (`hashlib.sha512(prefix(seed))`), index j and σ_j's bytes."""
    if type(j) is not int or not 0 <= j < 1 << 64:
        raise ValueError("a block index is a u64")
    h = head.copy()
    h.update(j.to_bytes(8, "big") + sigma)
    return h.digest()


def receipts_of(seed: bytes, sk: np.ndarray, j0: int) -> list[bytes]:
    """y of blocks j0, j0 + 1, … from their sketches (n × 16 integers)."""
    head, w = hashlib.sha512(prefix(seed)), np.asarray(sk).astype("<i4")
    return [receipt_of(head, j0 + v, w[v].tobytes()) for v in range(w.shape[0])]


class SketchReceipter:
    """Sketch receipts under one pass's seed: `receipter(j, block, z)` is `y_(i,j)`, as `receipts.Receipter`'s."""

    def __init__(self, seed: bytes) -> None:
        self.signs = signs(seed)
        self.head = hashlib.sha512(prefix(seed))

    def __call__(self, j: int, block: bytes, z: bytes) -> bytes:
        return receipt_of(self.head, j, sketch(self.signs, block, z))
