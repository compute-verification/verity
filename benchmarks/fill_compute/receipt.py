"""The filler's receipt: sketch-sha512, the one receipt form (`benchmarks/codec_tensor/sketch_receipt.py`, whose S_i, σ and y
these are, pinned by `tests/vectors.json`), over each tile's C̃ and U, unmasked.

~~~text
E_j      problem tile t's blocks (t = `tile_at`'s queue position mod the tile count, at (tm, tn) = bench.raster(t)): the
         T x T C̃ tile, row-major FP32 little-endian, cut into RB = T·T·4 / 4096 blocks (8 rows of 128 at T = 128, 16 of 64
         at T = 64), then U's; block b of matrix m (0 C̃, 1 U) is j = (2 t + m) RB + b
e_j      E_j as int8 (z = b"\\0": `SketchReceipter(seed)(j, E_j, b"\\0")`)
σ_j, y_j sketch_receipt's: S_i · e_j, 16 int32; SHA512(prefix(seed) ‖ u64be j ‖ σ_j), one compression after prefix(seed)
~~~

The kernel (`fill.cu`'s F_RCPT) writes tile t's 2 RB receipts, 64 bytes each in j order, at byte 2 RB · 64 · t of p9.
"""
from __future__ import annotations

import hashlib

import numpy as np

from verity.ml.randomness import derive, frame

KIND = "sketch-sha512"
SIGN_DOMAIN = "verity/remote-bandwidth/sketch-signs/v0"
SKETCH_RECEIPT_DOMAIN = b"verity/remote-bandwidth/sketch-receipt/v0\0"
ROWS = 16
BLOCK = 4096
SEED_BYTES = 64


def _seed(seed: bytes) -> bytes:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"a pass seed is {SEED_BYTES} bytes")
    return seed


def sign_bits(seed: bytes) -> np.ndarray:
    """S_i's sign bits, ROWS × 512 bytes: bit u of row c (byte u // 8, LSB first) is 1 where S_i[c][u] = −1."""
    key = derive(_seed(seed), SIGN_DOMAIN, {"tag": KIND, "k": ROWS, "n": BLOCK})
    return np.stack([np.frombuffer(key.stream(("row", c), BLOCK // 8), np.uint8) for c in range(ROWS)])


def signs(seed: bytes) -> np.ndarray:
    """S_i, ROWS × 4096 int8 of ±1."""
    bits = np.unpackbits(sign_bits(seed), axis=1, bitorder="little")
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8)


def sketches(s: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """σ of n blocks (n × 4096 bytes, as int8): n × 16 int32, by a float64 product (each term ±128 at most and each sum
    within 2^19, so every partial sum is an integer float64 holds exactly)."""
    e = np.asarray(blocks).view(np.int8).reshape(-1, BLOCK).astype(np.float64)
    return np.rint(e @ s.astype(np.float64).T).astype("<i4")


def prefix(seed: bytes) -> bytes:
    """`Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = SKETCH_RECEIPT_DOMAIN + frame(_seed(seed))
    return body + bytes(-len(body) % 128)


def receipts_of(seed: bytes, sk: np.ndarray, j0: int) -> list[bytes]:
    """y of blocks j0, j0 + 1, … from their sketches (n × 16 integers)."""
    head, w = hashlib.sha512(prefix(seed)), np.asarray(sk).astype("<i4")
    out = []
    for v in range(w.shape[0]):
        h = head.copy()
        h.update((j0 + v).to_bytes(8, "big") + w[v].tobytes())
        out.append(h.digest())
    return out


def tile_blocks(c: np.ndarray, u: np.ndarray, T: int, tm: int, tn: int) -> np.ndarray:
    """Tile (tm, tn)'s 2 RB blocks: its C̃ then its U (M x N of 4-byte words, FP32 or their bits), row-major, (2 RB) x 4096
    uint8."""
    if c.dtype.itemsize != 4 or u.dtype.itemsize != 4:
        raise ValueError("C~ and U are 4-byte words")
    parts = [np.ascontiguousarray(x[tm * T:(tm + 1) * T, tn * T:(tn + 1) * T]) for x in (c, u)]
    return np.concatenate([p.view(np.uint8).reshape(-1, BLOCK) for p in parts])


def all_receipts(seed: bytes, fc: np.ndarray, fu: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Every tile's receipts, tiles x (2 RB · 64) uint8, from fc, fu: tiles x T·T 4-byte words, tile t's C̃ and U row-major."""
    tiles = fc.shape[0]
    blocks = np.stack([np.ascontiguousarray(fc), np.ascontiguousarray(fu)], axis=1).view(np.uint8).reshape(-1, BLOCK)
    s = signs(seed)
    sk = np.concatenate([sketches(s, blocks[i:i + chunk]) for i in range(0, len(blocks), chunk)])
    return np.frombuffer(b"".join(receipts_of(seed, sk, 0)), np.uint8).reshape(tiles, -1)


def tile_receipts(seed: bytes, s: np.ndarray, c: np.ndarray, u: np.ndarray, T: int, t: int, tm: int, tn: int) -> bytes:
    """Tile t's 2 RB receipts, concatenated as the kernel writes them."""
    blocks = tile_blocks(c, u, T, tm, tn)
    return b"".join(receipts_of(seed, sketches(s, blocks), t * len(blocks)))


_K = [int(x, 16) for x in """
428a2f98d728ae22 7137449123ef65cd b5c0fbcfec4d3b2f e9b5dba58189dbbc 3956c25bf348b538 59f111f1b605d019 923f82a4af194f9b
ab1c5ed5da6d8118 d807aa98a3030242 12835b0145706fbe 243185be4ee4b28c 550c7dc3d5ffb4e2 72be5d74f27b896f 80deb1fe3b1696b1
9bdc06a725c71235 c19bf174cf692694 e49b69c19ef14ad2 efbe4786384f25e3 0fc19dc68b8cd5b5 240ca1cc77ac9c65 2de92c6f592b0275
4a7484aa6ea6e483 5cb0a9dcbd41fbd4 76f988da831153b5 983e5152ee66dfab a831c66d2db43210 b00327c898fb213f bf597fc7beef0ee4
c6e00bf33da88fc2 d5a79147930aa725 06ca6351e003826f 142929670a0e6e70 27b70a8546d22ffc 2e1b21385c26c926 4d2c6dfc5ac42aed
53380d139d95b3df 650a73548baf63de 766a0abb3c77b2a8 81c2c92e47edaee6 92722c851482353b a2bfe8a14cf10364 a81a664bbc423001
c24b8b70d0f89791 c76c51a30654be30 d192e819d6ef5218 d69906245565a910 f40e35855771202a 106aa07032bbd1b8 19a4c116b8d2d0c8
1e376c085141ab53 2748774cdf8eeb99 34b0bcb5e19b48a8 391c0cb3c5c95a63 4ed8aa4ae3418acb 5b9cca4f7763e373 682e6ff3d6b2b8a3
748f82ee5defb2fc 78a5636f43172f60 84c87814a1f0ab72 8cc702081a6439ec 90befffa23631e28 a4506cebde82bde9 bef9a3f7b2c67915
c67178f2e372532b ca273eceea26619c d186b8c721c0c207 eada7dd6cde0eb1e f57d4f7fee6ed178 06f067aa72176fba 0a637dc5a2c898a6
113f9804bef90dae 1b710b35131c471b 28db77f523047d84 32caab7b40c72493 3c9ebe0a15c9bebc 431d67c49c100d4c 4cc5d4becb3e42b6
597f299cfc657e2a 5fcb6fab3ad6faec 6c44198c4a475817""".split()]
IV = [0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1, 0x510E527FADE682D1,
      0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179]
_M = (1 << 64) - 1


def _rotr(x: int, n: int) -> int:
    return (x >> n | x << (64 - n)) & _M


def compress(state: list[int], block: bytes) -> list[int]:
    """FIPS 180-4 SHA-512's compression of one 128-byte block into `state` (8 words), returned as a new state."""
    w = [int.from_bytes(block[8 * i:8 * i + 8], "big") for i in range(16)]
    for i in range(16, 80):
        s0 = _rotr(w[i - 15], 1) ^ _rotr(w[i - 15], 8) ^ w[i - 15] >> 7
        s1 = _rotr(w[i - 2], 19) ^ _rotr(w[i - 2], 61) ^ w[i - 2] >> 6
        w.append((w[i - 16] + s0 + w[i - 7] + s1) & _M)
    a, b, c, d, e, f, g, h = state
    for i in range(80):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + (e & f ^ ~e & g) + _K[i] + w[i]) & _M
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + (a & b ^ a & c ^ b & c)) & _M
        a, b, c, d, e, f, g, h = (t1 + t2) & _M, a, b, c, (d + t1) & _M, e, f, g
    return [(x + y) & _M for x, y in zip(state, (a, b, c, d, e, f, g, h))]


def key(seed: bytes) -> np.ndarray:
    """The kernel's p15: SHA-512's state after prefix(seed) (8 u64, native order), then the sign bits, as bytes."""
    head = np.array(compress(IV, prefix(seed)), "<u8").view(np.uint8)
    return np.concatenate([head, sign_bits(seed).ravel()])
