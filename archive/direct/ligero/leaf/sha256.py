"""``leaf = sha256``: the operand rows hashed **in-circuit with SHA-256's compression function** under the core frame-v3 row
leaf ``sha256/row/v1`` (lane b-ligero-sha256): what the verifier ends up checking is the ordinary
``SHA-256(sha256_row_prefix(role, word_bits, n_words) || row_bytes)`` (``verity.ml.commitments.rowleaf.sha256_row_digest``) of
every committed row as the leaf value under the SHA-256 indexed trees -- anyone with ``hashlib`` recomputes it.

**Framing.**  The core schema's first block is the 64-byte public prefix (tag, role, word width, row length), so every row's
chain starts at a constant midstate ``M(role, word_bits, n_words)``; a K = 1536 row is 24 (E4M3) or 48 (BF16) whole
64-byte blocks, so the padding block (``0x80``, zeros, the message's bit length) is a public constant too.  **In-circuit we
compute exactly the data-block compressions and publish the chaining value after the last one** (``[header | H_0 .. H_7 as
16-bit limbs, low first]``, :data:`DIGEST_ELEMS` elements; the header is the constant ``role + 4 word_bits + 256 n_words``);
the verifier runs the padding compression natively in :meth:`leaf_bytes`.  Publishing that chaining value costs nothing in
soundness: two rows with the same one are a SHA-256 collision (same prefix, same length, so the same padding block).

**Columns.**  Whole blocks per column (the x4 folds: 128 bytes = 2 compressions per operand per column) are the intended
shape.  Half-block columns (the unfolded relations: 32 bytes per operand per column) pair up as in the BLAKE3 leaf: the even
column parks its 16 limbs in the message hold, the odd column compresses ``[hold | own bytes]``, the even column's
compression is discarded (``H_out = par ? new : H_in``) -- twice the rows per byte, supported so every relation composes.

**The gadget (rows per 64-byte block, BabyBear).**  A u32 is 32 boolean affine expressions.  A 2- or 3-input XOR of booleans
(Sigma0, Sigma1, sigma0, sigma1) is ONE row: ``c = [s >= 2]`` for ``s = x + y (+ z)`` (a ``sels`` program op), constrained by
``c^2 = c`` and ``(s - 2c)^2 = s - 2c``, so ``c`` is the carry ``floor(s / 2)`` and the XOR is the affine ``s - 2c``; the same
row is ``Maj(x, y, z)`` itself.  ``Ch(e, f, g) = g + e (f - g)``: one product row.  A sum mod 2^32 of ``n`` words is done on
16-bit limbs: the low limbs sum into ``16 + ceil(log2 n)`` bit rows, their carry bits feed the high limbs' sum (the
decomposition IS the modular addition constraint).  Per round: Sigma1, Ch, Sigma0, Maj 4 x 32 + the new ``e`` (6 addends)
and ``a`` (7 addends) 2 x 38 = 204 rows; per schedule word ``W_t`` (t = 16..63): sigma0, sigma1 2 x 32 + a 4-addend sum 36 =
100 rows; the feed-forward 8 x 34.  The message bits are the operand's own committed bit rows (``hashchain``'s
``hash.<a|b>[i].b<j>``, boolean-constrained there; checked here), so a block's 512 message bits cost no row.  The chaining
value entering a column is carried as 16 limbs (``H - M``: 0 at a chain start means the midstate) and re-derived into bits
once per column (256 rows).

Soundness of the arithmetic: every linear constraint is a bit decomposition ``e = sum 2^i b_i`` of an affine ``e`` whose
integer value (the addends are affine in booleans with limbs < 2^16) lies in ``[0, 2^nbits)`` with ``nbits <= 19``, so
``|e - sum| < 2^20 << p`` and the field identity is the integer identity.  XOR / Maj inputs are booleans (bit rows, the
operand bits, or affine XOR outputs whose booleanity is constrained).  Unsalted: a low-entropy row is preimage-searchable.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np

from verity.ml.commitments.rowleaf import SCHEMA_SHA256_ROW, sha256_row_digest, sha256_row_prefix

from ..compile import Expr, LigeroCtx, P, Quadratic
from .blake3 import _bits_raw, _const, _prod, row_bytes

__all__ = ["BLOCK_LEN", "CARRY_ELEMS", "DIGEST_ELEMS", "IV", "K", "K_VU", "Sha256Leaf", "census", "compress_np", "header",
           "leaf_bytes", "leaf_bytes_many", "midstate", "row_blocks"]

NAME = "sha256"
ASSUMPTION = ("SHA-256 collision resistance (frame-v3 sha256/row/v1 row leaves: the data-block compressions in-circuit, "
              "the chaining value published, the padding block by the verifier)")
SCHEMA = SCHEMA_SHA256_ROW
HASH_NAME = "sha256"
K_VU = 1536
BLOCK_LEN = 64
MASK32 = 0xFFFFFFFF
ROLE_NAMES = {1: "x-row", 2: "w-col"}   # rowleaf.ROLE_X / ROLE_W
DIGEST_ELEMS = 17                       # [header | H limbs]
HOLD = 16                               # carry offset of the half-block message hold (16 limbs)
PAR = 32                                # carry offset of the column parity (half-block columns)
CARRY_ELEMS = 33                        # [H - M (16 limbs) | msg hold (16) | par]
MALFORMED_TAG = b"verity/sha256-leaf/malformed-digest/v1\0"

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
K = (0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
     0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
     0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
     0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
     0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
     0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
     0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
     0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2)


# ---------------------------------------------------------------------------------------------------------------------
# Framing and the native reference
# ---------------------------------------------------------------------------------------------------------------------

def header(role: int, word_bits: int, n_words: int = K_VU) -> int:
    """Digest element 0: ``role + 4 word_bits + 256 n_words`` (the length and role the padding block and prefix depend on)."""
    return role + 4 * word_bits + 256 * n_words


def parse_header(h: int) -> tuple[int, int, int]:
    role, wb, n = h & 3, (h >> 2) & 63, h >> 8
    if role not in ROLE_NAMES or wb not in (8, 16, 32) or not n or (n * wb // 8) % BLOCK_LEN:
        raise ValueError(f"a SHA-256 leaf header is role (1, 2) + 4 word_bits (8, 16, 32) + 256 n_words (whole blocks), not {h}")
    return role, wb, n


def params_sha256() -> str:
    """The 32-byte parameters field of the v5 hash-auth block: the schema, the prefix rule and the digest framing."""
    return hashlib.sha256(b"sha256-leaf|schema=" + SCHEMA.encode() + b"|prefix=sha256_row_prefix(role,word_bits,n_words)"
                          b"|framing=header+cv-after-data-blocks-le16-limbs|padding=verifier|malformed=" + MALFORMED_TAG).hexdigest()


def _rotr(x: np.ndarray, n: int) -> np.ndarray:
    return (x >> np.uint32(n)) | (x << np.uint32(32 - n))


def compress_np(h: np.ndarray, block: np.ndarray) -> np.ndarray:
    """SHA-256's compression on N lanes: ``h`` (8, N), ``block`` (16, N) big-endian message words (uint32) -> (8, N)."""
    h = np.asarray(h, dtype=np.uint32)
    w = [np.asarray(x, dtype=np.uint32) for x in np.asarray(block, dtype=np.uint32)]
    with np.errstate(over="ignore"):
        for t in range(16, 64):
            x, y = w[t - 15], w[t - 2]
            s0 = _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> np.uint32(3))
            s1 = _rotr(y, 17) ^ _rotr(y, 19) ^ (y >> np.uint32(10))
            w.append(w[t - 16] + s0 + w[t - 7] + s1)
        a, b, c, d, e, f, g, hh = (h[i].copy() for i in range(8))
        for t in range(64):
            S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            ch = (e & f) ^ (~e & g)
            t1 = hh + S1 + ch + np.uint32(K[t]) + w[t]
            S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            mj = (a & b) ^ (a & c) ^ (b & c)
            hh, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + S0 + mj
        return (h + np.stack([a, b, c, d, e, f, g, hh])).astype(np.uint32)


def compress_one(h, block: bytes) -> tuple[int, ...]:
    """:func:`compress_np` of one lane in Python ints (a 64-byte ``block``): ~40x faster than a width-1 numpy call."""
    M = 0xFFFFFFFF
    r = lambda x, n: ((x >> n) | (x << (32 - n))) & M  # noqa: E731
    w = list(np.frombuffer(block, dtype=">u4").tolist())
    for t in range(16, 64):
        x, y = w[t - 15], w[t - 2]
        w.append((w[t - 16] + (r(x, 7) ^ r(x, 18) ^ (x >> 3)) + w[t - 7] + (r(y, 17) ^ r(y, 19) ^ (y >> 10))) & M)
    a, b, c, d, e, f, g, hh = (int(v) for v in h)
    for t in range(64):
        t1 = (hh + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K[t] + w[t]) & M
        t2 = ((r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & M, c, b, a, (t1 + t2) & M
    return tuple((int(x) + y) & M for x, y in zip(h, (a, b, c, d, e, f, g, hh)))


def _words_be(block: bytes) -> np.ndarray:
    return np.frombuffer(block, dtype=">u4").astype(np.uint32)


def midstate(role: int, word_bits: int, n_words: int = K_VU) -> tuple[int, ...]:
    """The chaining value after the schema's constant first block (``sha256_row_prefix``)."""
    pre = sha256_row_prefix(role, word_bits, n_words)
    h = compress_np(np.asarray(IV, dtype=np.uint32)[:, None], _words_be(pre)[:, None])
    return tuple(int(x) for x in h[:, 0])


def padding_block(n_bytes: int) -> bytes:
    """The final block of a ``64 + n_bytes``-byte message whose length is a multiple of 64: ``0x80``, zeros, bit length."""
    if n_bytes % BLOCK_LEN:
        raise ValueError("the row must be whole blocks")
    return b"\x80" + b"\0" * 55 + (8 * (BLOCK_LEN + n_bytes)).to_bytes(8, "big")


def row_blocks(rows: np.ndarray, word_bits: int) -> np.ndarray:
    """(R, n_words) words -> (R, n_blocks, 16) uint32 big-endian message words of the row bytes (whole blocks)."""
    rb = row_bytes(rows, word_bits)
    R, nb = rb.shape
    if nb % BLOCK_LEN:
        raise ValueError("rows must be whole 64-byte blocks (1536 or 3072 bytes are)")
    return np.ascontiguousarray(rb).view(">u4").reshape(R, nb // BLOCK_LEN, 16).astype(np.uint32)


def chain_states(rows: np.ndarray, word_bits: int, role: int) -> np.ndarray:
    """(R, n_blocks + 1, 8) uint32: the chaining value before the first data block (the midstate) and after every block."""
    blocks = row_blocks(rows, word_bits)
    R, nb = blocks.shape[:2]
    out = np.empty((R, nb + 1, 8), dtype=np.uint32)
    h = np.broadcast_to(np.asarray(midstate(role, word_bits, np.asarray(rows).shape[1]), dtype=np.uint32)[:, None], (8, R)).copy()
    out[:, 0] = h.T
    for j in range(nb):
        h = compress_np(h, blocks[:, j, :].T)
        out[:, j + 1] = h.T
    return out


_CHAIN_KERNEL = r"""
typedef unsigned char u8;

// row i (nblocks whole 64-byte blocks at rows + 64 nblocks i, 4-byte aligned): the chaining value before every block
// and after the last, from the midstate, as out[i][b][0..8)
extern "C" __global__
void sha256_row_chain(const u32* __restrict__ mid, const u8* __restrict__ rows, int nblocks, long long n,
                      u32* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 st[8], w[16];
    u32* o = out + i * (long long)(nblocks + 1) * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) { st[j] = mid[j]; o[j] = st[j]; }
    const u32* p = (const u32*)(rows + i * (long long)nblocks * 64);
    for (int b = 0; b < nblocks; ++b) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) w[j] = __byte_perm(p[16 * b + j], 0, 0x0123);
        sha256_compress(st, w);
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[8 * (b + 1) + j] = st[j];
    }
}
"""
_chain_fn = None


def chain_states_gpu(rb, role: int, word_bits: int, row_words: int):
    """:func:`chain_states` on the GPU: ``rb`` (R, n_bytes) uint8 cupy row bytes (whole blocks) -> (R, n_blocks + 1, 8)
    uint32 cupy, one thread per row (``hash_gpu/sha256_cuda.py``'s compression)."""
    global _chain_fn
    from .. import frame_gpu
    cp = frame_gpu.cp
    if _chain_fn is None:
        src = frame_gpu.fv3.sha256_cuda.SRC.split('extern "C"')[0] + _CHAIN_KERNEL
        _chain_fn = cp.RawModule(code=src, options=("-std=c++17",)).get_function("sha256_row_chain")
    rb = cp.ascontiguousarray(rb, dtype=cp.uint8)
    R, nbytes = rb.shape
    if nbytes % BLOCK_LEN:
        raise ValueError("rows must be whole 64-byte blocks (1536 or 3072 bytes are)")
    nb = nbytes // BLOCK_LEN
    out = cp.empty((R, nb + 1, 8), dtype=cp.uint32)
    if R:
        mid = cp.asarray(np.asarray(midstate(role, word_bits, row_words), dtype=np.uint32))
        _chain_fn((-(-R // 128),), (128,), (mid, rb, np.int32(nb), np.int64(R), out))
    return out


def _row_bytes_torch(rows, word_bits: int):
    """:func:`row_bytes` of (R, n_words) int64 torch words in range -> (R, n_bytes) uint8 torch (little-endian u16)."""
    import torch
    if word_bits == 8:
        return rows.to(torch.uint8).contiguous()
    if word_bits == 16:
        return torch.stack([rows & 0xFF, rows >> 8], -1).to(torch.uint8).reshape(rows.shape[0], -1).contiguous()
    raise ValueError(f"word_bits {word_bits}: rows are bytes or little-endian u16 words")


def _limbs_cp(h):
    """:func:`_limbs` of (..., 8) int64 cupy words."""
    import cupy as cp
    return cp.stack([h & 0xFFFF, h >> 16], -1).reshape(*h.shape[:-1], 16)


def _limbs(h: np.ndarray) -> np.ndarray:
    """(..., 8) u32 -> (..., 16) int64 limbs, word-major, low limb first."""
    h = np.asarray(h, dtype=np.int64)
    return np.stack([h & 0xFFFF, h >> 16], -1).reshape(*h.shape[:-1], 16)


def elems_to_cv(elems) -> tuple[np.ndarray, int, int, int]:
    e = np.asarray(elems, dtype=np.int64).reshape(-1)
    if e.shape[0] != DIGEST_ELEMS:
        raise ValueError(f"a SHA-256 leaf digest is {DIGEST_ELEMS} elements")
    role, wb, n = parse_header(int(e[0]))
    body = e[1:]
    if ((body < 0) | (body >= 1 << 16)).any():
        raise ValueError("a SHA-256 leaf digest is 16 limbs < 2^16")
    return (body[0::2] | (body[1::2] << 16)).astype(np.uint32), role, wb, n


def leaf_bytes(digest_row) -> bytes:
    """The leaf value of one row: the padding compression of the published chaining value, as the 32-byte big-endian
    SHA-256 digest -- ``sha256_row_digest(row, word_bits, role)``.  Total on field-element rows of the right length (the
    contract's conformance suite): a row that is not a well-formed frame (the gadget never pins one: its limbs are bit
    decompositions and its header a constant) maps to the tagged SHA-256 of its elements, a distinct domain that no row
    digest lands in except by collision.  Elements outside the field or a wrong length raise ``ValueError``."""
    e = np.asarray(digest_row, dtype=np.int64).reshape(-1)
    if e.shape[0] != DIGEST_ELEMS:
        raise ValueError(f"a SHA-256 leaf digest is {DIGEST_ELEMS} elements")
    if ((e < 0) | (e >= P)).any():
        raise ValueError("digest elements are BabyBear field elements")
    try:
        h, _role, wb, n = elems_to_cv(e)
    except ValueError:
        return hashlib.sha256(MALFORMED_TAG + e.astype("<u4").tobytes()).digest()
    return np.asarray(compress_one(h, padding_block(n * wb // 8)), dtype=">u4").tobytes()


def leaf_bytes_many(digests) -> list[bytes]:
    """:func:`leaf_bytes` of every row of ``digests`` (R, DIGEST_ELEMS): one lockstep padding compression when every row is a
    well-formed frame under one header (what :meth:`Sha256Leaf.native` returns); otherwise row by row."""
    e = np.asarray(digests, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != DIGEST_ELEMS:
        raise ValueError(f"a SHA-256 leaf digest is {DIGEST_ELEMS} elements")
    if e.shape[0] == 0:
        return []
    try:
        if (e[:, 0] != e[0, 0]).any():
            raise ValueError("mixed headers")
        _role, wb, n = parse_header(int(e[0, 0]))
    except ValueError:
        return [leaf_bytes(r) for r in e]
    body = e[:, 1:]
    if ((body < 0) | (body >= 1 << 16)).any():
        return [leaf_bytes(r) for r in e]
    h = (body[:, 0::2] | (body[:, 1::2] << 16)).astype(np.uint32).T
    pad = np.broadcast_to(_words_be(padding_block(n * wb // 8))[:, None], (16, e.shape[0]))
    out = np.ascontiguousarray(compress_np(h, pad).T).astype(">u4")
    return [bytes(r) for r in out]


# ---------------------------------------------------------------------------------------------------------------------
# The gadget
# ---------------------------------------------------------------------------------------------------------------------

class _W32:
    """A u32 in the circuit: 32 boolean affine expressions, least significant first (``None`` = a constant 0 bit of a
    shift); ``limbs`` the two 16-bit halves as affine expressions."""
    __slots__ = ("bits", "_limbs")

    def __init__(self, bits: list | None = None, limbs: tuple | None = None):
        self.bits = bits
        self._limbs = limbs

    @property
    def limbs(self) -> tuple[Expr, Expr]:
        if self._limbs is None:
            ctx = next(b.ctx for b in self.bits if isinstance(b, Expr))
            lo = sum((b * (1 << i) for i, b in enumerate(self.bits[:16]) if b is not None), _const(ctx, 0))
            hi = sum((b * (1 << i) for i, b in enumerate(self.bits[16:]) if b is not None), _const(ctx, 0))
            self._limbs = (lo, hi)
        return self._limbs

    @staticmethod
    def const(ctx: LigeroCtx, v: int) -> "_W32":
        return _W32(limbs=(_const(ctx, v & 0xFFFF), _const(ctx, v >> 16)))

    def rotr(self, n: int) -> list:
        return [self.bits[(i + n) % 32] for i in range(32)]

    def shr(self, n: int) -> list:
        return [self.bits[i + n] if i + n < 32 else None for i in range(32)]


def _carry(ctx: LigeroCtx, s: Expr, name: str) -> Expr:
    """``c = [s >= 2]`` for ``s`` a sum of 2 or 3 booleans: one row, ``c^2 = c`` and ``(s - 2c)^2 = s - 2c`` (so ``c =
    floor(s / 2)``, ``s - 2c`` = the XOR, ``c`` = the majority)."""
    c = ctx._row("sel", name, 0, 2, (s, 2))
    idx = next(iter(c.lin))
    ctx.sys.program.append(("sels", idx, [2], [3], s))
    ctx.sys.quadratic.append(Quadratic(f"{name}.bool", c, c, c))
    x = s - 2 * c
    ctx.sys.quadratic.append(Quadratic(f"{name}.xor", x, x, x))
    return c


def _xor(ctx: LigeroCtx, operands: Sequence[list], scope: str) -> _W32:
    """Bitwise XOR of 2 or 3 bit lists (``None`` entries are constant 0)."""
    out = []
    for i in range(32):
        xs = [op[i] for op in operands if op[i] is not None]
        if len(xs) == 1:
            out.append(xs[0])
            continue
        s = sum(xs[1:], xs[0])
        out.append(s - 2 * _carry(ctx, s, f"{scope}.x{i}"))
    return _W32(out)


def _maj(ctx: LigeroCtx, a: _W32, b: _W32, c: _W32, scope: str) -> _W32:
    return _W32([_carry(ctx, a.bits[i] + b.bits[i] + c.bits[i], f"{scope}.m{i}") for i in range(32)])


def _ch(ctx: LigeroCtx, e: _W32, f: _W32, g: _W32, scope: str) -> _W32:
    return _W32([g.bits[i] + _prod(ctx, e.bits[i], f.bits[i] - g.bits[i], scope + ".ch") for i in range(32)])


def _add(ctx: LigeroCtx, terms: Sequence[_W32], name: str) -> _W32:
    """Sum mod 2^32 of ``n`` words on 16-bit limbs; the output as 32 fresh bit rows (+ ``2 ceil(log2 n)`` carry rows)."""
    cb = max(1, math.ceil(math.log2(len(terms))))
    lo = sum((t.limbs[0] for t in terms), _const(ctx, 0))
    lo_bits = _bits_raw(ctx, lo, 16 + cb, name + ".lo")
    carry = sum((lo_bits[16 + i] * (1 << i) for i in range(cb)), _const(ctx, 0))
    hi = sum((t.limbs[1] for t in terms), carry)
    hi_bits = _bits_raw(ctx, hi, 16 + cb, name + ".hi")
    return _W32(lo_bits[:16] + hi_bits[:16])


def compress_gadget(ctx: LigeroCtx, h: list[_W32], w: list[_W32], scope: str) -> list[_W32]:
    """SHA-256's compression on circuit words: ``h`` 8 words (bits; ``d`` / ``h`` may be limb-only), ``w`` the 16 message
    words (bits).  Returns the 8 output words (bits)."""
    W = list(w)
    for t in range(16, 64):
        x, y = W[t - 15], W[t - 2]
        s0 = _xor(ctx, [x.rotr(7), x.rotr(18), x.shr(3)], f"{scope}.w{t}.s0")
        s1 = _xor(ctx, [y.rotr(17), y.rotr(19), y.shr(10)], f"{scope}.w{t}.s1")
        W.append(_add(ctx, [s1, W[t - 7], s0, W[t - 16]], f"{scope}.w{t}"))
    a, b, c, d, e, f, g, hh = h
    for t in range(64):
        s = f"{scope}.r{t}"
        S1 = _xor(ctx, [e.rotr(6), e.rotr(11), e.rotr(25)], s + ".S1")
        ch = _ch(ctx, e, f, g, s)
        S0 = _xor(ctx, [a.rotr(2), a.rotr(13), a.rotr(22)], s + ".S0")
        mj = _maj(ctx, a, b, c, s)
        kt = _W32.const(ctx, K[t])
        e_new = _add(ctx, [d, hh, S1, ch, kt, W[t]], s + ".e")
        a_new = _add(ctx, [hh, S1, ch, kt, W[t], S0, mj], s + ".a")
        hh, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new
    return [_add(ctx, [h[i], v], f"{scope}.ff{i}") for i, v in enumerate((a, b, c, d, e, f, g, hh))]


def _boolean_rows(ctx: LigeroCtx) -> set:
    """Rows the system constrains boolean (``r * r = r`` on the single row), memoized per system size."""
    memo = getattr(ctx, "_sha256_bool_rows", None)
    if memo is not None and memo[0] == len(ctx.sys.quadratic):
        return memo[1]
    out = set(memo[1]) if memo is not None else set()
    for q in ctx.sys.quadratic[memo[0] if memo is not None else 0:]:
        if len(q.a.lin) == 1 and q.a.c == 0 and q.a.key() == q.b.key() == q.c.key():
            (j, v), = q.a.lin.items()
            if v % P == 1:
                out.add(j)
    ctx._sha256_bool_rows = (len(ctx.sys.quadratic), out)
    return out


def _operand_bits(ctx: LigeroCtx, words: Sequence[Expr], word_bits: int, scope: str) -> list[list[Expr]]:
    """Each operand word's bits, least significant first: the committed bit rows the word is built from (``sum_j 2^j b_j``
    over rows the system already constrains boolean), else fresh bit rows."""
    boolean = _boolean_rows(ctx)
    out = []
    for i, w in enumerate(words):
        by_coef = {v % P: j for j, v in w.lin.items()}
        if (w.c % P == 0 and len(w.lin) == word_bits and set(by_coef) == {1 << j for j in range(word_bits)}
                and all(j in boolean for j in w.lin)):
            out.append([Expr(ctx, {by_coef[1 << j]: 1}, 0) for j in range(word_bits)])
        else:
            out.append(_bits_raw(ctx, w, word_bits, f"{scope}.op{i}"))
    return out


def _byte_bits(op_bits: list[list[Expr]], word_bits: int) -> list[list[Expr]]:
    """The column's bytes (row order, little-endian words) as 8 bits each, least significant first."""
    per = word_bits // 8
    return [wb[8 * q:8 * q + 8] for wb in op_bits for q in range(per)]


def _message_words(byte_bits: list[list[Expr]]) -> list[_W32]:
    """64 bytes -> 16 big-endian message words: bit i of word j is bit i % 8 of byte 4 j + 3 - i // 8."""
    if len(byte_bits) != BLOCK_LEN:
        raise ValueError(f"a block is 64 bytes, not {len(byte_bits)}")
    return [_W32([byte_bits[4 * j + 3 - i // 8][i % 8] for i in range(32)]) for j in range(16)]


def _limb_expr(ctx: LigeroCtx, lo_byte: list, hi_byte: list) -> Expr:
    return sum((b * (1 << i) for i, b in enumerate(lo_byte + hi_byte)), _const(ctx, 0))


class Sha256Leaf:
    """The ``LeafScheme`` (``leaf/base.py``) for ``sha256/row/v1``.  One instance serves every row format (8 / 16-bit words,
    whole- or half-block columns); the width enters through ``word_bits`` / ``row_words`` per call.

    Carried state per operand (:data:`CARRY_ELEMS` rows, 0 at a chain start): ``[H - M (16 limbs) | msg hold (16) | par]``:
    the chaining value entering the column minus the role's midstate, the previous column's 16 byte-pair limbs and the
    column parity (half-block columns; carried unchanged, i.e. 0, with whole blocks).

    Digest (:data:`DIGEST_ELEMS` public pins per operand at the chain end): ``[role + 4 word_bits + 256 n_words | H limbs]``;
    the verifier's :meth:`leaf_bytes` runs the padding compression and that 32-byte digest is the SHA-256 tree leaf."""

    name = NAME
    assumption = ASSUMPTION
    digest_elems = DIGEST_ELEMS
    carry_elems = CARRY_ELEMS
    schema = SCHEMA
    hash_name = HASH_NAME
    params_sha256 = params_sha256()
    test_only = False
    privacy_note = "deterministic, unsalted: hides only up to preimage search on a row"

    # --- committer / verifier side -----------------------------------------------------------------------------------
    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, DIGEST_ELEMS) int64: the header and the chaining value after the row's last data block as 16-bit limbs."""
        rows = np.asarray(rows, dtype=np.int64)
        cv = chain_states(rows, word_bits, role)[:, -1]
        out = np.empty((rows.shape[0], DIGEST_ELEMS), dtype=np.int64)
        out[:, 0] = header(role, word_bits, rows.shape[1])
        out[:, 1:] = _limbs(cv)
        return out

    def leaf_bytes(self, digest_row) -> bytes:
        return leaf_bytes(digest_row)

    def leaf_bytes_many(self, digests) -> list[bytes]:
        return leaf_bytes_many(digests)

    def reference_digest(self, rows: np.ndarray, *, word_bits: int, role: int) -> list[bytes]:
        """``verity.ml.commitments.rowleaf.sha256_row_digest`` per row -- the core reference :meth:`leaf_bytes` of
        :meth:`native` must equal."""
        return [sha256_row_digest([int(x) for x in r], word_bits, role) for r in np.asarray(rows, dtype=np.int64)]

    # --- the in-circuit hash -----------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int = K_VU,
               carry_in: Sequence[Expr], first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        """One column, one operand role: compress this column's blocks into the carried chaining value (see the class
        doc).  ``first`` / ``last`` are ignored (column-uniform circuit)."""
        if len(carry_in) != CARRY_ELEMS:
            raise ValueError(f"gadget: {len(carry_in)} carried rows, the scheme carries {CARRY_ELEMS}")
        k = len(words)
        col_bytes = k * word_bits // 8
        half = 2 * col_bytes == BLOCK_LEN
        if not half and (col_bytes % BLOCK_LEN or not col_bytes):
            raise ValueError(f"gadget: a column carries half a block or whole 64-byte blocks per operand, not {col_bytes} bytes")
        if row_words % k or (row_words * word_bits // 8) % BLOCK_LEN:
            raise ValueError("gadget: the row must be whole blocks tiled by the columns")
        nblk = 1 if half else col_bytes // BLOCK_LEN
        scope = f"sha256.{ROLE_NAMES[role]}"
        one = _const(ctx, 1)
        M = midstate(role, word_bits, row_words)
        m_limbs = [v for x in M for v in (x & 0xFFFF, x >> 16)]
        own = _byte_bits(_operand_bits(ctx, words, word_bits, scope), word_bits)
        # the chaining value entering the column: carried H - M, re-derived into bits (the range check of the carry)
        h_in_limbs = [carry_in[x] + m_limbs[x] for x in range(16)]
        h = [_W32(_bits_raw(ctx, h_in_limbs[2 * w], 16, f"{scope}.h{w}.lo") + _bits_raw(ctx, h_in_limbs[2 * w + 1], 16,
                                                                                         f"{scope}.h{w}.hi"))
             for w in range(8)]
        hold_in = list(carry_in[HOLD:HOLD + 16])
        par = carry_in[PAR]
        if half:
            held = []
            for q in range(16):
                b = _bits_raw(ctx, hold_in[q], 16, f"{scope}.hold{q}")
                held += [b[:8], b[8:]]
            h_new = compress_gadget(ctx, h, _message_words(held + own), f"{scope}.blk0")
            # the even (parking) column keeps the carried value; the odd one takes the compression
            out_limbs = [h_in_limbs[2 * w + i] + _prod(ctx, par, h_new[w].limbs[i] - h_in_limbs[2 * w + i], scope + ".hout")
                         for w in range(8) for i in range(2)]
            hold_out = [_prod(ctx, _limb_expr(ctx, own[2 * q], own[2 * q + 1]), one, scope + ".holdout") for q in range(16)]
            par_out = one - par
        else:
            for t in range(nblk):
                h = compress_gadget(ctx, h, _message_words(own[BLOCK_LEN * t:BLOCK_LEN * (t + 1)]), f"{scope}.blk{t}")
            out_limbs = [_prod(ctx, h[w].limbs[i], one, scope + ".hout") for w in range(8) for i in range(2)]
            hold_out, par_out = hold_in, par
        digest = [_const(ctx, header(role, word_bits, row_words))] + out_limbs
        carry_out = [out_limbs[x] - m_limbs[x] for x in range(16)] + list(hold_out) + [par_out]
        return digest, carry_out

    def carry_values(self, rows: np.ndarray, *, word_bits: int, role: int, steps: int) -> np.ndarray:
        """The honest carried rows of every column of every row: (R, steps, CARRY_ELEMS) int64 mod p -- ``carry_in`` of
        column ``j`` = the gadget's ``carry_out`` of column ``j - 1`` (0 at ``j = 0``)."""
        rows = np.asarray(rows, dtype=np.int64)
        R, row_words = rows.shape
        states = chain_states(rows, word_bits, role)          # (R, nb + 1, 8)
        nb = states.shape[1] - 1
        half = steps == 2 * nb
        if not half and nb % steps:
            raise ValueError("whole (or half) blocks per column")
        car = np.zeros((R, steps, CARRY_ELEMS), dtype=np.int64)
        m = _limbs(np.asarray(midstate(role, word_bits, row_words), dtype=np.uint32))
        blk_before = (np.arange(steps) // 2) if half else np.arange(steps) * (nb // steps)   # blocks done before column j
        car[:, :, :16] = (_limbs(states[:, blk_before]) - m) % P
        if half:
            rb = row_bytes(rows, word_bits).reshape(R, steps, 32)
            car[:, 1:, HOLD:HOLD + 16] = np.ascontiguousarray(rb[:, :-1, :]).view("<u2").astype(np.int64)
            car[:, :, PAR] = np.arange(steps) % 2
        return car

    def row_sponges(self, words, *, word_bits: int, role: int, row_words: int, n: int, steps: int):
        """The committer's chains of ``n`` whole rows at once on the GPU (:func:`chain_states_gpu`): ``words`` (n * steps,
        k) int64 CUDA torch, row-major -> ``(carry (n, steps, CARRY_ELEMS), digests (n, DIGEST_ELEMS))`` int64 torch =
        :meth:`carry_values` / :meth:`native`; ``None`` off CUDA or without the GPU committer (the caller falls back to
        :meth:`chain_witness`)."""
        from .. import frame_gpu
        if not words.is_cuda or not frame_gpu.available():
            return None
        import torch
        cp = frame_gpu.cp
        rows = words.reshape(n, row_words)
        if bool(((rows < 0) | (rows >= 1 << word_bits)).any()):
            raise ValueError("word out of range")
        with frame_gpu._torch_stream():
            rb = _row_bytes_torch(rows, word_bits)
            st = chain_states_gpu(cp.asarray(rb), role, word_bits, row_words).astype(cp.int64)    # (n, nb + 1, 8)
            nb = st.shape[1] - 1
            half = steps == 2 * nb
            if not half and nb % steps:
                raise ValueError("whole (or half) blocks per column")
            blk_before = (cp.arange(steps) // 2) if half else cp.arange(steps) * (nb // steps)
            m = cp.asarray(_limbs(np.asarray(midstate(role, word_bits, row_words), dtype=np.uint32)))
            car = cp.zeros((n, steps, CARRY_ELEMS), dtype=cp.int64)
            car[:, :, :16] = (_limbs_cp(st[:, blk_before]) - m) % P
            if half:
                prev = cp.asarray(rb).reshape(n, steps, 16, 2).astype(cp.int64)
                car[:, 1:, HOLD:HOLD + 16] = prev[:, :-1, :, 0] | (prev[:, :-1, :, 1] << 8)
                car[:, :, PAR] = cp.arange(steps)[None, :] % 2
            dg = cp.empty((n, DIGEST_ELEMS), dtype=cp.int64)
            dg[:, 0] = header(role, word_bits, row_words)
            dg[:, 1:] = _limbs_cp(st[:, -1])
            return torch.as_tensor(car, device=words.device), torch.as_tensor(dg, device=words.device)

    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """``words`` (l, k) int64 torch (VU v in columns [v steps, (v + 1) steps), pad columns 0) -> (carry (CARRY_ELEMS,
        l) torch: the carried rows entering every column, digests (n_vus, DIGEST_ELEMS) torch)."""
        import torch
        w = words.detach().cpu().numpy().astype(np.int64)
        k = w.shape[1]
        if steps * k != row_words:
            raise ValueError("chain_witness: steps x k != row_words")
        rows = w[:n_vus * steps].reshape(n_vus, row_words)
        car = self.carry_values(rows, word_bits=word_bits, role=role, steps=steps)
        carry = np.zeros((CARRY_ELEMS, l), dtype=np.int64)
        carry[:, :n_vus * steps] = car.reshape(n_vus * steps, CARRY_ELEMS).T
        dg = self.native(rows, word_bits=word_bits, role=role)
        dev = words.device
        return torch.as_tensor(carry, device=dev), torch.as_tensor(dg, device=dev)

    def witness(self):
        return None

    def census(self) -> dict:
        return {"sha256_rows_per_block": ROWS_PER_BLOCK, "sha256_framing": "data blocks in-circuit, CV published, padding by the verifier",
                "sha256_schema": SCHEMA}


ROWS_PER_BLOCK = 64 * 204 + 48 * 100 + 8 * 34     # rounds, schedule, feed-forward (the message bits are the operand's)

try:                                                          # the leaf registry (lane leaf-iface)
    from . import registry as _registry
    _registry.register(Sha256Leaf)
except ImportError:                                           # pragma: no cover
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------------------------------------------------

def census(word_bits: int = 8, rel_name: str | None = None) -> dict:
    """Rows per 64-byte block (one bare compression compiled on private inputs) and per column / VU of a composed relation
    (``rel_name``, default the x4 fold of the word width's Ada / Hopper relation)."""
    from ..relations import FP8_ADA_RELATION, relation
    from .blake3 import _count
    out: dict = {"word_bits": word_bits}
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    h = [_W32([ctx.hint(f"h{w}.b{i}") for i in range(32)]) for w in range(8)]
    msg = [_W32([ctx.hint(f"m{w}.b{i}") for i in range(32)]) for w in range(16)]
    m0 = ctx.sys.m
    compress_gadget(ctx, h, msg, "c")
    out["per_block_rows"] = ctx.sys.m - m0
    out["per_block_rows_formula"] = ROWS_PER_BLOCK
    out["per_block_detail"] = _count(ctx.sys)
    from .. import hashchain
    rel = relation(rel_name or {8: "fp8-ada-x4", 16: "bf16-hopper-x4"}[word_bits])
    hr = hashchain.compose(rel, NAME)
    out["relation"] = rel.name
    out["steps"] = rel.steps
    out["rows_per_column"] = hr.census["rows_total"]
    out["rows_base"] = hr.census["rows_base"]
    out["rows_hash_per_column"] = hr.census["rows_hash"]
    out["rows_per_vu"] = hr.census["rows_total"] * rel.steps
    return out


if __name__ == "__main__":
    import json
    import sys as _sys
    print(json.dumps(census(int(_sys.argv[1]) if len(_sys.argv) > 1 else 8, _sys.argv[2] if len(_sys.argv) > 2 else None), indent=1))
