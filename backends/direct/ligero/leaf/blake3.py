"""``leaf = blake3``: the operand rows hashed **in-circuit with BLAKE3's compression function** (lane blake3-leaf).

The conservative control of the leaf campaign: the same ``included-hash`` column as the Poseidon2 leaf (``hashchain.py``)
under a standard hash.  What the verifier ends up checking is the ordinary ``blake3.blake3(row_bytes, key=KEY).digest()``
of every committed row (keyed mode, one 32-byte key per word width; see :func:`key_for`) as the leaf value
under the SHA-256 indexed trees -- anyone with the ``blake3`` package recomputes it.

**Framing (decided; the rest of the module follows from it).**  A row is 1536 words: 1536 bytes (E4M3) or 3072 bytes
(BF16, little-endian u16) -- more than one BLAKE3 chunk (1024 B), so the standard chunk tree applies: chunk ``c`` =
bytes ``[1024 c, 1024 c + 1024)`` (16 blocks of 64 B; the last chunk of an E4M3 row has 8), its chaining value ``CV_c``
is 16 (8 for the tail) compressions with ``counter = c``, ``CHUNK_START`` on the first block and ``CHUNK_END`` on the last;
the root is the standard left-heavy parent fold of the chunk CVs (``PARENT``, ``ROOT`` on the top).  **In-circuit we compute
exactly the chunk compressions and publish the chunk CVs** (``digest_elems = 16 x n_chunks`` BabyBear elements: every u32
as two 16-bit limbs, low first); the 1-2 parent compressions are done natively by the verifier inside :meth:`leaf_bytes`
(one compression per row; microseconds).  Why: the parent nodes cannot be placed in a column-uniform chain (they run once
per row, after the last block), and publishing the CVs costs nothing in soundness -- two rows with the same CVs are a
BLAKE3 chunk collision, two CV pairs with the same root are a parent-compression collision; both are BLAKE3 collisions.
The relation binds the words to the CVs, the verifier binds the CVs to the leaf, the leaf is the real BLAKE3 digest.

**Granularity.**  One column of the chain must hold one whole block per operand (the compression needs all 16 message
words in round 1), i.e. 64 bytes of x and 64 of W: that is the ``x2`` fold of the fp8 / bf16 relations (``relations._folded``,
2 instruction steps per column; K = 64 E4M3 bytes or 32 BF16 words per operand per column), 24 (E4M3) or 48 (BF16) columns
per VU, one block of each operand per column.  The unfolded relations (32 B per operand per column) would need one
compression per two columns, which a column-uniform circuit can only do by running a full compression in every column and
using half of them (2x waste) -- measured in the census, rejected.

**The gadget (rows per 64-byte block, BabyBear).**  A u32 is 32 boolean rows; XOR is ``x + y - 2 x y`` (one product row
per bit, rotations are free re-indexing); an addition mod 2^32 is done on two 16-bit limbs (``2^33 > p``: the whole word
does not fit a linear constraint): the low limbs sum into ``16 + ceil(log2 n)`` bit rows (``n`` addends: 17 rows for
``a + b``, 18 for ``a + b + m``), the carry bits feed the high limb's sum -- the ``_bits`` decomposition IS the modular
addition constraint.  Per G: 4 additions (2 x 36 + 2 x 34 rows) + 4 XORs (4 x 32) = 268 rows; 56 G's per compression
(7 rounds x 8) = 15 008, minus the XORs with the constant ``d`` words of round 1, plus the 8 finalisation XORs
(``h_i = v_i ^ v_{i+8}``).  The lookup alternative (``ctx.lookup`` is a ONE-HOT selector over the table's distinct rows:
a 4-bit XOR table costs 256 selector rows + 8 key bits per nibble = 2 112 rows per u32 XOR vs 32) is 66x worse and was
not built; :func:`census` measures both.

Per column and operand role the gadget also selects the chunk-start CV (``cs ? KEY : carried``: 16 products), re-derives
the bits of ``cv[4..7]`` (the ``b`` words of round 1 enter an XOR: 128 bit rows), copies the output CV into 16 carried
rows, and holds the finished chunk CVs in ``16 (n_chunks - 1)`` carried rows (16 products each) until the chain end,
where the compose layer pins ``[CV_0 | CV_1 | ...]`` against the public digest.  Public layout pins (functions of the
column index, the verifier computes them): ``blake3.cs`` (chunk start), ``blake3.ce`` (chunk end), ``blake3.cnt<i>``
(chunk counter bits), ``blake3.hold<c>`` (this column ends chunk ``c < n_chunks - 1``).

Soundness of the arithmetic: every linear constraint here is a bit decomposition ``e = sum 2^i b_i`` of an affine ``e``
whose integer value (the addends are affine in booleans) lies in ``[0, 2^nbits)``; ``|e - sum| < 2^19 << p``, so the field
identity is the integer identity and the bits are the sum's bits.  XOR inputs are booleans by construction (bit rows,
public {0, 1} pins, constants, or XOR outputs of booleans with the exact product enforced by its quadratic constraint).
Unsalted (as the Poseidon2 leaf): the digest of a low-entropy row is preimage-searchable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..compile import Expr, LigeroCtx, Linear, P, Quadratic

__all__ = ["BLOCK_LEN", "CHUNK_LEN", "IV", "KEY_LEN", "K_VU", "Blake3Leaf", "census", "chunk_cvs_np", "compress_np",
           "digest_elems_for", "key_for", "leaf_bytes", "n_chunks", "root_from_cvs", "row_bytes"]

NAME = "blake3"
ASSUMPTION = "BLAKE3 collision resistance (keyed mode; standard ARX hash, chunk CVs published, parent fold by the verifier)"

IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
CHUNK_START, CHUNK_END, PARENT, ROOT, KEYED_HASH = 1, 2, 4, 8, 16
BLOCK_LEN, CHUNK_LEN, KEY_LEN = 64, 1024, 32
ROUNDS = 7
K_VU = 1536                      # words per committed row (K of the GEMM)
ROLE_NAMES = {1: "x-row", 2: "w-col"}   # poseidon2_babybear.ROLE_X / ROLE_W
MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------------------------------------------------

def key_for(word_bits: int) -> bytes:
    """The 32-byte BLAKE3 key of a row format: domain separation of the leaf from any other BLAKE3 use and of the row
    formats (a 1536-byte E4M3 row and a 768-word BF16 row are the same bytes), at zero circuit cost (the key words replace
    IV as the first chaining value of every chunk AND of every parent node; ``KEYED_HASH`` is a constant flag bit).  The
    operand role (x row / W column) is NOT in the key: the two roles live in different trees with different bindings, and
    ``leaf_bytes`` (the verifier's parent fold) must not need the role."""
    tag = f"verity/blake3-leaf/v1/w{word_bits}".encode()
    if len(tag) > KEY_LEN:
        raise ValueError("key tag too long")
    return tag.ljust(KEY_LEN, b"\0")


def key_words(word_bits: int) -> tuple[int, ...]:
    k = key_for(word_bits)
    return tuple(int.from_bytes(k[4 * i:4 * i + 4], "little") for i in range(8))


def row_bytes(rows: np.ndarray, word_bits: int) -> np.ndarray:
    """(R, n_words) words -> (R, n_words * word_bits / 8) uint8: little-endian words, the bytes BLAKE3 hashes."""
    w = np.asarray(rows, dtype=np.int64)
    if word_bits == 8:
        return w.astype(np.uint8)
    if word_bits == 16:
        return np.ascontiguousarray(w.astype("<u2")).view(np.uint8).reshape(w.shape[0], -1)
    raise ValueError(f"word_bits {word_bits}: rows are bytes or little-endian u16 words")


def n_chunks(word_bits: int, n_words: int = K_VU) -> int:
    return -(-(n_words * word_bits // 8) // CHUNK_LEN)


def digest_elems_for(word_bits: int, n_words: int = K_VU) -> int:
    """Chunk CVs as 16-bit limbs: 16 elements per chunk (2 chunks for E4M3 rows, 3 for BF16)."""
    return 16 * n_chunks(word_bits, n_words)


def _rotr(x: np.ndarray, n: int) -> np.ndarray:
    return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))) & np.uint32(MASK32)


def _g_np(v, a, b, c, d, mx, my):
    v[a] = v[a] + v[b] + mx
    v[d] = _rotr(v[d] ^ v[a], 16)
    v[c] = v[c] + v[d]
    v[b] = _rotr(v[b] ^ v[c], 12)
    v[a] = v[a] + v[b] + my
    v[d] = _rotr(v[d] ^ v[a], 8)
    v[c] = v[c] + v[d]
    v[b] = _rotr(v[b] ^ v[c], 7)


def compress_np(cv: np.ndarray, block: np.ndarray, counter, block_len, flags) -> np.ndarray:
    """The BLAKE3 compression function on N lanes: ``cv`` (8, N), ``block`` (16, N) u32 message words, ``counter`` /
    ``block_len`` / ``flags`` scalars or (N,) -> the 16-word output state (16, N) uint32 (before the ``v_i ^ v_{i+8}``
    truncation; :func:`_cv_of` takes the chaining value)."""
    cv = np.asarray(cv, dtype=np.uint32)
    N = cv.shape[1]
    m = np.asarray(block, dtype=np.uint32).copy()
    counter = np.broadcast_to(np.asarray(counter, dtype=np.uint64), (N,))
    v = np.empty((16, N), dtype=np.uint32)
    v[:8] = cv
    v[8:12] = np.asarray(IV[:4], dtype=np.uint32)[:, None]
    v[12] = (counter & np.uint64(MASK32)).astype(np.uint32)
    v[13] = (counter >> np.uint64(32)).astype(np.uint32)
    v[14] = np.broadcast_to(np.asarray(block_len, dtype=np.uint32), (N,))
    v[15] = np.broadcast_to(np.asarray(flags, dtype=np.uint32), (N,))
    with np.errstate(over="ignore"):
        for r in range(ROUNDS):
            _g_np(v, 0, 4, 8, 12, m[0], m[1])
            _g_np(v, 1, 5, 9, 13, m[2], m[3])
            _g_np(v, 2, 6, 10, 14, m[4], m[5])
            _g_np(v, 3, 7, 11, 15, m[6], m[7])
            _g_np(v, 0, 5, 10, 15, m[8], m[9])
            _g_np(v, 1, 6, 11, 12, m[10], m[11])
            _g_np(v, 2, 7, 8, 13, m[12], m[13])
            _g_np(v, 3, 4, 9, 14, m[14], m[15])
            if r != ROUNDS - 1:
                m = m[list(MSG_PERMUTATION)]
    return v


def _cv_of(state: np.ndarray) -> np.ndarray:
    return state[:8] ^ state[8:16]


def _blocks_of(rb: np.ndarray) -> np.ndarray:
    """(R, nbytes) bytes -> (R, n_blocks, 16) u32 little-endian message words (nbytes a multiple of 64)."""
    R, nb = rb.shape
    if nb % BLOCK_LEN:
        raise ValueError("rows must be whole blocks (1536 or 3072 bytes are)")
    return np.ascontiguousarray(rb).view("<u4").reshape(R, nb // BLOCK_LEN, 16).astype(np.uint32)


def block_flags(word_bits: int, n_words: int = K_VU) -> list[tuple[int, int]]:
    """Per block of a row: ``(chunk counter, flags)`` of its compression (the keyed-hash bit included)."""
    nb = n_words * word_bits // 8 // BLOCK_LEN
    per_chunk = CHUNK_LEN // BLOCK_LEN
    out = []
    for j in range(nb):
        c, i = divmod(j, per_chunk)
        last = (i == per_chunk - 1) or (j == nb - 1)
        out.append((c, KEYED_HASH | (CHUNK_START if i == 0 else 0) | (CHUNK_END if last else 0)))
    return out


def chunk_cvs_np(rows: np.ndarray, word_bits: int, role: int) -> tuple[np.ndarray, np.ndarray]:
    """All rows in lockstep: (block_out (R, n_blocks, 8) uint32: the chaining value after every block (the input of the
    next block of the same chunk), cvs (R, n_chunks, 8): the chunk chaining values)."""
    rb = row_bytes(rows, word_bits)
    R, n_words = rb.shape[0], np.asarray(rows).shape[1]
    blocks = _blocks_of(rb)                                   # (R, nb, 16)
    nb = blocks.shape[1]
    flags = block_flags(word_bits, n_words)
    key = np.asarray(key_words(word_bits), dtype=np.uint32)[:, None]
    h = np.broadcast_to(key, (8, R)).copy()
    out = np.empty((R, nb, 8), dtype=np.uint32)
    cvs = []
    for j in range(nb):
        c, f = flags[j]
        if f & CHUNK_START:
            h = np.broadcast_to(key, (8, R)).copy()
        h = _cv_of(compress_np(h, blocks[:, j, :].T, c, BLOCK_LEN, f))
        out[:, j, :] = h.T
        if f & CHUNK_END:
            cvs.append(h.T.copy())
    return out, np.stack(cvs, 1)


def root_from_cvs(cvs: np.ndarray, word_bits: int) -> np.ndarray:
    """(R, n_chunks, 8) chunk CVs -> (R, 32) uint8: the standard BLAKE3 tree root (left-heavy parent fold with the key as
    the parents' chaining value, ``ROOT`` on the top node; a single chunk would be the root itself with ``ROOT`` set on its
    last block -- not this row size)."""
    cvs = np.asarray(cvs, dtype=np.uint32)
    R, nch, _ = cvs.shape
    if nch < 2:
        raise ValueError("rows of one chunk are not framed here (the chunk's last block would carry ROOT)")
    nodes = [cvs[:, c, :].T for c in range(nch)]              # (8, R) each
    iv = np.asarray(key_words(word_bits), dtype=np.uint32)[:, None].repeat(R, 1)

    def fold(ns: list, top: bool) -> np.ndarray:
        """BLAKE3's tree: the left subtree holds the largest power of two of leaves strictly below the count."""
        if len(ns) == 1:
            return ns[0]
        k = 1 << ((len(ns) - 1).bit_length() - 1)
        st = compress_np(iv, np.concatenate([fold(ns[:k], False), fold(ns[k:], False)], 0), 0, BLOCK_LEN,
                         KEYED_HASH | PARENT | (ROOT if top else 0))
        return _cv_of(st)

    root = fold(nodes, True)                                   # (8, R)
    return np.ascontiguousarray(root.T.astype("<u4")).view(np.uint8).reshape(R, 32)


def cvs_to_elems(cvs: np.ndarray) -> np.ndarray:
    """(R, n_chunks, 8) u32 -> (R, 16 n_chunks) int64 limbs: chunk-major, word-major, low limb first."""
    c = np.asarray(cvs, dtype=np.int64)
    lo, hi = c & 0xFFFF, c >> 16
    return np.stack([lo, hi], -1).reshape(c.shape[0], -1)


def elems_to_cvs(elems) -> np.ndarray:
    e = np.asarray(elems, dtype=np.int64).reshape(-1)
    if e.shape[0] % 16 or ((e < 0) | (e >= 1 << 16)).any():
        raise ValueError("a BLAKE3 leaf digest is 16 limbs < 2^16 per chunk")
    lh = e.reshape(-1, 8, 2)
    return (lh[..., 0] | (lh[..., 1] << 16)).astype(np.uint32)[None]


def leaf_bytes(digest_row, word_bits: int) -> bytes:
    """The leaf value of one row: the BLAKE3 root (32 bytes) folded natively from the published chunk CVs."""
    return root_from_cvs(elems_to_cvs(digest_row), word_bits)[0].tobytes()


# ---------------------------------------------------------------------------------------------------------------------
# The gadget
# ---------------------------------------------------------------------------------------------------------------------

def _const(ctx: LigeroCtx, c: int) -> Expr:
    return Expr(ctx, {}, c)


def _prod(ctx: LigeroCtx, a: Expr, b: Expr, scope: str) -> Expr:
    """A product row without the compiler's interval analysis (quadratic in the number of learned facts; the
    hash rows are booleans whose ranges the constraints themselves establish)."""
    z = ctx._row("prod", f"{scope}.prod{len(ctx._prod_memo)}", None, None, (a, b))
    idx = next(iter(z.lin))
    ctx._prod_memo[(a.key(), b.key(), idx)] = idx
    ctx.sys.program.append(("prod", idx, a, b))
    ctx.sys.quadratic.append(Quadratic(f"{scope}.prod{idx}", a, b, z))
    return z


def _bits_raw(ctx: LigeroCtx, e: Expr, nbits: int, name: str) -> list[Expr]:
    """``e = sum_i 2^i b_i`` with fresh boolean rows ``b_i`` (the honest ``e`` is an integer in ``[0, 2^nbits)``), as one
    linear constraint; no interval bookkeeping (the caller guarantees ``|e - sum| < p``, here < 2^19)."""
    rows = []
    for i in range(nbits):
        r = ctx._row("bit", f"{name}.b{i}", 0, 2, (e, i))
        rows.append(r)
        ctx.sys.quadratic.append(Quadratic(f"{name}.bool{i}", r, r, r))
    idxs = [next(iter(r.lin)) for r in rows]
    if idxs != list(range(idxs[0], idxs[0] + nbits)):
        raise RuntimeError("bit rows must be consecutive")
    ctx.sys.program.append(("bits", idxs[0], nbits, e))
    lin = dict(e.lin)
    for i, j in enumerate(idxs):
        lin[j] = lin.get(j, 0) - (1 << i)
    ctx.sys.linear.append(Linear(f"{name}.decomp", lin, e.c, None))
    return rows


class _W32:
    """A u32 in the circuit: 32 boolean affine expressions (``bits``), or -- for values that only ever enter additions,
    the message words -- just the two 16-bit limbs."""
    __slots__ = ("bits", "_limbs")

    def __init__(self, bits: list | None = None, limbs: tuple | None = None):
        self.bits = bits
        self._limbs = limbs

    @property
    def limbs(self) -> tuple[Expr, Expr]:
        if self._limbs is None:
            ctx = self.bits[0].ctx if isinstance(self.bits[0], Expr) else None
            lo = sum((b * (1 << i) for i, b in enumerate(self.bits[:16])), _const(ctx, 0))
            hi = sum((b * (1 << i) for i, b in enumerate(self.bits[16:])), _const(ctx, 0))
            self._limbs = (lo, hi)
        return self._limbs

    @staticmethod
    def const(ctx: LigeroCtx, v: int) -> "_W32":
        return _W32([_const(ctx, (v >> i) & 1) for i in range(32)])

    def rotr(self, n: int) -> "_W32":
        return _W32([self.bits[(i + n) % 32] for i in range(32)])


def _xor(ctx: LigeroCtx, x: _W32, y: _W32, scope: str) -> _W32:
    out = []
    for xi, yi in zip(x.bits, y.bits):
        if xi.is_const():
            out.append(yi if xi.c == 0 else 1 - yi)
        elif yi.is_const():
            out.append(xi if yi.c == 0 else 1 - xi)
        else:
            out.append(xi + yi - 2 * _prod(ctx, xi, yi, scope))
    return _W32(out)


def _add(ctx: LigeroCtx, terms: Sequence[_W32], name: str) -> _W32:
    """Sum mod 2^32 of 2 or 3 words on 16-bit limbs; the output as 32 fresh bit rows (+ carry rows)."""
    n = len(terms)
    cb = max(1, math.ceil(math.log2(n)))
    lo = sum((t.limbs[0] for t in terms), _const(ctx, 0))
    lo_bits = _bits_raw(ctx, lo, 16 + cb, name + ".lo")
    carry = sum((lo_bits[16 + i] * (1 << i) for i in range(cb)), _const(ctx, 0))
    hi = sum((t.limbs[1] for t in terms), carry)
    hi_bits = _bits_raw(ctx, hi, 16 + cb, name + ".hi")
    return _W32(lo_bits[:16] + hi_bits[:16])


def _g(ctx: LigeroCtx, v: list, a: int, b: int, c: int, d: int, mx: _W32, my: _W32, scope: str) -> None:
    v[a] = _add(ctx, (v[a], v[b], mx), f"{scope}.a1")
    v[d] = _xor(ctx, v[d], v[a], scope).rotr(16)
    v[c] = _add(ctx, (v[c], v[d]), f"{scope}.c1")
    v[b] = _xor(ctx, v[b], v[c], scope).rotr(12)
    v[a] = _add(ctx, (v[a], v[b], my), f"{scope}.a2")
    v[d] = _xor(ctx, v[d], v[a], scope).rotr(8)
    v[c] = _add(ctx, (v[c], v[d]), f"{scope}.c2")
    v[b] = _xor(ctx, v[b], v[c], scope).rotr(7)


def compress_gadget(ctx: LigeroCtx, cv: list[_W32], msg: list[_W32], counter: _W32, block_len: _W32, flags: _W32,
                    scope: str) -> list[_W32]:
    """The compression on circuit words: ``cv`` 8 words (0..3 may be limb-only, 4..7 need bits), ``msg`` 16 words (limbs
    suffice), ``counter`` / ``block_len`` / ``flags`` as words (constants or public-pin bits).  Returns the 16-word state."""
    v = list(cv) + [_W32.const(ctx, IV[i]) for i in range(4)] + [counter, _W32.const(ctx, 0), block_len, flags]
    m = list(msg)
    for r in range(ROUNDS):
        s = f"{scope}.r{r}"
        _g(ctx, v, 0, 4, 8, 12, m[0], m[1], s + ".g0")
        _g(ctx, v, 1, 5, 9, 13, m[2], m[3], s + ".g1")
        _g(ctx, v, 2, 6, 10, 14, m[4], m[5], s + ".g2")
        _g(ctx, v, 3, 7, 11, 15, m[6], m[7], s + ".g3")
        _g(ctx, v, 0, 5, 10, 15, m[8], m[9], s + ".g4")
        _g(ctx, v, 1, 6, 11, 12, m[10], m[11], s + ".g5")
        _g(ctx, v, 2, 7, 8, 13, m[12], m[13], s + ".g6")
        _g(ctx, v, 3, 4, 9, 14, m[14], m[15], s + ".g7")
        if r != ROUNDS - 1:
            m = [m[i] for i in MSG_PERMUTATION]
    return v


def _message_words(ctx: LigeroCtx, words: Sequence[Expr], word_bits: int) -> list[_W32]:
    """This column's operand words (little-endian bytes of the row) as 16 limb-only u32 message words."""
    if len(words) * word_bits != 8 * BLOCK_LEN:
        raise ValueError(f"a column must carry exactly one 64-byte block per operand: {len(words)} x {word_bits} bits")
    per = 16 // word_bits                      # words per limb: 2 bytes or 1 u16
    limbs = []
    for i in range(0, len(words), per):
        limbs.append(sum((words[i + q] * (1 << (word_bits * q)) for q in range(per)), _const(ctx, 0)))
    return [_W32(limbs=(limbs[2 * w], limbs[2 * w + 1])) for w in range(16)]


@dataclass
class Blake3Leaf:
    """The ``LeafScheme`` (brief §2) for BLAKE3.  ``word_bits`` / ``n_words`` fix the row format (digest size)."""

    word_bits: int
    n_words: int = K_VU
    name: str = NAME
    assumption: str = ASSUMPTION

    # --- framing -----------------------------------------------------------------------------------------------------
    @property
    def n_chunks(self) -> int:
        return n_chunks(self.word_bits, self.n_words)

    @property
    def digest_elems(self) -> int:
        return digest_elems_for(self.word_bits, self.n_words)

    @property
    def n_carry(self) -> int:
        """Carried rows per operand role: the running CV (16 limbs) + one held CV per finished chunk but the last."""
        return 16 * self.n_chunks

    @property
    def blocks_per_column(self) -> int:
        return 1

    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, digest_elems) int64: the chunk CVs of every row as 16-bit limbs (what the gadget pins at the chain end)."""
        if word_bits != self.word_bits:
            raise ValueError("word width differs from the scheme's")
        _, cvs = chunk_cvs_np(np.asarray(rows, dtype=np.int64), word_bits, role)
        return cvs_to_elems(cvs)

    def leaf_bytes(self, digest_row) -> bytes:
        return leaf_bytes(digest_row, self.word_bits)

    def reference_digest(self, rows: np.ndarray, *, word_bits: int, role: int) -> list[bytes]:
        """``blake3.blake3(row_bytes, key=key_for(word_bits)).digest()`` per row -- the independent reference the
        conformance test compares :meth:`leaf_bytes` of :meth:`native` against."""
        import blake3
        rb = row_bytes(rows, word_bits)
        key = key_for(word_bits)
        return [blake3.blake3(bytes(r), key=key).digest() for r in rb]

    # --- layout pins -------------------------------------------------------------------------------------------------
    def layout_pin_names(self) -> list[str]:
        names = ["blake3.cs", "blake3.ce"]
        names += [f"blake3.cnt{i}" for i in range(max(1, (self.n_chunks - 1).bit_length()))]
        names += [f"blake3.hold{c}" for c in range(self.n_chunks - 1)]
        return names

    def layout_pin_values(self, steps: int) -> dict[str, np.ndarray]:
        """The public per-column pins of one VU of ``steps`` columns (one block per column): a function of the layout."""
        fl = block_flags(self.word_bits, self.n_words)
        if len(fl) != steps * self.blocks_per_column:
            raise ValueError(f"{steps} columns do not carry the row's {len(fl)} blocks")
        vals = {n: np.zeros(steps, dtype=np.int64) for n in self.layout_pin_names()}
        ends = 0
        for j, (c, f) in enumerate(fl):
            vals["blake3.cs"][j] = 1 if f & CHUNK_START else 0
            vals["blake3.ce"][j] = 1 if f & CHUNK_END else 0
            for i in range(max(1, (self.n_chunks - 1).bit_length())):
                vals[f"blake3.cnt{i}"][j] = (c >> i) & 1
            if f & CHUNK_END and c < self.n_chunks - 1:
                vals[f"blake3.hold{c}"][j] = 1
                ends += 1
        return vals

    def layout_pins(self, ctx: LigeroCtx) -> dict[str, Expr]:
        """The pins in ``ctx`` (created once per compile; memoised on the context)."""
        memo = getattr(ctx, "_blake3_pins", None)
        if memo is None:
            scope = ctx.scope
            ctx.scope = ""                     # public names carry their own prefix
            memo = {n: ctx.pin(n, 0, 2) for n in self.layout_pin_names()}
            ctx.scope = scope
            ctx._blake3_pins = memo
        return memo

    # --- the in-circuit hash -----------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, carry_in: Sequence[Expr],
               first: bool = False, last: bool = False) -> tuple[list[Expr], list[Expr]]:
        """One column, one operand role: compress this column's block into the carried chaining value.  ``carry_in``:
        ``n_carry`` rows (the previous column's ``carry_out``; 0 at a chain start), laid out ``[cv (16 limbs) | hold_0 (16)
        | ...]``.  Returns ``(digest_exprs, carry_out)``: ``digest_exprs`` = ``[CV_0 | ... | CV_last]`` limbs (meaningful at
        the last column; the caller pins them under ``is_end``), ``carry_out`` = ``n_carry`` single-row expressions.
        ``first`` / ``last`` are ignored: the circuit is column-uniform, the layout enters through the public pins."""
        if word_bits != self.word_bits or len(carry_in) != self.n_carry:
            raise ValueError("gadget: word width / carry shape differ from the scheme's")
        pins = self.layout_pins(ctx)
        scope = f"blake3.{ROLE_NAMES[role]}"
        one = _const(ctx, 1)
        cs, ce = pins["blake3.cs"], pins["blake3.ce"]
        key = key_words(word_bits)
        # chaining value in: cs ? key : carried (limbs); words 4..7 need bits (they enter round 1's b ^= c)
        cv_in = []
        for w in range(8):
            lo_c, hi_c = key[w] & 0xFFFF, key[w] >> 16
            lo = carry_in[2 * w] + lo_c * cs - _prod(ctx, cs, carry_in[2 * w], scope + ".cvsel")
            hi = carry_in[2 * w + 1] + hi_c * cs - _prod(ctx, cs, carry_in[2 * w + 1], scope + ".cvsel")
            if w < 4:
                cv_in.append(_W32(limbs=(lo, hi)))
            else:
                bits = _bits_raw(ctx, lo, 16, f"{scope}.cv{w}.lo") + _bits_raw(ctx, hi, 16, f"{scope}.cv{w}.hi")
                cv_in.append(_W32(bits))
        ncnt = max(1, (self.n_chunks - 1).bit_length())
        counter = _W32([pins[f"blake3.cnt{i}"] for i in range(ncnt)] + [_const(ctx, 0)] * (32 - ncnt))
        flags = _W32([cs, ce] + [_const(ctx, 0)] * 2 + [one] + [_const(ctx, 0)] * 27)      # CHUNK_START | CHUNK_END | KEYED
        msg = _message_words(ctx, words, word_bits)
        v = compress_gadget(ctx, cv_in, msg, counter, _W32.const(ctx, BLOCK_LEN), flags, scope)
        # output chaining value h_i = v_i ^ v_{i+8}, copied into single rows (the links and the digest read one row each)
        h_out = []
        for w in range(8):
            x = _xor(ctx, v[w], v[w + 8], scope + ".fin")
            lo, hi = x.limbs
            h_out.append(_prod(ctx, lo, one, scope + ".hout"))
            h_out.append(_prod(ctx, hi, one, scope + ".hout"))
        carry_out = list(h_out)
        digest = []
        for c in range(self.n_chunks - 1):
            e_c = pins[f"blake3.hold{c}"]
            held = carry_in[16 * (c + 1):16 * (c + 2)]
            hold_out = [h + _prod(ctx, e_c, h_out[i] - h, scope + f".hold{c}") for i, h in enumerate(held)]
            carry_out += hold_out
            digest += hold_out
        digest += h_out
        return digest, carry_out

    def carry_values(self, rows: np.ndarray, *, word_bits: int, role: int, steps: int) -> np.ndarray:
        """The honest carried rows of every column: (R, steps, n_carry) int64 -- ``carry_in`` of column ``j`` = the
        gadget's ``carry_out`` of column ``j - 1`` (0 at ``j = 0``)."""
        out_blocks, cvs = chunk_cvs_np(np.asarray(rows, dtype=np.int64), word_bits, role)
        R, nb = out_blocks.shape[:2]
        if nb != steps:
            raise ValueError("one block per column")
        car = np.zeros((R, steps, self.n_carry), dtype=np.int64)
        prev = out_blocks[:, :-1, :].astype(np.int64)                    # (R, steps-1, 8)
        car[:, 1:, 0:16] = np.stack([prev & 0xFFFF, prev >> 16], -1).reshape(R, steps - 1, 16)
        fl = block_flags(word_bits, self.n_words)
        for c in range(self.n_chunks - 1):
            end = max(j for j, (cc, f) in enumerate(fl) if cc == c)
            cv = cvs[:, c, :].astype(np.int64)
            limbs = np.stack([cv & 0xFFFF, cv >> 16], -1).reshape(R, 16)
            car[:, end + 1:, 16 * (c + 1):16 * (c + 2)] = limbs[:, None, :]
        return car

    def witness(self):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------------------------------------------------

def _fresh_ctx() -> LigeroCtx:
    from ..relations import FP8_ADA_RELATION
    return LigeroCtx(FP8_ADA_RELATION.params)


def _count(sys) -> dict:
    kinds: dict[str, int] = {}
    for r in sys.rows:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    lin_terms = sum(len(c.lin) for c in sys.linear)
    quad_terms = sum(len(q.a.lin) + len(q.b.lin) + len(q.c.lin) for q in sys.quadratic)
    return {"rows": sys.m, "by_kind": kinds, "linear": len(sys.linear), "linear_terms": lin_terms,
            "quadratic": len(sys.quadratic), "quadratic_terms": quad_terms}


def census(word_bits: int = 8) -> dict:
    """Rows per 64-byte block for the bit design (measured by compiling one compression / one full column gadget), the
    one-hot lookup alternative for XOR (measured: one 4-bit XOR nibble through ``ctx.lookup``), and the per-VU totals."""
    from verity_numerical.checker.ctx import Table
    out: dict = {"word_bits": word_bits}
    # (a) one bare compression on private inputs (cv bits + message limbs as hint rows), no selection / carry logic
    ctx = _fresh_ctx()
    cv = [_W32([ctx.hint(f"cv{w}.b{i}") for i in range(32)]) for w in range(8)]
    msg = [_W32(limbs=(ctx.hint(f"m{w}.lo"), ctx.hint(f"m{w}.hi"))) for w in range(16)]
    m0 = ctx.sys.m
    v = compress_gadget(ctx, cv, msg, _W32.const(ctx, 0), _W32.const(ctx, BLOCK_LEN), _W32.const(ctx, KEYED_HASH), "c")
    m1 = ctx.sys.m
    for w in range(8):
        _xor(ctx, v[w], v[w + 8], "fin")
    out["compression_rows"] = m1 - m0
    out["finalize_xor_rows"] = ctx.sys.m - m1
    out["per_block_bit_design"] = ctx.sys.m - m0
    out["per_block_detail"] = _count(ctx.sys)
    # (b) one full column gadget for one role (selection, cv bit re-derivation, copies, holds, layout pins)
    leaf = Blake3Leaf(word_bits)
    ctx = _fresh_ctx()
    k = 8 * BLOCK_LEN // word_bits
    words = [ctx.hint(f"w{i}") for i in range(k)]
    carry = [ctx.hint(f"carry{i}") for i in range(leaf.n_carry)]
    m0 = ctx.sys.m
    leaf.gadget(ctx, words, word_bits=word_bits, role=1, carry_in=carry)
    out["per_column_role_rows"] = ctx.sys.m - m0                        # includes the layout pins (shared by both roles)
    out["layout_pins"] = len(leaf.layout_pin_names())
    out["carry_rows_per_role"] = leaf.n_carry
    out["column_detail"] = _count(ctx.sys)
    # (c) the one-hot lookup alternative: a 4-bit XOR table through ctx.lookup, rows per nibble
    ctx = _fresh_ctx()
    xor4 = Table("xor4", 0, 256, ("z", "k"), lambda key: ((key >> 4) ^ (key & 15), key))   # distinct rows (no interval groups)
    key = ctx.hint("key")
    ctx.rng("keyr", key, 0, 256)
    m0 = ctx.sys.m
    ctx.lookup("x4", xor4, key, range_key=False)
    out["lookup_xor4_rows_per_nibble"] = ctx.sys.m - m0 + 8         # + the 8 key bits (the key must be range-checked)
    out["lookup_xor_rows_per_u32"] = 8 * out["lookup_xor4_rows_per_nibble"]
    out["bit_xor_rows_per_u32"] = 32
    out["lookup_xor8_rows_per_byte"] = 65536 + 16
    # (d) per VU
    nblocks = K_VU * word_bits // 8 // BLOCK_LEN
    hash_rows_per_vu = 2 * nblocks * out["per_column_role_rows"]
    out["blocks_per_row"] = nblocks
    out["blocks_per_vu"] = 2 * nblocks
    out["hash_rows_per_vu_unshared"] = hash_rows_per_vu
    out["hash_rows_per_vu_tile64"] = hash_rows_per_vu / 64
    bare = {8: 3724 * 48, 16: 3292 * 96}[word_bits]
    out["bare_rows_per_vu"] = bare
    out["private_operand_rows_per_vu"] = {8: 704 * 48, 16: 608 * 96}[word_bits]
    out["pct_unshared"] = 100.0 * (hash_rows_per_vu + out["private_operand_rows_per_vu"]) / bare
    out["pct_tile64"] = 100.0 * (hash_rows_per_vu / 64 + out["private_operand_rows_per_vu"]) / bare
    out["pct_tile64_hash_only"] = 100.0 * (hash_rows_per_vu / 64) / bare
    return out


if __name__ == "__main__":
    import json
    import sys as _sys
    wb = int(_sys.argv[1]) if len(_sys.argv) > 1 else 8
    print(json.dumps(census(wb), indent=1))
