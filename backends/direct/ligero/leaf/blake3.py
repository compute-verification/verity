"""``leaf = blake3``: the operand rows hashed **in-circuit with BLAKE3's compression function** (lane blake3-leaf).

The conservative control of the leaf campaign: the same ``included-hash`` column as the Poseidon2 leaf (``hashchain.py``)
under a standard hash.  What the verifier ends up checking is the ordinary ``blake3.blake3(row_bytes, key=key_for(role)).digest()``
of every committed row (keyed mode, one 32-byte key per operand role; see :func:`key_for`) as the leaf value
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
rows, and holds the finished chunk CVs in ``16 (N_CHUNKS_MAX - 1)`` carried rows (16 products each) until the chain
end, where the compose layer pins ``[n_chunks | CV_0 | CV_1 | ...]`` against the public digest.  The layout (chunk start
/ end, chunk counter, hold flags) is derived from one more carried row, the column position ``pos`` (0 at the chain
start, +1 per column: sound by the chain constraints), through its bit decomposition -- no public layout pins.

Soundness of the arithmetic: every linear constraint here is a bit decomposition ``e = sum 2^i b_i`` of an affine ``e``
whose integer value (the addends are affine in booleans) lies in ``[0, 2^nbits)``; ``|e - sum| < 2^19 << p``, so the field
identity is the integer identity and the bits are the sum's bits.  XOR inputs are booleans by construction (bit rows,
public {0, 1} pins, constants, or XOR outputs of booleans with the exact product enforced by its quadratic constraint).
Unsalted (as the Poseidon2 leaf): the digest of a low-entropy row is preimage-searchable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
import numpy as np

from ..compile import Expr, LigeroCtx, Linear, P, Quadratic

__all__ = ["BLOCK_LEN", "CARRY_ELEMS", "CHUNK_LEN", "DIGEST_ELEMS", "IV", "KEYS", "KEY_LEN", "K_VU", "N_CHUNKS_MAX", "Blake3Leaf",
           "census", "chunk_cvs_np", "compress_np", "digest_elems_for", "key_for", "leaf_bytes", "leaf_bytes_many", "n_chunks", "root_from_cvs",
           "row_bytes"]

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

KEY_PREFIX = b"verity/blake3-leaf/v1/"
KEYS = {role: (KEY_PREFIX + nm.split("-")[0].encode()).ljust(KEY_LEN, b"\0") for role, nm in ROLE_NAMES.items()}
N_CHUNKS_MAX = 3                        # E4M3 rows: 2 chunks, BF16 rows: 3 (one registry instance serves both widths)
DIGEST_ELEMS = 1 + 16 * N_CHUNKS_MAX    # [header | CV_0 limbs | CV_1 | CV_2]  (unused slots 0); header = n_chunks + 4 role
HOLD_MSG = 16 * N_CHUNKS_MAX            # carry offset of the half-block message hold (32-byte columns: x1 relations)
CARRY_ELEMS = HOLD_MSG + 16 + 1         # [running cv (16) | hold_0 (16) | hold_1 (16) | msg hold (16) | pos]
#: A row of more than N_CHUNKS_MAX chunks (a K = 2048 BF16 row is 4, K = 8192 BF16 16, K = 8192 E4M3 8) is framed by a SIZED
#: instance (``Blake3Leaf.for_row``): ``n_chunks_max`` = the row's chunk count, the same layout with more CV slots, and the role
#: moved above a 6-bit chunk count in the header (``n_chunks + 64 role``) -- the 3-chunk layout and its pins are unchanged.
MAX_CHUNKS_SIZED = 63


def digest_elems_of(ncm: int) -> int:
    return 1 + 16 * ncm


def hold_msg_of(ncm: int) -> int:
    return 16 * ncm


def carry_elems_of(ncm: int) -> int:
    return 16 * ncm + 16 + 1


def _role_shift(ncm: int) -> int:
    return 2 if ncm <= N_CHUNKS_MAX else 6


def _check_ncm(ncm: int) -> int:
    if not N_CHUNKS_MAX <= int(ncm) <= MAX_CHUNKS_SIZED:
        raise ValueError(f"a BLAKE3 leaf frames {N_CHUNKS_MAX}..{MAX_CHUNKS_SIZED} chunk slots, not {ncm}")
    return int(ncm)


def ncm_of_len(n_elems: int) -> int:
    """The chunk slots of a digest row of ``n_elems`` elements (``1 + 16 ncm``); ``ValueError`` for any other length."""
    if n_elems < DIGEST_ELEMS or (n_elems - 1) % 16:
        raise ValueError(f"a BLAKE3 leaf digest is 1 + 16 x (chunk slots) elements, not {n_elems}")
    return _check_ncm((n_elems - 1) // 16)
SCHEMA = "blake3-keyed/row/v2"
HASH_NAME = "blake3-keyed"
MALFORMED_TAG = b"verity/blake3-leaf/malformed-digest/v1\0"


def key_for(role: int) -> bytes:
    """The 32-byte BLAKE3 key of one operand role: ``"verity/blake3-leaf/v1/x"`` (x rows) / ``".../w"`` (W columns), zero
    padded.  Domain separation at zero circuit cost (the key words replace IV as the first chaining value of every chunk
    AND of every parent node; ``KEYED_HASH`` is a constant flag bit).  The role is in the key (red-team-leaf NIT: an x row
    and a W column with the same bytes must not share a digest) and, so that :func:`leaf_bytes` can fold with the right
    key, in the digest header; the word width is not (1536- vs 3072-byte messages differ anyway)."""
    return KEYS[role]


def key_words(role: int) -> tuple[int, ...]:
    k = key_for(role)
    return tuple(int.from_bytes(k[4 * i:4 * i + 4], "little") for i in range(8))


def header(nch: int, role: int, ncm: int = N_CHUNKS_MAX) -> int:
    """Digest element 0: ``n_chunks + 4 * role`` (6 / 10 for E4M3 rows, 7 / 11 for BF16); a sized frame (``ncm`` > 3 chunk
    slots) ``n_chunks + 64 * role``."""
    return nch + (role << _role_shift(ncm))


def parse_header(h: int, ncm: int = N_CHUNKS_MAX) -> tuple[int, int]:
    sh = _role_shift(ncm)
    nch, role = h & ((1 << sh) - 1), h >> sh
    if role not in ROLE_NAMES or not 2 <= nch <= ncm:
        raise ValueError(f"a BLAKE3 leaf digest header is n_chunks (2..{ncm}) + 2^{sh} x role (1, 2), not {h}")
    return nch, role


def params_sha256(ncm: int = N_CHUNKS_MAX) -> str:
    """The 32-byte parameters field of the v5 hash-auth block: the keys and the framing constants (the chunk slots
    included, so a sized frame is another parameter set)."""
    import hashlib
    return hashlib.sha256(b"blake3-leaf|keys=" + KEYS[1] + KEYS[2] + b"|framing=header+chunk-cvs-le16-limbs|max_chunks=" +
                          str(_check_ncm(ncm)).encode() + b"|malformed=" + MALFORMED_TAG + b"|schema=" + SCHEMA.encode()).hexdigest()


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
    """Meaningful digest elements of a row format: the count + 16 limbs per chunk (33 for E4M3 rows, 49 for BF16); the
    scheme publishes :data:`DIGEST_ELEMS` with the unused slots 0."""
    return 1 + 16 * n_chunks(word_bits, n_words)


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
    key = np.asarray(key_words(role), dtype=np.uint32)[:, None]
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


def root_from_cvs(cvs: np.ndarray, role: int) -> np.ndarray:
    """(R, n_chunks, 8) chunk CVs -> (R, 32) uint8: the standard BLAKE3 tree root (left-heavy parent fold with the role's
    key as the parents' chaining value, ``ROOT`` on the top node; a single chunk would be the root itself with ``ROOT``
    set on its last block -- not this row size)."""
    cvs = np.asarray(cvs, dtype=np.uint32)
    R, nch, _ = cvs.shape
    if nch < 2:
        raise ValueError("rows of one chunk are not framed here (the chunk's last block would carry ROOT)")
    nodes = [cvs[:, c, :].T for c in range(nch)]              # (8, R) each
    iv = np.asarray(key_words(role), dtype=np.uint32)[:, None].repeat(R, 1)

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


def cvs_to_elems(cvs: np.ndarray, role: int, ncm: int = N_CHUNKS_MAX) -> np.ndarray:
    """(R, n_chunks, 8) u32 -> (R, 1 + 16 ncm) int64: ``[header | limbs chunk-major, word-major, low limb first | 0..]``."""
    c = np.asarray(cvs, dtype=np.int64)
    R, nch, _ = c.shape
    if nch > ncm:
        raise ValueError(f"{nch} chunks exceed the frame's {ncm}")
    lo, hi = c & 0xFFFF, c >> 16
    out = np.zeros((R, digest_elems_of(ncm)), dtype=np.int64)
    out[:, 0] = header(nch, role, ncm)
    out[:, 1:1 + 16 * nch] = np.stack([lo, hi], -1).reshape(R, -1)
    return out


def elems_to_cvs(elems, ncm: int = N_CHUNKS_MAX) -> tuple[np.ndarray, int]:
    """One digest row -> ((1, n_chunks, 8) uint32 chunk CVs, role); ``ValueError`` on a malformed frame."""
    e = np.asarray(elems, dtype=np.int64).reshape(-1)
    if e.shape[0] != digest_elems_of(ncm):
        raise ValueError(f"a BLAKE3 leaf digest is {digest_elems_of(ncm)} elements")
    nch, role = parse_header(int(e[0]), ncm)
    body, tail = e[1:1 + 16 * nch], e[1 + 16 * nch:]
    if ((body < 0) | (body >= 1 << 16)).any() or tail.any():
        raise ValueError("a BLAKE3 leaf digest is 16 limbs < 2^16 per chunk, unused slots 0")
    lh = body.reshape(-1, 8, 2)
    return (lh[..., 0] | (lh[..., 1] << 16)).astype(np.uint32)[None], role


def leaf_bytes(digest_row, ncm: int = N_CHUNKS_MAX) -> bytes:
    """The leaf value of one row: the 32-byte BLAKE3 root folded natively (with the header's role key) from the published
    chunk CVs.  Total on field-element rows of the right length, as the contract's conformance suite requires: a row that
    is not a well-formed frame (the gadget never pins one -- its digest limbs are bit decompositions and its header a
    constant) maps to the tagged SHA-256 of its elements, a distinct 32-byte domain that no BLAKE3 root of a row lands in
    except by collision.  Elements outside the field or a wrong length raise ``ValueError``."""
    e = np.asarray(digest_row, dtype=np.int64).reshape(-1)
    if e.shape[0] != digest_elems_of(ncm):
        raise ValueError(f"a BLAKE3 leaf digest is {digest_elems_of(ncm)} elements")
    if ((e < 0) | (e >= P)).any():
        raise ValueError("digest elements are BabyBear field elements")
    try:
        cvs, role = elems_to_cvs(e, ncm)
    except ValueError:
        import hashlib
        return hashlib.sha256(MALFORMED_TAG + e.astype("<u4").tobytes()).digest()
    return root_from_cvs(cvs, role)[0].tobytes()


def leaf_bytes_many(digests, ncm: int = N_CHUNKS_MAX) -> list[bytes]:
    """:func:`leaf_bytes` of every row of ``digests`` (R, 1 + 16 ncm), the parent folds in lockstep when every row is a
    well-formed frame under one header (what the committer's :meth:`Blake3Leaf.native` returns); otherwise row by row."""
    e = np.asarray(digests, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != digest_elems_of(ncm):
        raise ValueError(f"a BLAKE3 leaf digest is {digest_elems_of(ncm)} elements")
    if e.shape[0] == 0:
        return []
    try:
        if (e[:, 0] != e[0, 0]).any():
            raise ValueError("mixed headers")
        nch, role = parse_header(int(e[0, 0]), ncm)
    except ValueError:
        return [leaf_bytes(r, ncm) for r in e]
    body, tail = e[:, 1:1 + 16 * nch], e[:, 1 + 16 * nch:]
    if ((body < 0) | (body >= 1 << 16)).any() or tail.any():
        return [leaf_bytes(r, ncm) for r in e]
    lh = body.reshape(e.shape[0], nch, 8, 2)
    roots = root_from_cvs((lh[..., 0] | (lh[..., 1] << 16)).astype(np.uint32), role)
    return [bytes(r) for r in roots]


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


def _limbs_of(ctx: LigeroCtx, words: Sequence[Expr], word_bits: int) -> list[Expr]:
    """Operand words (little-endian bytes of the row) as 16-bit limbs (2 bytes or 1 u16 each), low first."""
    per = 16 // word_bits                      # words per limb: 2 bytes or 1 u16
    return [sum((words[i + q] * (1 << (word_bits * q)) for q in range(per)), _const(ctx, 0))
            for i in range(0, len(words), per)]


def _message_words(limbs: Sequence[Expr]) -> list[_W32]:
    """32 limbs (one 64-byte block) -> 16 limb-only u32 message words."""
    if len(limbs) != 32:
        raise ValueError(f"a block is 32 limbs, not {len(limbs)}")
    return [_W32(limbs=(limbs[2 * w], limbs[2 * w + 1])) for w in range(16)]


def _indicator(ctx: LigeroCtx, bits: Sequence[Expr], value: int, memo: dict, scope: str) -> Expr:
    """``[bits == value]`` as a product of the bit literals, sharing prefixes through ``memo`` (keyed by the literal
    prefix); ``bits`` are booleans, so the result is boolean."""
    key = ()
    acc = None
    for i, b in enumerate(bits):
        lit = b if (value >> i) & 1 else 1 - b
        key = key + (((value >> i) & 1),)
        if acc is None:
            acc = lit
        else:
            hit = memo.get(key)
            if hit is None:
                hit = _prod(ctx, acc, lit, scope + ".ind")
                memo[key] = hit
            acc = hit
    return acc


class Blake3Leaf:
    """The ``LeafScheme`` (``leaf/base.py``) for BLAKE3.  One instance serves both row formats: the digest and carry
    layouts are fixed at ``n_chunks_max`` chunk slots (:data:`N_CHUNKS_MAX` for the registry's instance), the width enters
    through ``word_bits`` / ``row_words`` per call.  A longer row is framed by :meth:`for_row`'s sized instance.

    Carried state per operand (``CARRY_ELEMS`` rows, 0 at a chain start): ``[cv (16 limbs) | hold_0 (16) | hold_1 (16) |
    msg hold (16) | pos]``.  ``pos`` is the column's position in the VU (the chain forces ``pos_in = 0`` at the start and
    ``pos_out = pos_in + 1`` links the columns, so it is a sound private counter); its bits give the layout the Poseidon2
    leaf did not need: chunk start ``cs = [block mod 16 == 0]``, chunk end ``ce``, the chunk counter ``block >> 4`` and the
    ``hold_c`` flags ``[block == end of chunk c]`` -- no public layout pins, nothing for the verifier to compute per column.

    Columns carry 64 or 128 bytes per operand (x2 / x4 folds: 1 or 2 compressions per column) or 32 bytes (the unfolded
    relations: HALF a block).  Half-block columns pair up: the even column parks its 16 limbs in the msg hold, the odd
    column compresses ``[hold | its own limbs]`` -- but the circuit is column-uniform, so the even column runs a
    compression too and discards it (``cv_out = odd ? new : cv_in``): the x1 relations pay 2x the rows of the folds.

    Digest (``DIGEST_ELEMS`` public pins per operand at the chain end): ``[n_chunks + 4 role | CV_0 | ... | 0..]``; the
    verifier's :meth:`leaf_bytes` reads the header, folds the CVs with the role's key to the BLAKE3 root and that root is
    the SHA-256 tree leaf."""

    name = NAME
    assumption = ASSUMPTION
    digest_elems = DIGEST_ELEMS
    carry_elems = CARRY_ELEMS
    schema = SCHEMA
    hash_name = HASH_NAME
    params_sha256 = params_sha256()
    test_only = False
    privacy_note = "deterministic, unsalted: hides only up to preimage search on a row"   # red-team-leaf F5
    n_chunks_max = N_CHUNKS_MAX

    def __init__(self, n_chunks_max: int = N_CHUNKS_MAX):
        if n_chunks_max != N_CHUNKS_MAX:
            self.n_chunks_max = _check_ncm(n_chunks_max)
            self.digest_elems = digest_elems_of(n_chunks_max)
            self.carry_elems = carry_elems_of(n_chunks_max)
            self.params_sha256 = params_sha256(n_chunks_max)

    @property
    def _hold_msg(self) -> int:
        return hold_msg_of(self.n_chunks_max)

    def for_row(self, row_words: int, word_bits: int) -> "Blake3Leaf":
        """This scheme framed for rows of ``row_words`` ``word_bits``-bit words: itself when the row fits its chunk slots,
        else a fresh instance of the same class with ``n_chunks_max`` = the row's chunk count."""
        nch = n_chunks(word_bits, row_words)
        return self if nch <= self.n_chunks_max else type(self)(nch)

    # --- committer / verifier side -----------------------------------------------------------------------------------
    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, DIGEST_ELEMS) int64: the header and the chunk CVs of every row as 16-bit limbs (what the gadget pins at
        the chain end), keyed by ``role``."""
        rows = np.asarray(rows, dtype=np.int64)
        if n_chunks(word_bits, rows.shape[1]) < 2:
            raise ValueError("rows of one chunk are not framed by this scheme (the chunk's last block would carry ROOT)")
        _, cvs = chunk_cvs_np(rows, word_bits, role)
        return cvs_to_elems(cvs, role, self.n_chunks_max)

    def leaf_bytes(self, digest_row) -> bytes:
        return leaf_bytes(digest_row, self.n_chunks_max)

    def leaf_bytes_many(self, digests) -> list[bytes]:
        return leaf_bytes_many(digests, self.n_chunks_max)

    def reference_digest(self, rows: np.ndarray, *, word_bits: int, role: int) -> list[bytes]:
        """``blake3.blake3(row_bytes, key=key_for(role)).digest()`` per row -- the independent reference the conformance
        test compares :meth:`leaf_bytes` of :meth:`native` against."""
        import blake3
        return [blake3.blake3(bytes(r), key=key_for(role)).digest() for r in row_bytes(rows, word_bits)]

    # --- the in-circuit hash -----------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int = K_VU,
               carry_in: Sequence[Expr], first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        """One column, one operand role: compress this column's block (``words``: k affine words = 64 bytes) into the
        carried chaining value.  Returns ``(digest_exprs, carry_out)`` (see the class doc); ``first`` / ``last`` are
        ignored (column-uniform circuit)."""
        ncm, ce, hm = self.n_chunks_max, self.carry_elems, self._hold_msg
        if len(carry_in) != ce:
            raise ValueError(f"gadget: {len(carry_in)} carried rows, the scheme carries {ce}")
        k = len(words)
        col_bytes = k * word_bits // 8
        half = col_bytes * 2 == BLOCK_LEN                     # 32-byte columns: two columns per block
        if row_words % k or (not half and (col_bytes % BLOCK_LEN or (col_bytes // BLOCK_LEN) & (col_bytes // BLOCK_LEN - 1))):
            raise ValueError(f"gadget: a column must carry half or a power of two of 64-byte blocks per operand, not {col_bytes} bytes")
        nblk = 1 if half else col_bytes // BLOCK_LEN          # compressions per column: 1 (x1 / x2), 2 (x4 fold)
        cpb = 2 if half else 1                                # columns per block
        if nblk > 16:
            raise ValueError("gadget: at most one chunk per column")
        steps = row_words // k
        nch = n_chunks(word_bits, row_words)
        if not 2 <= nch <= ncm:
            raise ValueError(f"gadget: rows of {nch} chunks are outside 2..{ncm} (Blake3Leaf.for_row sizes the frame)")
        fl = block_flags(word_bits, row_words)
        if len(fl) * cpb != steps * nblk:
            raise ValueError("gadget: the row's blocks do not tile the columns")
        scope = f"blake3.{ROLE_NAMES[role]}"
        one, zero = _const(ctx, 1), _const(ctx, 0)
        key = key_words(role)
        # --- layout from the carried column position: block j sits at column j // nblk, sub-position j % nblk (half:
        # block j completes at the odd column 2 j + 1) ----------------------------------------------------------------
        col_of = (lambda j: 2 * j + 1) if half else (lambda j: j // nblk)
        sub_of = (lambda j: 0) if half else (lambda j: j % nblk)
        pos = carry_in[ce - 1]
        npb = max(6 if half else 5, (steps - 1).bit_length())  # 24 columns: 5 bits; 48: 6; 96: 7
        pbits = _bits_raw(ctx, pos, npb, f"{scope}.pos")
        memo: dict = {}
        ind = lambda value: _indicator(ctx, pbits, value, memo, scope)  # noqa: E731
        shift = 5 if half else 4 - (nblk.bit_length() - 1)   # chunk index = j >> 4 = pos >> shift
        counter = _W32(list(pbits[shift:]) + [zero] * (32 - (npb - shift)))
        h_in = [_W32(limbs=(carry_in[2 * w], carry_in[2 * w + 1])) for w in range(8)]   # the carried CV (limbs)
        h_out: list[Expr] = []
        digest = [_const(ctx, header(nch, role, ncm))]
        hold_out = [list(carry_in[16 * (c + 1):16 * (c + 2)]) for c in range(ncm - 1)]
        limbs = _limbs_of(ctx, words, word_bits)              # this column's operand bytes as 16-bit limbs
        msg_hold_in = list(carry_in[hm:hm + 16])
        for t in range(nblk):
            blocks_t = [j for j in range(len(fl)) if sub_of(j) == t]   # the blocks this sub-column position processes
            cs = sum((ind(col_of(j)) for j in blocks_t if fl[j][1] & CHUNK_START), zero)   # disjoint: boolean
            ce = sum((ind(col_of(j)) for j in blocks_t if fl[j][1] & CHUNK_END), zero)
            sc = f"{scope}.blk{t}"
            # chaining value in: cs ? KEY : previous (limbs); words 4..7 need bits (round 1's b ^= c)
            cv_in = []
            for w in range(8):
                lo_c, hi_c = key[w] & 0xFFFF, key[w] >> 16
                lo_p, hi_p = h_in[w].limbs
                lo = lo_p + lo_c * cs - _prod(ctx, cs, lo_p, sc + ".cvsel")
                hi = hi_p + hi_c * cs - _prod(ctx, cs, hi_p, sc + ".cvsel")
                if w < 4:
                    cv_in.append(_W32(limbs=(lo, hi)))
                else:
                    cv_in.append(_W32(_bits_raw(ctx, lo, 16, f"{sc}.cv{w}.lo") + _bits_raw(ctx, hi, 16, f"{sc}.cv{w}.hi")))
            flags = _W32([cs, ce, zero, zero, one] + [zero] * 27)      # CHUNK_START | CHUNK_END | KEYED_HASH
            msg = _message_words((msg_hold_in + limbs) if half else limbs[32 * t:32 * (t + 1)])
            fin = self._compress(ctx, cv_in, msg, counter, _W32.const(ctx, BLOCK_LEN), flags, sc)
            # output chaining value h_i = v_i ^ v_{i+8}, copied into single rows (the links / digest read one row each);
            # a half-block column keeps the carried CV unless it is the odd (completing) column
            h_out = []
            for w in range(8):
                for i, e in enumerate(fin(w).limbs):
                    if half:
                        prev = h_in[w].limbs[i]
                        h_out.append(prev + _prod(ctx, pbits[0], e - prev, sc + ".hout"))
                    else:
                        h_out.append(_prod(ctx, e, one, sc + ".hout"))
            h_in = [_W32(limbs=(h_out[2 * w], h_out[2 * w + 1])) for w in range(8)]
            # a chunk (but the last) ending at this block parks its CV in its hold slot
            for c in range(nch - 1):
                end_c = max(j for j, (cc, _) in enumerate(fl) if cc == c)
                if sub_of(end_c) == t:
                    e_c = ind(col_of(end_c))
                    hold_out[c] = [h + _prod(ctx, e_c, h_out[i] - h, sc + f".hold{c}") for i, h in enumerate(hold_out[c])]
        carry_out = list(h_out)
        slots = hold_out                                     # unused slots: carried unchanged (0 from the chain start)
        for c in range(ncm - 1):
            carry_out += hold_out[c]
        carry_out += limbs if half else msg_hold_in          # the msg hold: this column's limbs (half) / unchanged
        carry_out.append(pos + 1)
        for c in range(ncm):
            if c < nch - 1:
                digest += slots[c]
            elif c == nch - 1:
                digest += h_out
            else:
                digest += [zero] * 16
        return digest, carry_out

    def _compress(self, ctx: LigeroCtx, cv: list[_W32], msg: list[_W32], counter: _W32, block_len: _W32, flags: _W32,
                  scope: str):
        """``w -> h_w``, the compression's output chaining value word ``v_w ^ v_{w+8}``.  Each word's XOR rows are
        emitted when it is asked for: the pinned systems interleave them with the gadget's copy rows."""
        v = compress_gadget(ctx, cv, msg, counter, block_len, flags, scope)
        return lambda w: _xor(ctx, v[w], v[w + 8], scope + ".fin")

    def carry_values(self, rows: np.ndarray, *, word_bits: int, role: int, steps: int) -> np.ndarray:
        """The honest carried rows of every column of every row: (R, steps, CARRY_ELEMS) int64 -- ``carry_in`` of column
        ``j`` = the gadget's ``carry_out`` of column ``j - 1`` (0 at ``j = 0``)."""
        rows = np.asarray(rows, dtype=np.int64)
        row_words = rows.shape[1]
        out_blocks, cvs = chunk_cvs_np(rows, word_bits, role)
        R, nb = out_blocks.shape[:2]
        half = steps == 2 * nb
        if not half and nb % steps:
            raise ValueError("whole (or half) blocks per column")
        nblk = 1 if half else nb // steps
        ce, hm = self.carry_elems, self._hold_msg
        car = np.zeros((R, steps, ce), dtype=np.int64)
        fl = block_flags(word_bits, row_words)
        col_of = (lambda j: 2 * j + 1) if half else (lambda j: j // nblk)   # the column at which block j completes
        for j in range(nblk - 1, nb, nblk):                   # each column's last block
            cv = out_blocks[:, j, :].astype(np.int64)        # the CV after it: carried in from the next column on
            car[:, col_of(j) + 1:, 0:16] = np.stack([cv & 0xFFFF, cv >> 16], -1).reshape(R, 16)[:, None, :]
        for c in range(n_chunks(word_bits, row_words) - 1):
            end = col_of(max(j for j, (cc, f) in enumerate(fl) if cc == c))      # the column that ends chunk c
            cv = cvs[:, c, :].astype(np.int64)
            limbs = np.stack([cv & 0xFFFF, cv >> 16], -1).reshape(R, 16)
            car[:, end + 1:, 16 * (c + 1):16 * (c + 2)] = limbs[:, None, :]
        if half:                                              # the msg hold: the previous column's 16 limbs
            rb = row_bytes(rows, word_bits).reshape(R, steps, 32)
            car[:, 1:, hm:hm + 16] = np.ascontiguousarray(rb[:, :-1, :]).view("<u2").astype(np.int64)
        car[:, :, ce - 1] = np.arange(steps)[None, :]
        return car

    def row_sponges(self, words, *, word_bits: int, role: int, row_words: int, n: int, steps: int):
        """The committer's chains of ``n`` whole rows at once on the GPU (``frame_gpu.blake3_chain``): ``words`` (n * steps,
        k) int64 CUDA torch, row-major -> ``(carry (n, steps, CARRY_ELEMS), digests (n, DIGEST_ELEMS))`` =
        :meth:`carry_values` / :meth:`native`; ``None`` off CUDA or without the GPU committer (the caller falls back to
        :meth:`chain_witness`)."""
        from .. import frame_gpu
        if not words.is_cuda or not frame_gpu.available() or n_chunks(word_bits, row_words) < 2:
            return None
        return frame_gpu.blake3_chain(words, word_bits=word_bits, role=role, row_words=row_words, n=n, steps=steps,
                                      ncm=self.n_chunks_max)

    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """``words`` (l, k) int64 torch (VU v in columns [v steps, (v + 1) steps), pad columns 0) -> (carry (CARRY_ELEMS,
        l) torch: the carried rows entering every column, digests (n_vus, DIGEST_ELEMS) torch)."""
        import torch
        w = words.detach().cpu().numpy().astype(np.int64)
        k = w.shape[1]
        if steps * k != row_words:
            raise ValueError("chain_witness: steps x k != row_words")
        rows = w[:n_vus * steps].reshape(n_vus, row_words)
        car = self.carry_values(rows, word_bits=word_bits, role=role, steps=steps)     # (n_vus, steps, ce)
        carry = np.zeros((self.carry_elems, l), dtype=np.int64)
        carry[:, :n_vus * steps] = car.reshape(n_vus * steps, self.carry_elems).T
        dg = self.native(rows, word_bits=word_bits, role=role)
        dev = words.device
        return torch.as_tensor(carry, device=dev), torch.as_tensor(dg, device=dev)

    def witness(self):
        return None

    def census(self) -> dict:
        return {"blake3_compression_rows_per_block": 14880, "blake3_finalize_xor_rows": 256,
                "blake3_keys": {ROLE_NAMES[r]: k.rstrip(b"\0").decode() for r, k in KEYS.items()}, "blake3_framing": "chunk CVs in-circuit, parent fold by the verifier"}


try:                                                          # the leaf registry (lane leaf-iface); absent on plain main
    from . import registry as _registry
    _registry.register(Blake3Leaf)
except ImportError:                                           # pragma: no cover
    pass


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
    # (b) one full column gadget for one role (position bits + layout indicators, cv selection, cv bit re-derivation,
    #     copies, holds)
    leaf = Blake3Leaf()
    ctx = _fresh_ctx()
    k = 8 * BLOCK_LEN // word_bits
    words = [ctx.hint(f"w{i}") for i in range(k)]
    carry = [ctx.hint(f"carry{i}") for i in range(CARRY_ELEMS)]
    m0 = ctx.sys.m
    leaf.gadget(ctx, words, word_bits=word_bits, role=1, row_words=K_VU, carry_in=carry)
    out["per_column_role_rows"] = ctx.sys.m - m0
    out["layout_pins"] = 0
    out["carry_rows_per_role"] = CARRY_ELEMS
    out["digest_pins_per_role"] = DIGEST_ELEMS
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
