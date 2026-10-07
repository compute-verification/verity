"""Row leaves of frame-v3: one tensor row per leaf, committed as a row digest under the SHA-256 indexed trees of :mod:`merkle`.

A leaf is one *row* of a tensor in the orientation the GEMM consumes it -- a row of the activation matrix ``x`` (K
words), a column of the weight matrix ``W`` (K words).  The row's digest is the leaf *value* under
:meth:`CommitmentDomain.leaf` with the schema's string; ranks are row indices and the tree above is the ordinary
SHA-256 ``node`` / pad framing, so a verifier that has the digest checks an opening with SHA-256 alone.  A backend
proves only the row digest (its ``leaf_layout``) and publishes it.

Five schemas:

``sha256/row/v1``            ``SHA-256(sha256_row_prefix(role, word_bits, n_words) || row_bytes)``: a 64-byte constant
                             first block (tag, role, word width, row length), so the gadget starts from a fixed
                             midstate.
``blake3-keyed/row/v2``      ``BLAKE3(row_bytes, key=blake3_row_key(role))``: B-Ligero's ``leaf = blake3`` framing
                             (``archive/direct/ligero/leaf/blake3.py``), promoted byte for byte; its schema string and
                             keys are pinned by ``ligero-verify``.
``sha256/row-nvfp4/v1``      ``SHA-256(sha256_row_nvfp4_prefix(role, k) || nvfp4_row_bytes)`` for an NVFP4 row (``k`` E2M1
                             codes and their ``k/16`` UE4M3 scales).
``blake3-keyed/row-nvfp4/v1`` ``BLAKE3(nvfp4_row_bytes, key=blake3_row_key(role))``: the same row bytes, the frame-v3 keys.
``sha512/row/v1``            ``SHA-512(sha512_row_prefix(role, word_bits, n_words) || row_bytes)``: ``sha256/row/v1`` through SHA-512,
                             with a 128-byte constant first block (``frame-v3-sha512`` rows and ``hm96-sha512/v1``'s inner digest).
``sha512/row/v2``            ``SHA-512(sha512_row_v2_prefix(n_bits) || row)``: a row is a bit string (:func:`bit_row`), the
                             128-byte constant first block only its tag and length in bits; no format, role or word width.
``sha512/row-seg/v1``        a bit row (as ``sha512/row/v2``) cut into 1 KiB segments, each hashed on its own and the digests
                             hashed once more: ``SHA-512(Q(n_bits) || d_0 || ... || d_(s-1))``, ``d_j = SHA-512(S(j, s, m_j) ||
                             segment j)``, every prefix one constant block, so the segments hash in parallel (PoUW's A rows).
``hm96-sha512/row/v1``       the schema a ``frame-v3-sha512`` leaf binds over a hiding row's leaf digest:
                             ``hm96.SHA512.tree_leaf(key, b || c)`` over ``sha512/row/v1`` (64 bytes); ``hm96-sha512/row/v2`` over
                             ``sha512/row/v2``, ``hm96-sha512/row-seg/v1`` over ``sha512/row-seg/v1``.
``poseidon2-babybear-w24/row/v2h``  the Poseidon2 row digest. Admissible in a Table 2 cell (``frame-v3/poseidon2-babybear``),
                             but algebraic: its results are reported and flagged as algebraic in the security section,
                             and never count toward a standard-hash claim (Daniel's standing decision).

``row_bytes`` are the row's words little-endian, ``word_bits`` 8, 16 or 32 bits each, in row order.  ``nvfp4_row_bytes``
are the codes packed two per byte (code ``2i`` in the low nibble), then the scale bytes: the code buffer and the scale
buffer as NVFP4 tensors hold them, so a serving-side committer hashes rows without repacking (``bit_row`` gives the same
bytes).  Roles are :data:`ROLE_X` (1, an x row) and :data:`ROLE_W` (2, a W column).  Unsalted: a low-entropy row is
preimage-searchable.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.ml.crypto import blake3 as _blake3
from . import poseidon2_babybear as h2
from .merkle import CommitmentDomain
from .scheme import LeafLayout

__all__ = [
    "ROLE_W",
    "ROLE_X",
    "SCHEMA_BLAKE3_ROW",
    "SCHEMA_BLAKE3_ROW_NVFP4",
    "SCHEMA_HM96_SHA512_ROW",
    "SCHEMA_HM96_SHA512_ROW_SEG",
    "SCHEMA_HM96_SHA512_ROW_V2",
    "SCHEMA_ROW",
    "SCHEMA_SHA256_ROW",
    "SCHEMA_SHA256_ROW_NVFP4",
    "SCHEMA_SHA512_ROW",
    "SCHEMA_SHA512_ROW_SEG",
    "SCHEMA_SHA512_ROW_V2",
    "bit_row",
    "blake3_row_digest",
    "blake3_row_key",
    "blake3_row_layout",
    "blake3_row_nvfp4_digest",
    "blake3_row_nvfp4_layout",
    "nvfp4_row_bytes",
    "row_bytes",
    "row_digest",
    "row_leaf",
    "row_leaf_value",
    "sha256_row_digest",
    "sha256_row_layout",
    "sha256_row_nvfp4_digest",
    "sha256_row_nvfp4_layout",
    "sha256_row_nvfp4_prefix",
    "sha256_row_prefix",
    "sha512_row_digest",
    "sha512_row_layout",
    "sha512_row_prefix",
    "sha512_row_seg_digest",
    "sha512_row_seg_prefix",
    "sha512_row_seg_segment_prefix",
    "sha512_row_seg_segments",
    "sha512_row_seg_word_digest",
    "sha512_row_v2_digest",
    "sha512_row_v2_layout",
    "sha512_row_v2_prefix",
]

SCHEMA_ROW = "poseidon2-babybear-w24/row/v2h"
SCHEMA_SHA256_ROW = "sha256/row/v1"
SCHEMA_BLAKE3_ROW = "blake3-keyed/row/v2"
SCHEMA_SHA256_ROW_NVFP4 = "sha256/row-nvfp4/v1"
SCHEMA_BLAKE3_ROW_NVFP4 = "blake3-keyed/row-nvfp4/v1"
SCHEMA_SHA512_ROW = "sha512/row/v1"
SCHEMA_HM96_SHA512_ROW = "hm96-sha512/row/v1"
SCHEMA_SHA512_ROW_V2 = "sha512/row/v2"
SCHEMA_HM96_SHA512_ROW_V2 = "hm96-sha512/row/v2"
SCHEMA_SHA512_ROW_SEG = "sha512/row-seg/v1"
SCHEMA_HM96_SHA512_ROW_SEG = "hm96-sha512/row-seg/v1"
_DIGEST_SCHEMAS = {SCHEMA_SHA256_ROW: 32, SCHEMA_BLAKE3_ROW: 32, SCHEMA_SHA256_ROW_NVFP4: 32, SCHEMA_BLAKE3_ROW_NVFP4: 32, SCHEMA_SHA512_ROW: 64,
                   SCHEMA_HM96_SHA512_ROW: 64, SCHEMA_SHA512_ROW_V2: 64, SCHEMA_HM96_SHA512_ROW_V2: 64,
                   SCHEMA_SHA512_ROW_SEG: 64, SCHEMA_HM96_SHA512_ROW_SEG: 64}

ROLE_X, ROLE_W = h2.ROLE_X, h2.ROLE_W
_ROLE_KEY_NAME = {ROLE_X: b"x", ROLE_W: b"w"}
SHA256_ROW_TAG = b"verity/sha256-row/v1\0"
SHA256_ROW_NVFP4_TAG = b"verity/sha256-row-nvfp4/v1\0"
SHA512_ROW_TAG = b"verity/sha512-row/v1\0"
SHA512_ROW_V2_TAG = b"verity/sha512-row/v2\0"
SHA512_ROW_SEG_TAG = b"verity/sha512-row-seg/v1\0"
SHA512_ROW_SEG_SEGMENT_TAG = b"verity/sha512-row-seg/v1/segment\0"
#: bytes in a sha512/row-seg/v1 segment (eight SHA-512 blocks); the last segment of a row may be shorter
ROW_SEG_BYTES = 1024
NVFP4_SCALE_GROUP = 16
BLAKE3_KEY_PREFIX = b"verity/blake3-leaf/v1/"


def _role(role: int) -> int:
    if role not in _ROLE_KEY_NAME:
        raise ValueError(f"row role must be {ROLE_X} (x row) or {ROLE_W} (W column), not {role!r}")
    return role


def _width(word_bits: int) -> int:
    if word_bits not in (8, 16, 32):
        raise ValueError(f"row words are 8, 16 or 32 bits, not {word_bits!r}")
    return word_bits


def row_bytes(words: Sequence[int], word_bits: int) -> bytes:
    """The row's words as little-endian ``word_bits``-bit integers, in order: the bytes both row digests hash."""

    n = _width(word_bits) // 8
    bound = 1 << word_bits
    out = bytearray()
    for w in words:
        w = int(w)
        if not 0 <= w < bound:
            raise ValueError(f"row word {w} does not fit {word_bits} bits")
        out += w.to_bytes(n, "little")
    return bytes(out)


# --- sha256/row/v1 -----------------------------------------------------------------------------------------------


def sha256_row_prefix(role: int, word_bits: int, n_words: int) -> bytes:
    """The 64-byte first block: ``"verity/sha256-row/v1\\0" || u8 role || u8 word_bits || u32be n_words``, zero padded."""

    if not 0 < n_words < 1 << 32:
        raise ValueError("n_words must be a positive u32")
    head = SHA256_ROW_TAG + bytes([_role(role), _width(word_bits)]) + n_words.to_bytes(4, "big")
    return head.ljust(64, b"\0")


def sha256_row_layout(role: int, word_bits: int, n_words: int) -> LeafLayout:
    return LeafLayout("sha256", sha256_row_prefix(role, word_bits, n_words), n_words * word_bits // 8)


def sha256_row_digest(words: Sequence[int], word_bits: int, role: int) -> bytes:
    return sha256_row_layout(role, word_bits, len(words)).digest(row_bytes(words, word_bits))


# --- sha512/row/v1 -----------------------------------------------------------------------------------------------


def sha512_row_prefix(role: int, word_bits: int, n_words: int) -> bytes:
    """The 128-byte first block: ``"verity/sha512-row/v1\\0" || u8 role || u8 word_bits || u32be n_words``, zero padded."""

    if not 0 < n_words < 1 << 32:
        raise ValueError("n_words must be a positive u32")
    head = SHA512_ROW_TAG + bytes([_role(role), _width(word_bits)]) + n_words.to_bytes(4, "big")
    return head.ljust(128, b"\0")


def sha512_row_layout(role: int, word_bits: int, n_words: int) -> LeafLayout:
    return LeafLayout("sha512", sha512_row_prefix(role, word_bits, n_words), n_words * word_bits // 8)


def sha512_row_digest(words: Sequence[int], word_bits: int, role: int) -> bytes:
    return sha512_row_layout(role, word_bits, len(words)).digest(row_bytes(words, word_bits))


# --- sha512/row/v2 -----------------------------------------------------------------------------------------------


def bit_row(*parts: tuple[Sequence[int], int]) -> tuple[bytes, int]:
    """The row of ``(values, bits)`` parts, in order, each value its ``bits`` bits least significant first: ``(row, n_bits)``.

    Row bit ``i`` is bit ``i % 8`` of byte ``i // 8``, and the last byte's fill is zero bits.  For every format in use that
    is the tensor's memory: ``bit_row((codes, 4), (scales, 8))`` is ``nvfp4_row_bytes(codes, scales)``, an E4M3 row is its
    bytes, and ``bit_row((elements, 16))`` is BF16 elements little-endian."""

    out, acc, n, total = bytearray(), 0, 0, 0
    for values, bits in parts:
        if isinstance(bits, bool) or not isinstance(bits, int) or bits < 1:
            raise ValueError(f"an element is a positive number of bits, not {bits!r}")
        bound = 1 << bits
        for v in values:
            v = int(v)
            if not 0 <= v < bound:
                raise ValueError(f"row element {v} does not fit {bits} bits")
            acc |= v << n
            n += bits
            total += bits
            while n >= 8:
                out.append(acc & 0xFF)
                acc >>= 8
                n -= 8
    if n:
        out.append(acc)
    return bytes(out), total


def sha512_row_v2_prefix(n_bits: int) -> bytes:
    """``P(n_bits)``, the 128-byte first block: ``"verity/sha512-row/v2\\0" || u64be n_bits``, zero padded."""

    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or not 0 <= n_bits < 1 << 64:
        raise ValueError(f"a row's length is a u64 number of bits, not {n_bits!r}")
    return (SHA512_ROW_V2_TAG + n_bits.to_bytes(8, "big")).ljust(128, b"\0")


def sha512_row_v2_layout(n_bits: int) -> LeafLayout:
    """The ``ceil(n_bits / 8)`` row bytes after ``P(n_bits)``; a gadget wires the fill bits of the last byte as zeros."""

    return LeafLayout("sha512", sha512_row_v2_prefix(n_bits), (n_bits + 7) // 8)


def sha512_row_v2_digest(row: bytes, n_bits: int) -> bytes:
    """``SHA-512(P(n_bits) || row)`` for a row of :func:`bit_row`'s bytes, refusing a row whose fill bits are not zero."""

    layout, row = sha512_row_v2_layout(n_bits), bytes(row)
    if len(row) == layout.value_len and n_bits % 8 and row[-1] >> n_bits % 8:
        raise ValueError(f"a row of {n_bits} bits fills its last byte with zero bits")
    return layout.digest(row)


# --- sha512/row-seg/v1 -------------------------------------------------------------------------------------------


def _n_bits(n_bits: int) -> int:
    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or not 0 <= n_bits < 1 << 64:
        raise ValueError(f"a row's length is a u64 number of bits, not {n_bits!r}")
    return n_bits


def sha512_row_seg_segments(n_bits: int) -> list[int]:
    """The segments' lengths in bits, ``m_0 .. m_(s-1)``: ``s = ceil(n_bits / 8192)``, every segment 8192 bits but the last,
    which holds the rest (1 to 8192 bits); the empty row has no segment."""

    full, rest = divmod(_n_bits(n_bits), 8 * ROW_SEG_BYTES)
    return [8 * ROW_SEG_BYTES] * full + ([rest] if rest else [])


def sha512_row_seg_prefix(n_bits: int) -> bytes:
    """``Q(n_bits)``, the outer 128-byte first block: ``"verity/sha512-row-seg/v1\\0" || u64be n_bits || u64be s ||
    u32be 1024``, zero padded."""

    s = len(sha512_row_seg_segments(n_bits))
    return (SHA512_ROW_SEG_TAG + n_bits.to_bytes(8, "big") + s.to_bytes(8, "big") + ROW_SEG_BYTES.to_bytes(4, "big")).ljust(128, b"\0")


def sha512_row_seg_segment_prefix(j: int, s: int, m_bits: int) -> bytes:
    """``S(j, s, m)``, segment ``j``'s 128-byte first block: ``"verity/sha512-row-seg/v1/segment\\0" || u64be j || u64be s ||
    u64be m`` (``m`` the segment's length in bits), zero padded."""

    if not 0 <= j < s < 1 << 64 or not 0 < m_bits <= 8 * ROW_SEG_BYTES:
        raise ValueError(f"segment {j} of {s} holds 1 to {8 * ROW_SEG_BYTES} bits, not {m_bits}")
    return (SHA512_ROW_SEG_SEGMENT_TAG + j.to_bytes(8, "big") + s.to_bytes(8, "big") + m_bits.to_bytes(8, "big")).ljust(128, b"\0")


def sha512_row_seg_digest(row: bytes, n_bits: int) -> bytes:
    """``SHA-512(Q(n_bits) || d_0 || ... || d_(s-1))`` with ``d_j = SHA-512(S(j, s, m_j) || segment j)``, segment ``j`` the
    row bytes ``[1024 j, 1024 j + ceil(m_j / 8))``, for a row of :func:`bit_row`'s bytes; refuses a nonzero fill."""

    row = bytes(row)
    if len(row) != (_n_bits(n_bits) + 7) // 8:
        raise ValueError(f"a row of {n_bits} bits is {(n_bits + 7) // 8} bytes, not {len(row)}")
    if n_bits % 8 and row[-1] >> n_bits % 8:
        raise ValueError(f"a row of {n_bits} bits fills its last byte with zero bits")
    segments = sha512_row_seg_segments(n_bits)
    s = len(segments)
    outer = hashlib.sha512(sha512_row_seg_prefix(n_bits))
    for j, m in enumerate(segments):
        start = j * ROW_SEG_BYTES
        outer.update(hashlib.sha512(sha512_row_seg_segment_prefix(j, s, m) + row[start:start + (m + 7) // 8]).digest())
    return outer.digest()


def sha512_row_seg_word_digest(words: Sequence[int], word_bits: int) -> bytes:
    """A word row (BF16 or FP32 elements, ...) under ``sha512/row-seg/v1``: the bit row ``bit_row((words, word_bits))``, whose
    bytes are :func:`row_bytes` (the words little-endian) and whose length is ``word_bits * len(words)``.  No role or word
    width enters the digest, so it differs from the word row's ``sha512/row/v1`` digest."""

    return sha512_row_seg_digest(*bit_row((words, word_bits)))


# --- blake3-keyed/row/v2 -----------------------------------------------------------------------------------------


def blake3_row_key(role: int) -> bytes:
    """``"verity/blake3-leaf/v1/x"`` (x rows) or ``".../w"`` (W columns), zero padded to 32 bytes."""

    return (BLAKE3_KEY_PREFIX + _ROLE_KEY_NAME[_role(role)]).ljust(_blake3.KEY_LEN, b"\0")


def blake3_row_layout(role: int, word_bits: int, n_words: int) -> LeafLayout:
    if not 0 < n_words:
        raise ValueError("n_words must be positive")
    return LeafLayout("blake3", b"", n_words * _width(word_bits) // 8, key=blake3_row_key(role))


def blake3_row_digest(words: Sequence[int], word_bits: int, role: int) -> bytes:
    return blake3_row_layout(role, word_bits, len(words)).digest(row_bytes(words, word_bits))


# --- NVFP4 rows: sha256/row-nvfp4/v1 and blake3-keyed/row-nvfp4/v1 ------------------------------------------------


def _nvfp4_k(k: int) -> int:
    if not 0 < k < 1 << 32 or k % NVFP4_SCALE_GROUP:
        raise ValueError(f"an NVFP4 row is a positive multiple of {NVFP4_SCALE_GROUP} codes (u32), not {k!r}")
    return k


def nvfp4_row_bytes(codes: Sequence[int], scales: Sequence[int]) -> bytes:
    """``k`` E2M1 codes packed two per byte (code ``2i`` in the low nibble, ``2i+1`` in the high), then the ``k/16`` UE4M3
    scale bytes, both in row order: ``k/2 + k/16`` bytes (864 at k = 1536)."""

    k = _nvfp4_k(len(codes))
    if len(scales) != k // NVFP4_SCALE_GROUP:
        raise ValueError(f"{k} codes take {k // NVFP4_SCALE_GROUP} scale bytes, not {len(scales)}")
    c = [int(x) for x in codes]
    if any(not 0 <= x < 16 for x in c):
        raise ValueError("an E2M1 code is 4 bits")
    s = [int(x) for x in scales]
    if any(not 0 <= x < 256 for x in s):
        raise ValueError("a UE4M3 scale is one byte")
    return bytes(c[2 * i] | (c[2 * i + 1] << 4) for i in range(k // 2)) + bytes(s)


def _nvfp4_len(k: int) -> int:
    return k // 2 + k // NVFP4_SCALE_GROUP


def sha256_row_nvfp4_prefix(role: int, k: int) -> bytes:
    """The 64-byte first block: ``"verity/sha256-row-nvfp4/v1\\0" || u8 role || u8 4 || u32be k || u32be k/16``, zero padded."""

    k = _nvfp4_k(k)
    head = SHA256_ROW_NVFP4_TAG + bytes([_role(role), 4]) + k.to_bytes(4, "big") + (k // NVFP4_SCALE_GROUP).to_bytes(4, "big")
    return head.ljust(64, b"\0")


def sha256_row_nvfp4_layout(role: int, k: int) -> LeafLayout:
    return LeafLayout("sha256", sha256_row_nvfp4_prefix(role, k), _nvfp4_len(k))


def sha256_row_nvfp4_digest(codes: Sequence[int], scales: Sequence[int], role: int) -> bytes:
    return sha256_row_nvfp4_layout(role, len(codes)).digest(nvfp4_row_bytes(codes, scales))


def blake3_row_nvfp4_layout(role: int, k: int) -> LeafLayout:
    return LeafLayout("blake3", b"", _nvfp4_len(_nvfp4_k(k)), key=blake3_row_key(role))


def blake3_row_nvfp4_digest(codes: Sequence[int], scales: Sequence[int], role: int) -> bytes:
    return blake3_row_nvfp4_layout(role, len(codes)).digest(nvfp4_row_bytes(codes, scales))


# --- poseidon2-babybear-w24/row/v2h ------------------------------------------------------------------------------


def row_leaf_value(digest) -> bytes:
    """The 8 Poseidon2 digest lanes as 32 bytes (each lane a big-endian u32)."""
    d = [int(x) for x in digest]
    if len(d) != h2.OUT or any(not 0 <= x < h2.P for x in d):
        raise ValueError("a row digest is 8 BabyBear lanes")
    return b"".join(x.to_bytes(4, "big") for x in d)


def row_digest(words, word_bits: int, role: int) -> list[int]:
    """Poseidon2 digest (8 lanes) of one row of ``word_bits``-bit words (``role``: ``h2.ROLE_X`` or ``h2.ROLE_W``)."""
    return h2.hash_row(words, word_bits, role)


def row_leaf(domain: CommitmentDomain, rank: int, digest, schema: str = SCHEMA_ROW) -> bytes:
    """The frame-v3 leaf of the row at ``rank`` (rank == position in a range domain).

    ``digest`` is 8 Poseidon2 lanes for ``leaf/v2h`` (the default), the 32-byte SHA-256 / BLAKE3 row digest (four digest
    schemas, NVFP4 rows included), the 64-byte ``sha512/row/v1`` or ``sha512/row/v2`` digest (under a ``frame-v3-sha512``
    domain), or a hiding row's hm96 leaf digest under ``hm96-sha512/row/v1``, ``hm96-sha512/row/v2`` or
    ``hm96-sha512/row-seg/v1`` (64 bytes); ``sha512/row-seg/v1`` digests too.
    """
    value = row_leaf_value(digest) if schema == SCHEMA_ROW else bytes(digest)
    if schema != SCHEMA_ROW and len(value) != _DIGEST_SCHEMAS.get(schema, -1):
        raise ValueError(f"unknown row schema {schema!r} or a digest of the wrong size ({len(value)} bytes)")
    return domain.leaf(rank, rank, schema, value)
