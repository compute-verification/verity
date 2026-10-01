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
                             (``backends/direct/ligero/leaf/blake3.py``), promoted byte for byte; its schema string and
                             keys are pinned by ``ligero-verify``.
``sha256/row-nvfp4/v1``      ``SHA-256(sha256_row_nvfp4_prefix(role, k) || nvfp4_row_bytes)`` for an NVFP4 row (``k`` E2M1
                             codes and their ``k/16`` UE4M3 scales).
``blake3-keyed/row-nvfp4/v1`` ``BLAKE3(nvfp4_row_bytes, key=blake3_row_key(role))``: the same row bytes, the frame-v3 keys.
``sha512/row/v1``            ``SHA-512(sha512_row_prefix(role, word_bits, n_words) || row_bytes)``: ``sha256/row/v1`` through SHA-512,
                             with a 128-byte constant first block (``frame-v3-sha512`` rows and ``hm96-sha512/v1``'s inner digest).
``sha512/row-nvfp4/v1``, ``sha512/row-mxfp4/v1``, ``sha512/row-e4m3/v1``
                             ``SHA-512(sha512_element_row_prefix(format, role, k) || element_row_bytes)``: a row of ``k`` elements of
                             an :class:`ElementFormat` (its codes, then its scales, as the tensor holds them), the format, role, code
                             width, ``k`` and scale count in a 128-byte constant first block.
``hm96-sha256/row/v1``       the schema a frame-v3 leaf binds over a hiding row's leaf digest: ``hm96.SHA256.tree_leaf(key, b || c)``
                             over ``sha256/row/v1`` (32 bytes).
``hm96-sha512/row/v1``       the same under a ``frame-v3-sha512`` domain: ``hm96.SHA512.tree_leaf(key, b || c)`` over
                             ``sha512/row/v1`` (64 bytes); ``hm96-sha512/row-<format>/v1`` over an element row's digest.
``poseidon2-babybear-w24/row/v2h``  the Poseidon2 row digest. Admissible in a Table 2 cell (``frame-v3/poseidon2-babybear``),
                             but algebraic: its results are reported and flagged as algebraic in the security section,
                             and never count toward a standard-hash claim (Daniel's standing decision).

``row_bytes`` are the row's words little-endian, ``word_bits`` 8, 16 or 32 bits each, in row order.  ``nvfp4_row_bytes``
are the codes packed two per byte (code ``2i`` in the low nibble), then the scale bytes: the code buffer and the scale
buffer as NVFP4 tensors hold them, so a serving-side committer hashes rows without repacking.  ``element_row_bytes`` is
the same rule for every :class:`ElementFormat` (``nvfp4_row_bytes`` is its NVFP4 case; one E4M3 element a byte, no scales).
Roles are :data:`ROLE_X` (1, an x row) and :data:`ROLE_W` (2, a W column).  Unsalted: a low-entropy row is preimage-searchable.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from . import blake3 as _blake3
from . import poseidon2_babybear as h2
from .merkle import CommitmentDomain
from .scheme import LeafLayout

__all__ = [
    "E4M3",
    "ELEMENT_FORMATS",
    "MXFP4",
    "NVFP4",
    "ROLE_W",
    "ROLE_X",
    "SCHEMA_BLAKE3_ROW",
    "SCHEMA_BLAKE3_ROW_NVFP4",
    "SCHEMA_HM96_SHA256_ROW",
    "SCHEMA_HM96_SHA512_ROW",
    "SCHEMA_ROW",
    "SCHEMA_SHA256_ROW",
    "SCHEMA_SHA256_ROW_NVFP4",
    "SCHEMA_SHA512_ROW",
    "ElementFormat",
    "blake3_row_digest",
    "blake3_row_key",
    "blake3_row_layout",
    "blake3_row_nvfp4_digest",
    "blake3_row_nvfp4_layout",
    "element_row_bytes",
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
    "sha512_element_row_digest",
    "sha512_element_row_layout",
    "sha512_element_row_prefix",
    "sha512_row_digest",
    "sha512_row_layout",
    "sha512_row_prefix",
]

SCHEMA_ROW = "poseidon2-babybear-w24/row/v2h"
SCHEMA_SHA256_ROW = "sha256/row/v1"
SCHEMA_BLAKE3_ROW = "blake3-keyed/row/v2"
SCHEMA_SHA256_ROW_NVFP4 = "sha256/row-nvfp4/v1"
SCHEMA_BLAKE3_ROW_NVFP4 = "blake3-keyed/row-nvfp4/v1"
SCHEMA_SHA512_ROW = "sha512/row/v1"
SCHEMA_HM96_SHA256_ROW = "hm96-sha256/row/v1"
SCHEMA_HM96_SHA512_ROW = "hm96-sha512/row/v1"


@dataclass(frozen=True, slots=True)
class ElementFormat:
    """A row of ``k`` codes of ``code_bits`` bits (``code``: their encoding) and, when ``scale_group``, one ``scale`` byte per
    ``scale_group`` codes: an element row, hashed as the tensor holds it (frame_v3/PROTOCOL.md §6, element rows)."""

    name: str
    code: str
    code_bits: int
    scale: str | None = None
    scale_group: int = 0

    @property
    def schema(self) -> str:
        return f"sha512/row-{self.name}/v1"

    @property
    def hiding_schema(self) -> str:
        return f"hm96-sha512/row-{self.name}/v1"

    @property
    def tag(self) -> bytes:
        return f"verity/sha512-row-{self.name}/v1\0".encode()

    def check_k(self, k: int) -> int:
        group = self.scale_group or (8 // self.code_bits)
        if not 0 < k < 1 << 32 or k % group:
            many = f"a positive multiple of {group}" if group > 1 else "a positive number of"
            raise ValueError(f"an {self.name} row is {many} codes (u32), not {k!r}")
        return k

    def n_scales(self, k: int) -> int:
        return self.check_k(k) // self.scale_group if self.scale_group else 0

    def row_len(self, k: int) -> int:
        """The row's bytes: ``k * code_bits / 8`` code bytes, then the scale bytes (no padding)."""
        return self.check_k(k) * self.code_bits // 8 + self.n_scales(k)


NVFP4 = ElementFormat("nvfp4", "e2m1", 4, "ue4m3", 16)
MXFP4 = ElementFormat("mxfp4", "e2m1", 4, "ue8m0", 32)
E4M3 = ElementFormat("e4m3", "e4m3", 8)
ELEMENT_FORMATS = {f.name: f for f in (NVFP4, MXFP4, E4M3)}

_DIGEST_SCHEMAS = {SCHEMA_SHA256_ROW: 32, SCHEMA_BLAKE3_ROW: 32, SCHEMA_SHA256_ROW_NVFP4: 32, SCHEMA_BLAKE3_ROW_NVFP4: 32, SCHEMA_SHA512_ROW: 64,
                   SCHEMA_HM96_SHA256_ROW: 32, SCHEMA_HM96_SHA512_ROW: 64,
                   **{s: 64 for f in ELEMENT_FORMATS.values() for s in (f.schema, f.hiding_schema)}}

ROLE_X, ROLE_W = h2.ROLE_X, h2.ROLE_W
_ROLE_KEY_NAME = {ROLE_X: b"x", ROLE_W: b"w"}
SHA256_ROW_TAG = b"verity/sha256-row/v1\0"
SHA256_ROW_NVFP4_TAG = b"verity/sha256-row-nvfp4/v1\0"
SHA512_ROW_TAG = b"verity/sha512-row/v1\0"
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
    return NVFP4.check_k(k)


def _format(fmt: ElementFormat | str) -> ElementFormat:
    f = ELEMENT_FORMATS.get(fmt) if isinstance(fmt, str) else fmt
    if f not in ELEMENT_FORMATS.values():
        raise ValueError(f"element formats are {sorted(ELEMENT_FORMATS)}, not {fmt!r}")
    return f


def element_row_bytes(fmt: ElementFormat | str, codes: Sequence[int], scales: Sequence[int] = ()) -> bytes:
    """An element row's bytes: the ``k`` codes in row order (4-bit codes two per byte, code ``2i`` in the low nibble and
    ``2i+1`` in the high; 8-bit codes one per byte), then the ``k / scale_group`` scale bytes in row order."""

    f = _format(fmt)
    k = f.check_k(len(codes))
    if len(scales) != f.n_scales(k):
        raise ValueError(f"{k} codes take {f.n_scales(k)} scale bytes, not {len(scales)}")
    c = [int(x) for x in codes]
    if any(not 0 <= x < 1 << f.code_bits for x in c):
        raise ValueError(f"an {f.code.upper()} code is {f.code_bits} bits")
    s = [int(x) for x in scales]
    if any(not 0 <= x < 256 for x in s):
        raise ValueError(f"a {(f.scale or '').upper()} scale is one byte")
    body = bytes(c[2 * i] | (c[2 * i + 1] << 4) for i in range(k // 2)) if f.code_bits == 4 else bytes(c)
    return body + bytes(s)


def nvfp4_row_bytes(codes: Sequence[int], scales: Sequence[int]) -> bytes:
    """``k`` E2M1 codes packed two per byte (code ``2i`` in the low nibble, ``2i+1`` in the high), then the ``k/16`` UE4M3
    scale bytes, both in row order: ``k/2 + k/16`` bytes (864 at k = 1536)."""

    return element_row_bytes(NVFP4, codes, scales)


def _nvfp4_len(k: int) -> int:
    return NVFP4.row_len(k)


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


# --- element rows: sha512/row-nvfp4/v1, sha512/row-mxfp4/v1, sha512/row-e4m3/v1 ----------------------------------


def sha512_element_row_prefix(fmt: ElementFormat | str, role: int, k: int) -> bytes:
    """The 128-byte first block: ``"verity/sha512-row-<format>/v1\\0" || u8 role || u8 code_bits || u32be k || u32be n_scales``,
    zero padded."""

    f = _format(fmt)
    head = f.tag + bytes([_role(role), f.code_bits]) + f.check_k(k).to_bytes(4, "big") + f.n_scales(k).to_bytes(4, "big")
    return head.ljust(128, b"\0")


def sha512_element_row_layout(fmt: ElementFormat | str, role: int, k: int) -> LeafLayout:
    f = _format(fmt)
    return LeafLayout("sha512", sha512_element_row_prefix(f, role, k), f.row_len(k))


def sha512_element_row_digest(fmt: ElementFormat | str, codes: Sequence[int], scales: Sequence[int], role: int) -> bytes:
    return sha512_element_row_layout(fmt, role, len(codes)).digest(element_row_bytes(fmt, codes, scales))


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
    schemas, NVFP4 rows included), the 64-byte ``sha512/row/v1`` or ``sha512/row-<format>/v1`` digest (under a
    ``frame-v3-sha512`` domain), or a hiding row's hm96 leaf digest under ``hm96-sha256/row/v1`` (32 bytes),
    ``hm96-sha512/row/v1`` or ``hm96-sha512/row-<format>/v1`` (64 bytes).
    """
    value = row_leaf_value(digest) if schema == SCHEMA_ROW else bytes(digest)
    if schema != SCHEMA_ROW and len(value) != _DIGEST_SCHEMAS.get(schema, -1):
        raise ValueError(f"unknown row schema {schema!r} or a digest of the wrong size ({len(value)} bytes)")
    return domain.leaf(rank, rank, schema, value)
