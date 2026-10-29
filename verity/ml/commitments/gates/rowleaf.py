"""sha512/row-seg/v1's row digest (`verity.ml.commitments.rowleaf.sha512_row_seg_digest`) as Boolean Definitions: the
unsalted digest ``x`` of a bit row that a ``hm96-sha512/row-seg/v1`` leaf hides (`gates.hm96`'s ``x``).

    Sha512RowSeg_v1{N_BITS}
        (row: Array<N_BITS, Value<1>>) -> Digest
        Digest = Array<8, Array<64, Value<1>>>; N_BITS >= 1

``row`` is the bit row (`rowleaf.bit_row`): bit i is bit i % 8 of the row's byte i // 8, so a row of little-endian words is the
words' bits LSB first, one word after another. The digest is SHA-512(Q(N_BITS) || d_0 || ... || d_(s-1)) with
d_j = SHA-512(S(j, s, m_j) || segment j), segment j the row's bits [8192 j, 8192 j + m_j) as ceil(m_j / 8) bytes with zero
fill, s = ceil(N_BITS / 8192) and every m_j 8192 but the last (`rowleaf.sha512_row_seg_segments`). Every Q and S is one
constant 128-byte block, so the circuit starts each hash from its prefix's midstate and compresses only what follows it.

**The parts,** each its own traced Definition (`trace.part`, which C-Flock lowers by its builder), so a unit of a proof holds
at most one segment's hash:

    Sha512RowSegSegment_v1{M}         (cv: Digest, segment: Array<M, Value<1>>) -> Digest, 1 <= M <= 8192
        SHA-512 from the chaining value cv over the segment's ceil(M / 8) bytes and the padding of a message of 128 + ceil(M / 8)
        bytes: d_j when cv is S(j, s, M)'s midstate. One Definition serves every segment of a length, whatever j and s.
    Sha512RowSegOuter_v1{N_BITS}      (d: Array<s, Digest>) -> Digest, SHA-512(Q(N_BITS) || d_0 || ... || d_(s-1))

The row's body passes each segment its midstate as constant bits, so the segment's own constants (its zero fill, its padding)
fold into the traced circuit and its chaining value does not. **Cost:** a full segment is nine compressions (its 1,024 bytes and
a padding block), a short one ceil((ceil(m / 8) + 17) / 128), the outer hash s / 2 + 1 for even s and (s + 1) / 2 for odd s. A
64 × 64 Pearl-C tile's 32 KiB (``-h3``'s tile digest, 262,144 bits) is 32 × 9 + 17 = 305; ``PARTS`` and ``WHOLE`` in the tests
pin the gates.
"""
from __future__ import annotations

from typing import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ZERO, const_bits
from verity.ml.circuits.boolean.trace import part as _part
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array

from .. import rowleaf as R
from .sha512 import compress_gadget, midstate, pad_tail, words_be

__all__ = ["Sha512RowSeg", "Sha512RowSegOuter", "Sha512RowSegSegment", "arguments", "digest", "digest_bits", "digest_bytes",
           "outer_hash", "reference", "segment_hash"]

BIT = BL.BIT
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
#: bits in a full segment
SEG_BITS = 8 * R.ROW_SEG_BYTES


def reference(row: bytes, n_bits: int) -> bytes:
    """`Sha512RowSeg_v1{N_BITS = n_bits}`'s reference: `rowleaf.sha512_row_seg_digest`."""
    return R.sha512_row_seg_digest(row, n_bits)


def arguments(row: bytes, n_bits: int) -> list[int]:
    """`Sha512RowSeg_v1{N_BITS = n_bits}`'s argument, the row's bits in leaf order; refuses what the reference refuses."""
    _check_bits(n_bits)
    row = bytes(row)
    if len(row) != (n_bits + 7) // 8 or (n_bits % 8 and row[-1] >> n_bits % 8):
        raise ValueError(f"a row of {n_bits} bits is {(n_bits + 7) // 8} bytes with zero fill")
    return [(row[i >> 3] >> (i & 7)) & 1 for i in range(n_bits)]


def digest_bits(d: bytes) -> list[int]:
    """A 64-byte digest as a `Digest`'s 512 bits in leaf order (its eight words big-endian, each LSB first)."""
    if len(d) != 64:
        raise ValueError(f"a digest is 64 bytes, not {len(d)}")
    return [(w >> j) & 1 for w in words_be(d) for j in range(64)]


def digest_bytes(bits: Sequence[int]) -> bytes:
    """A `Digest`'s 512 bits in leaf order as its 64 bytes."""
    if len(bits) != 512:
        raise ValueError(f"a digest is 512 bits, not {len(bits)}")
    return b"".join(sum(int(bits[64 * k + j]) << j for j in range(64)).to_bytes(8, "big") for k in range(8))


def _check_bits(n_bits: int) -> None:
    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or not 1 <= n_bits < 1 << 64:
        raise ValueError(f"a row's N_BITS is an int in 1 ... 2^64 - 1, not {n_bits!r}")


def _check_segment(m: int) -> None:
    if isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= SEG_BITS:
        raise ValueError(f"a segment's M is an int in 1 ... {SEG_BITS}, not {m!r}")


# --- the parts over forms (a byte is 8 bits LSB first; a word 64 bits LSB first, its bytes big-endian) -------------------------

def _words(data: Sequence[Sequence]) -> list[list]:
    return [[x for b in range(8) for x in data[8 * k + 7 - b]] for k in range(len(data) // 8)]


def _sha(C, cv: Sequence[Sequence], msg: list[list]) -> list[list]:
    st = [list(w) for w in cv]
    for i in range(0, len(msg), 16):
        st = compress_gadget(C, st, msg[i:i + 16])
    return st


def segment_hash(C, m: int, cv: Sequence[Sequence], bits: Sequence) -> list[list]:
    """SHA-512 from ``cv`` over ``bits`` (m of them) as bytes with zero fill, padded as the end of a 128 + ceil(m / 8)-byte
    message."""
    n = (m + 7) // 8
    full = list(bits) + [ZERO] * (8 * n - m)
    data = [full[8 * o:8 * o + 8] for o in range(n)] + [const_bits(b, 8) for b in pad_tail(128 + n)]
    return _sha(C, cv, _words(data))


def outer_hash(C, n_bits: int, d: Sequence[Sequence]) -> list[list]:
    """SHA-512(Q(n_bits) || the digests ``d``, eight words each) from Q's midstate."""
    msg = [list(w) for w in d] + [const_bits(v, 64) for v in words_be(pad_tail(128 + 8 * len(d)))]
    return _sha(C, [const_bits(v, 64) for v in midstate(R.sha512_row_seg_prefix(n_bits))], msg)


# --- traced Definitions ------------------------------------------------------------------------------------------------------

def _sig_segment(S):
    _check_segment(S.M)
    return (("cv", DIGEST), ("segment", Array(S.M, BIT))), DIGEST


Sha512RowSegSegment = _part(
    "Sha512RowSegSegment", ("M",), _sig_segment, lambda C, S, *e: segment_hash(C, S.M, e[:8], e[8]),
    "sha512/row-seg/v1's segment hash from its prefix's midstate cv: SHA-512 continued over the segment's M bits as "
    "ceil(M / 8) bytes with zero fill, padded as a 128 + ceil(M / 8)-byte message")


def _segments(n_bits: int) -> list[int]:
    _check_bits(n_bits)
    return R.sha512_row_seg_segments(n_bits)


def _sig_outer(S):
    return (("d", Array(len(_segments(S.N_BITS)), DIGEST)),), DIGEST


Sha512RowSegOuter = _part(
    "Sha512RowSegOuter", ("N_BITS",), _sig_outer, lambda C, S, *e: outer_hash(C, S.N_BITS, e),
    "sha512/row-seg/v1's outer hash: SHA-512(Q(N_BITS) || d_0 || ... || d_(s-1)) from Q's midstate")


def _signature(S):
    _check_bits(S.N_BITS)
    return (("row", Array(S.N_BITS, BIT)),), DIGEST


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _body(B, S, row):
    bits = _refs(row)
    one = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]
    segments = _segments(S.N_BITS)
    d = []
    for j, m in enumerate(segments):
        cv = [one[(v >> k) & 1] for v in midstate(R.sha512_row_seg_segment_prefix(j, len(segments), m)) for k in range(64)]
        seg = bits[SEG_BITS * j:SEG_BITS * j + m]
        d += _refs(B.call(bind(Sha512RowSegSegment, M=m), Coll(DIGEST, TR._refs(cv)), Coll(Array(m, BIT), TR._refs(seg))))
    return B.call(bind(Sha512RowSegOuter, N_BITS=S.N_BITS), Coll(Array(len(segments), DIGEST), TR._refs(d)))


Sha512RowSeg = CompositeDefinition(
    "Sha512RowSeg", 1, ("N_BITS",), _signature, _body, conformance="exact-model-tested",
    doc="sha512/row-seg/v1's digest of an N_BITS-bit row (rowleaf.sha512_row_seg_digest): each 8192-bit segment hashed from its "
        "prefix's midstate (Sha512RowSegSegment_v1), then the digests from Q's (Sha512RowSegOuter_v1)")


def digest(n_bits: int):
    return bind(Sha512RowSeg, N_BITS=n_bits)
