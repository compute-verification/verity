"""SHA-512 continued over a byte stream as Boolean Definitions: a file's digest taken one row of 16-bit words at a time, so a
unit holds one row and the chain's state, never the file (a checkpoint tie: a registered weight's rows hashed in a
safetensors shard's byte order to the shard's SHA-512).

    Sha512StreamRow_v1{H, W}
        (cv: Digest, [carry: Array<C(H), Half>,] data: Array<W, Half>) -> Tuple<Digest, Array<C(H'), Half>>, or Digest if H' = 0
    Sha512StreamLast_v1{H, W, LEN}
        (cv: Digest, [carry: Array<C(H), Half>,] data: Array<W, Half>) -> Digest, LEN mod 128 = H'
        Digest = Array<8, Array<64, Value<1>>>, Half = Array<16, Value<1>>, C(h) = ceil(h / 2);
        0 <= H < 128, W >= 1, H' = (H + 2 W) mod 128, 0 <= LEN < 2^64

**The stream's state** after n bytes is the chaining value after its floor(n / 128) whole blocks (``cv``, from `sha512.IV`) and
its h = n mod 128 pending bytes (``carry``, absent when h = 0): byte 2i the low byte of word i and byte 2i + 1 its high byte, as a
little-endian 16-bit element is stored, an odd h's last word's high byte zero (a Row ignores it). `state` is the reference.

**A Row** appends its row's 2 W bytes (``data``, each word's low byte then its high byte: the row's elements as a safetensors
file holds them) and compresses every block that completes, floor((H + 2 W) / 128) of them; the state it returns is the next
Row's. **The Last** appends a stream's last row the same way and then ends it: the bytes left pending and the padding of a
LEN-byte message (`sha512.pad_tail`), one block or two (H' >= 112). Its Digest is SHA-512 of the stream, the digest's bytes
its words big-endian (`sha512.words_be`). The end is the last row's unit rather than one of its own because C-Flock proves a
unit only with a row that compresses before its last block (at least 112 bytes), and a stream's state alone, a 64-byte
chaining value and fewer than 128 pending bytes, need not have one.

Each compression is a call of `lms.Sha512Compress_v1`, one traced circuit for every block, so a binding costs no trace of its
own; the bodies only lay the bytes out (and write the padding's constant bits). **Cost:** 57,947 ANDs a compression
(`Sha512Compress_v1`, its whole message private): a Row floor((H + 2 W) / 128) of them, the Last one or two more.
"""
from __future__ import annotations

from collections.abc import Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Tuple

from .lms import Sha512Compress
from .sha512 import IV, compress_ref, pad_tail, words_be

__all__ = ["HALF", "Sha512StreamLast", "Sha512StreamRow", "carry_words", "end_reference", "last", "last_reference", "row",
           "row_reference", "state", "state_bits"]

BIT = BL.BIT
WORD = Array(64, BIT)
DIGEST = Array(8, WORD)
HALF = Array(16, BIT)
BLOCK = 128


def _int(name: str, v: object, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v < hi:
        raise ValueError(f"a stream's {name} is an int in {lo} ... {hi - 1}, not {v!r}")
    return v


def _carry(h: int) -> Array:
    return Array((h + 1) // 2, HALF)


# --- reference ---------------------------------------------------------------------------------------------------------------

def state(prefix: bytes, cv: Sequence[int] = IV) -> tuple[list[int], bytes]:
    """The stream's state after ``prefix`` from the chaining value ``cv``: (cv after its whole blocks, its pending bytes)."""
    cv, n = list(cv), len(prefix) - len(prefix) % BLOCK
    for i in range(0, n, BLOCK):
        cv = compress_ref(cv, words_be(prefix[i:i + BLOCK]))
    return cv, bytes(prefix[n:])


def carry_words(pending: bytes) -> list[int]:
    """Pending bytes as the carry's 16-bit words, an odd count's last high byte zero."""
    p = bytes(pending) + bytes(len(pending) % 2)
    return [p[i] | p[i + 1] << 8 for i in range(0, len(p), 2)]


def _row_bytes(data: Sequence[int]) -> bytes:
    if any(not 0 <= int(w) < 1 << 16 for w in data):
        raise ValueError("a row's words are 16 bits")
    return b"".join(int(w).to_bytes(2, "little") for w in data)


def row_reference(cv: Sequence[int], pending: bytes, data: Sequence[int]) -> tuple[list[int], bytes]:
    """`Sha512StreamRow_v1{H = len(pending), W = len(data)}`'s reference: the state after the row's bytes."""
    _int("H", len(pending), 0, BLOCK)
    return state(bytes(pending) + _row_bytes(data), cv)


def end_reference(cv: Sequence[int], pending: bytes, length: int) -> bytes:
    """A ``length``-byte stream's SHA-512 from its state, its pending bytes ``length mod 128`` of them."""
    if _int("LEN", length, 0, 1 << 64) % BLOCK != len(pending):
        raise ValueError(f"a {length}-byte stream ends with {length % BLOCK} pending bytes, not {len(pending)}")
    cv, rest = state(bytes(pending) + pad_tail(length), cv)
    assert not rest
    return b"".join(v.to_bytes(8, "big") for v in cv)


def last_reference(cv: Sequence[int], pending: bytes, data: Sequence[int], length: int) -> bytes:
    """`Sha512StreamLast_v1{H = len(pending), W = len(data), LEN = length}`'s reference: the stream's SHA-512 after the row."""
    return end_reference(*row_reference(cv, pending, data), length)


def state_bits(cv: Sequence[int], pending: bytes) -> list[list[int]]:
    """A state's arguments, each a flat list of bits in leaf order: ``cv``'s 512, then the carry's (none when nothing pends)."""
    out = [[(v >> j) & 1 for v in cv for j in range(64)]]
    if pending:
        out.append([(w >> j) & 1 for w in carry_words(pending) for j in range(16)])
    return out


# --- the bodies (a byte is 8 bits LSB first; a word 64 bits LSB first, its bytes big-endian) ---------------------------------

def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _bytes(halves: Sequence) -> list[list]:
    return [list(halves[8 * i:8 * i + 8]) for i in range(len(halves) // 8)]


def _words(data: Sequence[Sequence]) -> list:
    return [x for k in range(len(data) // 8) for b in range(8) for x in data[8 * k + 7 - b]]


def _compress(B, cv: list, stream: list[list]) -> tuple[list, list[list]]:
    """``cv`` after the stream's whole blocks, and the bytes left over."""
    n = len(stream) - len(stream) % BLOCK
    for i in range(0, n, BLOCK):
        cv = _refs(B.call(Sha512Compress, Coll(DIGEST, TR._refs(cv)), Coll(Array(16, WORD), TR._refs(_words(stream[i:i + BLOCK])))))
    return cv, stream[n:]


def _sig_row(S):
    _int("H", S.H, 0, BLOCK)
    _int("W", S.W, 1, 1 << 32)
    rest = (S.H + 2 * S.W) % BLOCK
    carry = (("carry", _carry(S.H)),) if S.H else ()
    return (("cv", DIGEST), *carry, ("data", Array(S.W, HALF))), Tuple(DIGEST, _carry(rest)) if rest else DIGEST


def _row_body(B, S, cv, *rest):
    carry = _bytes(_refs(rest[0]))[:S.H] if S.H else []
    out, left = _compress(B, _refs(cv), carry + _bytes(_refs(rest[-1])))
    if not left:
        return Coll(DIGEST, TR._refs(out))
    zero = B.call(BL.constant(0)).refs[0]
    tail = [x for b in left for x in b] + [zero] * (8 * (len(left) % 2))
    return Coll(_sig_row(S)[1], TR._refs(out + tail))


Sha512StreamRow = CompositeDefinition(
    "Sha512StreamRow", 1, ("H", "W"), _sig_row, _row_body, conformance="exact-model-tested",
    doc="SHA-512 continued over a row of W 16-bit words (each its low byte, then its high byte) from the state (cv, carry) with H "
        "pending bytes: the chaining value after every block that completes (Sha512Compress_v1) and the (H + 2 W) mod 128 bytes "
        "left pending, an odd count's last high byte zero")


def _sig_last(S):
    params, _ = _sig_row(S)
    if _int("LEN", S.LEN, 0, 1 << 64) % BLOCK != (S.H + 2 * S.W) % BLOCK:
        raise ValueError(f"a {S.LEN}-byte stream ends with {S.LEN % BLOCK} pending bytes, not the {(S.H + 2 * S.W) % BLOCK} "
                         f"a row of {S.W} words leaves after {S.H}")
    return params, DIGEST


def _last_body(B, S, cv, *rest):
    carry = _bytes(_refs(rest[0]))[:S.H] if S.H else []
    bit = [B.call(BL.constant(b)).refs[0] for b in (0, 1)]
    pad = [[bit[(v >> j) & 1] for j in range(8)] for v in pad_tail(S.LEN)]
    out, left = _compress(B, _refs(cv), carry + _bytes(_refs(rest[-1])) + pad)
    assert not left
    return Coll(DIGEST, TR._refs(out))


Sha512StreamLast = CompositeDefinition(
    "Sha512StreamLast", 1, ("H", "W", "LEN"), _sig_last, _last_body, conformance="exact-model-tested",
    doc="The SHA-512 of a LEN-byte stream whose last row is W 16-bit words (each its low byte, then its high byte), from the "
        "state (cv, carry) with H pending bytes before it: the pending bytes, the row's and the padding of a LEN-byte message "
        "compressed (Sha512Compress_v1), the digest's bytes its words big-endian")


def row(H: int, W: int):
    return bind(Sha512StreamRow, H=H, W=W)


def last(H: int, W: int, LEN: int):
    return bind(Sha512StreamLast, H=H, W=W, LEN=LEN)
