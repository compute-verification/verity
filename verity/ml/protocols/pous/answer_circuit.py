"""PoUS's audit answer as a Boolean circuit: the label the memory challenger received for round j is label i_j of C, by
its hiding leaf.  One input per round, k per audit.

    PousAnswerOpens_v1{N}           (block: Array<8N>, h: Array<512>, y: Array<1536>, bc: Array<1024>) -> Array<1>
    PousRowDigest_v1{N}             (row: Array<8N>) -> Array<512>                     sha512/row/v2 of an N-byte row
    PousRowDigestEnd_v1{N, END}     the digest's first block ("head"), last ("tail") or whole, when N < 128 ("short")
    PousSha512Compress_v1           (cv: Array<512>, msg: Array<1024>) -> Array<512>   one SHA-512 compression
    PousHm96Commit_v1               (x: Array<512>, y: Array<1536>) -> Array<1024>     hm96-sha512's b ‖ c
    PousEqual_v1{N}                 (a: Array<N>, b: Array<N>) -> Array<1>             a = b

Every array is of `Value<1>`.  Byte strings (the block, digests, salts, commit strings) are bits in byte order, bit t
being bit t mod 8 of byte ⌊t/8⌋ (hm96's order); a chaining value is SHA-512's eight words, each least significant bit
first.  Every compression is core's `verity.ml.commitments.gates.sha512.compress_gadget`, 58,120 ANDs at most.

**The statement.** For round j the memory challenger signs (i_j, send, receipt, h), where h is the digest of the 2,056
bytes it received.  C's leaf i_j is ``hm96-sha512/row/v2``: the frame leaf over the commit string ``bc = b ‖ c`` of the
inner digest x = SHA-512(P(16448) ‖ block) (`rowleaf.sha512_row_v2_prefix`), with salt y and hm96's pinned key.
`PousAnswerOpens_v1` is 1 iff

    x = SHA-512(P(8N) ‖ block) equals h,   and   hm96-sha512's commit string of (x, y) equals bc.

So h must be the label's ``sha512/row/v2`` digest: the memory challenger hashes P's constant midstate onward, the same
17 compressions as a plain SHA-512 of the label.  The signature over the record, the index and the leaf's place under
C's root are other parts of the session's statement.

At N = 2,056 the circuit is 19 compressions: 16 for the label's whole 128-byte blocks after P's constant first block,
one for its last 8 bytes and the padding, and hm96's two for c (b is XORs).  `verity/core/protocols/pous/tests/test_pous_answer_circuit.py` pins
the counts.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import and_reduce, const_bits
from verity.ml.circuits.boolean.trace import part
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.sha512 import compress_gadget, midstate, pad_tail, words_be
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

from .circuit import _check, _traced_body, bits

__all__ = ["PousAnswerOpens", "PousEqual", "PousHm96Commit", "PousRowDigest", "PousRowDigestEnd", "PousSha512Compress",
           "word_bit"]


def word_bit(t: int) -> int:
    """Where bit t of a byte string (bit t mod 8 of byte ⌊t/8⌋) sits among its big-endian 64-bit words, LSB first."""
    byte, b = t >> 3, t & 7
    return 64 * (byte >> 3) + 56 - 8 * (byte & 7) + b


def to_words(x):
    """A byte string's bits (a multiple of 64) as SHA-512 words."""
    flat = [None] * len(x)
    for t, f in enumerate(x):
        flat[word_bit(t)] = f
    return [flat[64 * i:64 * i + 64] for i in range(len(x) // 64)]


def from_words(words):
    flat = [f for w in words for f in w]
    return [flat[word_bit(t)] for t in range(len(flat))]


def _const_words(data: bytes):
    return [const_bits(v, 64) for v in words_be(data)]


def compress(C, cv, msg):
    return [f for w in compress_gadget(C, [cv[64 * i:64 * i + 64] for i in range(8)], to_words(msg)) for f in w]


def digest_end(C, N: int, END: str, *args):
    """The row digest's first message block from P's midstate (``head``), or its last bytes and padding to the digest
    (``tail``, from a chaining value; ``short``, the whole row when N < 128)."""
    mid = [f for w in midstate(sha512_row_v2_prefix(8 * N)) for f in const_bits(w, 64)]
    if END == "head":
        return compress(C, mid, args[0])
    cv, rest = (mid, args[0]) if END == "short" else (args[0], args[1] if len(args) > 1 else [])
    pad = pad_tail(128 + N)
    tail = list(rest) + [f for byte in pad for f in const_bits(byte, 8)]
    for i in range(0, len(tail), 1024):
        cv = compress(C, cv, tail[i:i + 1024])
    return from_words([cv[64 * i:64 * i + 64] for i in range(8)])


def hm96_commit(C, x, y):
    """hm96-sha512's commit string: b = x ⊕ M(key) y, c = SHA-512(salt prefix ‖ y)."""
    inst = hm96.SHA512
    cv = [f for w in midstate(inst.salt_prefix) for f in const_bits(w, 64)]
    tail = list(y) + [f for byte in pad_tail(128 + inst.salt_bytes) for f in const_bits(byte, 8)]
    for i in range(0, len(tail), 1024):
        cv = compress(C, cv, tail[i:i + 1024])
    c = from_words([cv[64 * i:64 * i + 64] for i in range(8)])
    b = []
    for i, row in enumerate(inst.rows_of(inst.default_key)):
        terms = [x[i]] + [y[j] for j in range(len(y)) if row >> j & 1]
        while len(terms) > 1:
            terms = [terms[k] ^ terms[k + 1] for k in range(0, len(terms) - 1, 2)] + terms[len(terms) & ~1:]
        b.append(terms[0])
    return b + c


def equal(C, a, b):
    return [and_reduce(C, [(p ^ q).inv() for p, q in zip(a, b)])]


# --- the Definitions -------------------------------------------------------------------------------------------------------

def _compress_sig(S):
    return (("cv", bits(512)), ("msg", bits(1024))), bits(512)


PousSha512Compress = part(
    "PousSha512Compress", (), _compress_sig, lambda C, S, cv, msg: compress(C, cv, msg),
    "One SHA-512 compression with the feed-forward (FIPS 180-4), core's compress_gadget: the chaining value as eight "
    "words LSB first, the message block as 128 bytes", conformance="protocol-defined")


def _end_sig(S):
    _check("PousRowDigestEnd", N=S.N)
    rest = S.N % 128
    if S.END == "head" and S.N >= 128:
        return (("msg", bits(1024)),), bits(512)
    if S.END == "tail" and S.N >= 128:
        return (("cv", bits(512)),) + ((("rest", bits(8 * rest)),) if rest else ()), bits(512)
    if S.END == "short" and S.N < 128:
        return (("rest", bits(8 * S.N)),), bits(512)
    raise ValueError(f"PousRowDigestEnd's END is head or tail when N >= 128, short below, not {S.END!r} at N = {S.N}")


PousRowDigestEnd = part(
    "PousRowDigestEnd", ("N", "END"), _end_sig, lambda C, S, *a: digest_end(C, S.N, S.END, *a),
    "sha512/row/v2's first block of an N-byte row (head), its last bytes and padding (tail), or all of it (short)",
    conformance="protocol-defined")


def _digest_sig(S):
    _check("PousRowDigest", N=S.N)
    return (("row", bits(8 * S.N)),), bits(512)


def _digest_body(B, S, row):
    N = S.N
    if N < 128:
        return B.call(bind(PousRowDigestEnd, N=N, END="short"), row)
    cv = B.call(bind(PousRowDigestEnd, N=N, END="head"), row[0:1024])
    for i in range(1, N // 128):
        cv = B.call(bind(PousSha512Compress), cv, row[1024 * i:1024 * (i + 1)])
    rest = (row[1024 * (N // 128):8 * N],) if N % 128 else ()
    return B.call(bind(PousRowDigestEnd, N=N, END="tail"), cv, *rest)


PousRowDigest = CompositeDefinition(
    "PousRowDigest", 1, ("N",), _digest_sig, _digest_body, conformance="protocol-defined",
    doc="The sha512/row/v2 digest of an N-byte row: SHA-512(P(8N) ‖ row), P's block a constant midstate")


def _commit_sig(S):
    return (("x", bits(512)), ("y", bits(1536))), bits(1024)


PousHm96Commit = CompositeDefinition(
    "PousHm96Commit", 1, (), _commit_sig, _traced_body(lambda S: ("hm96",), lambda S: hm96_commit, _commit_sig),
    conformance="protocol-defined",
    doc="hm96-sha512/v1's commit string b ‖ c of inner digest x under salt y and the pinned key: two compressions for c")


def _equal_sig(S):
    _check("PousEqual", N=S.N)
    return (("a", bits(S.N)), ("b", bits(S.N))), bits(1)


PousEqual = CompositeDefinition(
    "PousEqual", 1, ("N",), _equal_sig,
    _traced_body(lambda S: ("equal", S.N), lambda S: equal, _equal_sig), conformance="exact-model-tested",
    doc="1 iff the two N-bit strings are equal: N − 1 ANDs")


def _answer_sig(S):
    _check("PousAnswerOpens", N=S.N)
    return (("block", bits(8 * S.N)), ("h", bits(512)), ("y", bits(1536)), ("bc", bits(1024))), bits(1)


def _answer_body(B, S, block, h, y, bc):
    x = B.call(bind(PousRowDigest, N=S.N), block)
    com = B.call(bind(PousHm96Commit), x, y)
    return B.call(bind(PousEqual, N=1536), Coll(bits(1536), concat([x.refs, com.refs])),
                  Coll(bits(1536), concat([h.refs, bc.refs])))


PousAnswerOpens = CompositeDefinition(
    "PousAnswerOpens", 1, ("N",), _answer_sig, _answer_body, conformance="protocol-defined",
    doc="1 iff the N-byte block's sha512/row/v2 digest is h and its hm96-sha512 commit string under salt y is bc: the "
        "memory challenger's hash of round j's answer and C's hiding leaf i_j open to the same block (N = 2056 for P2)")
