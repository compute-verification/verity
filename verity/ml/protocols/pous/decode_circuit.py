"""P2's decode of one label as a Boolean circuit: ``m = sq(σ_K(sq(c))) − t mod p``, with `schemes.p2`'s v4 keys (K and
t from counter-mode SHA-512 of the label's expansion base), so that PoUS's setup can prove a label of C decodes to its
row of W.

    PousP2Decode_v2{W, C0, L}   (block: Array<W>, base: Array<8L>) -> Array<W − 15>    payload ‖ ok
    PousP2Sq_v1{W, C0}          (x: Array<W>) -> Array<W>                               sq(x)
    PousP2Negate_v1{W, C0}      (s: Array<W>, odd: Array<1>) -> Array<W>                odd ? p − s : s
    PousP2Sigma_v1{W, C0}       (r: Array<W>, k: Array<W>) -> Array<W>                  σ_K(r)
    PousP2Finish_v1{W, C0}      (u: Array<W>, t: Array<W>, c: Array<W>) -> Array<W − 15>
    PousSha512Mid_v1{M}         (prefix: Array<1024 M>) -> Array<512>                   SHA-512's chaining value after M blocks
    PousSha512Head_v1           (msg: Array<1024>) -> Array<512>                        its first block, from the IV
    PousSha512Ctr_v1{L, D, NB}  (cv: Array<512>, rest: Array<8 (L mod 128)>) -> Array<8 NB>
                                    the first NB bytes of SHA-512(base ‖ D ‖ u32be 0) ‖ SHA-512(base ‖ D ‖ u32be 1) ‖ …
    PousSha512CtrChunk_v1{L, D, I}  (cv, rest) -> Array<512>                            its chunk I
    PousByteSwap_v1{N}          (x: Array<8N>) -> Array<8N>                             N bytes reversed (no gates)

Every array is of `Value<1>`.  A byte string is bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋ (hm96's order);
an integer is bits least significant first.  A big-endian integer of N bytes and its byte string differ by
`PousByteSwap_v1{N}`, which is wiring.  A chaining value is SHA-512's eight words, each least significant bit first, and
every compression is `answer_circuit`'s `PousSha512Compress_v1` (core's `compress_gadget`).

**The decode** (`schemes.p2.decode_segment` for one label):

- ``block`` is the label's W/8 bytes, so c = int.from_bytes(block, "big"); ``base`` is `schemes.p2`'s v4 expansion base
  ``expand_prefix(params, salt) ‖ u64be(s) ‖ u64be(j)``, L bytes (144 at the registered point: the prefix is zero-padded
  to one 128-byte block);
- K is the first W/8 bytes of the stream ``D = "K"``, and t the first W/8 bytes of ``D = "t"``, both big-endian.  The
  base's whole blocks are compressed once, for both; each 64-byte chunk is then one compression of the base's last
  L mod 128 bytes, D, the counter and SHA-512's padding;
- the result is m = sq(σ_K(sq(c))) − t mod p as its (W − 16)/8 payload bytes, then ``ok``: c < p, t < p, and m < 2^(W−16).

When ``ok`` is 1 the payload is the reference's, byte for byte.  `schemes.p2` rejects a first t candidate ≥ p and reads
the next; the circuit reads only the first, and says so with ``ok`` = 0.  That happens with probability C0 / 2^W per
label (2^-16433.6 at the registered point), so an honest prover meets it never, and a circuit that read the next
candidate would cost 33 more compressions for it.

`sq(a)` is a² mod p, negated when a is odd: `PousModSquare_v1` (`pous.circuit`), then p − s when bit 0 of a is set.  For
a below p and odd, a² mod p is not 0 (p is prime), so p − s is below p; ``ok`` already rejects every input that could
make it p.

Costs at the registered point are pinned in `verity/core/protocols/pous/tests/test_pous_decode_circuit.py`: 31,960,530 ANDs per label, of which the
two `sq` are 28,147,478 (each 14,040,852 for the square and 32,887 for the negation), the 67 SHA-512 compressions
3,697,905 (one for the prefix, 33 per stream), σ_K 32,895 and the subtraction with its bounds 82,252.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce, const_bits
from verity.ml.circuits.boolean.trace import part
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.commitments.gates.sha512 import IV, pad_tail
from verity.ml.commitments.scheme import DIGEST_BYTES

from .answer_circuit import PousSha512Compress, compress, from_words
from .circuit import PousModSquare, _check, _traced_body, bits, column_sum

__all__ = ["PousByteSwap", "PousP2Decode", "PousP2Finish", "PousP2Negate", "PousP2Sigma", "PousP2Sq", "PousSha512Ctr",
           "PousSha512CtrChunk", "PousSha512Head", "PousSha512Mid", "P2_BASE_BYTES", "MAX_REST"]

P2_BASE_BYTES = 144
"""The expansion base's length at the registered point: `schemes.p2.expand_prefix` (one 128-byte block) and two u64be."""

MAX_REST = 128 - 1 - 4 - 17
"""The most bytes of the base past its whole blocks for which a chunk is one compression: the rest, D, the u32be
counter, then SHA-512's 0x80 and 16-byte length."""


# --- builders over forms ---------------------------------------------------------------------------------------------------

def _iv_bits():
    return [f for w in IV for f in const_bits(w, 64)]


def ctr_chunk(C, L: int, D: int, I: int, cv, rest):
    """Chunk I of the stream: one compression of the base's rest, D, u32be(I) and the padding, as its 64 bytes."""
    tail = bytes([D]) + I.to_bytes(4, "big") + pad_tail(L + 5)
    out = compress(C, cv, list(rest) + [f for byte in tail for f in const_bits(byte, 8)])
    return from_words([out[64 * i:64 * i + 64] for i in range(8)])


def byte_swap(x):
    n = len(x) // 8
    return [f for i in range(n) for f in x[8 * (n - 1 - i):8 * (n - i)]]


def less_than_p(C, x, C0: int):
    """x < 2^W − C0 for W-bit x: no carry out of x + C0."""
    carry = ZERO
    for i, f in enumerate(x):
        carry = (f ^ carry ^ C.AND(f, carry)) if C0 >> i & 1 else C.AND(f, carry)
    return carry.inv()


def negate(C, W: int, C0: int, s, odd):
    """odd ? (p − s mod 2^W) : s, with p − s = ¬s + p + 1."""
    p1 = (1 << W) - C0 + 1
    cols = [[f.inv()] + ([ONE] if p1 >> i & 1 else []) for i, f in enumerate(s)]
    n = column_sum(C, cols, W)
    return [a ^ C.AND(odd, a ^ b) for a, b in zip(s, n)]


def sigma(C, C0: int, r, k):
    """σ_K(r) = r ⊕ K when that is below p, else r."""
    lt = less_than_p(C, [a ^ b for a, b in zip(r, k)], C0)
    return [a ^ C.AND(lt, b) for a, b in zip(r, k)]


def finish(C, W: int, C0: int, u, t, c):
    """m = u − t mod p (u, t < p), as the low W − 16 bits' bytes, big-endian; then ok."""
    d = column_sum(C, [[a, b.inv()] + ([ONE] if i == 0 else []) for i, (a, b) in enumerate(zip(u, t))], W + 1)
    borrow = d[W].inv()
    p = (1 << W) - C0
    e = column_sum(C, [[f] + ([ONE] if p >> i & 1 else []) for i, f in enumerate(d[:W])], W)
    m = [a ^ C.AND(borrow, a ^ b) for a, b in zip(d[:W], e)]
    ok = and_reduce(C, [less_than_p(C, c, C0), less_than_p(C, t, C0)] + [f.inv() for f in m[W - 16:]])
    return byte_swap(m[:W - 16]) + [ok]


# --- the Definitions -------------------------------------------------------------------------------------------------------

PousSha512Head = part(
    "PousSha512Head", (), lambda S: ((("msg", bits(1024)),), bits(512)), lambda C, S, m: compress(C, _iv_bits(), m),
    "SHA-512's chaining value after one message block from the IV (FIPS 180-4)", conformance="protocol-defined")


def _mid_sig(S):
    _check("PousSha512Mid", M=S.M)
    return (("prefix", bits(1024 * S.M)),), bits(512)


def _mid_body(B, S, prefix):
    cv = B.call(bind(PousSha512Head), prefix[0:1024])
    for i in range(1, S.M):
        cv = B.call(bind(PousSha512Compress), cv, prefix[1024 * i:1024 * (i + 1)])
    return cv


PousSha512Mid = CompositeDefinition(
    "PousSha512Mid", 1, ("M",), _mid_sig, _mid_body, conformance="protocol-defined",
    doc="SHA-512's chaining value after M message blocks from the IV: a midstate, M compressions")


def _rest(name: str, L, D) -> int:
    _check(name, L=L)
    if not (L >= 128 and L % 128 <= MAX_REST and isinstance(D, int) and 0 <= D < 256):
        raise ValueError(f"{name} takes a base of whole 128-byte blocks and at most {MAX_REST} bytes more, and a byte D, "
                         f"not L = {L}, D = {D!r}")
    return L % 128


def _ctr_inputs(r: int):
    return (("cv", bits(512)),) + ((("rest", bits(8 * r)),) if r else ())


def _chunk_sig(S):
    r = _rest("PousSha512CtrChunk", S.L, S.D)
    if isinstance(S.I, bool) or not isinstance(S.I, int) or not 0 <= S.I < 1 << 32:
        raise ValueError(f"PousSha512CtrChunk's I is a u32 counter, not {S.I!r}")
    return _ctr_inputs(r), bits(512)


PousSha512CtrChunk = part(
    "PousSha512CtrChunk", ("L", "D", "I"), _chunk_sig,
    lambda C, S, cv, *rest: ctr_chunk(C, S.L, S.D, S.I, cv, rest[0] if rest else []),
    "SHA-512(base ‖ D ‖ u32be I) for an L-byte base, from the chaining value of its whole blocks and its rest: one "
    "compression", conformance="protocol-defined")


def _ctr_sig(S):
    r = _rest("PousSha512Ctr", S.L, S.D)
    _check("PousSha512Ctr", NB=S.NB)
    return _ctr_inputs(r), bits(8 * S.NB)


def _ctr_body(B, S, cv, *rest):
    n = DIGEST_BYTES["sha512"]
    out = [B.call(bind(PousSha512CtrChunk, L=S.L, D=S.D, I=i), cv, *rest).refs for i in range(-(-S.NB // n))]
    return Coll(bits(8 * n * len(out)), concat(out))[0:8 * S.NB]


PousSha512Ctr = CompositeDefinition(
    "PousSha512Ctr", 1, ("L", "D", "NB"), _ctr_sig, _ctr_body, conformance="protocol-defined",
    doc="The first NB bytes of SHA-512(base ‖ D ‖ u32be 0) ‖ SHA-512(base ‖ D ‖ u32be 1) ‖ … for an L-byte base, from "
        "the chaining value of its whole blocks and its rest: ⌈NB / 64⌉ compressions")


def _swap_sig(S):
    _check("PousByteSwap", N=S.N)
    return (("x", bits(8 * S.N)),), bits(8 * S.N)


PousByteSwap = CompositeDefinition(
    "PousByteSwap", 1, ("N",), _swap_sig, _traced_body(lambda S: ("swap", S.N), lambda S: lambda C, x: byte_swap(x), _swap_sig),
    conformance="exact-model-tested", doc="N bytes in reverse order: a big-endian integer's bytes to its bits, LSB first, "
                                          "and back (no gates)")


def _field_sig(name: str, S) -> int:
    _check(name, W=S.W, C0=S.C0)
    if S.W % 8 or S.W <= 16:
        raise ValueError(f"{name}'s W is a whole number of bytes above 16 bits, not {S.W}")
    bind(PousModSquare, W=S.W, C0=S.C0).signature()
    return S.W


def _negate_sig(S):
    W = _field_sig("PousP2Negate", S)
    return (("s", bits(W)), ("odd", bits(1))), bits(W)


PousP2Negate = CompositeDefinition(
    "PousP2Negate", 1, ("W", "C0"), _negate_sig,
    _traced_body(lambda S: ("negate", S.W, S.C0), lambda S: lambda C, s, o: negate(C, S.W, S.C0, s, o[0]), _negate_sig),
    conformance="exact-model-tested", doc="p − s mod 2^W when odd is set, else s, for p = 2^W − C0")


def _sq_sig(S):
    W = _field_sig("PousP2Sq", S)
    return (("x", bits(W)),), bits(W)


def _sq_body(B, S, x):
    s = B.call(bind(PousModSquare, W=S.W, C0=S.C0), x)
    return B.call(bind(PousP2Negate, W=S.W, C0=S.C0), s, x[0:1])


PousP2Sq = CompositeDefinition(
    "PousP2Sq", 1, ("W", "C0"), _sq_sig, _sq_body, conformance="exact-model-tested",
    doc="P2's sq(x): x² mod p, negated mod p when x is odd (`schemes.p2`), for x below p")


def _sigma_sig(S):
    W = _field_sig("PousP2Sigma", S)
    return (("r", bits(W)), ("k", bits(W))), bits(W)


PousP2Sigma = CompositeDefinition(
    "PousP2Sigma", 1, ("W", "C0"), _sigma_sig,
    _traced_body(lambda S: ("sigma", S.W, S.C0), lambda S: lambda C, r, k: sigma(C, S.C0, r, k), _sigma_sig),
    conformance="exact-model-tested", doc="P2's σ_K(r): r ⊕ K when below p, else r")


def _finish_sig(S):
    W = _field_sig("PousP2Finish", S)
    return (("u", bits(W)), ("t", bits(W)), ("c", bits(W))), bits(W - 15)


PousP2Finish = CompositeDefinition(
    "PousP2Finish", 1, ("W", "C0"), _finish_sig,
    _traced_body(lambda S: ("finish", S.W, S.C0), lambda S: lambda C, u, t, c: finish(C, S.W, S.C0, u, t, c), _finish_sig),
    conformance="exact-model-tested",
    doc="u − t mod p as (W − 16)/8 big-endian payload bytes, then ok: c < p, t < p and the result below 2^(W − 16)")


def _decode_sig(S):
    W = _field_sig("PousP2Decode", S)
    _rest("PousP2Decode", S.L, 0)
    return (("block", bits(W)), ("base", bits(8 * S.L))), bits(W - 15)


def _decode_body(B, S, block, base):
    W, C0, nb, M = S.W, S.C0, S.W // 8, S.L // 128
    swap = bind(PousByteSwap, N=nb)
    c = B.call(swap, block)
    cv = B.call(bind(PousSha512Mid, M=M), base[0:1024 * M])
    rest = (base[1024 * M:8 * S.L],) if S.L % 128 else ()
    k = B.call(swap, B.call(bind(PousSha512Ctr, L=S.L, D=ord("K"), NB=nb), cv, *rest))
    t = B.call(swap, B.call(bind(PousSha512Ctr, L=S.L, D=ord("t"), NB=nb), cv, *rest))
    sq = bind(PousP2Sq, W=W, C0=C0)
    u = B.call(sq, B.call(bind(PousP2Sigma, W=W, C0=C0), B.call(sq, c), k))
    return B.call(bind(PousP2Finish, W=W, C0=C0), u, t, c)


PousP2Decode = CompositeDefinition(
    "PousP2Decode", 2, ("W", "C0", "L"), _decode_sig, _decode_body, conformance="exact-model-tested",
    doc="P2 v4's decode of one block: sq(σ_K(sq(c))) − t mod p as its payload bytes, then ok, with K and t from "
        "counter-mode SHA-512 of the L-byte expansion base (`schemes.p2`); P2's point is W = 16448, C0 = 21065, L = 144")
