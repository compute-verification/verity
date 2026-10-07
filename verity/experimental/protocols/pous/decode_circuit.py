"""P2's decode of one block as a Boolean circuit: ``m = sq(σ_K(sq(c))) − t mod p``, with `schemes.p2`'s v3 keys (K and
t from SHAKE256 of the block's expansion base), so that PoUS's setup can prove a block of C decodes to its row of W.

    PousP2Decode_v1{W, C0, L}   (block: Array<W>, base: Array<8L>) -> Array<W − 15>    payload ‖ ok
    PousP2Sq_v1{W, C0}          (x: Array<W>) -> Array<W>                               sq(x)
    PousP2Negate_v1{W, C0}      (s: Array<W>, odd: Array<1>) -> Array<W>                odd ? p − s : s
    PousP2Sigma_v1{W, C0}       (r: Array<W>, k: Array<W>) -> Array<W>                  σ_K(r)
    PousP2Finish_v1{W, C0}      (u: Array<W>, t: Array<W>, c: Array<W>) -> Array<W − 15>
    PousShake256_v1{L, D, NB}   (msg: Array<8L>) -> Array<8 NB>                         SHAKE256(msg ‖ D), NB bytes
    PousShakePad_v1{L, D}       (msg: Array<8L>) -> Array<1600>                         the padded first block
    PousKeccakF1600_v1          (state: Array<1600>) -> Array<1600>                     Keccak-f[1600]
    PousKeccakRound_v1{R}       (state: Array<1600>) -> Array<1600>                     its round R
    PousByteSwap_v1{N}          (x: Array<8N>) -> Array<8N>                             N bytes reversed (no gates)

Every array is of `Value<1>`.  A byte string is bits in byte order, bit t being bit t mod 8 of byte ⌊t/8⌋ (hm96's order);
an integer is bits least significant first.  A big-endian integer of N bytes and its byte string differ by
`PousByteSwap_v1{N}`, which is wiring.  Keccak's state is its 200 bytes in that order (lane x + 5y little-endian).

**The decode** (`schemes.p2.decode_segment` for one block):

- ``block`` is the block's W/8 bytes, so c = int.from_bytes(block, "big"); ``base`` is `schemes.p2`'s v3 expansion base
  ``expand_prefix(params, salt) ‖ u64be(s) ‖ u64be(j)``, L bytes (121 at the registered point);
- K is the first W/8 bytes of SHAKE256(base ‖ "K"), and t the first W/8 bytes of SHAKE256(base ‖ "t"), both big-endian;
- the result is m = sq(σ_K(sq(c))) − t mod p as its (W − 16)/8 payload bytes, then ``ok``: c < p, t < p, and m < 2^(W−16).

When ``ok`` is 1 the payload is the reference's, byte for byte.  `schemes.p2` rejects a first t candidate ≥ p and reads
the next; the circuit reads only the first, and says so with ``ok`` = 0.  That happens with probability C0 / 2^W per
block (2^-16433.6 at the registered point), so an honest prover meets it never, and a circuit that read the next
candidate would cost 16 more permutations for it.

`sq(a)` is a² mod p, negated when a is odd: `PousModSquare_v1` (`pous.circuit`), then p − s when bit 0 of a is set.  For
a below p and odd, a² mod p is not 0 (p is prime), so p − s is below p; ``ok`` already rejects every input that could
make it p.

Costs at the registered point are pinned in `tests/test_pous_decode_circuit.py`: 29,491,425 ANDs per block, of which the
two `sq` are 28,147,478 (each 14,040,852 for the square and 32,887 for the negation), the 32 Keccak-f[1600] permutations
1,228,800, σ_K 32,895 and the subtraction with its bounds 82,252.
"""
from __future__ import annotations

from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat

from .circuit import PousModSquare, _check, _traced_body, bits, column_sum

__all__ = ["KECCAK_RC", "KECCAK_ROT", "PousByteSwap", "PousKeccakF1600", "PousKeccakRound", "PousP2Decode", "PousP2Finish",
           "PousP2Negate", "PousP2Sigma", "PousP2Sq", "PousShake256", "PousShakePad", "P2_BASE_BYTES", "RATE"]

P2_BASE_BYTES = 121
"""The expansion base's length at the registered point: `schemes.p2.expand_prefix` (105 bytes) and two u64be."""

RATE = 136
"""SHAKE256's rate in bytes."""

KECCAK_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
KECCAK_ROT = ((0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61), (28, 55, 25, 21, 56), (27, 20, 39, 8, 14))
"""ρ's offsets, ``KECCAK_ROT[x][y]`` (FIPS 202 §3.2.2)."""


# --- builders over forms ---------------------------------------------------------------------------------------------------

def keccak_round(C, R: int, s):
    """Round R of Keccak-f[1600] (θ, ρ, π, χ, ι) on the 1600-bit state, lane x + 5y at bits 64 (x + 5y) .. + 63."""
    a = [[s[64 * (x + 5 * y):64 * (x + 5 * y) + 64] for y in range(5)] for x in range(5)]
    col = [[a[x][0][z] ^ a[x][1][z] ^ a[x][2][z] ^ a[x][3][z] ^ a[x][4][z] for z in range(64)] for x in range(5)]
    d = [[col[(x - 1) % 5][z] ^ col[(x + 1) % 5][(z - 1) % 64] for z in range(64)] for x in range(5)]
    a = [[[a[x][y][z] ^ d[x][z] for z in range(64)] for y in range(5)] for x in range(5)]
    b = [[None] * 5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            r = KECCAK_ROT[x][y]
            b[y][(2 * x + 3 * y) % 5] = [a[x][y][(z - r) % 64] for z in range(64)]
    out = [[[b[x][y][z] ^ C.AND(b[(x + 1) % 5][y][z].inv(), b[(x + 2) % 5][y][z]) for z in range(64)] for y in range(5)]
           for x in range(5)]
    out[0][0] = [f.inv() if KECCAK_RC[R] >> z & 1 else f for z, f in enumerate(out[0][0])]
    return [f for y in range(5) for x in range(5) for f in out[x][y]]


def shake_pad(L: int, D: int, msg):
    """SHAKE256's first block of ``msg ‖ D`` (the state before the first permutation), when L + 1 < RATE."""
    bytes_ = [list(msg[8 * i:8 * i + 8]) for i in range(L)] + [[ONE if D >> k & 1 else ZERO for k in range(8)]]
    bytes_ += [[ZERO] * 8 for _ in range(200 - L - 1)]
    for k in range(8):
        if 0x1F >> k & 1:
            bytes_[L + 1][k] = bytes_[L + 1][k].inv()
    bytes_[RATE - 1][7] = bytes_[RATE - 1][7].inv()
    return [f for byte in bytes_ for f in byte]


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

def _round_sig(S):
    if not (isinstance(S.R, int) and 0 <= S.R < 24):
        raise ValueError(f"PousKeccakRound's R is a round index in 0..23, not {S.R!r}")
    return (("state", bits(1600)),), bits(1600)


PousKeccakRound = CompositeDefinition(
    "PousKeccakRound", 1, ("R",), _round_sig,
    _traced_body(lambda S: ("keccak", S.R), lambda S: lambda C, s: keccak_round(C, S.R, s), _round_sig),
    conformance="protocol-defined", doc="Round R of Keccak-f[1600] (FIPS 202), the state's 200 bytes in byte order")


def _keccak_body(B, S, s):
    for r in range(24):
        s = B.call(bind(PousKeccakRound, R=r), s)
    return s


PousKeccakF1600 = CompositeDefinition(
    "PousKeccakF1600", 1, (), lambda S: ((("state", bits(1600)),), bits(1600)), _keccak_body,
    conformance="protocol-defined", doc="The Keccak-f[1600] permutation (FIPS 202), the state's 200 bytes in byte order")


def _pad_sig(S):
    _check("PousShakePad", L=S.L)
    if not (S.L + 1 < RATE and isinstance(S.D, int) and 0 <= S.D < 256):
        raise ValueError(f"PousShakePad takes a message of L + 1 < {RATE} bytes and a byte D, not L = {S.L}, D = {S.D!r}")
    return (("msg", bits(8 * S.L)),), bits(1600)


PousShakePad = CompositeDefinition(
    "PousShakePad", 1, ("L", "D"), _pad_sig,
    _traced_body(lambda S: ("shake-pad", S.L, S.D), lambda S: lambda C, m: shake_pad(S.L, S.D, m), _pad_sig),
    conformance="protocol-defined", doc="SHAKE256's padded block of msg ‖ D, one rate block (no gates)")


def _shake_sig(S):
    _pad_sig(S)
    _check("PousShake256", NB=S.NB)
    return (("msg", bits(8 * S.L)),), bits(8 * S.NB)


def _shake_body(B, S, msg):
    st = B.call(bind(PousKeccakF1600), B.call(bind(PousShakePad, L=S.L, D=S.D), msg))
    out = [st[0:8 * RATE].refs]
    for _ in range(1, -(-S.NB // RATE)):
        st = B.call(bind(PousKeccakF1600), st)
        out.append(st[0:8 * RATE].refs)
    return Coll(bits(8 * RATE * len(out)), concat(out))[0:8 * S.NB]


PousShake256 = CompositeDefinition(
    "PousShake256", 1, ("L", "D", "NB"), _shake_sig, _shake_body, conformance="protocol-defined",
    doc="The first NB bytes of SHAKE256(msg ‖ D) (FIPS 202) for an L-byte msg, L + 1 < 136: ⌈NB / 136⌉ permutations")


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
    _check("PousP2Decode", L=S.L)
    if not S.L + 1 < RATE:
        raise ValueError(f"PousP2Decode's base and its purpose byte fit one SHAKE256 block: L + 1 < {RATE}, not L = {S.L}")
    return (("block", bits(W)), ("base", bits(8 * S.L))), bits(W - 15)


def _decode_body(B, S, block, base):
    W, C0, nb = S.W, S.C0, S.W // 8
    swap = bind(PousByteSwap, N=nb)
    c = B.call(swap, block)
    k = B.call(swap, B.call(bind(PousShake256, L=S.L, D=ord("K"), NB=nb), base))
    t = B.call(swap, B.call(bind(PousShake256, L=S.L, D=ord("t"), NB=nb), base))
    sq = bind(PousP2Sq, W=W, C0=C0)
    u = B.call(sq, B.call(bind(PousP2Sigma, W=W, C0=C0), B.call(sq, c), k))
    return B.call(bind(PousP2Finish, W=W, C0=C0), u, t, c)


PousP2Decode = CompositeDefinition(
    "PousP2Decode", 1, ("W", "C0", "L"), _decode_sig, _decode_body, conformance="exact-model-tested",
    doc="P2 v3's decode of one block: sq(σ_K(sq(c))) − t mod p as its payload bytes, then ok, with K and t from SHAKE256 of "
        "the L-byte expansion base (`schemes.p2`); P2's point is W = 16448, C0 = 21065, L = 121")
