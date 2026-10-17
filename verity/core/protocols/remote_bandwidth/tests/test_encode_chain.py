"""The falsification test of `Accounting.Assumptions.EncodeChainHardness` (Security/Models/Accounting/Assumptions/
EncodeChain.lean): every known construction of a squaring mod p1024 in the circuit model costs at least the conjectured
`e = L − d(ε)` budgeted units, `L = n/8` and `d(ε) = d₀ + log₂(1/ε)/8` with `d₀ = 4`.

A squaring here is the step an exponentiation chain repeats: 128 s8 limbs in, 128 s8 limbs out, congruent to the square
mod p. Products are free (s8-family steps from `C = 0`, their operands proven bytes by their producers' declared
intervals), and so are integer affine maps with constant coefficients. Every other gate is budgeted, one unit each,
reading at most `B = 128` operand bits, the declared bits of its operands (`OperandBits`). An s8 operand comes straight
from a budgeted gate declared in `[−128, 127]`, since interval analysis proves no affine map of a wide value narrow.
Each construction is run on seeded inputs and checked against `pow(a, 2, p)` on every input (`ε = 1`, so `e = L − d₀`);
a cheaper one, added here, fails the test and falsifies the conjecture at `e`. It is also the test of the per-squaring
statement `Accounting.Guarantees.Gates.Renormalization` at `c = 0`: its outputs are limbs congruent to the square.

The model's codec is held to this one: `codecRows` (Security/Proofs/Accounting/Gates.lean) are `codec.encode_word`'s
words at toy fields, which the Lean check evaluates `Accounting.Gates.encodeWord` on in the kernel
(`encodeWord_codecRows`), and `test_encode_word_agrees_with_lean` recomputes each here. `RhoPairs`, proved there for
every prime `p ≡ 3 mod 4`, is sampled on `codec.rho` at p1024.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from verity.core.protocols.remote_bandwidth import codec

LEAN_GATES = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "Gates.lean"
ROW = re.compile(r"\((\d+), (\d+), \[(\d+(?:, \d+)*)\], (\d+), (\d+), (\d+), (\d+), (\d+)\)")

F = codec.P1024
P = F.p
N_LIMBS = F.word_bytes
OPERAND_BITS = 128
D0 = 4
LOOKAHEAD = 3


def e_per_squaring(log2_inv_eps: int) -> int:
    """`L − d(ε)`, with `ε`'s share `log₂(1/ε)/8` limbs rounded up."""
    return N_LIMBS - D0 - -(-log2_inv_eps // 8)


def bits(lo: int, hi: int) -> int:
    """A declared interval's bits, `⌈log₂ size⌉` (`Gate.bits`)."""
    return (hi - lo).bit_length()


Val = tuple[int, int, int]


class Ckt:
    """Counts a construction's budgeted gates and checks the model's window on each value it makes. A value is
    `(v, lo, hi)`: its value and its declared (or, for a free gate, derived) interval."""

    def __init__(self) -> None:
        self.budgeted = 0

    def gate(self, f, *operands: Val, lo: int, hi: int) -> Val:
        assert all(o[1] <= o[0] <= o[2] for o in operands)
        assert sum(bits(o[1], o[2]) for o in operands) <= OPERAND_BITS, "a budgeted gate reading more than B bits"
        self.budgeted += 1
        v = f(*(o[0] for o in operands))
        assert lo <= v <= hi, "a budgeted value outside its declared interval"
        return v, lo, hi


def family(a: list[int], b: list[int]) -> int:
    """An s8-family coordinate chained over k blocks of 32 from `C = 0`: exact, and checked to stay in s32."""
    assert all(-128 <= v <= 127 for v in a + b), "an s8-family operand outside s8"
    total = 0
    for k in range(0, len(a), 32):
        total += sum(x * y for x, y in zip(a[k:k + 32], b[k:k + 32]))
        assert -(1 << 31) <= total < 1 << 31, "an s8-family accumulator outside s32"
    return total


def value(limbs: list[int]) -> int:
    return sum(d << (8 * m) for m, d in enumerate(limbs))


def folded_columns(d: list[int]) -> list[Val]:
    """The square's column sums (s8-family coordinates), folded mod p by the affine map `2^1024 ≡ c`, with the
    intervals interval analysis derives for them from s8 operands."""
    cols = []
    for m in range(2 * N_LIMBS):
        idx = range(max(0, m - N_LIMBS + 1), min(m, N_LIMBS - 1) + 1)
        v = family([d[i] for i in idx], [d[m - i] for i in idx]) if len(idx) else 0
        cols.append((v, -128 * 127 * len(idx), 128 * 128 * len(idx)))
    return [tuple(a + F.c * b for a, b in zip(cols[m], cols[m + N_LIMBS])) for m in range(N_LIMBS)]


def offset(c: int, *g: int) -> int:
    """`U = c + Σ (g_j + 128)·256^j`, whose base-256 digits less 128 are the signed limbs and whose top is the carry."""
    return c + sum((v + 128) << (8 * j) for j, v in enumerate(g))


def ripple(ckt: Ckt, f: list[Val]) -> tuple[list[Val], Val]:
    """A carry pass, two gates a limb: the limb and the carry, each its own budgeted gate (the red team's IMMA plus
    `I32Rem` and `I32Div`)."""
    out, c = [], (0, 0, 0)
    for m in range(N_LIMBS):
        out.append(ckt.gate(lambda c, g: offset(c, g) % 256 - 128, c, f[m], lo=-128, hi=127))
        c = ckt.gate(lambda c, g: offset(c, g) >> 8, c, f[m], lo=-(1 << 24), hi=1 << 24)
    return out, c


def lookahead(ckt: Ckt, f: list[Val]) -> tuple[list[Val], Val]:
    """A carry pass with 3-limb look-ahead: a gate for each limb and one for the carry past the group."""
    out, c = [], (0, 0, 0)
    for m in range(0, N_LIMBS, LOOKAHEAD):
        g = f[m:m + LOOKAHEAD]
        for i in range(len(g)):
            out.append(ckt.gate(lambda c, *h, i=i: (offset(c, *h) >> (8 * i)) % 256 - 128, c, *g[:i + 1],
                                lo=-128, hi=127))
        c = ckt.gate(lambda c, *h: offset(c, *h) >> (8 * len(h)), c, *g, lo=-(1 << 24), hi=1 << 24)
    return out, c


def square(carry_pass, d: list[int]) -> tuple[list[int], int]:
    ckt = Ckt()
    out, c = carry_pass(ckt, folded_columns(d))
    out[0] = tuple(a + F.c * b for a, b in zip(out[0], c))
    out, c = carry_pass(ckt, out)
    assert c[0] == 0
    return [v for v, _, _ in out], ckt.budgeted


def signed_limbs(a: int) -> list[int]:
    out = []
    for _ in range(N_LIMBS):
        out.append((a + 128) % 256 - 128)
        a = (a - out[-1]) >> 8
    assert a == 0
    return out


def test_encode_chain_hardness_known_constructions() -> None:
    rng = random.Random(20261017)
    for carry_pass in (ripple, lookahead):
        costs = set()
        for _ in range(32):
            a = rng.randrange(P // 4)
            out, cost = square(carry_pass, signed_limbs(a))
            assert all(-128 <= v <= 127 for v in out)
            assert value(out) % P == pow(a, 2, P), carry_pass.__name__
            costs.add(cost)
        assert min(costs) >= e_per_squaring(0), (carry_pass.__name__, min(costs))


def test_encode_chain_hardness_e_at_p1024_k3() -> None:
    squarings = ((P + 1) // 4).bit_length() - 1
    assert squarings == 1021
    assert codec.DEFAULT.encode_squarings == F.block_words * 3 * squarings
    assert e_per_squaring(0) == 124 and e_per_squaring(20) == 121
    assert 3 * squarings * e_per_squaring(20) == 370_623
    assert F.block_words * 3 * squarings * e_per_squaring(20) == 11_859_936


def test_rho_pairs_at_p1024() -> None:
    """`RhoPairs` on `codec.rho`: `ρ(p − x) = p − ρ(x)`, and `ρ(x)` is odd exactly at a non-square (Euler)."""
    rng = random.Random(1024)
    for _ in range(64):
        x = rng.randrange(1, P)
        r = codec.rho(x, F)
        assert codec.rho(P - x, F) == P - r
        assert r % 2 == (pow(x, (P - 1) // 2, P) != 1)


def lean_codec_rows() -> list[tuple[int, int, list[int], int, int, int, int, int]]:
    block = LEAN_GATES.read_text().split("def codecRows", 1)[1]
    lines = block[:block.index(")]") + 1].splitlines()[1:]
    found = [ROW.search(line) for line in lines]
    assert found and all(found), "a row of codecRows this test doesn't read"
    return [(int(n), int(c), [int(v) for v in cs.split(", ")], int(mask), int(tau), int(i), int(m), int(y))
            for n, c, cs, mask, tau, i, m, y in (r.groups() for r in found)]


def toy_key(n: int, c: int, cs: list[int], mask: int, tau: int) -> codec.Key:
    """A masked key at the toy field `p = 2^n − c`, which `codec.Codec` refuses (it takes p1024 and p512 only)."""
    toy = object.__new__(codec.Codec)
    for name, v in (("field", codec.Field(f"toy{n}", n, c)), ("rounds", len(cs)), ("masked", True)):
        object.__setattr__(toy, name, v)
    return codec.Key(toy, tuple(cs), tau, mask)


def test_encode_word_agrees_with_lean() -> None:
    rows = lean_codec_rows()
    arms = set()
    for n, c, cs, mask, tau, i, m, y in rows:
        k = toy_key(n, c, cs, mask, tau)
        assert codec.encode_word(m, i, k) == y, (n, c, cs, mask, tau, i, m)
        p, x = k.codec.field.p, m
        for r, cr in enumerate(cs):
            if r:
                arms.add("σ xor" if x ^ mask < p else "σ keep")
                x = k.sigma(x)
            a = (x + cr + tau * i) % p
            arms.add("ρ zero" if a == 0 else "ρ square" if pow(a, (p + 1) // 4, p) ** 2 % p == a else "ρ non-square")
            x = codec.rho(a, k.codec.field)
    assert arms == {"σ xor", "σ keep", "ρ zero", "ρ square", "ρ non-square"}, arms
