"""The falsification test of `Accounting.Assumptions.EncodeChainHardness` (Security/Models/Accounting/Assumptions/
EncodeChain.lean): every known construction of a squaring mod p1024 in the circuit model costs at least the conjectured
`e = n − 2` budgeted gates.

A squaring here is the step an exponentiation chain repeats: 128 s8 limbs in, 128 s8 limbs out, congruent to the square
mod p. Products are free (s8-family steps from `C = 0`, their operands proven bytes by their producers' declared
intervals), and so are integer affine maps with constant coefficients. Every other gate is budgeted, one unit each,
reading at most `f = 4` operands of at most 32 bits. An s8 operand comes straight from a budgeted gate declared in
`[−128, 127]`, since interval analysis proves no affine map of a wide value narrow. Each construction is run on seeded
inputs and checked against `pow(a, 2, p)`; a cheaper one, added here, fails the test and falsifies the conjecture at `e`.
It is also the test of the per-squaring statement `Accounting.Guarantees.Gates.Renormalization` at `d = 2`: its outputs
are limbs congruent to the square, which canonical bytes only make costlier.

The model's codec is held to this one: `codecRows` (Security/Proofs/Accounting/Gates.lean) are `codec.encode_word`'s
words at toy fields, which the Lean check evaluates `Accounting.Gates.encodeWord` on in the kernel
(`encodeWord_codecRows`), and `test_encode_word_agrees_with_lean` recomputes each here.
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
FAN_IN, WIDTH = 4, 32
E_PER_SQUARING = N_LIMBS - 2


class Ckt:
    """Counts a construction's budgeted gates and checks the model's window on each value it makes."""

    def __init__(self) -> None:
        self.budgeted = 0

    def gate(self, f, *operands: int, lo: int, hi: int) -> int:
        assert len(operands) <= FAN_IN, "a budgeted gate of fan-in above f"
        assert all(-(1 << WIDTH) <= v < 1 << WIDTH for v in operands), "an operand wider than w"
        assert -(1 << WIDTH) <= lo and hi < 1 << WIDTH
        self.budgeted += 1
        v = f(*operands)
        assert lo <= v <= hi, "a budgeted value outside its declared interval"
        return v


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


def folded_columns(d: list[int]) -> list[int]:
    """The square's column sums (s8-family coordinates), folded mod p by the affine map `2^1024 ≡ c`."""
    cols = []
    for m in range(2 * N_LIMBS):
        idx = range(max(0, m - N_LIMBS + 1), min(m, N_LIMBS - 1) + 1)
        cols.append(family([d[i] for i in idx], [d[m - i] for i in idx]) if len(idx) else 0)
    return [cols[m] + F.c * cols[m + N_LIMBS] for m in range(N_LIMBS)]


def offset(c: int, *g: int) -> int:
    """`U = c + Σ (g_j + 128)·256^j`, whose base-256 digits less 128 are the signed limbs and whose top is the carry."""
    return c + sum((v + 128) << (8 * j) for j, v in enumerate(g))


def ripple(ckt: Ckt, f: list[int]) -> tuple[list[int], int]:
    """A carry pass, two gates a limb: the limb and the carry, each its own budgeted gate."""
    out, c = [], 0
    for m in range(N_LIMBS):
        out.append(ckt.gate(lambda c, g: offset(c, g) % 256 - 128, c, f[m], lo=-128, hi=127))
        c = ckt.gate(lambda c, g: offset(c, g) >> 8, c, f[m], lo=-(1 << 24), hi=1 << 24)
    return out, c


def lookahead(ckt: Ckt, f: list[int]) -> tuple[list[int], int]:
    """A carry pass with (f − 1)-limb look-ahead: a gate for each limb and one for the carry past the group."""
    out, c = [], 0
    for m in range(0, N_LIMBS, FAN_IN - 1):
        g = f[m:m + FAN_IN - 1]
        for i in range(len(g)):
            out.append(ckt.gate(lambda c, *h, i=i: (offset(c, *h) >> (8 * i)) % 256 - 128, c, *g[:i + 1],
                                lo=-128, hi=127))
        c = ckt.gate(lambda c, *h: offset(c, *h) >> (8 * len(h)), c, *g, lo=-(1 << 24), hi=1 << 24)
    return out, c


def square(carry_pass, d: list[int]) -> tuple[list[int], int]:
    ckt = Ckt()
    out, c = carry_pass(ckt, folded_columns(d))
    out[0] += F.c * c
    out, c = carry_pass(ckt, out)
    assert c == 0
    return out, ckt.budgeted


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
        assert min(costs) >= E_PER_SQUARING, (carry_pass.__name__, min(costs))


def test_encode_chain_hardness_e_at_p1024_k3() -> None:
    squarings = ((P + 1) // 4).bit_length() - 1
    assert squarings == 1021
    assert codec.DEFAULT.encode_squarings == F.block_words * 3 * squarings
    assert 3 * squarings * 2 == 6_126 and F.block_words * 3 * squarings * 2 == 196_032
    assert 3 * squarings * E_PER_SQUARING == 385_938
    assert F.block_words * 3 * squarings * E_PER_SQUARING == 12_350_016


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
