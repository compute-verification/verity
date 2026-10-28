"""The falsification tests of `Accounting.Assumptions.SquaringFloor` and `ChainSum` (Security/Definitions/Accounting/
Assumptions/SquaringChain.lean), which replace the falsified `EncodeChainHardness` (EncodeChain.lean, refuted by
`Accounting.SecurityProofs.Gates.EncodeChainRefuted`): every known construction of a squaring mod p1024 in the circuit
model costs at least `e = (2L − 1) + (L − d(ε))` nonlinear units, `L = n/8` and `d(ε) = d₀ + log₂(1/ε)/8` with
`d₀ = 4`, at every `ε ≥ 2^−64` (`test_squaring_floor_known_constructions`), and chained squarings cost their sum
(`test_chain_sum_known_chains`).

A squaring here is the step an exponentiation chain repeats: 128 s8 limbs in, 128 s8 limbs out, congruent to the square
mod p. The only free operations are linear maps whose coefficients are fixed before the coins: integer affine maps with
constant coefficients, and s8-family lanes with a constant or registered multiplicand. A lane whose two multiplicands
both depend on run or coin data is a data product, one unit whatever their range; so is every lane of the square's
column sums (s8-family steps from `C = 0`, their operands proven bytes by their producers' declared intervals), `L²` of
them. Every other gate is budgeted, one unit each, reading at most `B = 128` operand bits, the declared bits of its
operands (`OperandBits`). The nonlinear units are the budgeted units and the data products. Each construction is run on
seeded inputs and checked against `pow(a, 2, p)` on every input (`ε = 1`, so `e = 3L − 1 − d₀`); a cheaper one, added
here, fails the test and falsifies `SquaringFloor` at `e`. Its budgeted units are also the test of the per-squaring
statement `Accounting.Guarantees.Gates.Renormalization` at `c = 0` (`L − d(ε)`): its outputs are limbs congruent to the
square. The bitwise codec, which refuted `EncodeChainHardness`, pays a data product an AND
(`benchmarks/property_redteam/tests/test_bitwise_codec.py`).

`Renormalization` has a floor on `ε` because circuits with no budgeted gate refute it below one: on the `a` whose
squares' column sums don't carry, the column sums (s8-family steps, folded mod `p`) are the square's bytes. The red
team's families and a sampled one at p1024 are kept here as
regressions the stated form must survive: each refutes the form with no floor, and sits below the floor `2^−64`.

The model's codec is held to this one: `codecRows` (Security/Proofs/Accounting/Gates.lean) are `codec.encode_word`'s
words at toy fields, which the Lean check evaluates `Accounting.Gates.encodeWord` on in the kernel
(`encodeWord_codecRows`), and `test_encode_word_agrees_with_lean` recomputes each here. `RhoPairs`, proved there for
every prime `p ≡ 3 mod 4`, is sampled on `codec.rho` at p1024.
"""

from __future__ import annotations

import itertools
import math
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
FLOOR = 64
HEADLINE = 40
LOOKAHEAD = 3
SQUARINGS = ((P + 1) // 4).bit_length() - 1


def renorm(log2_inv_eps: int) -> int:
    """Lemma R's budgeted units, `L − d(ε)`, with `ε`'s share `log₂(1/ε)/8` limbs rounded up."""
    return N_LIMBS - D0 - -(-log2_inv_eps // 8)


def e_per_squaring(log2_inv_eps: int) -> int:
    """`SquaringFloor`'s price in nonlinear units: a data product for each of the square's `2L − 1` column sums, and
    Lemma R's `L − d(ε)`."""
    return 2 * N_LIMBS - 1 + renorm(log2_inv_eps)


def bits(lo: int, hi: int) -> int:
    """A declared interval's bits, `⌈log₂ size⌉` (`IntGate.bits`)."""
    return (hi - lo).bit_length()


Val = tuple[int, int, int]


class Ckt:
    """Counts a construction's budgeted gates and data products, and checks the model's window on each value it makes.
    A value is `(v, lo, hi)`: its value and its declared (or, for a free gate, derived) interval."""

    def __init__(self) -> None:
        self.budgeted = 0
        self.products = 0

    @property
    def nonlinear(self) -> int:
        return self.budgeted + self.products

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


def folded_columns(ckt: Ckt, d: list[int], run_time: bool = True) -> list[Val]:
    """The square's column sums (s8-family coordinates over the limbs `d`, run-time or registered), folded mod p by the
    affine map `2^1024 ≡ c`, with the intervals interval analysis derives for them from s8 operands. A lane on two
    run-time limbs is a data product."""
    cols = []
    for m in range(2 * N_LIMBS):
        idx = range(max(0, m - N_LIMBS + 1), min(m, N_LIMBS - 1) + 1)
        ckt.products += len(idx) if run_time else 0
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
    """One squaring of the s8 limbs `d` (a coin's, or the last squaring's budgeted outputs): its limbs and nonlinear
    units."""
    ckt = Ckt()
    out, c = carry_pass(ckt, folded_columns(ckt, d))
    out[0] = tuple(a + F.c * b for a, b in zip(out[0], c))
    out, c = carry_pass(ckt, out)
    assert c[0] == 0
    assert all((lo, hi) == (-128, 127) for _, lo, hi in out) and ckt.products == N_LIMBS ** 2
    assert ckt.budgeted >= renorm(0)
    return [v for v, _, _ in out], ckt.nonlinear


def signed_limbs(a: int) -> list[int]:
    out = []
    for _ in range(N_LIMBS):
        out.append((a + 128) % 256 - 128)
        a = (a - out[-1]) >> 8
    assert a == 0
    return out


def test_squaring_floor_known_constructions() -> None:
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


def test_squaring_floor_data_products() -> None:
    """The data-product rule on the column sums: run-time limbs make every lane of the square a data product, `L²` of
    them whatever the limbs' range, and registered ones (a linear map by fixed coefficients) leave every lane free."""
    for run_time, want in ((True, N_LIMBS ** 2), (False, 0)):
        ckt = Ckt()
        folded_columns(ckt, [0] * N_LIMBS, run_time)
        assert (ckt.products, ckt.budgeted) == (want, 0), run_time
    assert N_LIMBS ** 2 > 2 * N_LIMBS - 1


def test_chain_sum_known_chains() -> None:
    """The byte-limb chain: each squaring's limbs are the next one's operands, and the chain costs the sum of its
    squarings, none shared, right on every input tried. At `k = 3` a word's `3 · 1021` squarings then cost at least
    `E = 3 · 1021 · e` at the headline `ε = 2^−40` and at `2^−20`."""
    rng = random.Random(20261026)
    chain = 64
    for carry_pass in (ripple, lookahead):
        for _ in range(2):
            a = rng.randrange(P // 4)
            d, costs = signed_limbs(a), []
            for _ in range(chain):
                d, cost = square(carry_pass, d)
                costs.append(cost)
            assert value(d) % P == pow(a, 2 ** chain, P), carry_pass.__name__
            assert len(set(costs)) == 1 and costs[0] >= e_per_squaring(0), (carry_pass.__name__, costs[0])
            per_word = codec.DEFAULT.rounds * SQUARINGS * costs[0]
            for log2_inv_eps in (HEADLINE, 20):
                assert per_word >= codec.DEFAULT.rounds * SQUARINGS * e_per_squaring(log2_inv_eps)


def test_chain_sum_e_at_p1024_k3() -> None:
    squarings = ((P + 1) // 4).bit_length() - 1
    assert squarings == 1021
    assert codec.DEFAULT.encode_squarings == F.block_words * 3 * squarings
    assert renorm(0) == 124 and renorm(20) == 121 and renorm(FLOOR) == 116 and renorm(HEADLINE) == 119
    assert e_per_squaring(0) == 379 and e_per_squaring(20) == 376 and e_per_squaring(FLOOR) == 371
    assert e_per_squaring(HEADLINE) == 374
    assert 3 * squarings * e_per_squaring(HEADLINE) == 1_145_562
    assert F.block_words * 3 * squarings * e_per_squaring(HEADLINE) == 36_657_984
    assert 3 * squarings * e_per_squaring(20) == 1_151_688


def folded_sums(d: list[int], p: int) -> list[int]:
    """The outputs of the circuit with no budgeted gate on `a = Σ d_l 256^l`: `a²`'s column sums, the top ones folded
    into the bottom by `256^L ≡ 256^L − p`."""
    L = len(d)
    t = [0] * (2 * L - 1)
    nz = [(i, v) for i, v in enumerate(d) if v]
    for i, u in nz:
        for j, v in nz:
            t[i + j] += u * v
    return [t[m] + ((256 ** L - p) * t[m + L] if m + L < 2 * L - 1 else 0) for m in range(L)]


def renormalization(L: int, eps_num: int, eps_den: int, budgeted: int, floor: int | None) -> bool:
    """`Renormalization`'s conclusion at `c = 0`, `d₀ = 4` and `ε = εnum/εden`, true where its floor `2^−floor`
    excludes `ε` (`floor=None`: the form with no floor)."""
    if floor is not None and eps_den > 2 ** floor * eps_num:
        return True
    return eps_num * 2 ** (8 * L) <= eps_den * 2 ** (8 * (budgeted + D0))


def test_renormalization_survives_non_carrying_families() -> None:
    """The red team's product families `{a : a_l ≤ D_l}`, from its greedy search: the all-max member's folded column
    sums bound every member's, so if they are bytes and their value is below `p`, the circuit with no budgeted gate
    outputs every member's canonical bytes (and is proven: its derived intervals are those sums)."""
    rng = random.Random(27)
    for p, top, log2_size in ((256 ** 27 - 377, [5] * 3 + [4] * 11 + [0] * 13, 33.30),
                              (P, [2] * 63 + [1] * 2 + [0] * 63, 101.85)):
        L = len(top)
        s = folded_sums(top, p)
        assert max(s) <= 255 and value(s) < p
        members = math.prod(v + 1 for v in top)
        assert round(math.log2(members), 2) == log2_size
        for _ in range(16):
            d = [rng.randint(0, v) for v in top]
            a = value(d)
            assert folded_sums(d, p) == list((a * a % p).to_bytes(L, "little"))
        assert not renormalization(L, members - 1, p, 0, None)
        assert renormalization(L, members - 1, p, 0, FLOOR)


def non_carrying(d: list[int]) -> bool:
    """Whether `a = Σ d_l 256^l` (digits below 2^8, at most 64 of them) squares with no carry: packed 16 bits apart,
    `a²`'s column sums don't overlap, and each must be a byte."""
    A = 0
    for i, v in enumerate(d):
        A |= v << (16 * i)
    S = A * A
    while S:
        if S & 0xFFFF > 255:
            return False
        S >>= 16
    return True


def test_slope_seven_refuted_below_the_floor() -> None:
    """A larger family at p1024, sampled: 64 digits, each from a geometric law truncated at 12 (means 1.75 to 2.75,
    higher at the edges, whose columns sum fewer products), kept where the square doesn't carry. Every member of
    `T = {d : −log₂ q(d) ≥ t, non-carrying}` has `q(d) ≤ 2^−t`, so `|T| ≥ q(T) · 2^t`, with `q(T)` bounded below by
    the hits less three standard deviations: `|T| ≥ 2^159.4`. It refutes the slope `1/7` (`d = 4 + log₂(1/ε)/7` allows a
    circuit with no budgeted gate at most `2^156` inputs) and every fixed `d₀` below 19.9 at slope `1/8`, and it sits
    far below the floor: its digits are at most 12, so `|T| ≤ 13^64`."""
    K, positions, draws = 12, 64, 10_000
    laws = []
    for i in range(positions):
        mu = 1.75 + math.exp(-min(i, positions - 1 - i) / 5)
        w = [(mu / (1 + mu)) ** k for k in range(K + 1)]
        laws.append([v / sum(w) for v in w])
    t = sum(-sum(v * math.log2(v) for v in q) for q in laws) - 4
    cum = [list(itertools.accumulate(q)) for q in laws]
    rng = random.Random(7)
    hits = []
    for _ in range(draws):
        d = [rng.choices(range(K + 1), cum_weights=c)[0] for c in cum]
        if -sum(math.log2(q[v]) for q, v in zip(laws, d)) >= t and non_carrying(d):
            hits.append(d)
    for d in hits[:8]:
        a = value(d)
        assert folded_sums(d + [0] * (N_LIMBS - positions), P) == list((a * a % P).to_bytes(N_LIMBS, "little"))
    log2_members = t + math.log2((len(hits) - 3 * math.sqrt(len(hits))) / draws)
    assert log2_members > 159.4
    assert 7 * (N_LIMBS - D0) > math.log2(P) - log2_members
    assert N_LIMBS - (math.log2(P) - log2_members) / 8 > 19.9
    assert P > 2 ** FLOOR * 13 ** positions


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
