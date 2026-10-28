"""The bit-level codec (`bitwise_codec.py`): a block of codec words for 8 units a key byte under Model.lean's budgeted
count, right on every draw tried, `Proven`, with no budgeted gate and no run input: the counterexample to
`EncodeChainHardness` at every `E ≥ 1`. In nonlinear units (`nonlinear`: the budgeted units and the data products) each
of its ANDs on two run-time bits is a data product and costs a unit, and its words cost what `SquaringFloor` and
`ChainSum` price."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import bitwise_codec as BC  # noqa: E402
import free_codec as FC  # noqa: E402
import gates as G  # noqa: E402
import squaring as S  # noqa: E402

# Security/Proofs/Accounting/Gates.lean's `codecRows`: (n, c, cs, K, τ, i, m, encodeWord).
CODEC_ROWS = [
    (8, 5, [17, 200, 99], 248, 3, 5, 219, 234), (8, 5, [92, 7], 253, 18, 54, 195, 147),
    (8, 5, [44, 209], 252, 132, 41, 181, 166), (8, 5, [139, 0], 254, 232, 36, 68, 225),
    (8, 5, [56, 35], 12, 69, 52, 101, 18), (8, 5, [88, 65], 93, 168, 12, 225, 4),
    (8, 5, [103, 10, 98], 252, 50, 51, 47, 10), (8, 5, [156, 202], 253, 47, 41, 11, 126),
    (16, 17, [26622, 53477], 34513, 22959, 218370, 28017, 16283),
    (16, 17, [27420, 1802, 17095], 22236, 6729, 817333, 53283, 48655),
    (16, 17, [60688, 23696, 15300], 349, 10049, 806969, 33473, 64012),
]


def targets(n: int, y: int) -> list[int]:
    return [(y // 2 * 2) >> (8 * l) & 255 for l in range(n // 8)]


def check_block(b, outs, x: dict, want: dict[int, list[int]]) -> None:
    vals, window, hold = b.evaluate(x)
    assert window and hold
    for i, gs in outs.items():
        assert [vals[g] for g in gs] == want[i], i


def check_costs(b, n: int, rounds: int) -> None:
    """No budgeted gate, no run input, `Restricted` at every `B`, and `8·(k + 2)·n/8` units, by the builder's running
    count and by `units` itself (only gathers can have units, and every gather's units are scanned literally)."""
    assert "budgeted" not in b.kinds
    assert "run" not in b.bindings.values()
    assert G.proven(b) == [] and G.operand_bits(0, b)
    assert all(b.lo[i] >= 0 and b.hi[i] <= 1 for i, k in enumerate(b.kinds) if k in ("mma", "gather"))
    want = 8 * (rounds + 2) * (n // 8)
    assert b.fixed_units == b.lean_units == want
    dyn = G.dynamic(b)
    gathers = [i for i, k in enumerate(b.kinds) if k == "gather"]
    assert sum(G.units(b, i, fixed=True, dyn=dyn) for i in gathers) == want


@pytest.mark.parametrize("rounds", [pytest.param(2, marks=pytest.mark.slow), 3])
def test_codec_rows_at_n8(rounds):
    rows = [r for r in CODEC_ROWS if r[0] == 8 and len(r[2]) == rounds]
    words = sorted({r[5] for r in rows})
    b, outs, _ = BC.encode_block(8, 5, rounds, words)
    check_costs(b, 8, rounds)
    for n, c, cs, K, tau, i, m, y in rows:
        p = (1 << n) - c
        assert FC.encode_word_ref(p, cs, K, tau * i % p, m) == y
        x = BC.draw_inputs(n, c, cs, K, tau, {j: m for j in words})
        vals, window, hold = b.evaluate(x)
        assert window and hold
        assert [vals[g] for g in outs[i]] == targets(n, y)


WORDS8 = [1, 2, 3, 7, 100, 200, 249, 250]


@pytest.fixture(scope="module")
def block8():
    """The eight-word block at ⟨8, 5⟩, three rounds (553k gates, about 3 s to build and 4 s more for its Program), shared by
    the tests that read it."""
    return BC.encode_block(8, 5, 3, WORDS8)


@pytest.mark.slow
def test_units_do_not_grow_with_the_block(block8):
    """One word or eight: the circuit reads the key once, and word `i`'s tweak is `i·t_1 mod p`, so the units stay
    `8·(k + 2)·n/8` = 40 at n = 8, k = 3, and the eight words are right on every draw tried."""
    n, c, k = 8, 5, 3
    p = (1 << n) - c
    words = WORDS8
    for b, outs, slots in (BC.encode_block(n, c, k, [1]), block8):
        check_costs(b, n, k)
        assert {s for s in slots if s[0] != "payload"} == (
            {("const", r, 0) for r in range(k)} | {("mask", 0), ("tweak", 1, 0)})
    rng = random.Random(26)
    for _ in range(2):
        cs, K, tau = [rng.randrange(p) for _ in range(k)], rng.randrange(1 << n), rng.randrange(1, p)
        payload = {i: rng.randrange(1 << n) for i in words}
        want = {i: targets(n, FC.encode_word_ref(p, cs, K, tau * i % p, payload[i])) for i in words}
        check_block(b, outs, BC.draw_inputs(n, c, cs, K, tau, payload), want)


def test_the_conjecture_fails_at_p1024():
    """`EncodeChainHardness p1024 3 B E ε`'s conclusion for this circuit, `q·E·(n − 1) ≤ budgeted·(n − 1) + 0·E`, with
    `budgeted = 8·(3 + 2)·128`: false at the headline `E` for one word, and at `E = 1` for 5,121 words (all below p)."""
    n, k, L = 1024, 3, 128
    units = 8 * (k + 2) * L
    assert units == 5120
    for E, q in ((364_497, 1), (24_504, 1), (1, units + 1)):
        assert not q * E * (n - 1) <= units * (n - 1)


def test_nonlinear_units_charge_each_and(block8):
    """Each AND (a `k = 1` s8-family step) on two run-time bits is a data product, by the builder's running count and by
    `products` itself, so the block's nonlinear units grow with its words, no longer `8·(k + 2)·n/8` for any number."""
    n, c, k = 8, 5, 3
    one, _, _ = BC.encode_block(n, c, k, [1])
    dyn = G.dynamic(one)
    ands = [i for i, kind in enumerate(one.kinds) if kind == "mma"]
    assert all(one.prims[i].k == 1 and one.lo[i] >= 0 and one.hi[i] <= 1 for i in ands)
    assert one.products == G.products(one) == sum(1 for i in ands if dyn[one.args[i][1]] and dyn[one.args[i][2]]) > 0
    assert one.nonlinear_units == G.nonlinear(one) == 8 * (k + 2) * (n // 8) + one.products
    eight, _, _ = block8
    assert eight.fixed_units == one.fixed_units and eight.products >= 8 * one.products


def test_bitwise_squaring_pays_its_partial_products():
    """A bitwise squaring (`Field.mul` of a word by itself, double-and-add) pays at least `n²` data products at three
    toy fields: its `n(n − 1)/2` distinct partial-product ANDs and its adders' carries. At p1024 the partial products
    alone are 523,776 a squaring, far above `SquaringFloor`'s `e = 374`, so the counterexample to `EncodeChainHardness`
    pays more than `ChainSum`'s `E = 3 · 1021 · 374` a word."""
    for n, c in ((8, 5), (16, 17), (32, 5)):
        b = G.Builder()
        B = BC.Bits(b)
        F = BC.Field(B, n, c)
        xs = [bit for l in range(n // 8) for bit in B.byte_bits(b.inp(("a", l), 0, 255, binding="coin"))]
        before = b.products
        F.mul(xs, xs)
        assert b.products - before >= n * n, (n, b.products - before)
    partial, e = 1024 * 1023 // 2, S.e_per_squaring(40)
    assert e == 374 and partial == 523_776
    assert 3 * S.SQUARINGS * partial > 3 * S.SQUARINGS * e == 1_145_562


@pytest.mark.slow
def test_codec_at_n16():
    """The 16-bit rows and random draws at `⟨16, 17⟩`, three rounds, two words: the gadgets at a second field."""
    n, c, k = 16, 17, 3
    p = (1 << n) - c
    rows = [r for r in CODEC_ROWS if r[0] == n and len(r[2]) == k]
    words = sorted({r[5] for r in rows})
    b, outs, _ = BC.encode_block(n, c, k, words)
    check_costs(b, n, k)
    for _, _, cs, K, tau, i, m, y in rows:
        assert FC.encode_word_ref(p, cs, K, tau * i % p, m) == y
        vals, window, hold = b.evaluate(BC.draw_inputs(n, c, cs, K, tau, {j: m for j in words}))
        assert window and hold
        assert [vals[g] for g in outs[i]] == targets(n, y)
    rng = random.Random(16)
    for _ in range(2):
        cs, K, tau = [rng.randrange(p) for _ in range(k)], rng.randrange(1 << n), rng.randrange(1, p)
        payload = {i: rng.randrange(1 << n) for i in words}
        want = {i: targets(n, FC.encode_word_ref(p, cs, K, tau * i % p, payload[i])) for i in words}
        check_block(b, outs, BC.draw_inputs(n, c, cs, K, tau, payload), want)
