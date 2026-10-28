"""The N_L search's restriction of the gate model, its tables, its circuits and its p1024 families compute what `renorm.py`
claims."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gates as G  # noqa: E402
import renorm as R  # noqa: E402


def test_mma_clamps_or_wraps_outside_s32_and_proven_refuses_a_wide_operand():
    b = G.Builder()
    big, zero = b.const(2 ** 20), b.const(0)
    sat = b.gate(G.op_mma(G.S8, G.S8, 1, sat=True), (zero, big, big), G.S32_MIN, G.S32_MAX)
    wrap = b.gate(G.op_mma(G.S8, G.S8, 1, sat=False), (zero, big, big), G.S32_MIN, G.S32_MAX)
    vs, window, _hold = b.evaluate({})
    assert vs[sat] == G.S32_MAX and vs[wrap] == (2 ** 40 - G.S32_MIN) % 2 ** 32 + G.S32_MIN
    assert not window  # the operand 2^20 leaves s8, and the exact sum leaves the declared interval
    assert any("outside" in m for m in G.proven(b))


def test_derive_is_the_interval_image_and_a_budgeted_gate_keeps_its_declaration():
    b = G.Builder()
    x, y, zero = b.inp(0, 0, 15), b.inp(1, -8, 7), b.const(0)
    m = b.gate(G.op_mma((0, 15), (-8, 7), 1), (zero, x, y), 0, 0)
    f = b.gate(G.op_affine(3, (2, -1)), (x, y), 0, 0)
    g = b.budgeted(lambda v: v[0] // 2, [x], 0, 7)
    ivs = G.intervals(b)
    assert ivs[m] == (-120, 105) and ivs[f] == (3 + 0 - 7, 3 + 30 + 8) and ivs[g] == (0, 7)


def test_fan_in_and_width_are_restricted_s():
    b = G.Builder()
    a = b.inp(0, 0, 255)
    b.budgeted(lambda v: v[0], [a] * 5, 0, 255)
    assert not R.fan_in(4, b) and R.fan_in(5, b)
    assert R.width(8, b) and not R.width(7, b)


@pytest.mark.parametrize("B,L", [(4, 4), (16, 3), (256, 2)])
def test_every_square_is_its_column_sums_plus_the_carry_lattice(B, L):
    F = R.Field.toy(B, L)
    assert F.p % 4 == 3 and R.is_prime(F.p) and not any(R.is_prime(q) and q % 4 == 3 for q in range(F.p + 1, B ** L))
    T = R.table(F)
    pw = np.array([B ** l for l in range(L)], dtype=np.int64)
    a = np.arange(F.p, dtype=np.int64)
    assert (T.y @ pw == a * a % F.p).all() and (T.digits @ pw == a).all()
    rebuilt = T.s.copy()
    rebuilt[:, 0] += F.delta * T.c[:, L - 1]
    rebuilt[:, 1:] += T.c[:, :L - 1]
    rebuilt -= B * T.c
    assert (rebuilt == T.y).all()


@pytest.mark.parametrize("B,L", [(4, 5), (16, 3), (256, 2)])
def test_the_column_sum_circuit_needs_no_budgeted_gate_on_the_carry_free_set(B, L):
    F = R.Field.toy(B, L)
    T = R.table(F)
    _m, names = R.monomials(T.digits[:1])
    eye = [[int(i == j) for j in range(L)] for i in range(L)]
    cn = R.Constraint(eye, [[0] * len(names)] * L, [0] * L, 0, "carry-free")
    wit = R.renormalization_witness(T, R.build(T, cn, names))
    assert wit["restricted"] and wit["run_bits"] == 0 and wit["budgeted"] == 0 and wit["window_on_good"]
    assert wit["good"] == int((T.c == 0).all(axis=1).sum()) == R.carry_free_count(F)


def test_the_honest_circuit_is_exact_everywhere_with_L_budgeted_gates():
    F = R.Field.toy(16, 3)
    T = R.table(F)
    _m, names = R.monomials(T.digits[:1])
    wit = R.renormalization_witness(T, R.build(T, None, names))
    assert wit["restricted"] and wit["good"] == F.p and wit["budgeted"] == F.L


def test_each_constraint_found_is_a_restricted_circuit_with_L_minus_k_budgeted_gates_exact_on_its_fiber():
    r = R.run_field(4, 4)  # run_field asserts the witness for every k it reports
    ks = [row["k"] for row in r["frontier"]]
    assert ks == list(range(1, 5))
    counts = [row["count"] for row in r["frontier"]]
    assert counts == sorted(counts, reverse=True) and counts[-1] >= r["carry_free"]


def test_integer_kernel_and_unimodular_completion():
    rnd = random.Random(3)
    for _ in range(40):
        L = rnd.randint(2, 6)
        k = rnd.randint(1, L)
        rows = [[rnd.randint(-3, 3) for _ in range(L)] for _ in range(k)]
        ker = R.integer_kernel(rows, L)
        assert all(sum(r[i] * v[i] for i in range(L)) == 0 for r in rows for v in ker)
        rank = int(np.linalg.matrix_rank(np.array(rows, dtype=float)))
        assert len(ker) == L - rank
        W = R.unimodular_completion(rows, L)
        if W is not None:
            assert W[:k] == rows and abs(R._det(W)) == 1
    assert R.unimodular_completion([[2, 0]], 2) is None and R.unimodular_completion([[2, 3]], 2) is not None


def test_p1024_small_digit_families_need_no_budgeted_gate():
    fam = R.p1024_family([2] * 63 + [1] + [0] * 64)
    assert fam["exact"] and fam["max_column"] == 252 and abs(fam["log2_size"] - 100.85) < 0.01
    assert fam["implied_d0"] > 12.6 and fam["formula_d_at_eps"] < 128
    assert not R.p1024_family([3] * 64 + [0] * 64)["exact"]  # its all-max member carries
    greedy = R.p1024_greedy_family()
    assert R.p1024_family(greedy)["exact"] and R.p1024_family(greedy)["log2_size"] > fam["log2_size"]
    rnd = random.Random(5)
    for _ in range(25):
        digits = [rnd.randint(0, m) for m in [2] * 63 + [1] + [0] * 64]
        a = sum(x << (8 * l) for l, x in enumerate(digits))
        assert list((a * a % R.P1024).to_bytes(128, "little")) == R.p1024_column_sums(digits)
    assert R.is_prime(R.P1024) and R.P1024 % 4 == 3


def test_a_27_byte_field_has_an_exact_family_with_no_budgeted_gate_past_the_formula():
    """The smallest byte field the greedy family refutes d = 4 + log₂(1/ε)/8 at: L = 27, p = 256^27 − 377."""
    rows = R.greedy_series(range(1, 28))
    assert all(r["exact"] for r in rows)
    assert [r["L"] for r in rows if r["implied_d0"] > 4] == [27]
    F = R.Field.toy(256, 27)
    D = R.p1024_greedy_family(27, F.delta)
    row = rows[-1]
    assert F.delta == 377 and row["formula_d_at_eps"] < 27 and abs(row["log2_size"] - 33.30) < 0.01
    rnd = random.Random(27)
    for _ in range(25):
        digits = [rnd.randint(0, m) for m in D]
        a = sum(x << (8 * l) for l, x in enumerate(digits))
        assert list((a * a % F.p).to_bytes(27, "little")) == R.p1024_column_sums(digits, 27, F.delta)


def test_p1024_carries_rebuild_the_square():
    c = R.p1024_carries(8, seed=2)
    rnd = random.Random(2)
    A = [rnd.randrange(R.P1024) for _ in range(8)]
    for i, a in enumerate(A):
        s = R.p1024_column_sums(list(a.to_bytes(128, "little")))
        y = [s[l] + (105 * c[i, 127] if l == 0 else c[i, l - 1]) - 256 * c[i, l] for l in range(128)]
        assert y == list((a * a % R.P1024).to_bytes(128, "little"))
