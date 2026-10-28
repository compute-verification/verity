"""The studies' old gate-model ports against the canonical circuit (`gates.py`): each old circuit, translated gate for gate onto
a `gates.Builder`, has every gate's value, its window and `DeclaredHold` equal on each input tried, and the same declaration
analyses."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import bitwise_codec as BC  # noqa: E402
import free_codec as FC  # noqa: E402
import gates as G  # noqa: E402
import renorm as R  # noqa: E402
import squaring as S  # noqa: E402

COUNTS: dict[str, int] = {}


def _from_squaring(old: S.Builder) -> G.Builder:
    b = G.Builder()
    for g in old.gates:
        op = g.op
        if op.kind == "input":
            b.inp(op.slot, g.lo, g.hi, op.binding)
        elif op.kind == "const":
            b.gate(G.op_const(op.c), g.args, g.lo, g.hi)
        elif op.kind == "affine":
            b.gate(G.op_affine(op.c, op.coeff), g.args, g.lo, g.hi)
        elif op.kind == "mma":
            b.gate(G.op_mma(op.fa, op.fb, op.k, op.sat, len(g.args)), g.args, g.lo, g.hi)
        elif op.kind == "gather":
            b.gate(G.op_gather(len(g.args), op.low), g.args, g.lo, g.hi)
        else:
            b.gate(G.op_budgeted(op.f, len(g.args), op.cls), g.args, g.lo, g.hi)
    return b


def _from_renorm(C: list) -> G.Builder:
    b = G.Builder()
    for g in C:
        op = g.op
        if op.kind == "input":
            b.inp(op.slot, g.lo, g.hi, op.binding)
        elif op.kind == "const":
            b.gate(G.op_const(op.c), g.args, g.lo, g.hi)
        elif op.kind == "affine":
            b.gate(G.op_affine(op.c, op.coeff), g.args, g.lo, g.hi)
        elif op.kind == "mma":
            b.gate(G.op_mma(R.fmt_range(op.fa), R.fmt_range(op.fb), op.k, op.sat, len(g.args)), g.args, g.lo, g.hi)
        elif op.kind == "gather":
            b.gate(G.op_gather(len(g.args)), g.args, g.lo, g.hi)
        else:
            f = op.f
            b.gate(G.op_budgeted(lambda v, f=f: int(f([np.array([x]) for x in v])[0]), len(g.args), op.cls), g.args, g.lo,
                   g.hi)
    return b


def _same_squaring(name: str, old: S.Builder, inputs: list[dict]) -> None:
    new = _from_squaring(old)
    prog, index = new.program()
    circ = prog.circuit
    assert all(circ[index[i]].prim is new.prims[i] for i in range(len(new)))
    assert [(g.lo, g.hi) for g in old.gates] == list(zip(new.lo, new.hi))
    assert old.ivs == new.ivs and old.dyn == new.dyn
    assert (old.lean_units, old.fixed_units, old.products) == (new.lean_units, new.fixed_units, new.products)
    assert S.proven(old.gates) == G.proven(new) and S.operand_bits(128, old.gates) == G.operand_bits(128, new)
    if len(old.gates) < 3000:
        assert S.dynamic(old.gates) == G.dynamic(new)
        assert S.budgeted(old.gates) == G.budgeted(new) and S.budgeted(old.gates, fixed=True) == G.budgeted(new, fixed=True)
        assert S.products(old.gates) == G.products(new) and S.nonlinear(old.gates) == G.nonlinear(new)
    for x in inputs:
        assert old.evaluate(x) == new.evaluate(x), name
    COUNTS[name] = COUNTS.get(name, 0) + len(inputs)


def test_squaring_constructions_agree():
    rng = random.Random(20261028)
    xs = [rng.randrange(1 << 1024) for _ in range(6)] + [S.P - 1, 0, 1 << 1023]
    xs.append(next(a for a in (rng.randrange(1 << 1024) for _ in range(1 << 15)) if S.seam_fails(a)))
    for name, fn in S.CONSTRUCTIONS.items():
        old = S.Builder()
        a = [old.inp(i, 0, 255) for i in range(S.N_LIMBS)]
        out, _ = fn(old, a)
        _same_squaring(name, old, [dict(enumerate(S.limbs(x))) for x in xs])


def test_squaring_unit_circuits_agree():
    old = S.Builder()
    r = old.inp(0, 0, 1000)
    S.free_bit(old, r, 5)
    _same_squaring("free_bit", old, [{0: v} for v in (0, 31, 32, 1000, -5, 5000)])
    old = S.Builder()
    i = old.inp(0, -1000, 1000)
    col = [old.const(c) for c in (5, -3, 9)]
    old.gather(i, col)
    old.gather(i, col, low=2)
    old.gather(i, [old.const(7)] * 3, low=2)
    old.gather(i, [old.const(1)] * 3, low=3)
    _same_squaring("gathers", old, [{0: v} for v in (-7, -1, 0, 1, 2, 3, 4, 6, 1000, -1000)])
    old = S.Builder()
    a = [old.inp(i, 0, 255) for i in range(S.N_LIMBS)]
    once, _ = S.holeb_square(old, a)
    old._append(S.Op("mma", fa=S.U8, fb=S.U8, k=1), (old.const(0), once[0], once[0]), S.S32_MIN, S.S32_MAX)
    old._append(S.Op("mma", fa=S.S8, fb=S.S8, k=1, sat=False), (old.const(2 ** 20), once[0], old.const(2 ** 20)),
                S.S32_MIN, S.S32_MAX)
    rng = random.Random(4)
    _same_squaring("holeb_square+mma", old, [dict(enumerate(S.limbs(rng.randrange(1 << 1024)))) for _ in range(4)])


def test_free_codec_agrees():
    rng = random.Random(3)
    for n, c, k in ((8, 5, 2), (8, 5, 3), (16, 17, 3)):
        p = (1 << n) - c
        old, _out, _slots = FC.free_encode_word(n, c, k)
        xs = []
        for _ in range(3):
            cs = [rng.randrange(p) for _ in range(k)]
            K, tau, i, m = rng.randrange(1 << n), rng.randrange(1, p), rng.randrange(1 << 20), rng.randrange(1 << n)
            xs.append(FC.draw_inputs(n, c, cs, K, tau, i, m))
        _same_squaring(f"free_encode_word{(n, c, k)}", old, xs)


def test_bitwise_codec_agrees():
    rng = random.Random(26)
    for n, c, k, words in ((8, 5, 2, [1, 5]), (8, 5, 3, [1, 2, 3, 7, 100, 200, 249, 250]), (16, 17, 3, [1])):
        p = (1 << n) - c
        old, _outs, _slots = BC.encode_block(n, c, k, words)
        xs = []
        for _ in range(2):
            cs, K, tau = [rng.randrange(p) for _ in range(k)], rng.randrange(1 << n), rng.randrange(1, p)
            xs.append(BC.draw_inputs(n, c, cs, K, tau, {i: rng.randrange(1 << n) for i in words}))
        _same_squaring(f"encode_block{(n, c, k, len(words))}", old, xs)


@pytest.mark.parametrize("B,L", [(4, 3), (4, 4), (4, 5), (16, 2), (16, 3), (256, 1), (256, 2)])
def test_renorm_circuits_agree(B, L):
    F = R.Field.toy(B, L)
    T = R.table(F)
    best = R.search(T)
    _m, names = R.monomials(T.digits[:1])
    eye = [[int(i == j) for j in range(L)] for i in range(L)]
    cns = [None, R.Constraint(eye, [[0] * len(names)] * L, [0] * L, 0, "carry-free"), *best.values()]
    x = {l: T.digits[:, l] for l in range(L)}
    for cn in cns:
        bc = R.build(T, cn, names)
        vs, window, hold = R.evaluate(bc.C, x, F.p)
        new = _from_renorm(bc.C)
        assert list(zip(new.lo, new.hi)) == [(g.lo, g.hi) for g in bc.C]
        assert (R.proven(bc.C) == []) == (G.proven(new) == [])
        rows = range(F.p) if F.p <= 5000 else sorted(set(random.Random(B * L).sample(range(F.p), 3000)) | {0, F.p - 1})
        for r in rows:
            got, w, h = new.evaluate({l: int(T.digits[r, l]) for l in range(L)})
            assert got == [int(v[r]) for v in vs] and w == bool(window[r]) and h == bool(hold[r]), (B, L, cn, r)
        COUNTS[f"renorm{(B, L)}"] = COUNTS.get(f"renorm{(B, L)}", 0) + len(rows)


def test_renorm_unit_gates_agree():
    big = R.Gate(R.Op("const", c=2 ** 20), (), 2 ** 20, 2 ** 20)
    zero = R.Gate(R.Op("const", c=0), (), 0, 0)
    sat = R.Gate(R.Op("mma", k=1, sat=True), (1, 0, 0), R.S32_MIN, R.S32_MAX)
    wrap = R.Gate(R.Op("mma", k=1, sat=False), (1, 0, 0), R.S32_MIN, R.S32_MAX)
    C = [big, zero, sat, wrap]
    vs, window, hold = R.evaluate(C, {}, 1)
    got = _from_renorm(C).evaluate({})
    assert got == ([int(v[0]) for v in vs], bool(window[0]), bool(hold[0]))
    a = R.Gate(R.Op("input", slot=0), (), 0, 15)
    b = R.Gate(R.Op("input", slot=1), (), -8, 7)
    m = R.Gate(R.Op("mma", fa=(False, 4), fb=(True, 4), k=1), (2, 0, 1), 0, 0)
    f = R.Gate(R.Op("affine", c=3, coeff=(2, -1)), (0, 1), 0, 0)
    g = R.Gate(R.Op("budgeted", f=lambda v: v[0] // 2), (0,), 0, 7)
    C = [a, b, zero, m, f, g]
    new = _from_renorm(C)
    assert R.intervals(C) == G.intervals(new)
    for u in range(-2, 18):
        for w in range(-9, 9):
            vs, window, hold = R.evaluate(C, {0: np.array([u]), 1: np.array([w])}, 1)
            assert new.evaluate({0: u, 1: w}) == ([int(v[0]) for v in vs], bool(window[0]), bool(hold[0]))


def test_report_counts():
    print({k: v for k, v in sorted(COUNTS.items())})
