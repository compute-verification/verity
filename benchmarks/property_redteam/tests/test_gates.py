"""The gate model on the canonical circuit (`gates.py`), pinned: each circuit's declarations, its values on fixed inputs as the
canonical evaluator computes them, its window, `DeclaredHold` and units, digested. A change to the model's semantics or to
a study's construction moves a digest."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import free_codec as FC  # noqa: E402
import gates as G  # noqa: E402
import renorm as R  # noqa: E402
import squaring as S  # noqa: E402
from verity.ml.circuits.evaluation.reference import evaluate  # noqa: E402

PINS = {
    "gathers": "eb1a6b31264359d8",
    "free_bit": "fdfd188ae23f4415",
    "mma_clamp_wrap": "5469f0a054cde1b5",
    "seam_square": "82c5e69ef36b37bd",
    "holeb_square": "cf439df9e78d3630",
    "free_codec_8_5_2": "f052d110c0e66734",
    "renorm_4_3_honest": "34935e4e87c119f4",
    "renorm_4_3_k1": "83da21cc7d91a614",
}


def _digest(rows, runs, extra) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(r).encode())
    for vals, window, hold in runs:
        h.update(repr(([int(v) for v in vals], bool(window), bool(hold))).encode())
    h.update(repr(extra).encode())
    return h.hexdigest()[:16]


def _rows(b: G.Builder):
    return [(b.kinds[i], tuple(b.args[i]), b.lo[i], b.hi[i]) for i in range(len(b))]


def _pin(b: G.Builder, xs) -> str:
    return _digest(_rows(b), [b.evaluate(x) for x in xs], (b.lean_units, b.fixed_units, b.products, G.proven(b) == []))


def _gathers():
    b = G.Builder()
    i = b.inp(0, -1000, 1000)
    col = [b.const(c) for c in (5, -3, 9)]
    b.gather(i, col)
    b.gather(i, col, low=2)
    b.gather(i, [b.const(7)] * 3, low=2)
    b.gather(i, [b.const(1)] * 3, low=3)
    return b, [{0: v} for v in (-7, -1, 0, 1, 2, 3, 4, 6, 1000)]


def _free_bit():
    b = G.Builder()
    S.free_bit(b, b.inp(0, 0, 1000, "run"), 5)
    return b, [{0: v} for v in (0, 31, 32, 1000)]


def _mma_clamp_wrap():
    b = G.Builder()
    x, zero = b.inp(0, -2 ** 20, 2 ** 20), b.const(0)
    b.gate(G.op_mma(G.S8, G.S8, 1, sat=True), (zero, x, x), G.S32_MIN, G.S32_MAX)
    b.gate(G.op_mma(G.S8, G.S8, 1, sat=False), (zero, x, x), G.S32_MIN, G.S32_MAX)
    b.budgeted(lambda v: v[0] // 3, [x], -350_000, 350_000)
    return b, [{0: v} for v in (2 ** 20, -(2 ** 20), 127, -128, 3)]


def _square(name):
    def build():
        b = G.Builder()
        S.CONSTRUCTIONS[name](b, [b.inp(i, 0, 255) for i in range(S.N_LIMBS)])
        return b, [dict(enumerate(S.limbs(v))) for v in (0x0123456789ABCDEF ** 16 % S.P, S.P - 1)]
    return build


def _free_codec():
    b, _out, _slots = FC.free_encode_word(8, 5, 2)
    return b, [FC.draw_inputs(8, 5, [92, 7], 253, 18, 54, 195)]


CIRCUITS = {"gathers": _gathers, "free_bit": _free_bit, "mma_clamp_wrap": _mma_clamp_wrap,
            "seam_square": _square("seam_square"), "holeb_square": _square("holeb_square"),
            "free_codec_8_5_2": _free_codec}


@pytest.mark.parametrize("name", sorted(CIRCUITS))
def test_the_circuits_values_window_and_units_are_pinned(name):
    b, xs = CIRCUITS[name]()
    assert _pin(b, xs) == PINS[name]


@pytest.mark.parametrize("tag", ["honest", "k1"])
def test_renorms_circuits_are_pinned_on_every_input(tag):
    """`build` at ⟨4, 3⟩ (p = 59): the honest circuit, and the one-constraint circuit `c_2 − c_0 + a_0 + 3·a_1·a_2 = 1`
    (the search's best row), on all 59 inputs."""
    F = R.Field.toy(4, 3)
    T = R.table(F)
    _m, names = R.monomials(T.digits[:1])
    cn = None if tag == "honest" else R.Constraint([[-1, 0, 1]], [[-1, 0, 0, 0, 0, 0, 0, -3, 0]], [1], 32, "row+q-lll")
    bc = R.build(T, cn, names)
    runs = [bc.C.evaluate({l: int(T.digits[a, l]) for l in range(F.L)}) for a in range(F.p)]
    w = R.renormalization_witness(T, bc)
    assert (w["budgeted"], w["good"]) == ((3, 59) if cn is None else (2, 32))
    extra = (w["restricted"], w["budgeted"], w["good"], w["window_on_good"])
    assert _digest(_rows(bc.C), runs, extra) == PINS[f"renorm_4_3_{tag}"]


def test_each_gate_is_its_primitive_on_its_operands():
    """The Builder's Program is the canonical circuit: its trace at every gate is that gate's primitive, through
    `reference.evaluate`, on its operands' encodings, and the inputs are the Program's first gates."""
    b, xs = _mma_clamp_wrap()
    prog, index = b.program()
    assert prog.gates == len(b) and index[0] == 0
    for x in xs:
        vals = b.values(x)
        for i in range(len(b)):
            if b.kinds[i] != "input":
                assert G.signed(evaluate(b.prims[i], *[G.enc(vals[j]) for j in b.args[i]])[0]) == vals[i]
