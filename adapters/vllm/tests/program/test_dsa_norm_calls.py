"""`boolean_dsa_norm`'s units written as Calls (`DsaRMSNormSumSq_v3`, `DsaRMSNormApply_v3`, `DsaRMSNorm_v3`): each has its traced
`_v2`'s word view and signature and is that `_v2` and the word bit for bit; the scale and the apply unit have the `_v2`'s gates,
count for count; the pieces are the word body's steps; at GLM-5.3's served rows the scale is the word, a thread's later registers
and second vector included; at the served statics a unit's description does not grow with its row, it fits a Flock block, and
its block is not over its word view's; and the pieces lower by their builders to the circuit their gates lower to."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.program import Program
from verity.definitions import boolean as SW
from verity.definitions.ops.boolean import cast as BC
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.registry import boolean_dsa_norm as BN
from verity.definitions.registry import dsa_norm as DN
from tests.program.test_boolean_fp8 import _flat, _inputs

#: GLM-5.3's rms_norm_eps, and its two rows: q_c [2048] (program 2) and kv_c [512] (program 1)
EPS, ROWS = 1e-5, (2048, 512)
#: each unit written as Calls and its traced version, at the smallest binding: a row of two registers a thread, 8 elements
UNITS = {"scale": (BN.DsaRMSNormSumSqV3, BN.DsaRMSNormSumSqV2, dict(N=256, EPS=EPS)),
         "apply": (BN.DsaRMSNormApplyV3, BN.DsaRMSNormApplyV2, dict(G=8)),
         "row": (BN.DsaRMSNormV3, BN.DsaRMSNormV2, dict(N=256, EPS=EPS))}
#: the units at GLM-5.3's served statics (the apply unit at the 64 columns its partitions name), each beside its smallest binding
#: that holds every piece: a row of 512, whose threads take a register past their second, and one output element
SERVED = [(BN.DsaRMSNormSumSqV3, dict(N=n, EPS=EPS), dict(N=512, EPS=EPS)) for n in ROWS]
SERVED += [(BN.DsaRMSNormApplyV3, dict(G=64), dict(G=1))]
SERVED += [(BN.DsaRMSNormV3, dict(N=n, EPS=EPS), dict(N=512, EPS=EPS)) for n in ROWS]
PIECES = (BN.DsaRMSNormSquares, BN.DsaRMSNormSquareFma, BN.DsaRMSNormMeanRsqrt, BN.DsaRMSNormScale, BN.DsaRMSNormOut)


def _differ(unit, got, want) -> None:
    for g, r in zip(_flat(got), _flat(want), strict=True):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{unit.id}: {bad.size} of {len(r)} instances differ, first {bad[:5]}"


@pytest.mark.slow
@pytest.mark.parametrize("name", list(UNITS))
def test_a_unit_as_calls_is_the_traced_unit(name: str) -> None:
    """Each `_v3` has its `_v2`'s word view and its function, special words included, and `boolean_version` still gives the
    `_v2`; the scale and the apply unit have the `_v2`'s gates, count for count (the row computes an element's own gates for
    its square and again for its output).  The row traces 1.6M ANDs (about a minute): only this holds a unit written as Calls
    to the traced one bit for bit."""
    d_calls, d_traced, S = UNITS[name]
    calls, traced = bind(d_calls, **S), bind(d_traced, **S)
    word = TR.word_of(calls)
    assert word.id == TR.word_of(traced).id and SW.boolean_version(word).id == traced.id
    assert calls.signature() == traced.signature()
    inputs = _inputs(word, 96, sum(map(ord, calls.id)))
    want = E.evaluate_batch(word, *inputs)
    for unit in (calls, traced):
        _differ(unit, BS.evaluate_words(unit, *inputs), want)
    if name != "row":
        assert calls.gates == traced.gates
    else:
        assert calls.gates > traced.gates


def test_the_pieces_are_the_word_bodys_steps() -> None:
    """`DsaRMSNormSquares` and `DsaRMSNormSquareFma` are the chain's steps over the word primitives, and `DsaRMSNormOut` over
    the scale's own gates (`DsaRMSNormScale`) is one element of `DsaRMSNormApply_v1`, bit for bit on random words and special
    ones; the apply unit's two pieces have the gates of its traced element, count for count."""
    fma, mul, cvt = (TR.word_of(d) for d in (EW.F32Fma, EW.F32Mul, BC.Bf16ToF32))
    one = bind(DN.DsaRMSNormApply, G=1)
    x, w, r = _inputs(one, 2000, 215)
    x, w = x.reshape(-1), w.reshape(-1)
    f = np.asarray(E.evaluate_batch(cvt, x)).reshape(-1)
    g = np.asarray(E.evaluate_batch(cvt, w)).reshape(-1)
    squares = E.evaluate_batch(fma, f, f, np.asarray(E.evaluate_batch(mul, g, g)).reshape(-1))
    _differ(BN.DsaRMSNormSquares, BS.evaluate_words(bind(BN.DsaRMSNormSquares), x, w), squares)
    _differ(BN.DsaRMSNormSquareFma, BS.evaluate_words(bind(BN.DsaRMSNormSquareFma), x, r), E.evaluate_batch(fma, f, f, r))
    rd = BS.evaluate_words(bind(BN.DsaRMSNormScale), r)
    _differ(BN.DsaRMSNormOut, BS.evaluate_words(bind(BN.DsaRMSNormOut), x, w, r, rd),
            E.evaluate_batch(one, x[:, None], w[:, None], r))
    parts = [BS.compile(bind(d)).counts for d in (BN.DsaRMSNormScale, BN.DsaRMSNormOut)]
    whole = BS.compile(bind(BN.DsaRMSNormApplyV2, G=1)).counts
    assert {k: parts[0][k] + parts[1][k] for k in whole} == whole


@pytest.mark.slow
@pytest.mark.parametrize("N", ROWS)
def test_the_scale_as_calls_is_the_word_at_the_served_rows(N: int) -> None:
    """At GLM-5.3's rows the scale written as Calls returns `DsaRMSNormSumSq_v1`'s word.  What the smallest row can't show: a
    thread's registers past its second (the rows of `DsaRMSNormSquareFma`), and at 2048 its second vector, 1024 elements on."""
    calls = bind(BN.DsaRMSNormSumSqV3, N=N, EPS=EPS)
    word = TR.word_of(calls)
    inputs = _inputs(word, 32, N)
    _differ(calls, BS.evaluate_words(calls, *inputs), E.evaluate_batch(word, *inputs))


@pytest.mark.parametrize("defn,S,least", SERVED, ids=[f"{d.name}-{'-'.join(map(str, S.values()))}" for d, S, _ in SERVED])
def test_a_unit_as_calls_stays_small_at_the_served_statics(defn, S: dict, least: dict) -> None:
    """At GLM-5.3's statics a unit written as Calls is a few rows of Calls: its Program's description is within 4 KiB of the
    one at its smallest binding that holds every piece (a traced unit's holds every gate, so it grows with the row) and under
    a byte for every 8 its traced version's holds at 56 bytes an AND; its Flock block, read off the Definition as
    circuit-check's ``block`` reads it, is within the proved range and not over its word view's."""
    from verity_flock import circuit as FC

    unit = bind(defn, **S)
    size, small = (len(canonical_json(encode_program(Program(u)))) for u in (unit, bind(defn, **least)))
    b, word = FC.unit_block(unit), FC.unit_block(TR.word_of(unit))
    assert size <= small + 4096, (unit.id, size, small)
    assert size * 8 < 56 * b["ands"], (unit.id, size, b["ands"])
    assert FC.past_block(b) is None, (unit.id, b["k_log"])
    assert b["bits"] <= word["bits"], (unit.id, b["bits"], word["bits"])


def _lowered(unit) -> tuple[list, int]:
    """Flock's flat lowering of ``unit`` (`ir_lower.lower`, as a served unit's stager runs it): each committed bit and the
    output's forms, and how many activations of the norm's pieces it lowered by a builder."""
    from verity.circuits.evaluation.reference import signature_of
    from verity_flock import ir_lower as IL
    from verity_flock.class_statement import _UnitCircuit

    C, built = _UnitCircuit(), []
    activation = IL._lower_activation

    def counted(C, s, *rest):
        built.append(s.spec.id.split("_v")[0])
        return activation(C, s, *rest)

    params, _ = signature_of(unit)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for t in params for w in t.leaf_widths())]
    IL._lower_activation = counted
    try:
        out = [b for w in IL.lower(C, unit, ins) for b in w]
    finally:
        IL._lower_activation = activation
    bits = [(k, (tuple(d[0].s), d[0].c, tuple(d[1].s), d[1].c) if k == "and" else d) for k, d in zip(C.kind, C.data)]
    return [bits, [(tuple(x.s), x.c) for x in out]], sum(n in {p.name for p in PIECES} for n in built)


@pytest.mark.slow
def test_the_pieces_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each piece carries a builder, so Flock's flat lowering runs each of its activations and keeps no word a gate of it.  The
    circuit is the one its gates lower to, AND for AND: each piece alone, an apply unit of 3 elements (its `DsaRMSNormOut`s
    read the scale's own gates from `DsaRMSNormScale`'s wires) and the smallest scale (128 `DsaRMSNormSquares` and the
    epilogue, which reads the adds' wires).  The scale lowers 0.8M ANDs twice (about a minute); only a unit has a builder
    read its operands from the other Calls' wires."""
    units = [bind(BN.DsaRMSNormSquares), bind(BN.DsaRMSNormSquareFma), bind(BN.DsaRMSNormMeanRsqrt, N=256, EPS=EPS),
             bind(BN.DsaRMSNormScale), bind(BN.DsaRMSNormOut), bind(BN.DsaRMSNormApplyV3, G=3),
             bind(BN.DsaRMSNormSumSqV3, N=256, EPS=EPS)]
    with_builders = [_lowered(u) for u in units]
    assert [n for _, n in with_builders] == [1, 1, 1, 1, 1, 1 + 3, 128 + 1]
    for piece in PIECES:
        monkeypatch.delattr(piece, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, n = _lowered(unit)
        assert n == 0 and by_gates == circuit, unit.id
