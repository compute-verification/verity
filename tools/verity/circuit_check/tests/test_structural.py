"""`circuit_check.structural` (``circuit-check --structural``): a ``flock-ir-tmpl`` template is accepted as its Definition's
lowering by construction when its canonical gates are the flat lowering's, on every input or on random ones when they are
other gates computing the same bits, and refused by name when it computes anything else or is a form neither covers."""
from __future__ import annotations

import dataclasses as dc
import json

import pytest
from circuit_check import cli
from circuit_check import structural as ST
from circuit_check import targets as TG
from verity_flock import ir_lower as IL
from verity_flock import tmpl as TM

from verity.ml.circuits.boolean.forms import maj
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.ops.boolean import scalar as SB

MAJ = definition("StructuralTestMaj", 1, lambda C, a, b, c: [maj(C, a[0], b[0], c[0])],
                 (("a", SB.BIT), ("b", SB.BIT), ("c", SB.BIT)), SB.BIT, register=False)

W = ST.WORD
K = 3 * W  # a hand-built template's constant: its inputs from column 0, its rows from W, its outputs from 2 W


def _unit(ins: list[int], outs: list[int], rows, calls=(), callees=()) -> TM.Tmpl:
    """A template with input ports of widths ``ins`` from column 0, ``rows`` (their columns from W) and ``calls``, and output
    ports of widths ``outs`` from 2 W: its root is the last Definition, after ``callees``."""
    groups = [1, 0, 1, len(ins), *ins, 1, 2, 1, len(outs), *outs]
    inputs = tuple((c, (c,), (c,)) for c in range(sum(ins)))
    root = TM.Def(K + 1, 0, inputs + tuple(rows), tuple(calls))
    return TM.Tmpl("t", len(callees), [str(g) for g in groups], [*callees, root])


def _verdict(target, tm, **kw) -> tuple[str, dict]:
    r = ST.check(target, tm, **kw)
    return r["structural"]["verdict"], r


def _codes(r: dict) -> list[str]:
    return [f["code"] for f in r["failures"]]


# --- the canonical gates ---------------------------------------------------------------------------------------------

def test_an_and_is_one_node_whichever_way_round_and_through_whichever_pair_it_is_written():
    K_ = ST.Canon(3)
    x, y, z = K_.inputs
    xy = K_.AND(x, y)
    assert K_.AND(y, x) == xy
    assert K_.AND(x, x ^ y) == x ^ xy                    # x (x + y) = x + xy
    assert K_.AND(x ^ y, y) == y ^ xy                    # (x + y) y = y + xy
    assert K_.AND(x ^ 1, y ^ 1) == xy ^ x ^ y ^ 1        # (x + 1)(y + 1) = xy + x + y + 1
    assert K_.AND(x, z) != xy and K_.next == 5


def test_constants_and_repeats_fold():
    K_ = ST.Canon(2)
    x, _y = K_.inputs
    assert K_.AND(x, 1) == x and K_.AND(1, x) == x
    assert K_.AND(x, ST.ZERO) == ST.ZERO
    assert K_.AND(x, x) == x and K_.AND(x, x ^ 1) == ST.ZERO
    assert K_.next == 2


def test_a_verdict_does_not_depend_on_the_labels_draw():
    tm = _emitted(SB.E4m3ToF32)
    a, b = (ST.check(SB.E4m3ToF32, tm)["structural"] for _ in range(2))
    assert a["verdict"] == b["verdict"] == "syntactic" and a["canonical_nodes"] == b["canonical_nodes"]
    assert a["labels"] == b["labels"] == {"source": "os.urandom", "bits": 512}


# --- accepted --------------------------------------------------------------------------------------------------------

def _emitted(target) -> TM.Tmpl:
    """``target``'s flat ``flock-ir-unit/v2`` lowering as `tmpl.emit` templates it."""
    C, _ins, words = ST._lowered(target)
    lay = IL.layout(C, list(C.inputs), [f"o{j}" for j in range(len(words))])
    return TM.emit(TM.parse_flat(IL.netlist(lay, "t").splitlines()))[0]


def test_the_emitted_template_of_a_flat_lowering_is_its_lowering_by_construction():
    tm = _emitted(SB.E4m3ToF32)
    assert any(d.calls for d in tm.defs)
    v, r = _verdict(SB.E4m3ToF32, TM.write(tm))
    assert (v, r["failures"], r["warnings"]) == ("syntactic", [], [])
    assert r["structural"]["template"]["output_bits"] == r["structural"]["lowering"]["output_bits"] == 32


def _mux(cols_row=((0, (1,), (2, 3)),)) -> TM.Def:
    """``c (a + b)`` on its inputs ``c, a, b`` (local columns 1, 2, 3): `forms.mux`'s AND."""
    return TM.Def(1, 3, cols_row, ())


SEL_IN = [1] * 33           # c, then a's 16 bits, then b's
A0, B0 = 1, 17
SEL_SEGS = ((1, 0, 0), (1, A0, 1), (1, B0, 1))


def _segs(segs) -> tuple:
    """One-level ``(len, at, step)`` segments as ``flock-ir-tmpl/v2`` writes them: one rep, no rstride."""
    return tuple(s + (1, 0) for s in segs)


def _select(mux: TM.Def | None = None, *, count: int = 16, segs=SEL_SEGS, base: int = W, extra=()) -> TM.Tmpl:
    """SelectBf16 (``c ? a : b``, 16 bits) as one run of 16 calls of ``mux`` from column ``base``, each output bit ``b_j``
    plus its call's columns."""
    mux = mux or _mux()
    outs = tuple((2 * W + j, tuple(W + j * mux.cols + i for i in range(mux.cols)) + (B0 + j,), (K,)) for j in range(16))
    return _unit(SEL_IN, [16], outs + tuple(extra), [TM.Call(0, count, base, mux.cols, _segs(segs))], [mux])


def test_a_run_of_calls_is_the_lowering_it_unrolls_to():
    v, r = _verdict(SB.SelectBf16, _select())
    assert (v, r["failures"]) == ("syntactic", [])
    assert r["structural"]["template_calls"] == 1 and r["structural"]["template"]["live_ands"] == 16


def test_other_gates_computing_the_same_bits_pass_on_every_input_when_the_inputs_are_few():
    """maj = xy + z (x + y), two ANDs, where `forms.maj` writes z + (x + z)(y + z): other nodes, the same function on all
    eight inputs."""
    assert _verdict(MAJ, _unit([1, 1, 1], [1], [(W, (0, 2), (1, 2)), (2 * W, (W, 2), (K,))]))[0] == "syntactic"
    v, r = _verdict(MAJ, _unit([1, 1, 1], [1], [(W, (0,), (1,)), (W + 1, (2,), (0, 1)), (2 * W, (W, W + 1), (K,))]))
    assert (v, r["failures"], r["warnings"]) == ("exhaustive", [], [])
    assert r["structural"]["semantic"] == {"inputs": 3, "exhaustive": True, "vectors": 8, "mismatched_output_bits": 0}
    assert "other forms" in r["structural"]["why_not_syntactic"]


def test_other_gates_on_many_inputs_pass_on_random_ones_with_a_warning():
    """``b + c a + c b`` per bit (two ANDs a call) for `forms.mux`'s ``b + c (a + b)``: 33 input bits, past the
    exhaustive limit, so equal on random inputs only."""
    alt = TM.Def(2, 3, ((0, (2,), (3,)), (1, (2,), (4,))), ())
    v, r = _verdict(SB.SelectBf16, _select(alt), vectors=256)
    assert (v, r["failures"]) == ("random", [])
    assert [w["code"] for w in r["warnings"]] == ["random-only"] and r["warnings"][0]["count"] == 16
    assert r["structural"]["semantic"]["vectors"] == 256 and not r["structural"]["semantic"]["exhaustive"]


# --- refused ---------------------------------------------------------------------------------------------------------

MUTATIONS = {
    "the call reads a where it reads a + b": (_select(_mux(((0, (1,), (2,)),))), "mismatch"),
    "one call fewer": (_select(count=15), "mismatch"),
    "a's segment does not step": (_select(segs=((1, 0, 0), (1, A0, 0), (1, B0, 1))), "mismatch"),
    "the select bit's segment reads a": (_select(segs=((1, A0, 0), (1, A0, 1), (1, B0, 1))), "mismatch"),
    "the callee's row reads its own column": (_select(_mux(((0, (0,), (2, 3)),))), "order"),
    "an output reads the next output's column": (
        dc.replace(_select(), defs=[_mux(), dc.replace(_select().defs[1], rows=tuple(
            (c, a + (c + 1,) if c == 2 * W else a, b) for c, a, b in _select().defs[1].rows))]), "order"),
    "an input's row is not its own": (
        dc.replace(_select(), defs=[_mux(), dc.replace(_select().defs[1], rows=tuple(
            (c, a, (0,) if c == A0 else b) for c, a, b in _select().defs[1].rows))]), "ports"),
    "the ports are other widths": (dc.replace(_select(), groups=[str(g) for g in [1, 0, 1, 2, 1, 32, 1, 2, 1, 1, 16]]),
                                   "ports"),
    "an output group's spare bit is written": (_select(extra=[(2 * W + 16, (0,), (K,))]), "ports"),
    "a column written twice": (_select(extra=[(2 * W, (0,), (K,))]), "malformed"),
    "the run's last call writes the constant": (_select(base=K - 15), "malformed"),
}


@pytest.mark.parametrize("name", MUTATIONS)
def test_a_mutated_template_is_refused(name):
    tm, code = MUTATIONS[name]
    v, r = _verdict(SB.SelectBf16, tm, vectors=256)
    assert v == "refused" and _codes(r) == [code], r["failures"]


def test_a_template_computing_another_function_is_refused_on_every_input():
    v, r = _verdict(MAJ, _unit([1, 1, 1], [1], [(W, (0,), (1,)), (2 * W, (W, 2), (K,))]))   # xy + z
    assert v == "refused" and _codes(r) == ["mismatch"]
    assert r["failures"][0]["exhaustive"] and r["structural"]["semantic"]["mismatched_output_bits"] == 1


def test_a_dead_row_that_reads_ahead_is_refused():
    """Every column, read or not, is a function of the inputs: a row nothing reads may not read a later column either."""
    v, r = _verdict(MAJ, _unit([1, 1, 1], [1], [(W, (0, 2), (1, 2)), (W + 1, (W + 2,), (K,)), (W + 2, (0,), (1,)),
                                                (2 * W, (W, 2), (K,))]))
    assert v == "refused" and _codes(r) == ["order"]


def test_text_that_is_no_template_is_refused():
    v, r = _verdict(MAJ, f"{TM.FORMAT} t 0 1\nDEF x\n")
    assert v == "refused" and _codes(r) == ["malformed"]


# --- the command -----------------------------------------------------------------------------------------------------

def test_the_command_reports_the_verdict_and_fails_a_refused_template(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(TG, "definition", lambda spec: {"SelectBf16_v2": SB.SelectBf16}[spec])
    good, bad, out = tmp_path / "good.tmpl", tmp_path / "bad.tmpl", tmp_path / "r.json"
    good.write_text(TM.write(_select()))
    bad.write_text(TM.write(_select(count=15)))
    assert cli.main(["--structural", str(good), "--out", str(out), "SelectBf16_v2"]) == 0
    rep = json.loads(out.read_text())
    (t,) = rep["targets"]
    assert (rep["schema"], t["kind"], t["structural"]["verdict"]) == ("verity-circuit-check/v1", "structural", "syntactic")
    assert cli.main(["--structural", str(bad), "SelectBf16_v2"]) == 1
    assert "FAIL structural/mismatch" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--structural", str(good), "SelectBf16_v2", "MAJ"])


def test_the_command_resolves_a_definition_id(tmp_path):
    p = tmp_path / "unit.tmpl"
    p.write_text(TM.write(_emitted(SB.E4m3ToF32)))
    rep = cli.structural("E4m3ToF32_v2", p)
    assert rep["targets"][0]["structural"]["verdict"] == "syntactic"


# --- the structural lowering ---------------------------------------------------------------------------------------

def test_the_structural_lowering_is_its_definitions_lowering():
    """`verity_flock.struct_lower` (TMPL1) emits each specialized Definition once and keeps its calls as calls: its template
    of a GEMM scan is the flat lowering by construction."""
    S = pytest.importorskip("verity_flock.struct_lower")
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as GB

    t = bind(GB.GemmAccumulatorS8, K=128, DOT=GB.BlackwellS8ImmaDot32)
    tm = S.lower(t)[0]
    assert any(c.count > 1 for d in tm.defs for c in d.calls)
    assert _verdict(t, tm)[0] == "syntactic"


def test_the_struct_lower_realization_is_held_to_the_reference_from_its_template(monkeypatch):
    """``c-flock:struct_lower`` (`checks.struct_lowering`, under ``Options.struct_max_gates``): a scan's structural template,
    evaluated from the template, equals the reference on every correspondence vector; a template whose outputs differ is
    refused by name on every vector."""
    from circuit_check import checks as CK
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.reference import resolve, signature_of
    from verity.ml.circuits.ops.boolean import gemm as GB

    t = resolve(bind(GB.GemmAccumulatorS8, K=128, DOT=GB.BlackwellS8ImmaDot32))
    opts = CK.Options(vectors=16, struct_max_gates=1 << 30)
    F = CK.Findings()
    _corr, arrays, ref = CK.correspondence(t, opts, F)
    params, _ = signature_of(t)
    rec = CK.struct_lowering(t, params, arrays, ref, F)
    assert (rec["via"], rec["lowered"], rec["ordered"], rec["mismatches"]) == (CK.STRUCT_VIA, True, True, 0)
    assert rec["compared"] == sum(r is not None for r in ref) > 0 and not F.failures
    evaluate = TM.evaluate
    monkeypatch.setattr(TM, "evaluate", lambda *a, **k: ~evaluate(*a, **k))
    rec = CK.struct_lowering(t, params, arrays, ref, F)
    assert rec["mismatches"] == rec["compared"]
    assert [f["code"] for f in F.failures] == ["lowering-mismatch"] and CK.STRUCT_VIA in F.failures[0]["name"]


# --- a proof block on its row ports ----------------------------------------------------------------------------------

def test_a_structural_stagers_unit_is_the_flat_unit_on_its_ports_by_construction():
    """`check_unit`: the proof block a structural stager writes (`struct_lower` on `class_statement.unit_ports`' u16 row
    ports) is the flat proof block the flat stager lays out on those ports (`class_statement.unit_circuit`) by construction,
    with the bound on a wrong verdict recorded. The same template held to rows pairing each x with the next w computes another function
    and is refused; the Definition's own template, on its leaves, is refused by its ports."""
    S = pytest.importorskip("verity_flock.struct_lower")
    from verity_flock import class_statement as CS

    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as GB

    t = bind(GB.GemmAccumulatorS8, K=64, DOT=GB.BlackwellS8ImmaDot32)
    rows = [(f"p{r}", [f"x[{r}]", f"wrow[{r}]"]) for r in range(64)]
    tm = CS.unit_lowering(t, "unit/structural-test", rows, (), True)[0].template
    r = ST.check_unit(t, rows, tm, name="unit/structural-test")
    s = r["structural"]
    assert (s["verdict"], s["unit"], r["failures"]) == ("syntactic", "unit/structural-test", [])
    assert s["error_bound"]["M"] == 4 * (s["template"]["live_rows"] + s["lowering"]["live_ands"]) + 32 + 1
    assert s["error_bound"]["log2"] < -400
    shifted = [(f"p{r}", [f"x[{r}]", f"wrow[{(r + 1) % 64}]"]) for r in range(64)]
    r = ST.check_unit(t, shifted, tm)
    assert r["structural"]["verdict"] == "refused" and _codes(r) == ["mismatch"]
    r = ST.check_unit(t, rows, S.lower(t)[0])
    assert r["structural"]["verdict"] == "refused" and _codes(r) == ["ports"]
