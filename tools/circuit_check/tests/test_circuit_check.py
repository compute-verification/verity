"""circuit-check over every registered Definition and subcircuit template, a served row's request Program, and the checks' own
negatives (each check fires by name on a circuit built to break it).

A failure is a correctness or invariant violation.  The ones `main` already has are listed in `circuit_check.known` with their
cause; a test whose only failures are known is an expected failure, and `test_known_failures_are_current` fails once a known one
is fixed, so the entry is dropped.  Redundant and dead gates are warnings: counted in the report, never a test failure here.
The per-target tests are marked `circuit_suite`: `check` deselects them and runs `circuit-check --all`, the same checks.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from circuit_check import checks as CK
from circuit_check import known as KN
from circuit_check import targets as TG
from circuit_check import vectors as VX

REPO = Path(__file__).resolve().parents[3]
OPTS = CK.Options()
MEMO: dict = {}
PINS = CK.load_pins()
SUITE = TG.suite()
TEMPLATES = {label: (sub, sts) for label, sub, sts in TG.templates()}

_known = KN.known


def _clean(entry: dict) -> None:
    new = [f["name"] for f in entry["failures"] if _known(f) is None]
    assert not new, "\n".join(new)
    known = sorted({_known(f) for f in entry["failures"]} - {None})
    if known:
        pytest.xfail(" | ".join(known))


# --- coverage -----------------------------------------------------------------------------------------------------------------

def test_every_registered_definition_is_checked():
    """Every registered Definition family is checked at a binding the catalog reaches, and every catalog root builds: a new
    Definition without one fails here by name (add its binding to `circuit_check.targets`)."""
    assert not TG.ROOT_ERRORS, TG.ROOT_ERRORS
    assert len(TG.REGISTERED) > 200
    checked = {TG.family(fn) for fn in SUITE.values()}
    missing = sorted(set(TG.REGISTERED) - checked)
    assert not missing, f"registered Definitions no circuit-check binding reaches: {missing}"


def test_every_template_is_checked():
    from verity_numerical.bench import templates as TM

    assert {sub.template.id for sub, _ in TEMPLATES.values()} == set(TM.TEMPLATES)


def test_the_partition_applies_to_the_calls_of_served_programs():
    calls = TG.call_families()
    assert {"Gemm_v1", "RMSNormTriton_v1", "Attention_v3", "TokenSelect_v1", "GumbelTopPTokenSelect_v1", "RoPEHead_v1"} <= calls
    assert not {"AttnBlock_v3", "DotBf16_v1", "Serve_v1", "TopPMask_v1"} & calls


# --- the suite ----------------------------------------------------------------------------------------------------------------

@pytest.mark.circuit_suite
@pytest.mark.parametrize("fid", sorted(SUITE))
def test_definition(fid, results):
    e = CK.check_definition(SUITE[fid], OPTS, memo=MEMO, pins=PINS)
    results.append({**e, "kind": "definition", "target": fid})
    _clean(e)


@pytest.mark.circuit_suite
@pytest.mark.parametrize("label", sorted(TEMPLATES))
def test_template(label, results):
    sub, sts = TEMPLATES[label]
    r = CK.check_subcircuit(sub, OPTS, instance_statics=sts, memo=MEMO, pins=PINS)
    results.append({**r, "kind": "template", "target": label})
    assert all(low["lowered"] or low.get("reason") for low in r["lowering"])
    _clean(r)


def _request_program(tmp_path: Path, **kw) -> Path:
    """The vLLM tests' synthetic request Program (per token Embedding -> RMSNorm -> Gemm -> RoPE -> SiluMul -> Gemm -> RMSNorm, per
    step the LM head and TokenSelect; ``attn``: the attention and its o_proj, whose output is a leaf) as ``instances.json.gz``."""
    path = REPO / "integrations" / "vllm" / "tests" / "check" / "test_sampled_replay.py"
    spec = importlib.util.spec_from_file_location("circuit_check_request_program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst, _w = mod._program(**kw)
    out = tmp_path / "instances.json.gz"
    out.write_bytes(gzip.compress(json.dumps(inst).encode()))
    return out


def test_row(tmp_path, results):
    r = CK.check_row(_request_program(tmp_path, rope=True), OPTS, memo=MEMO, pins=PINS)
    results.append({**r, "kind": "row", "target": "synthetic request Program"})
    assert r["calls"] == 42 and len(r["definitions"]) >= 8
    _clean(r)


def test_a_call_output_no_commitment_holds_is_named(tmp_path):
    """The attention fixture's o_proj Gemm returns a value no later Call reads and the query does not require: a unit would end
    at an uncommitted value, across Calls."""
    r = CK.check_row(_request_program(tmp_path, attn=True), OPTS, memo=MEMO, pins=PINS)
    assert "partition/output-not-committed Gemm_v1{K=48,N=16} (4 Call(s) under model.model.layers.0.self_attn.o_proj)" in \
        [f["name"] for f in r["failures"]]


@pytest.mark.circuit_suite
def test_known_failures_are_current():
    """Each known entry still fails somewhere: a fixed one is dropped from `circuit_check.known`."""
    fails = [f for fid in sorted(SUITE) for f in CK.check_definition(SUITE[fid], OPTS, memo=MEMO, pins=PINS)["failures"]]
    assert not KN.stale(fails), KN.stale(fails)


# --- the vectors ----------------------------------------------------------------------------------------------------------------

def test_the_vectors_start_with_every_edge():
    from verity.ir.types import Array, Value

    a, x = VX.vectors([Value(32), Array(3, Value(16))], 40, 1)
    assert a.dtype == np.uint32 and x.shape == (40, 3) and x.dtype == np.uint16
    e32, e16 = VX.edges(32), VX.edges(16)
    for w in (0x7FC00000, 0xFFC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0, 0x80000000, 1, 0x007FFFFF, 0x7F7FFFFF, 0xFF7FFFFF):
        assert w in e32
    for w in (0x7FC0, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x007F, 0x7F7F, 0x7C00, 0x7E00, 0x7BFF):   # bf16 and fp16
        assert w in e16
    for w in (0x7F, 0xFF, 0x7E, 0x01, 0x7C, 0x7B):                                                  # e4m3 and e5m2
        assert w in VX.edges(8)
    assert a[:len(e32)].tolist() == list(e32) and (x[:len(e16)] == np.array(e16)[:, None]).all()


# --- negatives: each check fires, by name ---------------------------------------------------------------------------------------

def _composite(name, sig, body):
    from verity.ir.defs import CompositeDefinition, bind

    return bind(CompositeDefinition(name, 1, (), lambda S: sig, body, register=False))


def _f32x2():
    from verity.ir.types import Array, Value

    return (("x", Array(2, Value(32))),)


def test_a_value_computed_twice_in_one_unit_is_a_redundant_gate():
    from verity.ir.types import Value
    from verity_vllm.program.registry import prims as P

    fn = _composite("CircuitCheckRedundant", (_f32x2(), Value(32)),
                    lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])))
    e = CK.check_definition(fn, OPTS, as_call=True)
    (w,) = [w for w in e["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"]
    assert w["count"] == 1 and w["where"] == {"F32Add_v1 @ CircuitCheckRedundant_v1": 1}
    assert not [f for f in e["failures"] if f["check"] == "partition"]


def test_a_value_computed_in_two_units_is_a_recompute_failure():
    """Two identical calls, each returned by its own unit (the body reads only its parameters, which `word.unit_rule` partitions
    node by node; the whole Definition's cut compares them)."""
    from verity.ir.refs import array_of
    from verity.ir.types import Array, Value
    from verity_vllm.program.registry import prims as P

    fn = _composite("CircuitCheckRecompute", (_f32x2(), Array(2, Value(32))),
                    lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])]))
    e = CK.check_definition(fn, OPTS, as_call=True)
    assert "partition/gate-recomputed CircuitCheckRecompute_v1 (1 gate(s))" in [f["name"] for f in e["failures"]]
    inner = CK.check_definition(fn, OPTS, as_call=False)                 # not a Call: its copies are redundant gates, not a failure
    assert not [f for f in inner["failures"] if f["check"] == "partition"] and inner["partition"]["isolated"]
    assert [w["count"] for w in inner["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"] == [1]


def test_a_gate_no_output_reads_is_a_dead_gate():
    from verity.ir.types import Value
    from verity_vllm.program.registry import prims as P

    def body(B, S, x):
        B.call(P.F32Mul, x[0], x[1])
        return B.call(P.F32Add, x[0], x[1])

    e = CK.check_definition(_composite("CircuitCheckDead", (_f32x2(), Value(32)), body), OPTS)
    (w,) = [w for w in e["warnings"] if w["check"] == "dead-gates"]
    assert w["count"] == 1 and w["where"] == {"F32Mul_v1 @ CircuitCheckDead_v1": 1}


def test_a_unit_wider_than_the_word_fails_the_width_rule():
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.types import Value

    wide = PrimitiveDefinition("CircuitCheckWide", 1, (("a", Value(32)), ("b", Value(32))), Value(64), lambda a, b: (a << 32) | b,
                               register=False)
    fn = _composite("CircuitCheckWidth", ((("a", Value(32)), ("b", Value(32))), Value(64)), lambda B, S, a, b: B.call(wide, a, b))
    names = [f["name"] for f in CK.check_definition(fn, OPTS, as_call=True)["failures"]]
    assert any(n.startswith("partition/width CircuitCheckWidth_v1 output:CircuitCheckWide_v1") for n in names), names


def test_a_partial_reference_is_named():
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.types import Value

    def finite_only(a):
        if (a >> 23) & 0xFF == 0xFF:
            raise ValueError("non-finite")
        return a

    prim = PrimitiveDefinition("CircuitCheckPartial", 1, (("a", Value(32)),), Value(32), finite_only, register=False)
    (f,) = CK.check_definition(prim, OPTS)["failures"]
    assert f["name"].startswith("correspondence/reference-not-total CircuitCheckPartial_v1 raised on")


def test_a_kernel_wrong_only_on_edge_words_is_caught():
    """A kernel that returns 0 for a NaN operand: uniform random words rarely hold a NaN, the edge vectors always do."""
    from verity.evaluation import register_kernel
    from verity.evaluation.batch import _KERNELS
    from verity.ir.defs import PrimitiveDefinition
    from verity.ir.types import Value

    prim = PrimitiveDefinition("CircuitCheckKernel", 1, (("a", Value(32)),), Value(32), lambda a: a ^ 1, register=False)
    nan = lambda a: ((a >> 23) & 0xFF == 0xFF) & ((a & 0x7FFFFF) != 0)  # noqa: E731
    register_kernel(prim, lambda a: (np.where(nan(a), 0, a ^ 1).astype(np.uint32), None))
    try:
        e = CK.check_definition(prim, OPTS)
    finally:
        del _KERNELS[prim.id]
    (f,) = e["failures"]
    assert f["name"].startswith("correspondence/kernel-mismatch CircuitCheckKernel_v1 [numpy]")


def test_a_lowering_that_differs_from_the_ir_is_caught(monkeypatch):
    from verity_flock import ir_lower as IL
    from verity_vllm.program.registry import prims as P

    add = IL.PIECES["F32Add_v1"]
    monkeypatch.setitem(IL.PIECES, "F32Mul_v1", lambda C, a, b: add(C, a, b))
    e = CK.check_definition(P.F32Mul, OPTS, pins=PINS)
    assert any(f["name"].startswith("correspondence/lowering-mismatch F32Mul_v1 [c-flock]") for f in e["failures"])


def test_a_lowering_count_off_its_pin_is_named():
    from verity_vllm.program.registry import prims as P

    pins = {"pieces": {"F32Mul_v1": {"and": 1}}, "units": {}}
    e = CK.check_definition(P.F32Mul, OPTS, pins=pins)
    assert any(f["name"].startswith("lowering/pin-mismatch F32Mul_v1 [F32Mul_v1]: and 1 -> ") for f in e["failures"])


def test_redundant_ands_are_counted():
    from verity_flock import gf2

    C = gf2.Circuit()
    a, b = C.input("a", 2)
    x = C.AND(a, b)
    y = C.AND(b, a)                       # the same product again
    z = C.AND(x ^ b, y ^ b)               # equal operands once y is x
    C.output("o", [x, y, z])
    assert CK.redundant_ands(C)[0] == 2


def test_the_cli_writes_a_report_and_fails_by_name(tmp_path, capsys):
    from circuit_check import cli

    out = tmp_path / "r.json"
    assert cli.main(["Bf16ToF32_v1", "RoPEHead_v1{D=8}", "--out", str(out), "-q"]) == 0
    r = json.loads(out.read_text())
    assert r["schema"] == CK.SCHEMA and [t["target"] for t in r["targets"]] == ["Bf16ToF32_v1", "RoPEHead_v1{D=8}"]
    assert r["summary"]["ok"] and all(t["kind"] == "definition" for t in r["targets"])
    capsys.readouterr()
    assert cli.main(["F32Add_v1", "-q"]) == 0                                    # a known failure: named, not failing
    assert "known correspondence/lowering-mismatch F32Add_v1 [c-flock]" in capsys.readouterr().out
    assert cli.main(["F32Add_v1", "-q", "--strict"]) == 1
