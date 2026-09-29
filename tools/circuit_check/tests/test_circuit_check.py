"""circuit-check over every registered Definition and subcircuit template, a served row's request Program, and the checks' own
negatives (each check fires by name on a circuit built to break it).

A failure is a correctness or invariant violation.  The ones `main` already has are listed in `circuit_check.known` with their
cause; a test whose only failures are known is an expected failure, and `test_known_failures_are_current` fails once a known one
is fixed, so the entry is dropped.  Redundant and dead gates are warnings: counted in the report, never a test failure here.
The per-target tests are marked `circuit_suite` and collected only under `-m circuit_suite` (see conftest.py): `check` runs the
same checks as `circuit-check --all`.
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


@pytest.mark.slow
def test_row(tmp_path, results):
    r = CK.check_row(_request_program(tmp_path, rope=True), OPTS, memo=MEMO, pins=PINS)
    results.append({**r, "kind": "row", "target": "synthetic request Program"})
    assert r["calls"] == 42 and len(r["definitions"]) >= 8
    _clean(r)


@pytest.mark.slow
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


@pytest.mark.slow
def test_the_cli_writes_a_report_and_fails_by_name(tmp_path, capsys):
    from circuit_check import cli

    out = tmp_path / "r.json"
    assert cli.main(["Bf16ToF32_v1", "RoPEHead_v1{D=8}", "--out", str(out), "-q"]) == 0
    r = json.loads(out.read_text())
    assert r["schema"] == CK.SCHEMA and [t["target"] for t in r["targets"]] == ["Bf16ToF32_v1", "RoPEHead_v1{D=8}"]
    assert r["summary"]["ok"] and all(t["kind"] == "definition" for t in r["targets"])
    capsys.readouterr()
    fp8 = "ScaledMmFp8Block_v1{K=128,N=128,G=128}"
    assert cli.main([fp8, "-q"]) == 0                                            # a known failure: named, not failing
    assert f"known partition/gate-recomputed {fp8}" in capsys.readouterr().out
    assert cli.main([fp8, "-q", "--strict"]) == 1


def test_the_cache_reuses_a_targets_result_only_while_every_read_is_unchanged(tmp_path):
    """Two runs of one target: the second reuses it.  An entry whose recorded digest of one file it read no longer matches (as
    after an edit) is not reused, and the run records a new entry."""
    import subprocess
    import sys

    def run(*targets):
        r = subprocess.run([sys.executable, "-m", "circuit_check", *targets, "-q", "--cache", str(tmp_path)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout

    assert "0 of 1 targets reused" in run("F32Mul_v1") and "1 of 1 targets reused" in run("F32Mul_v1")
    assert "1 of 2 targets reused" in run("F32Mul_v1", "rope-head")           # per target: the new one runs, the old one is reused
    (entry,) = [p for p in tmp_path.glob("*/*/*.json") if json.loads(p.read_text())["key"]["request"]["target"][1] == "F32Mul_v1"]
    e = json.loads(entry.read_text())
    assert "packages/verity/src/verity/ir/defs.py" in e["reads"]["files"] and e["reads"]["dirs"]
    assert e["key"]["version"] == entry.parent.parent.name and e["key"]["request"]["target"] == ["definition", "F32Mul_v1"]
    e["reads"]["files"]["packages/verity/src/verity/ir/defs.py"] = "0" * 64
    entry.write_text(json.dumps(e))
    assert "0 of 1 targets reused" in run("F32Mul_v1") and len(list(entry.parent.glob("*.json"))) == 2


def test_a_result_after_a_read_the_tracer_cannot_see_is_not_stored(tmp_path):
    """A check that starts a program (its reads are untraced) is not stored, nor is any later one in that process; a library
    built in the tree and loaded by the dynamic loader is an input by its bytes."""
    import subprocess
    import sys

    script = f"""
import json, ctypes.util
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli
CC.trace(cli.REPO)
ok = lambda: {{"failures": [], "warnings": []}}
def spawns():
    import subprocess
    subprocess.run(["true"])
    return ok()
cli._tasks = lambda *a: [("definition", "A", ok), ("definition", "B", spawns), ("definition", "C", ok)]
rep = cli.run([], CK.Options(), log=None, cache=Path({str(tmp_path)!r}))
print(json.dumps(rep["cache"]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    assert stats["stored"] == 1 and [u["target"] for u in stats["untraced"]] == ["B", "C"]
    assert "subprocess.Popen" in stats["untraced"][0]["because"][0]


def test_a_check_that_raised_or_read_another_trees_module_is_not_stored(tmp_path):
    """A check that raised is no verdict (T11: an ImportError from a tree a concurrent run was rewriting, replayed on the next
    tree); a module imported from outside the tree is a read no key holds, so that result and later ones are not stored."""
    import subprocess
    import sys

    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "other_tree_mod.py").write_text("X = 1\n")
    script = f"""
import json, sys
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli
CC.trace(cli.REPO)
ok = lambda: {{"failures": [], "warnings": []}}
def broken():
    from verity_numerical.bench import no_such_module  # noqa: F401
def outside():
    sys.path.insert(0, {str(tmp_path / 'elsewhere')!r})
    import other_tree_mod  # noqa: F401
    return ok()
cli._tasks = lambda *a: [("definition", "A", ok), ("definition", "B", broken), ("definition", "C", outside), ("definition", "D", ok)]
rep = cli.run([], CK.Options(), log=None, cache=Path({str(tmp_path / 'cache')!r}))
print(json.dumps(rep["cache"]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    assert stats["stored"] == 1 and stats["errors"] == ["B"] and [u["target"] for u in stats["untraced"]] == ["C", "D"]
    assert "imported other_tree_mod from" in stats["untraced"][0]["because"][0] and "outside the tree" in stats["untraced"][0]["because"][0]


def test_a_result_is_not_stored_when_the_tree_changed_while_it_ran(tmp_path):
    """The run remembers each read's digest from its first look; the process that ran the check digests its reads again right
    after it. When they differ (a concurrent run rewrote the tree), nothing is stored: the report may be of a tree no digest names."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("")
    got = {"files": ["a.txt"], "dirs": ["pkg"], "probes": [], "libs": []}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    k, rep = {"version": "v", "request": {}}, {"failures": [], "warnings": []}
    CC._digests(tmp_path, got)                                          # the run's first look, before the check
    (tmp_path / "pkg" / "m.py").unlink()                                # a concurrent run removes a file while it runs
    CC._VISIBLE.clear()
    assert CC.store(tmp_path / "cache", k, tmp_path, got, rep, CC.seen(tmp_path, got)) is None
    assert not list((tmp_path / "cache").rglob("*.json"))
    (tmp_path / "pkg" / "m.py").write_text("")
    CC._VISIBLE.clear()
    assert CC.store(tmp_path / "cache", k, tmp_path, got, rep, CC.seen(tmp_path, got))    # the same tree: stored
    CC._VISIBLE.clear()
    CC._MEMO.clear()


def test_parallel_jobs_give_the_serial_report(tmp_path):
    from circuit_check import cli

    specs = ["F32Mul_v1", "RMSNormTriton_v1{N=64,EPS=1e-05}", "TokenSelect_v1{V=64}", "rope-head"]
    strip = lambda rep: [{k: v for k, v in t.items() if k != "seconds"} for t in rep["targets"]]  # noqa: E731
    assert strip(cli.run(specs, OPTS, log=None, jobs=3)) == strip(cli.run(specs, OPTS, log=None, jobs=1))


def test_the_cache_reads_the_tree_as_git_has_it(tmp_path):
    """A build or bytecode inside the tree (ignored by git) changes no recorded read; a new file git sees does."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "a.txt").write_text("a")
    got = {"files": ["a.txt", "build/lib.so"], "dirs": ["."], "probes": ["build/lib.so", "b.txt"]}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    before = CC._digests(tmp_path, got)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "lib.so").write_bytes(b"built")
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    assert CC._digests(tmp_path, got) == before
    (tmp_path / "b.txt").write_text("b")
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    after = CC._digests(tmp_path, got)
    assert after["dirs"] != before["dirs"] and after["probes"]["b.txt"] == "file"
    CC._VISIBLE.clear()
    CC._MEMO.clear()
