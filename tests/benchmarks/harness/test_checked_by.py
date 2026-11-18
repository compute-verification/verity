"""Who ran a check our benchmarks print or store (`verity_bench.checked_by`): proofs' runners and no word of our own, and
`not recorded` for a result that predates the field."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from verity_bench import checked_by as C
from verity_bench.vllm import table
from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import expand_sweep, resolve_spec
from verity_bench.vllm.runner import run_benchmark

from test_served_table import COMMIT_SPEC, RTX, ScaledFactory

REPO = Path(__file__).resolve().parents[3]


def _ran_by() -> dict:
    tree = ast.parse((REPO / "benchmarks/one_stage/served_verdicts.py").read_text())
    [value] = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "RAN_BY" for t in node.targets)]
    return ast.literal_eval(value)


def test_the_runners_are_proofs_and_no_word_of_ours():
    assert list(C.CHECKED_BY) == list(_ran_by())
    assert C.CHECKED_BY["device"] == _ran_by()["device"]
    assert "no auditor-side entry runs it" in C.CHECKED_BY["piece"]


def test_the_words_name_the_runner_or_say_not_recorded_and_refuse_any_other():
    assert C.words("piece") == ("checked by: piece (run by the developer's side on the served node: the benchmark's own "
                                "process or one it starts there; no auditor-side entry runs it)")
    assert C.words("lean") == ("checked by: lean (recomputed by the Lean verifier in the run's own job, on the developer's "
                               "node)")
    assert C.words(None) == "checked by: not recorded"
    with pytest.raises(ValueError, match="'verified': one of 'piece', 'lean', 'device'"):
        C.words("verified")
    assert C.of([{"checked_by": "piece"}, {"checked_by": "piece"}]) == "piece"
    assert C.of([{"checked_by": "piece"}, {}]) is None and C.of([]) is None


def _result() -> dict:
    [point] = [p for p in expand_sweep("quick") if p.id == "b8-p512-mixed"]
    arms = (Arm("stock-fp8"), Arm("commit", control="stock-fp8"))
    return run_benchmark(resolve_spec(COMMIT_SPEC).preset, arms, (point,), sweep="quick", reps=2,
                         factory=ScaledFactory(1.0, RTX), git_commit="c" * 40)


def test_every_gate_a_vllm_run_stores_is_its_own_process_and_its_table_says_so():
    doc = _result()
    assert doc["gates"] and all(g["checked_by"] == "piece" for g in doc["gates"])
    lines = table.render(doc).splitlines()
    assert f"Gates {C.words('piece')}." in lines
    assert not any("verified" in line for line in lines)


def test_a_vllm_result_that_predates_the_field_renders_not_recorded():
    doc = _result()
    old = {**doc, "gates": [{k: v for k, v in g.items() if k != "checked_by"} for g in doc["gates"]]}
    assert "Gates checked by: not recorded." in table.render(old).splitlines()
