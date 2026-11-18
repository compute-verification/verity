"""Who ran a check our benchmarks print or store (`kit_report.CHECKED_BY`): proofs' words for the developer's piece and
no word of our own, and `not recorded` for a result that predates the field."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from verity_bench import kit_report as K
from verity_bench.vllm import table
from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import expand_sweep, resolve_spec
from verity_bench.vllm.runner import run_benchmark

from bench_standins import CPUS
from test_served_table import COMMIT_SPEC, RTX, ScaledFactory

REPO = Path(__file__).resolve().parents[3]


def _proofs(name: str) -> str:
    tree = ast.parse((REPO / "benchmarks/one_stage/served_verdicts.py").read_text())
    [value] = [node.value for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
               and any(isinstance(t, ast.Name) and t.id == name
                       for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    return ast.literal_eval(value)


def test_the_piece_is_in_proofs_words_and_no_word_of_ours():
    assert K.CHECKED_BY[K.PIECE] == _proofs("PIECE")
    assert set(K.CHECKED_BY) == {"piece", "lean"}


def test_the_words_name_the_runner_or_say_not_recorded_and_refuse_any_other():
    assert K.checked_by("piece") == "checked by: piece (run by the developer's piece; no auditor-side entry runs it)"
    assert K.checked_by("lean") == ("checked by: lean (recomputed by the Lean verifier in the run's own job, on the "
                                    "developer's node)")
    assert K.checked_by(None) == "checked by: not recorded"
    with pytest.raises(ValueError, match="'verified': one of 'piece', 'lean'"):
        K.checked_by("verified")
    assert K.checked_by_of([{"checked_by": "piece"}, {"checked_by": "piece"}]) == "piece"
    assert K.checked_by_of([{"checked_by": "piece"}, {}]) is None and K.checked_by_of([]) is None


def _result() -> dict:
    [point] = [p for p in expand_sweep("quick") if p.id == "b8-p512-mixed"]
    arms = (Arm("stock-fp8"), Arm("commit", control="stock-fp8"))
    return run_benchmark(resolve_spec(COMMIT_SPEC).preset, arms, (point,), sweep="quick", reps=2,
                         factory=ScaledFactory(1.0, RTX), git_commit="c" * 40, cpus=CPUS)


def test_every_gate_a_vllm_run_stores_is_its_own_process_and_its_table_says_so():
    doc = _result()
    assert doc["gates"] and all(g["checked_by"] == "piece" for g in doc["gates"])
    lines = table.render(doc).splitlines()
    assert f"Gates {K.checked_by('piece')}." in lines
    assert not any("verified" in line for line in lines)


def test_a_vllm_result_that_predates_the_field_renders_not_recorded():
    doc = _result()
    old = {**doc, "gates": [{k: v for k, v in g.items() if k != "checked_by"} for g in doc["gates"]]}
    assert "Gates checked by: not recorded." in table.render(old).splitlines()
