"""The commit-cost benchmark runs both schemes, writes a valid result, and its Tool declaration loads by name."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from research import result as research_result
from research.store import tools_registry

ROOT = Path(__file__).resolve().parents[1]


def _bench():
    spec = importlib.util.spec_from_file_location("commit_cost", ROOT / "benchmarks" / "commitments" / "commit_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_commit_cost_writes_a_valid_result(tmp_path: Path) -> None:
    assert _bench().main(["--leaves", "37", "--leaf-bytes", "40", "--reps", "1", "--out", str(tmp_path)]) == 0
    check = research_result.check(tmp_path, None)
    assert check.valid, check.errors
    assert check.validation_status == "passed"
    rows = json.loads((tmp_path / "commit_cost.json").read_text())
    assert [r["scheme"] for r in rows] == ["frame-v3", "vllm-production/v1"]
    assert rows[0]["root"] != rows[1]["root"]
    names = {m["name"] for m in check.doc["measurements"]}
    assert {"frame-v3.commit_s", "vllm-production.commit_s", "vllm-production.leaves_per_s"} <= names


def test_commit_cost_root_is_the_reference_root() -> None:
    from verity.commitments import vllm_production as vp

    bench = _bench()
    row = bench.measure("vllm-production", 9, 16, 1)
    assert row["root"] == vp.SCHEME.root([vp.pos_leaf(v) for v in bench.values(9, 16)]).hex()


def test_commit_cost_tool_is_registered() -> None:
    tool = tools_registry.load("commit_cost", ROOT)
    assert tool.name == "commit_cost"
    params, key = tool.resolve_params(["--scheme", "vllm-production", "--leaves", "1024"], ROOT)
    assert key == {"scheme": "vllm-production", "leaves": 1024, "leaf_bytes": 256}
