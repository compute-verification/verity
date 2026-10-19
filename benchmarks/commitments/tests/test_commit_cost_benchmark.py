"""The commit-cost benchmarks run, write a valid result, and the commit-cost Tool declaration loads by name."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from research import result as research_result
from research.store import tools_registry

ROOT = Path(__file__).resolve().parents[3]


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
    assert [r["scheme"] for r in rows] == ["frame-v3", "frame-v3-sha256-row", "vllm-v1"]
    assert len({r["root"] for r in rows}) == 3
    names = {m["name"] for m in check.doc["measurements"]}
    assert {"frame-v3.commit_s", "frame-v3-sha256-row.tree_s", "vllm-v1.commit_s", "vllm-v1.leaves_per_s"} <= names


def test_commit_cost_sha512_framings(tmp_path: Path) -> None:
    assert _bench().main(["--scheme", "frame-v3-sha512", "--scheme", "frame-v3-sha512-row", "--scheme", "vllm-v1-sha512", "--leaves", "7",
                          "--leaf-bytes", "64", "--reps", "1", "--out", str(tmp_path)]) == 0
    rows = json.loads((tmp_path / "commit_cost.json").read_text())
    assert [r["hashes"] for r in rows] == [["sha512"]] * 3 and all(len(r["root"]) == 128 and r["opening_verified"] for r in rows)


def test_commit_cost_blake3_rows(tmp_path: Path) -> None:
    assert _bench().main(["--scheme", "frame-v3-blake3-row", "--leaves", "5", "--leaf-bytes", "64", "--out", str(tmp_path)]) == 0


def test_commit_cost_root_is_the_reference_root() -> None:
    from verity.ml.commitments import vllm_v1

    bench = _bench()
    row = bench.measure("vllm-v1", 9, 16, 1)
    domain = bench.setup("vllm-v1", 9, 16).ports["v"]
    assert row["root"] == domain.bind(vllm_v1.fold([vllm_v1.pos_leaf(v) for v in bench.values(9, 16)])).hex()


def test_hiding_leaf_cost_writes_a_valid_result(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("hiding_leaf_cost", ROOT / "benchmarks" / "commitments" / "hiding_leaf_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--row-bytes", "3072", "--rows", "16", "--reps", "1", "--out", str(tmp_path)]) == 0
    check = research_result.check(tmp_path, None)
    assert check.valid, check.errors
    assert check.validation_status == "passed"
    rows = {r["rule"]: r for r in json.loads((tmp_path / "hiding_leaf_cost.json").read_text())}
    assert rows["blake3-keyed/row/v2"]["ands"] == 50 * 10_416 and rows["hm96/sha256-row"]["ands"] == 52 * 22_696
    assert rows["hm96/pos-leaf"]["stored_bytes"] == 128 and rows["vllm-v1/pos-leaf"]["stored_bytes"] == 0
    assert rows["hm96-sha512/sha512-row"]["ands"] == 27 * 58_120 and rows["sha512/row/v1"]["ands"] == 25 * 58_120
    assert rows["hm96-sha512/pos-leaf"]["stored_bytes"] == 192 and rows["hm96-sha512/pos-leaf"]["sha512"] == 25 + 4
    tool = tools_registry.load("hiding_leaf_cost", ROOT)
    assert tool.resolve_params(["--rows", "64"], ROOT)[1] == {"row_bytes": "4,256,1536,3072,4096,8192", "rows": 64}


@pytest.mark.slow  # the suite's first import of verity_vllm: 8-9 s on the shared pod, a fraction of a second warm
def test_step_root_cost_writes_a_valid_result_whose_root_is_the_references(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("step_root_cost", ROOT / "benchmarks" / "commitments" / "step_root_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--batch", "3", "--layers", "2", "--k-scale", "256", "--reps", "1", "--out", str(tmp_path)]) == 0
    check = research_result.check(tmp_path, None)
    assert check.valid, check.errors
    assert check.validation_status == "passed"
    got = json.loads((tmp_path / "step_root_cost.json").read_text())
    assert got["tables"] == 8 and got["rows"] == 24 and got["reference_root_equal"]
    assert set(got["by_workers"]) == set(got["sha512_floor_s"]) == set(got["sha512_floor_cpu_s"]) == {"1"}
    assert {"step_roots.w1.cpu_seconds", "salts.cpu_seconds"} <= {m["name"] for m in check.doc["measurements"]}


def test_commit_cost_tool_is_registered() -> None:
    tool = tools_registry.load("commit_cost", ROOT)
    assert tool.name == "commit_cost"
    params, key = tool.resolve_params(["--scheme", "vllm-v1", "--leaves", "1024"], ROOT)
    assert key == {"scheme": "vllm-v1", "leaves": 1024, "leaf_bytes": 256}
