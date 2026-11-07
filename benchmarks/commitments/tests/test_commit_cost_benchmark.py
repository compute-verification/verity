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
    assert [r["scheme"] for r in rows] == ["frame-v3", "frame-v3-sha256-row"]
    assert len({r["root"] for r in rows}) == 2
    names = {m["name"] for m in check.doc["measurements"]}
    assert {"frame-v3.commit_s", "frame-v3-sha256-row.tree_s", "frame-v3-sha256-row.leaves_per_s"} <= names


def test_commit_cost_sha512_framings(tmp_path: Path) -> None:
    assert _bench().main(["--scheme", "frame-v3-sha512", "--scheme", "frame-v3-sha512-row", "--leaves", "7",
                          "--leaf-bytes", "64", "--reps", "1", "--out", str(tmp_path)]) == 0
    rows = json.loads((tmp_path / "commit_cost.json").read_text())
    assert [r["hashes"] for r in rows] == [["sha512"]] * 2 and all(len(r["root"]) == 128 and r["opening_verified"] for r in rows)


def test_commit_cost_blake3_rows(tmp_path: Path) -> None:
    assert _bench().main(["--scheme", "frame-v3-blake3-row", "--leaves", "5", "--leaf-bytes", "64", "--out", str(tmp_path)]) == 0


def test_hiding_leaf_cost_sha512_pos_leaf_is_the_pinned_leaf() -> None:
    import hashlib

    from verity_vllm.commit import scheme

    pinned = {b"\x00" * 4: "57270a148d084b84aba8ab1e679e1cd01f11b2e9a8cca85aacdb63dde44b3085"
                           "6a5ecad7bc135f8e9ca5486666ad73fe23fa360826fe2d0a544b8fc7a7fd1b17",
              bytes(range(256)) * 12: "440a3c2d8e3031973f506bd4b86d9737c4c4a8054174364f6aad2ed4e1c50a58"
                                      "d5c7ab36a860507688bda813eccb9ccd1911f07a34aac3c24ba27cedb547cbe9"}
    for value, leaf in pinned.items():
        assert scheme.pos_leaf(memoryview(value), hashlib.sha512).hex() == leaf


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


def test_the_ripple_carry_model_counts_sampled_proofs_sha512_ands() -> None:
    """hiding_leaf_cost's model of a SHA-512 compression (80 rounds of 7 additions, Ch and Maj, 64 schedule words of 3
    additions and 8 feed-forward additions, a w-bit addition w - 1 ANDs at w = 64) gives the count it reads."""
    from verity.core.protocols.sampled_proofs.optimize import SHA512_ANDS

    add64 = 63
    assert 80 * (7 * add64 + 64 + 64) + 64 * 3 * add64 + 8 * add64 == SHA512_ANDS


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
    params, key = tool.resolve_params(["--scheme", "frame-v3-sha512", "--leaves", "1024"], ROOT)
    assert key == {"scheme": "frame-v3-sha512", "leaves": 1024, "leaf_bytes": 256}
