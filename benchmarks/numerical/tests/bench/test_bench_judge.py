"""``benchmarks/judge.py``: refuses across identity fields, and judges otherwise."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import JUDGE, conforming_result, write_run

from verity_numerical.bench import contract


def judge(*args: str, runs_dir: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(JUDGE), *args]
    if runs_dir is not None:
        cmd += ["--runs-dir", str(runs_dir)]
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = conforming_result("r-base")
    cand = conforming_result("r-cand")
    for m in cand["measurements"]:
        if m["name"] in ("t.witness", "t.total", "vu.seconds_per_vu"):
            m["value"] *= 2.0
        if m["name"] == "vu.per_second":
            m["value"] /= 2.0
    cand["workload_fingerprint"]["software"]["backend"]["version"] = "1"
    return tmp_path, write_run(tmp_path, "r-base", base), write_run(tmp_path, "r-cand", cand)


def test_same_campaign_number_is_judged(pair) -> None:
    root, base, cand = pair
    proc = judge("r-base", "r-cand", runs_dir=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "REFUSED" not in out
    assert re.search(r"software\.backend\.version\s+0\s+->\s+1", out)
    assert "t.total" in out and "x2.0000" in out and "x1.0000" in out
    assert "conforms" in out
    # directories work as well as ids, and so does --json
    proc = judge(str(base), str(cand), "--json")
    rep = json.loads(proc.stdout)
    assert proc.returncode == 0 and rep["refused"] == []
    ratios = {m["name"]: m["ratio"] for m in rep["measurements"]}
    assert ratios["t.total"] == 2.0 and ratios["t.lookup"] == 1.0 and ratios["own_name"] == 1.0
    assert rep["fingerprint_deltas"] == [{"field": "software.backend.version", "baseline": "0", "candidate": "1"}]


@pytest.mark.parametrize(
    ("field", "value"),
    [("K", 32), ("proof_class", "ARITHMETIC_DIAGNOSTIC"), ("authentication", "excluded"), ("domain", "total"), ("profile", "other-target")],
)
def test_identity_difference_is_refused(tmp_path: Path, field: str, value) -> None:
    base = write_run(tmp_path, "r-base", conforming_result("r-base"))
    cand = write_run(tmp_path, "r-cand", conforming_result("r-cand", **{field: value}))
    proc = judge(str(base), str(cand))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stdout and f"{field}:" in proc.stdout and "DIFFERS" in proc.stdout
    assert "[measurements]" not in proc.stdout and "x1.0000" not in proc.stdout, "a refused pair shows no ratio"


def test_missing_identity_field_is_refused(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", conforming_result("r-base"))
    lacking = conforming_result("r-cand")
    del lacking["workload_fingerprint"]["authentication"]
    cand = write_run(tmp_path, "r-cand", lacking)
    proc = judge(str(base), str(cand))
    assert proc.returncode == 2
    assert "authentication: included vs <missing>" in proc.stdout
    assert "workload_fingerprint.authentication missing" in proc.stdout


def test_allow_lets_a_named_field_differ(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", conforming_result("r-base"))
    cand = write_run(tmp_path, "r-cand", conforming_result("r-cand", K=1536))
    assert judge(str(base), str(cand)).returncode == 2
    proc = judge(str(base), str(cand), "--allow", "K")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "(allowed to differ)" in proc.stdout and re.search(r"\n  K\s+16\s+->\s+1536\s+\(allowed to differ\)", proc.stdout)
    # --allow does not cover another field
    cand2 = write_run(tmp_path, "r-cand2", conforming_result("r-cand2", K=1536, proof_class="NO_PROOF"))
    assert judge(str(base), str(cand2), "--allow", "K").returncode == 2
    assert judge(str(base), str(cand2), "--allow", "B").returncode == 2  # argparse error: not an identity field


def test_unreadable_run_is_an_error(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", conforming_result("r-base"))
    proc = judge(str(base), "r-nope", runs_dir=tmp_path)
    assert proc.returncode == 1 and "not a run under" in proc.stderr
    (tmp_path / "r-empty").mkdir()
    proc = judge(str(base), "r-empty", runs_dir=tmp_path)
    assert proc.returncode == 1 and "no result.json" in proc.stderr


# --- instances --------------------------------------------------------------------------------------------------------------------------

SHA = "f2b7cd4ca31e4adf0ce1e656bb0a2ef882f02301d5058670350b9c2b8d1f81a6"


def _with_instances(run_id: str, lo: int, hi: int, sha: str = SHA, **overrides):
    inst = contract.instances_ref("vu-k1536", lo, hi, sha)
    res = conforming_result(run_id, B=hi - lo, instances=inst, **overrides)
    for m in res["measurements"]:
        if m["name"] == "vu.count":
            m["value"] = hi - lo
        if m["name"] == "vu.seconds_per_vu":
            m["value"] = 4.0 / (hi - lo)
        if m["name"] == "vu.per_second":
            m["value"] = (hi - lo) / 4.0
    return res


def test_same_instances_are_judged(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", _with_instances("r-base", 0, 64))
    cand = write_run(tmp_path, "r-cand", _with_instances("r-cand", 0, 64))
    proc = judge(str(base), str(cand))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING" not in proc.stdout and "REFUSED" not in proc.stdout
    assert re.search(r"\n  instances\s+.*bench-instances/v1.*\s+=\s+", proc.stdout)


def test_disagreeing_instances_are_refused(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", _with_instances("r-base", 0, 64))
    # another manifest
    cand = write_run(tmp_path, "r-cand", _with_instances("r-cand", 0, 64, sha="1" * 64))
    proc = judge(str(base), str(cand))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stdout and "\n  instances: " in proc.stdout and "[measurements]" not in proc.stdout
    # other ids for the same batch size
    cand2 = write_run(tmp_path, "r-cand2", _with_instances("r-cand2", 64, 128))
    assert judge(str(base), str(cand2)).returncode == 2
    # other ids because B differs: a scaling point -- refused until B is named as the experiment (B is an identity
    # field), then judged with the range delta visible
    cand3 = write_run(tmp_path, "r-cand3", _with_instances("r-cand3", 0, 4096))
    assert judge(str(base), str(cand3)).returncode == 2
    proc = judge(str(base), str(cand3), "--allow", "B")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "instances.range" in proc.stdout  # visible in the fingerprint deltas


def test_missing_instances_warns_and_judges(tmp_path: Path) -> None:
    base = write_run(tmp_path, "r-base", _with_instances("r-base", 0, 1))
    cand = write_run(tmp_path, "r-cand", conforming_result("r-cand"))  # instances: null
    proc = judge(str(base), str(cand))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING: candidate instances is null: not measured on the shared input set" in proc.stdout
    lacking = conforming_result("r-old")
    del lacking["workload_fingerprint"]["instances"]
    old = write_run(tmp_path, "r-old", lacking)
    proc = judge(str(base), str(old))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "WARNING: candidate does not record instances" in proc.stdout
    assert "workload_fingerprint.instances missing" in proc.stdout
    rep = json.loads(judge(str(base), str(old), "--json").stdout)
    assert rep["warnings"] and rep["refused"] == []
