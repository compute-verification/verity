"""``benchmarks/judge.py``: refuses across identity fields, judges otherwise, and reads the recorded runs that exist."""

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

K16_RUN = "r20260921-215829-9796"     # SP1 core, K=16, recorded 2026-09-21 before the contract existed
K1536_RUN = "r20260921-220722-019b"   # SP1 core, K=1536, same


def judge(*args: str, runs_dir: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(JUDGE), *args]
    if runs_dir is not None:
        cmd += ["--runs-dir", str(runs_dir)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, env={**os.environ})


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


def _recorded(run_id: str) -> Path | None:
    root = Path(os.environ.get("RESEARCH_RUNS") or Path.home() / ".research" / "runs")
    d = root / run_id
    return d if (d / "result.json").is_file() else None


@pytest.mark.skipif(not (_recorded(K16_RUN) and _recorded(K1536_RUN)), reason="the recorded K=16 / K=1536 SP1 runs are not fetched on this machine")
def test_the_recorded_k16_and_k1536_runs_differ_in_k() -> None:
    proc = judge(K16_RUN, K1536_RUN)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "K: 16 vs 1536" in proc.stdout
    assert "derived from fixtures/typed-obligation-v0/accept--packed-word-row.json" in proc.stdout, "pre-contract runs: the identity is derived and said to be"
    for field in contract.REQUIRED_FIELDS:
        assert f"workload_fingerprint.{field} missing" in proc.stdout
    proc = judge(K16_RUN, K1536_RUN, "--allow", "K")
    assert proc.returncode == 2, "pre-contract runs recorded neither hardware nor a soundness target: still refused with only K allowed"
    assert "hardware: <missing> vs <missing>" in proc.stdout
    legacy_allow = ("--allow", "K", "--allow", "hardware", "--allow", "security.target")
    proc = judge(K16_RUN, K1536_RUN, *legacy_allow)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "prove_seconds" in proc.stdout and "(allowed to differ)" in proc.stdout
    rep = json.loads(judge(K16_RUN, K1536_RUN, *legacy_allow, "--json").stdout)
    ratios = {m["name"]: m["ratio"] for m in rep["measurements"]}
    assert ratios["shards"] == 2.0 and ratios["prove_seconds"] == pytest.approx(3.580119322 / 0.892078765)
    assert rep["derived"]["baseline"]["fields"] == ["B", "K", "authentication", "domain", "profile", "proof_class"]
