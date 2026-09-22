"""The checker sweep over bench-instances/v1: the committed negatives tier is always available; the positives' operand
arrays are built, not committed, so those tiers run only where they exist (the pod run is the record)."""

import json
from pathlib import Path

import pytest
from verity_numerical.bench import instances as I
from verity_numerical.checker import sweep as S

ROOT = I.DEFAULT_ROOT


def test_negatives_tier_all_behave() -> None:
    rep = S.sweep_tier("vu-k1536-neg", ROOT, procs=1, log=lambda *_: None)
    assert rep["checked"] == rep["of"] == 52 and rep["bad"] == 0, rep["failures"][:5]
    assert set(rep["by_mistake"]) >= {"ulp-plus", "c0-stale", "reversed-k", "nonfinite-operand", "sat-intermediate"}
    assert all(d["bad"] == 0 for d in rep["by_mistake"].values())


@pytest.mark.parametrize("tier", ["vu-k1536", "tu-k16"])
def test_positive_tiers_when_built(tier: str) -> None:
    try:
        rep = S.sweep_tier(tier, ROOT, limit=8, procs=1, log=lambda *_: None)
    except FileNotFoundError:
        pytest.skip(f"{tier} operand arrays are not built here")
    assert rep["checked"] == 8 and rep["bad"] == 0, rep["failures"]


def test_cli_writes_report_and_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    rc = S.main(["--tiers", "vu-k1536-neg", "--limit", "4", "--procs", "1"])
    assert rc == 0
    doc = json.loads((tmp_path / "checker_sweep.json").read_text())
    assert doc["complete"] is False and doc["passed"] is False, "a --limit run is not the record"
    env = json.loads((tmp_path / "result.json").read_text())
    assert env["validation"]["status"] == "not_run" and env["run_id"] == "r-test"
    assert {m["name"] for m in env["measurements"]} == {"vu-k1536-neg.checked", "vu-k1536-neg.bad", "vu-k1536-neg.seconds"}
