"""The re-baseline tool (`tests.regression.rebaseline`) and the oracle switch: `write` keeps the contract's shape and takes v2's values,
records the decision that explains a difference from the R2 record, refuses a recorded failure, keeps the v1 reference as history;
`resolver.oracle()` defaults to `expected`, `frozen` lifts from the records; `record_result` writes the document `write` reads.
Self-contained (a temporary expected/ and record dir; no store, no VERITY_REGRESSION=1).  Run from integrations/vllm:
    python -m pytest tests/regression/test_rebaseline.py -q
"""
from __future__ import annotations

import json

import pytest

from . import rebaseline as RB
from . import resolver

ROW = "gemma2-2b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-eager"       # #57: the row with labelled decisions


def _row() -> resolver.Row:
    return next(r for r in resolver.rows() if r.key == ROW)


def _result(check: str, exp: dict, act: dict, problems: list[str], *, matched: str) -> dict:
    return {"schema": "verity-vllm/regression-result/v1", "row": ROW, "check": check, "expected": exp, "actual": act,
            "problems": problems, "matched": matched, "engine": "v2", "oracle": "frozen"}


def test_oracle_default_and_values(monkeypatch):
    monkeypatch.delenv(resolver.ORACLE_ENV, raising=False)
    assert resolver.oracle() == "expected"
    monkeypatch.setenv(resolver.ORACLE_ENV, "frozen")
    assert resolver.oracle() == "frozen"
    monkeypatch.setenv(resolver.ORACLE_ENV, "v1")
    with pytest.raises(ValueError):
        resolver.oracle()
    monkeypatch.delenv(resolver.ENGINE_ENV, raising=False)
    assert resolver.engine() == resolver.DEFAULT_ENGINE == "v2" and resolver.BASELINE_ENGINE == "v1"


def test_record_result_writes_the_document_write_reads(monkeypatch, tmp_path):
    monkeypatch.setenv(resolver.RECORD_ACTUAL_ENV, str(tmp_path))
    monkeypatch.setenv(resolver.ORACLE_ENV, "frozen")
    act = {"a": 1, "source": "recomputed", "sanctioned_by": "F-x", "decision_status": "accepted"}
    p = resolver.record_result(ROW, "executed_prefix", {"a": 1}, act, [])
    assert p == tmp_path / ROW / "executed_prefix.json"
    res = RB.results(tmp_path)
    doc = res[ROW]["executed_prefix"]
    assert doc["matched"] == "decision F-x (accepted)" and doc["oracle"] == "frozen" and doc["expected"] == {"a": 1} and doc["actual"]["a"] == 1
    monkeypatch.delenv(resolver.RECORD_ACTUAL_ENV)
    assert resolver.record_result(ROW, "x", {}, {}, []) is None


def test_write_takes_v2_values_keeps_shape_and_records_the_decision(monkeypatch, tmp_path):
    row = _row()
    old = {"row": 57, "class": "FAIL", "reference": {"tag": "v1-reference", "commit": "89af", "release": "r19-int-21", "branch": "lane/vllm-integration"},
           "checks": {"manifest_digest": {"applies": True, "tier": "T0", "strength": "strong"}, "executed_prefix": {"applies": True, "tier": "T0", "strength": "strong"}},
           "manifest_digest": {"complete": False, "identities": 10, "manifest_digest": "old"},
           "executed_prefix": {"g6_lengths": {"r0": 81}}}
    dec_keys = sorted({k for d in row.decisions("manifest_digest") for k in d["expected"]})
    assert dec_keys, "#57 carries a manifest_digest decision in fixtures.toml"
    act_md = {"complete": True, "identities": 12, "manifest_digest": "new", "source": "recomputed", "sanctioned_by": "F-r19-int-20",
              "decision_status": "accepted", "beyond_contract": 1}
    for k in dec_keys:
        act_md.setdefault(k, f"v2-{k}")
    res = {"manifest_digest": _result("manifest_digest", old["manifest_digest"], act_md, [], matched="decision F-r19-int-20 (accepted)"),
           "executed_prefix": _result("executed_prefix", old["executed_prefix"], {"g6_lengths": {"r0": 81}, "source": "recomputed"}, [], matched="frozen")}
    ref = {"tag": "v2-reference", "commit": "b353ed2", "release": "v2-flip", "branch": "lane/vllm-v2-default"}
    doc, notes = RB.merge(row, old, res, reference=ref, force=False)
    md = doc["manifest_digest"]
    assert md["complete"] is True and md["identities"] == 12 and md["manifest_digest"] == "new"
    assert "beyond_contract" not in md and not (set(md) & set(RB.VOLATILE))            # shape kept: contract keys + decision keys only
    assert all(k in md for k in dec_keys)                                                 # the decision's keys join the contract
    assert doc["executed_prefix"] == {"g6_lengths": {"r0": 81}}
    assert doc["checks"]["manifest_digest"]["rebaselined"]["matched"] == "decision F-r19-int-20 (accepted)"
    assert doc["checks"]["executed_prefix"]["rebaselined"]["matched"] == "frozen"
    assert doc["reference"]["tag"] == "v2-reference" and doc["reference"]["frozen"]["tag"] == "v1-reference"
    assert any(n.startswith("coverage: no recorded result") for n in notes)              # untouched checks are listed, not dropped
    # idempotent on the reference history: a second re-baseline keeps the v1 block, not a v2 one
    doc2, _ = RB.merge(row, doc, res, reference=dict(ref, commit="c0ffee"), force=False)
    assert doc2["reference"]["commit"] == "c0ffee" and doc2["reference"]["frozen"]["tag"] == "v1-reference"


def test_write_refuses_a_recorded_failure_unless_forced():
    row = _row()
    old = {"row": 57, "class": "FAIL", "checks": {}, "executed_prefix": {"g6_lengths": {"r0": 81}}}
    res = {"executed_prefix": _result("executed_prefix", old["executed_prefix"], {"g6_lengths": {"r0": 80}}, ["$.g6_lengths.r0: 81 != 80"], matched="neither")}
    with pytest.raises(SystemExit):
        RB.merge(row, old, res, reference={"tag": "v2-reference"}, force=False)
    doc, _ = RB.merge(row, old, res, reference={"tag": "v2-reference"}, force=True)
    assert doc["executed_prefix"] == {"g6_lengths": {"r0": 80}} and doc["checks"]["executed_prefix"]["rebaselined"]["forced_over"]


def test_write_main_dry_run_touches_nothing(monkeypatch, tmp_path):
    rec = tmp_path / "rec"
    (rec / ROW).mkdir(parents=True)
    (rec / ROW / "executed_prefix.json").write_text(json.dumps(_result("executed_prefix", {"a": 1}, {"a": 2}, [], matched="expected")))
    exp_dir = tmp_path / "expected"
    exp_dir.mkdir()
    monkeypatch.setattr(resolver, "EXPECTED_DIR", exp_dir)
    rc = RB.main(["write", "--record", str(rec), "--commit", "abc", "--branch", "b", "--release", "r", "--dry-run"])
    assert rc == 0 and not list(exp_dir.iterdir())
    rc = RB.main(["write", "--record", str(rec), "--commit", "abc", "--branch", "b", "--release", "r", "--run-id", "r2026-1"])
    assert rc == 0
    out = json.load(open(exp_dir / f"{ROW}.json"))
    assert out["executed_prefix"] == {"a": 2} and out["reference"]["runs"] == ["r2026-1"] and out["row"] == 57
