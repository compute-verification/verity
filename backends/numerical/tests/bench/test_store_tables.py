"""The ledger -> store import and the store-built like-for-like table: targets, labels, idempotency, the proof-class policy, and
the row-by-row agreement with the ledger-built table (every row as a ledger-row artifact when no attempt is present)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_numerical.bench import decision_tables as DT
from verity_numerical.bench import ledger, ledger_to_store as L2S, store_tables as ST

pytest.importorskip("research.store.local", reason="research.store (tools/research) is not importable")
from research.store.local import LocalStore  # noqa: E402


def _entry(lane: str, label: str, run_id: str | None, proof_class: str = "NON_ZK_PROOF_DIAGNOSTIC", **kw) -> dict:
    return ledger.entry(lane=lane, track="B", label=label, scope="vu", proof_class=proof_class, hardware="L40S", security_target="2^-128",
                        seconds_per_vu=kw.pop("spv", 3.4e-4), B=4096, assumptions=["hash"], omitted=["auth"], note="n", run_id=run_id,
                        ts="2026-09-22T11:07:02+00:00", **kw)


def _attempt(run_id: str) -> dict:
    conds = {"hw.family": "l40s"}
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bash", "version": None}, "derivation": "drv:" + "1" * 64,
            "reuse_key": None, "argv": ["bash"], "params": {}, "key_params": [], "inputs": {}, "conditions": conds, "key_conditions": [],
            "source": {"commit": None, "tree": None, "dirty": None, "dirty_digest": None, "lockfiles": {}}, "closure": {},
            "execution": {"start_utc": None, "end_utc": None, "wall_s": None, "host": None, "project": None, "campaign": None, "run_dir": "/x",
                          "stage": None, "exclusive": None, "load_start": None, "load_end": None, "remote": None},
            "cost": None, "status": {"state": "done", "rc": 0, "class": None, "failure": None, "validation": None, "result": "absent", "timed_out": False},
            "outputs": {"result": None, "run_files": None}, "telemetry": None}


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStore:
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "norepo"))
    return LocalStore(tmp_path / "store")


def test_candidate_from_lane() -> None:
    assert [L2S.candidate(x) for x in ("a-gpu2", "b-air-v2", "soundness-fixes", "b-ligero2", "b-zk-fix", "b-encode", "c-vole", "sp1", "gate4-h100", "red-team-zk")] == \
        ["A", "B-AIR", "B-AIR", "B-Ligero", "B-Ligero", "B-Ligero", "C", "SP1", "shared", "shared"]


def test_import_targets_labels_and_idempotency(store: LocalStore, tmp_path: Path) -> None:
    d = tmp_path / "ledger"
    run = "r20260922-105905-2021"
    ledger.add(_entry("b-zk-fix", "ZK interactive: 0.340 ms/VU", run, "COMPLETE_ZK_BACKEND", breakthrough=True), d)
    ledger.add(_entry("b-zk-fix", "non-ZK control: 0.333 ms/VU", run, spv=3.33e-4), d)
    ledger.add(_entry("b-ligero", "hints: 2.30 ms/VU", None, spv=2.3e-3), d)
    store.put_attempt(_attempt(run))
    # the campaign's REDTEAM block names campaign rows; for this tiny ledger, one verdict dated by the b-ligero row
    verdict = {"by": "red-team-x", "ref": "r20260922-112029-2db3", "ts_from": ("b-ligero", "hints"),
               "rows": (("b-zk-fix", run, lambda r: r["proof_class"] == "COMPLETE_ZK_BACKEND"),), "labels": (("proof_class", "NON_ZK_PROOF_DIAGNOSTIC"),)}
    saved = L2S.REDTEAM
    L2S.REDTEAM = (verdict,)
    try:
        dry = L2S.import_ledger(store, d, dry_run=True)
        assert dry["dry_run"] and dry["on_attempts"] == 2 and dry["on_artifacts"] == 1 and dry["labels_added"] > 0 and store.labels(run) == []
        stats = L2S.import_ledger(store, d)
        again = L2S.import_ledger(store, d)
    finally:
        L2S.REDTEAM = saved
    assert stats["rows"] == 3 and stats["on_attempts"] == 2 and stats["on_artifacts"] == 1 and stats["artifacts_new"] == 1 and stats["redteam_added"] == 1
    assert again["labels_added"] == 0 and again["labels_skipped"] == stats["labels_added"] and again["redteam_added"] == 0 and again["artifacts_new"] == 0
    labs = store.labels(run, by_glob="b-zk-fix")
    refs = {lab.ref for lab in labs}
    assert refs == {L2S.ledger_ref(d / "b-zk-fix.jsonl", 1), L2S.ledger_ref(d / "b-zk-fix.jsonl", 2)}, "one ref per ledger row groups a row's labels"
    row1 = {lab.key: lab.value for lab in labs if lab.ref.endswith(":1")}
    assert row1["label"] == "ZK interactive: 0.340 ms/VU" and row1["proof_class"] == "COMPLETE_ZK_BACKEND" and row1["soundness"] == "2^-128"
    assert row1["assumptions"] == ["hash"] and row1["breakthrough"] is True and row1["B"] == 4096 and row1["candidate"] == "B-Ligero"
    assert all(lab.ts == "2026-09-22T11:07:02Z" for lab in labs), "the assertion is dated by the ledger row, not by the import"
    rt = store.labels(run, by_glob="red-team-*")
    assert [(l.key, l.value, l.ref) for l in rt] == [("proof_class", "NON_ZK_PROOF_DIAGNOSTIC", "r20260922-112029-2db3")]
    art = stats["targets"][L2S.ledger_ref(d / "b-ligero.jsonl", 1)]
    m = store.get_manifest(art)
    assert m.kind == "ledger-row/v1" and m.meta["lane"] == "b-ligero" and m.meta["run_id"] is None and m.payload is None
    assert {lab.key for lab in store.labels(art)} >= {"label", "proof_class", "hardware", "soundness", "candidate", "seconds_per_vu", "overhead"}
    # store_tables reads the same rows back
    idx = store.index
    assert ST.resolve_target(idx, "b-zk-fix", run, "ZK interactive") == run and ST.resolve_target(idx, "b-ligero", None, "hints") == art
    with pytest.raises(LookupError):
        ST.resolve_target(idx, "b-ligero", None, "no such row")
    asserted = ST._assertions(idx, run)
    row = ST.lane_row(asserted, "b-zk-fix", "ZK interactive")
    assert row["proof_class"] == "COMPLETE_ZK_BACKEND" and row["seconds_per_vu"] == pytest.approx(3.4e-4) and row["B"] == 4096
    assert ST.policy_proof_class(asserted, "b-zk-fix", "COMPLETE_ZK_BACKEND", None)[0] == "COMPLETE_ZK_BACKEND"
    assert ST.policy_proof_class(asserted, "b-zk-fix", "COMPLETE_ZK_BACKEND", "red-team-*")[0] == "NON_ZK_PROOF_DIAGNOSTIC", "downgrades stand"
    assert ST.policy_proof_class(asserted, "b-zk-fix", "NON_ZK_PROOF_DIAGNOSTIC", "red-team-*")[0] == "NON_ZK_PROOF_DIAGNOSTIC", "nothing is upgraded"
    store.add_label(run, "proof_class", "COMPLETE_ZK_BACKEND", "red-team-y", ref="r20260922-120000-0000", ts="2026-09-22T12:00:00Z")
    asserted = ST._assertions(idx, run)
    assert ST.policy_proof_class(asserted, "b-zk-fix", "COMPLETE_ZK_BACKEND", "red-team-*")[0] == "COMPLETE_ZK_BACKEND", "the latest audit event wins"
    assert ST.policy_proof_class(asserted, "b-zk-fix", "COMPLETE_ZK_BACKEND", "red-team-x")[0] == "NON_ZK_PROOF_DIAGNOSTIC", "policy picks the asserter"


def test_ts_and_ref_helpers(tmp_path: Path) -> None:
    assert L2S.ts_z("2026-09-22T09:20:19+00:00") == "2026-09-22T09:20:19Z" and L2S.ts_z("2026-09-22T11:20:19+02:00") == "2026-09-22T09:20:19Z"
    assert L2S.ledger_ref(ledger.DIR / "b-zk.jsonl", 4) == "backends/numerical/reports/ledger/b-zk.jsonl:4"
    assert L2S.ledger_ref(tmp_path / "x.jsonl", 2) == f"{tmp_path.resolve().as_posix()}/x.jsonl:2"


def test_diff_tables_compares_label_by_class() -> None:
    a = "| design | label | lane / run id |\n|---|---|---|\n| d | NON_ZK_PROOF_DIAGNOSTIC (red-team-zk r1; lane says X) | l / r |\n| e | X | l / r |"
    b = "| design | label | lane / run id |\n|---|---|---|\n| d | NON_ZK_PROOF_DIAGNOSTIC (hand-written prose) | l / r |\n| e | Y | l / r |"
    ok, n, diffs = ST.diff_tables(a, b)
    assert (ok, n) == (1, 2) and len(diffs) == 1 and "[label]: store='X' ledger='Y'" in diffs[0]


@pytest.mark.skipif(not ledger.DIR.is_dir(), reason="campaign ledger not present")
def test_campaign_ledger_round_trips_through_a_fresh_store(store: LocalStore) -> None:
    """No attempts at all: every row is a ledger-row artifact, the red-team verdicts land on those, and the like-for-like table
    rebuilt from the store matches decision_tables' ledger-built table row for row under the red-team policy -- and differs from it
    exactly on the rows the campaign carried as label overrides without the policy."""
    stats = L2S.import_ledger(store, ledger.DIR)
    entries = DT._entries(ledger.DIR)
    assert stats["rows"] == len(entries) and stats["on_attempts"] == 0 and stats["on_artifacts"] == len(entries) and stats["redteam_added"] > 0
    assert L2S.import_ledger(store, ledger.DIR)["labels_added"] == 0
    theirs = DT.render_like_for_like(entries)
    ok, n, diffs = ST.diff_tables(ST.render_like_for_like(store.index, "red-team-*"), theirs)
    assert (ok, n, diffs) == (len(DT.LIKE_FOR_LIKE), len(DT.LIKE_FOR_LIKE), [])
    ok2, _, diffs2 = ST.diff_tables(ST.render_like_for_like(store.index, None), theirs)
    overridden = sum(1 for r in DT.LIKE_FOR_LIKE if r.label_override)
    assert ok2 == n - overridden and len(diffs2) == overridden and all("[label]" in d for d in diffs2)


def test_dry_run_writes_nothing(store: LocalStore, tmp_path: Path) -> None:
    d = tmp_path / "ledger"
    ledger.add(_entry("b-ligero", "x", None), d)
    saved = L2S.REDTEAM
    L2S.REDTEAM = ()
    try:
        L2S.import_ledger(store, d, dry_run=True)
    finally:
        L2S.REDTEAM = saved
    assert store.artifact_ids() == [] and not any((store.root / "labels").iterdir())
    assert json.loads(json.dumps(L2S.row_labels(_entry("b-ligero", "x", None))))["candidate"] == "B-Ligero"
