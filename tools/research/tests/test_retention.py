"""research.retention: the `research/retention/v0` record, implicit records from `.custody`, the report, and gc's default policy.
Pure Python over tmp trees; the clock, the disk, the remote check and the in-use check are fakes (an fs remote for the real check)."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research import retention as R
from research.cli import main as research_main
from research.store import custody as CU
from research.store.local import LocalStore

NOW = datetime(2026, 10, 1, 19, 30, tzinfo=timezone.utc)
PAST, FUTURE = "2026-09-25T00:00:00Z", "2026-10-08T00:00:00Z"
NODE = "vy-test"
ART = "art:" + "a" * 64
UNIT = 50_000
VERIFIED_FROM = 4102444800.0          # after every mtime a test can write, so the marker vouches for the files it lists


def clock() -> datetime:
    return NOW


def data(root: Path, rel: str, size: int = UNIT) -> Path:
    """A directory `rel` holding one file of `size` bytes."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "blob.bin").write_bytes(b"x" * size)
    return d


def kept(root: Path, rel: str, *, priority: str = "low", expires: str | None = PAST, regenerate: str | None = "make it",
         preserved: str | None = None, node: str = NODE, size: int = UNIT, owner: str = "@compute-accounting") -> Path:
    d = data(root, rel, size)
    R.keep(d, root=root, owner=owner, expires=expires, priority=priority, regenerate=regenerate, preserved=preserved, now=clock, node=node)
    return d


def run_dir(root: Path, rel: str, *, preserved: bool = True, lane: str | None = None, campaign: str | None = None,
            created: str | None = "2026-09-20T10:00:00.123Z", verified: str = "2026-09-21T00:00:00Z", host: str = NODE,
            extra: dict[str, bytes] | None = None) -> Path:
    """A run dir with job.json, a log, the `extra` files, and the `.custody` marker the runner writes (store.custody)."""
    d = root / rel
    d.mkdir(parents=True)
    job = {"ownership": {"lane": lane}, "store": {"campaign": campaign}, "run": {"created_utc": created}}
    (d / "job.json").write_text(json.dumps(job))
    (d / "stdout.log").write_bytes(b"y" * UNIT)
    for name, body in (extra or {}).items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(body)
    files = {r: [(d / r).stat().st_size, hashlib.sha256((d / r).read_bytes()).hexdigest()] for r in CU.run_files(d)}
    (d / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": d.name, "run_record": ART, "preserved": preserved,
                                           "verified_from": VERIFIED_FROM, "verified_utc": verified, "files": files, "host": host}))
    return d


class Disk:
    """shutil.disk_usage over a filesystem of `total` bytes holding `base` bytes besides the watched paths, each UNIT bytes while it
    exists: deleting one frees UNIT."""

    def __init__(self, total: int, base: int, watched: list[Path]):
        self.total, self.base, self.watched = total, base, watched

    def __call__(self, _path: Path) -> tuple[int, int, int]:
        used = self.base + UNIT * sum(1 for p in self.watched if p.exists())
        return self.total, used, self.total - used


def never_in_use(_path: Path) -> None:
    return None


def preserved_ok(_art: str, _path: Path) -> None:
    return None


def run_gc(root: Path, disk: Disk, **kw):
    kw = {"now": clock, "usage": disk, "remote_check": preserved_ok, "in_use": never_in_use, "node": NODE, **kw}
    return R.gc(root, **kw)


def by_path(plan: dict) -> dict[str, dict]:
    return {a["path"]: a for a in plan["actions"]}


# ---- the record -----------------------------------------------------------------------------------------------------------------------

def test_a_record_round_trips_through_its_sidecar_and_an_update_keeps_created(tmp_path):
    d = data(tmp_path, "pouw/mvp-e2e/passes/job-7")
    rec = R.keep(d, root=tmp_path, owner="@compute-accounting", expires=FUTURE, priority="low", regenerate="research run ... at abc",
                 why="job B's passes", by="https://cursor.com/agents/bc-1", now=clock, node=NODE)
    assert (d / R.SIDECAR).is_file() and sorted(p.name for p in d.iterdir()) == [R.SIDECAR, "blob.bin"], "atomic: no temp file left"
    assert list(rec) == list(R.FIELDS)
    assert rec == {"schema": R.SCHEMA, "path": "pouw/mvp-e2e/passes/job-7", "node": NODE, "owner": "@compute-accounting",
                   "by": "https://cursor.com/agents/bc-1", "created": "2026-10-01T19:30:00Z", "expires": FUTURE, "priority": "low",
                   "preserved": None, "regenerate": "research run ... at abc", "why": "job B's passes"}
    assert R.load(d) == rec and json.loads((d / R.SIDECAR).read_text()) == rec

    later = datetime(2026, 10, 2, tzinfo=timezone.utc)
    upd = R.keep(d, root=tmp_path, owner="@infra", expires=None, priority="keep", preserved=ART, regenerate="", now=lambda: later, node=NODE)
    assert upd["created"] == "2026-10-01T19:30:00Z", "an update keeps created"
    assert (upd["owner"], upd["expires"], upd["priority"], upd["preserved"]) == ("@infra", None, "keep", ART)
    assert upd["regenerate"] is None, "an empty option clears its field"
    assert upd["why"] == "job B's passes" and upd["by"] == "https://cursor.com/agents/bc-1", "an option not given keeps its value"


def test_a_file_records_beside_itself(tmp_path):
    f = tmp_path / "hf" / "model.safetensors"
    f.parent.mkdir()
    f.write_bytes(b"w" * 10)
    R.keep(f, root=tmp_path, owner="@infra", expires=None, priority="normal", regenerate="huggingface-cli download m", now=clock, node=NODE)
    assert R.sidecar(f) == tmp_path / "hf" / "model.safetensors.retention.json" and R.sidecar(f).is_file()
    assert R.load(f)["path"] == "hf/model.safetensors"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into every directory")
def test_a_read_only_directory_records_beside_itself_and_gc_leaves_it_whole(tmp_path):
    """compute-accounting's dr-xr-xr-x cancel-pair units (1 Oct): keep failed on `.retention.json.tmp` inside them."""
    d = data(tmp_path, "pouw/gpu3-fp8/cancel-pair@flat@1")
    d.chmod(0o555)
    try:
        R.keep(d, root=tmp_path, owner="@compute-accounting", expires=PAST, priority="low", regenerate="make it", now=clock, node=NODE)
        sc = tmp_path / "pouw/gpu3-fp8/cancel-pair@flat@1.retention.json"
        assert R.sidecar(d) == sc and sc.is_file() and not (d / R.SIDECAR).exists()
        assert R.find(d / "blob.bin", tmp_path).record["priority"] == "low"
        rows = {r["path"]: r for r in R.report(tmp_path, now=clock, usage=lambda _p: (10, 5, 5), node=NODE)["rows"]}
        assert rows["pouw/gpu3-fp8/cancel-pair@flat@1"]["state"] == "expired"
        plan = run_gc(tmp_path, Disk(100 * UNIT, 99 * UNIT, [d]), apply=True, force=True, low_pct=0)
        assert "is read-only" in by_path(plan)["pouw/gpu3-fp8/cancel-pair@flat@1"]["error"] and (d / "blob.bin").exists() and sc.exists()
    finally:
        d.chmod(0o755)
    plan = run_gc(tmp_path, Disk(100 * UNIT, 99 * UNIT, [d]), apply=True, force=True, low_pct=0)
    assert by_path(plan)["pouw/gpu3-fp8/cancel-pair@flat@1"]["error"] is None and not d.exists() and not sc.exists()


def test_a_directory_with_a_record_inside_and_beside_it_is_invalid(tmp_path):
    d = kept(tmp_path, "pouw/twice")
    R.beside(d).write_text((d / R.SIDECAR).read_text())
    with pytest.raises(R.RecordError, match="both record"):
        R.load(d)
    plan = run_gc(tmp_path, Disk(100 * UNIT, 99 * UNIT, [d]), apply=True, force=True, low_pct=0)
    assert by_path(plan)["pouw/twice"]["action"] == "skip" and d.exists()


@pytest.mark.parametrize("change, fix", [
    ({"owner": "compute-accounting"}, "lane handle such as @compute-accounting"),
    ({"priority": "urgent"}, "one of low, normal, high, keep"),
    ({"expires": "next week"}, "UTC time"),
    ({"expires": "2026-10-08T00:00:00"}, "UTC time"),
    ({"created": None}, "created must be a UTC time"),
    ({"preserved": "art:abc"}, "full art: id"),
    ({"regenerate": " "}, "command that rebuilds"),
    ({"path": "/workspace/pouw/x"}, "relative to the root"),
    ({"path": "pouw/../x"}, "'..'"),
    ({"schema": "research/retention/v1"}, "schema must be"),
    ({"expiry": FUTURE}, "unknown field(s) expiry"),
])
def test_validation_rejects_a_bad_field_and_says_how_to_fix_it(tmp_path, change, fix):
    d = kept(tmp_path, "pouw/out", expires=FUTURE)
    doc = {**R.load(d), **change}
    assert any(fix in p for p in R.problems(doc)), R.problems(doc)
    (d / R.SIDECAR).write_text(json.dumps(doc))
    with pytest.raises(R.RecordError, match="retention.json"):
        R.load(d)
    with pytest.raises(R.RecordError):
        R.write(d, doc)


def test_validation_names_a_missing_field_and_an_unreadable_sidecar(tmp_path):
    d = kept(tmp_path, "pouw/out")
    doc = R.load(d)
    del doc["why"]
    assert R.problems(doc) == ["missing why (every field is present, null where it has no value)"]
    (d / R.SIDECAR).write_text("{not json")
    with pytest.raises(R.RecordError, match="unreadable .*research keep"):
        R.load(d)


def test_keep_refuses_the_root_a_sidecar_a_missing_path_and_a_path_outside_the_root(tmp_path):
    root = tmp_path / "ws"
    d = kept(root, "pouw/out")
    args = {"owner": "@x", "expires": None, "priority": "low", "now": clock, "node": NODE}
    with pytest.raises(R.RecordError, match="never the root itself"):
        R.keep(root, root=root, **args)
    with pytest.raises(R.RecordError, match="is a record, not data"):
        R.keep(d / R.SIDECAR, root=root, **args)
    with pytest.raises(R.RecordError, match="does not exist"):
        R.keep(root / "absent", root=root, **args)
    with pytest.raises(R.RecordError, match="not under the root"):
        R.keep(data(tmp_path, "other"), root=root, **args)


# ---- implicit records ---------------------------------------------------------------------------------------------------------------

def test_a_preserved_custody_marker_is_an_implicit_record_owned_by_the_lane_then_the_campaign_then_runner(tmp_path):
    a = run_dir(tmp_path, "research/runs/r1", lane="compute-accounting", campaign="pouw")
    found = R.record_at(a, "research/runs/r1")
    assert found.source == "custody" and not (a / R.SIDECAR).exists(), "nothing is written for an implicit record"
    assert found.record == {**found.record, "schema": R.SCHEMA, "path": "research/runs/r1", "node": NODE, "owner": "compute-accounting",
                            "created": "2026-09-20T10:00:00Z", "expires": "2026-09-23T00:00:00Z", "priority": "normal",
                            "preserved": ART, "regenerate": None}
    assert R.record_at(run_dir(tmp_path, "research/runs/r2", campaign="pouw"), "research/runs/r2").record["owner"] == "pouw"
    r3 = R.record_at(run_dir(tmp_path, "research/runs/r3", created=None), "research/runs/r3").record
    assert (r3["owner"], r3["created"]) == ("runner", "2026-09-21T00:00:00Z"), "no lane or campaign: runner; no created_utc: verified_utc"


def test_an_unpreserved_marker_gives_no_record_and_a_sidecar_wins_over_custody(tmp_path):
    run_dir(tmp_path, "research/runs/r1", preserved=False)
    assert R.record_at(tmp_path / "research/runs/r1", "research/runs/r1") is None
    rows = {r.path: r for r in R.scan(tmp_path).rows}
    assert list(rows) == ["research"] and rows["research"].record is None and rows["research"].bytes > UNIT, "its area is unowned"

    b = run_dir(tmp_path, "research/runs/r2")
    R.keep(b, root=tmp_path, owner="@infra", expires=None, priority="keep", now=clock, node=NODE)
    found = R.record_at(b, "research/runs/r2")
    assert (found.source, found.record["priority"]) == ("sidecar", "keep")


# ---- the report -----------------------------------------------------------------------------------------------------------------------

def test_the_report_lists_unowned_and_expired_first_with_totals_and_never_descends_into_a_record(tmp_path):
    data(tmp_path, "cache/uv", 3 * UNIT)                                        # unowned, top level
    kept(tmp_path, "pouw/expired-small", expires=PAST, size=UNIT // 2)
    big = kept(tmp_path, "pouw/expired-big", expires=PAST, size=2 * UNIT)
    kept(big, "inner", priority="keep", expires=None, size=10)               # a record inside a recorded path: covered, not listed
    kept(tmp_path, "pouw/kept", expires=FUTURE, size=4 * UNIT)
    data(tmp_path, "pouw/fill-out", UNIT)                                      # unowned, second level
    (tmp_path / "pouw" / "loose.txt").write_bytes(b"z" * 7)
    run_dir(tmp_path, "research/runs/r1")
    data(tmp_path, "research/runs/unrecorded", 2 * UNIT)                       # third level: in research/runs's unrecorded rest

    rep = R.report(tmp_path, now=clock, usage=lambda _p: (1000 * UNIT, 820 * UNIT, 180 * UNIT), node=NODE)
    rows = [(d["state"], d["path"]) for d in rep["rows"]]
    assert rows == [("unowned", "cache"), ("unowned", "research/runs"), ("unowned", "pouw/fill-out"), ("unowned", "pouw"),
                    ("expired", "pouw/expired-big"), ("expired", "research/runs/r1"), ("expired", "pouw/expired-small"),
                    ("recorded", "pouw/kept")]
    docs = {d["path"]: d for d in rep["rows"]}
    assert docs["pouw"]["rest"] and docs["pouw"]["bytes"] == 7 and docs["research/runs"]["rest"]
    assert docs["cache"]["bytes"] == 3 * UNIT and not docs["cache"]["rest"]
    assert not any(p.startswith("pouw/expired-big/") for _, p in rows), "a record covers everything beneath it"
    assert docs["pouw/expired-big"]["nested"] == ["pouw/expired-big/inner/.retention.json"]
    assert docs["pouw/expired-big"]["bytes"] > 2 * UNIT and docs["research/runs/r1"]["source"] == "custody"
    assert (docs["pouw/kept"]["regenerable"], docs["research/runs/r1"]["preserved"], docs["pouw/kept"]["expired"]) == (True, ART, False)
    t = rep["totals"]
    assert (t["unowned"]["paths"], t["expired"]["paths"], t["recorded"]["paths"], t["collectable"]["paths"]) == (4, 3, 1, 3)
    assert t["unowned"]["bytes"] == sum(docs[p]["bytes"] for p in ("cache", "research/runs", "pouw/fill-out", "pouw"))
    assert rep["filesystem"]["pct"] == 82.0


def test_the_report_marks_an_invalid_sidecar_and_keeps_it_out_of_unowned(tmp_path):
    d = kept(tmp_path, "pouw/bad")
    (d / R.SIDECAR).write_text(json.dumps({**R.load(d), "priority": "urgent"}))
    (row,) = [r for r in R.report(tmp_path, now=clock, usage=lambda _p: (10, 5, 5))["rows"] if r["path"] == "pouw/bad"]
    assert row["state"] == "invalid" and "priority must be one of" in row["error"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_a_directory_the_report_cannot_read_is_unknown_not_a_record_to_rewrite(tmp_path):
    """Node 2's root-only lost+found (1 Oct): the report told the user to rewrite a record that doesn't exist."""
    locked = data(tmp_path, "lost+found")
    locked.chmod(0)
    try:
        (row,) = [r for r in R.report(tmp_path, now=clock, usage=lambda _p: (10, 5, 5))["rows"] if r["path"] == "lost+found"]
    finally:
        locked.chmod(0o755)
    assert row["state"] == "invalid" and "whether it has a record is unknown" in row["error"] and "research keep" not in row["error"]


# ---- gc: what it never deletes --------------------------------------------------------------------------------------------------------

def test_gc_deletes_only_what_a_valid_expired_collectable_recoverable_unused_record_allows(tmp_path):
    root = tmp_path / "ws"
    queue = tmp_path / "fill" / "queue"
    queue.mkdir(parents=True)
    ok = kept(root, "pouw/ok")
    keep_ = kept(root, "pouw/keep", priority="keep")
    high = kept(root, "pouw/high", priority="high")
    fresh = kept(root, "pouw/fresh", expires=FUTURE)
    never = kept(root, "pouw/never", expires=None)
    lost = kept(root, "pouw/lost", regenerate=None)
    unverified = kept(root, "pouw/unverified", regenerate=None, preserved=ART)
    regen_unverified = kept(root, "pouw/regen-unverified", preserved=ART)
    named = kept(root, "pouw/named")
    elsewhere = kept(root, "pouw/elsewhere", node="vy-other")
    parent = kept(root, "pouw/parent")
    kept(parent, "child", priority="keep", expires=None, size=10)
    unowned = data(root, "pouw/unowned")
    invalid = kept(root, "pouw/invalid")
    (invalid / R.SIDECAR).write_text(json.dumps({**R.load(invalid), "owner": "nobody"}))
    unpreserved_run = run_dir(root, "research/runs/r-unpreserved", preserved=False)
    changed_run = run_dir(root, "research/runs/r-changed")
    (changed_run / "late.txt").write_text("written after custody")
    (queue / "job-1.sh").write_text(f"#!/bin/bash\n# fill: owner=bc-1\npython verify.py --passes {root}/pouw/named/blob.bin\n")
    checked = []

    def remote(art: str, path: Path) -> str | None:
        checked.append(path.name)
        return "objects/sha256/…: absent"

    plan = run_gc(root, Disk(100 * UNIT, 99 * UNIT, []), apply=True, force=True, low_pct=0, remote_check=remote,
                  in_use=R.JobScripts([queue], root))
    acts = by_path(plan)
    assert [p for p, a in acts.items() if a["action"] == "delete"] == ["pouw/ok"] and not ok.exists()
    for d in (keep_, high, fresh, never, lost, unverified, regen_unverified, named, elsewhere, parent, unowned, invalid,
              unpreserved_run, changed_run):
        assert d.exists(), d
    reasons = {p: a["reason"] for p, a in acts.items()}
    assert reasons["pouw/keep"] == "priority keep: never collected" and reasons["pouw/high"] == "priority high: never collected"
    assert reasons["pouw/fresh"] == f"not expired: expires {FUTURE}" and reasons["pouw/never"] == "never expires"
    assert reasons["pouw/lost"].startswith("neither preserved nor regenerable")
    assert reasons["pouw/unverified"] == f"preserved {ART} is not verified on the remote: objects/sha256/…: absent"
    assert reasons["pouw/regen-unverified"].startswith(f"preserved {ART} is not verified"), "a failed verification is never waived"
    assert reasons["pouw/named"] == f"in use: {queue / 'job-1.sh'} names it"
    assert reasons["pouw/elsewhere"].startswith("recorded on vy-other, and this is vy-test")
    assert reasons["pouw/parent"].startswith("holds 1 nested record(s), e.g. pouw/parent/child/.retention.json")
    assert reasons["pouw/unowned"].startswith("no record: a path without one is never deleted")
    assert reasons["pouw/invalid"].startswith("invalid record:") and "lane handle" in reasons["pouw/invalid"]
    assert reasons["research/runs"].startswith("no record"), "an unpreserved run dir is unowned"
    assert reasons["research/runs/r-changed"] == "its custody marker no longer vouches for it: late.txt is not in the run record (added after custody)"
    assert sorted(checked) == ["regen-unverified", "unverified"], "the remote is checked just before a deletion, and only then"


def test_gc_does_nothing_unless_the_filesystem_is_over_high_pct_or_forced(tmp_path):
    a = kept(tmp_path, "pouw/a")
    disk = Disk(100 * UNIT, 79 * UNIT, [a])                                   # 80%: not over 80
    plan = run_gc(tmp_path, disk, apply=True)
    assert (plan["acting"], plan["actions"], plan["pct_before"]) == (False, [], 80.0) and a.exists()
    plan = run_gc(tmp_path, disk, apply=True, force=True, low_pct=0)
    assert plan["acting"] and by_path(plan)["pouw/a"]["action"] == "delete" and not a.exists()


def test_gc_deletes_low_before_normal_then_the_oldest_expiry_and_stops_under_low_pct(tmp_path):
    normal_old = kept(tmp_path, "pouw/normal-old", priority="normal", expires="2026-09-01T00:00:00Z")
    low_new = kept(tmp_path, "pouw/low-new", expires="2026-09-30T00:00:00Z")
    low_old = kept(tmp_path, "pouw/low-old", expires="2026-09-10T00:00:00Z")
    normal_new = kept(tmp_path, "pouw/normal-new", priority="normal", expires="2026-09-29T00:00:00Z")
    disk = Disk(20 * UNIT, 14 * UNIT, [normal_old, low_new, low_old, normal_new])   # 90%; each deletion frees 5%
    plan = run_gc(tmp_path, disk, apply=True, low_pct=78)
    acts = [(a["path"], a["action"], a["pct_after"]) for a in plan["actions"] if a["path"].startswith("pouw/")]
    assert acts == [("pouw/low-old", "delete", 85.0), ("pouw/low-new", "delete", 80.0), ("pouw/normal-old", "delete", 75.0),
                    ("pouw/normal-new", "skip", None)]
    assert by_path(plan)["pouw/normal-new"]["reason"] == "not needed: the filesystem is at 75.0%, under 78%"
    assert [d.exists() for d in (low_old, low_new, normal_old, normal_new)] == [False, False, False, True]
    assert (plan["pct_after"], plan["freed_bytes"] >= 3 * UNIT) == (75.0, True)


def test_a_dry_run_deletes_nothing_and_reads_no_remote(tmp_path):
    a = kept(tmp_path, "pouw/a", preserved=ART, regenerate=None)
    b = kept(tmp_path, "pouw/b")
    run_dir(tmp_path, "research/runs/r1")

    def remote(_art: str, _path: Path) -> str | None:
        raise AssertionError("a dry run reads no remote")

    plan = run_gc(tmp_path, Disk(20 * UNIT, 15 * UNIT, [a, b]), remote_check=remote, low_pct=0)
    deletes = [x for x in plan["actions"] if x["action"] == "delete"]
    assert {x["path"] for x in deletes} == {"pouw/a", "pouw/b", "research/runs/r1"} and not plan["apply"]
    assert all(x["reason"].startswith("would delete: ") for x in deletes)
    assert "re-verified on the remote just before an --apply deletes it" in by_path(plan)["pouw/a"]["reason"]
    assert a.exists() and b.exists() and (tmp_path / "research/runs/r1").exists()


def test_gc_deletes_a_recorded_file_with_its_sidecar_and_an_expired_implicit_run_dir(tmp_path):
    f = tmp_path / "jobs" / "bitset.bin"
    f.parent.mkdir()
    f.write_bytes(b"b" * UNIT)
    R.keep(f, root=tmp_path, owner="@pouw", expires=PAST, priority="low", regenerate="python bitset.py", now=clock, node=NODE)
    r1 = run_dir(tmp_path, "research/runs/r1")
    seen = []
    plan = run_gc(tmp_path, Disk(20 * UNIT, 18 * UNIT, []), apply=True, force=True, low_pct=0,
                  remote_check=lambda art, p: seen.append((art, p)) or None)
    assert not f.exists() and not R.sidecar(f).exists() and not r1.exists()
    assert seen == [(ART, r1)], "the run record is what is re-verified"
    assert by_path(plan)["research/runs/r1"]["reason"].startswith("deleted: normal, expired 2026-09-23T00:00:00Z, preserved art:")


def test_gc_skips_a_path_a_job_queued_during_gc_names(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    a = kept(tmp_path / "ws", "pouw/a")
    jobs = R.JobScripts([queue], tmp_path / "ws")
    calls = []

    def in_use(p: Path) -> str | None:
        calls.append(p)
        if len(calls) == 2:                                                    # between the qualification and the deletion
            (queue / "late.sh").write_text("cat pouw/a/blob.bin\n")
        return jobs(p)

    plan = run_gc(tmp_path / "ws", Disk(20 * UNIT, 15 * UNIT, [a]), apply=True, force=True, low_pct=0, in_use=in_use)
    assert a.exists() and by_path(plan)["pouw/a"]["reason"] == f"in use: {queue / 'late.sh'} names it"


# ---- the default checks ---------------------------------------------------------------------------------------------------------------

def test_job_scripts_match_the_path_or_beneath_it_never_a_longer_name_and_fail_closed(tmp_path):
    root = tmp_path / "ws"
    q = tmp_path / "queue"
    q.mkdir()
    (q / "a.sh").write_text(f"OUT={root}/pouw/out/job-70\nINPUT=pouw/out/job-8/part-1.bin\n")
    jobs = R.JobScripts([q], root)
    assert jobs(root / "pouw/out/job-7") is None, "job-70 is another path"
    assert jobs(root / "pouw/out/job-70") == f"{q / 'a.sh'} names it"
    assert jobs(root / "pouw/out/job-8") == f"{q / 'a.sh'} names it", "a file beneath the path names it too"
    assert R.JobScripts([tmp_path / "absent"], root)(root / "pouw/out/job-7").startswith("cannot read the job scripts")


@pytest.fixture
def vault_store(tmp_path, monkeypatch) -> Path:
    """An evidence store whose remote is a directory (remote.FsRemote, the store's test double)."""
    cfg = tmp_path / "store.toml"
    cfg.write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def test_the_default_check_needs_the_artifact_to_hold_the_paths_files_and_be_preserved(tmp_path, vault_store):
    root = tmp_path / "ws"
    d = data(root, "research/runs/r1")
    st = LocalStore(vault_store)
    art = st.put_artifact("run-record/v1", {"run_id": "r1"}, tree=d)
    assert st.push(art, preserve=True)["preserved"]
    (d / CU.MARKER).write_text("{}")                                          # custody's marker and the sidecar are not data
    R.keep(d, root=root, owner="@x", expires=PAST, priority="low", preserved=art, now=clock, node=NODE)
    check = R.store_check(str(vault_store))
    assert check(art, d) is None
    (d / "late.bin").write_bytes(b"l")
    assert check(art, d) == "1 file(s) under it are not in the artifact at their size, e.g. late.bin"
    (d / "late.bin").unlink()
    (d / "blob.bin").write_bytes(b"x" * (UNIT - 1))
    assert "e.g. blob.bin" in check(art, d)
    (d / "blob.bin").write_bytes(b"x" * UNIT)
    blob = hashlib.sha256(b"x" * UNIT).hexdigest()
    (tmp_path / "vault" / "objects" / "sha256" / blob).unlink()
    assert check(art, d) == f"objects/sha256/{blob}: absent"
    assert R.store_check(str(vault_store))("art:" + "f" * 64, d).startswith("NotFound")


def test_a_run_dir_whose_record_custody_trimmed_is_never_deleted(tmp_path, vault_store, monkeypatch):
    """Custody marks a run dir preserved while its run record only lists an undeclared file over the cap (custody.select); those
    bytes are on no remote, so gc keeps the dir."""
    root = tmp_path / "ws"
    d = run_dir(root, "research/runs/r1", extra={"work/big.bin": b"t" * UNIT})
    monkeypatch.setattr(CU, "UNDECLARED_CAP_BYTES", 0)
    st = LocalStore(vault_store)
    art = CU.put_run_record(st, d, "r1")
    assert st.push(art, preserve=True)["preserved"] and [t["path"] for t in CU.trimmed_of(st, art)] == ["work/big.bin"]
    (d / CU.MARKER).write_text(json.dumps({**json.loads((d / CU.MARKER).read_text()), "run_record": art}))
    assert CU.marker_problem(d) is None, "custody vouches for the dir"
    plan = run_gc(root, Disk(20 * UNIT, 18 * UNIT, []), apply=True, force=True, low_pct=0, remote_check=R.store_check(str(vault_store)))
    assert d.exists() and (d / "work/big.bin").exists()
    assert by_path(plan)["research/runs/r1"]["reason"] == (f"preserved {art} is not verified on the remote: 1 file(s) under it are not "
                                                         "in the artifact at their size, e.g. work/big.bin")


# ---- the command line -----------------------------------------------------------------------------------------------------------------

def test_keep_preserve_then_show_then_gc_through_research(tmp_path, vault_store, monkeypatch, capsys):
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    root = tmp_path / "ws"
    d = data(root, "pouw/gpu3-fp8/bitsets")
    rc = research_main(["keep", str(d), "--owner", "@fp8-security", "--expires", "2026-09-30", "--priority", "low", "--preserve",
                        "--why", "the bitsets, compressed", "--root", str(root), "--store", str(vault_store)])
    assert rc == 0, capsys.readouterr().err
    rec = R.load(d)
    assert LocalStore(vault_store).get_manifest(rec["preserved"]).kind == R.KIND
    assert (rec["expires"], rec["created"], rec["why"]) == ("2026-09-30T00:00:00Z", "2026-10-01T19:30:00Z", "the bitsets, compressed")
    capsys.readouterr()

    assert research_main(["keep", "show", str(d / "blob.bin"), "--root", str(root)]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == rec and "is covered by the record of" in out.err
    run_dir(root, "research/runs/r1", host=socket.gethostname())
    assert research_main(["keep", "show", str(root / "research/runs/r1"), "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["implicit"] == str(root / "research/runs/r1" / CU.MARKER)
    assert research_main(["keep", "show", str(root / "research"), "--root", str(root)]) == 1
    assert "unowned" in capsys.readouterr().err

    monkeypatch.setattr(R, "disk_usage", lambda _p: (100, 90, 10))
    assert research_main(["retention", "gc", "--root", str(root), "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert by_path(plan)["pouw/gpu3-fp8/bitsets"]["action"] == "delete" and d.exists(), "a dry run by default"
    real = R.store_check
    monkeypatch.setattr(R, "store_check", lambda store: lambda art, p: "unreachable" if p.name == "r1" else real(store)(art, p))
    assert research_main(["retention", "gc", "--root", str(root), "--store", str(vault_store), "--apply"]) == 0
    out = capsys.readouterr().out
    assert not d.exists() and (root / "research/runs/r1").exists()
    assert "DELETE" in out and "pouw/gpu3-fp8/bitsets" in out and "is not verified on the remote: unreachable" in out


def test_keep_refuses_a_bad_owner_and_a_bad_expiry_and_report_prints_a_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "disk_usage", lambda _p: (1 << 40, 1 << 39, 1 << 39))
    d = data(tmp_path, "pouw/x")
    assert research_main(["keep", str(d), "--owner", "pouw", "--expires", "never", "--priority", "low", "--root", str(tmp_path)]) == 2
    assert "lane handle" in capsys.readouterr().err
    assert research_main(["keep", str(d), "--owner", "@pouw", "--expires", "soon", "--priority", "low", "--root", str(tmp_path)]) == 2
    assert "give never, a date" in capsys.readouterr().err and not (d / R.SIDECAR).exists()
    assert research_main(["keep", str(d), "--owner", "@pouw", "--expires", "7d", "--priority", "low", "--root", str(tmp_path)]) == 0
    assert R.load(d)["expires"] == "2026-10-08T19:30:00Z"
    data(tmp_path, "cache/uv")
    capsys.readouterr()
    assert research_main(["retention", "report", "--root", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].startswith("UNOWNED") and lines[2].endswith(" cache") and lines[3].startswith("recorded")
    assert lines[-1].startswith("totals: unowned ")


# ---- rm -----------------------------------------------------------------------------------------------------------------------------------

REF = "C0C5RCXL66N/1791060198.615929"
BY = "https://cursor.com/agents/bc-test"


def passes(root: Path, preserved: str | None = ART) -> list[Path]:
    """The 3 Oct shape: one record on pouw/mvp-e2e, owned by @compute-accounting and unexpired, over pass dirs without their own."""
    area = data(root, "pouw/mvp-e2e")
    R.keep(area, root=root, owner="@compute-accounting", expires=FUTURE, priority="normal", preserved=preserved,
           why="PoUW served-path e2e", now=clock, node=NODE)
    return [data(root, f"pouw/mvp-e2e/passes/r{i}") for i in range(2)]


def log_lines(root: Path) -> list[dict]:
    p = root / R.LOG
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_rm_refuses_a_kept_pass_on_another_lanes_yes_and_names_its_owner(tmp_path):
    root = tmp_path / "ws"
    p0, p1 = passes(root)
    plan = R.rm([p0, p1], approved_by="@memory-accounting", ref=REF, root=root, apply=True, by=BY, in_use=never_in_use, now=clock,
                node=NODE)
    assert p0.exists() and p1.exists() and plan["freed_bytes"] == 0
    a = plan["actions"][0]
    assert a["action"] == "refuse" and a["owners"] == ["@compute-accounting"] and a["bytes"] == UNIT
    assert a["reason"] == ("pouw/mvp-e2e is kept by @compute-accounting until 2026-10-08T00:00:00Z (priority normal: PoUW served-path "
                           "e2e), and --approved-by @memory-accounting is not its owner: only @compute-accounting can say yes")
    lines = log_lines(root)
    assert [x["action"] for x in lines] == ["refuse", "refuse"] and {x["schema"] for x in lines} == {R.LOG_SCHEMA}
    assert lines[0]["by"] == BY and lines[0]["owner"] == "@compute-accounting" and lines[0]["path"] == str(p0)
    assert "approved by @memory-accounting (C0C5RCXL66N/1791060198.615929)" in lines[0]["reason"]


def test_rm_deletes_with_the_owners_yes_leaves_the_covering_record_and_logs_who(tmp_path):
    root = tmp_path / "ws"
    p0, p1 = passes(root)
    dry = R.rm([p0], approved_by="@compute-accounting", ref=REF, root=root, in_use=never_in_use, now=clock, node=NODE)
    assert dry["actions"][0]["action"] == "delete" and p0.exists() and not (root / R.LOG).exists(), "a dry run deletes and logs nothing"
    plan = R.rm([p0], approved_by="@compute-accounting", ref=REF, root=root, apply=True, by=BY, in_use=never_in_use, now=clock,
                node=NODE, remote_check=preserved_ok)
    assert not p0.exists() and p1.exists() and R.load(root / "pouw/mvp-e2e")["owner"] == "@compute-accounting"
    assert plan["freed_bytes"] == UNIT and plan["actions"][0]["reason"].endswith("and its owner said yes")
    (line,) = log_lines(root)
    assert (line["action"], line["by"], line["bytes"], line["files"]) == ("delete", BY, UNIT, 1)


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_rm_refuses_unowned_keep_another_owner_beneath_in_use_read_only_and_non_data(tmp_path):
    root = tmp_path / "ws"
    data(root, "cache/uv")
    kept(root, "pouw/a", owner="@compute-accounting", priority="keep", expires=None)
    area = kept(root, "pouw/b", owner="@compute-accounting", expires=FUTURE)
    kept(root, "pouw/b/inner", owner="@circuits", expires=FUTURE)
    kept(root, "pouw/c", owner="@compute-accounting", expires=FUTURE)
    ro = kept(root, "pouw/d", owner="@compute-accounting", expires=FUTURE)
    (ro / "sub").mkdir()
    (ro / "sub").chmod(0o555)
    lane_area = kept(root, "research/old", owner="@x", expires=FUTURE)
    run_dir(root, "research/old/r1", lane="y")
    try:
        paths = [root / "cache/uv", root / "pouw/a", area, root / "pouw/c", ro, lane_area, root / "nope", area / R.SIDECAR, root]
        plan = R.rm(paths, approved_by="@compute-accounting", ref=REF, root=root, now=clock, node=NODE,
                    in_use=lambda p: "q.sh names it" if p.name == "c" else None)
    finally:
        (ro / "sub").chmod(0o755)
    why = {Path(a["path"]).relative_to(root).as_posix() if a["path"] != str(root) else ".": a["reason"] for a in plan["actions"]}
    assert all(a["action"] == "refuse" for a in plan["actions"])
    assert why["cache/uv"].startswith("no record covers it, so nobody can say yes")
    assert "priority keep is never deleted" in why["pouw/a"]
    assert why["pouw/b"].startswith("pouw/b/inner is kept by @circuits") and "only @circuits can say yes" in why["pouw/b"]
    assert why["pouw/c"] == "in use: q.sh names it"
    assert why["pouw/d"] == f"{ro / 'sub'} is read-only: its owner makes it writable first"
    assert "research/old is kept by @x" in why["research/old"], "the area's own owner is checked before the run beneath it"
    assert why["nope"].endswith("does not exist") and "is a record, not data" in why[f"pouw/b/{R.SIDECAR}"] and "is the root" in why["."]
    plan = R.rm([lane_area], approved_by="@x", ref=REF, root=root, now=clock, node=NODE, in_use=never_in_use)
    assert plan["actions"][0]["reason"].startswith("research/old/r1 is kept by @y"), "an implicit record beneath counts, its lane as @y"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_rm_apply_needs_who_and_a_writable_log_and_rereads_the_records(tmp_path):
    root = tmp_path / "ws"
    p0, _ = passes(root)
    with pytest.raises(R.RecordError, match="--by: say who deletes"):
        R.rm([p0], approved_by="@compute-accounting", ref=REF, root=root, apply=True, in_use=never_in_use, now=clock)
    with pytest.raises(R.RecordError, match="--ref"):
        R.rm([p0], approved_by="@compute-accounting", ref=" ", root=root, in_use=never_in_use, now=clock)
    with pytest.raises(R.RecordError, match="lane handle"):
        R.rm([p0], approved_by="compute-accounting", ref=REF, root=root, in_use=never_in_use, now=clock)
    (tmp_path / "ro-log").mkdir(mode=0o555)
    try:
        with pytest.raises(R.RecordError, match="can't write the deletion log"):
            R.rm([p0], approved_by="@compute-accounting", ref=REF, root=root, apply=True, by=BY, in_use=never_in_use, now=clock,
                 log=tmp_path / "ro-log" / "sub" / "deletions.jsonl")
    finally:
        (tmp_path / "ro-log").chmod(0o755)

    def renewing(p: Path) -> None:
        R.keep(root / "pouw/mvp-e2e", root=root, owner="@compute-accounting", expires="2026-10-09T00:00:00Z", priority="normal",
               now=clock, node=NODE)

    plan = R.rm([p0], approved_by="@compute-accounting", ref=REF, root=root, apply=True, by=BY, in_use=renewing, now=clock, node=NODE)
    assert p0.exists() and plan["actions"][0]["action"] == "failed"
    assert plan["actions"][0]["error"] == "its records changed during rm: run it again" and log_lines(root)[-1]["action"] == "failed"


def test_rm_deletes_passes_only_once_their_record_names_an_artifact_verified_on_the_remote(tmp_path):
    root = tmp_path / "ws"
    p0, _ = passes(root, preserved=None)
    kw = {"approved_by": "@compute-accounting", "ref": REF, "root": root, "by": BY, "in_use": never_in_use, "now": clock, "node": NODE}
    for p in (p0, root / "pouw/mvp-e2e"):
        a = R.rm([p], apply=True, remote_check=preserved_ok, **kw)["actions"][0]
        assert a["action"] == "refuse" and p.exists(), "the owner's yes alone deletes neither a pass nor the area that holds them"
        assert a["reason"].startswith("it holds retained passes and pouw/mvp-e2e is kept by @compute-accounting")
    passes(root)
    seen = []
    plan = R.rm([p0], apply=True, remote_check=lambda art, p: seen.append((art, p)) or "404 on the remote", **kw)
    a = plan["actions"][0]
    assert a["action"] == "failed" and p0.exists() and seen == [(ART, root / "pouw/mvp-e2e")], "the record's whole tree is checked"
    assert a["error"] == f"it holds retained passes, and preserved {ART} is not verified on the remote: 404 on the remote"
    assert R.rm([p0], remote_check=lambda *_: pytest.fail("a dry run reads no remote"), **kw)["actions"][0]["action"] == "delete"


def test_rm_through_research(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "store_check", lambda _root: preserved_ok)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    monkeypatch.delenv("RESEARCH_AGENT_URL", raising=False)
    root = tmp_path / "ws"
    p0, p1 = passes(root)
    base = ["retention", "rm", str(p0), str(p1), "--ref", REF, "--root", str(root)]
    assert research_main([*base, "--approved-by", "@memory-accounting", "--apply", "--by", BY]) == 1
    out = capsys.readouterr().out
    assert out.count("REFUSED") == 2 and "only @compute-accounting can say yes" in out and p0.exists()
    assert research_main([*base, "--approved-by", "@compute-accounting", "--apply"]) == 2
    assert "--by: say who deletes" in capsys.readouterr().err
    monkeypatch.setenv("RESEARCH_AGENT_URL", BY)
    assert research_main([*base, "--approved-by", "@compute-accounting", "--json"]) == 0
    assert [a["action"] for a in json.loads(capsys.readouterr().out)["actions"]] == ["delete", "delete"] and p0.exists()
    assert research_main([*base, "--approved-by", "@compute-accounting", "--apply"]) == 0
    assert not p0.exists() and not p1.exists() and "deleted 2 of 2 path(s)" in capsys.readouterr().out
    assert [x["by"] for x in log_lines(root)][-2:] == [BY, BY]
