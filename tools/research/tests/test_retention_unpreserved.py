"""The record of a run that custody never preserved (research.retention): a finished or runner-dead run that custody did not
preserve gets a low-priority retention record that expires UNPRESERVED_RUN_EXPIRY after the run directory's last write, so the
collector takes it. Which directories qualify and which never do, where the record lives, what gc and the report print, and the
README's statement of the rule. Every directory is under tmp_path, with each entry's mtime set by hand and the process table a
directory the test writes."""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_retention import NOW, REPO, UNIT, Disk, by_path, clock, kept, never_in_use, run_dir

from research import clone as CL
from research import retention as R
from research import runs
from research.cli import main as research_main
from research.store import custody as CU

HOST = socket.gethostname()
RUNS = "research/runs"
RUN, RUN2, RUN3 = "r20260920-100000-ab12", "r20260921-100000-cd34", "r20260922-100000-ef56"
WRITTEN = NOW - timedelta(days=10)             # every entry's mtime, unless a test writes another
EXPIRES = R.fmt_utc(WRITTEN + R.UNPRESERVED_RUN_EXPIRY)
PID, TICKS = 4242, 777
RUNNING = {"state": "running", "transitions": [{"state": "running", "pid": PID, "starttime_ticks": TICKS}]}
KEEP = "; to keep it: `research keep {d} --owner @LANE --expires DATE --priority P`, or preserve it: `research data custody {run} --publish`"


def process(table: Path, pid: int, ticks: int | None) -> None:
    """A process in the table `table`, started at `ticks` (the 20th field after its command's closing parenthesis); None: a stat
    line without a start time."""
    (table / str(pid)).mkdir(parents=True, exist_ok=True)
    fields = ["S", *["0"] * 18, *([str(ticks), "0", "0"] if ticks is not None else [])]
    (table / str(pid) / "stat").write_text(f"{pid} (python3 (run)) {' '.join(fields)}\n")


@pytest.fixture(autouse=True)
def table(tmp_path, monkeypatch) -> Path:
    """The process table the rule reads: pid 1 alone, until a test starts a process in it."""
    t = tmp_path / "proc"
    process(t, 1, 1)
    monkeypatch.setattr(R, "PROC", t)
    return t


def written(path: Path, when: datetime = WRITTEN) -> Path:
    """Set the mtime of `path` and of everything under it to `when`, following no symlink."""
    for base, dirs, files in os.walk(path):
        for name in dirs + files:
            os.utime(os.path.join(base, name), (when.timestamp(), when.timestamp()), follow_symlinks=False)
    os.utime(path, (when.timestamp(), when.timestamp()), follow_symlinks=False)
    return path


def ended(root: Path, name: str = RUN, *, under: str = RUNS, status: dict | str | None = "failed", marker: bool | None = None,
          lane: str | None = None, campaign: str | None = None, when: datetime = WRITTEN, **job: object) -> Path:
    """A run directory as `research run` leaves one on a machine: job.json, status.json (a final state's name, a document, or None for
    none), a log and an output, and with `marker` the `.custody` custody wrote. `job` replaces fields of job.json's `run` and
    `environment` (None removes one). Every entry was last written at `when`."""
    d = root / under / name
    (d / "out").mkdir(parents=True)
    doc = {"run": {"id": name, "runs_root": str(d.parent), "created_utc": "2026-09-20T10:00:00.123Z"},
           "environment": {"hostname": HOST}, "ownership": {"lane": lane}, "store": {"campaign": campaign}}
    for k, v in job.items():
        part = doc["environment" if k == "hostname" else "run"]
        part.pop(k, None) if v is None else part.update({k: v})
    (d / "job.json").write_text(json.dumps(doc))
    (d / "stdout.log").write_bytes(b"y" * UNIT)
    (d / "out/result.bin").write_bytes(b"z" * UNIT)
    if status is not None:
        final = {"state": status, "transitions": [RUNNING["transitions"][0], {"state": status}]}
        (d / "status.json").write_text(json.dumps(final if isinstance(status, str) else status))
    if marker is not None:
        (d / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": name, "preserved": marker, "error": "the upload failed",
                                               "verified_utc": "2026-09-21T00:00:00Z", "host": HOST}))
    return written(d, when)


def record(d: Path, root: Path) -> R.Found | None:
    return R.record_at(d, R.relative(d, root))


def entries(d: Path) -> dict[str, int]:
    return {os.path.join(base, n): os.lstat(os.path.join(base, n)).st_mtime_ns for base, dirs, files in os.walk(d) for n in dirs + files}


def collect(root: Path, paths: list[Path], **kw: object) -> dict:
    kw = {"now": clock, "usage": Disk(100 * UNIT, 90 * UNIT, paths), "in_use": never_in_use, "node": HOST, "force": True, "low_pct": 0, **kw}
    return R.gc(root, **kw)


# ---- 1: which run directories qualify -------------------------------------------------------------------------------------------------

def test_a_finished_run_that_custody_did_not_preserve_has_a_low_record_that_expires_7_days_after_its_last_write(tmp_path):
    root = tmp_path / "ws"
    d = ended(root, lane="compute-accounting", campaign="pouw")
    before = entries(d)
    found = record(d, root)
    assert found.source == R.UNPRESERVED and entries(d) == before, "the record is computed: nothing is written into the run directory"
    assert found.record == {"schema": R.SCHEMA, "path": f"{RUNS}/{RUN}", "node": HOST, "owner": "compute-accounting", "by": None,
                            "created": "2026-09-20T10:00:00Z", "expires": EXPIRES, "priority": R.PRIORITIES[0], "preserved": None,
                            "regenerate": None,
                            "why": "no run custody (it ended failed): this directory is the run's only copy, and it expires 7 d "
                                   "after its last write"}
    assert R.implicit(d, f"{RUNS}/{RUN}") == found.record and R.find(d / "out", root).path == d, "it covers everything beneath it"
    assert R.expired(found.record, NOW) and R.PRIORITIES[0] == "low" and R.UNPRESERVED_RUN_EXPIRY == timedelta(days=7)

    for name, state in ((RUN2, "done"), (RUN3, "cancelled")):
        r = record(ended(root, name, status=state, campaign="pouw"), root).record
        assert (r["owner"], r["why"].split(":")[0]) == ("pouw", f"no run custody (it ended {state})"), "no lane: the campaign owns it"
    no_created = record(ended(root, "r20260923-100000-0001", created_utc=None), root).record
    assert (no_created["owner"], no_created["created"]) == ("runner", None), "no lane or campaign: runner"


def test_a_run_whose_custody_failed_or_that_was_refused_has_the_record_and_a_preserved_one_keeps_custodys(tmp_path):
    root = tmp_path / "ws"
    failed = record(ended(root, marker=False), root)
    assert (failed.source, failed.record["expires"]) == (R.UNPRESERVED, EXPIRES), "`.custody` says preserved: false"
    refused = ended(root, RUN2, status=None)
    (refused / "refused.json").write_text("{}")
    r = record(written(refused), root).record
    assert (r["why"].split(":")[0], r["expires"]) == ("no run custody (it ended refused)", EXPIRES)

    kept_by_custody = ended(root, RUN3, marker=True)
    found = record(kept_by_custody, root)
    assert (found.source, found.record["priority"], found.record["expires"]) == ("custody", "normal", "2026-09-23T00:00:00Z")
    for body in ("{", json.dumps({"schema": "another/v1", "preserved": False}), json.dumps({"schema": CU.SCHEMA, "preserved": 0}),
                 json.dumps({"schema": CU.SCHEMA})):
        (kept_by_custody / CU.MARKER).write_text(body)
        assert record(written(kept_by_custody), root) is None, f"a marker that does not say what custody did gives no record: {body}"


def test_a_runner_is_dead_only_when_its_pid_at_its_start_time_is_gone_from_its_own_hosts_process_table(tmp_path, table):
    root = tmp_path / "ws"
    d = ended(root, status=RUNNING)
    gone = record(d, root).record
    assert (gone["why"].split(":")[0], gone["expires"]) == ("no run custody (its runner is gone)", EXPIRES), "no process has the pid"

    process(table, PID, TICKS)
    assert record(d, root) is None, "the runner is alive: the pid, started when the run recorded"
    process(table, PID, TICKS + 1)
    assert record(d, root).source == R.UNPRESERVED, "the pid is another process's now: it started at another time"
    process(table, PID, None)
    assert record(d, root) is None, "a pid alone never shows a runner dead: its start time can't be read"

    (table / str(PID) / "stat").unlink()
    assert record(d, root).source == R.UNPRESERVED
    (table / "1" / "stat").unlink()
    assert record(d, root) is None, "a process table that does not show pid 1 shows nobody dead"
    process(table, 1, 1)
    assert record(ended(root, RUN2, status=RUNNING, hostname="another-host"), root) is None, "the pid was another host's"
    assert record(ended(root, RUN3, status={"state": "running", "transitions": [{"state": "running"}]}), root) is None, "no pid recorded"

    if os.geteuid():       # root reads every file
        process(table, PID, TICKS + 1)
        (table / str(PID) / "stat").chmod(0)
        found = record(d, root)        # None where the process reader answers "not known gone", a Found where it raises
        assert found is None or (found.record, found.source) == (None, R.UNPRESERVED), "a process that can't be read is not shown dead"
        assert found is None or found.error == (f"whether the run of {d} ended can't be read ([Errno 13] Permission denied: "
                                                f"'{table}/{PID}/stat')")


def test_an_explicit_record_on_the_run_or_above_it_always_wins(tmp_path):
    root = tmp_path / "ws"
    d = ended(root)
    for priority, expires in (("keep", None), ("low", "2026-01-01T00:00:00Z"), ("high", "2027-01-01T00:00:00Z")):
        R.keep(d, root=root, owner="@infra", expires=expires, priority=priority, now=clock, node=HOST)
        found = record(written(d), root)
        assert (found.source, found.record["priority"], found.record["expires"]) == ("sidecar", priority, expires)
    (d / R.SIDECAR).write_text("{")
    assert record(written(d), root).error, "an invalid record is still the run's record"
    (d / R.SIDECAR).unlink()
    assert record(written(d), root).source == R.UNPRESERVED

    R.keep(root / RUNS, root=root, owner="@infra", expires="2026-01-01T00:00:00Z", priority="low", regenerate="rerun", now=clock, node=HOST)
    assert record(d, root) is None and R.find(d, root).path == root / RUNS, "a record above covers it, whatever its priority and expiry"
    assert [r.path for r in R.scan(root).rows] == [RUNS]
    (root / RUNS / R.SIDECAR).write_text("{")
    assert record(d, root) is None, "nor under a record that is invalid"
    (root / RUNS / R.SIDECAR).unlink()
    R.beside(root / RUNS).write_text(json.dumps({**R.implicit(d, RUNS), "path": RUNS, "owner": "@infra"}))
    assert record(d, root) is None, "nor under a record that lies beside its directory"


# ---- 2: which never qualify -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("status", [{"state": "submitted"}, {"state": "admitted"}, {"state": "preserving"}, {"state": "running"}, {},
                                    "{", None, RUNNING],
                         ids=["submitted", "admitted", "preserving", "running, no pid", "no state", "not JSON", "no status.json",
                              "waiting in a slot line"])
def test_a_run_not_known_to_have_ended_has_no_record(tmp_path, table, status):
    root = tmp_path / "ws"
    process(table, PID, TICKS)
    d = ended(root, status=None if status in (None, "{") else status)
    if status == "{":
        (d / "status.json").write_text(status)
    if status is RUNNING:      # a run reads `running` from its launch, and its slot record says what it waits for
        (d / runs.SLOT_FILE).write_text(json.dumps({"state": "waiting", "since": "2026-09-20T10:00:05Z"}))
    assert record(written(d), root) is None
    rows = R.scan(root).rows
    assert [(r.path, r.record) for r in rows] == [("research", None)], "its area is unowned, so gc never deletes it"


def test_a_queued_run_and_a_directory_that_is_not_a_run_directory_have_no_record(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    queued = root / RUNS / RUN
    queued.mkdir(parents=True)
    (queued / "launch.json").write_text(json.dumps({"runner_pid": PID}))
    assert record(written(queued), root) is None, "a queued run has no job.json yet"

    assert record(ended(root, "results"), root) is None, "not named as a run"
    assert record(ended(root, RUN2, id=RUN3), root) is None, "its job.json is another run's"
    assert record(ended(root, RUN3, id=None), root) is None, "its job.json names no run"
    elsewhere = ended(root, "r20260923-100000-0001", runs_root=str(root / "other"))
    (root / "other").mkdir()
    assert record(elsewhere, root) is None, "its job.json places it in another runs root: a copy"
    relative = ended(root, "r20260923-100000-0002", runs_root=RUNS)
    monkeypatch.chdir(root)
    assert record(relative, root) is None, "a relative runs root places it nowhere, wherever the collector runs"
    assert record(ended(root, "r20260923-100000-0003", runs_root=None), root) is None
    assert record(ended(root, "r20260923-100000-0004", hostname=None), root) is None, "its job.json names no host"
    fetched = ended(root, "r20260923-100000-0005")
    (fetched / CU.LAUNCHER_RECORD).write_text("{}")
    assert record(written(fetched), root) is None, "the launcher's copy of a run on another machine"
    verified = ended(root, "r20260923-100000-0008")
    (verified / "preserved.json").write_text("{}")
    assert CU.has_custody(verified)[0] and record(written(verified), root) is None, "a fetched copy that custody vouches for"

    outer = ended(root, "r20260923-100000-0006")
    inner = ended(root, RUN, under=f"{RUNS}/r20260923-100000-0006/out")
    written(outer)
    assert record(inner, root) is None and record(outer, root).source == R.UNPRESERVED, "a run inside a run's directory is its data"
    link = root / RUNS / "r20260923-100000-0007"
    link.symlink_to(outer)
    assert record(link, root) is None, "a link to a run directory"
    assert sorted(r.path for r in R.scan(root).rows if r.record) == [f"{RUNS}/r20260923-100000-0006"]


def test_the_last_write_is_the_newest_mtime_of_anything_under_the_run_directory(tmp_path):
    root = tmp_path / "ws"
    d = ended(root)
    deep = d / "out/a/b/c"
    deep.mkdir(parents=True)
    written(d)
    young = NOW - timedelta(days=2)
    (deep / "late.bin").write_bytes(b"x")
    written(d)
    os.utime(deep / "late.bin", (young.timestamp(), young.timestamp()))
    rec = record(d, root).record
    assert rec["expires"] == R.fmt_utc(young + R.UNPRESERVED_RUN_EXPIRY) and not R.expired(rec, NOW), "one young file, four levels down"

    written(d)
    os.utime(deep, (young.timestamp(), young.timestamp()))
    assert record(d, root).record["expires"] == R.fmt_utc(young + R.UNPRESERVED_RUN_EXPIRY), "an empty directory's own mtime counts"
    written(d)
    os.utime(d, (young.timestamp(), young.timestamp()))
    assert record(d, root).record["expires"] == R.fmt_utc(young + R.UNPRESERVED_RUN_EXPIRY), "and the run directory's"

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    os.utime(outside, (NOW.timestamp(), NOW.timestamp()))
    (d / "out/link").symlink_to(outside)
    assert record(written(d), root).record["expires"] == EXPIRES, "a link's target is not under the run directory"

    edge = NOW - R.UNPRESERVED_RUN_EXPIRY
    assert R.expired(record(written(d, edge), root).record, NOW), "7 days after its last write, to the second"
    assert not R.expired(record(written(d, edge + timedelta(seconds=1)), root).record, NOW), "one second short of 7 days"
    os.utime(written(d, edge) / "stdout.log", (edge.timestamp() + 0.25, edge.timestamp() + 0.25))
    assert record(d, root).record["expires"] == R.fmt_utc(NOW + timedelta(seconds=1)), "a fraction of a second rounds up"
    with pytest.raises(FileNotFoundError):
        R._last_write(d / "absent")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory and file")
def test_a_run_directory_that_cannot_be_read_has_no_record_is_listed_with_what_could_not_be_read_and_is_never_deleted(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    d = ended(root)
    (d / "out/private").mkdir()
    written(d)
    through_research(monkeypatch, [d])
    for unreadable, mode, error in (
            (d / "out/private", 0o755, f"when {d} was last written can't be read ([Errno 13] Permission denied: '{d}/out/private')"),
            (d / "status.json", 0o644, f"whether the run of {d} ended can't be read (cannot read {d}/status.json: [Errno 13] Permission "
                                       f"denied: '{d}/status.json')")):
        unreadable.chmod(0)
        try:
            found = record(d, root)
            assert (found.record, found.source, found.error) == (None, R.UNPRESERVED, error), "what can't be read may be newer, or live"
            with pytest.raises(R.RecordError):
                R.implicit(d, f"{RUNS}/{RUN}")
            plan = collect(root, [d], apply=True, unpreserved=True)
            act = by_path(plan)[f"{RUNS}/{RUN}"]
            assert d.exists() and (act["action"], act["state"]) == ("skip", "invalid")
            assert act["reason"] == f"no record: {error}" + KEEP.format(d=d, run=RUN)
            assert (plan["unpreserved"]["invalid"]["paths"], plan["unpreserved"]["expired"]["paths"]) == (1, 0)

            unread = f"; 1 run(s), {R._size(act['bytes'])}, {act['inodes']} inode(s) with no record, since what decides it can't be read"
            assert research_main(["retention", "gc", "--root", str(root), "--apply", "--force", "--low-pct", "0", "--unpreserved-runs"]) == 0
            out = capsys.readouterr().out.splitlines()
            assert d.exists() and out[out.index(R.UNPRESERVED_HEAD) + 1].split()[0] == "skip"
            assert [x for x in out if x.startswith("unpreserved runs, ")][0].endswith("0 inode(s) not yet expired" + unread)
            assert research_main(["retention", "report", "--root", str(root)]) == 0
            out = capsys.readouterr().out.splitlines()
            listed = out[out.index(R.UNPRESERVED_HEAD) + 1]
            assert listed.startswith("INVALID ") and listed.endswith(f"{RUNS}/{RUN}  {error}")
            assert out[-1].endswith("0 inode(s) not yet expired" + unread)
            assert research_main(["keep", "show", str(d), "--root", str(root)]) == 1
            assert capsys.readouterr().err == f"research keep show: {error}\n"
        finally:
            unreadable.chmod(mode)
    assert act["bytes"] == nbytes(d) and act["inodes"] == 7, "with only its status unread, the whole directory is counted"
    found = record(d, root)
    assert (found.source, found.error, found.record["expires"]) == (R.UNPRESERVED, None, EXPIRES)


def test_the_record_never_expires_before_the_custody_key_its_runner_publishes_with(tmp_path):
    root = tmp_path / "ws"
    request = tmp_path / "requests" / RUN / "request.json"
    cred = request.parent / CU.DIR_NAME / CU.CRED_FILE
    cred.parent.mkdir(parents=True)
    d = ended(root, remote={"request": str(request)})
    assert record(d, root).record["expires"] == EXPIRES, "no key: its runner deleted it, or custody was off"

    cred.write_text(json.dumps({"expires_utc": "2026-10-02T06:00:00Z"}))
    rec = record(d, root).record
    assert rec["expires"] == "2026-10-02T06:00:00Z" and not R.expired(rec, NOW), "the runner may still be publishing with that key"
    cred.write_text(json.dumps({"expires_utc": "2026-09-21T06:00:00Z"}))
    assert record(d, root).record["expires"] == EXPIRES, "a key that expired before the 7 days were over changes nothing"
    for body in ("{", json.dumps({}), json.dumps({"expires_utc": "soon"})):
        cred.write_text(body)
        found = record(d, root)
        assert (found.record, found.source) == (None, R.UNPRESERVED), f"a key whose expiry can't be read gives no record: {body}"
        assert found.error == f"the expiry of the custody key {cred} can't be read, so whether the runner of {d} still publishes is unknown"
    cred.unlink()
    assert record(d, root).record["expires"] == EXPIRES


@pytest.mark.skipif(os.geteuid() == 0, reason="root searches every directory")
def test_a_request_directory_that_cannot_be_searched_for_the_custody_key_gives_no_record(tmp_path):
    root = tmp_path / "ws"
    request = tmp_path / "requests" / RUN / "request.json"
    custody = request.parent / CU.DIR_NAME
    custody.mkdir(parents=True)
    d = ended(root, remote={"request": str(request)})
    custody.chmod(0)
    try:
        found = record(d, root)
        assert (found.record, found.source) == (None, R.UNPRESERVED), "a runner may hold a key there that this user can't see"
        assert found.error == (f"whether the runner of {d} still holds its custody key can't be read ([Errno 13] Permission denied: "
                               f"'{custody / CU.CRED_FILE}')")
    finally:
        custody.chmod(0o755)
    assert record(d, root).record["expires"] == EXPIRES


# ---- 3: where the record lives --------------------------------------------------------------------------------------------------------

def test_the_record_is_computed_at_each_pass_and_a_write_during_the_pass_keeps_the_run(tmp_path):
    root = tmp_path / "ws"
    d, quiet = ended(root), ended(root, RUN2)
    before = entries(root)
    assert record(d, root).record == record(d, root).record and [r.source for r in R.scan(root).rows] == [R.UNPRESERVED] * 2
    assert entries(root) == before and not (d / R.SIDECAR).exists() and not R.beside(d).exists(), "no file holds the record"

    def a_lane_writes(path: Path) -> None:
        if path == d and not (d / "out/more.bin").exists():
            (d / "out/more.bin").write_bytes(b"x")
            written(d / "out", NOW)

    plan = collect(root, [d, quiet], apply=True, unpreserved=True, in_use=a_lane_writes)
    acts = by_path(plan)
    assert d.exists() and not quiet.exists()
    assert acts[f"{RUNS}/{RUN}"]["reason"] == "its record changed during gc: run gc again" + KEEP.format(d=d, run=RUN)
    after = record(d, root).record
    assert after["expires"] == R.fmt_utc(NOW + R.UNPRESERVED_RUN_EXPIRY) and not R.expired(after, NOW), "the write gave it 7 more days"
    assert by_path(collect(root, [d], apply=True, unpreserved=True))[f"{RUNS}/{RUN}"]["reason"].startswith("not expired: expires ")


# ---- 4: what the collector prints -----------------------------------------------------------------------------------------------------

def workspace(root: Path) -> tuple[Path, Path, Path, Path]:
    """Two expired unpreserved runs, one written two days ago, and an expired regenerable path with an explicit record."""
    a = ended(root, lane="compute-accounting")
    b = ended(root, RUN2, status=RUNNING)
    young = ended(root, RUN3, when=NOW - timedelta(days=2))
    return a, b, young, kept(root, "pouw/cache", node=HOST)


def through_research(monkeypatch: pytest.MonkeyPatch, paths: list[Path]) -> None:
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    monkeypatch.setattr(R, "disk_usage", Disk(100 * UNIT, 90 * UNIT, paths))
    monkeypatch.setattr(R, "inode_usage", lambda _p: (0, 0))


def nbytes(*runs: Path) -> int:
    return sum(f.lstat().st_size for d in runs for f in d.rglob("*") if f.is_file())


def counted(*runs: Path) -> str:
    """How gc and the report count `runs`: each is the six inodes `ended` makes (the run directory, out/ and four files)."""
    return f"{len(runs)} run(s), {R._size(nbytes(*runs))}, {6 * len(runs)} inode(s)"


def line(verb: str, d: Path, root: Path, reason: str) -> str:
    return f"{verb:<7} {R._size(nbytes(d)):>10}  {R.relative(d, root)}  {reason}"


ONLY_COPY = f"low, expired {EXPIRES}, the only copy of a run that custody did not preserve"


def test_gc_lists_the_unpreserved_runs_apart_and_deletes_none_without_unpreserved_runs(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    a, b, young, cache = workspace(root)
    through_research(monkeypatch, [a, b, young, cache])
    assert research_main(["retention", "gc", "--root", str(root), "--apply", "--force", "--low-pct", "0"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert a.exists() and b.exists() and young.exists() and not cache.exists(), "the pass the timer runs takes what it took before"
    head = out.index(R.UNPRESERVED_HEAD)
    assert out[1].startswith("DELETE ") and "pouw/cache" in out[1] and head == 2, "after every other path"
    left = f"left: {ONLY_COPY}, which only --unpreserved-runs deletes"
    assert sorted(out[head + 1: head + 4]) == sorted([
        line("skip", a, root, left + KEEP.format(d=a, run=RUN)), line("skip", b, root, left + KEEP.format(d=b, run=RUN2)),
        line("skip", young, root, f"not expired: expires {R.fmt_utc(NOW + timedelta(days=5))}" + KEEP.format(d=young, run=RUN3))])
    assert out[head + 4].startswith("deleted 1 path(s), ")
    assert out[head + 5] == (f"unpreserved runs, of the paths above: {counted(a, b)} expired, none of which goes without "
                             f"--unpreserved-runs; {counted(young)} not yet expired")

    plan = collect(root, [a, b, young], apply=True)
    assert a.exists() and b.exists() and plan["deletes_unpreserved"] is False and plan["unpreserved"] == {
        "expired": {"paths": 2, "bytes": nbytes(a, b), "inodes": 12}, "recorded": {"paths": 1, "bytes": nbytes(young), "inodes": 6},
        "invalid": {"paths": 0, "bytes": 0, "inodes": 0}, "deleted": {"paths": 0, "bytes": 0, "inodes": 0}}
    assert by_path(plan)[f"{RUNS}/{RUN}"]["inodes"] == 6, "the run directory, out/ and its four files"


def test_gc_with_unpreserved_runs_reports_what_it_would_delete_and_with_apply_deletes_it(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    a, b, young, cache = workspace(root)
    through_research(monkeypatch, [a, b, young, cache])
    gone, stays = counted(a, b), counted(young)
    would = [line("DELETE", a, root, f"would delete: {ONLY_COPY}" + KEEP.format(d=a, run=RUN)),
             line("DELETE", b, root, f"would delete: {ONLY_COPY}" + KEEP.format(d=b, run=RUN2))]
    deleted = [line("DELETE", a, root, f"deleted: {ONLY_COPY}"), line("DELETE", b, root, f"deleted: {ONLY_COPY}")]
    argv = ["retention", "gc", "--root", str(root), "--force", "--low-pct", "0", "--unpreserved-runs"]
    assert research_main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert all(p.exists() for p in (a, b, young, cache)), "a dry run deletes nothing"
    head = out.index(R.UNPRESERVED_HEAD)
    assert out[0].startswith("research retention gc: policy default/v0, dry run (--apply deletes): ")
    assert [x.split("  -> ")[0] for x in out[head + 1: head + 3]] == would
    assert out[head + 3].startswith(line("skip", young, root, "not expired: expires "))
    assert out[head + 4].startswith("would delete 3 path(s), ") and out[head + 4].endswith("; nothing was deleted")
    assert out[head + 5] == (f"unpreserved runs, of the paths above: {gone} expired, of which this pass would delete {gone}; "
                             f"{stays} not yet expired")

    assert research_main([*argv, "--apply"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert not a.exists() and not b.exists() and not cache.exists() and young.exists()
    head = out.index(R.UNPRESERVED_HEAD)
    assert [x.split("  -> ")[0] for x in out[head + 1: head + 3]] == deleted, "a deleted run's line names no way to keep it"
    assert out[head + 5] == f"unpreserved runs, of the paths above: {gone} expired, of which this pass deleted {gone}; {stays} not yet expired"
    assert research_main([*argv, "--apply", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["deletes_unpreserved"] is True and [x["path"] for x in plan["actions"] if x["source"]] == [f"{RUNS}/{RUN3}"]
    assert plan["unpreserved"]["recorded"]["paths"] == 1 and plan["unpreserved"]["deleted"]["paths"] == 0


def test_gc_takes_an_unpreserved_run_before_a_preserved_one_and_never_one_of_another_node_or_one_in_use(tmp_path):
    root = tmp_path / "ws"
    a = ended(root)
    preserved = run_dir(root, f"{RUNS}/r1", host=HOST)
    other = ended(root, RUN2, hostname="another-host")
    used = ended(root, RUN3)
    plan = collect(root, [a, preserved, other, used], unpreserved=True, in_use=lambda p: "a queued job names it" if p == used else None)
    acts = by_path(plan)
    assert [x["path"] for x in plan["actions"] if x["action"] == "delete"] == [f"{RUNS}/{RUN}", f"{RUNS}/r1"], "low before normal"
    assert acts[f"{RUNS}/{RUN3}"]["reason"] == "in use: a queued job names it" + KEEP.format(d=used, run=RUN3)
    assert acts[f"{RUNS}/{RUN2}"]["reason"] == (f"recorded on another-host, and this is {HOST}: `research keep` it here"
                                                + KEEP.format(d=other, run=RUN2)), "a run this node did not run is never this node's to take"


def test_the_report_lists_the_unpreserved_runs_apart_with_their_count_bytes_and_inodes(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    a, b, young, cache = workspace(root)
    preserved = run_dir(root, f"{RUNS}/r1", host=HOST)
    rep = R.report(root, now=clock, usage=Disk(100 * UNIT, 90 * UNIT, [a, b, young, cache, preserved]), node=HOST)
    rows = {r["path"]: r for r in rep["rows"]}
    assert rep["unpreserved"] == {"expired": {"paths": 2, "bytes": nbytes(a, b), "inodes": 12},
                                  "recorded": {"paths": 1, "bytes": nbytes(young), "inodes": 6},
                                  "invalid": {"paths": 0, "bytes": 0, "inodes": 0}}
    assert (rows[f"{RUNS}/{RUN}"]["source"], rows[f"{RUNS}/{RUN}"]["inodes"], rows[f"{RUNS}/{RUN}"]["bytes"]) == (R.UNPRESERVED, 6, nbytes(a))
    assert rows[f"{RUNS}/r1"]["source"] == "custody" and rows["pouw/cache"]["source"] == "sidecar"

    through_research(monkeypatch, [a, b, young, cache, preserved])
    assert research_main(["retention", "report", "--root", str(root)]) == 0
    out = capsys.readouterr().out.splitlines()
    head = out.index(R.UNPRESERVED_HEAD)
    assert all(RUN not in x and RUN2 not in x and RUN3 not in x for x in out[:head]), "after every other path"
    assert any(x.endswith(f"{RUNS}/r1 (implicit: .custody)") for x in out[:head])
    assert sorted(x.split()[0] + " " + x.rsplit("  ", 1)[-1] for x in out[head + 1: head + 4]) == [
        f"EXPIRED {RUNS}/{RUN} (implicit: unpreserved run)", f"EXPIRED {RUNS}/{RUN2} (implicit: unpreserved run)",
        f"recorded {RUNS}/{RUN3} (implicit: unpreserved run)"]
    assert out[head + 4].startswith("totals: ")
    assert out[head + 5] == (f"unpreserved runs, counted in the totals above: {counted(a, b)} expired, which `research retention gc "
                             f"--unpreserved-runs` takes when the disk is full; {counted(young)} not yet expired")


def test_keep_show_names_the_record_and_how_to_keep_the_run_and_keep_or_the_owners_rm_decide_it(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    d = ended(root, lane="compute-accounting")
    through_research(monkeypatch, [d])
    assert research_main(["keep", "show", str(d), "--root", str(root)]) == 0
    io = capsys.readouterr()
    shown = json.loads(io.out)
    assert (shown["implicit"], shown["priority"], shown["expires"], shown["owner"]) == ("unpreserved run", "low", EXPIRES, "compute-accounting")
    assert io.err.strip() == "research keep show: " + KEEP.format(d=d, run=RUN)[2:]

    other = R.rm([d], approved_by="@circuits", ref="a thread", root=root, in_use=never_in_use, now=clock, node=HOST)
    assert other["actions"][0]["action"] == "refuse" and "only @compute-accounting can say yes" in other["actions"][0]["reason"]
    owner = R.rm([d], approved_by="@compute-accounting", ref="a thread", root=root, in_use=never_in_use, now=clock, node=HOST)
    assert owner["actions"][0]["action"] == "delete" and d.exists(), "its owner's yes deletes it; this was a dry run"

    assert research_main(["keep", str(d), "--root", str(root), "--owner", "@compute-accounting", "--expires", "never", "--priority",
                          "keep", "--why", "the paper's table reads it"]) == 0
    capsys.readouterr()
    found = record(d, root)
    assert (found.source, found.record["priority"], found.record["expires"]) == ("sidecar", "keep", None)
    plan = collect(root, [d], apply=True, unpreserved=True)
    assert d.exists() and by_path(plan)[f"{RUNS}/{RUN}"]["reason"] == "priority keep: never collected"
    assert plan["unpreserved"]["expired"]["paths"] == 0, "a kept run is no unpreserved run"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into every directory")
def test_an_unpreserved_run_this_user_cannot_remove_is_left_and_takes_none_of_the_passes_paths(tmp_path):
    root = tmp_path / "ws"
    stuck, free = ended(root), ended(root, RUN2, when=WRITTEN + timedelta(hours=1))
    (stuck / "out").chmod(0o555)
    try:
        plan = collect(root, [stuck, free], apply=True, unpreserved=True, max_paths=1)
        acts = by_path(plan)
        assert stuck.exists() and not free.exists() and plan["capped"] == 0, "the run that can't go did not use the pass's one path"
        assert acts[f"{RUNS}/{RUN}"]["action"] == "skip" and acts[f"{RUNS}/{RUN}"]["error"] is None
        assert acts[f"{RUNS}/{RUN}"]["reason"] == (f"{stuck / 'out'} can't be written by this user, so the tree can't be removed whole: "
                                                 "its owner makes it writable, or removes it" + KEEP.format(d=stuck, run=RUN))
    finally:
        (stuck / "out").chmod(0o755)


def clone(root: Path, name: str = RUN) -> Path:
    """The clone `research run --cwd clone` made for the run `name` on a machine, where `clone.home` puts it: beside runs/."""
    c = CL.home(root / RUNS, name, machine=True)
    (c / "verity").mkdir()
    (c / "verity/module.py").write_bytes(b"c" * UNIT)
    return written(c)


CLONE_FIRST = ("its clone {c} still stands, and the clone goes first: what removes a run's clone decides that its run is over from this "
               "directory's launch.json and status.json, so this directory goes at a pass after the clone has gone")


def test_gc_never_takes_an_unpreserved_run_whose_clone_still_stands_and_says_that_the_clone_goes_first(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    waits, free, linked = ended(root, status=RUNNING), ended(root, RUN2, when=WRITTEN + timedelta(hours=1)), ended(root, RUN3)
    c = clone(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (c.parent / RUN3).symlink_to(elsewhere, target_is_directory=True)
    assert c == root / "research" / R.CLONES / RUN and c.is_dir(), "gc looks for a clone where `clone.home` makes a machine's"
    assert R.expired(record(waits, root).record, NOW), "the run keeps its record: it is ended, unpreserved and past its 7 days"
    reason, taken = CLONE_FIRST.format(c=c) + KEEP.format(d=waits, run=RUN), nbytes(free, linked)

    plan = collect(root, [waits, free, linked], apply=True, unpreserved=True, max_paths=2)
    acts = by_path(plan)
    assert waits.exists() and c.exists(), "neither the run nor its clone went"
    assert not free.exists() and not linked.exists() and plan["capped"] == 0, "a link in scratch/ is no clone, and the run left for its " \
                                                                             "clone used none of the pass's paths"
    assert (acts[f"{RUNS}/{RUN}"]["action"], acts[f"{RUNS}/{RUN}"]["reason"], acts[f"{RUNS}/{RUN}"]["error"]) == ("skip", reason, None)
    assert plan["unpreserved"]["expired"]["paths"] == 3 and plan["unpreserved"]["deleted"] == {"paths": 2, "bytes": taken, "inodes": 12}

    flagless = by_path(collect(root, [waits], apply=True))
    assert flagless[f"{RUNS}/{RUN}"]["reason"] == reason, "the pass the timer runs says it too, so a first reading shows what waits"
    through_research(monkeypatch, [waits])
    assert research_main(["retention", "gc", "--root", str(root), "--force", "--low-pct", "0", "--unpreserved-runs"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert line("skip", waits, root, reason) in out[out.index(R.UNPRESERVED_HEAD) + 1:]
    assert out[-1] == (f"unpreserved runs, of the paths above: {counted(waits)} expired, of which this pass would delete "
                       f"{counted()}; {counted()} not yet expired")


def test_a_run_goes_at_the_next_pass_once_its_clone_is_gone(tmp_path):
    root = tmp_path / "ws"
    d, c = ended(root), clone(root)
    first = collect(root, [d], apply=True, unpreserved=True)
    assert d.exists() and by_path(first)[f"{RUNS}/{RUN}"]["action"] == "skip" and first["unpreserved"]["deleted"]["paths"] == 0
    assert CL.remove(c) is None and c.parent.is_dir(), "the clone went, as what removes a clone takes it; scratch/ itself stays"
    second = collect(root, [d], apply=True, unpreserved=True)
    assert not d.exists() and by_path(second)[f"{RUNS}/{RUN}"]["reason"] == f"deleted: {ONLY_COPY}"
    assert second["unpreserved"]["deleted"] == {"paths": 1, "bytes": by_path(first)[f"{RUNS}/{RUN}"]["bytes"], "inodes": 6}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every directory")
def test_an_unpreserved_run_stays_while_whether_its_clone_stands_cannot_be_read(tmp_path):
    root = tmp_path / "ws"
    d, c = ended(root), clone(root)
    c.parent.chmod(0o000)
    try:
        plan = collect(root, [d], apply=True, unpreserved=True)
        reason = by_path(plan)[f"{RUNS}/{RUN}"]["reason"]
    finally:
        c.parent.chmod(0o755)
    assert d.exists() and reason.startswith(f"whether its clone {c} still stands can't be read ([Errno 13] Permission denied: ")
    assert reason.endswith("), and a clone goes before its run's directory" + KEEP.format(d=d, run=RUN))


# ---- 6: a launch that was refused before it ran ---------------------------------------------------------------------------------------

KEEP_REFUSED = "; to keep it: `research keep {d} --owner @LANE --expires DATE --priority P`"
REFUSED_WHY = ("no run custody (its launch was refused before it ran): this directory is the run's only copy, and it expires 7 d after "
               "its last write")


def refused(root: Path, name: str = RUN, *, under: str = RUNS, runner: int | None = None, host: str = HOST, when: datetime = WRITTEN,
            request: dict | None = {}, **doc: object) -> Path:        # noqa: B006  (read, never changed)
    """The directory of a launch that was refused before it ran, as a machine's launch leaves one: refused.json alone, or with
    `runner` also the launch.json and the launcher.log of the runner it started on `host`; and, beside the directory it lies in,
    its request. `request` replaces fields of request.json (None: no request, a field's None removes it), `doc` those of
    refused.json. Every entry was last written at `when`."""
    d = root / under / name
    d.mkdir(parents=True)
    q = d.parent.parent / R.REQUESTS / name
    body = {"run_id": name, "reason": "the shipped source is not ready", "fault": "SourceNotReady", "at_utc": "2026-09-20T10:00:01.000Z"}
    (d / "refused.json").write_text(json.dumps({k: v for k, v in {**body, **doc}.items() if v is not None}))
    if runner is not None:
        (d / "launch.json").write_text(json.dumps({"run_id": name, "claimed_utc": "2026-09-20T10:00:00.000Z", "runner_pid": runner,
                                                   "argv": ["python3", "-m", "research"], "cwd": None, "request": str(q / "request.json"),
                                                   "hostname": host}))
        (d / "launcher.log").write_bytes(b"l" * UNIT)
    if request is not None:
        q.mkdir(parents=True, exist_ok=True)
        asked = {"schema": "research/launch-request/v0", "run_id": name, "runs_root": str(d.parent), "command": ["true"], "env": {}}
        (q / "request.json").write_text(json.dumps({k: v for k, v in {**asked, **request}.items() if v is not None}))
        written(q, when)
    return written(d, when)


def test_a_launch_that_was_refused_before_it_ran_gets_the_same_record(tmp_path, monkeypatch, capsys):
    root = tmp_path / "ws"
    alone, started = refused(root), refused(root, RUN2, runner=PID)
    request = root / "research" / R.REQUESTS / RUN
    before = entries(root)
    for d in (alone, started):
        found = record(d, root)
        assert found.source == R.UNPRESERVED and found.error is None and found.record == {
            "schema": R.SCHEMA, "path": R.relative(d, root), "node": HOST, "owner": "runner", "by": None, "created": None,
            "expires": EXPIRES, "priority": "low", "preserved": None, "regenerate": None, "why": REFUSED_WHY}
        assert R.expired(found.record, NOW)
    assert entries(root) == before and R.REQUESTS == "requests", "nothing is written, and the request is where a launch stages it"
    assert R.keeping(alone) == KEEP_REFUSED.format(d=alone)[2:], "a launch that ran nothing has nothing custody could preserve"

    left = by_path(collect(root, [alone, started], apply=True))
    assert alone.exists() and started.exists(), "the same switch: the pass the timer runs takes neither"
    assert left[f"{RUNS}/{RUN}"]["reason"] == (f"left: {ONLY_COPY}, which only --unpreserved-runs deletes" + KEEP_REFUSED.format(d=alone))
    through_research(monkeypatch, [alone, started])
    assert research_main(["retention", "report", "--root", str(root)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert sorted(x.split()[0] + " " + x.rsplit("  ", 1)[-1] for x in out[out.index(R.UNPRESERVED_HEAD) + 1:][:2]) == [
        f"EXPIRED {RUNS}/{RUN} (implicit: unpreserved run)", f"EXPIRED {RUNS}/{RUN2} (implicit: unpreserved run)"]

    plan = collect(root, [alone, started], apply=True, unpreserved=True)
    assert not alone.exists() and not started.exists() and by_path(plan)[f"{RUNS}/{RUN}"]["reason"] == f"deleted: {ONLY_COPY}"
    assert plan["unpreserved"]["deleted"]["paths"] == 2 and plan["unpreserved"]["deleted"]["inodes"] == 2 + 4
    assert {k: v for k, v in entries(root).items() if R.REQUESTS in k} == {k: v for k, v in before.items() if R.REQUESTS in k}
    assert (request / "request.json").exists(), "the request is not the collector's: it goes after its run directory, by its own rule"


def test_a_refused_launch_whose_runner_is_not_known_gone_has_no_record(tmp_path, table):
    root = tmp_path / "ws"
    d = refused(root, runner=PID)
    assert record(d, root).source == R.UNPRESERVED, "no process has the runner's pid"

    process(table, PID, TICKS)
    assert record(d, root) is None, "launch.json records no start time: a pid some process has now is not known gone"
    (table / str(PID) / "stat").unlink()
    assert record(d, root) is None, "whatever that process is"
    (table / str(PID)).rmdir()
    assert record(d, root).source == R.UNPRESERVED
    (table / "1" / "stat").unlink()
    assert record(d, root) is None, "a process table that does not show pid 1 shows nobody gone"
    process(table, 1, 1)

    assert record(refused(root, RUN2, runner=PID, host="another-host"), root) is None, "the pid was another host's"
    for n, launch in enumerate(({"runner_pid": str(PID), "hostname": HOST}, {"runner_pid": True, "hostname": HOST}, {"hostname": HOST},
                                {"runner_pid": PID}, [PID], "{")):
        odd = refused(root, f"r20260924-100000-000{n}")
        (odd / "launch.json").write_text(launch if isinstance(launch, str) else json.dumps(launch))
        assert record(written(odd), root) is None, f"a launch.json that names no runner of this host says nothing of one: {launch}"


def test_an_empty_claim_and_a_directory_that_cannot_show_it_is_a_refused_launchs_have_no_record(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    claim = refused(root)
    (claim / "refused.json").unlink()
    assert not list(claim.iterdir()) and record(written(claim), root) is None, "an empty directory is a claim: its launch may be starting"
    (claim / "launcher.log").write_bytes(b"l")
    (claim / "launch.json").write_text(json.dumps({"run_id": RUN, "runner_pid": PID, "hostname": HOST}))
    assert record(written(claim), root) is None, "a runner was started and nothing says the launch was refused"

    cases = {
        "its refused.json names another run": dict(run_id=RUN3),
        "its refused.json names no run": dict(run_id=None),
        "it has no request": dict(request=None),
        "its request names another run": dict(request={"run_id": RUN3}),
        "its request names no run": dict(request={"run_id": None}),
        "its request places the run in another runs root: a copy, or a run moved aside": dict(request={"runs_root": str(root / "other")}),
        "its request names no runs root": dict(request={"runs_root": None}),
        "a relative runs root places it nowhere, wherever the collector runs": dict(request={"runs_root": RUNS}),
    }
    (root / "other").mkdir()
    monkeypatch.chdir(root)
    for n, (why, case) in enumerate(cases.items()):
        assert record(refused(root, f"r20260925-100000-000{n}", **case), root) is None, why
    for n, (name, body) in enumerate([("refused.json", "{"), ("refused.json", json.dumps([RUN])), ("request.json", "{"),
                                      ("request.json", json.dumps(["r20260926-100000-0003"]))]):
        d = refused(root, f"r20260926-100000-000{n}")
        (d if name == "refused.json" else root / "research" / R.REQUESTS / d.name).joinpath(name).write_text(body)
        assert record(written(d), root) is None, f"a {name} that is not a JSON object names nothing: {body}"

    named = refused(root, "results", run_id="results", request={"run_id": "results"})
    assert record(named, root) is None, "not named as a run"
    (named / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": "results", "preserved": False, "host": HOST}))
    assert record(written(named), root) is None, "nor with a marker that says custody preserved nothing"
    aside = refused(root, "r20260927-100000-0001", under="research/runs-aside", request={"runs_root": str(root / RUNS)})
    assert record(aside, root) is None, "moved aside: its request names the runs root it was claimed in"
    status = refused(root, "r20260927-100000-0002")
    (status / "status.json").write_text(json.dumps({"state": "done"}))
    assert record(written(status), root) is None, "a status.json and no job.json: not a launch refused before it ran"
    launcher = refused(root, "r20260927-100000-0003")
    (launcher / CU.LAUNCHER_RECORD).write_text("{}")
    assert record(written(launcher), root) is None, "the launcher's copy of a run on another machine"
    outer = ended(root, "r20260927-100000-0004")
    inner = refused(root, "r20260927-100000-0005", under=f"{RUNS}/r20260927-100000-0004/out")
    assert (outer / R.REQUESTS / inner.name / "request.json").exists(), "with a request beside the directory it lies in"
    assert record(inner, root) is None and record(written(outer), root).source == R.UNPRESERVED, "inside a run's directory: its data"
    kept_above = refused(root, "r20260927-100000-0006")
    R.keep(root / "research", root=root, owner="@infra", expires=None, priority="keep", now=clock, node=HOST)
    assert record(written(kept_above), root) is None and R.find(kept_above, root).source == "sidecar", "a record above it wins"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_a_refused_launch_whose_records_cannot_be_read_is_listed_with_what_could_not_be_read_and_is_never_deleted(tmp_path):
    root = tmp_path / "ws"
    d = refused(root, runner=PID)
    for unreadable in (root / "research" / R.REQUESTS / RUN / "request.json", d / "refused.json", d / "launch.json"):
        error = f"whether the run of {d} ended can't be read (cannot read {unreadable}: [Errno 13] Permission denied: '{unreadable}')"
        unreadable.chmod(0)
        try:
            found = record(d, root)
            assert (found.record, found.source, found.error) == (None, R.UNPRESERVED, error), "what can't be read may say it is live"
            act = by_path(collect(root, [d], apply=True, unpreserved=True))[f"{RUNS}/{RUN}"]
            assert d.exists() and (act["action"], act["state"]) == ("skip", "invalid") and act["reason"].startswith(f"no record: {error}")
        finally:
            unreadable.chmod(0o644)
    assert record(d, root).record["expires"] == EXPIRES


def test_a_refused_launchs_record_follows_its_last_write_its_custody_key_and_its_clone(tmp_path):
    root = tmp_path / "ws"
    d = refused(root, runner=PID, fault="CloneRefused")
    young = NOW - timedelta(days=2)
    os.utime(d / "launcher.log", (young.timestamp(), young.timestamp()))
    rec = record(d, root).record
    assert rec["expires"] == R.fmt_utc(young + R.UNPRESERVED_RUN_EXPIRY) and not R.expired(rec, NOW), "7 days after its last write"
    written(d)

    cred = root / "research" / R.REQUESTS / RUN / CU.DIR_NAME / CU.CRED_FILE
    cred.parent.mkdir()
    cred.write_text(json.dumps({"expires_utc": "2026-10-02T06:00:00Z"}))
    assert record(d, root).record["expires"] == "2026-10-02T06:00:00Z", "a custody key its refusal did not delete bounds it as a run's"
    cred.write_text("{")
    assert record(d, root).error == f"the expiry of the custody key {cred} can't be read, so whether the runner of {d} still publishes is unknown"
    cred.unlink()

    c = clone(root)
    act = by_path(collect(root, [d], apply=True, unpreserved=True))[f"{RUNS}/{RUN}"]
    assert d.exists() and act["reason"] == CLONE_FIRST.format(c=c) + KEEP_REFUSED.format(d=d), "its clone goes first, as a run's"
    assert CL.remove(c) is None and collect(root, [d], apply=True, unpreserved=True)["unpreserved"]["deleted"]["paths"] == 1
    assert not d.exists()


# ---- 5: the constant ------------------------------------------------------------------------------------------------------------------

def test_the_readme_states_the_rule_with_the_constants_value_and_both_ways_to_keep_a_run():
    text = (REPO / "tools/research/README.md").read_text()
    section = text[text.index("### Retention of node data"):text.index("### Slack")]
    days = R.UNPRESERVED_RUN_EXPIRY // timedelta(days=1)
    rule = (f"A finished or runner-dead run that custody did not preserve gets a low-priority retention record that expires {days} days "
            "after the run directory's last write, so the collector takes it")
    assert rule in " ".join(section.split()), "the README states the rule with UNPRESERVED_RUN_EXPIRY's value"
    assert "`UNPRESERVED_RUN_EXPIRY`" in section and f"before the {days} days are over" in " ".join(section.split())
    for needed in ("`research keep RUNDIR ...`", "`research data custody RUN --publish`", "`--unpreserved-runs`", "[--unpreserved-runs]"):
        assert needed in section, f"the retention section names {needed}"
    assert f"{days} d after its last write" in R.UNPRESERVED_HEAD and f"expires {R.UNPRESERVED_RUN_EXPIRY.days} days" in rule
    assert f"one whose clone stands waits (`{R.CLONES}/<run id>` goes first)" in " ".join(section.split())
    assert "(`UNPRESERVED_RUN_EXPIRY`). So does a refused launch." in " ".join(section.split())
