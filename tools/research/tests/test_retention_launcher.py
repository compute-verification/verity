"""The record of a launcher's copy of a run on another machine (research.retention, gc's --launcher-copies): which directories
are such copies, when one expires, and what the remote must say before an --apply deletes one. The machine a run ran on is a
directory whose run record and attempt a store pushes to an fs remote; the launcher's copy is what `fetch` left of it, with the
launcher's remote.json, each entry's mtime set by hand; the clock and the disk are fakes."""
from __future__ import annotations

import json
import shutil
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from test_retention import NOW, REPO, UNIT, Disk, aged, by_path, clock, never_in_use, stored, vault_store  # noqa: F401  (a fixture)
from test_retention_unpreserved import written

from research import retention as R
from research.cli import main as research_main
from research.store import custody as CU
from research.store import vocab
from research.store.local import LocalStore, object_key

HOST = socket.gethostname()
MACHINE = "vy-elsewhere"
RUN, RUN2, RUN3, RUN4 = "r20260920-100000-ab12", "r20260921-100000-cd34", "r20260922-100000-ef56", "r20260923-100000-0a0b"
OLD = NOW - R.LAUNCHER_COPY_EXPIRY - timedelta(minutes=1)      # a last write just past the 48 hours
RECORDS = ("job.json", "status.json", "events.jsonl", "launcher.log", "stdout.log")


def ran(tmp_path: Path, run: str = RUN, *, state: str | None = "done", record: bool = True, big: bool = False) -> tuple[Path, str | None]:
    """A run as its machine leaves it once its workload has exited: its directory, its run record pushed to the remote a day
    before NOW, and with `state` its attempt there in that state, naming the record when `record`. `big` adds an input that
    custody trims from the record. Returns the machine's directory and the record's id."""
    d = tmp_path / "machine" / "runs" / run
    (d / "out" / "work" / "0").mkdir(parents=True)
    (d / "inputs").mkdir()
    (d / "job.json").write_text(json.dumps({"run": {"id": run, "runs_root": str(d.parent)}, "environment": {"hostname": MACHINE},
                                            "ownership": {"lane": "infra"}}))
    (d / "status.json").write_text(json.dumps({"state": state or "done", "rc": 0, "transitions": [{"state": "running"}, {"state": "done"}]}))
    (d / "events.jsonl").write_text('{"event": "start"}\n{"event": "end"}\n')
    (d / "launcher.log").write_text("started\nexited 0\ncustody: pushing\n")
    (d / "stdout.log").write_text(f"{run} out\n")
    (d / "out" / "results.json").write_text(json.dumps({"run": run, "pass": True}))
    (d / "out" / "work" / "0" / "tree.txt").write_text(f"the tree of task 0 of {run}\n")
    (d / "inputs" / "manifest.json").write_text(json.dumps({"tasks": [run]}))
    st = LocalStore(tmp_path / "machine" / "store")
    if big:
        (d / "inputs" / "reads.json").write_text("r" * 4096)
        cap, CU.UNDECLARED_CAP_BYTES = CU.UNDECLARED_CAP_BYTES, 1024
    try:
        art = CU.put_run_record(st, d, run)
    finally:
        if big:
            CU.UNDECLARED_CAP_BYTES = cap
    assert st.push(art, preserve=True)["preserved"]
    aged(Path(st.remote.root), art, days=1)
    if state:
        st.put_attempt({"schema": "research/attempt/v0.1", "id": run, "tool": {"name": "check", "version": "1"}, "derivation": "drv:" + "a" * 64,
                        "reuse_key": None, "argv": ["x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
                        "source": {"commit": "c" * 40, "dirty": False}, "closure": None,
                        "execution": {"start_utc": "2026-09-20T10:00:05Z", "host": MACHINE, "project": "verity", "campaign": "check"},
                        "cost": None, "status": {"state": state, "rc": 0, "validation": "passed"},
                        "outputs": {"run_record": art} if record else {}, "telemetry": None})
        st.push_attempt(run)
    return d, art


def copy(root: Path, machine_dir: Path, *, when: datetime = OLD, mid_run: bool = False, whole: bool = True) -> Path:
    """The launcher's copy of the run of `machine_dir` under `root`, last written at `when`: its remote.json, the run's records and
    its `out/`, and with `whole` its `inputs/` too, as `fetch --all` brings them. `mid_run`: the records as they were fetched
    while the run still ran, so its status reads `running` and its logs are shorter than the record's."""
    d = root / machine_dir.name
    d.mkdir(parents=True)
    (d / CU.LAUNCHER_RECORD).write_text(json.dumps({"schema": "research/remote/v0.1", "run_id": d.name, "machine": MACHINE, "project": "verity",
                                                    "phase": "running" if mid_run else "done", "created_utc": "2026-09-20T10:00:00.123Z"}))
    for name in RECORDS:
        shutil.copy(machine_dir / name, d / name)
    shutil.copytree(machine_dir / "out", d / "out")
    if whole:
        shutil.copytree(machine_dir / "inputs", d / "inputs")
    if mid_run:
        (d / "status.json").write_text(json.dumps({"state": "running", "transitions": [{"state": "running"}]}))
        (d / "events.jsonl").write_text('{"event": "start"}\n')
        (d / "launcher.log").write_text("started\n")
    return written(d, when)


def collect(root: Path, **kw: object) -> dict:
    """One forced pass over `root` with --launcher-copies, --apply unless told otherwise, and the real check against the remote."""
    kw = {"now": clock, "usage": Disk(100 * UNIT, 90 * UNIT, []), "in_use": never_in_use, "node": HOST, "force": True, "low_pct": 0,
          "apply": True, "launcher": True, "copy_check": R.launcher_check(clock), **kw}
    return R.gc(root, **kw)


def blob_of(tmp_path: Path, machine_dir: Path, rel: str) -> Path:
    """Where the remote keeps the blob of the run's file `rel`."""
    import hashlib

    return tmp_path / "vault" / object_key(hashlib.sha256((machine_dir / rel).read_bytes()).hexdigest())


# ---- the copy that goes -----------------------------------------------------------------------------------------------------------------

def test_a_copy_of_an_ended_stored_run_goes_48_hours_after_its_last_write(tmp_path, vault_store):
    """The rule's one deletion: the remote holds the run's attempt, ended, and a run record with every file of the copy and
    every blob, and the copy was last written over 48 hours ago."""
    root = tmp_path / "launcher"
    machine_dir, art = ran(tmp_path)
    d = copy(root, machine_dir)
    found = R.record_at(d, RUN, launcher=True)
    assert found.source == R.LAUNCHER and R.LAUNCHER_COPY_EXPIRY == timedelta(hours=48)
    assert found.record == {"schema": R.SCHEMA, "path": RUN, "node": HOST, "owner": "infra", "by": None, "created": "2026-09-20T10:00:00Z",
                            "expires": R.fmt_utc(OLD + R.LAUNCHER_COPY_EXPIRY), "priority": "normal", "preserved": None, "regenerate": None,
                            "why": f"the launcher's copy of a run on {MACHINE}: it expires 48 h after its last write, and goes once the "
                                   "remote holds the run's attempt, ended, and a run record with its files"}
    plan = collect(root)
    a = by_path(plan)[RUN]
    assert not d.exists() and (a["action"], a["stored"], a["error"]) == ("delete", "verified", None) and plan["stored"] == stored(verified=1)
    assert a["reason"] == (f"deleted: normal, expired {found.record['expires']}, a launcher's copy of a run (its run ended done, and the "
                           f"remote holds its run record {art}: every blob at its size)")
    assert machine_dir.exists() and plan["deletes_launcher_copies"] is True


def test_a_copy_that_reads_running_goes_once_the_remote_says_its_run_ended(tmp_path, vault_store):
    """A copy fetched while its run still ran reads `running` for ever, and its logs are shorter than the record's: what says the
    run is over is the attempt its machine published, never the copy."""
    root = tmp_path / "launcher"
    machine_dir, _ = ran(tmp_path, state="failed")
    d = copy(root, machine_dir, mid_run=True, whole=False)
    assert json.loads((d / "status.json").read_text())["state"] == "running"
    assert (d / "launcher.log").stat().st_size < (machine_dir / "launcher.log").stat().st_size
    a = by_path(collect(root))[RUN]
    assert not d.exists() and a["stored"] == "verified" and "its run ended failed" in a["reason"]


# ---- the copies that stay ---------------------------------------------------------------------------------------------------------------

def test_a_copy_written_within_48_hours_stays_whatever_the_remote_holds(tmp_path, vault_store):
    """Too new: the 48 hours count from the newest write anywhere in the copy, so one file fetched late keeps the whole copy,
    and the remote is not asked about it."""
    root = tmp_path / "launcher"
    young = copy(root, ran(tmp_path)[0], when=NOW - timedelta(hours=47, minutes=59))
    late = copy(root, ran(tmp_path, RUN2)[0])
    written(late / "out" / "work" / "0" / "tree.txt", NOW - timedelta(hours=1))
    plan = collect(root, copy_check=lambda p: pytest.fail(f"the remote was asked about {p}"))
    acts = by_path(plan)
    assert young.exists() and late.exists() and plan["stored"] == stored()
    assert acts[RUN]["reason"] == f"not expired: expires {R.fmt_utc(NOW + timedelta(minutes=1))}"
    assert acts[RUN2]["reason"] == f"not expired: expires {R.fmt_utc(NOW + timedelta(hours=47))}"


@pytest.mark.parametrize("local_state", ["done", "running"])
def test_a_copy_of_a_run_that_still_runs_stays_however_old_and_whatever_its_own_status_says(tmp_path, vault_store, local_state):
    """A list made from the launcher's files alone would take this one: the copy is old and may even read `done`. Its run has
    published no attempt, which its machine does when the workload exits, so nothing says it is over."""
    root = tmp_path / "launcher"
    machine_dir, _ = ran(tmp_path, state=None)
    d = copy(root, machine_dir, mid_run=local_state == "running", when=NOW - timedelta(days=30))
    plan = collect(root)
    a = by_path(plan)[RUN]
    assert d.exists() and (a["action"], a["stored"]) == ("skip", "unknown") and plan["stored"] == stored(unknown=1)
    assert a["reason"] == (f"the remote doesn't say that its run is over and stored: the remote has no attempt for {RUN}, which the run's "
                           "machine publishes once its workload has exited: nothing says the run is over")


def test_a_copy_whose_attempt_is_not_of_a_run_that_ended_stays(tmp_path, vault_store):
    """An attempt published by force while its runner was thought dead freezes the run as `running`: it does not say the run
    is over. And one published without custody names no run record, so the remote holds none of the copy's files."""
    root = tmp_path / "launcher"
    forced = copy(root, ran(tmp_path, state="running")[0])
    bare = copy(root, ran(tmp_path, RUN2, record=False)[0])
    plan = collect(root)
    acts = by_path(plan)
    assert forced.exists() and bare.exists() and plan["stored"] == stored(unknown=2)
    assert acts[RUN]["reason"].endswith(f"the attempt the remote holds for {RUN} is 'running', which is not a run that is over")
    assert acts[RUN2]["reason"].endswith(f"the attempt of {RUN2} names no run record (published without custody), so the remote holds none "
                                         "of its files")


def test_a_copy_that_a_file_in_the_jobs_dir_names_stays(tmp_path, vault_store):
    """How a service keeps the copies it still needs: it lists their run ids in a file of a --jobs-dir. A copy the file names
    stays, one it doesn't goes, and a list that can't be read keeps every copy."""
    root, names = tmp_path / "launcher", tmp_path / "named-runs"
    named, free = copy(root, ran(tmp_path)[0]), copy(root, ran(tmp_path, RUN2)[0])
    names.mkdir()
    (names / "runs").write_text(f"{RUN}\n{RUN3}\n")
    plan = collect(root, in_use=R.JobScripts([names], root))
    acts = by_path(plan)
    assert named.exists() and not free.exists() and plan["stored"] == stored(verified=1)
    assert acts[RUN]["reason"] == f"in use: {names / 'runs'} names it"
    (names / "runs").unlink()
    names.rmdir()
    plan = collect(root, in_use=R.JobScripts([names], root))
    assert named.exists() and by_path(plan)[RUN]["reason"].startswith("in use: cannot read the job scripts") and plan["stored"] == stored()


def test_a_copy_named_or_written_while_the_remote_is_asked_stays(tmp_path, vault_store):
    """Asking the remote about a copy's blobs takes a while, and the answer is about what the copy held and who named it before.
    A copy the service names meanwhile, and one a fetch writes into meanwhile, stay though the remote said yes; the pass after
    takes the one nobody names once its 48 hours have run again."""
    root, names = tmp_path / "launcher", tmp_path / "named-runs"
    named, grown = copy(root, ran(tmp_path)[0]), copy(root, ran(tmp_path, RUN2)[0])
    names.mkdir()
    (names / "runs").write_text("")
    ask = R.launcher_check(clock)

    def meanwhile(p: Path) -> R.Stored:
        found = ask(p)
        if p == named:
            (names / "runs").write_text(f"{RUN}\n")
        else:
            (p / "out" / "late.json").write_text("{}")
            written(p, NOW - timedelta(minutes=1))
        return found

    plan = collect(root, in_use=R.JobScripts([names], root), copy_check=meanwhile)
    acts = by_path(plan)
    assert named.exists() and grown.exists() and plan["stored"] == stored(verified=2), "the remote said yes to both"
    assert acts[RUN]["action"] == "skip" and acts[RUN]["reason"] == f"in use: {names / 'runs'} names it"
    assert acts[RUN2]["action"] == "skip" and acts[RUN2]["reason"] == "it was written while the remote was asked about it: run gc again"
    (names / "runs").write_text("")
    assert by_path(collect(root, in_use=R.JobScripts([names], root)))[RUN]["action"] == "delete" and not named.exists() and grown.exists()


def test_a_deletion_cut_short_leaves_a_copy_the_next_pass_still_takes(tmp_path, vault_store, monkeypatch):
    """A copy has its record by its remote.json alone. A deletion that stops partway (the pass killed at its time limit, a
    reboot) and had taken that file first would leave a directory with no record, which no pass ever deletes. It goes last:
    what is left is still a copy, and it goes once its 48 hours have run again."""
    root = tmp_path / "launcher"
    d = copy(root, ran(tmp_path)[0])
    rmtree = shutil.rmtree

    def cut_short(path, *a, **kw):
        if Path(path) == d / "out":
            rmtree(d / "out" / "work")
            raise OSError("the pass was stopped here")
        return rmtree(path, *a, **kw)

    monkeypatch.setattr(R.shutil, "rmtree", cut_short)
    a = by_path(collect(root))[RUN]
    assert a["action"] == "delete" and a["error"] == "OSError: the pass was stopped here"
    left = sorted(p.name for p in d.iterdir())
    assert left == ["out", CU.LAUNCHER_RECORD, "status.json", "stdout.log"], "what sorts before `out` went, and remote.json is still there"
    monkeypatch.setattr(R.shutil, "rmtree", rmtree)
    assert R.record_at(d, RUN, launcher=True).source == R.LAUNCHER
    written(d, OLD)
    assert by_path(collect(root))[RUN]["action"] == "delete" and not d.exists()


def test_a_copy_holding_a_file_the_run_record_lacks_stays(tmp_path, vault_store):
    """The copy may hold the only bytes of a file: one something wrote into it under `out/`, one whose size is not the record's,
    and one custody trimmed, which the record only lists. A log at the copy's top that the machine went on writing is the
    record's by its path, so it keeps nothing."""
    root = tmp_path / "launcher"
    extra = copy(root, ran(tmp_path)[0])
    (extra / "out" / "notes.txt").write_text("written on the launcher")
    resized = copy(root, ran(tmp_path, RUN2)[0])
    (resized / "out" / "results.json").write_text("{}")
    trimmed = copy(root, ran(tmp_path, RUN3, big=True)[0])
    shorter = copy(root, ran(tmp_path, RUN4)[0])
    (shorter / "stdout.log").write_text("")
    for d in (extra, resized, trimmed, shorter):
        written(d, OLD)
    plan = collect(root)
    acts = by_path(plan)
    assert [d.exists() for d in (extra, resized, trimmed, shorter)] == [True, True, True, False]
    assert plan["stored"] == stored(uncovered=3, verified=1)
    for run, rel in ((RUN, "out/notes.txt"), (RUN2, "out/results.json"), (RUN3, "inputs/reads.json")):
        assert acts[run]["reason"].endswith(f"doesn't hold all of it: 1 file(s) under it are not in the artifact at their size, e.g. {rel}")


def test_a_copy_whose_run_record_lost_a_blob_stays_unless_the_stores_own_rule_gave_it_up(tmp_path, vault_store):
    """The remote holds the manifest and lacks a blob. Under 7 days old nothing explains that, so the copy may be the only one
    and stays; from 7 days on the store's time to live dropped it, and a withdrawal drops one at any age: the store gave those
    up by its own rule, so their copies go. A blob of another size is never the record's."""
    root, vault = tmp_path / "launcher", tmp_path / "vault"
    runs = {}
    for run, days in ((RUN, 6.9), (RUN2, 7), (RUN3, 1), (RUN4, 1)):
        machine_dir, art = ran(tmp_path, run)
        runs[run] = (copy(root, machine_dir), art)
        if run == RUN4:
            blob_of(tmp_path, machine_dir, "stdout.log").write_text("another file")
        else:
            blob_of(tmp_path, machine_dir, "out/work/0/tree.txt").unlink()
        aged(vault, art, days=days)
    elsewhere = LocalStore(tmp_path / "elsewhere")
    elsewhere.add_label(runs[RUN3][1], vocab.WITHDRAWN, "values that must not be shared", "bc-elsewhere", ts="2026-09-30T00:00:00Z")
    assert not elsewhere.push_labels([runs[RUN3][1]])["failed"]
    plan = collect(root)
    acts = by_path(plan)
    assert [runs[r][0].exists() for r in (RUN, RUN2, RUN3, RUN4)] == [True, False, False, True]
    assert plan["stored"] == stored(mismatch=2, expired=1, withdrawn=1)
    assert acts[RUN]["reason"].endswith(": absent, which neither a withdrawal nor its kind's time to live explains")
    assert acts[RUN2]["reason"].endswith("expired on the remote: a run-record/v1 lives 7 d there, and its manifest is 7.0 d old)")
    assert acts[RUN3]["reason"].endswith("was withdrawn from the remote: 'values that must not be shared', by bc-elsewhere)")
    assert ": remote size 12 != " in acts[RUN4]["reason"]


def test_a_pass_that_cannot_ask_the_remote_keeps_every_copy_and_exits_1(tmp_path, vault_store, monkeypatch, capsys):
    """Without a remote to ask (no key in the pass's environment) nothing says a run is over or stored: the copy stays and the
    pass fails, so the unit that runs it shows as failed instead of quietly deleting nothing."""
    monkeypatch.setattr(R, "utc_now", clock)
    monkeypatch.setattr(R, "DEFAULT_JOBS_DIRS", ())
    root = tmp_path / "launcher"
    d = copy(root, ran(tmp_path)[0])
    (tmp_path / "store.toml").write_text("")
    rc = research_main(["retention", "gc", "--root", str(root), "--launcher-copies", "--force", "--low-pct", "0", "--apply"])
    out = capsys.readouterr().out
    assert rc == 1 and d.exists()
    assert "the remote doesn't say that its run is over and stored: no remote can be asked from here (none is configured)" in out
    assert out.splitlines()[-1].endswith("0 uncovered, 0 mismatch, 0 unknown, 1 unreachable (kept); a copy that could not be asked about "
                                         "fails the pass")


# ---- what the flag changes, and what is a copy ------------------------------------------------------------------------------------------

def test_without_the_flag_a_launchers_copy_has_no_record_and_is_never_deleted(tmp_path, vault_store):
    """The hourly collector on every node runs without --launcher-copies: for it nothing changes. A copy has no record, a
    forced --apply leaves it, and neither the report nor `research keep show` gives it one."""
    root = tmp_path / "launcher"
    d = copy(root, ran(tmp_path)[0])
    assert R.record_at(d, RUN) is None and R.find(d, root) is None and R.implicit(d, RUN) is None
    assert [(r.path, r.record) for r in R.scan(root).rows] == [(RUN, None)]
    plan = collect(root, launcher=False, copy_check=lambda p: pytest.fail(f"the remote was asked about {p}"))
    assert d.exists() and plan["deletes_launcher_copies"] is False
    assert by_path(plan)[RUN]["reason"] == "no record: a path without one is never deleted (`research keep PATH ...` gives it one)"


def test_a_dry_run_lists_what_would_go_and_asks_no_remote(tmp_path, vault_store):
    """The list a dry run prints is made from the launcher's files alone, so each line says what an --apply still asks."""
    root = tmp_path / "launcher"
    d = copy(root, ran(tmp_path, state=None)[0])
    plan = collect(root, apply=False, copy_check=lambda p: pytest.fail(f"the remote was asked about {p}"))
    a = by_path(plan)[RUN]
    assert d.exists() and a["action"] == "delete" and plan["stored"] == stored()
    assert a["reason"].endswith("a launcher's copy of a run, if the remote says just before an --apply deletes it that the run is over and stored")


def test_a_copy_someone_keeps_stays_until_they_let_it_go(tmp_path, vault_store):
    """`research keep COPY` is how a person keeps a failed run's tree on the launcher past the 48 hours: the sidecar wins over
    the implicit record, whatever the remote holds. With the sidecar removed the copy is the launcher's again, and goes."""
    root = tmp_path / "launcher"
    d = copy(root, ran(tmp_path)[0])
    R.keep(d, root=root, owner="@infra", expires="2026-10-08T00:00:00Z", priority="high", why="looking at task 0", now=clock, node=HOST)
    assert R.record_at(d, RUN, launcher=True).source == "sidecar"
    plan = collect(root, copy_check=lambda p: pytest.fail(f"the remote was asked about {p}"))
    assert d.exists() and by_path(plan)[RUN]["reason"] == "priority high: never collected"
    (d / R.SIDECAR).unlink()
    written(d, OLD)
    assert not (collect(root) and d.exists())


def test_only_a_launchers_copy_of_a_run_gets_the_record(tmp_path, vault_store):
    """What the flag never touches: a directory not named as a run id, one whose remote.json is not the launcher's record of
    that run, a run's own directory on the machine it ran on, and anything under a recorded path or inside another run."""
    root = tmp_path / "launcher"
    machine_dir, _ = ran(tmp_path)
    launched = json.loads((copy(root, machine_dir) / CU.LAUNCHER_RECORD).read_text())
    shutil.copytree(root / RUN, tmp_path / "a-copy")

    def record(d: Path) -> R.Found | None:
        return R.record_at(d, R.relative(d, root), launcher=True)

    def other(name: str, under: str = "", **remote: object) -> Path:
        d = root / under / name
        shutil.copytree(tmp_path / "a-copy", d)
        (d / CU.LAUNCHER_RECORD).write_text(json.dumps({**launched, "run_id": name, **remote}))
        return written(d, OLD)

    assert record(root / RUN).source == R.LAUNCHER
    assert record(other("results")) is None, "not named as a run"
    assert record(other(RUN2, run_id=RUN3)) is None, "its remote.json is another run's"
    assert record(other(RUN3, schema="research/job/v1")) is None, "its remote.json is not a launcher's record"
    assert record(other(RUN4, machine=None)) is None, "its remote.json names no machine"
    empty = other("r20260924-100000-0001")
    (empty / CU.LAUNCHER_RECORD).write_text("{}")
    assert record(empty) is None
    here = other("r20260924-100000-0002")
    (here / "job.json").write_text(json.dumps({"run": {"id": here.name, "runs_root": str(root)}, "environment": {"hostname": HOST}}))
    assert record(here) is None, "the run's own directory on this host: never a copy"
    assert record(other("r20260924-100000-0003", under=f"{RUN}/out")) is None, "inside another run's directory: that run's data"
    (root / "area").mkdir()
    R.keep(root / "area", root=root, owner="@infra", expires=None, priority="keep", now=clock, node=HOST)
    assert record(other("r20260924-100000-0004", under="area")) is None, "under a recorded path: that record covers it"
    link = root / "r20260924-100000-0005"
    link.symlink_to(root / RUN)
    assert record(link) is None, "a link to a copy"


def test_a_copy_with_a_fetched_custody_marker_goes_by_the_launchers_rule(tmp_path, vault_store):
    """`fetch --all` brings the machine's `.custody` into the copy. Without the flag that marker gives the copy the machine's
    record, which this host never collects; with it the copy is the launcher's, and the marker is no file the record lacks."""
    root = tmp_path / "launcher"
    machine_dir, art = ran(tmp_path)
    d = copy(root, machine_dir)
    (d / CU.MARKER).write_text(json.dumps({"schema": CU.SCHEMA, "run_id": RUN, "run_record": art, "preserved": True,
                                           "verified_utc": "2026-09-21T00:00:00Z", "host": MACHINE}))
    written(d, OLD)
    assert R.record_at(d, RUN).source == "custody"
    plan = collect(root, launcher=False)
    assert d.exists() and by_path(plan)[RUN]["reason"] == f"recorded on {MACHINE}, and this is {HOST}: `research keep` it here"
    assert not (collect(root) and d.exists())


def test_the_docs_state_the_rule_and_its_one_number():
    readme = " ".join((REPO / "tools/research/README.md").read_text().split())
    for needed in ("[--launcher-copies]", "`--launcher-copies`", "48 hours after its last write", "vy-launcher-copies-gc"):
        assert needed in readme, f"tools/research/README.md no longer states {needed!r}"
    assert "LAUNCHER_COPY_EXPIRY (48 hours)" in R.__doc__ and "[--launcher-copies]" in R.__doc__
