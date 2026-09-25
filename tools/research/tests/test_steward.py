"""The steward off the laptop (steward.py, step e): liveness from the notes history and origin's lane/* tips (E1, E2), relaunch from
origin when the dead lane's machine is gone (E5), the steward lease (E3: a second copy refuses and names the holder), and the service
definitions (E4).  Local git repos and a lease file or an in-memory S3 double; no network."""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import io
import json
import os
import plistlib
import subprocess
import urllib.error
from pathlib import Path

import pytest

from research import notes, steward

NOW = dt.datetime(2026, 9, 23, 21, 0, tzinfo=dt.timezone.utc)
FRONT = "---\nlane: {lane}\nkind: report\ncreated: 2026-09-23T17:00Z\nstatus: open\n---\n\n"


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}).stdout.strip()


def _init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    return path


def _write(root: Path, rel: str, text: str, minutes_ago: float) -> Path:
    """Write, set the mtime, and commit with that author time (what a machine that syncs every write leaves behind)."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    t = (NOW - dt.timedelta(minutes=minutes_ago)).timestamp()
    os.utime(p, (t, t))
    stamp = email.utils.formatdate(t, usegmt=True)
    _git(root, "add", rel)
    _git(root, "commit", "-qm", rel, env={"GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp})
    return p


@pytest.fixture
def notes_pair(tmp_path) -> tuple[Path, Path]:
    """(the laptop's notes, a fresh clone of them: every mtime is the checkout time)."""
    laptop = _init(tmp_path / "laptop")
    _write(laptop, "lanes/live/20260923T1700Z-report-live.md", FRONT.format(lane="live") + "CHECKPOINT abc1234 (20:50Z) [open] going\n", 10)
    _write(laptop, "lanes/dead/20260923T1700Z-report-dead.md", FRONT.format(lane="dead") + "CHECKPOINT def5678 (19:55Z) [open] last\n", 65)
    _write(laptop, "lanes/dead/20260923T2030Z-handoff-coordinator.md", "# please stop\n", 30)
    _write(laptop, "lanes/dead/binding.json", json.dumps({"final": "22:00Z", "branch": "lane/dead"}) + "\n", 120)
    _write(laptop, "lanes/live/20260923T1600Z-report-live-old.md", FRONT.format(lane="live") + "CHECKPOINT 0000000 (16:00Z) old\n", 300)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(laptop), str(clone)], check=True)
    return laptop, clone


def _rows(root: Path) -> dict[str, dict]:
    keep = ("state", "age_min", "stale", "checkpoint_at", "report", "inbox")
    out = {}
    for r in notes.status(root, stale_min=30, since_hours=8, now=NOW):
        out[r["lane"]] = {k: (Path(r[k]).name if k == "report" else r.get(k)) for k in keep}
    return out


def test_e1_a_fresh_clone_gives_the_laptops_rows(notes_pair, monkeypatch):
    laptop, clone = notes_pair
    monkeypatch.delenv(steward.LIVENESS_ENV, raising=False)
    on_laptop = _rows(laptop)
    assert on_laptop["dead"]["stale"] and not on_laptop["live"]["stale"]
    assert on_laptop["live"]["report"] == "20260923T1700Z-report-live.md"
    stale_by_mtime = _rows(clone)                           # the clone's mtimes are now: every lane looks fresh (or out of the window)
    assert stale_by_mtime != on_laptop
    monkeypatch.setenv(steward.LIVENESS_ENV, "git")
    assert _rows(clone) == on_laptop
    items = notes.inbox(clone, "dead")
    assert [x["path"].name for x in items] == ["20260923T2030Z-handoff-coordinator.md"]
    assert items[0]["mtime"] == pytest.approx((NOW - dt.timedelta(minutes=30)).timestamp())
    assert notes.final_deadline(clone, "dead", "22:00Z") == dt.datetime(2026, 9, 23, 22, 0, tzinfo=dt.timezone.utc)


def test_an_uncommitted_write_keeps_its_mtime(notes_pair, monkeypatch):
    _, clone = notes_pair
    monkeypatch.setenv(steward.LIVENESS_ENV, "git")
    p = clone / "lanes/dead/20260923T1700Z-report-dead.md"
    p.write_text(p.read_text() + "more\n")
    t = (NOW - dt.timedelta(minutes=2)).timestamp()
    os.utime(p, (t, t))
    notes.refresh_times()
    assert notes.file_time(clone, p) == t
    new = clone / "lanes/dead/untracked-handoff.md"
    new.write_text("x")
    os.utime(new, (t, t))
    assert notes.file_time(clone, new) == t


@pytest.fixture
def code(tmp_path) -> tuple[Path, Path]:
    """(a checkout, its origin) with lane/dead pushed at one commit and a newer local commit never pushed."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    repo = _init(tmp_path / "repo")
    _git(repo, "remote", "add", "origin", str(origin))
    old = email.utils.formatdate((NOW - dt.timedelta(minutes=90)).timestamp(), usegmt=True)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base", env={"GIT_AUTHOR_DATE": old, "GIT_COMMITTER_DATE": old})
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "checkout", "-q", "-b", "lane/dead")
    mid = email.utils.formatdate((NOW - dt.timedelta(minutes=50)).timestamp(), usegmt=True)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "pushed", env={"GIT_AUTHOR_DATE": mid, "GIT_COMMITTER_DATE": mid})
    _git(repo, "push", "-q", "origin", "lane/dead")
    new = email.utils.formatdate((NOW - dt.timedelta(minutes=1)).timestamp(), usegmt=True)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "local only", env={"GIT_AUTHOR_DATE": new, "GIT_COMMITTER_DATE": new})
    _git(repo, "checkout", "-q", "main")
    return repo, origin


def test_e2_the_branch_signal_is_origins_tip_and_no_worktree_is_read(notes_pair, code, monkeypatch):
    _, clone = notes_pair
    repo, _ = code
    monkeypatch.delenv(steward.LIVENESS_ENV, raising=False)
    by_local = {r["lane"]: r for r in notes.status(clone, 30, 8, now=NOW, repo=repo)}["dead"]
    monkeypatch.setenv(steward.LIVENESS_ENV, "git")
    assert notes.fetch_lanes(repo) is None
    row = {r["lane"]: r for r in notes.status(clone, 30, 8, now=NOW, repo=repo)}["dead"]
    assert by_local["commit_min"] == pytest.approx(1, abs=0.1)          # the unpushed commit counts on the laptop only
    assert row["commit_min"] == pytest.approx(50, abs=0.1) and row["dirty"] is None and row["edit_min"] is None
    assert row["stale"] and row["life_via"] == "commit"


def test_e5_relaunch_from_origin_when_the_machine_is_gone(notes_pair, code, monkeypatch):
    _, clone = notes_pair
    repo, origin = code
    notes.bind(clone, "dead", worktree="/nonexistent/machine/verity-wt/dead")
    monkeypatch.setenv(steward.LIVENESS_ENV, "git")
    s, done, msg = notes.relaunch(clone, "dead", repo=repo, now=NOW)
    tip = _git(Path(origin), "rev-parse", "--short=8", "refs/heads/lane/dead")
    assert s == "dead-2" and f"base: origin/lane/dead @ {tip}" in done
    assert f"`lane/dead` @ {tip} on origin" in msg and "git checkout -B lane/dead origin/lane/dead" in msg
    b = json.loads((clone / "lanes/dead-2/binding.json").read_text())
    assert b["branch"] == "lane/dead" and b["succeeds"] == "dead" and "worktree" not in b
    assert f"CHECKPOINT {tip} (21:00Z) [superseded] by dead-2" in notes.report_for(clone, "dead").read_text()


def test_run_custody_fails_closed_without_the_launchers_records(monkeypatch, tmp_path):
    monkeypatch.setenv(steward.LIVENESS_ENV, "git")
    ok, why = notes.run_custody("pod123", runs_dir=tmp_path, registered=lambda _: True)
    assert not ok and "step f" in why


# ---- the lease ---------------------------------------------------------------------------------------------------------------------------

class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_e3_one_holder_a_second_refuses_and_takes_over_only_when_free(tmp_path):
    clock, be = Clock(), steward.FileLeaseBackend(tmp_path / "lease.json")
    a = steward.Lease(be, owner="host-a", ttl_s=900, now=clock, host="host-a", pid=11)
    b = steward.Lease(be, owner="host-b", ttl_s=900, now=clock, host="host-b", pid=22)
    assert a.ensure()[0]
    held, doc = b.ensure()
    assert not held and doc["owner"] == "host-a" and "host-a (host host-a, pid 11)" in steward.holder(doc, clock())
    clock.t += 600
    assert a.ensure()[0]                                    # renewal pushes the expiry
    clock.t += 600
    assert not b.ensure()[0]
    clock.t += 901                                          # a stopped renewing: expired
    assert b.ensure()[0]
    assert not a.ensure()[0] and not a.held                 # the old holder learns it lost
    b.release()
    c = steward.Lease(be, owner="host-c", now=clock, host="host-c", pid=33)
    assert c.ensure()[0]


def test_a_dead_pid_on_the_same_host_frees_the_lease_a_live_one_does_not(tmp_path):
    clock, be = Clock(), steward.FileLeaseBackend(tmp_path / "lease.json")
    assert steward.Lease(be, now=clock, host="h", pid=11, alive=lambda pid: True).ensure()[0]
    assert not steward.Lease(be, now=clock, host="h", pid=12, alive=lambda pid: True).ensure()[0]      # a second copy on one host
    assert steward.Lease(be, now=clock, host="h", pid=12, alive=lambda pid: pid != 11).ensure()[0]     # after a reboot


class FakeS3:
    """The two calls the lease makes, with S3's conditional semantics: If-None-Match: * and If-Match: <etag> answer 412."""

    def __init__(self, honour: bool = True):
        self.objects: dict[str, bytes] = {}
        self.honour = honour

    def describe(self) -> str:
        return "s3://fake"

    def _etag(self, key: str) -> str:
        return '"' + hashlib.md5(self.objects[key]).hexdigest() + '"'

    class _R:
        def __init__(self, body: bytes, headers: dict):
            self.body, self.headers = body, headers

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return self.body

    def _request(self, method: str, key: str, body: bytes | None = None, headers: dict | None = None):
        h = headers or {}
        if method == "GET":
            if key not in self.objects:
                raise urllib.error.HTTPError(key, 404, "Not Found", {}, io.BytesIO(b""))
            return self._R(self.objects[key], {"ETag": self._etag(key)})
        assert method == "PUT"
        if self.honour and ((h.get("if-none-match") == "*" and key in self.objects) or
                            ("if-match" in h and (key not in self.objects or h["if-match"] != self._etag(key)))):
            raise urllib.error.HTTPError(key, 412, "Precondition Failed", {}, io.BytesIO(b""))
        self.objects[key] = body
        return self._R(b"", {"ETag": self._etag(key)})


def test_the_s3_lease_is_a_compare_and_swap():
    s3, clock = FakeS3(), Clock()
    be = steward.S3LeaseBackend(s3)
    a = steward.Lease(be, owner="a", now=clock, host="a", pid=1)
    b = steward.Lease(be, owner="b", now=clock, host="b", pid=2)
    assert a.ensure()[0] and not b.ensure()[0]
    doc, etag = be.read()
    with pytest.raises(steward.LeaseConflict):
        be.write({**doc, "owner": "x"}, None)                # create over an existing lease
    clock.t += 60
    a.ensure()
    with pytest.raises(steward.LeaseConflict):
        be.write({**doc, "owner": "x"}, etag)                # a stale etag: someone renewed since
    clock.t += 10_000
    assert b.ensure()[0] and json.loads(s3.objects[steward.LEASE_KEY])["owner"] == "b"


def test_a_store_that_ignores_the_condition_still_yields_one_holder():
    s3, clock = FakeS3(honour=False), Clock()
    be = steward.S3LeaseBackend(s3)
    a = steward.Lease(be, owner="a", now=clock, host="a", pid=1)
    b = steward.Lease(be, owner="b", now=clock, host="b", pid=2)
    real_write = be.write

    def a_then_b(doc: dict, etag: str | None) -> None:      # both read "free"; b's unconditional write lands right after a's
        real_write(doc, etag)
        if doc["owner"] == "a":
            real_write(b._doc(None), None)
    be.write = a_then_b
    held_a, doc = a.ensure()
    assert not held_a and doc["owner"] == "b"               # a's read-back finds b's token: a does not act
    be.write = real_write
    assert b.ensure()[0]


# ---- the watch with a lease (E3) and the service definition (E4) -------------------------------------------------------------------------

def test_e3_a_second_watcher_refuses_and_names_the_holder(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(notes, "R2_ENV", tmp_path / "absent.env")
    root = tmp_path / "notes"
    (root / "lanes").mkdir(parents=True)
    lease_file = tmp_path / "lease.json"
    holder = steward.Lease(steward.FileLeaseBackend(lease_file), owner="the-server", host="elsewhere", pid=7)
    assert holder.ensure()[0]
    args = ["watch", "--root", str(root), "--passes", "1", "--every", "0", "--guardian-log", "none", "--lease", str(lease_file)]
    assert notes._main(args) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "the-server (host elsewhere, pid 7)" in err
    assert not (root / notes.STATE_FILE).exists()
    assert notes._main([*args, "--standby"]) == 0
    out = capsys.readouterr().out
    assert "STANDBY: the steward lease is held by the-server" in out and not (root / notes.STATE_FILE).exists()
    holder.release()
    assert notes._main(args) == 0
    assert (root / notes.STATE_FILE).exists()               # the holder's pass ran the steward rules
    doc = json.loads(lease_file.read_text())
    assert doc["released"] and doc["pid"] == os.getpid()    # and released the lease on exit


def test_e4_service_definitions_restart_and_start_at_boot(tmp_path):
    u = steward.unit("systemd", "/opt/venv/bin/python", Path("/srv/verity"), env_file=Path("/etc/research/r2.env"),
                     extra_env={"RESEARCH_REPO": "/srv/verity"})
    assert "Restart=always" in u and "WantedBy=default.target" in u and "EnvironmentFile=-/etc/research/r2.env" in u
    exec_ = next(x for x in u.splitlines() if x.startswith("ExecStart="))
    assert exec_.startswith("ExecStart=/opt/venv/bin/python -m research notes watch ")
    for flag in ("--lease", "--standby", "--liveness git", "--reap", "--sync", "--guardian-log none", "'--exclude' 'vllm*'"):
        assert flag.replace("'", "") in exec_.replace("'", "")
    p = steward.unit("launchd", "/usr/bin/python3", Path("/Users/d/verity"), log=tmp_path / "steward.log")
    doc = plistlib.loads(p.encode())
    assert doc["Label"] == steward.LAUNCHD_LABEL and doc["RunAtLoad"] and doc["KeepAlive"]
    assert doc["ProgramArguments"][:5] == ["/usr/bin/python3", "-m", "research", "notes", "watch"] and "--lease" in doc["ProgramArguments"]
    assert "--guardian-log" not in doc["ProgramArguments"]  # the guardian routing stays on the laptop, where launchd runs it
    custom = steward.unit("systemd", "py", Path("/w"), watch_args=["--every", "5", "--lease"])
    assert "watch --every 5 --lease" in custom


def test_service_command_prints_the_unit(capsys):
    assert notes.main(["service", "systemd", "--python", "/p/python", "--workdir", "/", "--", "--every", "3", "--lease"]) == 0
    assert "ExecStart=/p/python -m research notes watch --every 3 --lease" in capsys.readouterr().out
