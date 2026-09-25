"""`ship_source` verifies before it marks, and the machine executes only from READY trees (r20 incident: a stalled `git archive | ssh
tar -x` left a half-tree that later runs used).  The transport is exercised for real over a local `sh -c` (the same argv `ssh_run`
would hand to ssh), so the remote script, the staged archive, the digest and the rename all run; truncation is injected by
shortening the bytes on stdin."""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from research import remote
from research import runs as R
from research.store import attempt as A

PY = sys.executable
SRC = str(Path(remote.__file__).resolve().parents[1])
LOCAL = ["sh", "-c"]                     # `ssh_run(LOCAL, argv)` runs the remote argv on this machine


@pytest.fixture(autouse=True)
def _stream_transport(monkeypatch):
    """These tests are the archive stream's (truncation, digests, legacy trees); the git transport has test_remote_ship_git.py."""
    monkeypatch.setenv(remote.SHIP_ENV, "stream")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    d = tmp_path / "repo"
    d.mkdir()
    (d / "a.txt").write_text("alpha\n")
    (d / "pkg").mkdir()
    (d / "pkg" / "b.py").write_text("B = 2\n")
    (d / "pkg" / "c.bin").write_bytes(bytes(range(256)) * 40)
    _git(d, "init", "-q")
    _git(d, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(d, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "one")
    return d


def _archive(repo: Path) -> bytes:
    return subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=repo, capture_output=True, check=True).stdout


def _tree_files(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file() and p.name != remote.READY_FILE}


def test_complete_ship_lands_ready_with_identity_and_is_idempotent(repo, tmp_path, monkeypatch):
    root = tmp_path / "root"
    sha, dest, ready = remote.ship_source(LOCAL, str(root), repo)
    assert sha == _git(repo, "rev-parse", "HEAD") and dest == f"{root}/src/{sha}"
    doc = json.loads((Path(dest) / remote.READY_FILE).read_text())
    assert doc == ready and doc["schema"] == remote.READY_SCHEMA and doc["identity"] == sha and doc["commit"] == sha
    assert doc["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert doc["archive_sha256"] == hashlib.sha256(_archive(repo)).hexdigest() and doc["files"] == 3 and doc["verified"] == "archive-sha256+file-manifest"
    assert doc["shipped_at_utc"] and doc["launcher"]
    assert _tree_files(Path(dest)) == remote.archive_manifest(_archive(repo))["files"], "the tree on the machine is the archive"
    assert not (Path(dest) / ".archive.tar").exists() and not (Path(dest) / remote.LEGACY_COMPLETE).exists()
    assert [p.name for p in (root / "src").iterdir()] == [sha], "no partial directory left behind"
    # a READY tree is not shipped again: one probe, no transfer
    calls = []
    real = remote.ssh_run
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, **kw: calls.append(argv) or real(ssh, argv, **kw))
    assert remote.ship_source(LOCAL, str(root), repo) == (sha, dest, doc)
    assert len(calls) == 1 and not any("tar" in a for a in calls[0])
    assert remote.require_ready_source(dest, sha) == doc


def test_an_older_commit_shipped_after_a_newer_build_is_not_mistaken_for_built(repo, tmp_path):
    """A shared build dir (Cargo's `/workspace/sp1-target`) calls an output fresh when it is newer than every source.  `git archive`
    stamps files with the commit time, so extracting with those times made an older commit's tree look already built and it ran
    the newer commit's binary.  The shipped tree must carry extraction times."""
    root = tmp_path / "root"
    remote.ship_source(LOCAL, str(root), repo)
    output = tmp_path / "shared-target" / "binary"                  # the last build, just finished, from the first commit
    output.parent.mkdir()
    output.write_text("built from commit one\n")
    built_at = time.time() - 5
    os.utime(output, (built_at, built_at))
    (repo / "pkg" / "b.py").write_text("B = 3\n")                 # a commit whose date is before that build (a baseline re-run)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "older"], cwd=repo, check=True,
                   env={**os.environ, "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z"})   # `git archive` stamps the committer date
    sha, dest, _ = remote.ship_source(LOCAL, str(root), repo)
    sources = [p for p in Path(dest).rglob("*") if p.is_file() and p.name != remote.READY_FILE]
    newest = max(p.stat().st_mtime for p in sources)
    assert newest > output.stat().st_mtime, "the build output looks newer than every shipped source: the stale binary would run"
    assert min(p.stat().st_mtime for p in sources) >= built_at


class _Cut:
    """A pipe that stops accepting after `limit` bytes, as a stalled or severed stream does (the feed sees a closed pipe)."""

    def __init__(self, w, limit):
        self.w, self.left = w, limit

    def write(self, b):
        if len(b) > self.left:
            self.w.write(b[: self.left])
            self.left = 0
            raise BrokenPipeError("cut")
        self.left -= len(b)
        return self.w.write(b)

    def flush(self):
        self.w.flush()


def _cutting(real, limit):
    def stream(ssh, argv, feed, *, timeout):
        return real(ssh, argv, lambda w: feed(_Cut(w, limit)), timeout=timeout)
    return stream


def test_truncated_transfer_is_refused_and_leaves_nothing_ready(repo, tmp_path, monkeypatch):
    root = tmp_path / "root"
    size = len(_archive(repo))
    monkeypatch.setattr(remote, "ssh_stream", _cutting(remote.ssh_stream, size * 2 // 3))   # the stream stalls two thirds of the way through
    with pytest.raises(remote.SourceNotReady, match=r"archive sha256 mismatch.*truncated transfer"):
        remote.ship_source(LOCAL, str(root), repo)
    sha = _git(repo, "rev-parse", "HEAD")
    assert not (root / "src" / sha).exists(), "a failed ship never creates <root>/src/<identity>/"
    assert not list((root / "src").glob("*.partial-*")), "the partial directory is removed"
    # and the launcher's own timeout is the same fault, not a traceback
    def timing_out(ssh, argv, feed, *, timeout):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)

    monkeypatch.setattr(remote, "ssh_stream", timing_out)
    with pytest.raises(remote.SourceNotReady, match="exceeded 600s"):
        remote.ship_source(LOCAL, str(root), repo)
    assert not (root / "src" / sha).exists() and not list((root / "src").glob("*.partial-*"))
    assert issubclass(remote.SourceNotReady, remote.Refused), "the launcher's `refused:` path names it"
    assert remote.ship_timeout(229_017_600) > 3000 and remote.ship_timeout(1000) == remote.SHIP_SOURCE_TIMEOUT_S, "the timeout grows with the archive"


def test_machine_removes_its_own_partial_when_the_launcher_cannot(repo, tmp_path, monkeypatch):
    """A launcher that dies mid-ship cannot run its cleanup: the machine sees the short archive and removes the partial itself."""
    root = tmp_path / "root"
    monkeypatch.setattr(remote, "_rm_rf", lambda ssh, path: None)                           # the launcher's cleanup never runs
    monkeypatch.setattr(remote, "ssh_stream", _cutting(remote.ssh_stream, 4096))
    with pytest.raises(remote.SourceNotReady, match="archive sha256 mismatch"):
        remote.ship_source(LOCAL, str(root), repo)
    assert list((root / "src").iterdir()) == [], "nothing at all is left on the machine, not even a partial"


def test_ship_streams_the_archive_instead_of_holding_it(tmp_path):
    """The launcher's memory does not grow with the tree: a 24 MiB archive ships with a few MiB of Python allocations (the old
    launcher held it two to three times over; 0.55-0.69 GB launchers of the verity tree were killed by the laptop's guardian)."""
    import os
    import tracemalloc

    big = tmp_path / "big"
    big.mkdir()
    (big / "blob.bin").write_bytes(os.urandom(24 << 20))
    (big / "small.txt").write_text("s\n")
    _git(big, "init", "-q")
    _git(big, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(big, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "big")
    root = tmp_path / "root"
    tracemalloc.start()
    try:
        sha, dest, doc = remote.ship_source(LOCAL, str(root), big)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert doc["files"] == 2 and doc["bytes"] > 24 << 20 and remote.require_ready_source(dest, sha) == doc
    assert hashlib.sha256((Path(dest) / "blob.bin").read_bytes()).hexdigest() == remote.archive_manifest_git(big)["files"]["blob.bin"]
    assert peak < 6 << 20, f"peak Python allocation during the ship: {peak / 2**20:.1f} MiB"


def test_manifest_streamed_from_git_equals_the_in_memory_one(repo):
    assert remote.archive_manifest_git(repo) == remote.archive_manifest(_archive(repo))
    assert remote.archive_manifest(_archive(repo))["size"] == len(_archive(repo))


def test_launcher_killed_mid_ship_leaves_nothing_executable(repo, tmp_path):
    """SIGKILL the launcher while the archive is in flight (a throttled feed): no <root>/src/<sha>/ appears, the machine-side gate
    refuses the identity, and the orphaned transport removes its partial once it sees the short stream."""
    import os
    import signal

    root = tmp_path / "root"
    sha = _git(repo, "rev-parse", "HEAD")
    child = f"""
import pathlib, sys, time
sys.path.insert(0, {SRC!r})
from research import remote
real_feed = remote._archive_feed
class Slow:
    def __init__(self, w): self.w = w
    def write(self, b):
        for i in range(0, len(b), 1024):
            self.w.write(b[i:i + 1024]); self.w.flush(); print("SENT", flush=True); time.sleep(0.05)
    def flush(self): self.w.flush()
remote._archive_feed = lambda git_dir, sent: (lambda w, inner=real_feed(git_dir, sent): inner(Slow(w)))
remote.ship_source(["sh", "-c"], {str(root)!r}, pathlib.Path({str(repo)!r}))
"""
    p = subprocess.Popen([PY, "-c", child], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdout.readline().strip() == b"SENT", p.stderr.read()
    p.stdout.readline()
    assert list((root / "src").glob("*.partial-*")), "the ship is in flight"
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    deadline = time.time() + 10
    while time.time() < deadline and list((root / "src").glob("*.partial-*")):
        time.sleep(0.1)
    assert not (root / "src" / sha).exists(), "nothing executable by name"
    assert not list((root / "src").glob("*.partial-*")), "the orphaned transport removed the partial when the stream ended short"
    req, _ = _request(tmp_path, "rk", [PY, "-c", "print('ran')"], remote_sec={"source_sha": sha, "source_dir": str(root / "src" / sha)})
    out = remote.launch_request(req)
    assert out["started"] is False and out["fault"] == "SourceNotReady"


def test_a_dead_ships_stale_partial_is_removed_by_the_next_ship(repo, tmp_path):
    import os

    root = tmp_path / "root"
    stale = root / "src" / "f00d.partial-deadbeef"
    (stale / "pkg").mkdir(parents=True)
    (stale / "pkg" / "half.py").write_text("x")
    old = time.time() - 3600
    for p in (stale / "pkg" / "half.py", stale / "pkg", stale):
        os.utime(p, (old, old))
    fresh = root / "src" / "cafe.partial-0000beef"                  # another launcher's ship in progress: kept
    fresh.mkdir()
    (fresh / ".archive.tar").write_bytes(b"in flight")
    sha, dest, _ = remote.ship_source(LOCAL, str(root), repo)
    assert sorted(p.name for p in (root / "src").iterdir()) == sorted([sha, fresh.name])


def test_extraction_manifest_mismatch_is_refused(repo, tmp_path, monkeypatch):
    """The digest can match and the extraction still be wrong (tar variant, disk full mid-extract): the file set is checked too."""
    root = tmp_path / "root"
    real = remote.ssh_stream

    def dropping(ssh, argv, feed, *, timeout):
        r = real(ssh, argv, feed, timeout=timeout)
        lines = r.stdout.decode().splitlines()
        r.stdout = "\n".join(lines[:-1]).encode()                      # the machine lists one file fewer than it should
        return r

    monkeypatch.setattr(remote, "ssh_stream", dropping)
    with pytest.raises(remote.SourceNotReady, match="manifest mismatch.*1 missing"):
        remote.ship_source(LOCAL, str(root), repo)
    assert not list((root / "src").glob("*")), "nothing READY, nothing partial"


def test_legacy_complete_tree_is_adopted_when_identical_and_quarantined_when_not(repo, tmp_path, capsys):
    root = tmp_path / "root"
    sha = _git(repo, "rev-parse", "HEAD")
    legacy = root / "src" / sha
    legacy.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(_archive(repo)), mode="r:") as tf:
        tf.extractall(legacy, filter="data")
    (legacy / remote.LEGACY_COMPLETE).touch()
    with pytest.raises(remote.SourceNotReady, match="legacy"):
        remote.require_ready_source(legacy, sha)                        # `.complete` alone never executes
    _, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    assert dest == str(legacy) and doc["verified"] == "legacy-sha256-per-file" and doc["identity"] == sha
    assert json.loads((legacy / remote.READY_FILE).read_text()) == doc and remote.require_ready_source(legacy, sha) == doc

    # the r20 shape: a `.complete` tree with a file missing and one truncated -> moved aside, shipped afresh, READY
    (legacy / remote.READY_FILE).unlink()
    (legacy / "pkg" / "b.py").unlink()
    (legacy / "pkg" / "c.bin").write_bytes(b"trunc")
    _, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    assert doc["verified"] == "archive-sha256+file-manifest" and _tree_files(Path(dest)) == remote.archive_manifest(_archive(repo))["files"]
    q = list((root / "src" / ".quarantine").iterdir())
    assert len(q) == 1 and q[0].name.startswith(sha + "-") and (q[0] / remote.LEGACY_COMPLETE).exists() and not (q[0] / "pkg" / "b.py").exists()
    err = capsys.readouterr().err
    assert "did not match the archive" in err and "missing pkg/b.py" in err and "quarantine" in err

    # a READY.json naming another identity is never shipped over
    (legacy / remote.READY_FILE).write_text(json.dumps({**doc, "identity": "f" * 40}))
    with pytest.raises(remote.SourceNotReady, match="names identity"):
        remote.ship_source(LOCAL, str(root), repo)


def _request(tmp_path, run_id, command, *, remote_sec):
    root = tmp_path / "root"
    (root / "requests" / run_id).mkdir(parents=True, exist_ok=True)
    req = {"schema": remote.REQUEST_SCHEMA, "run_id": run_id, "runs_root": str(root / "runs"), "tool": SRC, "python": PY, "cwd": None,
           "env": {}, "harness_args": ["--stage", "t", "--no-sampler"], "command": command, "exclusive": False,
           "cost": {"provider": "runpod", "pod_id": "p1", "rate_usd_per_hour": 0.06, "rate_observed_utc": "t0", "currency": "USD", "reason": None},
           "remote": {"machine": "m", "pod_id": "p1", "root": str(root), **remote_sec}}
    p = root / "requests" / run_id / "request.json"
    p.write_text(json.dumps(req))
    return p, root


def _wait_terminal(run_dir, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = R.summarize(run_dir)
        if s["state"] in R.TERMINAL:
            return s
        time.sleep(0.1)
    pytest.fail(f"run did not finish: {R.summarize(run_dir)}")


def test_machine_refuses_to_execute_without_ready_or_with_foreign_identity(tmp_path):
    sha = "a" * 40
    src = tmp_path / "root" / "src" / sha
    src.mkdir(parents=True)
    (src / remote.LEGACY_COMPLETE).touch()                              # the old marker on an unverified tree
    req, root = _request(tmp_path, "r1", [PY, "-c", "print('ran')"], remote_sec={"source_sha": sha, "source_dir": str(src)})
    out = remote.launch_request(req)
    assert out["claimed"] and out["started"] is False and out["fault"] == "SourceNotReady" and "no READY.json" in out["refused"] and "legacy" in out["refused"]
    refused = json.loads((root / "runs" / "r1" / "refused.json").read_text())
    assert refused["fault"] == "SourceNotReady" and not (root / "runs" / "r1" / "launch.json").exists() and not (root / "runs" / "r1" / "stdout.log").exists()

    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": "b" * 40, "archive_sha256": "x" * 64}))
    req2, _ = _request(tmp_path, "r2", [PY, "-c", "print('ran')"], remote_sec={"source_sha": sha, "source_dir": str(src)})
    out = remote.launch_request(req2)
    assert out["started"] is False and out["fault"] == "SourceNotReady" and "names identity bbbb" in out["refused"] and "asked for aaaa" in out["refused"]

    # a request from an older launcher (no source_dir) is gated on <root>/src/<sha> just the same
    req3, _ = _request(tmp_path, "r3", [PY, "-c", "print('ran')"], remote_sec={"source_sha": sha})
    assert remote.launch_request(req3)["fault"] == "SourceNotReady"

    # READY for the requested identity -> it runs, and the attempt says which verified tree ran it
    ready = {"schema": remote.READY_SCHEMA, "identity": sha, "commit": sha, "tree": None, "archive_sha256": "c" * 64, "files": 1, "bytes": 1,
             "shipped_at_utc": "t", "launcher": "l", "verified": "archive-sha256+file-manifest"}
    (src / remote.READY_FILE).write_text(json.dumps(ready))
    req4, _ = _request(tmp_path, "r4", [PY, "-c", "import os; print(os.environ['RESEARCH_SOURCE_SHA'])"],
                       remote_sec={"source_sha": sha, "source_dir": str(src), "source_archive_sha256": "c" * 64, "source_verified": ready["verified"]})
    out = remote.launch_request(req4)
    assert out["started"], out
    s = _wait_terminal(root / "runs" / "r4")
    assert s["state"] == "done" and (root / "runs" / "r4" / "stdout.log").read_text().strip() == sha
    doc = A.build(root / "runs" / "r4")
    assert doc["execution"]["remote"]["source_sha"] == sha and doc["execution"]["remote"]["source_archive_sha256"] == "c" * 64
    assert doc["execution"]["remote"]["source_dir"] == str(src) and doc["execution"]["remote"]["source_verified"] == ready["verified"]
    # a run that shipped no tree carries no source provenance and is not gated
    req5, _ = _request(tmp_path, "r5", [PY, "-c", "pass"], remote_sec={})
    assert remote.launch_request(req5)["started"]
    _wait_terminal(root / "runs" / "r5")
    assert A.build(root / "runs" / "r5")["execution"]["remote"]["source_archive_sha256"] is None
