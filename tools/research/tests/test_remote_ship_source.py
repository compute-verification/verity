"""`ship_source` verifies before it marks, and the machine executes only from READY trees (r20 incident: a stalled `git archive | ssh
tar -x` left a half-tree that later runs used).  The transport is exercised for real over a local `sh -c` (the same argv `ssh_run`
would hand to ssh), so the remote script, the staged archive, the digest and the rename all run; truncation is injected by
shortening the bytes on stdin."""
from __future__ import annotations

import hashlib
import io
import json
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


def test_truncated_transfer_is_refused_and_leaves_nothing_ready(repo, tmp_path, monkeypatch):
    root = tmp_path / "root"
    real = remote.ssh_run

    def truncating(ssh, argv, *, stdin=None, timeout=120):
        if stdin is not None and any(".archive.tar" in a for a in argv):
            stdin = stdin[: len(stdin) * 2 // 3]                       # the stream stalls two thirds of the way through
        return real(ssh, argv, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(remote, "ssh_run", truncating)
    with pytest.raises(remote.SourceNotReady, match="archive sha256 mismatch"):
        remote.ship_source(LOCAL, str(root), repo)
    sha = _git(repo, "rev-parse", "HEAD")
    assert not (root / "src" / sha).exists(), "a failed ship never creates <root>/src/<identity>/"
    assert not list((root / "src").glob("*.partial-*")), "the partial directory is removed"
    # and the launcher's own timeout is the same fault, not a traceback
    def timing_out(ssh, argv, *, stdin=None, timeout=120):
        if stdin is not None and any(".archive.tar" in a for a in argv):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return real(ssh, argv, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(remote, "ssh_run", timing_out)
    with pytest.raises(remote.SourceNotReady, match="exceeded"):
        remote.ship_source(LOCAL, str(root), repo)
    assert not (root / "src" / sha).exists() and not list((root / "src").glob("*.partial-*"))
    assert issubclass(remote.SourceNotReady, remote.Refused), "the launcher's `refused:` path names it"


def test_extraction_manifest_mismatch_is_refused(repo, tmp_path, monkeypatch):
    """The digest can match and the extraction still be wrong (tar variant, disk full mid-extract): the file set is checked too."""
    root = tmp_path / "root"
    real = remote.ssh_run

    def dropping(ssh, argv, *, stdin=None, timeout=120):
        r = real(ssh, argv, stdin=stdin, timeout=timeout)
        if stdin is not None and any(".archive.tar" in a for a in argv):
            lines = r.stdout.decode().splitlines()
            r.stdout = "\n".join(lines[:-1]).encode()                  # the machine lists one file fewer than it should
        return r

    monkeypatch.setattr(remote, "ssh_run", dropping)
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
