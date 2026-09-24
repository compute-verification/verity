"""`ship_source` verifies before it marks, and the machine executes only from READY trees (r20 incident: a stalled `git archive | ssh
tar -x` left a half-tree that later runs used).  The transport is exercised for real over a local `sh -c` (the same argv `ssh_run`
would hand to ssh), so the remote script, the staged archive, the digest and the rename all run; truncation is injected by
shortening the bytes on stdin."""
from __future__ import annotations

import gzip
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
    sha, dest, ready = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert sha == _git(repo, "rev-parse", "HEAD") and dest == f"{root}/src/{sha}"
    doc = json.loads((Path(dest) / remote.READY_FILE).read_text())
    assert doc == ready and doc["schema"] == remote.READY_SCHEMA and doc["identity"] == sha and doc["commit"] == sha
    assert doc["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert doc["archive_sha256"] == hashlib.sha256(_archive(repo)).hexdigest() and doc["files"] == 3 and doc["verified"] == "file-sha256+mode-manifest"
    assert doc["shipped_at_utc"] and doc["launcher"]
    assert _tree_files(Path(dest)) == remote.archive_manifest(_archive(repo))["files"], "the tree on the machine is the archive"
    assert doc["reused_files"] == 0 and doc["shipped_files"] == 3, "a first ship sends every file"
    assert not (Path(dest) / ".fill.tar.gz").exists() and not (Path(dest) / ".want.json.gz").exists() and not (Path(dest) / remote.LEGACY_COMPLETE).exists()
    assert [p.name for p in (root / "src").iterdir()] == [sha], "no partial directory left behind"
    # a READY tree is not shipped again: one probe, no transfer
    calls = []
    real = remote.ssh_run
    monkeypatch.setattr(remote, "ssh_run", lambda ssh, argv, **kw: calls.append(argv) or real(ssh, argv, **kw))
    assert remote.ship_source(LOCAL, str(root), repo, python=PY) == (sha, dest, doc)
    assert len(calls) == 1 and not any("tar" in a for a in calls[0])
    assert remote.require_ready_source(dest, sha) == doc


def test_truncated_transfer_is_refused_and_leaves_nothing_ready(repo, tmp_path, monkeypatch):
    root = tmp_path / "root"
    real = remote.ssh_run

    def truncating(ssh, argv, *, stdin=None, timeout=120):
        if stdin is not None and any(".fill.tar.gz" in a for a in argv):
            stdin = stdin[: len(stdin) * 2 // 3]                       # the stream stalls two thirds of the way through
        return real(ssh, argv, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(remote, "ssh_run", truncating)
    with pytest.raises(remote.SourceNotReady, match="archive sha256 mismatch"):
        remote.ship_source(LOCAL, str(root), repo, python=PY)
    sha = _git(repo, "rev-parse", "HEAD")
    assert not (root / "src" / sha).exists(), "a failed ship never creates <root>/src/<identity>/"
    assert not list((root / "src").glob("*.partial-*")), "the partial directory is removed"
    # and the launcher's own timeout is the same fault, not a traceback
    def timing_out(ssh, argv, *, stdin=None, timeout=120):
        if stdin is not None and any(".fill.tar.gz" in a for a in argv):
            raise subprocess.TimeoutExpired(cmd="ssh", timeout=timeout)
        return real(ssh, argv, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(remote, "ssh_run", timing_out)
    with pytest.raises(remote.SourceNotReady, match="exceeded"):
        remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert not (root / "src" / sha).exists() and not list((root / "src").glob("*.partial-*"))
    assert issubclass(remote.SourceNotReady, remote.Refused), "the launcher's `refused:` path names it"


def test_extraction_manifest_mismatch_is_refused(repo, tmp_path, monkeypatch):
    """The digest can match and the tree still be wrong (a fill that lacks a file, tar variant, disk full mid-extract): every file of the
    assembled tree is checked against the manifest."""
    root = tmp_path / "root"
    real = remote.fill_archive
    monkeypatch.setattr(remote, "fill_archive", lambda tar, paths: real(tar, set(sorted(paths)[:-1])))   # one file never sent
    with pytest.raises(remote.SourceNotReady, match="manifest mismatch.*1 missing.*pkg/c.bin"):
        remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert not list((root / "src").glob("*")), "nothing READY, nothing partial"


def _commit(repo: Path, msg: str = "next", *extra: str) -> str:
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", msg, *extra)
    return _git(repo, "rev-parse", "HEAD")


def test_a_repeat_ship_sends_only_the_files_the_machine_lacks_and_still_verifies_every_file(repo, tmp_path, monkeypatch, capsys):
    """r21: `run --on <pod> --source .` sent the whole tree (168 MB at 0.3 MB/s) and gave up after 600 s; the workaround was a manual
    `git archive | tar -x` into <root>/src/<sha>/.  Now the machine copies whatever its trees (that one included) already hold by content
    and only the rest crosses the link; READY still means every file's sha256 and executable bit was checked on the machine."""
    root = tmp_path / "root"
    sha1 = _git(repo, "rev-parse", "HEAD")
    manual = root / "src" / sha1                                        # the r21 workaround: an unmarked tree of the previous commit
    manual.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(_archive(repo)), mode="r:") as tf:
        tf.extractall(manual, filter="data")
    real = remote.ssh_run
    sent: list[list[str]] = []

    def spy(ssh, argv, *, stdin=None, timeout=120):
        if stdin is not None and any(".fill.tar.gz" in a for a in argv):
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(stdin))) as tf:
                sent.append(sorted(tf.getnames()))
        return real(ssh, argv, stdin=stdin, timeout=timeout)

    monkeypatch.setattr(remote, "ssh_run", spy)
    (repo / "a.txt").write_text("alpha, edited\n")
    (repo / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (repo / "run.sh").chmod(0o755)
    sha2 = _commit(repo)
    got, dest2, doc2 = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert got == sha2 and sent == [["a.txt", "run.sh"]], "the unchanged files are copied on the machine, not sent"
    assert doc2["reused_files"] == 2 and doc2["shipped_files"] == 2 and doc2["files"] == 4 and doc2["verified"] == "file-sha256+mode-manifest"
    assert _tree_files(Path(dest2)) == remote.archive_manifest(_archive(repo))["files"] and remote.require_ready_source(dest2, sha2) == doc2
    assert os.access(Path(dest2) / "run.sh", os.X_OK) and not os.access(Path(dest2) / "pkg" / "b.py", os.X_OK)
    assert "2 of 4 files already on the machine; shipping 2" in capsys.readouterr().err
    index = root / "cache" / "src-index"
    assert {p.name for p in index.iterdir()} == {f"{sha1}.sha256", f"{sha2}.sha256"}, "the manual tree was indexed on demand"
    assert not (manual / remote.READY_FILE).exists(), "a reuse source is read, never marked or modified"

    # the same content under a new commit: nothing crosses the link, the tree is still verified before READY
    sha3 = _commit(repo, "empty", "--allow-empty")
    _, dest3, doc3 = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert len(sent) == 1 and doc3["identity"] == sha3 and doc3["reused_files"] == 4 and doc3["shipped_files"] == 0 and doc3["shipped_bytes"] == 0
    assert _tree_files(Path(dest3)) == _tree_files(Path(dest2))

    # a source edited in place after it was indexed (a workload writing into its tree): hashed as it is copied, not used, sent instead
    for tree in (manual, Path(dest2), Path(dest3)):
        (tree / "pkg" / "c.bin").write_bytes(b"edited in place")
    (repo / "a.txt").write_text("alpha, again\n")
    sha4 = _commit(repo)
    _, dest4, doc4 = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert sent[-1] == ["a.txt", "pkg/c.bin"] and doc4["reused_files"] == 2 and doc4["identity"] == sha4
    assert _tree_files(Path(dest4)) == remote.archive_manifest(_archive(repo))["files"]
    left = {p.name.removesuffix(".sha256") for p in index.glob("*.sha256")}
    assert sha4 in left and len(left & {sha1, sha2, sha3}) == 2, "the stale source's index is dropped (rebuilt when next needed)"


def test_the_machine_side_verify_names_missing_extra_differing_and_mode_mismatched_files(tmp_path, capsys):
    from research import source_ship as S

    root = tmp_path / "root"
    partial = root / "src" / "x.partial-1"
    partial.mkdir(parents=True)
    h = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    want = {"a": [h(b"A"), False], "b": [h(b"B"), True], "c": [h(b"C"), False], "d/e": [h(b"E"), True]}

    def verify() -> dict:
        (partial / S.WANT).write_bytes(gzip.compress(json.dumps(want).encode()))
        S.verify(str(root), str(partial), "x")
        return json.loads(capsys.readouterr().out)

    (partial / "a").write_bytes(b"A")
    (partial / "b").write_bytes(b"B")                                   # should be executable
    (partial / "c").write_bytes(b"not C")
    (partial / "zz").write_bytes(b"not in the archive")                 # and d/e is missing
    out = verify()
    assert not out["ok"] and out["files"] == 4
    assert {k: out[k]["e.g."] for k in ("missing", "extra", "differs", "mode")} == {"missing": ["d/e"], "extra": ["zz"], "differs": ["c"], "mode": ["b"]}
    assert not (root / "cache" / "src-index" / "x.sha256").exists() and not (partial / S.WANT).exists()
    (partial / "b").chmod(0o755)
    (partial / "c").write_bytes(b"C")
    (partial / "zz").unlink()
    (partial / "d").mkdir()
    (partial / "d" / "e").write_bytes(b"E")
    (partial / "d" / "e").chmod(0o755)
    out = verify()
    assert out["ok"] and out["files"] == 4
    assert (root / "cache" / "src-index" / "x.sha256").read_text().splitlines() == [f"{h(b'A')}  a", f"{h(b'B')}  b", f"{h(b'C')}  c", f"{h(b'E')}  d/e"]


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
    _, dest, doc = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert dest == str(legacy) and doc["verified"] == "legacy-sha256-per-file" and doc["identity"] == sha
    assert json.loads((legacy / remote.READY_FILE).read_text()) == doc and remote.require_ready_source(legacy, sha) == doc

    # the r20 shape: a `.complete` tree with a file missing and one truncated -> moved aside, shipped afresh, READY
    (legacy / remote.READY_FILE).unlink()
    (legacy / "pkg" / "b.py").unlink()
    (legacy / "pkg" / "c.bin").write_bytes(b"trunc")
    _, dest, doc = remote.ship_source(LOCAL, str(root), repo, python=PY)
    assert doc["verified"] == "file-sha256+mode-manifest" and _tree_files(Path(dest)) == remote.archive_manifest(_archive(repo))["files"]
    q = list((root / "src" / ".quarantine").iterdir())
    assert len(q) == 1 and q[0].name.startswith(sha + "-") and (q[0] / remote.LEGACY_COMPLETE).exists() and not (q[0] / "pkg" / "b.py").exists()
    err = capsys.readouterr().err
    assert "did not match the archive" in err and "missing pkg/b.py" in err and "quarantine" in err

    # a READY.json naming another identity is never shipped over
    (legacy / remote.READY_FILE).write_text(json.dumps({**doc, "identity": "f" * 40}))
    with pytest.raises(remote.SourceNotReady, match="names identity"):
        remote.ship_source(LOCAL, str(root), repo, python=PY)


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
