"""`ship_source` over git (pod code delivery): the commit is `git push`ed into the machine's bare repo through the launcher's own
ssh argv, extracted there and verified per file against the launcher's archive manifest; a machine without git, or any failure of
that path, falls back to the archive stream with one line; `RESEARCH_SHIP=stream` opts out.  The machine is a local `sh -c`, as in
test_remote_ship_source.py, so `git receive-pack` really runs through `git_push_url`."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from research import remote

LOCAL = ["sh", "-c"]


def _git(repo: Path, *args: str, **env: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, capture_output=True, text=True, check=True,
                          env={**os.environ, **env}).stdout.strip()


@pytest.fixture(autouse=True)
def _git_transport(monkeypatch):
    monkeypatch.delenv(remote.SHIP_ENV, raising=False)


@pytest.fixture
def repo(tmp_path) -> Path:
    d = tmp_path / "repo"
    (d / "pkg").mkdir(parents=True)
    (d / "a.txt").write_text("alpha\n")
    (d / "pkg" / "b.py").write_text("B = 2\n")
    (d / "pkg" / "c.bin").write_bytes(bytes(range(256)) * 40)
    (d / "with space.txt").write_text("spaced\n")
    _git(d, "init", "-q")
    _git(d, "add", ".")
    _git(d, "commit", "-q", "-m", "one", GIT_COMMITTER_DATE="2020-01-01T00:00:00Z")
    return d


def _tree(d: Path) -> dict[str, str]:
    import hashlib
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*"))
            if p.is_file() and p.name != remote.READY_FILE}


def _no_git_machine(tmp_path: Path) -> list[str]:
    """A machine whose PATH has every tool but git."""
    fake = tmp_path / "nogit-bin"
    fake.mkdir()
    for d in ("/usr/bin", "/bin"):
        for p in Path(d).iterdir():
            if not p.name.startswith("git") and not (fake / p.name).exists():
                (fake / p.name).symlink_to(p)
    return ["env", f"PATH={fake}", "sh", "-c"]


def test_first_and_delta_ships_go_as_git_objects_and_verify_per_file(repo, tmp_path, capsys):
    root = tmp_path / "root"
    t0 = time.time() - 1
    sha, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    assert doc["verified"] == "git-push+tree+file-sha256" and doc["transport"] == "git" and doc["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    man = remote.archive_manifest_git(repo)
    assert doc["archive_sha256"] == man["archive_sha256"] and doc["files"] == 4 and doc["pod_archive_sha256"]
    assert _tree(Path(dest)) == man["files"], "the tree on the machine is the commit, file by file"
    assert min(p.stat().st_mtime for p in Path(dest).rglob("*") if p.is_file()) >= t0, "extraction mtimes, not the 2020 commit time"
    bare = root / remote.POD_GIT_REPO
    assert _git(bare, "rev-parse", f"refs/research/src/{sha}") == sha
    assert [p.name for p in (root / "src").iterdir()] == [sha] and "archive stream" not in capsys.readouterr().err
    assert remote.require_ready_source(dest, sha) == doc
    (repo / "pkg" / "b.py").write_text("B = 3\n")
    _git(repo, "commit", "-qam", "two")
    sha2, dest2, doc2 = remote.ship_source(LOCAL, str(root), repo)
    assert doc2["transport"] == "git" and (Path(dest2) / "pkg" / "b.py").read_text() == "B = 3\n" and sha2 != sha
    assert (Path(dest) / "pkg" / "b.py").read_text() == "B = 2\n", "the older tree is untouched"
    assert remote.ship_source(LOCAL, str(root), repo) == (sha2, dest2, doc2), "a READY tree is not shipped again"


def test_a_machine_without_git_gets_the_archive_stream_with_one_line(repo, tmp_path, capsys):
    root = tmp_path / "root"
    sha, dest, doc = remote.ship_source(_no_git_machine(tmp_path), str(root), repo)
    assert doc["verified"] == "archive-sha256+file-manifest" and doc["transport"] == "stream"
    err = capsys.readouterr().err
    assert "git ship to" in err and "(the machine has no git); shipping the archive stream" in err
    assert _tree(Path(dest)) == remote.archive_manifest_git(repo)["files"] and not (root / remote.POD_GIT_REPO).exists()


def test_a_tampered_extraction_is_never_marked_ready_and_the_stream_ships_instead(repo, tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    real = remote._git_ship_script
    monkeypatch.setattr(remote, "_git_ship_script", lambda root_, repo_, partial, sha: real(root_, repo_, partial, sha).replace(
        "trap - EXIT", f"echo tampered >> {partial}/a.txt; trap - EXIT"))
    sha, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    err = capsys.readouterr().err
    assert "extracted tree does not match the archive manifest (1: differs a.txt)" in err and doc["transport"] == "stream"
    assert (Path(dest) / "a.txt").read_text() == "alpha\n" and not list((root / "src").glob("*.partial-*"))


def test_stream_opt_out_and_the_push_url_carries_no_credential(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(remote.SHIP_ENV, "stream")
    _, _, doc = remote.ship_source(LOCAL, str(tmp_path / "root"), repo)
    assert doc["transport"] == "stream" and not (tmp_path / "root" / remote.POD_GIT_REPO).exists()
    url = remote.git_push_url(["ssh", "-i", "/k/my key", "-p", "2222", "root@1.2.3.4"], "/workspace/research/git/verity.git")
    assert url == "ext::ssh -i /k/my% key -p 2222 root@1.2.3.4 git% receive-pack% /workspace/research/git/verity.git"


def test_nothing_credential_like_is_left_on_the_machine(repo, tmp_path):
    root = tmp_path / "root"
    remote.ship_source(LOCAL, str(root), repo)
    bare = root / remote.POD_GIT_REPO
    cfg = (bare / "config").read_text()
    assert "url" not in cfg and "credential" not in cfg and "extraheader" not in cfg.lower()
    assert json.loads((root / "src" / _git(repo, "rev-parse", "HEAD") / remote.READY_FILE).read_text())["transport"] == "git"
