"""`ship_source` over git (pod code delivery): the commit is `git push`ed into the machine's bare repo through the launcher's own
ssh argv, extracted there and verified per file against the launcher's archive manifest; a machine without git, or any failure of
that path, falls back to the archive stream with one line; `RESEARCH_SHIP=stream` opts out.  The machine is a local `sh -c`, as in
test_remote_ship_source.py, so `git receive-pack` really runs through `git_push_url`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research import remote

LOCAL = ["sh", "-c"]
REPLICA_TEST = Path(__file__).resolve().parent / "test_repo_replicas.py"


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
            if p.is_file() and p.name != remote.READY_FILE and p.relative_to(d).parts[0] != ".git"}


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
    sha, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    assert doc["verified"] == "git-push+tree+file-sha256" and doc["transport"] == "git" and doc["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    man = remote.archive_manifest_git(repo)
    assert doc["archive_sha256"] == man["archive_sha256"] and doc["files"] == 4 and doc["pod_archive_sha256"]
    assert _tree(Path(dest)) == man["files"], "the tree on the machine is the commit, file by file"
    assert min(p.stat().st_mtime for p in Path(dest).rglob("*") if p.is_file()) >= 1_609_459_200, "extraction mtimes, not the 2020 commit time"   # 2021-01-01
    bare = root / remote.POD_GIT_REPO
    assert _git(bare, "rev-parse", f"refs/research/src/{sha}") == sha
    assert [p.name for p in (root / "src").iterdir()] == [sha] and "archive stream" not in capsys.readouterr().err
    assert remote.require_ready_source(dest, sha) == doc
    assert doc["checkout"] is True and _git(Path(dest), "rev-parse", "HEAD") == sha and _git(Path(dest), "status", "--porcelain") == ""
    assert _git(Path(dest), "ls-files", "-co", "--exclude-standard") == _git(repo, "ls-files"), "what `check.tracked_files` lists"
    assert (Path(dest) / ".git" / "objects" / "info" / "alternates").read_text().strip() == str(bare / "objects"), "no object copied"
    (repo / "pkg" / "b.py").write_text("B = 3\n")
    _git(repo, "commit", "-qam", "two")
    sha2, dest2, doc2 = remote.ship_source(LOCAL, str(root), repo)
    assert doc2["transport"] == "git" and (Path(dest2) / "pkg" / "b.py").read_text() == "B = 3\n" and sha2 != sha
    assert (Path(dest) / "pkg" / "b.py").read_text() == "B = 2\n", "the older tree is untouched"
    assert remote.ship_source(LOCAL, str(root), repo) == (sha2, dest2, doc2), "a READY tree is not shipped again"


def test_a_shipped_checkouts_origin_main_is_the_newest_launchers_and_never_moves_back(repo, tmp_path):
    root = tmp_path / "root"
    remote.ship_source(LOCAL, str(root), repo)
    bare = root / remote.POD_GIT_REPO
    assert subprocess.run(["git", "rev-parse", "--verify", "-q", "refs/heads/main"], cwd=bare, capture_output=True).returncode != 0, \
        "a launcher without origin/main ships and leaves main alone"
    old = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", old)
    (repo / "a.txt").write_text("beta\n")
    _git(repo, "commit", "-qam", "two")
    new = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", new)
    _, dest, _ = remote.ship_source(LOCAL, str(root), repo)
    assert _git(Path(dest), "rev-parse", "origin/main") == new, "suites.py --changed origin/main diffs against the launcher's main"
    _git(repo, "update-ref", "refs/remotes/origin/main", old)
    (repo / "a.txt").write_text("gamma\n")
    _git(repo, "commit", "-qam", "three")
    _, dest3, doc3 = remote.ship_source(LOCAL, str(root), repo)
    assert doc3["transport"] == "git" and _git(Path(dest3), "rev-parse", "origin/main") == new, "an older launcher doesn't move main back"


def test_a_machine_without_git_gets_the_archive_stream_with_one_line(repo, tmp_path, capsys):
    root = tmp_path / "root"
    sha, dest, doc = remote.ship_source(_no_git_machine(tmp_path), str(root), repo)
    assert doc["verified"] == "archive-sha256+file-manifest" and doc["transport"] == "stream"
    err = capsys.readouterr().err
    assert "git ship to" in err and "(the machine has no git); shipping the archive stream" in err
    assert _tree(Path(dest)) == remote.archive_manifest_git(repo)["files"] and not (root / remote.POD_GIT_REPO).exists()
    assert not (Path(dest) / ".git").exists() and "checkout" not in doc


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


def _replica_repo(tmp_path: Path) -> Path:
    """A repository whose replica map registers a tracked fixture and a fetched one (gitignored, so no commit holds it), with this
    suite's own replica test at its usual path."""
    from research.store.fixtures import fixture_id

    d = tmp_path / "replicas"
    (d / "fixtures").mkdir(parents=True)
    (d / "tools" / "research" / "tests").mkdir(parents=True)
    shutil.copy(REPLICA_TEST, d / "tools" / "research" / "tests" / REPLICA_TEST.name)
    (d / "fixtures" / "small.bin").write_bytes(b"tracked\n")
    (d / ".gitignore").write_text("/fixtures/big.bin\n")
    reg = {fixture_id(d / "fixtures" / "small.bin"): "fixtures/small.bin", "art:" + "b" * 64: "fixtures/big.bin"}
    (d / "fixtures" / "artifacts.json").write_text(json.dumps(reg))
    _git(d, "init", "-q")
    _git(d, "add", ".")
    _git(d, "commit", "-q", "-m", "replicas")
    return d


def _replica_test_in(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tree / "tools" / "research" / "tests" / REPLICA_TEST.name)],
                          cwd=tree, capture_output=True, text=True)


def test_a_shipped_tree_answers_git_as_the_launchers_checkout_does(tmp_path, monkeypatch):
    """#371: the replica test skips a fetched fixture only where `git ls-files` answers.  A shipped `git archive` had no .git, so in a
    recorded check every migrated fixture read as missing.  Shipped over git, the tree is a checkout, and the same test passes there."""
    repo = _replica_repo(tmp_path)
    _, dest, _ = remote.ship_source(LOCAL, str(tmp_path / "git-root"), repo)
    p = _replica_test_in(Path(dest))
    assert p.returncode == 0 and "3 passed" in p.stdout, p.stdout + p.stderr
    assert _git(Path(dest), "check-ignore", "fixtures/big.bin") == "fixtures/big.bin"
    monkeypatch.setenv(remote.SHIP_ENV, "stream")
    _, bare_tree, _ = remote.ship_source(LOCAL, str(tmp_path / "stream-root"), repo)
    p = _replica_test_in(Path(bare_tree))
    assert p.returncode == 1 and "fixtures/big.bin is not in the working tree" in p.stdout, "without .git, #371's failure: " + p.stdout


def test_nothing_credential_like_is_left_on_the_machine(repo, tmp_path):
    root = tmp_path / "root"
    remote.ship_source(LOCAL, str(root), repo)
    bare = root / remote.POD_GIT_REPO
    cfg = (bare / "config").read_text()
    assert "url" not in cfg and "credential" not in cfg and "extraheader" not in cfg.lower()
    tree = root / "src" / _git(repo, "rev-parse", "HEAD")
    assert json.loads((tree / remote.READY_FILE).read_text())["transport"] == "git"
    keys = [line.split("=", 1)[0] for line in _git(tree, "config", "--local", "--list").splitlines()]
    assert _git(tree, "remote", "get-url", "origin") == str(bare), "the checkout's one remote is the machine's own bare repository"
    assert not [k for k in keys if k.startswith("credential") or "extraheader" in k], keys


def test_a_run_that_edits_the_shared_tree_is_not_what_the_next_run_of_that_commit_gets(repo, tmp_path, capsys):
    """INF-225: a probe ran `git checkout origin/main -- FILE` inside the node's shared tree for a commit, and every later run of that
    commit ran main's file.  The machine refuses such a tree, and the next ship sets it aside and ships the commit afresh."""
    root = tmp_path / "root"
    sha, dest, doc = remote.ship_source(LOCAL, str(root), repo)
    (Path(dest) / ".lake").mkdir()
    (Path(dest) / ".lake" / "build.olean").write_text("built\n")
    assert remote.tracked_dirt(Path(dest)) == [], "untracked build output is not dirt"
    (Path(dest) / "pkg" / "b.py").write_text("B = 99\n")
    assert remote.tracked_dirt(Path(dest)) == ["pkg/b.py"]
    with pytest.raises(remote.SourceNotReady, match=r"edited its tracked files \(pkg/b\.py\)"):
        remote.require_ready_source(dest, sha)
    sha2, dest2, doc2 = remote.ship_source(LOCAL, str(root), repo)
    assert (sha2, dest2) == (sha, dest) and (Path(dest) / "pkg" / "b.py").read_text() == "B = 2\n"
    assert remote.require_ready_source(dest, sha) == doc2 and remote.tracked_dirt(Path(dest)) == []
    (moved,) = (root / "src" / ".quarantine").iterdir()
    assert moved.name.startswith(f"{sha}-") and (moved / "pkg" / "b.py").read_text() == "B = 99\n", "kept for inspection"
    assert "an earlier run edited (pkg/b.py)" in capsys.readouterr().err
