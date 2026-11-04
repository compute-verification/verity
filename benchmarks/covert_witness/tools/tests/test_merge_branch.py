"""merge_branch on a toy remote: an island's branch merges while it leaves te_core alone and is refused once it touches
it; the toolsmith's VERSION merges with te_core exactly theirs, a te_core conflict included, is refused when it moves
products.CAPTURE or a version gate, and needs the coordinator's ruling when it changes the capture's code; --check
changes nothing."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import merge_branch as M

LANE, ISLAND = "cursor/covert-witness-4207", "cursor/te-alpha-4207"
PRODUCTS = f"{M.TE_CORE}/capture/products.py"
FORMALISM = f"{M.TE_CORE}/formalism.py"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bare origin and a clone on the lane's branch, whose te_core has the frozen capture and two gates."""
    (tmp_path / "gitconfig").write_text("")
    for name, value in {"GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}.items():
        monkeypatch.setenv(name, value)
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", LANE, str(origin))
    git(tmp_path, "init", "-q", "-b", LANE, str(work))
    git(work, "remote", "add", "origin", str(origin))
    write(work, {PRODUCTS: f'CAPTURE = "{M.FROZEN}"\n', FORMALISM: 'ELEMENTWISE_FROM = "4.12.9"  # why\nX = 1\n',
                 f"{M.TE_CORE}/tests/test_products.py": 'CAPTURE = "4.0"\n', "README.md": "harness\n"}, "start")
    git(work, "push", "-q", "origin", LANE)
    monkeypatch.chdir(work)
    return work


def write(work: Path, files: dict[str, str | None], msg: str, branch: str | None = None) -> str:
    """Commit `files` (None deletes one) on `branch` (made from the lane's when new), push it and return to the lane."""
    if branch:
        exists = git(work, "branch", "--list", branch).strip()
        git(work, "checkout", "-q", *([branch] if exists else ["-b", branch, LANE]))
    for path, text in files.items():
        f = work / path
        if text is None:
            f.unlink()
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", msg)
    sha = git(work, "rev-parse", "HEAD").strip()
    if branch:
        git(work, "push", "-q", "origin", branch)
        git(work, "checkout", "-q", LANE)
    return sha


def te_core(work: Path, rev: str) -> str:
    return git(work, "rev-parse", f"{rev}:{M.TE_CORE}").strip()


def test_an_island_merges_while_it_leaves_te_core_alone(work: Path) -> None:
    sha = write(work, {"benchmarks/covert_witness/islands/alpha/BOARD.md": "10:00Z alpha [TOOL] hi\n"}, "board", ISLAND)
    assert M.main([ISLAND, "-m", "alpha's board"]) == 0
    assert git(work, "log", "-1", "--format=%s").strip() == f"Merge {ISLAND} at {sha[:10]} (alpha's board)"
    assert git(work, "rev-parse", "HEAD^2").strip() == sha
    assert M.main([ISLAND]) == 0 and git(work, "rev-parse", "HEAD^2").strip() == sha


def test_an_island_that_changes_te_core_is_refused(work: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(work, {f"{M.TE_CORE}/frontier_fit.py": "fit = 2\n", "islands.txt": "x\n"}, "a local fix", ISLAND)
    head = git(work, "rev-parse", "HEAD")
    assert M.main([ISLAND, "--check"]) == 1 and M.main([ISLAND]) == 1
    err = capsys.readouterr().err
    assert "changes te_core (frontier_fit.py)" in err and M.TOOLSMITH in err
    assert git(work, "rev-parse", "HEAD") == head and not git(work, "status", "--porcelain")


def test_the_toolsmiths_version_merges_with_te_core_theirs(work: Path) -> None:
    sha = write(work, {FORMALISM: 'ELEMENTWISE_FROM = "4.12.9"  # why\nX = 3\n', f"{M.TE_CORE}/priced.py": "p\n",
                       f"{M.TE_CORE}/tests/test_products.py": 'CAPTURE = "4.12.13"\n', "README.md": "theirs\n",
                       f"{M.TE_CORE}/capture/README.md": "the capture\n",
                       f"{M.TE_CORE}/capture/tests/test_new.py": "x = 1\n"},
                "te_core 4.12.13", M.TOOLSMITH)
    write(work, {f"{M.TE_CORE}/local.py": "mine\n", FORMALISM: 'ELEMENTWISE_FROM = "4.12.9"  # why\nX = 2\n'},
          "a te_core edit on the lane, conflicting with the VERSION's")
    assert M.main([M.TOOLSMITH, "--at", sha[:10]]) == 0
    assert git(work, "rev-parse", "HEAD^2").strip() == sha and te_core(work, "HEAD") == te_core(work, sha)
    assert not (work / M.TE_CORE / "local.py").exists() and (work / "README.md").read_text() == "theirs\n"
    assert not git(work, "status", "--porcelain")


@pytest.mark.parametrize("files, named", [
    ({PRODUCTS: 'CAPTURE = "4.12.13"\n'}, 'moves products.CAPTURE to "4.12.13"'),
    ({PRODUCTS: None}, "moves products.CAPTURE to nothing"),
    ({FORMALISM: 'ELEMENTWISE_FROM = "4.12.13"\n'}, 'ELEMENTWISE_FROM "4.12.9" -> "4.12.13"'),
    ({f"{M.TE_CORE}/frontier.py": "MEASURED_FROM = (4, 13)\n"}, "frontier.py:MEASURED_FROM absent -> (4, 13)"),
])
def test_a_version_that_moves_the_capture_or_a_gate_is_refused(work: Path, capsys: pytest.CaptureFixture[str],
                                                              files: dict[str, str | None], named: str) -> None:
    write(work, files, "te_core 4.12.13", M.TOOLSMITH)
    head = git(work, "rev-parse", "HEAD")
    assert M.main([M.TOOLSMITH]) == 1
    err = capsys.readouterr().err
    assert named in err and f"frozen at {M.FROZEN}" in err
    assert git(work, "rev-parse", "HEAD") == head and not git(work, "status", "--porcelain")


def test_a_change_to_the_captures_code_merges_only_once_ruled(work: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sha = write(work, {PRODUCTS: f'CAPTURE = "{M.FROZEN}"\nLISTS = "fewer"\n'}, "te_core 4.12.13", M.TOOLSMITH)
    head = git(work, "rev-parse", "HEAD")
    assert M.main([M.TOOLSMITH]) == 1
    err = capsys.readouterr().err
    assert f"changes the capture's code (capture/products.py) under products.CAPTURE {M.FROZEN}" in err
    assert "--capture-ruled" in err and git(work, "rev-parse", "HEAD") == head
    assert M.main([M.TOOLSMITH, "-m", "4.12.13", "--capture-ruled", "the coordinator: it only removes listings"]) == 0
    assert git(work, "log", "-1", "--format=%s").strip() == (
        f"Merge {M.TOOLSMITH} at {sha[:10]} (4.12.13; its capture code change ruled: the coordinator: it only "
        f"removes listings)")


def test_check_changes_nothing_and_a_commit_off_the_branch_is_refused(work: Path,
                                                                      capsys: pytest.CaptureFixture[str]) -> None:
    write(work, {f"{M.TE_CORE}/priced.py": "p\n"}, "te_core 4.12.13", M.TOOLSMITH)
    off = write(work, {"other.txt": "o\n"}, "not on the toolsmith's branch", "cursor/te-beta-4207")
    head = git(work, "rev-parse", "HEAD")
    assert M.main([M.TOOLSMITH, "--check"]) == 0 and "may merge" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="is not on"):
        M.main([M.TOOLSMITH, "--at", off])
    assert git(work, "rev-parse", "HEAD") == head and not git(work, "status", "--porcelain")
