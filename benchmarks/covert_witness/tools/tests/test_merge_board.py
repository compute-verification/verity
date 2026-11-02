"""merge_board on a toy remote: every search branch's board, the integrator's and the integration branch's merge into
the digest once each, in time order across midnight, with the digest's own text and lines kept, and `--push` commits a
change once."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import merge_board

ISLANDS = "benchmarks/covert_witness/islands"
LANE, DIGEST = "cursor/training-efficiency-4207", "cursor/te-digest-4207"
HEAD = "# The board\n\nOne line per entry.\n\n## Entries"
TAIL = "## Board move\nWrite to your own branch."
MERGED = f"""{HEAD}

### 2026-11-02
19:58Z integrator [TOOL] pinned
20:10Z integrator [TOOL] on the integrator's branch alone
20:30Z coordinator [DIGEST] written in the digest
23:50Z alpha [FINDING] before midnight

### 2026-11-03
00:10Z alpha [NEGATIVE] after midnight
00:30Z beta [TOOL] a lowercase board
00:45Z beta [FINDING] stamped ahead of its commit

{TAIL}
"""


def git(cwd: Path, *args: str, at: str | None = None) -> str:
    env = os.environ | ({"GIT_AUTHOR_DATE": at, "GIT_COMMITTER_DATE": at} if at else {})
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A bare origin holding the lane's branches, and a fresh clone of it to merge from."""
    (tmp_path / "gitconfig").write_text("")
    for name, value in {"GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1",
                        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}.items():
        monkeypatch.setenv(name, value)
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", LANE, str(origin))
    git(tmp_path, "init", "-q", "-b", LANE, str(work))
    git(work, "remote", "add", "origin", str(origin))

    def append(branch: str, path: str, text: str, at: str) -> None:
        if git(work, "branch", "--show-current").strip() != branch:
            exists = git(work, "branch", "--list", branch).strip()
            git(work, "checkout", "-q", *([branch] if exists else ["-b", branch, LANE]))
        f = work / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text((f.read_text() if f.exists() else "") + text)
        git(work, "add", path)
        git(work, "commit", "-q", "-m", f"{branch}: {path}", at=at)
        git(work, "push", "-q", "origin", branch)

    append(LANE, f"{ISLANDS}/integrator/BOARD.md", "# The integrator's board\n\n19:58Z integrator [TOOL] pinned\n",
           "2026-11-02T20:00:00Z")
    append("cursor/covert-witness-4207", f"{ISLANDS}/integrator/BOARD.md",
           "20:10Z integrator [TOOL] on the integrator's branch alone\n", "2026-11-02T20:15:00Z")
    append(DIGEST, "benchmarks/covert_witness/BOARD.md",
           f"{HEAD}\n19:58Z integrator [TOOL] pinned\n20:30Z coordinator [DIGEST] written in the digest\n\n{TAIL}\n",
           "2026-11-02T21:00:00Z")
    append("cursor/te-alpha-4207", f"{ISLANDS}/alpha/BOARD.md", "23:50Z alpha [FINDING] before midnight\n",
           "2026-11-02T23:55:00Z")
    append("cursor/te-alpha-4207", f"{ISLANDS}/alpha/BOARD.md", "00:10Z alpha [NEGATIVE] after midnight\n",
           "2026-11-03T00:20:00Z")
    append("cursor/te-beta-4207", f"{ISLANDS}/beta/board.md",
           "notes, not an entry\n23:50Z alpha [FINDING] before midnight\n00:30Z beta [TOOL] a lowercase board\n"
           "00:45Z beta [FINDING] stamped ahead of its commit\n", "2026-11-03T00:40:00Z")
    append("cursor/other-4207", f"{ISLANDS}/other/BOARD.md", "00:50Z other [FINDING] not a search branch\n",
           "2026-11-03T00:50:00Z")
    git(tmp_path, "clone", "-q", str(origin), "clone")
    return tmp_path / "clone"


def test_the_digest_holds_every_board_once_in_time_order(clone: Path, tmp_path: Path) -> None:
    out = tmp_path / "digest.md"
    assert merge_board.main(["--repo", str(clone), "--out", str(out)]) == 0
    assert out.read_text() == MERGED


def test_push_commits_a_changed_digest_once(clone: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pushed days later, the digest keeps each entry's day: a board's line is dated by its board, not by the digest."""
    for name in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):
        monkeypatch.setenv(name, "2026-11-05T12:00:00Z")
    before = git(clone, "rev-parse", f"origin/{DIGEST}").strip()
    assert merge_board.main(["--repo", str(clone), "--push"]) == 0
    pushed = git(clone, "ls-remote", "origin", f"refs/heads/{DIGEST}").split()[0]
    assert git(clone, "rev-parse", f"{pushed}^") == before + "\n"
    assert git(clone, "show", f"{pushed}:benchmarks/covert_witness/BOARD.md") == MERGED
    assert merge_board.main(["--repo", str(clone), "--push"]) == 0
    assert git(clone, "ls-remote", "origin", f"refs/heads/{DIGEST}").split()[0] == pushed


def test_a_new_digest_branch_starts_from_the_integration_branch(clone: Path) -> None:
    assert merge_board.main(["--repo", str(clone), "--push", "--digest", "cursor/te-new-digest-4207"]) == 0
    new = git(clone, "ls-remote", "origin", "refs/heads/cursor/te-new-digest-4207").split()[0]
    assert git(clone, "rev-parse", f"{new}^") == git(clone, "rev-parse", f"origin/{LANE}")
    text = git(clone, "show", f"{new}:benchmarks/covert_witness/BOARD.md")
    assert text.startswith(merge_board.NEW + "\n### 2026-11-02\n19:58Z integrator [TOOL] pinned\n")
