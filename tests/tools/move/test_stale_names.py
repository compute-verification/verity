"""`stale_names.py`: a PR merged with `main` after a word move merges clean and still spells the old word in its own new
lines; the tool lists each such line, with what the move's map would have written."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "stale_names.py"
ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
MAP = '[words]\n"flock-verify" = "verity-verify-proof"\n'
RUN = "#!/bin/sh\nflock-verify --in session.bin\n"


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def commit(repo: Path, msg: str, files: dict[str, str]) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def merged(tmp_path):
    """main renames the command flock-verify everywhere (a word move under its move line); the PR, from before the move,
    adds a script that calls it by the old name. The merge has no conflict."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    commit(r, "old", {"bin/run.sh": RUN, "README.md": "readme\n"})
    git(r, "branch", "pr")
    commit(r, "rename: flock-verify to verity-verify-proof\n\nMove: tools/move/verify_words_map.toml",
           {"tools/move/verify_words_map.toml": MAP, "bin/run.sh": RUN.replace("flock-verify", "verity-verify-proof")})
    git(r, "checkout", "-q", "pr")
    commit(r, "a check script", {"bin/check.sh": "#!/bin/sh\nset -e\nflock-verify --check \"$1\"\n",
                                 "bin/new.sh": "#!/bin/sh\nverity-verify-proof --in x\n"})
    git(r, "merge", "-q", "--no-edit", "main")
    return r


def run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], cwd=repo, capture_output=True, text=True)


def test_a_clean_merge_that_keeps_an_old_name_in_the_pr_s_own_line_is_listed(merged):
    assert git(merged, "show", "HEAD:bin/check.sh").count("flock-verify") == 1
    p = run(merged, "HEAD", "--json")
    assert p.returncode == 1, p.stderr
    r = json.loads(p.stdout)
    assert [(x["file"], x["line"], x["would_be"]) for x in r["lines"]] == [
        ("bin/check.sh", 3, 'verity-verify-proof --check "$1"')]
    assert r["paths"] == []


def test_a_merge_whose_own_lines_use_the_new_names_lists_nothing(merged):
    git(merged, "rm", "-q", "bin/check.sh")
    git(merged, "commit", "-q", "-m", "drop the check script")
    p = run(merged, "HEAD", "--onto", "main", "--head", "pr")
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.startswith("0 own lines still spell a name the maps rename; 0 added files")


def test_a_result_that_is_no_merge_is_refused_with_the_fix(merged):
    p = run(merged, "pr^1")
    assert p.returncode == 1 and "is no merge: commit the merge first" in p.stderr, p.stderr


def test_a_map_no_move_line_names_is_applied_with_map(merged):
    """A move commit whose message names its map only in prose is no move to restack.py; `--map` applies its map."""
    git(merged, "checkout", "-q", "main")
    commit(merged, "Rename by tools/move/session_words_map.toml",
           {"tools/move/session_words_map.toml": '[words]\n"session" = "challenge_phase"\n',
            "bin/run.sh": RUN.replace("flock-verify", "verity-verify-proof").replace("session", "challenge_phase")})
    git(merged, "checkout", "-q", "pr")
    git(merged, "reset", "-q", "--hard", "HEAD^")
    commit(merged, "a session reader", {"bin/read.sh": "#!/bin/sh\ncat session.bin\n"})
    git(merged, "merge", "-q", "--no-edit", "main")
    without = {x["file"] for x in json.loads(run(merged, "HEAD", "--json").stdout)["lines"]}
    assert "bin/read.sh" not in without
    r = json.loads(run(merged, "HEAD", "--json", "--map", "main:tools/move/session_words_map.toml").stdout)
    assert ("bin/read.sh", 2, "cat challenge_phase.bin") in [(x["file"], x["line"], x["would_be"]) for x in r["lines"]]
