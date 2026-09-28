"""What conftest.py keeps every test from reading off the machine it runs on."""
from __future__ import annotations

import subprocess


def test_git_reads_only_the_tests_own_config_and_commits_unsigned(tmp_path):
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    scoped = [line.split("\t", 1) for line in git("config", "--list", "--show-scope").splitlines()]
    assert {scope for scope, _ in scoped} <= {"global", "local"}
    assert sorted(kv for scope, kv in scoped if scope == "global") == [
        "commit.gpgsign=false", "tag.gpgsign=false", "user.email=t@t", "user.name=t"]
    git("commit", "-q", "--allow-empty", "-m", "one")
    assert "gpgsig" not in git("cat-file", "commit", "HEAD")
