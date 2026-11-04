"""`research` runs origin/main's code when the checkout's research code is only behind it (`research.at_main`)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from research import at_main

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def commit(repo: Path, path: str, text: str) -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", path)
    return git(repo, "rev-parse", "HEAD")


def checkout(tmp_path: Path) -> Path:
    """A repository on branch `lane`, cut from main, with `origin/main` one research commit and one other commit ahead."""
    repo = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    commit(repo, "tools/research/src/research/cli.py", "old\n")
    git(repo, "checkout", "-q", "-b", "lane")
    git(repo, "checkout", "-q", "main")
    commit(repo, "tools/research/src/research/cli.py", "new\n")
    main = commit(repo, "README.md", "docs\n")
    git(repo, "update-ref", "refs/remotes/origin/main", main)
    git(repo, "checkout", "-q", "lane")
    return repo


def test_a_checkout_only_behind_runs_origin_mains_research_from_a_cached_worktree_moved_with_main(tmp_path):
    repo = checkout(tmp_path)
    commit(repo, "verity/x.py", "the lane's own work elsewhere\n")
    main, n = at_main.behind(repo)
    assert main == git(repo, "rev-parse", "origin/main") and n == 1
    wt = at_main.tree(repo, main, tmp_path / "cache")
    assert (wt / "tools/research/src/research/cli.py").read_text() == "new\n"
    assert at_main.tree(repo, main, tmp_path / "cache") == wt
    git(repo, "checkout", "-q", "main")
    later = commit(repo, "tools/research/src/research/cli.py", "newer\n")
    git(repo, "update-ref", "refs/remotes/origin/main", later)
    git(repo, "checkout", "-q", "lane")
    assert at_main.tree(repo, later, tmp_path / "cache") == wt
    assert (wt / "tools/research/src/research/cli.py").read_text() == "newer\n"


def test_a_checkout_that_changes_research_code_or_is_level_with_main_runs_its_own(tmp_path):
    repo = checkout(tmp_path)
    (repo / "tools/research/src/research/cli.py").write_text("being edited\n")
    assert at_main.behind(repo) is None, "uncommitted research code"
    git(repo, "checkout", "-q", "--", ".")
    commit(repo, "infra/ops/verity_ops/x.py", "the lane's change to ops\n")
    assert at_main.behind(repo) is None, "a committed change of its own"
    git(repo, "checkout", "-q", "-B", "level", "origin/main")
    assert at_main.behind(repo) is None, "level with origin/main"
    git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    assert at_main.behind(repo) is None, "no origin/main"


def test_the_opt_out_and_a_tree_outside_git_run_this_copy(tmp_path, monkeypatch):
    repo, ran = checkout(tmp_path), []
    monkeypatch.setenv("RESEARCH_AT_CHECKOUT", "1")
    at_main.reexec(["--version"], repo, tmp_path / "cache", lambda *a: ran.append(a))
    monkeypatch.delenv("RESEARCH_AT_CHECKOUT")
    at_main.reexec(["--version"], tmp_path / "not-a-checkout", tmp_path / "cache", lambda *a: ran.append(a))
    assert not ran and not (tmp_path / "cache").exists()


def test_the_shim_runs_origin_mains_code_for_a_stale_checkout_and_the_checkouts_own_otherwise(tmp_path, monkeypatch):
    """What /usr/local/bin/research runs: a checkout from before `at_main` can't re-execute itself, so the shim decides."""
    monkeypatch.delenv("RESEARCH_AT_CHECKOUT", raising=False)
    repo, ran = checkout(tmp_path), []
    at_main.shim(["queue", "status"], repo, tmp_path / "cache", lambda exe, args, env: ran.append((args, env)))
    (args, env), = ran[:1]                              # the stub returns, so the shim's fallback start follows it
    wt = at_main.tree(repo, git(repo, "rev-parse", "origin/main"), tmp_path / "cache")
    assert args[1:] == ["-m", "research", "queue", "status"] and env["RESEARCH_AT_CHECKOUT"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(wt / "tools/research/src")
    ran.clear()
    commit(repo, "tools/research/src/research/cli.py", "the lane's own\n")
    at_main.shim(["queue", "status"], repo, tmp_path / "cache", lambda exe, args, env: ran.append((args, env)))
    assert [env["PYTHONPATH"].split(os.pathsep)[0] for _, env in ran] == [str(repo / "tools/research/src")]


def test_a_research_a_test_starts_never_loads_at_main():
    """Other packages' suites start `python -m research` without declaring at_main.py, so loading it failed their read
    guard; under a test the tree's code runs anyway."""
    probe = ("import sys; sys.argv = ['research', '--help']; from research import cli; cli.main(); "
             "print('at_main' if 'research.at_main' in sys.modules else 'clean')")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[-1] == "clean"
