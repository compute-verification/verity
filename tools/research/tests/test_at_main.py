"""`research` runs origin/main's code when the checkout's research code is only behind it (`research.at_main`)."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research import at_main

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
OWN = ["queue", "status"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def to_its_end(argv, timeout, **kw):
    """The set-up's runner without its bound: git takes as long as the machine makes it."""
    return subprocess.run(argv, **kw)


@pytest.fixture(autouse=True)
def no_bound_on_git(monkeypatch):
    """No test here depends on how long git takes: the set-up's bound is reached only when a test's own runner or clock
    says so."""
    monkeypatch.setattr(at_main, "run", to_its_end)
    monkeypatch.setattr(at_main, "clock", lambda: 0.0)
    monkeypatch.delenv(at_main.OPT_OUT, raising=False)


def run_shim(repo: Path, cache: Path, capsys) -> tuple[list[str], str]:
    """What `research queue status` runs for the checkout ``repo``, as the first entry of each started process's PYTHONPATH
    (the stub returns, so the checkout's own start follows origin/main's), and what it said on stderr."""
    ran = []
    at_main.shim(OWN, repo, cache, lambda exe, args, env: ran.append(env["PYTHONPATH"].split(os.pathsep)[0]))
    return ran, capsys.readouterr().err


def own(repo: Path) -> str:
    return str(repo / "tools/research/src")


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


def at_worktree_add(does):
    """A runner that is git's, but for `git worktree add`, where ``does(argv, timeout)`` stands in."""
    return lambda argv, timeout, **kw: does(argv, timeout) if argv[3:5] == ["worktree", "add"] else subprocess.run(argv, **kw)


def still_running(argv, timeout):
    raise subprocess.TimeoutExpired(argv, timeout)


CANT_SET_UP = {
    "a git call still running at the bound": (at_worktree_add(still_running), "its 60 s ran out at `git worktree add`"),
    "git fails": (at_worktree_add(lambda argv, timeout: subprocess.CompletedProcess(
        argv, 128, "", "fatal: could not create leading directories\nhint: two lines\n")),
        "`git worktree add` exited 128: fatal: could not create leading directories hint: two lines"),
    "a cache that can't be written": (None, "Not a directory"),
}


@pytest.mark.parametrize("case", list(CANT_SET_UP))
def test_a_worktree_that_cant_be_set_up_runs_this_checkouts_copy_and_says_why_in_one_line(tmp_path, monkeypatch, capsys, case):
    runner, why = CANT_SET_UP[case]
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    if runner is None:
        cache.write_text("a file where the cache's directory goes\n")
    else:
        monkeypatch.setattr(at_main, "run", runner)
    ran, err = run_shim(repo, cache, capsys)
    assert ran == [own(repo)]
    assert err.count("\n") == 1 and err.startswith("research: running this checkout's copy: origin/main's could not be set up (")
    assert why in err, err
    monkeypatch.setattr(at_main, "run", to_its_end)                     # the cause gone, the next command runs origin/main's
    if runner is None:
        cache.unlink()
    ran, err = run_shim(repo, cache, capsys)
    assert len(ran) == 2 and (Path(ran[0]) / "research/cli.py").read_text() == "new\n" and "behind origin/main" in err


def test_every_git_call_of_one_set_up_draws_on_the_one_bound(tmp_path, monkeypatch, capsys):
    repo, now, given = checkout(tmp_path), [0.0], []

    def takes_25_s(argv, timeout, **kw):
        given.append(timeout)
        now[0] += 25
        return subprocess.run(argv, **kw)

    monkeypatch.setattr(at_main, "run", takes_25_s)
    monkeypatch.setattr(at_main, "clock", lambda: now[0])
    ran, err = run_shim(repo, tmp_path / "cache", capsys)
    assert given == [60, 35, 10], "the fourth call, with nothing left, isn't started"
    assert ran == [own(repo)] and "(its 60 s ran out at `git diff`)" in err and err.count("\n") == 1


def test_a_command_that_cant_take_the_caches_lock_in_time_runs_this_checkouts_copy(tmp_path, monkeypatch, capsys):
    repo, cache, now = checkout(tmp_path), tmp_path / "cache", [0.0]
    wt = at_main.tree(repo, git(repo, "rev-parse", "origin/main"), cache)
    monkeypatch.setattr(at_main, "clock", lambda: now[0])
    monkeypatch.setattr(at_main, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    with open(wt.parent / "lock") as held:                              # another command's hold on the cache
        fcntl.flock(held, fcntl.LOCK_EX)
        ran, err = run_shim(repo, cache, capsys)
    assert ran == [own(repo)] and at_main.SETUP_S <= now[0] < at_main.SETUP_S + 2 * at_main.LOCK_POLL_S
    assert f"(its 60 s ran out at the wait for {wt.parent / 'lock'}, which another research command holds)" in err
    assert err.count("\n") == 1
    assert run_shim(repo, cache, capsys)[0] == [str(wt / "tools/research/src"), own(repo)]


def killed_checkout(argv, timeout, **kw):
    """git, but a checkout is killed part-way at the bound, leaving what a killed `git worktree add` or `git checkout`
    leaves: HEAD where it was set, the entry in the repository locked as one being made, the index locked, one file
    written and the rest not."""
    if argv[3] != "checkout":
        return subprocess.run(argv, **kw)
    wt = Path(argv[2])
    entry = Path((wt / ".git").read_text().split(": ", 1)[1].strip())
    (entry / "locked").write_text("initializing\n")
    (entry / "index.lock").touch()
    (wt / "README.md").write_text("half of what the checkout was writing\n")
    raise subprocess.TimeoutExpired(argv, timeout)


@pytest.mark.parametrize("case", ["made", "moved", "a move git refuses"])
def test_a_worktree_a_killed_git_left_half_made_is_never_run_from_and_the_next_command_makes_it_again(
        tmp_path, monkeypatch, capsys, case):
    """A `git worktree add` killed at its timeout left a tree whose HEAD was already origin/main's, so the next command
    took it for whole and ran from it, with no research code in it, or half of it, until origin/main moved; and then
    `git worktree add` refused its locked entry for good."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    main, text = git(repo, "rev-parse", "origin/main"), "new\n"
    if case != "made":
        wt = at_main.tree(repo, main, cache)
        git(repo, "checkout", "-q", "main")
        main, text = commit(repo, "tools/research/src/research/cli.py", "newer\n"), "newer\n"
        git(repo, "update-ref", "refs/remotes/origin/main", main)
        git(repo, "checkout", "-q", "lane")
    if case == "a move git refuses":
        (Path(git(wt, "rev-parse", "--absolute-git-dir")) / "index.lock").touch()
    else:
        monkeypatch.setattr(at_main, "run", killed_checkout)
        ran, err = run_shim(repo, cache, capsys)
        assert ran == [own(repo)] and "(its 60 s ran out at `git checkout`)" in err and err.count("\n") == 1
        wt = next(cache.glob("*/tree"))
        if case == "made":
            assert git(wt, "rev-parse", "HEAD") == main and not (wt / "tools").exists(), "it looks made, and isn't"
        monkeypatch.setattr(at_main, "run", to_its_end)
    assert run_shim(repo, cache, capsys)[0] == [str(wt / "tools/research/src"), own(repo)]
    assert (wt / "tools/research/src/research/cli.py").read_text() == text and (wt / "README.md").read_text() == "docs\n"
    assert git(wt, "rev-parse", "HEAD") == main and git(wt, "status", "--porcelain") == ""
    assert len(git(repo, "worktree", "list").splitlines()) == 2, "one entry for the cached tree, and none locked"
    assert not (Path(git(wt, "rev-parse", "--absolute-git-dir")) / "locked").exists()


def test_a_research_a_test_starts_never_loads_at_main():
    """Other packages' suites start `python -m research` without declaring at_main.py, so loading it failed their read
    guard; under a test the tree's code runs anyway."""
    probe = ("import sys; sys.argv = ['research', '--help']; from research import cli; cli.main(); "
             "print('at_main' if 'research.at_main' in sys.modules else 'clean')")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[-1] == "clean"
