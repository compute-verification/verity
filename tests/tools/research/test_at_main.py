"""`research` runs origin/main's code when the checkout's research code is only behind it (`research.at_main`)."""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research import at_main, guard_notice

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
ARGV = ["queue", "status"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def to_its_end(argv, timeout, **kw):
    """The set-up's runner without its bound: git takes as long as the machine makes it."""
    return subprocess.run(argv, **kw)


#: the wall clock's reading in every test: a pick is dated at it, and an entry's idleness read against it
NOW = 2_000_000_000.0


@pytest.fixture(autouse=True)
def no_bound_on_git(monkeypatch, tmp_path):
    """No test here depends on how long git takes: the set-up's bound is reached only when a test's own runner or clock
    says so. Each test's in-use holds are its own, let go when it ends, and the only live processes a test's pruner sees
    are the ones it lists (`process`)."""
    monkeypatch.setattr(at_main, "run", to_its_end)
    monkeypatch.setattr(at_main, "clock", lambda: 0.0)
    monkeypatch.setattr(at_main, "wall", lambda: NOW)
    monkeypatch.setattr(at_main, "held", [])
    (tmp_path / "proc").mkdir()
    monkeypatch.setattr(at_main, "PROC", tmp_path / "proc")
    monkeypatch.delenv(at_main.OPT_OUT, raising=False)
    yield
    exited()


def exited() -> None:
    """The command this test played has exited: the in-use holds it took are let go."""
    for f in at_main.held:
        f.close()
    at_main.held.clear()


def run_shim(repo: Path, cache: Path, capsys) -> tuple[list[str], str]:
    """What `research queue status` runs for the checkout ``repo``, as the first entry of each started process's PYTHONPATH
    (the stub returns, so the checkout's own start follows origin/main's), and what it said on stderr."""
    ran = []
    at_main.shim(ARGV, repo, cache, lambda exe, args, env: ran.append(env["PYTHONPATH"].split(os.pathsep)[0]))
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


def moved_on(repo: Path) -> str:
    """origin/main one research commit later: that commit."""
    git(repo, "checkout", "-q", "main")
    later = commit(repo, "tools/research/src/research/cli.py", "newer\n")
    git(repo, "update-ref", "refs/remotes/origin/main", later)
    git(repo, "checkout", "-q", "lane")
    return later


def test_a_checkout_only_behind_runs_origin_mains_research_from_a_cached_worktree_moved_with_main(tmp_path):
    repo = checkout(tmp_path)
    commit(repo, "verity/x.py", "the lane's own work elsewhere\n")
    main, n = at_main.behind(repo)
    assert main == git(repo, "rev-parse", "origin/main") and n == 1
    wt = at_main.tree(repo, main, tmp_path / "cache")
    assert (wt / "tools/research/src/research/cli.py").read_text() == "new\n"
    assert at_main.tree(repo, main, tmp_path / "cache") == wt
    later = moved_on(repo)
    assert at_main.tree(repo, later, tmp_path / "cache") == wt
    assert (wt / "tools/research/src/research/cli.py").read_text() == "newer\n"


def noting(seen: list[str]):
    """The set-up's runner without its bound, noting in ``seen`` what it is given: git's subcommand, else the program."""
    def runner(argv, timeout, **kw):
        seen.append(" ".join(a for a in argv[3:5] if not a.startswith("-")) if argv[0] == "git" else argv[0])
        return subprocess.run(argv, **kw)
    return runner


def made(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A checkout, its cache, and the cache's tree made at origin/main, holding one file more that a tree made again
    wouldn't have."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    wt = at_main.tree(repo, git(repo, "rev-parse", "origin/main"), cache)
    (wt / "kept").write_text("no commit has this file\n")
    return repo, cache, wt


def test_a_whole_tree_at_origin_mains_commit_is_reused_as_it_is(tmp_path, monkeypatch):
    """Were the stamp never written, or not read as this commit's, every command would remove the tree and make it again."""
    (repo, cache, wt), seen = made(tmp_path), []
    main = git(repo, "rev-parse", "origin/main")
    monkeypatch.setattr(at_main, "run", noting(seen))
    assert at_main.tree(repo, main, cache) == wt
    assert seen == ["rev-parse"], "the one git call finds the cache: nothing is removed, made or moved"
    assert (wt / "kept").exists() and (wt.parent / "at").read_text().split() == [main]


def test_a_whole_tree_is_moved_to_origin_mains_new_commit_and_not_made_again(tmp_path, monkeypatch):
    (repo, cache, wt), seen = made(tmp_path), []
    (wt / "README.md").write_text("an edit made in the cache, to a file the move doesn't change\n")
    later = moved_on(repo)
    monkeypatch.setattr(at_main, "run", noting(seen))
    assert at_main.tree(repo, later, cache) == wt
    assert seen == ["rev-parse", "checkout"], "one checkout moves it: nothing is removed or made"
    assert (wt / "kept").exists() and (wt / "tools/research/src/research/cli.py").read_text() == "newer\n"
    assert git(wt, "rev-parse", "HEAD") == later and git(wt, "status", "--porcelain") == "?? kept", "every file is the commit's"
    assert (wt.parent / "at").read_text().split() == [later], "the stamp follows the tree"
    assert at_main.tree(repo, later, cache) == wt and seen == ["rev-parse", "checkout", "rev-parse"], "and it is reused there"


def test_a_stamp_whose_tree_is_gone_counts_for_nothing(tmp_path, capsys):
    repo, cache, wt = made(tmp_path)
    shutil.rmtree(wt)
    assert run_shim(repo, cache, capsys)[0] == [str(wt / "tools/research/src"), own(repo)]
    assert (wt / "tools/research/src/research/cli.py").read_text() == "new\n" and not (wt / "kept").exists()


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


#: a research package's `__main__`, which says whose copy it is and what it was asked
WHOSE = "import sys\nprint({!r}, *sys.argv[1:])\n"


def two_researches(tmp_path: Path) -> Path:
    """A repository on branch `lane`, cut from main, whose research `python -m research` runs, saying it is the
    checkout's, with `origin/main` one research commit ahead, whose research says it is origin/main's."""
    repo, main_py = tmp_path / "repo", "tools/research/src/research/__main__.py"
    git(tmp_path, "init", "-q", "-b", "main", str(repo))
    commit(repo, "tools/research/src/research/__init__.py", "")
    commit(repo, main_py, WHOSE.format("the checkout's"))
    git(repo, "checkout", "-q", "-b", "lane")
    git(repo, "checkout", "-q", "main")
    git(repo, "update-ref", "refs/remotes/origin/main", commit(repo, main_py, WHOSE.format("origin/main's")))
    git(repo, "checkout", "-q", "lane")
    return repo


def as_the_shim_runs_it(repo: Path, cache: Path, *argv: str, **env: str) -> subprocess.CompletedProcess:
    """`research ARGV` as /usr/local/bin/research (`tools/agent-guard/research.sh`) runs it: this file's source as
    `python -c SOURCE ARGV`, with no package, and `RESEARCH_CHECKOUT` naming the checkout."""
    base = {k: v for k, v in ENV.items() if k != at_main.OPT_OUT}
    return subprocess.run([sys.executable, "-c", Path(at_main.__file__).read_text(), *argv], cwd=repo, capture_output=True,
                          text=True, env={**base, "RESEARCH_CHECKOUT": str(repo), "XDG_CACHE_HOME": str(cache), **env})


def test_the_shims_script_form_runs_origin_mains_copy_or_with_the_opt_out_the_checkouts(tmp_path):
    """A read of the settings registry that needs the package (`from . import settings`) failed every command the shim
    ran, before any of this module ran; the tests above import it as `research.at_main`, where it has one."""
    repo, cache = two_researches(tmp_path), tmp_path / "cache"
    behind = as_the_shim_runs_it(repo, cache, "--version")
    assert behind.returncode == 0 and "Traceback" not in behind.stderr, behind.stderr
    assert behind.stdout == "origin/main's --version\n"
    assert behind.stderr.startswith("research: this checkout's research code is 1 commit(s) behind origin/main")
    opted_out = as_the_shim_runs_it(repo, cache, "--version", RESEARCH_AT_CHECKOUT="1")
    assert (opted_out.returncode, opted_out.stdout, opted_out.stderr) == (0, "the checkout's --version\n", "")


#: an agent's conversation id, as Cursor's shell sets it
AGENT = "bc-00000000-0000-4000-8000-000000000001"


def tree_at(root: Path, git_checkout: bool = True) -> Path:
    """A tree at `root` holding a research package, a git checkout's (its `.git` a worktree's file) unless not: the
    package's directory."""
    package = root / "tools/research/src/research"
    package.mkdir(parents=True)
    if git_checkout:
        (root / ".git").write_text("gitdir: elsewhere\n")
    return package


def on_an_agents_vm(tmp_path: Path, monkeypatch) -> Path:
    """An agent's VM: its venv /workspace's, the shell in /workspace, no PYTHONPATH and no shim's checkout. /workspace."""
    workspace = tmp_path / "workspace"
    tree_at(workspace)
    monkeypatch.setattr(sys, "prefix", str(workspace / ".venv"))
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CURSOR_CONVERSATION_ID", AGENT)
    for name in ("PYTHONPATH", "RESEARCH_CHECKOUT", "RESEARCH_AT_CHECKOUT"):
        monkeypatch.delenv(name, raising=False)
    return workspace


def said(tmp_path: Path, monkeypatch, here: Path, under_test: bool = False) -> str | None:
    """What `pythonpath_notice` says for the package `here`, as a command outside a test (unless `under_test`) says it."""
    if not under_test:
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    return at_main.pythonpath_notice(tmp_path / "cache", here, tmp_path / "marks")


def test_a_pythonpath_that_runs_a_third_trees_research_is_said_once_an_agents_session(tmp_path, monkeypatch, capsys):
    """A startup file's PYTHONPATH ran another tree's research on every command, and nothing said so."""
    on_an_agents_vm(tmp_path, monkeypatch)
    third = tree_at(tmp_path / "wt-main")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(third.parent), "/elsewhere"]))
    line = (f"research: PYTHONPATH puts {tmp_path / 'wt-main'}'s research ahead of this checkout's and of origin/main's, "
            f"so that copy runs; unset PYTHONPATH, or put it on the one command")
    assert said(tmp_path, monkeypatch, third) == line and capsys.readouterr().err == line + "\n"
    assert said(tmp_path, monkeypatch, third) is None and capsys.readouterr().err == "", "once a session"


def at_mains_re_exec(tmp_path, monkeypatch, workspace):
    here = tree_at(tmp_path / "cache/0123456789ab/tree")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(here.parent), str(workspace / "tools/research/src")]))
    monkeypatch.setenv("RESEARCH_AT_CHECKOUT", "1")
    return here


def a_worktrees_own_from_inside_it(tmp_path, monkeypatch, workspace):
    here = tree_at(tmp_path / "wt")
    monkeypatch.setenv("PYTHONPATH", str(here.parent))
    monkeypatch.setenv("RESEARCH_AT_CHECKOUT", "1")
    monkeypatch.chdir(here.parents[2])
    return here


def the_shims_checkout(tmp_path, monkeypatch, workspace):
    here = tree_at(tmp_path / "wt")
    monkeypatch.setenv("PYTHONPATH", str(here.parent))
    monkeypatch.setenv("RESEARCH_CHECKOUT", str(tmp_path / "wt"))
    monkeypatch.chdir(tmp_path)
    return here


def the_venvs_own_from_outside_any_checkout(tmp_path, monkeypatch, workspace):
    monkeypatch.setenv("PYTHONPATH", str(workspace / "tools/research/src"))
    monkeypatch.chdir(tmp_path)
    return workspace / "tools/research/src/research"


def a_third_tree_not_on_pythonpath(tmp_path, monkeypatch, workspace):
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    return tree_at(tmp_path / "wt-main")


def a_nodes_installed_copy(tmp_path, monkeypatch, workspace):
    here = tree_at(tmp_path / "usr/local/lib/vy-monitoring/check-slot", git_checkout=False)
    monkeypatch.setenv("PYTHONPATH", str(here.parent))
    return here


def a_node_with_no_agent(tmp_path, monkeypatch, workspace):
    monkeypatch.delenv("CURSOR_CONVERSATION_ID")
    return a_third_tree_on_pythonpath(tmp_path, monkeypatch, workspace)


def a_third_tree_on_pythonpath(tmp_path, monkeypatch, workspace):
    here = tree_at(tmp_path / "wt-main")
    monkeypatch.setenv("PYTHONPATH", str(here.parent))
    return here


@pytest.mark.parametrize("case", [at_mains_re_exec, a_worktrees_own_from_inside_it, the_shims_checkout,
                                  the_venvs_own_from_outside_any_checkout, a_third_tree_not_on_pythonpath,
                                  a_nodes_installed_copy, a_node_with_no_agent, "a test"], ids=lambda c: getattr(c, "__name__", c))
def test_the_copy_a_command_is_meant_to_run_and_a_node_or_a_test_say_nothing_of_pythonpath(tmp_path, monkeypatch, capsys,
                                                                                         case):
    workspace = on_an_agents_vm(tmp_path, monkeypatch)
    if case == "a test":
        assert said(tmp_path, monkeypatch, a_third_tree_on_pythonpath(tmp_path, monkeypatch, workspace), under_test=True) is None
    else:
        assert said(tmp_path, monkeypatch, case(tmp_path, monkeypatch, workspace)) is None
    assert capsys.readouterr().err == ""


def no_guard_notice(env) -> None:
    """`guard_notice.notice` with nothing to say: no test asks the VM's own guard."""


def test_a_mark_directory_that_cant_be_read_is_one_line_and_the_command_goes_on(tmp_path, monkeypatch, capsys):
    """Another user's mode-700 mark directory raised PermissionError from `mark.exists()`, which stopped the command."""
    workspace = on_an_agents_vm(tmp_path, monkeypatch)
    third = tree_at(tmp_path / "wt-main")
    monkeypatch.setenv("PYTHONPATH", str(third.parent))
    monkeypatch.setattr(at_main, "__file__", str(third / "at_main.py"))
    monkeypatch.setattr(at_main.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    closed, exists = tmp_path / "tmp/research-pythonpath-notice", Path.exists

    def denied(self, *args, **kwargs):
        if self.is_relative_to(closed):
            raise PermissionError(13, "Permission denied", str(self))
        return exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", denied)
    monkeypatch.setattr(guard_notice, "notice", no_guard_notice)
    monkeypatch.setenv(at_main.OPT_OUT, "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert at_main.reexec(["--version"], workspace, tmp_path / "cache") is None
    assert capsys.readouterr().err == (f"research: pythonpath_notice could not run, and the command goes on: [Errno 13] "
                                       f"Permission denied: '{closed / AGENT}'\n")


def test_no_home_costs_no_command_that_returns_before_it_needs_the_cache(tmp_path, monkeypatch, capsys):
    """`Path.home()` raises RuntimeError with no HOME and no passwd entry: computed first, it stopped every command."""
    def no_home(cls):
        raise RuntimeError("Could not determine home directory.")

    for name in ("HOME", "XDG_CACHE_HOME", "CURSOR_CONVERSATION_ID", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(no_home))
    monkeypatch.setattr(guard_notice, "notice", no_guard_notice)
    monkeypatch.setenv(at_main.OPT_OUT, "1")
    assert at_main.reexec(["--version"], checkout(tmp_path)) is None
    assert capsys.readouterr().err == ""


def test_a_guard_notice_that_raises_is_one_line_and_the_command_goes_on(tmp_path, monkeypatch, capsys):
    def unreadable(env):
        raise OSError("tools/agent-guard/inbox.py: Permission denied")

    monkeypatch.setattr(guard_notice, "notice", unreadable)
    monkeypatch.setenv(at_main.OPT_OUT, "1")
    assert at_main.reexec(["--version"], checkout(tmp_path), tmp_path / "cache") is None
    assert capsys.readouterr().err == ("research: guard_notice could not run, and the command goes on: "
                                       "tools/agent-guard/inbox.py: Permission denied\n")


def test_research_run_from_a_third_tree_by_pythonpath_says_so_once_and_exits_as_it_would_have(tmp_path):
    """The line comes from the package that runs: a copy of this one, in a git checkout no command is for."""
    third = tmp_path / "wt-main"
    shutil.copytree(Path(at_main.__file__).parent, third / "tools/research/src/research",
                    ignore=shutil.ignore_patterns("__pycache__"))
    git(tmp_path, "init", "-q", str(third))
    git(tmp_path, "init", "-q", str(tmp_path / "checkout"))
    (tmp_path / "tmp").mkdir()
    base = {k: v for k, v in ENV.items() if k not in ("PYTEST_CURRENT_TEST", "CURSOR_CONVERSATION_ID", "RESEARCH_CHECKOUT",
                                                      at_main.OPT_OUT)}
    base |= {"PYTHONPATH": str(third / "tools/research/src"), "TMPDIR": str(tmp_path / "tmp"),
             "XDG_CACHE_HOME": str(tmp_path / "cache")}

    def research(**env: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "research", "--version"], cwd=tmp_path / "checkout",
                              capture_output=True, text=True, env={**base, **env})

    plain, told, again = research(), research(CURSOR_CONVERSATION_ID=AGENT), research(CURSOR_CONVERSATION_ID=AGENT)
    line = (f"research: PYTHONPATH puts {third}'s research ahead of this checkout's and of origin/main's, so that copy "
            f"runs; unset PYTHONPATH, or put it on the one command\n")
    assert (plain.returncode, plain.stderr) == (0, ""), plain.stderr
    assert (told.returncode, told.stdout, told.stderr) == (plain.returncode, plain.stdout, line)
    assert (again.returncode, again.stdout, again.stderr) == (plain.returncode, plain.stdout, "")


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


def test_every_git_call_of_one_set_up_shares_the_one_bound(tmp_path, monkeypatch, capsys):
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
        main, text = moved_on(repo), "newer\n"
    if case == "a move git refuses":
        (Path(git(wt, "rev-parse", "--absolute-git-dir")) / "index.lock").touch()
    else:
        monkeypatch.setattr(at_main, "run", killed_checkout)
        ran, err = run_shim(repo, cache, capsys)
        assert ran == [own(repo)] and "(its 60 s ran out at `git checkout`)" in err and err.count("\n") == 1
        wt = next(cache.glob("*/tree"))
        assert not (wt.parent / "at").exists(), "no stamp calls whole a tree git was killed changing"
        if case == "made":
            assert git(wt, "rev-parse", "HEAD") == main and not (wt / "tools").exists(), "it looks made, and isn't"
        monkeypatch.setattr(at_main, "run", to_its_end)
    assert run_shim(repo, cache, capsys)[0] == [str(wt / "tools/research/src"), own(repo)]
    assert (wt / "tools/research/src/research/cli.py").read_text() == text and (wt / "README.md").read_text() == "docs\n"
    assert git(wt, "rev-parse", "HEAD") == main and git(wt, "status", "--porcelain") == ""
    assert len(git(repo, "worktree", "list").splitlines()) == 2, "one entry for the cached tree, and none locked"
    assert not (Path(git(wt, "rev-parse", "--absolute-git-dir")) / "locked").exists()


IDLE = at_main.IDLE_S


def entry(cache: Path, n: int, idle_s: float, whole: bool = True) -> Path:
    """An entry of the cache as `tree` leaves one (a tree, its stamp, its lock), last picked ``idle_s`` before NOW; with
    ``whole`` false, one whose checkout never finished (no stamp)."""
    home = cache / f"{n:012x}"
    (home / "tree/tools").mkdir(parents=True)
    (home / "tree/tools/f").write_text("a file of the tree\n")
    if whole:
        (home / "at").write_text("c" * 40 + "\n")
    (home / "lock").touch()
    os.utime(home / "lock", (NOW - idle_s, NOW - idle_s))
    return home


def whole(home: Path) -> bool:
    return (home / "lock").is_file() and (home / "at").is_file() and (home / "tree").is_dir()


def entries(cache: Path) -> list[Path]:
    return sorted(cache.iterdir())


def test_each_command_removes_the_longest_unpicked_entry_past_the_newest_once_idle_s_has_passed(tmp_path, capsys):
    """The cache grew by one entry for every repository a machine ran `research` from and never shrank."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    newest = [entry(cache, i, IDLE * (i + 1) / at_main.KEEP) for i in range(at_main.KEEP - 1)]
    not_yet = entry(cache, 100, IDLE)
    old, oldest = entry(cache, 200, 2 * IDLE), entry(cache, 201, 3 * IDLE)
    ran, err = run_shim(repo, cache, capsys)
    own = Path(ran[0]).parents[3]
    assert len(ran) == 2 and own.parent == cache and err.count("\n") == 1, "origin/main's runs, and nothing failed"
    assert entries(cache) == sorted([own, *newest, not_yet, old]) and not oldest.exists(), \
        "one removal a command, the longest unpicked first"
    run_shim(repo, cache, capsys)
    assert entries(cache) == sorted([own, *newest, not_yet])
    assert run_shim(repo, cache, capsys)[0][0] == ran[0] and entries(cache) == sorted([own, *newest, not_yet])
    assert all(whole(h) for h in [own, *newest, not_yet])


def test_the_most_recently_picked_entries_stay_however_long_unpicked(tmp_path, capsys):
    cache = tmp_path / "cache"
    homes = [entry(cache, i, (10 + i) * IDLE) for i in range(at_main.KEEP + 2)]
    for n in (at_main.KEEP + 1, at_main.KEEP, at_main.KEEP):
        at_main.prune(cache, None, 1.0)
        assert entries(cache) == homes[:n]
    assert capsys.readouterr().err == ""


def test_the_entry_a_live_process_runs_from_stays_however_long_unpicked_and_goes_once_it_exits(tmp_path):
    """`start` replaces the command with the one it runs from the tree, which keeps the hold `tree` took: a child started
    with only the inheritable descriptors stands in for it."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    home = at_main.tree(repo, git(repo, "rev-parse", "origin/main"), cache).parent
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    os.utime(home / "lock", (NOW - 10 * IDLE, NOW - 10 * IDLE))
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE, close_fds=False)
    exited()
    assert at_main.due(cache, None) == [home]
    at_main.prune(cache, None, 1.0)
    assert whole(home), "the process it started still runs from it"
    child.stdin.close()
    assert child.wait() == 0
    at_main.prune(cache, home, 1.0)
    assert whole(home), "the command's own entry"
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer


def test_an_entry_being_made_stays_however_long_unpicked_and_one_abandoned_half_made_goes(tmp_path, capsys):
    """A tree is half-made while its maker holds the entry's lock, with no stamp yet; once that lock is free, a stampless
    tree is one a killed git left, which the next command for it would remove and make again anyway."""
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    making = entry(cache, 100, 10 * IDLE, whole=False)
    with open(making / "lock") as maker:
        fcntl.flock(maker, fcntl.LOCK_EX)
        at_main.prune(cache, None, 1.0)
        assert (making / "tree/tools/f").is_file() and (making / "lock").is_file()
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""


def test_two_pruners_at_once_remove_different_entries_and_leave_every_other_whole(tmp_path, monkeypatch, capsys):
    cache, inner = tmp_path / "cache", []
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    olds = [entry(cache, 100 + i, (10 + i) * IDLE) for i in range(3)]

    def another_pruner_meanwhile(argv, timeout, **kw):
        if argv[0] == "rm" and not inner:
            inner.append(argv[-1])
            at_main.prune(cache, None, 1.0)
        return subprocess.run(argv, **kw)

    monkeypatch.setattr(at_main, "run", another_pruner_meanwhile)
    at_main.prune(cache, None, 1.0)
    assert inner == [str(olds[2] / "tree")], "the first takes the longest unpicked"
    assert entries(cache) == sorted([*newer, olds[0]]), "the other, finding it locked, takes the next"
    assert all(whole(h) for h in entries(cache)) and capsys.readouterr().err == ""


def test_an_entry_that_vanishes_before_or_during_its_removal_is_no_failure(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    gone = entry(cache, 100, 10 * IDLE)
    shutil.rmtree(gone)
    assert at_main.remove(gone, 1.0) is False
    vanishing = entry(cache, 101, 10 * IDLE)

    def removed_by_another_meanwhile(argv, timeout, **kw):
        if argv[0] == "rm":
            shutil.rmtree(vanishing)
        return subprocess.run(argv, **kw)

    monkeypatch.setattr(at_main, "run", removed_by_another_meanwhile)
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""


def test_removals_that_fail_are_one_line_on_stderr_and_the_command_runs_origin_mains(tmp_path, monkeypatch, capsys):
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    olds = [entry(cache, i, (10 + i) * IDLE) for i in range(at_main.KEEP + 1)]

    def not_its_files(argv, timeout, **kw):
        if argv[0] == "rm" and Path(argv[-1]).parent in olds:
            return subprocess.CompletedProcess(argv, 1, "", f"rm: cannot remove '{argv[-1]}/tools/f': Permission denied\n")
        return subprocess.run(argv, **kw)

    monkeypatch.setattr(at_main, "run", not_its_files)
    ran, err = run_shim(repo, cache, capsys)
    assert len(ran) == 2 and (Path(ran[0]) / "research/cli.py").read_text() == "new\n", "origin/main's runs"
    lines = err.splitlines()
    assert len(lines) == 2 and "behind origin/main" in lines[0]
    worst = olds[-1] / "tree"
    assert lines[1] == (f"research: 2 old tree(s) of origin/main's research under {cache} could not be removed "
                        f"(`rm -rf {worst}` exited 1: rm: cannot remove '{worst}/tools/f': Permission denied)")
    assert all((h / "tree/tools/f").is_file() for h in olds)


def registrations(repo: Path) -> list[str]:
    """What `repo` has registered as its worktrees, by name (git's `.git/worktrees`)."""
    d = repo / ".git/worktrees"
    return sorted(os.listdir(d)) if d.exists() else []


def idle_and_past_the_newest(cache: Path, home: Path) -> list[Path]:
    """The command that picked `home` has exited, long ago, and `KEEP` entries were picked since: those."""
    exited()
    os.utime(home / "lock", (NOW - 10 * IDLE, NOW - 10 * IDLE))
    return [entry(cache, i, 1) for i in range(at_main.KEEP)]


def test_an_entry_is_removed_together_with_what_its_repository_registered_for_its_tree(tmp_path, capsys):
    """`rm -rf` of the tree alone left the repository listing a worktree that is gone, which git prunes months later: a
    shipped source tree read as one with a worktree in use, and was kept on its node for good."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "another"), "main")
    home = at_main.tree(repo, git(repo, "rev-parse", "origin/main"), cache).parent
    assert registrations(repo) == ["another", "tree"]
    newer = idle_and_past_the_newest(cache, home)
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""
    assert registrations(repo) == ["another"], "its own registration, and no other"
    listed = git(repo, "worktree", "list", "--porcelain")
    assert listed.count("worktree ") == 2 and "prunable" not in listed, "the repository and the other worktree, both there"
    assert git(tmp_path / "another", "status", "--porcelain") == ""


def test_a_tree_removed_to_be_made_again_leaves_no_registration_when_it_then_cant_be_made(tmp_path, monkeypatch, capsys):
    """Made again, a tree takes the place of its old registration; when `git worktree add` fails or is cut short, nothing
    does, and the old one would stay with no tree."""
    repo, cache, wt = made(tmp_path)
    (wt.parent / "at").unlink()                                         # as a tree a killed checkout left: not whole
    monkeypatch.setattr(at_main, "run", at_worktree_add(still_running))
    ran, err = run_shim(repo, cache, capsys)
    assert ran == [own(repo)] and "(its 60 s ran out at `git worktree add`)" in err
    assert not wt.exists() and registrations(repo) == []
    monkeypatch.setattr(at_main, "run", to_its_end)
    assert run_shim(repo, cache, capsys)[0] == [str(wt / "tools/research/src"), own(repo)] and registrations(repo) == ["tree"]


def test_only_a_registration_that_names_the_tree_back_is_removed_with_it(tmp_path, capsys):
    """A tree's `.git` says where its repository registered it; what is found there is removed only when it is a
    registration (a directory of some `worktrees`) and that tree's (its `gitdir` names the tree's `.git`), so a `.git`
    that names another worktree's registration, a directory elsewhere, or a repository costs that nothing."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "another"), "main")
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    olds = [entry(cache, 100 + i, (10 + i) * IDLE) for i in range(3)]
    elsewhere = tmp_path / "elsewhere/plain"
    elsewhere.mkdir(parents=True)
    (elsewhere / "gitdir").write_text(f"{olds[1] / 'tree/.git'}\n")
    (olds[0] / "tree/.git").write_text(f"gitdir: {repo / '.git/worktrees/another'}\n")
    (olds[1] / "tree/.git").write_text(f"gitdir: {elsewhere}\n")
    (olds[2] / "tree/.git").write_text(f"gitdir: {repo / '.git'}\n")
    for _ in olds:
        at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""
    assert registrations(repo) == ["another"] and git(tmp_path / "another", "status", "--porcelain") == ""
    assert (elsewhere / "gitdir").exists() and git(repo, "status", "--porcelain") == ""


def test_a_pick_renews_its_entry_so_a_pruner_that_listed_it_as_due_leaves_it(tmp_path, monkeypatch):
    """The race: one command picks an entry (the commit it was told to run from) while another has judged it old. The
    judgement that counts is the one made under the entry's lock, where the pick is dated and held."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    main = git(repo, "rev-parse", "origin/main")
    wt = at_main.tree(repo, main, cache)
    home = wt.parent
    exited()
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    os.utime(home / "lock", (NOW - 2 * IDLE, NOW - 2 * IDLE))
    assert at_main.due(cache, None) == [home], "the pruner lists it"
    assert at_main.tree(repo, main, cache) == wt, "the other command picks it"
    assert (home / "lock").stat().st_mtime == NOW, "and the pick is dated"
    assert at_main.remove(home, 1.0) is False and whole(home), "judged under the lock, it was picked just now"
    exited()
    assert at_main.remove(home, 1.0) is False and whole(home), "and still within IDLE_S once that command exited"
    monkeypatch.setattr(at_main, "wall", lambda: NOW + IDLE + 1)
    assert at_main.remove(home, 1.0) is True and entries(cache) == newer


def test_a_command_waiting_for_the_lock_of_an_entry_a_pruner_removes_makes_it_again_under_the_new_lock(
        tmp_path, monkeypatch, capsys):
    """A lock taken on a lock file the pruner had removed guards nothing: a third command creates the file again and takes
    it, and two commands make one tree at once."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    main = git(repo, "rev-parse", "origin/main")
    wt = at_main.tree(repo, main, cache)
    home = wt.parent
    exited()
    (wt / "kept").write_text("no commit has this file\n")
    pruner = open(home / "lock")
    fcntl.flock(pruner, fcntl.LOCK_EX)

    def the_pruner_finishes(s):
        shutil.rmtree(home)
        pruner.close()

    third = []

    def a_third_command_tries_the_lock(argv, timeout, **kw):
        if argv[3:5] == ["worktree", "add"]:
            home.mkdir(parents=True, exist_ok=True)
            with open(home / "lock", "a") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    third.append("took it")
                except BlockingIOError:
                    third.append("waits")
        return subprocess.run(argv, **kw)

    monkeypatch.setattr(at_main, "sleep", the_pruner_finishes)
    monkeypatch.setattr(at_main, "run", a_third_command_tries_the_lock)
    ran, err = run_shim(repo, cache, capsys)
    assert third == ["waits"], "the command making it holds the lock at its path"
    assert ran[0] == str(wt / "tools/research/src") and "behind origin/main" in err and err.count("\n") == 1
    assert whole(home) and not (wt / "kept").exists() and (wt / "tools/research/src/research/cli.py").read_text() == "new\n"
    assert (home / "at").read_text().split() == [main]


def process(pid: int, cwd: str = "/", exe: str = "/usr/bin/python3", cmdline=("python3",), environ=()) -> Path:
    """A live process of this user as `PROC` shows one: its two links, and its command line and environment."""
    p = at_main.PROC / str(pid)
    p.mkdir()
    os.symlink(cwd, p / "cwd")
    os.symlink(exe, p / "exe")
    (p / "cmdline").write_bytes(b"".join(os.fsencode(w) + b"\0" for w in cmdline))
    (p / "environ").write_bytes(b"".join(os.fsencode(w) + b"\0" for w in environ))
    return p


NAMES = {
    "its working directory": lambda t: {"cwd": f"{t}/tools/research"},
    "its executable": lambda t: {"exe": f"{t}/bin/python3"},
    "a word of its command line": lambda t: {"cmdline": ["python3", "-m", "pytest", f"--rootdir={t}/tools/research"]},
    "a path in its environment": lambda t: {"environ": ["HOME=/home/u", f"PYTHONPATH={t}/tools/research/src:/elsewhere"]},
}


@pytest.mark.parametrize("how", list(NAMES))
def test_an_entry_a_live_process_names_stays_however_long_unpicked_and_goes_once_it_exits(tmp_path, capsys, how):
    """A command started by a copy of this module from before `inuse` holds nothing, and runs for as long as it runs: a
    queue loop or a watch outlives `IDLE_S`, and fails on its next lazy import once its tree is gone. Such a copy's `start`
    names the tree in PYTHONPATH, the environment's case."""
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    old = entry(cache, 100, 10 * IDLE)
    running = process(4242, **NAMES[how](old / "tree"))
    at_main.prune(cache, None, 1.0)
    assert whole(old) and (old / "tree/tools/f").is_file()
    shutil.rmtree(running)
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""


def test_a_tree_a_process_names_by_the_caches_real_path_stays(tmp_path):
    """A process's working directory and executable read back resolved, so a cache reached through a link is matched by
    its real path too."""
    (tmp_path / "real").mkdir()
    (tmp_path / "cache").symlink_to(tmp_path / "real")
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    old = entry(cache, 100, 10 * IDLE)
    process(4242, cwd=os.path.realpath(old / "tree/tools"))
    at_main.prune(cache, None, 1.0)
    assert whole(old) and entries(cache) == sorted([*newer, old])


def test_an_entry_no_process_names_goes_and_a_process_that_cant_be_read_names_nothing(tmp_path, capsys):
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    old = entry(cache, 100, 10 * IDLE)
    tree = old / "tree"
    process(1, cwd=f"{tree}-other", exe=f"{old}/python3", cmdline=["python3", f"{tree}house/x"],
            environ=[f"PYTHONPATH={old}:/elsewhere", f"OLD={tree}s"])
    unreadable = at_main.PROC / "2"
    for name in ("cwd", "exe", "cmdline", "environ"):
        (unreadable / name).mkdir(parents=True)                         # each read fails, as another user's would
    (at_main.PROC / "3").mkdir()                                        # a process that exited while it was listed
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == newer and capsys.readouterr().err == ""


def emptied(cache: Path, n: int, idle_s: float) -> Path:
    """An entry that holds nothing, its directory last changed ``idle_s`` before NOW: what a pruner that died between
    removing the lock file and the directory leaves."""
    home = cache / f"{n:012x}"
    home.mkdir(parents=True)
    os.utime(home, (NOW - idle_s, NOW - idle_s))
    return home


def test_an_empty_entry_is_removed_once_idle_s_has_passed_since_it_changed_and_holds_no_place(tmp_path, capsys):
    """An entry with no lock file counted as picked now, so one a pruner left empty stayed for good, in a `KEEP` place."""
    cache = tmp_path / "cache"
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    left = emptied(cache, 100, 2 * IDLE)
    making = emptied(cache, 101, 1)                                     # a command's `mkdir`, just before its lock file
    assert at_main.due(cache, None) == [left]
    at_main.prune(cache, None, 1.0)
    assert entries(cache) == sorted([*newer, making]) and capsys.readouterr().err == ""


def test_a_command_that_comes_for_an_empty_entry_as_it_is_removed_makes_it_again(tmp_path, monkeypatch, capsys):
    """The empty entry has no lock to judge it under: a command's `mkdir` may come just before the `rmdir`, whose
    command then finds no directory to open its lock file in and makes it again."""
    repo, cache = checkout(tmp_path), tmp_path / "cache"
    main = git(repo, "rev-parse", "origin/main")
    home = at_main.tree(repo, main, cache).parent
    exited()
    shutil.rmtree(home)
    emptied(cache, int(home.name, 16), 2 * IDLE)
    newer = [entry(cache, i, 1) for i in range(at_main.KEEP)]
    real, seen = Path.mkdir, []

    def another_pruner_after_it(self, *a, **kw):
        real(self, *a, **kw)
        if self == home and not seen:
            at_main.prune(cache, None, 1.0)
            seen.append(home.exists())

    monkeypatch.setattr(Path, "mkdir", another_pruner_after_it)
    ran, err = run_shim(repo, cache, capsys)
    assert seen == [False], "the other pruner removed it between this command's mkdir and its open"
    assert ran[0] == str(home / "tree/tools/research/src") and err.count("\n") == 1
    assert whole(home) and entries(cache) == sorted([*newer, home])


def test_an_empty_entry_a_command_has_opened_its_lock_in_stays(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    for i in range(at_main.KEEP):
        entry(cache, i, 1)
    left = emptied(cache, 100, 2 * IDLE)
    assert at_main.due(cache, None) == [left]
    real, command = at_main.empty, []

    def a_command_opens_its_lock_meanwhile(home):
        found = real(home)
        command.append(at_main.lock(home, 60.0))
        return found

    monkeypatch.setattr(at_main, "empty", a_command_opens_its_lock_meanwhile)
    assert at_main.remove(left, 1.0) is False
    assert at_main.same(command[0], left / "lock"), "the command holds the lock at its path"
    command[0].close()


def test_a_research_a_test_starts_never_loads_at_main():
    """Other packages' suites start `python -m research` without declaring at_main.py, so loading it failed their read
    guard; under a test the tree's code runs anyway."""
    probe = ("import sys; sys.argv = ['research', '--help']; from research import cli; cli.main(); "
             "print('at_main' if 'research.at_main' in sys.modules else 'clean')")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[-1] == "clean"
