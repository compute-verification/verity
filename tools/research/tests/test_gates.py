"""`research.gates`: the environment every gate's subprocess gets and the one refusal path."""
from __future__ import annotations

import subprocess

from research import gates


def test_git_local_env_names_every_variable_git_says_picks_another_repository():
    """A git that grows a repository-local variable must not leak it into a gate: each one `git rev-parse --local-env-vars`
    names is dropped, or kept on purpose (`GIT_KEPT_ENV`)."""
    named = set(subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, check=True).stdout.split())
    assert named and not (named - set(gates.GIT_LOCAL_ENV) - set(gates.GIT_KEPT_ENV))
    assert not set(gates.GIT_LOCAL_ENV) & set(gates.GIT_KEPT_ENV)


def test_git_env_drops_a_hooks_repository_and_the_callers_slots_and_import_path(monkeypatch):
    leaked = {"GIT_DIR": "/elsewhere/.git", "GIT_WORK_TREE": "/elsewhere", "GIT_INDEX_FILE": "/tmp/scratch-index",
              "LEAN_SLOT": "2", "LEAN_SLOT_POOL": "/pool", "CHECK_SLOT": "a", "PYTHONPATH": "/caller/src"}
    kept = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_GLOBAL": "/cfg", "HOME": "/home/x", "RESEARCH_STORE": "/store"}
    for k, v in {**leaked, **kept}.items():
        monkeypatch.setenv(k, v)
    env = gates.git_env(GIT_TERMINAL_PROMPT="0", HOME="/home/y")
    assert not set(leaked) & set(env)
    assert env == {**env, **kept, "GIT_TERMINAL_PROMPT": "0", "HOME": "/home/y"}


def test_refuse_says_why_on_stderr_and_exits_refused(capsys):
    assert gates.refuse("research merge", "no passing `check` attempt") == gates.REFUSED == 3
    assert capsys.readouterr() == ("", "research merge: refused: no passing `check` attempt\n")
