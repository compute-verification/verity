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
    kept = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "credential.helper", "GIT_CONFIG_VALUE_0": "!gh auth git-credential",
            "GIT_CONFIG_GLOBAL": "/cfg", "HOME": "/home/x", "RESEARCH_STORE": "/store"}
    for k, v in {**leaked, **kept}.items():
        monkeypatch.setenv(k, v)
    env = gates.git_env(GIT_TERMINAL_PROMPT="0", HOME="/home/y")
    assert not set(leaked) & set(env)
    driver = {"GIT_CONFIG_KEY_1": "merge.lean-audit.name", "GIT_CONFIG_KEY_2": "merge.lean-audit.driver",
              "GIT_CONFIG_KEY_3": "merge.lean-imports.name", "GIT_CONFIG_KEY_4": "merge.lean-imports.driver",
              "GIT_CONFIG_KEY_5": "merge.fixture-map.name", "GIT_CONFIG_KEY_6": "merge.fixture-map.driver"}
    assert env == {**env, **kept, **driver, "GIT_CONFIG_COUNT": "7", "GIT_TERMINAL_PROMPT": "0", "HOME": "/home/y"}


def test_a_gates_merge_runs_the_record_driver_in_a_clone_without_it(tmp_path):
    """A clone's config holds `merge.lean-audit.driver` only after `merge.py --install`, so the queue's own clone and a fresh VM's
    checkout merged two sides' `lean-audit.json` line by line: a conflict where the driver keeps both."""
    def git(*a):
        return subprocess.run(["git", "-c", "commit.gpgsign=false", *a], cwd=tmp_path, check=True, capture_output=True, text=True,
                              env=gates.git_env(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t",
                                                GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"))

    driver = tmp_path / "tools/verity/lean/merge.py"
    driver.parent.mkdir(parents=True)
    driver.write_text("import json, sys\no, a, b = (json.load(open(p)) for p in sys.argv[1:4])\n"
                      "json.dump({**a, **b}, open(sys.argv[2], 'w'), indent=1, sort_keys=True)\n")
    (tmp_path / ".gitattributes").write_text("lean-audit.json merge=lean-audit\n")
    (tmp_path / "lean-audit.json").write_text('{\n "m": 1\n}\n')
    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "o")
    git("checkout", "-q", "-b", "b")
    (tmp_path / "lean-audit.json").write_text('{\n "b": 1,\n "m": 1\n}\n')
    git("commit", "-q", "-am", "b")
    git("checkout", "-q", "main")
    (tmp_path / "lean-audit.json").write_text('{\n "a": 1,\n "m": 1\n}\n')
    git("commit", "-q", "-am", "a")
    assert subprocess.run(["git", "config", "merge.lean-audit.driver"], cwd=tmp_path, capture_output=True).returncode == 1
    git("merge", "-q", "--no-edit", "b")
    assert (tmp_path / "lean-audit.json").read_text() == '{\n "a": 1,\n "b": 1,\n "m": 1\n}'


def test_refuse_says_why_on_stderr_and_exits_refused(capsys):
    assert gates.refuse("research merge", "no passing `check` attempt") == gates.REFUSED == 3
    assert capsys.readouterr() == ("", "research merge: refused: no passing `check` attempt\n")
