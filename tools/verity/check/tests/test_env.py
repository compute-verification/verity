"""`check.env()`: the steps get an allowlisted environment, and never what `check` sets for its own steps from the environment
(#420: inside a real recorded check, its steps' CHECK_STORE_CUSTODY and VERITY_STORE_REQUIRED leaked into `env()` and into the
tests that check it)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("check_env_under_test", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)

#: what a check pod's runner environment can hold: what the steps need, and what they must never see
MACHINE = {"PATH": "/usr/bin", "HOME": "/root", "TMPDIR": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "XDG_CACHE_HOME": "/root/.cache",
           "UV_CACHE_DIR": "/root/.cache/uv", "VIRTUAL_ENV": "/w/src/abc/.venv", "CARGO_HOME": "/root/.cargo", "https_proxy": "http://p:3128",
           "VERITY_SLOW": "1", "FLOCK_CHECKOUT": "/root/flock", "AWS_ACCESS_KEY_ID": "dummy", "AWS_SESSION_TOKEN": "dummy",
           "R2_ENDPOINT": "https://r2.invalid", "RUNPOD_API_KEY": "dummy", "GITHUB_TOKEN": "dummy", "SSH_AUTH_SOCK": "/tmp/agent",
           "RESEARCH_STORE": "/w/store", "RESEARCH_RUN_ID": "r0", "PYTHONPYCACHEPREFIX": "/w/cache", "LD_LIBRARY_PATH": "/usr/local/cuda",
           "HOSTNAME": "pod", "SOME_TOOL_SETTING": "x"}


def test_the_steps_get_only_what_the_allowlist_names(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {**MACHINE, "RESEARCH_RUN_DIR": str(tmp_path / "runs" / "r0")})
    env = CHECK.env()
    assert set(env) == {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "XDG_CACHE_HOME", "UV_CACHE_DIR", "VIRTUAL_ENV", "CARGO_HOME",
                        "https_proxy", "VERITY_SLOW", "FLOCK_CHECKOUT", CHECK.STORE_REQUIRED_ENV, CHECK.REGRESSION_ENV, "GIT_AUTHOR_NAME",
                        "GIT_COMMITTER_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"}, sorted(env)
    assert {k: env[k] for k in ("PATH", "VERITY_SLOW", "FLOCK_CHECKOUT")} == {"PATH": "/usr/bin", "VERITY_SLOW": "1", "FLOCK_CHECKOUT": "/root/flock"}


def test_a_step_never_inherits_what_check_set_for_it(tmp_path, monkeypatch):
    """A step of a recorded check has the run's custody dir and the store required, and no RESEARCH_*: a check it starts (its
    own tests call `env()`) is not a recorded run, so it passes neither on.  A recorded run of its own passes its own."""
    step = {"PATH": "/usr/bin", "HOME": "/root", CHECK.STORE_REQUIRED_ENV: "1", CHECK.CUSTODY_ENV: "/w/requests/r0/custody",
            CHECK.REGRESSION_ENV: "1"}
    monkeypatch.setattr(os, "environ", dict(step))
    env = CHECK.env()
    assert CHECK.STORE_REQUIRED_ENV not in env and CHECK.CUSTODY_ENV not in env and CHECK.REGRESSION_ENV not in env
    run, custody = tmp_path / "research" / "runs" / "r1", tmp_path / "research" / "requests" / "r1" / "custody"
    run.mkdir(parents=True)
    custody.mkdir(parents=True)
    (custody / "store.toml").write_text("[remote]\n")
    (custody / "cred.json").write_text("{}")
    monkeypatch.setattr(os, "environ", {**step, "RESEARCH_RUN_ID": "r1", "RESEARCH_RUN_DIR": str(run)})
    env = CHECK.env()
    assert env[CHECK.STORE_REQUIRED_ENV] == "1" and env[CHECK.CUSTODY_ENV] == str(custody.resolve()) and env[CHECK.REGRESSION_ENV] == "1"


def test_checks_own_tests_pass_inside_a_recorded_checks_step():
    """#420's own test failed in train TB's recorded check: the step's variables reached it."""
    leaked = {**os.environ, CHECK.STORE_REQUIRED_ENV: "1", CHECK.CUSTODY_ENV: "/w/requests/r0/custody"}
    test = "tests/test_check.py::test_a_recorded_runs_steps_require_the_store_and_get_its_custody_dir"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], cwd=ROOT / "tools" / "verity" / "check", env=leaked,
                       capture_output=True, text=True)
    assert p.returncode == 0 and "1 passed" in p.stdout, p.stdout + p.stderr
