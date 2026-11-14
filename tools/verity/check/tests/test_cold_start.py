"""Each entry point a run starts, started cold: in a tree shipped as `research run` ships one (`remote._git_ship_script`: a
`clone --shared` of the machine's bare repository detached at the commit, READY.json beside it and out of git's view), with no
`.venv`, under the runner's environment (`remote._runner_env`: the tool snapshot, then the tree's import roots and pyshim, on
PYTHONPATH), and run until its first real step. A unit test of an entry point runs it from this checkout's warm `.venv`, so
one that breaks only on a fresh tree (a workspace member the sync misses, an import the runner's PYTHONPATH shadows, a step that
needs a package `uv run` alone doesn't install) passes them all and fails every run. Each test here syncs a fresh environment,
which is why they are slow. The Lean check, which runs `lake`, has its smoke in its own suite (tools/verity/lean)."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from research import remote

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
UV = shutil.which("uv")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(UV is None, reason="no uv")]
#: what a run's environment doesn't inherit from this test's: credentials, a recording run's and `check`'s variables, and
#: this suite's own environment and import path
SCRUBBED = ("AWS_", "R2_", "RESEARCH_", "CHECK_", "VERITY_", "CLOUD_AGENT_", "SLACK_", "GH_", "GITHUB_", "PYTEST_")
DROPPED = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT")
FIRST_STEP = r"^check: preflight-lock: passed"


class Names(dict):
    """An environment whose repr shows none of its values: a failing test's traceback prints the test's arguments."""

    def __repr__(self) -> str:
        return f"<environment of {len(self)} names>"


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def machine(tmp_path_factory) -> tuple[Path, str]:
    """(the machine's bare repository, the commit it ships): this checkout's working tree as one commit, its `main`, with the
    objects borrowed from this checkout's."""
    m = tmp_path_factory.mktemp("machine")
    env = Names({**os.environ, "GIT_INDEX_FILE": str(m / "index")})
    _git(ROOT, "read-tree", "HEAD", env=env)
    _git(ROOT, "add", "-A", env=env)
    tree, head = _git(ROOT, "write-tree", env=env), _git(ROOT, "rev-parse", "HEAD")
    sha = head if tree == _git(ROOT, "rev-parse", "HEAD^{tree}") else _git(
        ROOT, "-c", "user.name=cold-start", "-c", "user.email=cold-start@verity", "commit-tree", tree, "-p", head, "-m", "cold start")
    bare = m / "git" / "verity.git"
    _git(m, "init", "-q", "--bare", str(bare))
    objects = Path(_git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "objects"
    (bare / "objects" / "info" / "alternates").write_text(f"{objects}\n")
    _git(bare, "update-ref", "refs/heads/main", sha)
    return bare, sha


@pytest.fixture
def shipped(machine, tmp_path, monkeypatch):
    """(the shipped tree, its runner environment), removed after the test: its `.venv` is a whole workspace's."""
    bare, sha = machine
    tree = tmp_path / "root" / "src" / sha
    _git(tmp_path, "clone", "-q", "--shared", "--no-checkout", str(bare), str(tree))
    _git(tree, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha)
    (tree / ".git" / "info").mkdir(exist_ok=True)
    with (tree / ".git" / "info" / "exclude").open("a") as f:
        f.write(f"/{remote.READY_FILE}\n")
    (tree / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": sha, "checkout": True}) + "\n")
    data, digest = remote.tool_tarball(tree / "tools" / "research" / "src" / "research")
    tool = tmp_path / "root" / "tool" / digest
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(tool, filter="data")
    uv_cache = subprocess.run([UV, "cache", "dir"], capture_output=True, text=True, check=True).stdout.strip()
    venv = Path(sys.prefix).resolve()
    path = [p for p in os.get_exec_path() if p and not Path(p).resolve().is_relative_to(venv)]
    for k in list(os.environ):
        if k.startswith(SCRUBBED) or k in DROPPED:
            monkeypatch.delenv(k)
    monkeypatch.setenv("PATH", os.pathsep.join(path))
    env = Names(remote._runner_env({"tool": str(tool)}, tmp_path / "root" / "runs", str(tree), {"source_sha": sha}))
    (tmp_path / "tmp").mkdir()
    env.update(UV_CACHE_DIR=uv_cache, XDG_CACHE_HOME=str(tmp_path / "cache"), TMPDIR=str(tmp_path / "tmp"))
    yield tree, env
    shutil.rmtree(tmp_path / "root", ignore_errors=True)


def _until(argv: list[str], cwd: Path, env: dict[str, str], line: str) -> tuple[bool, str]:
    """Run ``argv`` in a session of its own, reading its output until a line matches ``line``, then stop the session: (whether
    one did, the output)."""
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    out, hit = [], False
    for ln in p.stdout:
        out.append(ln)
        if not hit and re.search(line, ln):
            hit = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGTERM)
    p.wait()
    return hit, "".join(out)


def _tail(out: str) -> str:
    return "\n".join(out.splitlines()[-30:])


def test_suites_py_syncs_a_fresh_tree_and_runs_a_suite(shipped):
    """`uv run tools/verity/check/suites.py` in a fresh tree, as `research run` workloads start it: it finds `uv run`'s
    environment missing the workspace, syncs it and runs the suite."""
    tree, env = shipped
    out = env["TMPDIR"] + "/outcome.json"
    p = subprocess.run([UV, "run", "tools/verity/check/suites.py", "--fresh", "--json", out, "tools/config"], cwd=tree, env=env,
                       capture_output=True, text=True)
    assert "holds none of the workspace yet" in p.stderr, _tail(p.stdout + p.stderr)
    assert p.returncode == 0, _tail(p.stdout + p.stderr)
    [r] = json.loads(Path(out).read_text())
    assert r["status"] == "passed" and r["counts"].get("passed"), r


def test_check_py_reaches_its_first_step_from_a_fresh_tree(shipped):
    """`uv run tools/verity/check/check.py`, the check `--record` launches, in a fresh tree: it syncs the workspace and passes
    its first step."""
    tree, env = shipped
    hit, out = _until([UV, "run", "tools/verity/check/check.py"], tree, env, FIRST_STEP)
    assert hit, _tail(out)
    assert "holds none of the workspace yet" in out, _tail(out)


def test_the_verity_command_reaches_checks_first_step_from_a_fresh_tree(shipped):
    """`uv run --extra torch-cpu verity check`, AGENTS.md's whole check, in a fresh tree: the CLI installs, finds check.py and
    passes its first step."""
    tree, env = shipped
    hit, out = _until([UV, "run", "--extra", "torch-cpu", "verity", "check"], tree, env, FIRST_STEP)
    assert hit, _tail(out)


def test_circuit_check_checks_a_target_from_a_fresh_tree(shipped):
    """`uv run circuit-check TARGET` in a fresh tree: the command installs and checks the target."""
    tree, env = shipped
    p = subprocess.run([UV, "run", "circuit-check", "F32Add_v1", "--no-cache"], cwd=tree, env=env, capture_output=True, text=True)
    assert p.returncode == 0, _tail(p.stdout + p.stderr)
    assert re.search(r"^circuit-check: 1 target\(s\), 0 failure\(s\)", p.stdout, re.M), _tail(p.stdout + p.stderr)


STAND_IN = '''import os, sys
__path__.append(os.path.join(os.getcwd(), "tools", "research", "src", "research"))
'''
LAUNCH = '''import json, os, sys
with open(os.environ["COLD_START_LAUNCH"], "w") as f:
    json.dump(sys.argv[1:], f)
print("research: run r20991231-000000-c01d launched on vy-cold-start", flush=True)
'''


def test_train_sh_launches_the_check_from_a_fresh_worktree(shipped, tmp_path):
    """`tools/verity/check/train.sh COMMIT MACHINE`, the lander's launch, from a checkout of the queue: it makes the commit's
    worktree, whose `uv run -q python check.py --record` starts in a fresh environment, and prints the run that launch names.
    `research run` itself is a stand-in that records its argv (the package's other modules are the tree's), so nothing reaches a
    machine or the store; the store key is a dummy that train.sh's own refusal checks for and nothing signs with."""
    tree, env = shipped
    standin = tmp_path / "standin" / "research"
    standin.mkdir(parents=True)
    (standin / "__init__.py").write_text(STAND_IN)
    (standin / "__main__.py").write_text(LAUNCH)
    launch = tmp_path / "launch.json"
    env.update(PYTHONPATH=f"{standin.parent}{os.pathsep}{env['PYTHONPATH']}", COLD_START_LAUNCH=str(launch),
               TRAIN_TREES=str(tmp_path / "root" / "trains"), AWS_ACCESS_KEY_ID="cold-start", R2_ENDPOINT="https://r2.invalid",
               RESEARCH_STORE=str(tmp_path / "store"), RESEARCH_STORE_CONFIG=str(tmp_path / "store.toml"))
    sha = tree.name
    p = subprocess.run(["bash", "tools/verity/check/train.sh", sha, "vy-cold-start"], cwd=tree, env=env, capture_output=True, text=True)
    log = tmp_path / "root" / "trains" / f"train-{sha[:12]}.log"
    shown = _tail(p.stdout + p.stderr + (log.read_text() if log.is_file() else ""))
    assert p.returncode == 0, shown
    assert "RUN r20991231-000000-c01d" in p.stdout, shown
    argv = json.loads(launch.read_text())
    assert argv[:3] == ["run", "--tool", "check"] and argv[argv.index("--on") + 1] == "vy-cold-start", argv
    assert (tmp_path / "root" / "trains" / f"train-{sha[:12]}" / ".venv").is_dir(), shown
