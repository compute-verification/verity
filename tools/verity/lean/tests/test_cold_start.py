"""The Lean check started cold: `python3 tools/verity/lean/check.py --build PKG` in a tree shipped as `research run` ships one
(`remote._git_ship_script`: a `clone --shared` detached at the commit, READY.json beside it and out of git's view), with no
`.lake` anywhere, under the runner's environment (`remote._runner_env`), until it audits the package. The other tests here run
this directory's check.py on a copy of a fixture, so one that breaks only in a shipped tree (a path it resolves from the tree, a
file it needs that a fresh checkout lacks, an import the runner's PYTHONPATH shadows) passes them and fails every run.
`tools/verity/check/tests/test_cold_start.py` holds the other entry points' smokes."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from research import remote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PKG = "tools/verity/lean/tests/fixtures/control"


class Names(dict):
    """An environment whose repr shows none of its values: a failing test's traceback prints its frames' arguments."""

    def __repr__(self) -> str:
        return f"<environment of {len(self)} names>"


def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def _shipped(tmp_path: Path) -> Path:
    """This checkout's working tree as one commit, checked out as a run's shipped tree, its objects borrowed from this checkout's."""
    env = Names({**os.environ, "GIT_INDEX_FILE": str(tmp_path / "index")})
    _git(ROOT, "read-tree", "HEAD", env=env)
    _git(ROOT, "add", "-A", env=env)
    tree, head = _git(ROOT, "write-tree", env=env), _git(ROOT, "rev-parse", "HEAD")
    sha = head if tree == _git(ROOT, "rev-parse", "HEAD^{tree}") else _git(
        ROOT, "-c", "user.name=cold-start", "-c", "user.email=cold-start@verity", "commit-tree", tree, "-p", head, "-m", "cold start")
    src = tmp_path / "root" / "src" / sha
    _git(tmp_path, "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(src))
    _git(src, "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha)
    with (src / ".git" / "info" / "exclude").open("a") as f:
        f.write(f"/{remote.READY_FILE}\n")
    (src / remote.READY_FILE).write_text(json.dumps({"schema": remote.READY_SCHEMA, "identity": sha, "checkout": True}) + "\n")
    return src


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_lean_check_builds_and_audits_a_package_in_a_fresh_tree(tmp_path, monkeypatch):
    """`--build` of the control package from a fresh shipped tree, to its `AUDIT` line. It can take over a minute (a whole
    tree's checkout and the package's first Lake build); what it catches that the fixture-copy tests here can't is a Lean
    check that works from this checkout and not from a tree a run ships: a path it resolves through the tree, `lean_slot.py`
    loaded from it, the runner's PYTHONPATH."""
    tree = _shipped(tmp_path)
    data, digest = remote.tool_tarball(tree / "tools" / "research" / "src" / "research")
    tool = tmp_path / "root" / "tool" / digest
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(tool, filter="data")
    venv = Path(sys.prefix).resolve()
    monkeypatch.setenv("PATH", os.pathsep.join(p for p in os.get_exec_path()
                                               if p and not Path(p).resolve().is_relative_to(venv)))
    for k in list(os.environ):
        if k.startswith(("AWS_", "R2_", "RESEARCH_", "VERITY_", "PYTEST_")) or k in ("VIRTUAL_ENV", "PYTHONPATH"):
            monkeypatch.delenv(k)
    env = Names(remote._runner_env({"tool": str(tool)}, tmp_path / "root" / "runs", str(tree), {"source_sha": tree.name}))
    # no node's slot policy (whether its slots list an audit pool, how long they're held) decides a one-package build here
    env.update(XDG_CACHE_HOME=str(tmp_path / "cache"), LEAN_SLOTS=str(tmp_path / "no-lean-slots"))
    p = subprocess.run(["python3", "tools/verity/lean/check.py", "--build", "--out", str(tmp_path / "out"), PKG], cwd=tree, env=env,
                       capture_output=True, text=True)
    shown = "\n".join((p.stdout + p.stderr).splitlines()[-30:])
    assert "Traceback" not in p.stderr, shown
    assert re.search(rf"^AUDIT \S*{PKG}: (PASS|FAIL) +\d+ declarations in \d+ modules", p.stdout, re.M), shown
