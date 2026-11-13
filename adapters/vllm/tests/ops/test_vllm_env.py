"""verity_vllm/ops/vllm_env.sh: which builds it installs, and what it does with a directory that already exists.

The install itself (a wheel, torch) cannot run here.  What can: the script parses; with the pin it installs exactly
`engine.stock.BUILDS`; it refuses the pin, whose environment is pod_bootstrap.sh's, and a build it doesn't know; it
leaves a directory that holds the build as it is and prints its path last; and it refuses a directory that holds
anything else, or an install that did not finish, without touching it."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from verity_vllm.engine import stock

SCRIPT = Path(__file__).resolve().parents[2] / "verity_vllm" / "ops" / "vllm_env.sh"
BASH = shutil.which("bash") or "/bin/bash"
NEW = stock.BUILDS[1]


def run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(SCRIPT)], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "WORK": str(tmp_path / "work"), **env})


def venv(tmp_path: Path, commit: str | None, finished: bool = True) -> Path:
    """A directory that holds a vLLM of ``commit`` as an environment does (None: no vLLM): the file setuptools-scm writes
    into the package, which the script reads the build from; with the ``freeze.txt`` an install writes last when it
    ``finished``."""
    d = tmp_path / "venv"
    (d / "bin").mkdir(parents=True)
    if commit:
        pkg = d / "lib" / "python3.12" / "site-packages" / "vllm"
        pkg.mkdir(parents=True)
        (pkg / "_version.py").write_text(f"__version__ = version = '0.0'\n\n__commit_id__ = commit_id = 'g{commit}'\n")
    if finished:
        (d / "freeze.txt").write_text("vllm==0\n")
    return d


def test_script_parses() -> None:
    r = subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_it_has_a_case_for_every_build_of_the_tree() -> None:
    cases = re.findall(r"^  ([0-9a-f]{9})\)", SCRIPT.read_text(), re.M)
    assert sorted(cases) == sorted(stock.BUILDS), "a build of engine.stock.BUILDS has no case in vllm_env.sh, or one too many"
    assert f"BUILD=${{SERVE_VLLM:-{NEW}}}" in SCRIPT.read_text()


@pytest.mark.parametrize("build, says", [(stock.BUILDS[0], "pod_bootstrap.sh"), ("0123456789", "no build of engine.stock.BUILDS")])
def test_it_refuses_the_pin_and_a_build_it_does_not_know(tmp_path: Path, build: str, says: str) -> None:
    r = run(tmp_path, SERVE_VLLM=build)
    assert r.returncode == 3 and r.stderr.startswith("VLLM_ENV_FAIL_BUILD: ") and says in r.stderr, r.stderr
    assert not r.stdout and not (tmp_path / "work").exists()


def test_a_directory_that_holds_the_build_is_left_as_it_is(tmp_path: Path) -> None:
    d = venv(tmp_path, NEW)
    before = sorted(p.relative_to(d) for p in d.rglob("*"))
    r = run(tmp_path, VENV=str(d))
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-1] == str(d) and sorted(p.relative_to(d) for p in d.rglob("*")) == before


@pytest.mark.parametrize("commit, finished", [(stock.BUILDS[0], True), (None, True), (NEW, False)])
def test_a_directory_that_holds_anything_else_is_refused_and_kept(tmp_path: Path, commit: str | None, finished: bool) -> None:
    d = venv(tmp_path, commit, finished)
    before = sorted(p.relative_to(d) for p in d.rglob("*"))
    r = run(tmp_path, VENV=str(d))
    assert r.returncode == 3 and r.stderr.startswith("VLLM_ENV_FAIL_OTHER: ") and not r.stdout, (r.stdout, r.stderr)
    assert sorted(p.relative_to(d) for p in d.rglob("*")) == before
