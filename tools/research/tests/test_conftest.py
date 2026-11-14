"""What conftest.py keeps every test from reading off the machine it runs on."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[1]


def test_git_reads_only_the_tests_own_config_and_commits_unsigned(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    scoped = [line.split("\t", 1) for line in git("config", "--list", "--show-scope").splitlines()]
    assert {scope for scope, _ in scoped} <= {"global", "local"}
    assert sorted(kv for scope, kv in scoped if scope == "global") == [
        "commit.gpgsign=false", "tag.gpgsign=false", "user.email=t@t", "user.name=t"]
    git("commit", "-q", "--allow-empty", "-m", "one")
    assert "gpgsig" not in git("cat-file", "commit", "HEAD")


def test_the_tests_and_the_pythons_they_start_import_this_trees_research():
    """The research the tests import, and the one a Python they start imports, is this tree's, with store.pod.toml beside it.
    The next test runs this one with another `research` first on PYTHONPATH."""
    from research.pods import registry

    assert Path(registry.__file__).resolve() == TREE / "src" / "research" / "pods" / "registry.py"
    assert (Path(registry.__file__).resolve().parents[2].parent / "store.pod.toml").is_file()
    child = subprocess.run([sys.executable, "-c", "import research; print(research.__file__)"], capture_output=True, text=True,
                           check=True, cwd="/")
    assert Path(child.stdout.strip()).resolve() == TREE / "src" / "research" / "__init__.py"


def test_a_research_first_on_pythonpath_is_not_what_the_tests_check(tmp_path, monkeypatch):
    """A `research run` job puts its tool snapshot, a `research` with no store.pod.toml beside it, first on PYTHONPATH: a test
    run there with pytest alone must still check this tree, and read the files beside its code."""
    decoy = tmp_path / "decoy"
    (decoy / "research" / "pods").mkdir(parents=True)
    for f in ("__init__.py", "pods/__init__.py", "pods/registry.py"):
        (decoy / "research" / f).write_text("")
    monkeypatch.setenv("PYTHONPATH", str(decoy), prepend=os.pathsep)
    seen = subprocess.run([sys.executable, "-c", "import research; print(research.__file__)"], capture_output=True, text=True,
                          check=True, cwd="/")
    assert Path(seen.stdout.strip()) == decoy / "research" / "__init__.py", "the decoy is what a Python on this PYTHONPATH imports"
    probe = f"{Path(__file__).name}::test_the_tests_and_the_pythons_they_start_import_this_trees_research"
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'probe'}",
                          f"tests/{probe}"], capture_output=True, text=True, cwd=TREE)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
