"""A toy checkout for the graph's tests: a git repository holding a `tasks.toml`, runner stubs and a few files, and a lister
over it.  `toy_tree(tmp_path)` builds one; a family's own tests write their table and files into it the same way."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from verity_graph import graph

SPEC = """
[slot]
cores = 4
mem_gb = 8

[env.py]
spec = ["uv.lock", "pyproject.toml"]

[env.lean]
spec = ["**/lean-toolchain"]

[family.toy]
files = ["src/*.txt"]
kinds = ["toy", "toy-b"]
env = "py"
about = "toy {name}"
resources = { cores = 1, mem_gb = 1, disk_gb = 1 }
timeout_s = 60

[family.other]
files = ["lib/**"]
kinds = ["other"]
env = "lean"
about = "other {name} with {flavor}"
resources = { cores = 2, mem_gb = 2, disk_gb = 2 }
timeout_s = 30
"""
FILES = {"src/a.txt": "a\n", "src/b.txt": "b\n", "lib/x.lean": "def x := 1\n", "lib/lean-toolchain": "v4\n",
         "uv.lock": "version = 1\n", "pyproject.toml": "[project]\n", "README": "toy\n",
         f"{graph.RUNNERS}/__init__.py": "", f"{graph.RUNNERS}/toy.py": "", f"{graph.RUNNERS}/toy_b.py": "",
         f"{graph.RUNNERS}/other.py": ""}


def git(root: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=toy", "-c", "user.email=toy@example.invalid",
                           "-c", "commit.gpgsign=false", *argv], capture_output=True, text=True, check=True).stdout.strip()


def toy_tree(root: Path, spec: str = SPEC, files: dict[str, str] | None = None) -> Path:
    """A committed checkout at ``root`` with ``spec`` as its tasks.toml and ``files``."""
    for p, text in {**FILES, **(files or {}), graph.SPEC: spec}.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(text)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "toy")
    return root


def toy(root: Path, files: list[str]) -> list[dict]:
    return [{"id": "toy:a", "inputs": ["src/a.txt"], "outputs": [{"name": "o", "path": "out/a"}]},
            {"id": "toy-b:b", "kind": "toy-b", "inputs": ["src/b.txt"], "reads": [{"task": "toy:a", "output": "o"}],
             "outputs": [{"name": "o", "path": "out/b"}]}]


def other(root: Path, files: list[str]) -> list[dict]:
    return [{"id": "other:x", "inputs": ["lib/x.lean"], "vars": {"flavor": "plain"}}]


LISTERS = {"toy": toy, "other": other}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    return toy_tree(tmp_path / "tree")
