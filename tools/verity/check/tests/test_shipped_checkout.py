"""`research run --on` ships the tree as a git checkout, so `suites.py` keys a commit on a machine as it does in the launcher's
checkout, and `--explain` names a miss's files there from git's trees.  The machine is a local `sh -c`, as in `research`'s tests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from research import remote

HERE = Path(__file__).resolve().parents[1]
LOCAL = ["sh", "-c"]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                   capture_output=True, check=True)


def _repo(root: Path) -> Path:
    """A checkout with this runner and one package suite whose test passes."""
    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (root / "pkg" / "tests").mkdir(parents=True)
    (root / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (root / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "one")
    return root


def _suites(tree: Path, cache: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(VERITY_TEST_CACHE=str(cache), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(tree / "tools" / "verity" / "check" / "suites.py"), "--jobs", "1", *args], cwd=tree, env=env,
                          capture_output=True, text=True)


def _keys(tree: Path, cache: Path) -> dict[str, str]:
    out = cache / "outcome.json"
    p = _suites(tree, cache, "--fresh", "--json", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    return {r["name"]: r["key"] for r in json.loads(out.read_text())}


def test_a_shipped_checkout_keys_its_suites_as_the_launchers_checkout_does(tmp_path):
    repo = _repo(tmp_path / "repo")
    _, shipped, doc = remote.ship_source(LOCAL, str(tmp_path / "machine"), repo)
    assert doc["checkout"] is True
    assert _keys(Path(shipped), tmp_path / "pod-cache") == _keys(repo, tmp_path / "laptop-cache")


def test_explain_on_the_machine_names_the_files_a_miss_changed(tmp_path):
    repo, machine, cache = _repo(tmp_path / "repo"), tmp_path / "machine", tmp_path / "pod-cache"
    _, one, _ = remote.ship_source(LOCAL, str(machine), repo)
    assert _suites(Path(one), cache).returncode == 0
    (repo / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    _git(repo, "commit", "-qam", "two")
    _, two, _ = remote.ship_source(LOCAL, str(machine), repo)
    p = _suites(Path(two), cache, "--explain", "pkg")
    assert p.returncode == 0 and "input pkg changed" in p.stdout, p.stdout
    assert "pkg/tests/test_a.py" in p.stdout.split("input pkg changed")[1].split("\n")[0], p.stdout
