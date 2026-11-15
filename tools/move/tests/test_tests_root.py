"""`tests_root.py` on a toy tree: its inventory names each layout snag, `split` moves a distribution's suite into its tests/,
and the maps it writes move every test under the root's `tests/` with `layout.py`, each suite still collecting its tests
from its own directory."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

MOVE = Path(__file__).resolve().parents[3] / "tools" / "move"
spec = importlib.util.spec_from_file_location("tests_root_under_test", MOVE / "tests_root.py")
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
TREE = {
    "pyproject.toml": '[project]\nname = "verity"\nversion = "0"\n\n[tool.uv.workspace]\n'
                      'members = ["adapters/dist", "tools/plain"]\n\n[tool.uv.sources]\nverity = { workspace = true }\n\n'
                      '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
    "tests/test_tree.py": "def test_tree():\n    pass\n",
    "verity/__init__.py": "",
    "verity/a/__init__.py": "",
    "verity/a/mod.py": "X = 1\n",
    "verity/a/reader.py": "from pathlib import Path\n\nHELPER = Path(__file__).resolve().parent / \"tests/helper.py\"\n",
    "verity/a/tests/helper.py": "H = 1\n",
    "verity/a/pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["."]\n\n[tool.verity.tests]\n'
                               'untraced = ["tests/test_a.py"]\n',
    "verity/a/tests/test_a.py": "from pathlib import Path\n\nfrom verity.a import mod\n\nCODE = Path(__file__).parent / \"../..\" / \"a\" / \"mod.py\"\n\n\n"
                                "def test_a():\n    assert mod.X and CODE.is_file()\n",
    "adapters/dist/pyproject.toml": '[project]\nname = "dist"\nversion = "0"\nrequires-python = ">=3.12"\n\n'
                                    '[project.optional-dependencies]\nx = ["plain"]\ny = ["requests"]\n\n[build-system]\nrequires = ["hatchling"]\n'
                                    'build-backend = "hatchling.build"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
                                    'pythonpath = ["."]\n\n[tool.verity.tests]\nuntraced = ["tests/test_d.py"]\n'
                                    '# what its dependents read\nexports = ["adapters/dist/dist"]\n',
    "adapters/dist/conftest.py": "from __future__ import annotations\n\nOUTER = 1\n",
    "adapters/dist/dist/__init__.py": "",
    "adapters/dist/tests/__init__.py": "",
    "adapters/dist/tests/conftest.py": "from __future__ import annotations\n\nINNER = 1\n",
    "adapters/dist/tests/helpers.py": "H = 1\n",
    "adapters/dist/tests/test_d.py": "from pathlib import Path\n\nfrom tests.helpers import H\n\nHERE = Path(__file__).resolve().parent\n"
                                     "CODE = HERE.parent / \"dist\" / \"__init__.py\"\n\n\ndef test_d():\n    assert H and CODE.is_file()\n",
    "tools/plain/pyproject.toml": '[project]\nname = "plain"\nversion = "0"\n\n[tool.uv]\npackage = false\n\n'
                                  '[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["../../verity/a"]\n',
    "tools/plain/tests/test_p.py": "from pathlib import Path\n\nA = Path(__file__).resolve().parents[3] / \"verity\" / \"a\"\n"
                                   "HELPER = A / \"tests\" / \"helper.py\"\nROOT = Path(__file__).resolve().parents[3]\n"
                                   "CODE = ROOT / \"verity\" / \"a\"\n\n\ndef test_p():\n"
                                   "    assert HELPER.is_file() and (CODE / \"tests/helper.py\").is_file()\n",
    "tools/research/src/research/store/moves.json": '{"moves": []}\n',
}


def git(r: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=r, env=ENV, capture_output=True, text=True, check=True).stdout


def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    for p, t in TREE.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r


def test_the_inventory_names_each_tests_directory_with_its_suite_and_what_the_rule_cant_move_as_stated(tmp_path):
    """Each `X/tests` lands at `tests/X/`, but for a package its modules import as `tests`, which keeps its level; a suite
    that builds a distribution and a conftest beside a suite's pyproject are snags by name."""
    inv = TR.inventory(TR.Tree(repo(tmp_path)))
    rows = {r["dir"]: (r["kind"], r["lands"]) for r in inv["rows"]}
    assert rows == {"adapters/dist/tests": ("distribution", "tests/adapters/dist/tests/"),
                    "tools/plain/tests": ("virtual", "tests/tools/plain/"),
                    "verity/a/tests": ("suite-only", "tests/verity/a/")}
    assert inv["snags"]["a suite whose pyproject builds a distribution (split first)"] == ["adapters/dist/tests (dist)"]
    assert inv["snags"]["a tests/ package its modules import as `tests` (one level kept)"] == ["adapters/dist/tests"]
    assert "adapters/dist/conftest.py" in inv["snags"][
        "a test-like file outside every tests/ (conftest beside a suite's pyproject, or a module named test_*)"]


def test_split_moves_a_distributions_suite_to_a_virtual_member_in_its_tests_once(tmp_path):
    """The member depends on the distribution with its extras that name a workspace package, its lists are written from its new directory, the root
    lists it beside the distribution, and the conftest beside the pyproject joins the one in tests/. What dependents key
    on (`exports`) stays the distribution's, and the member, which no longer holds it, reads it by name. A rerun splits
    none."""
    r = repo(tmp_path)
    assert TR.run_split(TR.Tree(r), []) == ["adapters/dist"]
    member = tomllib.loads((r / "adapters/dist/tests/pyproject.toml").read_text())
    assert member["project"]["name"] == "dist-tests" and member["project"]["dependencies"] == ["dist[x]"]
    assert member["tool"]["uv"] == {"package": False}
    assert member["tool"]["pytest"]["ini_options"] == {"testpaths": ["."], "pythonpath": [".."]}
    assert member["tool"]["verity"]["tests"] == {"untraced": ["test_d.py"], "inputs": ["adapters/dist"]}
    dist = (r / "adapters/dist/pyproject.toml").read_text()
    assert tomllib.loads(dist)["tool"] == {"verity": {"tests": {"exports": ["adapters/dist/dist"]}}}
    assert "# what its dependents read\nexports" in dist
    uv = tomllib.loads((r / "pyproject.toml").read_text())["tool"]["uv"]
    assert uv["workspace"]["members"] == ["adapters/dist", "adapters/dist/tests", "tools/plain"]
    assert uv["sources"] == {"verity": {"workspace": True}, "dist": {"workspace": True}}
    assert not (r / "adapters/dist/conftest.py").exists()
    conftest = (r / "adapters/dist/tests/conftest.py").read_text()
    assert conftest.index("OUTER") < conftest.index("INNER") and conftest.count("from __future__") == 1
    assert conftest.startswith("from __future__ import annotations\n")
    assert TR.run_split(TR.Tree(r), []) == []


def test_the_maps_move_every_test_under_the_roots_tests_with_each_suites_pyproject_beside_them(tmp_path):
    """After `split`, `maps`, its move, `suites-map`, its move and `anchors`: every test is under `tests/`, each suite's
    pyproject beside its tests collects them from its own directory and puts the same directories on its path, a suite
    whose own directory held more than its tests reads it still, and a reference through `Path(__file__)` or a name bound
    to it still reaches what it did: a test's to its code, a test's to another's tests, and code's to its tests."""
    r = repo(tmp_path)
    TR.run_split(TR.Tree(r), [])
    git(r, "commit", "-q", "-m", "split")
    base = git(r, "rev-parse", "HEAD").strip()

    def move(m: str):
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", f"map {m}")
        p = subprocess.run([sys.executable, str(MOVE / "layout.py"), "--root", str(r), "--map", m, "--no-lock"], cwd=r,
                           env=ENV, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", m)

    assert TR.main(["maps"], root=r) == 0
    files = tomllib.loads((r / TR.FILES_MAP).read_text())
    assert files["paths"] == {"adapters/dist/tests/": "tests/adapters/dist/tests/", "tools/plain/tests/": "tests/tools/plain/",
                              "verity/a/tests/": "tests/verity/a/"}
    assert "split" not in files and files["rewrite"]["keep_relative"] == [f"{d}*.py" for d in files["paths"]]
    move(TR.FILES_MAP)
    assert TR.main(["suites-map"], root=r) == 0
    suites = tomllib.loads((r / TR.SUITES_MAP).read_text())
    assert suites["paths"] == {"tools/plain/pyproject.toml": "tests/tools/plain/pyproject.toml",
                               "verity/a/pyproject.toml": "tests/verity/a/pyproject.toml"}
    assert suites["pyproject"]["tests/verity/a/pyproject.toml"]["add_inputs"] == ["verity/a"]
    assert "add_inputs" not in suites["pyproject"]["tests/tools/plain/pyproject.toml"]
    move(TR.SUITES_MAP)
    left = [f for f in git(r, "ls-files").split() if "/tests/" in f and not f.startswith(("tests/", "tools/move/"))]
    assert left == []
    a = tomllib.loads((r / "tests/verity/a/pyproject.toml").read_text())
    assert a["tool"]["pytest"]["ini_options"] == {"testpaths": ["."], "pythonpath": ["../../../verity/a"]}
    assert a["tool"]["verity"]["tests"] == {"untraced": ["test_a.py"], "inputs": ["verity/a"]}
    dist = tomllib.loads((r / "tests/adapters/dist/tests/pyproject.toml").read_text())
    assert dist["tool"]["pytest"]["ini_options"] == {"testpaths": ["."], "pythonpath": ["../../../../adapters/dist"]}
    plain = tomllib.loads((r / "tests/tools/plain/pyproject.toml").read_text())
    assert plain["tool"]["pytest"]["ini_options"] == {"testpaths": ["."], "pythonpath": ["../../../verity/a"]}
    members = tomllib.loads((r / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    assert members == ["adapters/dist", "tests/adapters/dist/tests", "tests/tools/plain"]
    assert TR.main(["anchors", base, "--write"], root=r) == 0
    assert 'CODE = HERE.parents[3] / "adapters" / "dist" / "dist" / "__init__.py"' in (
        r / "tests/adapters/dist/tests/test_d.py").read_text()
    assert 'CODE = Path(__file__).parents[3] / "verity" / "a" / "mod.py"' in (r / "tests/verity/a/test_a.py").read_text()
    assert 'HELPER = Path(__file__).resolve().parents[2] / "tests" / "verity" / "a" / "helper.py"' in (
        r / "verity/a/reader.py").read_text()
    p = (r / "tests/tools/plain/test_p.py").read_text()
    assert 'HELPER = A.parents[1] / "tests" / "verity" / "a" / "helper.py"' in p and "assert HELPER.is_file() and" in p
    assert '(CODE.parents[1] / "tests" / "verity" / "a" / "helper.py").is_file()' in p
    assert TR.anchors(TR.Tree(r), base, write=False) == {}
