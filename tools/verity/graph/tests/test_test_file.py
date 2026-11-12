"""The test-file family: on this tree its tasks are the test files `suites.py` runs; on a toy checkout its listing is the same
bytes from any checkout of the same files, a suite check runs after its Lean steps makes none, and a test file's inputs are its conftests, the `__init__.py` files their import runs,
their import closure, what they start, the suite's declared data and reads, the pyprojects above it and the read guard; and its runner's
status is pytest's verdict on the file under the read guard: a pass, a fail when a test or its collection failed, an error when
the file read outside its inputs or pytest was killed."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from conftest import toy_tree

from verity_graph import ROOT, graph, runners
from verity_graph.families import build
from verity_graph.families import test_file as family
from verity_graph.runners import test_file as runner


def _table() -> str:
    """`[family.test-file]` as this tree's tasks.toml has it."""
    text = (ROOT / graph.SPEC).read_text()
    start = text.index("[family.test-file]")
    end = text.find("\n[", start)
    return text[start:] if end < 0 else text[start:end]


SPEC = '[env.py]\nspec = ["uv.lock", "pyproject.toml"]\n\n' + _table()
LISTERS = {"test-file": family.tasks}
#: what every toy checkout holds besides its suite: the files the core and the family make inputs of every task
BASE = {f"{graph.RUNNERS}/test_file.py": "", **{p: "" for p in family.GUARD}}

SUITE = {
    **BASE,
    "pyproject.toml": '[project]\nname = "toy"\n\n[tool.uv.workspace]\nmembers = ["lib", "s"]\n',
    "lib/pyproject.toml": ('[project]\nname = "toy-lib"\n\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
                           '[tool.hatch.build.targets.wheel]\npackages = ["toy_lib"]\n'),
    "lib/toy_lib/__init__.py": "from .core import f\n",
    "lib/toy_lib/core.py": "import json\n\nfrom . import helper\n\n\ndef f():\n    return helper.g()\n",
    "lib/toy_lib/helper.py": "def g():\n    return 1\n",
    "lib/toy_lib/extra.py": "X = 1\n",
    "lib/toy_lib/cli.py": "print('cli')\n",
    "lib/toy_lib/unused.py": "import toy_lib.core\n",
    "s/pyproject.toml": ('[project]\nname = "toy-suite"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                         '[tool.verity.tests]\ninputs = ["s/data", "s/table.csv"]\ntimeout_s = 120.5\nworker_gb = 6\n'),
    "s/table.csv": "a,b\n",
    "s/data/x.json": "{}\n",
    "s/data/gen.py": "print(1)\n",
    "s/data/README.md": "notes\n",
    "s/notes.txt": "n\n",
    "s/scripts/run_me.py": "import toy_lib.helper\n",
    "s/tests/conftest.py": "import toy_lib\n",
    "s/tests/helpers.py": 'from pathlib import Path\n\nGOLDEN = Path(__file__).resolve().parents[2] / "golden"\n\n\ndef h():\n    return 2\n',
    "golden/g.txt": "g\n",
    "s/tests/test_a.py": ('import importlib\n\nfrom helpers import h\nfrom toy_lib import core\n\n\ndef test_a():\n'
                          '    importlib.import_module("toy_lib.extra")\n    name = "toy_lib." + "helper"\n'
                          '    importlib.import_module(name)\n    assert core.f() + h() == 3\n'),
    "s/tests/test_c.py": ('import subprocess\nimport sys\n\n\ndef test_c():\n'
                          '    subprocess.run([sys.executable, "scripts/run_me.py"], check=True)\n'
                          '    subprocess.run([sys.executable, "-m", "toy_lib.cli"], check=True)\n'),
    "s/tests/sub/__init__.py": "",
    "s/tests/sub/conftest.py": "",
    "s/tests/sub/local.py": "V = 3\n",
    "s/tests/sub/test_b.py": ('from toy_lib import adapters\n\nfrom .local import V\n\n\ndef test_b(monkeypatch):\n'
                              '    monkeypatch.setattr("toy_lib.extra.X", 2)\n    assert V == 3 and adapters.load("x").X == 1\n'),
    "s/tests/fixtures/test_data.py": "def test_never():\n    pass\n",
    "s/tests/test_d.py": ('import sys\nfrom pathlib import Path\n\nHERE = Path(__file__).resolve().parent\n'
                          'ASSETS = Path(__file__).resolve().parents[2] / "lib" / "assets"\n'
                          'sys.path.insert(0, str(HERE.parent / "tools"))\n\nimport finder  # noqa: E402\n'
                          'from toy_lib import plugins  # noqa: E402\n'),
    "s/tools/finder.py": 'PATTERN = "*/pyproject.toml"\n',
    "lib/settings.py": "REGISTRY = None\n",
    "lib/assets/a.txt": "a\n",
    "lib/toy_lib/plugins/__init__.py": ('import importlib\nimport pkgutil\n\n\ndef load():\n    for m in pkgutil.iter_modules(__path__):\n'
                                        '        importlib.import_module(f"toy_lib.plugins.{m.name}")\n'),
    "lib/toy_lib/plugins/a.py": "A = 1\n",
    "lib/toy_lib/plugins/b.py": "from verity_config.settings import of\n\nsettings = of(__file__)\n",
    "lib/toy_lib/plugins/table.json": "{}\n",
    "lib/toy_lib/plugins/data/t.bin": "t\n",
    "lib/toy_lib/adapters/__init__.py": ('import importlib\n\nADAPTERS = {"x": "xa"}\n\n\ndef load(name):\n'
                                         '    module = f"{__name__}.{ADAPTERS[name]}"\n    return importlib.import_module(module)\n'),
    "lib/toy_lib/adapters/xa.py": "X = 1\n",
}
#: in every toy task: the suite's declared inputs, Python included, but its directory's Markdown, its own directory's data, the
#: pyprojects above, the guard
COMMON = [*family.GUARD, "pyproject.toml", "s/data/gen.py", "s/data/x.json", "s/notes.txt", "s/pyproject.toml", "s/table.csv"]
LIB = ["lib/toy_lib/__init__.py", "lib/toy_lib/core.py", "lib/toy_lib/helper.py"]


def test_its_tasks_are_the_test_files_suites_py_runs():
    """HEAD's, which both read: `suites.worktree()` would read git's index, which is no file of the tree."""
    spec = importlib.util.spec_from_file_location("graph_suites", ROOT / "tools" / "verity" / "check" / "suites.py")
    suites = sys.modules["graph_suites"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suites)
    pkgs = suites.members()
    want = {s.path: suites.test_files(s, "HEAD", suites.dropped(s, suites.inputs_of(s, pkgs))) for s in suites.discover()}
    ours = family.Tree(ROOT, [p for p in graph.git(ROOT, "ls-tree", "-r", "-z", "--name-only", "HEAD").decode().split("\0") if p])
    found = family.discover(ours)
    assert {s.path: family.suite_tests(ours, s, found) for s in found} == want
    assert family.STORE_READS == suites.STORE_READS, "what its guard lets every suite read, which a whole-tree task declares"


def test_the_listing_is_the_same_bytes_from_any_checkout(tmp_path):
    one, two = (toy_tree(tmp_path / d, SPEC, SUITE) for d in ("one", "two/deeper"))
    assert json.dumps(family.tasks(one, graph.tracked(one))) == json.dumps(family.tasks(two, graph.tracked(two)))
    assert family.tasks(one, graph.tracked(one)) == family.tasks(one, graph.tracked(one))
    a, b = (graph.listing(r, listers=LISTERS)["tasks"] for r in (one, two))
    assert a == b and all(t["key"] for t in a)


def test_a_test_file_s_inputs(tmp_path):
    root = toy_tree(tmp_path / "tree", SPEC, SUITE)
    by = {t["id"]: t for t in family.tasks(root, graph.tracked(root))}
    assert sorted(by) == ["test-file:s/tests/sub/test_b.py", "test-file:s/tests/test_a.py", "test-file:s/tests/test_c.py",
                          "test-file:s/tests/test_d.py"], "a data directory's test_*.py is no test file"
    a = by["test-file:s/tests/test_a.py"]
    assert a["inputs"] == sorted([*COMMON, *LIB, "golden/g.txt", "lib/toy_lib/extra.py", "s/tests/conftest.py", "s/tests/helpers.py",
                                  "s/tests/test_a.py"]), \
        "its conftest, the library it imports, the module named to import_module and the directory its helper names"
    assert a["vars"] == {"suite": "s"} and a["timeout_s"] == 121 and a["resources"] == {"mem_gb": 6}
    assert by["test-file:s/tests/sub/test_b.py"]["inputs"] == sorted(
        [*COMMON, *LIB, "lib/toy_lib/adapters/__init__.py", "lib/toy_lib/adapters/xa.py", "lib/toy_lib/extra.py",
         "s/tests/conftest.py", "s/tests/sub/__init__.py", "s/tests/sub/conftest.py", "s/tests/sub/local.py",
         "s/tests/sub/test_b.py"]), \
        ("both conftests, its package's __init__.py, its relative import, the module of the target it monkeypatches and each "
         "module of the package a name holding an f-string names")
    assert by["test-file:s/tests/test_c.py"]["inputs"] == sorted(
        [*COMMON, *LIB, "lib/toy_lib/cli.py", "s/scripts/run_me.py", "s/tests/conftest.py", "s/tests/test_c.py"]), \
        "the script it runs and the module it runs with -m, with what they import"
    beside_registry = [f for f in graph.tracked(root) if f.startswith("lib/") and f.count("/") == 1 and not f.endswith(".py")]
    assert by["test-file:s/tests/test_d.py"]["inputs"] == sorted(
        [*COMMON, *LIB, *beside_registry, "lib/assets/a.txt", "lib/settings.py", "lib/toy_lib/plugins/__init__.py",
         "lib/toy_lib/plugins/a.py", "lib/toy_lib/plugins/b.py", "lib/toy_lib/plugins/data/t.bin",
         "lib/toy_lib/plugins/table.json", "s/tests/conftest.py", "s/tests/test_d.py", "s/tools/finder.py"]), \
        ("the module a directory it puts on sys.path serves, each module of the package an f-string names, the data beside "
         "them, the settings registry of(__file__) loads, the directory it names, and every pyproject.toml for a module that "
         "lists them")
    tree = family.Tree(root, graph.tracked(root))
    assert family.Imports(tree, family.Layout(tree)).info("s/tests/test_a.py").unfollowed == 1, \
        "import_module(name) is followed only for a constant name"


def test_a_suite_check_runs_after_its_lean_steps_makes_no_test_file_task(tmp_path):
    """A suite `suites.AFTER_LEAN` names, by its name or its path, runs once: as the build family's lean-suites task, after
    lean-build has built what it runs, never as test-file tasks besides; in a tree without the verifier, which lists no
    lean-suites task, its files are test-file tasks, as check runs it with the rest there."""
    pytest_ = '\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    files = {**SUITE, build.SUITES: 'AFTER_LEAN = ("toy-suite", "t")\n',
             "t/pyproject.toml": f'[project]\nname = "toy-t"\n{pytest_}', "t/tests/test_t.py": "def test_t():\n    pass\n",
             "u/pyproject.toml": f'[project]\nname = "toy-u"\n{pytest_}', "u/tests/test_u.py": "def test_u():\n    pass\n"}
    plain = toy_tree(tmp_path / "plain", SPEC, files)
    assert [t["id"] for t in family.tasks(plain, graph.tracked(plain))] == [
        "test-file:s/tests/sub/test_b.py", "test-file:s/tests/test_a.py", "test-file:s/tests/test_c.py",
        "test-file:s/tests/test_d.py", "test-file:t/tests/test_t.py", "test-file:u/tests/test_u.py"]
    assert build.tasks(plain, graph.tracked(plain)) == []
    lean = {f"{build.LEAN}/lakefile.toml": 'defaultTargets = ["v"]\n[[lean_exe]]\nname = "v"\n', f"{build.LEAN}/V.lean": ""}
    root = toy_tree(tmp_path / "tree", SPEC, {**files, **lean})
    assert [t["id"] for t in family.tasks(root, graph.tracked(root))] == ["test-file:u/tests/test_u.py"]
    assert [t["id"] for t in build.tasks(root, graph.tracked(root)) if t["kind"] == "lean-suites"] == [
        "lean-suites:toy-suite", "lean-suites:t"]


WALKS = {
    **SUITE,
    "t/pyproject.toml": ('[project]\nname = "toy-walks"\ndependencies = ["toy-lib"]\n\n[tool.pytest.ini_options]\n'
                         'testpaths = ["tests"]\n\n[tool.verity.tests]\nuntraced = ["tests/test_m.py"]\n\n[[tool.verity.tests.reads]]\ntests = ["tests/test_w*.py"]\npaths = ["lib/assets", "lib/toy_lib/plugins/*.json"]\n'),
    "t/tests/__init__.py": "",
    "t/tests/deep/test_e.py": ('import subprocess\nimport sys\n\n\ndef test_e():\n'
                               '    subprocess.run([sys.executable, "-c", "import json; from toy_lib import helper\\nimport toy_lib.cli"])\n'),
    "t/tests/test_m.py": ('import tomllib\nfrom pathlib import Path\n\nROOT = Path(__file__).parents[2]\n\n\ndef test_m():\n'
                          '    for m in tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]:\n'
                          '        assert (ROOT / m / "pyproject.toml").is_file()\n'),
    "t/tests/build.py": "from torch.utils.cpp_extension import load_inline\n",
    "t/tests/test_native.py": "import build\n\n\ndef test_native():\n    pass\n",
    "t/tests/test_walk.py": ('from pathlib import Path\n\n\ndef test_walk():\n'
                             '    assert sorted(Path(__file__).parents[2].glob("lib/**/*.txt"))\n'),
}


def test_what_a_test_file_reads_that_its_source_does_not_name(tmp_path):
    root = toy_tree(tmp_path / "tree", SPEC, WALKS)
    by = {t["id"]: set(t["inputs"]) for t in family.tasks(root, graph.tracked(root))}
    e = by["test-file:t/tests/deep/test_e.py"]
    assert "t/tests/__init__.py" in e, "pytest imports a Package node's __init__.py above a directory that isn't one"
    assert {*LIB, "lib/toy_lib/cli.py"} <= e, "the modules the program it hands `python -c` imports"
    m = by["test-file:t/tests/test_m.py"]
    assert {"lib/pyproject.toml", "s/pyproject.toml"} <= m, "each member's pyproject for a test that reads the root's members"
    assert "lib/pyproject.toml" not in e
    tree = family.Tree(root, graph.tracked(root))
    assert {*tree.under("lib"), *tree.under("t")} <= m and not m & set(tree.under("s/tests")), \
        "a file its suite says starts a program the guard can't trace is keyed on the suite's directory and its dependencies'"
    t = {t["id"]: t for t in family.tasks(root, graph.tracked(root))}
    assert t["test-file:t/tests/test_m.py"]["vars"] == {"suite": "t", "untraced": "suite"}
    assert t["test-file:t/tests/deep/test_e.py"]["vars"] == {"suite": "t"} and "lib/assets/a.txt" not in e
    assert t["test-file:t/tests/test_native.py"]["vars"] == {"suite": "t", "untraced": "compilers"}, \
        "a file whose closure builds native code may start a compiler"
    plain = toy_tree(tmp_path / "plain", SPEC, {**WALKS, "t/pyproject.toml": WALKS["t/pyproject.toml"].partition("\n[[")[0]})
    walked = {"lib/assets/a.txt", "lib/toy_lib/plugins/table.json"}
    assert not walked & next(set(t["inputs"]) for t in family.tasks(plain, graph.tracked(plain))
                             if t["id"] == "test-file:t/tests/test_walk.py")
    assert walked <= by["test-file:t/tests/test_walk.py"] and not walked & e, \
        "what its suite's [[tool.verity.tests.reads]] declare it reads, for the test files it names alone"


def test_a_task_declares_the_registry_its_closure_names_and_an_untraced_one_what_its_suite_s_guard_allows(tmp_path):
    """`suites.py`'s guard lets every suite read each pyproject.toml and `STORE_READS`, and keys a pass on them only when its
    tests read them, which no key of the graph can: a task declares the registry where a module of its closure names it (the
    store's replica map), and a task keyed on its suite's tree all of them."""
    store = {family.REGISTRY: "{}\n", family.STORE_READS[1]: "",
             "lib/toy_lib/store.py": 'from pathlib import Path\n\nREPLICAS = Path("fixtures") / "artifacts.json"\n',
             "s/tests/test_store.py": "from toy_lib import store\n\n\ndef test_store():\n    assert store.REPLICAS\n"}
    root = toy_tree(tmp_path / "tree", SPEC, {**WALKS, **store})
    by = {t["id"]: set(t["inputs"]) for t in family.tasks(root, graph.tracked(root))}
    assert family.REGISTRY in by["test-file:s/tests/test_store.py"] and family.REGISTRY not in by["test-file:s/tests/test_a.py"]
    tree = family.Tree(root, graph.tracked(root))
    assert {*family.STORE_READS, *tree.pyprojects} <= by["test-file:t/tests/test_m.py"]
    assert not {*family.STORE_READS, "s/pyproject.toml"} & by["test-file:t/tests/deep/test_e.py"]


def test_built_paths_are_the_expression_s_matches():
    texts = ['a/"b"', 'x = a.b /\n  "c" / "d"', 'Path(__file__)/"x"', ' / "x"', 'a\u2003/"x"', 'f(a)\t/ \'b\' / "c"',
             'a / b / "c"', '"q"/"x"', 'a[0].parent/"x"/"y" HERE/"z"', '\u00e9a/"x"', 'a.b.c/"x"a/"y"', "x/'", 'a/"b"/"c"/',
             'a()/"b"', (ROOT / "tools/verity/graph/tests/test_test_file.py").read_text()]
    for t in texts:
        assert [m.span() for m in family._built_paths(t)] == [m.span() for m in family._BUILT_PATH.finditer(t)], t[:60]


@pytest.mark.parametrize("reads, refused", [
    ('tests = ["tests/test_gone.py"]\npaths = ["lib/assets"]\n', "names none of the suite's test files"),
    ('tests = ["tests/test_walk.py"]\npaths = ["lib/gone"]\n', "is no file or directory of the tree"),
    ('tests = ["tests/test_walk.py"]\npaths = ["."]\n', "is no file or directory of the tree"),
    ('tests = "tests/test_walk.py"\npaths = ["lib/assets"]\n', "each a non-empty list of strings"),
])
def test_a_reads_entry_that_declares_nothing_is_refused(tmp_path, reads, refused):
    py = WALKS["t/pyproject.toml"].partition("[[tool.verity.tests.reads]]")[0] + "[[tool.verity.tests.reads]]\n" + reads
    root = toy_tree(tmp_path / "tree", SPEC, {**WALKS, "t/pyproject.toml": py})
    with pytest.raises(ValueError, match=refused):
        family.tasks(root, graph.tracked(root))


def test_the_core_completes_them_from_the_table(tmp_path):
    root = toy_tree(tmp_path / "tree", SPEC, SUITE)
    t = next(t for t in graph.listing(root, listers=LISTERS)["tasks"] if t["id"] == "test-file:s/tests/sub/test_b.py")
    assert (t["kind"], t["env"], t["argv"]) == ("test-file", "py", ["verity", "graph", "run", t["id"]])
    assert t["about"] == "pytest on s/tests/sub/test_b.py alone, as suite s runs it"
    assert t["resources"] == {"cores": 1, "mem_gb": 6, "disk_gb": 2} and t["timeout_s"] == 121
    assert set(graph.runner_sources("test-file")) <= set(t["inputs"]) and not t["reads"] and not t["outputs"]


#: a module a test loads by a path it computes, which puts its own directory's `lib/` on `sys.path` and imports from it
LOADED = ('import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))\n\n'
          'import dep  # noqa: E402\n\nV = dep.V\n')


def _loads(where: str) -> str:
    """A test file that runs ``where``/p.py by a path no string of it names."""
    head, tail = where[:2], where[2:]
    return ('import runpy\nfrom pathlib import Path\n\n\ndef test_loads():\n'
            f'    where, name = "{head}" + "{tail}", "p" + ".py"\n'
            '    assert runpy.run_path(str(Path(__file__).parents[2] / where / name))["V"] == 1\n')


RUNS = {
    **BASE,
    "s/pyproject.toml": ('[project]\nname = "toy-suite"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                         '[tool.verity.tests]\nuntraced = ["tests/test_git_keyed.py"]\ninputs = ["plugins/p.py"]\n\n'
                         '[[tool.verity.tests.reads]]\ntests = ["tests/test_walked.py"]\npaths = ["walked/p.py"]\n'),
    "plugins/p.py": LOADED,
    "plugins/lib/dep.py": "V = 1\n",
    "walked/p.py": LOADED,
    "walked/lib/dep.py": "V = 1\n",
    "s/tests/test_declared.py": _loads("plugins"),
    "s/tests/test_walked.py": _loads("walked"),
    "s/notes.txt": "n\n",
    "s/tests/test_git.py": "import subprocess\n\n\ndef test_git():\n    subprocess.run(['git', '--version'], check=True)\n",
    "s/tests/test_cc.py": ("import shutil\nimport subprocess\n\nimport pytest\n\n\n"
                           "@pytest.mark.skipif(not shutil.which('cc'), reason='no C compiler')\ndef test_cc():\n"
                           "    subprocess.run(['cc', '--version'], check=True)\n    subprocess.run(['git', '--version'], check=True)\n"),
    "s/tests/test_git_keyed.py": "import subprocess\n\n\ndef test_git():\n    subprocess.run(['git', '--version'], check=True)\n",
    "s/tests/test_ok.py": ('from pathlib import Path\n\n\ndef test_reads_its_env_s_spec():\n'
                           '    assert (Path(__file__).parents[2] / "uv.lock").read_text()\n'),
    "s/tests/test_bad.py": "def test_good():\n    pass\n\n\ndef test_bad():\n    assert False\n",
    "s/tests/test_broken.py": "import toy_module_nowhere\n",
    "s/tests/test_none.py": "X = 1\n",
    "elsewhere/notes.txt": "n\n",
    "s/tests/test_reads.py": ('from pathlib import Path\n\n\ndef test_reads_what_it_did_not_declare():\n'
                              '    where = "else" + "where"\n'
                              '    assert (Path(__file__).parents[2] / where / "notes.txt").read_text() == "n\\n"\n'),
    "s/tests/test_python3.py": ("import subprocess\nimport sys\n\n\ndef test_python3_is_this_python():\n"
                                "    out = subprocess.run(['python3', '-c', 'import sys; print(sys.prefix)'], capture_output=True, text=True)\n"
                                "    assert out.stdout.strip() == sys.prefix\n"),
    "s/tests/test_dies.py": "import os\nimport signal\n\n\ndef test_dies():\n    os.kill(os.getpid(), signal.SIGKILL)\n",
    ".gitignore": ".lake/\n",
    "s/tests/test_lock.py": ('from pathlib import Path\n\n\ndef test_locks_a_build():\n'
                             '    lock = Path(__file__).parents[2] / ".lake" / "build.lock"\n    lock.parent.mkdir(exist_ok=True)\n'
                             '    with open(lock, "a"):\n        pass\n'),
}


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> dict[str, dict]:
    """Each toy test file's result/v1 (`runners.execute`), by file name, the runner pointed at the toy checkout."""
    root = toy_tree(tmp_path_factory.mktemp("runs"), SPEC, RUNS)
    work = tmp_path_factory.mktemp("work")
    doc = graph.listing(root, listers=LISTERS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "ROOT", root)
        tools = {Path(w).parent for w in map(shutil.which, ("git", "cc")) if w} - {Path(sys.executable).parent}
        mp.setenv("PATH", os.pathsep.join(map(str, sorted(tools))))
        return {Path(t["id"]).name: runners.execute(t, doc, work / Path(t["id"]).stem) for t in doc["tasks"]}


def _counts(r: dict) -> dict:
    return {k: r["detail"][k] for k in ("passed", "failed", "skipped", "errors", "exit")}


def test_a_file_whose_tests_pass_passes(results):
    r = results["test_ok.py"]
    assert r["status"] == "pass" and not r["undeclared_reads"], "a read of its env's spec is declared"
    assert _counts(r) == {"passed": 1, "failed": 0, "skipped": 0, "errors": 0, "exit": 0}
    assert results["test_none.py"]["status"] == "pass" and results["test_none.py"]["detail"]["exit"] == 5
    assert results["test_python3.py"]["status"] == "pass", "a test's python3 is the runner's Python, as under `uv run`"


def test_a_failing_test_or_collection_is_a_fail(results):
    r = results["test_bad.py"]
    assert r["status"] == "fail" and _counts(r) == {"passed": 1, "failed": 1, "skipped": 0, "errors": 0, "exit": 1}
    assert r["detail"]["failed_tests"] == ["tests/test_bad.py::test_bad"]
    assert results["test_broken.py"]["status"] == "fail" and results["test_broken.py"]["detail"]["exit"] == 2


def test_a_read_outside_its_inputs_is_an_error(results):
    r = results["test_reads.py"]
    assert r["status"] == "error" and r["undeclared_reads"] == ["elsewhere/notes.txt"] and r["detail"]["passed"] == 1
    r = results["test_lock.py"]
    assert r["status"] == "pass" and not r["undeclared_reads"] and r["detail"]["ignored"] == [".lake/build.lock"], \
        "a file git ignores (a build's lock) is none the graph lists, so no input a key could hold"


def test_python_a_task_declares_is_an_input_with_what_it_imports(results):
    """Its suite's declared Python and a reads entry's: its suite's guard lets the file read them, and a test that loads one by a
    path or a name it computes (circuit-check's sidecars, a registry's tool) reads it and runs its imports, which the test
    file's own closure never reaches. Declaring less made the executor's runs of such files errors that `suites.py` passes."""
    for f in ("test_declared.py", "test_walked.py"):
        r = results[f]
        assert r["status"] == "pass" and not r["undeclared_reads"] and r["detail"]["passed"] == 1, (f, r)


def test_a_program_the_guard_cannot_trace_is_an_error_unless_the_task_is_keyed_on_its_suite(results):
    r = results["test_git.py"]
    assert r["status"] == "error" and r["detail"]["untraced"] == ["git"] and "[tool.verity.tests] untraced" in r["detail"]["error"]
    r = results["test_git_keyed.py"]
    assert r["status"] == "pass" and r["detail"]["untraced"] == ["git"]
    r = results["test_cc.py"]
    assert r["status"] == "error" and "(cc, git)" in r["detail"]["error"], \
        "a task whose closure builds native code may start a compiler, and nothing else untraced"
    assert runner.allowed({"vars": {"untraced": "compilers"}}, {"/usr/bin/g++", "nvcc"})
    assert not runner.allowed({"vars": {"untraced": "compilers"}}, {"g++", "git"}) and not runner.allowed({"vars": {}}, {"g++"})


def test_a_pytest_killed_is_an_error(results):
    r = results["test_dies.py"]
    assert r["status"] == "error" and r["detail"]["error"].startswith("pytest died by SIGKILL")


def test_it_fetches_the_registered_fixtures_git_does_not_hold(tmp_path, monkeypatch):
    reg = {"art:a": "s/fx/a.bin", "art:b": "s/fx/b.bin", "art:c": "other/c.bin", "art:d": "data/d.bin"}
    root = toy_tree(tmp_path / "tree", SPEC, {**RUNS, family.REGISTRY: json.dumps(reg), "s/fx/b.bin": "b\n"})
    monkeypatch.setattr(runner, "ROOT", root)
    task = {"vars": {"suite": "s"}, "inputs": [family.REGISTRY]}
    assert runner.fixtures(task, {"inputs": ["data"]}) == ["data/d.bin", "s/fx/a.bin"]
    assert runner.fixtures({**task, "inputs": []}, {}) == [], "a task that doesn't read the registry fetches nothing"
