"""`suites.py` in a tree without git, as `research run --source` ships one: suites are found, and keyed from their files."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _suites(monkeypatch, root: Path):
    spec = importlib.util.spec_from_file_location("suites_under_test", HERE / "suites.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)           # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod, "GIT", False)
    return mod


def test_a_shipped_tree_is_discovered_and_keyed_from_its_files(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (tmp_path / "pkg" / "tests").mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"1")
    S = _suites(monkeypatch, tmp_path)

    [suite] = S.discover()
    assert (suite.name, suite.path) == ("pkg", "pkg")
    assert S.metadata() == ["pkg/pyproject.toml", "pyproject.toml", "uv.lock"]
    before = S.object_ids(None, ["pkg", "uv.lock"], S.digests())
    assert before["uv.lock"] == "absent"

    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"2")      # what a machine adds is not source
    assert S.object_ids(None, ["pkg", "uv.lock"], S.digests()) == before
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    assert S.object_ids(None, ["pkg"], S.digests())["pkg"] != before["pkg"]


def test_a_declared_build_output_keys_the_suite(tmp_path, monkeypatch):
    """Git ignores build outputs, so a suite whose tests run or skip on one declares it (`[tool.verity.tests] artifacts`), and
    its pass is reused only for the same build."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                                                     '[tool.verity.tests]\nartifacts = ["pkg/.lake/build/bin/tool", "pkg/.lake/cache"]\n')
    S = _suites(monkeypatch, tmp_path)
    [suite] = S.discover()
    assert suite.artifacts == ("pkg/.lake/build/bin/tool", "pkg/.lake/cache")
    keys = [S.key(suite, {}, {}, [])]
    assert S.artifact_ids(suite) == {"pkg/.lake/build/bin/tool": "absent", "pkg/.lake/cache": "absent"}
    (tmp_path / "pkg" / ".lake" / "build" / "bin").mkdir(parents=True)
    (tmp_path / "pkg" / ".lake" / "build" / "bin" / "tool").write_bytes(b"one build")
    keys.append(S.key(suite, {}, {}, []))
    (tmp_path / "pkg" / ".lake" / "build" / "bin" / "tool").write_bytes(b"another build")
    (tmp_path / "pkg" / ".lake" / "cache").mkdir()
    keys.append(S.key(suite, {}, {}, []))
    assert len(set(keys)) == 3 and S.artifact_ids(suite)["pkg/.lake/cache"] == "directory"


def test_jobs_default_to_the_cpus_this_process_may_use(tmp_path, monkeypatch):
    S = _suites(monkeypatch, tmp_path)
    assert 1 <= S.cpus() <= (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())


def test_a_shipped_tree_reports_writes_as_git_would(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("**/target/\n")
    (tmp_path / "pkg" / "target").mkdir(parents=True)
    (tmp_path / "pkg" / "target" / "kept.bin").write_bytes(b"a tracked file the rules cover")
    (tmp_path / "pkg" / "a.py").write_text("A = 1\n")
    S = _suites(monkeypatch, tmp_path)
    before = S.digests()
    assert "pkg/target/kept.bin" in before

    (tmp_path / "pkg" / "target" / "build.o").write_bytes(b"x")        # a build output: ignored, as in a checkout
    assert S.changed(before) == []
    (tmp_path / "pkg" / "a.py").write_text("A = 2\n")
    (tmp_path / "pkg" / "new.json").write_text("{}\n")
    (tmp_path / "pkg" / "target" / "kept.bin").unlink()
    assert S.changed(before) == ["pkg/a.py", "pkg/new.json", "pkg/target/kept.bin"]


def _workspace(root: Path) -> None:
    """Members `a` (on numpy, with a `plot` extra on matplotlib) and `b` (on nothing), and the workspace's dev group."""
    (root / "pyproject.toml").write_text('[project]\nname = "ws"\n\n[tool.uv.workspace]\nmembers = ["a", "b"]\n\n'
                                         '[dependency-groups]\ndev = ["pytest>=8"]\n')
    for name, deps in (("a", '["numpy>=1"]'), ("b", "[]")):
        (root / name / "tests").mkdir(parents=True)
        (root / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"\ndependencies = {deps}\n\n'
                                                    '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _lock(root, numpy="2.0", pytest="8.0")


def _lock(root: Path, numpy: str, pytest: str) -> None:
    (root / "uv.lock").write_text(f'''version = 1
[[package]]
name = "a"
version = "0"
source = {{ editable = "a" }}
dependencies = [{{ name = "numpy" }}]
optional-dependencies = {{ plot = [{{ name = "matplotlib" }}] }}

[[package]]
name = "b"
version = "0"
source = {{ editable = "b" }}

[[package]]
name = "numpy"
version = "{numpy}"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "matplotlib"
version = "3.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "pytest"
version = "{pytest}"
source = {{ registry = "https://pypi.org/simple" }}
dependencies = [{{ name = "iniconfig" }}]

[[package]]
name = "iniconfig"
version = "2.0"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "ws"
version = "0"
source = {{ virtual = "." }}
dev-dependencies = {{ dev = [{{ name = "pytest" }}] }}
''')


def _keys(S) -> dict[str, str]:
    pkgs, lock, env = S.members(), S.locked(), S.environment()
    files = S.digests()
    out = {}
    for s in S.discover():
        paths = S.inputs_of(s, pkgs)
        out[s.name] = S.key(s, S.object_ids(None, sorted(set(paths) | set(S.RUNNER)), files), env, [], S.resolved(S.roots(s, pkgs), lock))
    return out


def test_a_lock_or_metadata_change_rekeys_only_the_suites_it_resolves_to(tmp_path, monkeypatch):
    _workspace(tmp_path)
    S = _suites(monkeypatch, tmp_path)
    pkgs, lock = S.members(), S.locked()
    [a, b] = S.discover()
    assert S.roots(a, pkgs) == ["a", "pytest"] and S.roots(b, pkgs) == ["b", "pytest"]
    assert sorted(S.resolved(S.roots(a, pkgs), lock)) == ["a", "iniconfig", "matplotlib", "numpy", "pytest"]
    assert sorted(S.resolved(S.roots(b, pkgs), lock)) == ["b", "iniconfig", "pytest"]
    before = _keys(S)
    _lock(tmp_path, numpy="2.1", pytest="8.0")                     # numpy moves: only a resolves to it
    after = _keys(S)
    assert after["a"] != before["a"] and after["b"] == before["b"]
    (tmp_path / "pyproject.toml").write_text((tmp_path / "pyproject.toml").read_text() + "\n# a comment\n")
    assert _keys(S) == after                                        # the root's metadata alone re-keys nothing
    _lock(tmp_path, numpy="2.1", pytest="8.1")                      # pytest moves: every suite runs with it
    assert all(_keys(S)[n] != after[n] for n in ("a", "b"))


def test_a_pass_is_reused_only_while_what_it_used_beside_its_key_holds(tmp_path, monkeypatch):
    _workspace(tmp_path)
    S = _suites(monkeypatch, tmp_path)
    lock, files = S.locked(), S.digests()
    r = S.Result(S.discover()[1], "passed", watched=["a/pyproject.toml", "pyproject.toml"], distributions=["NumPy", "pytest"],
                 missing=["torch"])
    deps = S.resolved(["b", "pytest"], lock)
    hit = S.used(r, ["b"], deps, None, files, lock)
    assert sorted(hit["watched"]) == ["a/pyproject.toml", "pyproject.toml"] and list(hit["imported"]) == ["numpy"]
    assert S.holds(hit, None, files, lock, set())
    assert not S.holds(hit, None, files, lock, {"torch"})                           # an import that found nothing now would
    (tmp_path / "pyproject.toml").write_text((tmp_path / "pyproject.toml").read_text() + "\n# read by b's tests\n")
    assert not S.holds(hit, None, S.digests(), lock, set())
    _lock(tmp_path, numpy="2.1", pytest="8.0")
    assert not S.holds({**hit, "watched": {}}, None, files, S.locked(), set())      # numpy, imported outside b's key, moved


def test_fetched_fixtures_are_the_registry_entries_git_does_not_hold(tmp_path, monkeypatch):
    import json
    import subprocess

    (tmp_path / "fixtures").mkdir()
    (tmp_path / "pkg" / "fx").mkdir(parents=True)
    (tmp_path / "pkg" / "fx" / "small.json").write_text("{}\n")
    reg = {"art:" + "a" * 64: "pkg/fx/small.json", "art:" + "b" * 64: "pkg/fx/big.proof", "art:" + "c" * 64: "other/big.bin",
           "previous": {"art:" + "a" * 64: ["art:" + "d" * 64]}}
    (tmp_path / "fixtures" / "artifacts.json").write_text(json.dumps(reg))
    S = _suites(monkeypatch, tmp_path)
    assert S.fetched_fixtures(None, {"pkg"}) == {"pkg/fx/small.json": "art:" + "a" * 64, "pkg/fx/big.proof": "art:" + "b" * 64}

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    tree = subprocess.run(["git", "-C", str(tmp_path), "write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    assert S.fetched_fixtures(tree, {"pkg"}) == {"pkg/fx/big.proof": "art:" + "b" * 64}     # the tracked entry is keyed by git
    assert S.fetched_fixtures(tree, {"."}) == {"pkg/fx/big.proof": "art:" + "b" * 64, "other/big.bin": "art:" + "c" * 64}


def _shipped_workspace(root: Path, *, absent: bool) -> None:
    """A tree as `research run --source` ships one: no .git, this runner under tools/check, one package suite whose test reads a
    registered fixture; with `absent`, the registry also names a fetched entry the tree lacks."""
    import hashlib
    import json
    import shutil

    (root / "tools" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (root / "pkg" / "tests" / "fixtures").mkdir(parents=True)
    (root / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    data = b'{"n": 1}\n'
    (root / "pkg" / "tests" / "fixtures" / "small.json").write_bytes(data)
    (root / "pkg" / "tests" / "test_a.py").write_text(
        "from pathlib import Path\n\ndef test_reads_its_fixture():\n"
        "    assert (Path(__file__).parent / 'fixtures' / 'small.json').read_text() == '{\"n\": 1}\\n'\n")
    from research.store.fixtures import fixture_id

    reg = {fixture_id(root / "pkg" / "tests" / "fixtures" / "small.json"): "pkg/tests/fixtures/small.json"}
    if absent:
        reg["art:" + hashlib.sha256(b"never shipped").hexdigest()] = "pkg/tests/fixtures/big.proof"
    (root / "fixtures").mkdir()
    (root / "fixtures" / "artifacts.json").write_text(json.dumps(reg))


def _run_shipped(root: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(RESEARCH_STORE=str(tmp_path / "pod-store"), RESEARCH_STORE_CONFIG=str(tmp_path / "absent.toml"),
               RESEARCH_PUBLIC_URL="", VERITY_TEST_CACHE=str(tmp_path / "cache"), VERITY_SKIP_STORE="1")
    return subprocess.run([sys.executable, str(root / "tools" / "check" / "suites.py"), "--fresh", "--jobs", "1"], cwd=root, env=env,
                          capture_output=True, text=True)


def test_a_shipped_tree_with_no_store_runs_the_suites_its_tracked_fixtures_feed(tmp_path):
    root = tmp_path / "tree"
    _shipped_workspace(root, absent=False)
    p = _run_shipped(root, tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "passed  pkg" in p.stdout and "1 passed" in p.stdout


def test_a_shipped_tree_with_no_store_fails_only_on_a_fixture_it_cannot_fetch(tmp_path):
    root = tmp_path / "tree"
    _shipped_workspace(root, absent=True)
    p = _run_shipped(root, tmp_path)
    assert p.returncode == 1
    assert "its fixtures could not be fetched: pkg/tests/fixtures/big.proof" in p.stdout, p.stdout


def test_what_a_module_can_reach_by_name(tmp_path, monkeypatch):
    """Imports and `test_...` words in strings are what a module mentions; a computed import leaves nothing out, unless it is computed
    from the module's own package's name; a top-level call or an assignment to an attribute or item acts on the process."""
    S = _suites(monkeypatch, tmp_path)
    s = S.scan("import a.b\nfrom .c import d\nX = 'tests/test_e.py::t'\nfrom importlib import import_module\n"
               "def f(n):\n    return import_module(f'{__name__}.{n}')\n")
    assert {"a", "b", "c", "d", "test_e"} <= s.names and s.package and not s.computed and not s.effects
    assert S.scan("import importlib\ndef f(n):\n    return importlib.import_module(n)\n").computed
    assert S.scan("import glob\nFILES = glob.glob('tests/test_*.py')\n").computed          # a pattern names test modules too
    assert not S.scan("import pytest\nnp = pytest.importorskip('numpy')\n").computed          # a literal name is a mention
    for effect in ("import sys\nsys.path.insert(0, '.')\n", "import os\nos.environ['A'] = '1'\n", "import m\nm.X = 1\n",
                   "try:\n    import x\nexcept ImportError:\n    print('no x')\n"):
        assert S.scan(effect).effects, effect
    assert not S.scan("'''doc'''\nfrom a import b\nA, B = 1, b()\nclass C:\n    pass\n").effects
    assert S.scan("def (:\n") is None


def _per_test_workspace(root: Path) -> None:
    """A shipped tree with this runner and one parallel suite: `test_heavy` (on a helper), `test_sibling` (failing until fixed),
    `test_uses` (importing `test_shared`), and `test_shared`."""
    import shutil

    (root / "tools" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (root / "pkg" / "tests").mkdir(parents=True)
    (root / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                                                 '[tool.verity.tests]\nparallel = true\n')
    for name, text in {"helpers.py": "VALUE = 1\n",
                       "test_heavy.py": "from helpers import VALUE\n\ndef test_heavy():\n    assert VALUE == 1\n\ndef test_heavy_too():\n    pass\n",
                       "test_sibling.py": "def test_sibling():\n    assert False\n",
                       "test_uses.py": "from test_shared import X\n\ndef test_uses():\n    assert X == 1\n",
                       "test_shared.py": "X = 1\n\ndef test_shared():\n    pass\n"}.items():
        (root / "pkg" / "tests" / name).write_text(text)


def _run_suite(root: Path, tmp_path: Path, *args: str, extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(RESEARCH_STORE=str(tmp_path / "pod-store"), RESEARCH_STORE_CONFIG=str(tmp_path / "absent.toml"), RESEARCH_FIXTURES_URL="",
               VERITY_TEST_CACHE=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1", **(extra or {}))
    return subprocess.run([sys.executable, str(root / "tools" / "check" / "suites.py"), "--jobs", "1", "--logs", str(tmp_path / "logs"), *args],
                          cwd=root, env=env, capture_output=True, text=True)


def _reused(tmp_path: Path) -> list[str]:
    import json

    p = tmp_path / "logs" / "pkg.reuse.json"
    return sorted(json.loads(p.read_text())) if p.exists() else []


def test_a_failed_suite_keeps_its_passes_and_a_sibling_fix_reuses_them(tmp_path):
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    p = _run_suite(root, tmp_path)
    assert p.returncode == 1 and "4 per-test passes kept" in p.stdout, p.stdout

    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")       # the fix: another module
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "1 passed, 4 reused" in p.stdout, p.stdout
    assert _reused(tmp_path) == ["tests/test_heavy.py::test_heavy", "tests/test_heavy.py::test_heavy_too",
                                 "tests/test_shared.py::test_shared", "tests/test_uses.py::test_uses"]

    (tests / "test_shared.py").write_text("X = 1\n\n\ndef test_shared():\n    pass\n")       # reaches test_uses, which imports it
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "2 passed, 3 reused" in p.stdout, p.stdout
    assert _reused(tmp_path) == ["tests/test_heavy.py::test_heavy", "tests/test_heavy.py::test_heavy_too", "tests/test_sibling.py::test_sibling"]

    (tests / "helpers.py").write_text("VALUE = 1  # not a test module: in every key\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "5 passed in" in p.stdout and _reused(tmp_path) == [], p.stdout

    (tests / "test_effect.py").write_text("import os\nos.environ['PKG_FLAG'] = '1'\n\ndef test_effect():\n    pass\n")
    (tests / "test_dyn.py").write_text("import importlib\n\ndef test_dyn():\n    assert importlib.import_module('test_' + 'shared').X == 1\n")
    p = _run_suite(root, tmp_path)                        # new modules: the one acting on import joins every key
    assert p.returncode == 0 and "7 passed" in p.stdout, p.stdout
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert 1\n")
    p = _run_suite(root, tmp_path)                        # test_dyn imports by a computed name: no test module is left out of its key
    assert p.returncode == 0 and "2 passed, 5 reused" in p.stdout and "tests/test_dyn.py::test_dyn" not in _reused(tmp_path), p.stdout
    (tests / "test_effect.py").write_text("import os\nos.environ['PKG_FLAG'] = '2'\n\ndef test_effect():\n    pass\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "7 passed" in p.stdout and _reused(tmp_path) == [], p.stdout


def test_a_fresh_run_that_fails_what_a_kept_pass_answers_for_says_so(tmp_path):
    """The nightly cold check's signal: a key that misses an input shows as a fresh failure a cached pass contradicts."""
    import json

    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("import os\n\ndef test_sibling():\n    assert not os.environ.get('PKG_BREAK')\n")
    assert _run_suite(root, tmp_path).returncode == 0
    p = _run_suite(root, tmp_path, "--fresh")                                   # nothing reused, yet it all passes again
    assert p.returncode == 0 and _reused(tmp_path) == [] and "CONTRADICTS" not in p.stdout
    out = tmp_path / "suites.json"                   # PKG_BREAK: an input no key covers (a variable outside VERITY_ and FLOCK_)
    p = _run_suite(root, tmp_path, "--fresh", "--json", str(out), extra={"PKG_BREAK": "1"})
    assert p.returncode == 1 and "CONTRADICTS a cached pass: (the suite), tests/test_sibling.py::test_sibling" in p.stdout, p.stdout
    assert json.loads(out.read_text())[0]["contradicts"] == ["(the suite)", "tests/test_sibling.py::test_sibling"]


def test_an_interrupted_run_keeps_what_finished(tmp_path):
    """A stop (check.py's SIGINT to the step's processes) mid-suite: the tests that passed before it are kept."""
    import signal

    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    gate = tmp_path / "gate"
    os.mkfifo(gate)
    for f in ("test_sibling.py", "test_uses.py", "test_shared.py"):
        (tests / f).unlink()
    (tests / "test_z_block.py").write_text(f"import signal\n\ndef test_block():\n    with open({str(gate)!r}, 'w') as f:\n"
                                           "        f.write('x')\n    signal.pause()\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(VERITY_TEST_CACHE=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([sys.executable, str(root / "tools" / "check" / "suites.py"), "--jobs", "1", "--logs", str(tmp_path / "logs")],
                         cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    with open(gate) as f:                                  # test_heavy's two tests have passed; test_block is running
        f.read()
    os.killpg(p.pid, signal.SIGINT)
    out = p.communicate()[0]
    assert p.returncode != 0 and "2 per-test passes kept" in out, out
    assert len(list((tmp_path / "cache" / "per-test" / "pkg").glob("pkg~tests~test_heavy.py--*.json"))) == 1
