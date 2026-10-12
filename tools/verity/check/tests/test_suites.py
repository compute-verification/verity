"""`suites.py`: suites are found and keyed from git's view of the tree, on a laptop's checkout and on the checkout `research run
--source` ships alike."""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _suites(monkeypatch, root: Path):
    spec = importlib.util.spec_from_file_location("suites_under_test", HERE / "suites.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)           # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", root)
    return mod


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_a_tree_is_discovered_and_keyed_from_gits_view_of_it(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (tmp_path / "pkg" / "tests").mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"1")
    _git(tmp_path, "init", "-q")
    S = _suites(monkeypatch, tmp_path)

    [suite] = S.discover()
    assert (suite.name, suite.path) == ("pkg", "pkg")
    assert S.metadata() == ["pkg/pyproject.toml", "pyproject.toml", "uv.lock"]
    before = S.object_ids(S.worktree(), ["pkg", "uv.lock"])
    assert before["uv.lock"] == "absent"

    (tmp_path / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"2")      # what the ignore rules cover is not source
    assert S.object_ids(S.worktree(), ["pkg", "uv.lock"]) == before
    (tmp_path / "pkg" / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")     # uncommitted, and it counts
    assert S.object_ids(S.worktree(), ["pkg"])["pkg"] != before["pkg"]


def test_a_tree_that_is_not_a_checkout_is_refused(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = []\n')
    S = _suites(monkeypatch, tmp_path)
    assert S.main(["--list"]) == 2 and "is not a git checkout" in capsys.readouterr().err


def test_the_tests_get_the_callers_pythonpath_entries_inside_the_tree_only(tmp_path, monkeypatch):
    """An `export PYTHONPATH=...` naming another checkout, left in a shell, would run this tree's tests against that checkout."""
    S = _suites(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/tmp/s1350/packages/verity/src", str(tmp_path / "src"), ""]))
    assert S.tests_pythonpath() == os.pathsep.join([str(S.CHILD), str(tmp_path / "src")])
    monkeypatch.delenv("PYTHONPATH")
    assert S.tests_pythonpath() == str(S.CHILD)


_HATCH = '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'


def _member(root: Path, path: str, name: str, packages: list[str] | None, build: bool = True) -> None:
    (root / path).mkdir(parents=True, exist_ok=True)
    text = f'[project]\nname = "{name}"\n' + (_HATCH if build else '')
    if packages is not None:
        text += f"\n[tool.hatch.build.targets.wheel]\npackages = {packages!r}\n".replace("'", '"')
    (root / path / "pyproject.toml").write_text(text)
    for p in packages or []:
        (root / path / p).mkdir(parents=True)
        (root / path / p / "__init__.py").write_text("")


def test_a_python_that_imports_the_trees_packages_from_another_checkout_is_named(tmp_path, monkeypatch):
    tree, other = tmp_path / "tree", tmp_path / "other"
    for root in (tree, other):
        _member(root, "protocols/pous", "verity-pous", ["verity_pous"])
        _member(root, "tools/check", "check", None, build=False)
    S = _suites(monkeypatch, tree)
    pkgs = {"verity-pous": "protocols/pous", "check": "tools/check"}
    env = dict(os.environ, PYTHONPATH=str(other / "protocols" / "pous"))
    assert S.foreign(pkgs, env) == f"verity_pous from {other / 'protocols' / 'pous' / 'verity_pous' / '__init__.py'}"
    env["PYTHONPATH"] = str(tree / "protocols" / "pous")
    assert S.foreign(pkgs, env) is None


def test_every_declared_package_is_probed_whatever_its_name_and_one_that_wont_import_is_named(tmp_path, monkeypatch):
    """The names come from each member's build configuration, not a `verity` prefix: a package moved to `catalog/` is probed like
    `verity_pous`, and one the Python can't find at all is named rather than passed over."""
    tree, other = tmp_path / "tree", tmp_path / "other"
    for root in (tree, other):
        _member(root, "catalog", "catalog", ["src/catalog"])
        _member(root, "archive", "archive", ["python/archive_old"])
    S = _suites(monkeypatch, tree)
    pkgs = {"catalog": "catalog", "archive": "archive"}
    assert S.importable(pkgs) == {"archive_old": "archive", "catalog": "catalog"}
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(other / "catalog" / "src"), str(tree / "archive" / "python")]))
    assert S.foreign(pkgs, env) == f"catalog from {other / 'catalog' / 'src' / 'catalog' / '__init__.py'}"
    env["PYTHONPATH"] = str(tree / "catalog" / "src")
    assert S.foreign(pkgs, env) == "archive_old from nowhere (not importable)"


def test_a_member_that_builds_but_declares_no_package_is_named(tmp_path, monkeypatch):
    _member(tmp_path, "odd", "odd", None)
    S = _suites(monkeypatch, tmp_path)
    assert S.foreign({"odd": "odd"}, dict(os.environ)) == "odd's package from nowhere (its build configuration declares none)"
    (tmp_path / "odd" / "src" / "odd").mkdir(parents=True)
    (tmp_path / "odd" / "src" / "odd" / "__init__.py").write_text("")       # hatch's default layout
    assert S.importable({"odd": "odd"}) == {"odd": "odd"}


def test_suites_refuse_a_python_that_imports_the_trees_packages_from_elsewhere(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg"]\n')
    (tmp_path / "pkg" / "tests").mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _git(tmp_path, "init", "-q")
    S = _suites(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", f"/tmp/explorer/a{os.pathsep}{tmp_path / 'pkg'}")
    probed = []
    monkeypatch.setattr(S, "foreign", lambda pkgs, env: probed.append(env["PYTHONPATH"]) or "verity_pous from /tmp/s1350/x.py")
    assert S.main([]) == 2
    assert "against another checkout's code (verity_pous from /tmp/s1350/x.py)" in capsys.readouterr().err
    assert probed == [os.pathsep.join([str(S.CHILD), str(tmp_path / "pkg")])]
    assert S.main(["--list"]) == 0 and len(probed) == 1


def test_a_declared_build_output_keys_the_suite(tmp_path, monkeypatch):
    """Git ignores build outputs, so a suite whose tests run or skip on one declares it (`[tool.verity.tests] artifacts`), and
    its pass is reused only for the same build."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                                                     '[tool.verity.tests]\nartifacts = ["pkg/.lake/build/bin/tool", "pkg/.lake/cache"]\n')
    _git(tmp_path, "init", "-q")
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


def test_jobs_default_to_no_more_test_processes_than_available_memory_holds(tmp_path, monkeypatch):
    S = _suites(monkeypatch, tmp_path)
    monkeypatch.setattr(S, "cpus", lambda: 4)
    meminfo = tmp_path / "meminfo"
    for gb, want in ((13, 1), (20, 2), (64, 4), (1, 1)):
        meminfo.write_text(f"MemTotal:       99999999 kB\nMemAvailable:   {gb << 20} kB\n")
        assert S.default_jobs(meminfo) == want
    assert S.default_jobs(tmp_path / "absent") == 4


def _workspace(root: Path) -> None:
    """Members `a` (on numpy, with a `plot` extra on matplotlib) and `b` (on nothing), and the workspace's dev group."""
    (root / "pyproject.toml").write_text('[project]\nname = "ws"\n\n[tool.uv.workspace]\nmembers = ["a", "b"]\n\n'
                                         '[dependency-groups]\ndev = ["pytest>=8"]\n')
    for name, deps in (("a", '["numpy>=1"]'), ("b", "[]")):
        (root / name / "tests").mkdir(parents=True)
        (root / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"\ndependencies = {deps}\n\n'
                                                    '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _lock(root, numpy="2.0", pytest="8.0")
    _git(root, "init", "-q")


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
    pkgs, lock, env, tree = S.members(), S.locked(), S.environment(), S.worktree()
    out = {}
    for s in S.discover():
        paths = S.inputs_of(s, pkgs)
        out[s.name] = S.key(s, S.input_ids(tree, sorted(set(paths) | set(S.RUNNER)), S.markdown(s)), env, [],
                            S.resolved(S.roots(s, pkgs), lock, s.path))
    return out


def _move(root: Path, old: str, new: str) -> None:
    """Move the member at ``old`` to ``new`` as a layout change does: the directory, the workspace's members and the lock."""
    (root / new).parent.mkdir(parents=True, exist_ok=True)
    _git(root, "mv", old, new)
    for f, was in (("pyproject.toml", f'"{old}"'), ("uv.lock", f'editable = "{old}"')):
        if (root / f).exists():
            (root / f).write_text((root / f).read_text().replace(was, was.replace(f'"{old}"', f'"{new}"')))


def test_a_suite_moved_whole_keys_as_it_did_and_any_change_misses(tmp_path, monkeypatch):
    """The key is content: a suite renamed with every file as it was hits, its uv.lock source included. Another depth misses (its
    tests count parents to the root), as do a moved dependency directory and one changed byte."""
    _workspace(tmp_path)
    (tmp_path / "b" / "pyproject.toml").write_text('[project]\nname = "b"\ndependencies = ["a"]\n\n[tool.uv.sources]\n'
                                                   'a = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "a" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    before = _keys(S)

    _move(tmp_path, "b", "c")
    assert S.locked()["b"]["entries"][0]["source"] == {"editable": "c"}
    assert _keys(S) == before                                       # renamed beside itself, nothing else changed
    _move(tmp_path, "c", "nested/b")
    assert _keys(S)["b"] != before["b"] and _keys(S)["a"] == before["a"]     # another depth
    _move(tmp_path, "nested/b", "b")
    assert _keys(S) == before

    _move(tmp_path, "a", "z")                                       # b's dependency, outside it, moves: b misses, a hits
    after = _keys(S)
    assert after["a"] == before["a"] and after["b"] != before["b"]

    (tmp_path / "z" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n ")      # one byte
    changed = _keys(S)
    assert changed["a"] != after["a"] and changed["b"] != after["b"]


def test_the_lock_keys_a_suite_s_own_paths_from_it_and_others_as_they_are(tmp_path, monkeypatch):
    S = _suites(monkeypatch, tmp_path)
    entry = [{"name": "b", "source": {"editable": "pkgs/b"}, "metadata": {"requires-dist": [
        {"name": "a", "editable": "pkgs/a"}, {"name": "t", "editable": "pkgs/b/tools"}, {"name": "x", "editable": "pkgs/bb"}]}}]
    assert S._relocated(entry, "pkgs/b") == [{"name": "b", "source": {"editable": "<suite>"}, "metadata": {"requires-dist": [
        {"name": "a", "editable": "pkgs/a"}, {"name": "t", "editable": "<suite>/tools"}, {"name": "x", "editable": "pkgs/bb"}]}}]
    assert S._relocated(entry, ".") == entry
    assert S._abs("pkgs/b", S._rel("pkgs/b", "pkgs/b/tools")) == "pkgs/b/tools" and S._rel("pkgs/b", "pkgs/bb") == "pkgs/bb"


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


def test_changed_selects_the_suites_a_change_touches_by_path_runner_or_lock(tmp_path, monkeypatch):
    """`--changed REV`, the quick tier `research queue ready` runs: a suite whose inputs, runner or resolution the change moves."""
    _workspace(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")

    def touched():
        return [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())]

    assert touched() == []
    (tmp_path / "a" / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")      # uncommitted, and it counts
    assert touched() == ["a"]
    (tmp_path / "a" / "tests" / "test_a.py").unlink()
    _lock(tmp_path, numpy="2.1", pytest="8.0")                     # only a resolves to numpy
    assert touched() == ["a"]
    _lock(tmp_path, numpy="2.0", pytest="8.1")                     # every suite runs with pytest
    assert touched() == ["a", "b"]
    _lock(tmp_path, numpy="2.0", pytest="8.0")
    assert touched() == []
    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    keys = _keys(S)
    (tmp_path / "tools" / "verity" / "check" / "check.py").write_text("# check, which runs the runner\n")
    assert touched() == [] and _keys(S) == keys                     # check's own code decides no other suite's verdict
    (tmp_path / "tools" / "verity" / "check" / "suites.py").write_text("# the runner\n")
    assert touched() == ["a", "b"] and all(_keys(S)[n] != keys[n] for n in keys)


def test_markdown_rekeys_and_selects_only_the_suites_that_list_markdown(tmp_path, monkeypatch):
    """#1594's 13 `.md` files picked 35 suites. A directory keys without its `*.md` unless its suite lists a `.md`, whose guard
    then lets it read Markdown (test_guard.py holds the others to reading none)."""
    _workspace(tmp_path)
    (tmp_path / "a" / "README.md").write_text("prose\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")

    def touched():
        return [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())]

    keys = _keys(S)
    (tmp_path / "a" / "README.md").write_text("other prose\n")
    (tmp_path / "a" / "GUIDE.md").write_text("new prose\n")
    assert touched() == [] and _keys(S) == keys
    (tmp_path / "a" / "data.txt").write_text("not prose\n")
    assert touched() == ["a"] and _keys(S)["a"] != keys["a"]
    (tmp_path / "a" / "data.txt").unlink()
    (tmp_path / "b" / "pyproject.toml").write_text((tmp_path / "b" / "pyproject.toml").read_text()
                                                   + '\n[tool.verity.tests]\ninputs = ["a/README.md"]\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "b reads a's README")
    base = _git(tmp_path, "rev-parse", "HEAD")
    keys = _keys(S)
    (tmp_path / "a" / "README.md").write_text("third prose\n")
    assert touched() == ["b"]                                       # b reads a's README: its edit reaches b alone
    assert _keys(S)["a"] == keys["a"] and _keys(S)["b"] != keys["b"]


def test_changed_selects_the_suite_a_file_is_moved_out_of(tmp_path, monkeypatch):
    """git reports a rename by its destination alone unless asked not to, which would leave out the suite the file left."""
    _workspace(tmp_path)
    (tmp_path / "a" / "mod.py").write_text("".join(f"X{i} = {i}\n" for i in range(20)))
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "mv", "a/mod.py", "b/mod.py")
    assert [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())] == ["a", "b"]


def test_a_dependencys_dependency_group_is_not_in_its_dependents_closure(tmp_path, monkeypatch):
    """`b`'s dev group (its own tests' tooling, `a`) keys `b`'s suite but not `c`'s, which depends on `b`: code importing `b`
    never reads `a`, so a change there need not run `c`."""
    _workspace(tmp_path)
    (tmp_path / "pyproject.toml").write_text((tmp_path / "pyproject.toml").read_text().replace('["a", "b"]', '["a", "b", "c"]'))
    (tmp_path / "b" / "pyproject.toml").write_text('[project]\nname = "b"\n\n[dependency-groups]\ndev = ["a"]\n\n[tool.uv.sources]\n'
                                                   'a = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "c" / "tests").mkdir(parents=True)
    (tmp_path / "c" / "pyproject.toml").write_text('[project]\nname = "c"\ndependencies = ["b"]\n\n[tool.uv.sources]\n'
                                                   'b = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    by = {s.name: s for s in S.discover()}
    assert S.inputs_of(by["b"], pkgs) == ["a", "b"] and S.inputs_of(by["c"], pkgs) == ["b", "c"]
    (tmp_path / "a" / "mod.py").write_text("X = 1\n")
    assert [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())] == ["a", "b"]


def test_a_dependency_that_lists_exports_is_those_paths_to_its_dependents(tmp_path, monkeypatch):
    """`a` lists `a/lib.py` under `exports`: `b`, which depends on it, keys on that file and not on the rest of `a`, so a change
    to `a/cli.py` runs `a`'s suite alone."""
    _workspace(tmp_path)
    text = (tmp_path / "a" / "pyproject.toml").read_text()
    (tmp_path / "a" / "pyproject.toml").write_text(text + '\n[tool.verity.tests]\nexports = ["a/lib.py"]\n')
    (tmp_path / "b" / "pyproject.toml").write_text('[project]\nname = "b"\ndependencies = ["a"]\n\n[tool.uv.sources]\n'
                                                   'a = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    for f in ("lib.py", "cli.py"):
        (tmp_path / "a" / f).write_text("X = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    by = {s.name: s for s in S.discover()}
    assert S.inputs_of(by["a"], pkgs) == ["a"] and S.inputs_of(by["b"], pkgs) == ["a/lib.py", "b"]
    (tmp_path / "a" / "cli.py").write_text("X = 2\n")
    assert [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())] == ["a"]
    (tmp_path / "a" / "lib.py").write_text("X = 2\n")
    assert [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())] == ["a", "b"]


def test_a_declared_package_is_its_directory_without_its_dependencies(tmp_path, monkeypatch):
    """A test that reads a declared package's dependency fails the guard, which allows only `inputs_of`, so a change there
    need not run the suite that declared the package."""
    _workspace(tmp_path)
    (tmp_path / "pyproject.toml").write_text((tmp_path / "pyproject.toml").read_text().replace('["a", "b"]', '["a", "b", "c"]'))
    (tmp_path / "b" / "pyproject.toml").write_text('[project]\nname = "b"\ndependencies = ["a"]\n\n[tool.uv.sources]\n'
                                                   'a = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "c" / "tests").mkdir(parents=True)
    (tmp_path / "c" / "pyproject.toml").write_text('[project]\nname = "c"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                                                   '[tool.verity.tests]\ninputs = ["b"]\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    by = {s.name: s for s in S.discover()}
    assert S.inputs_of(by["b"], pkgs) == ["a", "b"] and S.inputs_of(by["c"], pkgs) == ["b", "c"]

    def touched():
        return [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())]

    (tmp_path / "a" / "mod.py").write_text("X = 1\n")
    assert touched() == ["a", "b"]
    (tmp_path / "b" / "mod.py").write_text("Y = 1\n")
    assert touched() == ["a", "b", "c"]


def test_an_inputs_only_suite_keys_on_what_it_declares_of_its_dependencies(tmp_path, monkeypatch):
    """`inputs_only = true`: a suite depends on `a` for its package but its tests read only `a/used.py`, so a change elsewhere in
    `a` need not run it; the guard allows only `inputs_of`, so a read of the rest of `a` fails it."""
    _workspace(tmp_path)
    (tmp_path / "b" / "pyproject.toml").write_text('[project]\nname = "b"\ndependencies = ["a"]\n\n[tool.uv.sources]\n'
                                                   'a = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n'
                                                   '[tool.verity.tests]\ninputs = ["a/used.py"]\ninputs_only = true\n')
    (tmp_path / "a" / "used.py").write_text("X = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    by = {s.name: s for s in S.discover()}
    assert by["b"].inputs_only and not by["a"].inputs_only
    assert S.inputs_of(by["b"], pkgs) == ["a/used.py", "b"]

    def touched():
        return [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())]

    (tmp_path / "a" / "other.py").write_text("Y = 1\n")
    assert touched() == ["a"]
    (tmp_path / "a" / "used.py").write_text("X = 2\n")
    assert touched() == ["a", "b"]


def test_the_roots_own_distribution_is_a_member_and_its_dependents_key_on_the_packages_it_ships(tmp_path, monkeypatch):
    """The root's pyproject.toml builds `core`, so uv makes it a member; a suite that depends on it keys on `core/`, not on the
    whole tree its directory holds."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "core"\n' + _HATCH + '\n[tool.hatch.build.targets.wheel]\n'
                                             'packages = ["core"]\n\n[tool.uv.workspace]\nmembers = ["p"]\n')
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "__init__.py").write_text("")
    (tmp_path / "p" / "tests").mkdir(parents=True)
    (tmp_path / "p" / "pyproject.toml").write_text('[project]\nname = "p"\ndependencies = ["core"]\n\n[tool.uv.sources]\n'
                                                   'core = { workspace = true }\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    S = _suites(monkeypatch, tmp_path)
    pkgs = S.members()
    assert pkgs == {"core": ".", "p": "p"} and S.importable(pkgs) == {"core": "."}
    assert S.inputs_of({s.name: s for s in S.discover()}["p"], pkgs) == ["core", "p"]


def test_a_pass_is_reused_only_while_what_it_used_beside_its_key_holds(tmp_path, monkeypatch):
    _workspace(tmp_path)
    S = _suites(monkeypatch, tmp_path)
    lock, tree = S.locked(), S.worktree()
    r = S.Result(S.discover()[1], "passed", watched=["a/pyproject.toml", "pyproject.toml"], distributions=["NumPy", "pytest"],
                 missing=["torch"])
    deps = S.resolved(["b", "pytest"], lock)
    hit = S.used(r, ["b"], deps, tree, lock)
    assert sorted(hit["watched"]) == ["a/pyproject.toml", "pyproject.toml"] and list(hit["imported"]) == ["numpy"]
    assert S.holds(hit, tree, lock, set())
    assert not S.holds(hit, tree, lock, {"torch"})                                   # an import that found nothing now would
    (tmp_path / "pyproject.toml").write_text((tmp_path / "pyproject.toml").read_text() + "\n# read by b's tests\n")
    assert not S.holds(hit, S.worktree(), lock, set())
    _lock(tmp_path, numpy="2.1", pytest="8.0")
    assert not S.holds({**hit, "watched": {}}, tree, S.locked(), set())             # numpy, imported outside b's key, moved


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
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    tree = subprocess.run(["git", "-C", str(tmp_path), "write-tree"], capture_output=True, text=True, check=True).stdout.strip()
    assert S.fetched_fixtures(tree, {"pkg"}) == {"pkg/fx/big.proof": "art:" + "b" * 64}     # the tracked entry is keyed by git
    assert S.fetched_fixtures(tree, {"."}) == {"pkg/fx/big.proof": "art:" + "b" * 64, "other/big.bin": "art:" + "c" * 64}


def _shipped_workspace(root: Path, *, absent: bool) -> None:
    """A tree as `research run --source` ships one: a checkout of the commit with READY.json beside it, git's view leaving it out;
    this runner under tools/check, one package suite whose test reads a registered fixture; with `absent`, the registry also names
    a fetched entry the tree lacks."""
    import hashlib
    import json
    import shutil

    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
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
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "shipped")
    with open(root / ".git" / "info" / "exclude", "a") as f:
        f.write("/READY.json\n")
    (root / "READY.json").write_text("{}\n")


def _run_shipped(root: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(RESEARCH_STORE=str(tmp_path / "pod-store"), RESEARCH_STORE_CONFIG=str(tmp_path / "absent.toml"),
               RESEARCH_PUBLIC_URL="", VERITY_TEST_CACHE=str(tmp_path / "cache"), VERITY_SKIP_STORE="1")
    return subprocess.run([sys.executable, str(root / "tools" / "verity" / "check" / "suites.py"), "--fresh", "--jobs", "1"], cwd=root, env=env,
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
    assert "Traceback" not in p.stderr and "cached  pkg" not in p.stdout, p.stderr     # a suite that never ran has no pass to cite
    assert "1 suites: 0 passed (0 cached), 1 failed" in p.stdout, p.stdout


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
    """A checkout with this runner and one parallel suite: `test_heavy` (on a helper), `test_sibling` (failing until fixed),
    `test_uses` (importing `test_shared`), and `test_shared`."""
    import shutil

    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
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
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "shipped")


def _run_suite(root: Path, tmp_path: Path, *args: str, extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_", "VERITY_"))}
    env.update(RESEARCH_STORE=str(tmp_path / "pod-store"), RESEARCH_STORE_CONFIG=str(tmp_path / "absent.toml"), RESEARCH_FIXTURES_URL="",
               VERITY_TEST_CACHE=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1")
    env.update(extra or {})
    return subprocess.run([sys.executable, str(root / "tools" / "verity" / "check" / "suites.py"), "--jobs", "1", "--logs", str(tmp_path / "logs"), *args],
                          cwd=root, env=env, capture_output=True, text=True)


def _env_workspace(root: Path, members: dict[str, str]) -> None:
    """A checkout with this runner and a member per (path: test module) in ``members``, each named its path."""
    import shutil

    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "pyproject.toml").write_text(f'[tool.uv.workspace]\nmembers = {list(members)!r}\n'.replace("'", '"'))
    for path, test in members.items():
        (root / path / "tests").mkdir(parents=True)
        (root / path / "pyproject.toml").write_text(f'[project]\nname = "{path}"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
        (root / path / "tests" / "test_it.py").write_text(test)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "shipped")


def test_the_tests_get_neither_the_runners_research_variables_nor_its_pycache_prefix(tmp_path):
    """Inside a `research run`, the run's RESEARCH_* and the machine's PYTHONPYCACHEPREFIX reached the tests, which check's steps
    never get: test_remote_cwd's inner run inherited the prefix and wrote no bytecode where it looked (r20261008-010104-2b69,
    passing with the prefix unset in r20261008-010441-bb7b). A VERITY_ switch still reaches them."""
    root = tmp_path / "tree"
    _env_workspace(root, {"pkg": "import os\n\ndef test_env():\n    assert os.environ.get('VERITY_SWITCH') == '1'\n"
                                 "    assert not [k for k in os.environ if k.startswith('RESEARCH_') or k == 'PYTHONPYCACHEPREFIX']\n"})
    p = _run_suite(root, tmp_path, "--fresh", extra={"RESEARCH_RUN_ID": "r1", "RESEARCH_RUN_DIR": str(tmp_path / "run"),
                                                     "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"), "VERITY_SWITCH": "1"})
    assert p.returncode == 0 and "passed  pkg" in p.stdout, p.stdout + p.stderr


def test_a_pass_earned_on_one_machine_is_reused_on_another_and_names_its_run(tmp_path):
    """Daniel's CI update (7 Oct): a suite's pass is reused on any machine while its inputs, the base image and the check code
    are as they were.  Machine a's passes travel as a verdict pack; on machine b a change to one suite's inputs runs that suite
    alone, and the other is listed as reused, citing the run that earned its pass."""
    root = tmp_path / "tree"
    _env_workspace(root, {"one": "def test_one():\n    pass\n", "two": "def test_two():\n    pass\n"})
    shutil.copy(HERE / "verdicts.py", root / "tools" / "verity" / "check" / "verdicts.py")
    a, b = {"VERITY_TEST_CACHE": str(tmp_path / "a")}, {"VERITY_TEST_CACHE": str(tmp_path / "b")}
    p = _run_suite(root, tmp_path, "--run", "r-a", extra=a)
    assert p.returncode == 0 and "passed  one" in p.stdout and "passed  two" in p.stdout, p.stdout + p.stderr
    pack = tmp_path / "verdicts.tar.gz"
    verdicts = [sys.executable, str(root / "tools" / "verity" / "check" / "verdicts.py")]
    subprocess.run([*verdicts, "export", str(pack)], env={**os.environ, **a, "CHECK_CACHE": str(tmp_path / "a-check")}, check=True)
    subprocess.run([*verdicts, "import", str(pack)], env={**os.environ, **b, "CHECK_CACHE": str(tmp_path / "b-check")}, check=True)

    (root / "one" / "tests" / "test_it.py").write_text("def test_one():\n    assert 1\n")
    p = _run_suite(root, tmp_path, "--run", "r-b", extra=b)
    assert p.returncode == 0 and "passed  one" in p.stdout and "1 suites" not in p.stdout, p.stdout + p.stderr
    assert re.search(r"cached  two +passed at \S+ \S+ \S+ in run r-a\n", p.stdout), p.stdout
    assert "2 suites: 2 passed (1 cached)" in p.stdout, p.stdout


def test_the_suites_on_the_lean_verifier_run_after_its_build_as_inside_check(tmp_path):
    """check's verity-flock and verity-lean-audit run after its lean-build step, with `lake` on the PATH; run on their own they
    skipped the Lean tests or found no `flock-verify`. So suites.py builds the verifier first, only for those suites, not when
    check says it's built (`--lean-built`), and a failed build fails them unrun."""
    root, log = tmp_path / "tree", tmp_path / "lake.log"
    _env_workspace(root, {"verity-flock": "import shutil\n\ndef test_lake():\n    assert shutil.which('lake')\n",
                          "other": "def test_other():\n    pass\n"})
    (root / "verity" / "core" / "service" / "flock").mkdir(parents=True)
    (root / "verity" / "core" / "service" / "flock" / "lakefile.toml").write_text('name = "flock"\n')
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "verifier")
    # ELAN puts $HOME/.elan/bin first on the PATH, so on a machine with elan its real lake would shadow one only on the PATH
    home = tmp_path / "home"
    bin_ = home / ".elan" / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "lake").write_text('#!/bin/sh\necho "$PWD $*" >> "$FAKE_LAKE_LOG"\nexit "${FAKE_LAKE_RC:-0}"\n')
    (bin_ / "lake").chmod(0o755)
    extra = {"HOME": str(home), "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "FAKE_LAKE_LOG": str(log)}

    p = _run_suite(root, tmp_path, "--fresh", "other", extra=extra)
    assert p.returncode == 0 and not log.exists(), p.stdout
    p = _run_suite(root, tmp_path, "--fresh", "--lean-built", "verity-flock", extra=extra)
    assert p.returncode == 0 and not log.exists(), p.stdout
    p = _run_suite(root, tmp_path, "--fresh", extra=extra)
    assert p.returncode == 0 and "passed  verity-flock" in p.stdout and "passed  other" in p.stdout, p.stdout
    assert log.read_text() == f"{root / 'verity' / 'core' / 'service' / 'flock'} build\n"

    p = _run_suite(root, tmp_path, "--fresh", extra={**extra, "FAKE_LAKE_RC": "1"})
    assert p.returncode == 1 and "failed  verity-flock" in p.stdout and "`lake build` in verity/core/service/flock exited 1" in p.stdout
    assert "passed  other" in p.stdout, p.stdout


def _reused(tmp_path: Path) -> list[str]:
    import json

    p = tmp_path / "logs" / "pkg.reuse.json"
    return sorted(json.loads(p.read_text())) if p.exists() else []


def test_a_killed_suite_is_reported_by_its_signal_and_what_it_left_in_its_group_is_killed(tmp_path):
    """A pytest the OOM killer ends wrote no report and left its forkserver running (network-accounting, #1202's tier): the
    suite's own process group takes what it left with it, and the signal and peak RSS are said."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    for f in tests.glob("test_*.py"):
        f.unlink()
    left = tmp_path / "left.pid"
    (tests / "test_killed.py").write_text(
        "import os, signal, subprocess, sys\n\n"
        "def test_killed():\n"
        "    p = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])\n"
        f"    open({str(left)!r}, 'w').write(str(p.pid))\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "killed")
    p = _run_suite(root, tmp_path, "--fresh")
    assert p.returncode == 1 and "killed by SIGKILL: pytest died by SIGKILL at a peak RSS of" in p.stdout, p.stdout
    assert "killed 1 processes its tests left in the suite's process group" in p.stdout, p.stdout
    stat = Path("/proc") / left.read_text() / "stat"
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] in ("Z", "X"), stat.read_text()


def test_a_suite_past_its_bound_is_killed_and_fails_as_timed_out(tmp_path):
    """45fe's test_suites.py waited 82 min on a FIFO nobody wrote until it was killed by hand (7 Oct). Here the test blocks the
    same way, so the run ends only by its bound, whenever that fires: its group is killed, it fails timed out, and no pass is
    kept."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    for f in tests.glob("test_*.py"):
        f.unlink()
    fifo = tmp_path / "never-written"
    os.mkfifo(fifo)
    (tests / "test_hangs.py").write_text(f"def test_hangs():\n    open({str(fifo)!r}).read()\n")
    pyproject = root / "pkg" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + "timeout_s = 0.5\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "hangs")
    p = _run_suite(root, tmp_path, "--fresh", "--timeouts", "--json", str(tmp_path / "out.json"))
    assert p.returncode == 1 and "TIMED OUT: still running after 0.5 s, so its process group was killed" in p.stdout, p.stdout
    import json

    [r] = json.loads((tmp_path / "out.json").read_text())
    assert r["status"] == "failed" and r["timed_out"] == 0.5
    assert [f.name for f in (tmp_path / "cache").glob("*.json")] == ["durations.json"], "no pass kept"


def test_oom_kills_reads_the_kernels_count(monkeypatch, tmp_path):
    S = _suites(monkeypatch, tmp_path)
    n = S.oom_kills()
    assert n is None if not Path("/proc/vmstat").exists() else isinstance(n, int) and n >= 0


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


def test_a_moved_suite_reuses_its_tests_passes_and_its_own_until_a_byte_changes(tmp_path):
    """A layout change moves a suite whole: its kept passes, the suite's and each test's, answer for it where it lands."""
    import json

    root = tmp_path / "tree"
    _per_test_workspace(root)
    p = _run_suite(root, tmp_path)
    assert p.returncode == 1 and "4 per-test passes kept" in p.stdout, p.stdout

    _move(root, "pkg", "moved")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 1 and "1 failed" in p.stdout and "4 reused" in p.stdout, p.stdout
    assert sorted(json.loads((tmp_path / "logs" / "moved.reuse.json").read_text())) == [
        "tests/test_heavy.py::test_heavy", "tests/test_heavy.py::test_heavy_too", "tests/test_shared.py::test_shared",
        "tests/test_uses.py::test_uses"]

    (root / "moved" / "tests" / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    assert _run_suite(root, tmp_path).returncode == 0
    _move(root, "moved", "again")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" in p.stdout and "1 suites: 1 passed (1 cached)" in p.stdout, p.stdout

    _move(root, "again", "deeper/pkg")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" not in p.stdout and "5 passed" in p.stdout, p.stdout     # another depth

    helpers = root / "deeper" / "pkg" / "tests" / "helpers.py"
    helpers.write_text(helpers.read_text() + " ")                   # one byte, which every module's key holds
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" not in p.stdout and "5 passed" in p.stdout, p.stdout


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
    doc = json.loads(out.read_text())[0]
    assert doc["contradicts"] == ["(the suite)", "tests/test_sibling.py::test_sibling"]
    assert doc["nondeterministic"] == ["tests/test_sibling.py::test_sibling"] and "NONDETERMINISTIC: tests/test_sibling.py" in p.stdout
    (tests / "test_shared.py").write_text("X = 1\n\n\ndef test_shared():\n    pass\n")      # the suite's key moves; the flaky test re-runs
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "tests/test_sibling.py::test_sibling" not in _reused(tmp_path), p.stdout


def test_a_suite_that_is_not_parallel_records_outcomes_but_reuses_none(tmp_path):
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    pyproject = root / "pkg" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("parallel = true", "parallel = false"))
    (tests / "test_sibling.py").write_text("import os\n\ndef test_sibling():\n    assert not os.environ.get('PKG_BREAK')\n")
    assert _run_suite(root, tmp_path).returncode == 0
    (tests / "test_shared.py").write_text("X = 1\n\n\ndef test_shared():\n    pass\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and _reused(tmp_path) == [] and "5 passed in" in p.stdout, p.stdout
    p = _run_suite(root, tmp_path, "--fresh", extra={"PKG_BREAK": "1"})
    assert p.returncode == 1 and "NONDETERMINISTIC: tests/test_sibling.py::test_sibling" in p.stdout, p.stdout


def test_explain_names_what_a_miss_changed(tmp_path):
    """Against the nearest kept pass: the input that moved and its files (git's trees), and per test module what it reaches."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    g = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "add", "-A"], check=True)
    subprocess.run([*g, "commit", "-qm", "tree"], check=True)
    assert _run_suite(root, tmp_path).returncode == 0
    p = _run_suite(root, tmp_path, "--explain", "pkg")
    assert p.returncode == 0 and "pkg: cached: a pass under this key" in p.stdout, p.stdout
    (tests / "helpers.py").write_text("VALUE = 1  # moved\n")
    p = _run_suite(root, tmp_path, "--explain", "pkg")
    assert "pkg: missed; the nearest pass kept here" in p.stdout and "input pkg changed" in p.stdout, p.stdout
    assert "pkg/tests/helpers.py" in p.stdout.split("input pkg changed")[1].split("\n")[0], p.stdout
    assert "per-test: 0 tests of 0 of 4 test modules would be reused" in p.stdout
    assert "pkg/tests/test_heavy.py: it read pkg/tests/helpers.py, which changed" in p.stdout, p.stdout      # its traced read set
    (tests / "helpers.py").write_text("VALUE = 1\n")
    (tests / "test_shared.py").write_text("X = 1\n\n\ndef test_shared():\n    pass\n")
    p = _run_suite(root, tmp_path, "--explain", "pkg")
    assert "per-test: 3 tests of 2 of 4 test modules would be reused" in p.stdout, p.stdout
    assert "pkg/tests/test_uses.py: it reaches pkg/tests/test_shared.py, which changed" in p.stdout, p.stdout


def test_each_suite_that_runs_says_what_forced_it(tmp_path):
    """`--json`'s `forced_by`, which check's reuse report reads: no pass to compare, then the input that moved against the
    nearest kept pass (whose run and commit it names), and `--fresh`; a cached suite has none."""
    import json

    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    _git(root, "commit", "-qam", "passing")
    out = tmp_path / "suites.json"

    def forced(*args: str) -> dict | None:
        p = _run_suite(root, tmp_path, "--json", str(out), *args)
        assert p.returncode == 0, p.stdout
        [doc] = json.loads(out.read_text())
        assert (doc["forced_by"] is None) == (doc["status"] == "cached")
        assert doc["forced_by"] is None or f"ran because: {doc['forced_by']['why'][0]}"[:60] in p.stdout, p.stdout
        return doc["forced_by"]

    assert forced()["why"] == ["no pass of it kept here records its key's components (passes kept before `--explain` don't)"]
    assert forced() is None
    (tests / "helpers.py").write_text("VALUE = 1  # moved\n")
    got = forced()
    assert got["why"][0].startswith("input pkg changed") and "pkg/tests/helpers.py" in got["why"][0], got
    assert got["nearest"]["commit"] == _git(root, "rev-parse", "HEAD") and set(got["nearest"]) == {"run", "commit", "when"}
    assert forced("--fresh") == {"why": ["--fresh"]}


def _until_written_or_exited(fifo: Path, p: subprocess.Popen) -> tuple[bool, bytes]:
    """(whether something was written to ``fifo`` before ``p`` closed its stdout, ``p``'s output read meanwhile), never waiting
    on a bare blocking ``open``: a runner that dies before the writer starts would leave that waiting forever."""
    import select

    fd, out, buf = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK), p.stdout.fileno(), []
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    poll.register(out, select.POLLIN)
    try:
        while True:
            for ready, _ in poll.poll():
                if ready == fd:
                    if os.read(fd, 1):
                        return True, b"".join(buf)
                    poll.unregister(fd)                    # a writer closed without writing
                elif not (chunk := os.read(out, 65536)):
                    return False, b"".join(buf)
                else:
                    buf.append(chunk)
    finally:
        os.close(fd)


@pytest.mark.parametrize("stop", ["SIGINT", "SIGTERM"])
def test_an_interrupted_run_keeps_what_finished(tmp_path, stop):
    """A stop mid-suite (check.py's SIGINT to the step's processes, or `research cancel`'s SIGTERM to the run's group): the
    tests that passed before it are kept, and the suite's `--basetemp` is gone."""
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
    (tmp := tmp_path / "tmp").mkdir()
    env.update(VERITY_TEST_CACHE=str(tmp_path / "cache"), PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp))
    p = subprocess.Popen([sys.executable, str(root / "tools" / "verity" / "check" / "suites.py"), "--jobs", "1", "--logs", str(tmp_path / "logs")],
                         cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    written, before = _until_written_or_exited(gate, p)    # test_heavy's two tests have passed; test_block is running
    assert written, f"the runner ended before test_block ran:\n{before.decode(errors='replace')}"
    assert len(list(tmp.glob("verity-suite-[0-9]*"))) == 1
    os.killpg(p.pid, getattr(signal, stop))
    out = (before + p.communicate()[0]).decode(errors="replace")
    assert p.returncode != 0 and "2 per-test passes kept" in out, out
    assert list(tmp.glob("verity-suite-[0-9]*")) == [], out
    assert len(list((tmp_path / "cache" / "per-test" / "pkg").glob("tests~test_heavy.py--*.json"))) == 1


_CHILD_TEST = '''import glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLINGS = os.path.join(HERE, "*.py")


def test_a_child_python_reads_its_siblings():
    subprocess.run([sys.executable, "-c", "import glob, sys\\nfor p in glob.glob(sys.argv[1]):\\n    open(p).read()", SIBLINGS], check=True)


def test_a_fork_reads_its_siblings_and_ends_without_flushing():
    pid = os.fork()
    if pid == 0:
        for p in glob.glob(SIBLINGS):
            open(p).read()
        os._exit(0)
    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0
'''


def test_what_a_child_or_a_fork_reads_joins_the_module_s_reads(tmp_path):
    """A child Python and a forked child that end with `os._exit` still count: their test module re-runs when a sibling it read
    changes, which a key narrowed from the module's source alone (it names no test module) would miss."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    (tests / "test_child.py").write_text(_CHILD_TEST)
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "7 per-test passes kept" in p.stdout, p.stdout
    (tests / "test_heavy.py").write_text((tests / "test_heavy.py").read_text() + "\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0, p.stdout
    reused = _reused(tmp_path)
    assert "tests/test_uses.py::test_uses" in reused and not [n for n in reused if n.startswith(("tests/test_child.py", "tests/test_heavy.py"))]


def _elsewhere(root: Path) -> str:
    """A repository file outside `pkg`'s inputs, as a test module's expression for its path."""
    (root / "elsewhere").mkdir()
    (root / "elsewhere" / "data.txt").write_text("not an input of pkg\n")
    return "os.path.join(os.path.dirname(__file__), '..', '..', 'elsewhere', 'data.txt')"


def test_what_a_child_reads_outside_the_inputs_fails_the_suite_and_names_the_test(tmp_path):
    """A child's read outside the inputs is its test's own: the suite fails and keeps no pass, as for the test's own read, until
    the file is declared. The other modules' per-test passes are kept, so the next run re-runs that module alone."""
    import json

    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    (tests / "test_outside.py").write_text(
        f"import os, subprocess, sys\n\ndef test_outside():\n    p = {_elsewhere(root)}\n"
        "    subprocess.run([sys.executable, '-c', 'import sys; open(sys.argv[1]).read()', p], check=True)\n")
    out, cache = tmp_path / "suites.json", tmp_path / "cache"
    p = _run_suite(root, tmp_path, "--json", str(out))
    assert p.returncode == 1 and "failed  pkg" in p.stdout and "6 passed" in p.stdout, p.stdout
    assert "elsewhere/data.txt: read by a process that tests/test_outside.py::test_outside started" in p.stdout, p.stdout
    assert "list each file under [tool.verity.tests] inputs in pkg/pyproject.toml" in p.stdout, p.stdout
    assert "log:" not in p.stdout, p.stdout                                     # pytest passed: no log tail to read
    assert json.loads(out.read_text())[0]["child_outside"] == {"pkg/tests/test_outside.py": ["elsewhere/data.txt"]}
    assert not [f for f in cache.glob("*.json") if json.loads(f.read_text()).get("suite") == "pkg"]
    assert not list((cache / "per-test" / "pkg").glob("tests~test_outside.py--*.json"))
    assert list((cache / "per-test" / "pkg").glob("tests~test_heavy.py--*.json"))

    p = _run_suite(root, tmp_path)                                              # nothing changed: no pass to reuse
    assert p.returncode == 1 and "1 passed, 5 reused" in p.stdout and "elsewhere/data.txt: read by" in p.stdout, p.stdout

    pyproject = root / "pkg" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("parallel = true", 'parallel = true\ninputs = ["elsewhere"]'))
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "passed  pkg" in p.stdout and "outside its inputs" not in p.stdout, p.stdout
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" in p.stdout, p.stdout


def test_every_suite_may_read_the_fixture_registry_and_a_pass_that_read_it_holds_while_it_is_unchanged(tmp_path):
    """`research data fetch` on an empty local store indexes the remote's manifest, which reads `fixtures/artifacts.json` for
    repo replicas: in the test's process or a child's, that read fails no suite, and the test's own read is watched."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    (root / "fixtures").mkdir()
    (root / "fixtures" / "artifacts.json").write_text("{}\n")
    (tests / "test_registry.py").write_text(
        "import os, subprocess, sys\n\ndef test_registry():\n"
        "    p = os.path.join(os.path.dirname(__file__), '..', '..', 'fixtures', 'artifacts.json')\n"
        "    open(p).read()\n"
        "    subprocess.run([sys.executable, '-c', 'import sys; open(sys.argv[1]).read()', p], check=True)\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "registry")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "passed  pkg" in p.stdout and "outside its inputs" not in p.stdout, p.stdout
    assert "cached  pkg" in _run_suite(root, tmp_path).stdout
    (root / "fixtures" / "artifacts.json").write_text('{"art:x": "elsewhere/big.bin"}\n')
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "passed  pkg" in p.stdout, p.stdout


def test_a_fork_a_collection_or_a_child_whose_tracer_fails_fails_the_suite_under_xdist(tmp_path):
    """In xdist workers too: a forked child's read (it ends through `os._exit`), one made while a module is collected, and a child
    Python whose tracer can't start, so that nothing vouches for what it read."""
    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    data = _elsewhere(root)
    (tests / "test_fork.py").write_text(
        f"import os\n\ndef test_fork():\n    pid = os.fork()\n    if pid == 0:\n        open({data}).read()\n        os._exit(0)\n"
        "    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 0\n")
    (tests / "test_collected.py").write_text(
        f"import os, subprocess, sys\n\nsubprocess.run([sys.executable, '-c', 'import sys; open(sys.argv[1]).read()', {data}], check=True)\n\n"
        "def test_collected():\n    pass\n")
    (tests / "test_blind.py").write_text(
        "import os, subprocess, sys\n\ndef test_blind():\n"
        "    subprocess.run([sys.executable, '-c', 'pass'], env={**os.environ, 'VERITY_SUITE_ALLOW': 'not json'}, check=True)\n")
    p = _run_suite(root, tmp_path, "--jobs", "2")
    assert p.returncode == 1 and "on 2 workers" in p.stdout and "8 passed" in p.stdout, p.stdout
    assert ("elsewhere/data.txt: read by a process that tests/test_collected.py (on import) and tests/test_fork.py::test_fork "
            "started") in p.stdout, p.stdout
    assert "a Python that tests/test_blind.py::test_blind started could not start its read guard (JSONDecodeError: " in p.stdout, p.stdout


def test_the_child_tracer_hands_on_to_the_interpreter_s_own_sitecustomize(tmp_path):
    """It shadows the system's `sitecustomize` on PYTHONPATH, so it loads that one after itself."""
    (tmp_path / "sys").mkdir()
    (tmp_path / "sys" / "sitecustomize.py").write_text("import builtins\nbuiltins.SYSTEM_SITECUSTOMIZE = True\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("VERITY_")}
    env["PYTHONPATH"] = os.pathsep.join([str(HERE / "guard" / "child"), str(tmp_path / "sys")])
    p = subprocess.run([sys.executable, "-c", "import builtins, sitecustomize; print(builtins.SYSTEM_SITECUSTOMIZE, sitecustomize.__file__)"],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("True ") and "guard/child/sitecustomize.py" in p.stdout, p.stdout + p.stderr


def _traced_workspace(root: Path) -> None:
    """A checkout with this runner and a parallel suite `pkg` that depends on a package `lib`: `test_a` imports `lib/used.py`,
    `test_b` imports nothing, `test_spawn` starts a program that isn't Python, and `test_probe` tries to open `lib/maybe.txt`."""
    import shutil

    (root / "tools" / "verity" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "verity" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "verity" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pkg", "lib"]\n')
    (root / "lib").mkdir()
    (root / "lib" / "pyproject.toml").write_text('[project]\nname = "lib"\n')
    (root / "lib" / "used.py").write_text("VALUE = 1\n")
    (root / "lib" / "unused.py").write_text("OTHER = 2\n")
    (root / "pkg" / "tests").mkdir(parents=True)
    (root / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\ndependencies = ["lib"]\n\n[tool.pytest.ini_options]\n'
                                                 'testpaths = ["tests"]\n\n[tool.verity.tests]\nparallel = true\n')
    for name, text in {"conftest.py": "import pathlib, sys\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'lib'))\n",
                       "test_a.py": "from used import VALUE\n\ndef test_a():\n    assert VALUE == 1\n",
                       "test_b.py": "def test_b():\n    pass\n",
                       "test_spawn.py": "import subprocess\n\ndef test_spawn():\n    subprocess.run(['sh', '-c', ':'], check=True)\n",
                       "test_probe.py": "import pathlib\n\ndef test_probe():\n    p = pathlib.Path(__file__).resolve().parents[2] / 'lib' / 'maybe.txt'\n"
                                        "    try:\n        p.read_text()\n    except FileNotFoundError:\n        pass\n"}.items():
        (root / "pkg" / "tests" / name).write_text(text)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "shipped")


def test_a_traced_pass_is_reused_until_a_file_its_run_read_changes(tmp_path):
    """The suite depends on `lib` whole, so any edit there misses the suite's key. A module's traced pass misses only when a file
    or listing its run read moved, or an absent file it looked for appears; a module whose test started a program no tracer sees
    into is kept on the suite's inputs whole."""
    import json

    root, lib = tmp_path / "tree", tmp_path / "tree" / "lib"
    _traced_workspace(root)
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "4 per-test passes kept" in p.stdout, p.stdout
    kept = {json.loads(f.read_text())["module"]: json.loads(f.read_text()) for f in (tmp_path / "cache" / "per-test" / "pkg").glob("*.json")}
    assert sorted(m for m, e in kept.items() if e.get("traced")) == ["pkg/tests/test_a.py", "pkg/tests/test_b.py", "pkg/tests/test_probe.py"]
    reads = json.loads((tmp_path / "cache" / "per-test" / "pkg" / "traced" / f"{kept['pkg/tests/test_a.py']['traced']}.json").read_text())["reads"]
    assert "F\tlib/used.py" in reads and "F\tlib/unused.py" not in reads and reads["F\tlib/maybe.txt"] == "absent", sorted(reads)
    assert not [i for i in reads if i.startswith("F\tpkg/tests/test_")]            # the suite's own test modules: `keep` holds them

    (lib / "unused.py").write_text("OTHER = 3\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "1 passed, 3 reused" in p.stdout, p.stdout
    assert _reused(tmp_path) == ["tests/test_a.py::test_a", "tests/test_b.py::test_b", "tests/test_probe.py::test_probe"]

    (lib / "used.py").write_text("VALUE = 1  # read by test_a\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "4 passed in" in p.stdout and _reused(tmp_path) == [], p.stdout

    (lib / "maybe.txt").write_text("now it's here\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "4 passed in" in p.stdout, p.stdout


def test_a_suite_that_is_not_parallel_reuses_its_whole_traced_pass_until_a_file_its_run_read_changes(tmp_path):
    """Its tests may depend on their order, so it reuses no test's pass, but a run of all of them that the guard traced answers
    for the suite while every read holds; a run in which a test started a program no tracer sees into keeps no such pass."""
    import json

    root, lib, tests = tmp_path / "tree", tmp_path / "tree" / "lib", tmp_path / "tree" / "pkg" / "tests"
    _traced_workspace(root)
    pyproject = root / "pkg" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("parallel = true", "parallel = false"))
    spawn = (tests / "test_spawn.py").read_text()
    (tests / "test_spawn.py").unlink()
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "3 passed in" in p.stdout, p.stdout
    [whole] = (tmp_path / "cache" / "per-test" / "pkg").glob("(suite)--*.json")
    reads = json.loads((whole.parent / "traced" / f"{json.loads(whole.read_text())['traced']}.json").read_text())["reads"]
    assert "F\tlib/used.py" in reads and "F\tlib/unused.py" not in reads, sorted(reads)

    (lib / "unused.py").write_text("OTHER = 3\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" in p.stdout and "traced: every file its run read" in p.stdout, p.stdout
    assert "pkg: cached: a traced pass of the whole suite" in _run_suite(root, tmp_path, "--explain", "pkg").stdout

    (lib / "used.py").write_text("VALUE = 1  # read by test_a\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "3 passed in" in p.stdout and _reused(tmp_path) == [], p.stdout
    (tests / "test_b.py").write_text("def test_b():\n    assert True\n")          # a test module: in the key, read or not
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "3 passed in" in p.stdout, p.stdout

    (tests / "test_spawn.py").write_text(spawn)
    assert _run_suite(root, tmp_path).returncode == 0
    (lib / "unused.py").write_text("OTHER = 4\n")
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "4 passed in" in p.stdout, p.stdout


def test_the_tracer_names_what_a_key_narrowed_to_the_tests_reads_must_hold(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("reads_under_test", HERE / "guard" / "child" / "verity_suite_reads.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    child = str(HERE / "guard" / "child")
    env = {"VERITY_SUITE_ROOT": str(tmp_path), "VERITY_SUITE_READS": str(tmp_path / "reads"), "VERITY_SUITE_MODULE": "pkg/tests/test_x.py",
           "PYTHONPATH": child, "VERITY_SUITE_ALLOW": '["lib/used.py"]'}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = R.Reads()
    assert r.traced(str(tmp_path / "lib" / "__pycache__" / "used.cpython-314.pyc")) == "lib/used.py"
    assert r.traced(str(tmp_path / "lib" / "__pycache__" / "used.cpython-314.pyc.140214632246000")) == "lib/used.py"
    assert r.traced(str(tmp_path / "__pycache__" / "top.cpython-314.opt-1.pyc")) == "top.py"
    for cached in ("used.cpython-314.pyc", "used.cpython-314.pyc.140214632246000"):    # an input file's import, and its write
        assert r.classify(str(tmp_path / "lib" / "__pycache__" / cached)) is None
    assert r.classify(str(tmp_path / "lib" / "__pycache__" / "other.cpython-314.pyc")) == ("O", "lib/__pycache__/other.cpython-314.pyc")
    assert r.traced(str(tmp_path / "lib" / "data.json")) == "lib/data.json" and r.traced("/elsewhere/x") is None
    assert r.traced(str(Path(child) / "sitecustomize.py")) is None
    assert r.trace("open", (str(tmp_path / "lib" / "a.py"), "r", 0)) == [("F", "lib/a.py")]
    assert r.trace("os.scandir", (str(tmp_path / "lib"),)) == [("L", "lib")]
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, None)) == []
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-I", "-c", "pass"], None, None)) == [("X", sys.executable)]
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, {"PATH": "/bin"}), ) == [("X", sys.executable)]
    assert r.trace("subprocess.Popen", (None, ["git", "status"], None, None)) == [("X", "git")]
    assert r.trace("subprocess.Popen", (None, ["uv", "run", "python"], None, None)) == [("X", "uv")]
    assert r.trace("os.system", ("ls",)) == [("X", "os.system")]
    assert r.trace("subprocess.Popen", (None, ["uname", "-p"], None, None)) == []       # a machine probe reads no repository file
    monkeypatch.delenv("VERITY_SUITE_MODULE")                  # outside a test module sitecustomize doesn't trace
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, None)) == [("X", sys.executable)]


def test_a_declared_build_output_is_traced_by_the_repository_paths_it_is_given(tmp_path, monkeypatch):
    """A Lean executable the suite declares under `artifacts` reads what its arguments name, so the tests that start it keep
    traced per-test keys: its run is the files it was given (a file with those beside it, a directory whole), outside the inputs
    a failure; `lake build` of declared outputs reads nothing the key doesn't hold. Any other program is still untraced."""
    spec = importlib.util.spec_from_file_location("reads_under_test", HERE / "guard" / "child" / "verity_suite_reads.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    pkg, bin_ = tmp_path / "lean", tmp_path / "lean" / ".lake" / "build" / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "verify").write_text("")
    (bin_ / "other").write_text("")
    for f in ("fx/a.bin", "fx/b.bin", "fx/sub/c.json", "out/x.bin"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    monkeypatch.setenv("VERITY_SUITE_ROOT", str(tmp_path))
    monkeypatch.setenv("VERITY_SUITE_ALLOW", '["fx"]')
    monkeypatch.setenv("VERITY_SUITE_PROGRAMS", '["lean/.lake/build/bin/verify"]')
    r = R.Reads()
    exe = str(bin_ / "verify")
    pop = lambda argv, cwd=None: r.trace("subprocess.Popen", (None, argv, cwd, None))     # noqa: E731
    assert pop([exe, "check", "--circuit", str(tmp_path / "fx" / "a.bin"), "/elsewhere/tmp/p.bin"]) == [("F", "fx/a.bin"), ("F", "fx/b.bin")]
    assert pop([exe, "--dir=fx/sub"], cwd=tmp_path) == [("F", "fx/sub/c.json")]
    assert pop([exe, "out/x.bin"], cwd=tmp_path) == [("F", "out/x.bin"), ("O", "out/x.bin")]
    monkeypatch.chdir(tmp_path)
    assert r.trace("os.posix_spawn", (exe, [exe, "fx"], None)) == [("F", "fx/a.bin"), ("F", "fx/b.bin"), ("F", "fx/sub/c.json")]
    assert pop(["lake", "build", "verify"], cwd=pkg) == []
    assert pop(["lake", "build", "verify", "other"], cwd=pkg) == [("X", "lake")]
    assert pop(["lake", "env", "lean", "x.lean"], cwd=pkg) == [("X", "lake")]
    assert pop([str(bin_ / "other"), "fx"]) == [("X", str(bin_ / "other"))]
    assert pop(["verify", "fx"], cwd=bin_) == [("X", "verify")]                         # found on PATH: not the declared one


def _guard(monkeypatch):
    """The guard module, loaded without its audit hook (which it installs only when `suites.py` runs it)."""
    monkeypatch.delenv("VERITY_SUITE_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_passed_test_s_seconds_hold_its_setup_so_a_module_fixture_s_first_test_counts_as_heavy(monkeypatch):
    """Ordering by call alone started `test_lean_witness`'s tests late behind 1,270-1,475 s of their `f1` setup per worker
    (proofs, 12 Oct); the teardown is not counted, and a failed phase still drops the test."""
    from types import SimpleNamespace as R

    G = _guard(monkeypatch)
    for node, phases in (("t::fixture_first", [("setup", "passed", 1200.0), ("call", "passed", 1.5), ("teardown", "passed", 9.0)]),
                         ("t::plain", [("setup", "passed", 0.25), ("call", "passed", 3.0), ("teardown", "passed", 0.0)]),
                         ("t::bad", [("setup", "passed", 5.0), ("call", "failed", 1.0), ("teardown", "passed", 0.0)])):
        for when, outcome, s in phases:
            G.pytest_runtest_logreport(R(nodeid=node, when=when, outcome=outcome, duration=s, keywords={"slow": 1}))
    assert G._outcomes == {"t::fixture_first": 1201.5, "t::plain": 3.25, "t::bad": None}


def test_each_heavy_test_heads_its_own_worker_s_share_under_worksteal(monkeypatch):
    """The placement matches the split of the pinned pytest-xdist's `worksteal`: each worker is sent a contiguous share at
    once, and runs it from the head, while steals take from the tail."""
    from xdist.scheduler.worksteal import WorkStealingScheduling

    G = _guard(monkeypatch)
    ids = [f"tests/test_{i:02d}.py::test" for i in range(23)]
    heavy = [ids[22], ids[17], ids[9], "tests/test_gone.py::test"]

    class Node:
        shutting_down = False

        def __init__(self, i):
            self.sent, self.gateway = [], type("Gateway", (), {"id": f"gw{i}"})

        def send_runtest_some(self, indices):
            self.sent += indices

        def shutdown(self):
            pass

    for workers in (2, 3, 5, 30):
        order = [ids[i] for i in G.heavy_first(ids, heavy, workers)]
        assert sorted(order) == sorted(ids)
        assert [n for n in order if n not in heavy] == [n for n in ids if n not in heavy]
        sched = WorkStealingScheduling.__new__(WorkStealingScheduling)
        sched.numnodes, sched.node2collection, sched.node2pending, sched.pending = workers, {}, {}, []
        sched.collection, sched.log, sched.config, sched.steal_requested_from_node = None, lambda *a: None, None, None
        nodes = [Node(i) for i in range(workers)]
        for n in nodes:
            sched.add_node(n)
            sched.add_node_collection(n, order)
        sched.schedule()
        heads = [order[n.sent[0]] for n in nodes if n.sent]
        k = min(3, len(heads))
        assert heads[:k] == heavy[:k], (workers, heads)


def test_the_longest_tests_are_read_from_the_newest_kept_passes(tmp_path, monkeypatch):
    import json

    S = _suites(monkeypatch, tmp_path)
    monkeypatch.setattr(S, "CACHE", tmp_path / "cache")
    d = tmp_path / "cache" / "per-test" / "pkg"
    d.mkdir(parents=True)
    (d / "a--old.json").write_text(json.dumps({"when": "2026-09-29 10:00", "passed": {"t::a": 90.0, "t::b": 1.0}}))
    (d / "a--new.json").write_text(json.dumps({"when": "2026-09-30 10:00", "passed": {"t::a": 3.0}}))
    (d / "b--one.json").write_text(json.dumps({"when": "2026-09-30 09:00", "passed": {"t::c": 5.0, "t::d": 5.0}}))
    suite = S.Suite("pkg", "pkg", ("pkg",), ())
    assert S.longest(suite, 3) == ["t::c", "t::d", "t::a"]
    assert S.longest(S.Suite("other", "other", ("other",), ()), 3) == []


def test_a_parallel_suite_starts_the_test_its_kept_pass_says_is_longest(tmp_path):
    import json

    root, tests = tmp_path / "tree", tmp_path / "tree" / "pkg" / "tests"
    _per_test_workspace(root)
    (tests / "test_sibling.py").write_text("def test_sibling():\n    assert True\n")
    (tests / "test_z.py").write_text("import os\n\ndef test_z():\n    pass\n\ndef test_zz():\n    pass\n")
    for f in tests.glob("test_*.py"):
        f.write_text(f.read_text() + "\n\nimport os as _os, pytest as _pytest\n\n\n@_pytest.fixture(autouse=True)\n"
                     "def _order(request):\n    with open(_os.environ['ORDER_LOG'], 'a') as f:\n"
                     "        f.write(_os.environ.get('PYTEST_XDIST_WORKER', '-') + ' ' + request.node.nodeid + '\\n')\n")
    kept = tmp_path / "cache" / "per-test" / "pkg"
    kept.mkdir(parents=True)
    (kept / "tests~test_z.py--earlier.json").write_text(json.dumps({"when": "2026-09-30 00:00", "passed": {"tests/test_z.py::test_zz": 120.0}}))
    log = tmp_path / "order.log"
    p = _run_suite(root, tmp_path, "--jobs", "2", extra={"ORDER_LOG": str(log)})
    assert p.returncode == 0 and "on 2 workers" in p.stdout, p.stdout
    assert json.loads((tmp_path / "logs" / "pkg.heavy.json").read_text())[0] == "tests/test_z.py::test_zz"
    first = {}
    for line in log.read_text().splitlines():
        worker, node = line.split(" ", 1)
        first.setdefault(worker, node)
    assert "tests/test_z.py::test_zz" in first.values(), log.read_text()


def test_a_change_under_a_name_git_quotes_selects_the_suite_that_owns_it(tmp_path, monkeypatch):
    """`git diff --name-only` quotes a name with bytes outside ASCII (`"a/\\303\\251.py"`), which is under no suite's inputs, so
    `--changed` (the quick tier `research queue ready` runs) skipped the suite owning such a file."""
    _workspace(tmp_path)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    S = _suites(monkeypatch, tmp_path)
    pkgs, base = S.members(), _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "a" / "é.py").write_text("x = 1\n")
    assert [s.name for s in S.touched(S.discover(), pkgs, base, S.worktree(), S.locked())] == ["a"]


def test_a_package_under_a_name_git_quotes_is_a_suite(tmp_path, monkeypatch):
    """`git ls-files` quotes a name with bytes outside ASCII (`"pk\\303\\251/pyproject.toml"`), a path to no file, so the package
    there was no suite and its tests never ran."""
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["pké"]\n')
    (tmp_path / "pké" / "tests").mkdir(parents=True)
    (tmp_path / "pké" / "pyproject.toml").write_text('[project]\nname = "pke"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    _git(tmp_path, "init", "-q")
    S = _suites(monkeypatch, tmp_path)
    assert S.metadata() == ["pké/pyproject.toml", "pyproject.toml", "uv.lock"]
    assert [s.path for s in S.discover()] == ["pké"]


def test_the_suites_in_one_process_get_the_fewest_slots_that_finish_them_with_their_longest(tmp_path, monkeypatch):
    """ee02's pytest step ran its parallel suites one at a time (650 s) before circuit_check's one process (2,730 s) started, and
    its lean-suites ran evaluate (527 s) only after flock (862 s): the suites in one process now run beside the parallel ones."""
    S = _suites(monkeypatch, tmp_path)
    assert S.serial_slots([2730.0, 231.0, 203.0, 138.0, 98.0, 94.0, 36.0], 8, 32) == 2
    assert S.serial_slots([527.0, 289.0, 62.0, 52.0, 46.0, 42.0, 7.0], 1, 32) == 2
    assert S.serial_slots([100.0, 100.0, 100.0, None], 1, 32) == 4, "a suite never timed gets a slot of its own"
    assert S.serial_slots([None] * 40, 2, 32) == 8, "a fresh test cache times none: the parallel lane keeps 24 of 32"
    assert S.serial_slots([100.0, 100.0] + [None] * 40, 2, 32) == 10
    assert S.serial_slots([100.0] * 40, 1, 32) == 31, "the parallel suites keep a slot"
    assert S.serial_slots([5.0], 0, 32) == 32 and S.serial_slots([], 3, 32) == 0


def test_a_parallel_suite_runs_beside_the_others_on_the_slots_they_leave(tmp_path):
    import json

    root = tmp_path / "tree"
    _env_workspace(root, {"wide": "def test_a():\n    pass\n\ndef test_b():\n    pass\n", "one": "def test_one():\n    pass\n"})
    wide = root / "wide" / "pyproject.toml"
    wide.write_text(wide.read_text() + "\n[tool.verity.tests]\nparallel = true\n")
    _git(root, "commit", "-qam", "wide is parallel")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "durations.json").write_text(json.dumps({"one": 100.0, "wide": 50.0}))
    p = _run_suite(root, tmp_path, "--fresh", "--jobs", "3")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "1 parallel suites one at a time on 2 processes, beside 1 others on 1" in p.stdout, p.stdout
    assert re.search(r"passed  wide .* on 2 workers", p.stdout) and "passed  one" in p.stdout, p.stdout
