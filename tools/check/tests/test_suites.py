"""`suites.py`: suites are found and keyed from git's view of the tree, on a laptop's checkout and on the checkout `research run
--source` ships alike."""
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
        out[s.name] = S.key(s, S.object_ids(tree, sorted(set(paths) | set(S.RUNNER))), env, [], S.resolved(S.roots(s, pkgs), lock))
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
    (tmp_path / "tools" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "check" / "suites.py").write_text("# the runner\n")
    assert touched() == ["a", "b"]


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
    """A checkout with this runner and one parallel suite: `test_heavy` (on a helper), `test_sibling` (failing until fixed),
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
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "shipped")


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
    assert not list((cache / "per-test" / "pkg").glob("pkg~tests~test_outside.py--*.json"))
    assert list((cache / "per-test" / "pkg").glob("pkg~tests~test_heavy.py--*.json"))

    p = _run_suite(root, tmp_path)                                              # nothing changed: no pass to reuse
    assert p.returncode == 1 and "1 passed, 5 reused" in p.stdout and "elsewhere/data.txt: read by" in p.stdout, p.stdout

    pyproject = root / "pkg" / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace("parallel = true", 'parallel = true\ninputs = ["elsewhere"]'))
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "passed  pkg" in p.stdout and "outside its inputs" not in p.stdout, p.stdout
    p = _run_suite(root, tmp_path)
    assert p.returncode == 0 and "cached  pkg" in p.stdout, p.stdout


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

    (root / "tools" / "check").mkdir(parents=True)
    shutil.copy(HERE / "suites.py", root / "tools" / "check" / "suites.py")
    shutil.copytree(HERE / "guard", root / "tools" / "check" / "guard", ignore=shutil.ignore_patterns("__pycache__"))
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
    assert r.trace("open", (str(tmp_path / "lib" / "a.py"), "r", 0)) == ("F", "lib/a.py")
    assert r.trace("os.scandir", (str(tmp_path / "lib"),)) == ("L", "lib")
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, None)) is None
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-I", "-c", "pass"], None, None)) == ("X", sys.executable)
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, {"PATH": "/bin"}), ) == ("X", sys.executable)
    assert r.trace("subprocess.Popen", (None, ["git", "status"], None, None)) == ("X", "git")
    assert r.trace("subprocess.Popen", (None, ["uv", "run", "python"], None, None)) == ("X", "uv")
    assert r.trace("os.system", ("ls",)) == ("X", "os.system")
    assert r.trace("subprocess.Popen", (None, ["uname", "-p"], None, None)) is None       # a machine probe reads no repository file
    monkeypatch.delenv("VERITY_SUITE_MODULE")                  # outside a test module sitecustomize doesn't trace
    assert r.trace("subprocess.Popen", (None, [sys.executable, "-c", "pass"], None, None)) == ("X", sys.executable)


def _guard(monkeypatch):
    """The guard module, loaded without its audit hook (which it installs only when `suites.py` runs it)."""
    monkeypatch.delenv("VERITY_SUITE_ROOT", raising=False)
    spec = importlib.util.spec_from_file_location("guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    (kept / "pkg~tests~test_z.py--earlier.json").write_text(json.dumps({"when": "2026-09-30 00:00", "passed": {"tests/test_z.py::test_zz": 120.0}}))
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
