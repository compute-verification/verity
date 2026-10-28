"""The read guard in one process and across pytest-xdist workers: a file the tests open outside the suite's inputs fails the
suite whichever process opened it, and a suite nothing vouches for (no report, a worker without a record) fails too. And
each suite's own `--basetemp`, which no sweep takes while anything still uses it."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)                # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def _suite(tmp_path: Path, monkeypatch, reads_outside: bool):
    """A suite `pkg` whose conftest (on import) and four tests each read a file, outside its inputs or inside them."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    monkeypatch.setattr(S, "ROOT", tmp_path)
    (tmp_path / "outside").mkdir()
    for f in ("by_conftest.txt", "by_test.txt"):
        (tmp_path / "outside" / f).write_text("not an input\n")
    tests = tmp_path / "pkg" / "tests"
    tests.mkdir(parents=True)
    (tmp_path / "pkg" / "pyproject.toml").write_text('[project]\nname = "pkg"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "pkg" / "data.txt").write_text("an input\n")
    where = "(ROOT / 'outside' / '{}')" if reads_outside else "(ROOT / 'pkg' / 'data.txt')"
    head = "from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n"
    (tests / "conftest.py").write_text(head + f"{where.format('by_conftest.txt')}.read_text()\n")
    (tests / "test_reads.py").write_text(head + "".join(f"\ndef test_{i}():\n    {where.format('by_test.txt')}.read_text()\n"
                                                        for i in range(4)))
    (tmp_path / "logs").mkdir()
    return S, S.Suite("pkg", "pkg", (), ())


@pytest.mark.parametrize("workers", [1, 2])
def test_what_any_process_reads_outside_the_inputs_fails_the_suite(tmp_path, monkeypatch, workers):
    if workers > 1:
        pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=True)
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", workers)
    assert (r.status, r.workers, r.unguarded) == ("failed", workers, [])
    assert r.outside == {"outside/by_conftest.txt": "*", "outside/by_test.txt": "pkg/tests/test_reads.py"}
    assert r.counts["passed"] == 4


def test_a_parallel_suite_that_reads_only_its_inputs_passes(tmp_path, monkeypatch):
    pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", 2)
    assert (r.status, r.outside, r.unguarded, r.counts["passed"]) == ("passed", {}, [], 4), r.log.read_text()


def test_a_cached_module_read_outside_the_inputs_is_named_as_its_source(tmp_path, monkeypatch):
    """Train 21b3d26fe1a6 (14 Oct): a suite imported an undeclared helper through its warm `__pycache__`, and the failure named
    only `...cpython-*.pyc`, which no PR's diff touches, so drive found no culprit. The read is named as the `.py` source,
    attributed to the test module that imported it, warm or cold cache alike."""
    import py_compile
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    (tmp_path / "outside" / "helper.py").write_text("X = 1\n")
    py_compile.compile(str(tmp_path / "outside" / "helper.py"))
    (tmp_path / "pkg" / "tests" / "test_reads.py").write_text(
        "import importlib.util\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n\n"
        "def test_imports_a_helper():\n"
        "    spec = importlib.util.spec_from_file_location('helper_under_test', ROOT / 'outside' / 'helper.py')\n"
        "    mod = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(mod)\n    assert mod.X == 1\n")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs")
    assert (r.status, r.counts["passed"]) == ("failed", 1), r.log.read_text()
    assert r.outside == {"outside/helper.py": "pkg/tests/test_reads.py"}, r.log.read_text()


def test_a_suites_tmp_paths_are_under_its_own_basetemp_and_gone_when_it_ends(tmp_path, tmp_path_factory, monkeypatch):
    """pytest keeps its last three base temps (a flock suite's is ~45 GB: vy-cpu-1 and vy-cpu-2 filled on 11 Oct); each suite gets
    its own `--basetemp`, removed when it ends, unless the caller passed one. While it runs, its owner marker names this process
    and pytest's session (which the sweep of a later suites.py checks: the sweep tests below)."""
    import tempfile
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    tmp = tmp_path_factory.mktemp("tmp")                         # outside the suite's tree, as /tmp is
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setenv("EXPECT", str(tmp / f"{S.BASETEMP}-{os.getpid()}m"))
    monkeypatch.setenv("OWNER", str(os.getpid()))
    (tmp_path / "pkg" / "tests" / "test_reads.py").write_text(
        "import json, os\n\ndef test_tmp(tmp_path):\n    (tmp_path / 'big').write_bytes(b'x' * 4096)\n"
        "    assert str(tmp_path).startswith(os.environ['EXPECT'])\n"
        "    if os.environ['OWNER']:\n"
        "        doc = json.loads((tmp_path.parents[1] / 'owner.json').read_text())\n"
        "        assert [o[0] for o in doc['owners']] == [int(os.environ['OWNER'])], doc\n"
        "        assert [s[0] for s in doc['sessions']] == [os.getpid()] == [os.getsid(0)], doc\n")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs")
    assert (r.status, r.counts["passed"]) == ("passed", 1), r.log.read_text()
    assert list(tmp.iterdir()) == []
    mine = tmp_path_factory.mktemp("mine")
    monkeypatch.setenv("EXPECT", str(mine))
    monkeypatch.setenv("OWNER", "")
    r = S.run(suite, ["pkg"], [f"--basetemp={mine}"], tmp_path / "logs")
    assert r.status == "passed" and (mine / "test_tmp0" / "big").exists() and list(tmp.iterdir()) == []



NOW = 2_000_000_000.0
HOUR = 3600.0


def _proc(root: Path, procs: dict[int, tuple[int, int]], boot: str = "boot-a", ns: str = "pid:[1]") -> Path:
    """A /proc of this boot and pid namespace, holding ``procs`` (pid: (start time, session)) and nothing else."""
    if root.exists():
        shutil.rmtree(root)
    (root / "sys" / "kernel" / "random").mkdir(parents=True)
    (root / "sys" / "kernel" / "random" / "boot_id").write_text(boot + "\n")
    (root / "self" / "ns").mkdir(parents=True)
    os.symlink(ns, root / "self" / "ns" / "pid")
    for pid, (start, sid) in {**procs, "self": procs.get(os.getpid(), (7, 7))}.items():
        (root / str(pid)).mkdir(exist_ok=True)
        (root / str(pid) / "stat").write_text(f"{pid} (a) b) S 1 {sid} {sid} " + "0 " * 15 + f"{start} 0 0\n")
    return root


def _basetemp(tmp: Path, name: str, owner: dict | None, age_h: float, newest_h: float | None = None) -> Path:
    """A basetemp named ``name`` with ``owner`` as its marker (None: none), unchanged for ``age_h`` hours but for a file deep in
    it changed ``newest_h`` hours ago."""
    import json

    d = tmp / name
    (d / "b" / "popen-gw0" / "deep").mkdir(parents=True)
    f = d / "b" / "popen-gw0" / "deep" / "file"
    f.write_text("x")
    if owner is not None:
        (d / "owner.json").write_text(json.dumps(owner))
    for p in (f, d / "b" / "popen-gw0" / "deep", d / "b" / "popen-gw0", d / "b", *([d / "owner.json"] if owner else []), d):
        t = NOW - (newest_h if newest_h is not None and p == f else age_h) * HOUR
        os.utime(p, (t, t))
    return d


def _owner(*owners: tuple[int, int], sessions=(), boot: str = "boot-a", ns: str = "pid:[1]") -> dict:
    return {"boot": boot, "pidns": ns, "owners": [list(o) for o in owners], "sessions": [list(s) for s in sessions]}


def test_a_basetemp_another_namespace_or_boot_owns_is_never_swept_while_anything_in_it_changes(tmp_path, monkeypatch):
    """A sweep that can't see a basetemp's owner (another pid namespace, another boot, a /proc that hides it) never takes
    it as gone, whatever the owner's pid is here; it waits until nothing in it changed for a day."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    (tmp := tmp_path / "tmp").mkdir()
    proc = _proc(tmp_path / "proc", {})
    for age, newest in ((1, None), (48, 2), (23.9, None)):
        _basetemp(tmp, f"verity-suite-mns{age}", _owner((42, 5), ns="pid:[2]"), age, newest)
        _basetemp(tmp, f"verity-suite-mboot{age}", _owner((42, 5), boot="boot-b"), age, newest)
    assert S.sweep_basetemps(tmp, proc=proc, now=NOW) == []
    old = {_basetemp(tmp, "verity-suite-mns-old", _owner((42, 5), ns="pid:[2]"), 25),
           _basetemp(tmp, "verity-suite-mboot-old", _owner((42, 5), boot="boot-b"), 25)}
    assert set(S.sweep_basetemps(tmp, proc=proc, now=NOW)) == old and not any(d.exists() for d in old)
    assert len(list(tmp.iterdir())) == 6


def test_a_basetemp_whose_owner_died_here_is_swept_and_a_reused_pid_does_not_keep_it(tmp_path, monkeypatch):
    """Under this boot and namespace an owner is alive only at its pid with its start time: one gone, or whose pid another process
    took since (a different start time), leaves its basetemp to the sweep at once; one alive keeps it."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    (tmp := tmp_path / "tmp").mkdir()
    proc = _proc(tmp_path / "proc", {42: (500, 42), 43: (999, 43)})
    live = _basetemp(tmp, "verity-suite-mlive", _owner((42, 500)), 0)
    dead = _basetemp(tmp, "verity-suite-mdead", _owner((44, 500)), 0)
    reused = _basetemp(tmp, "verity-suite-mreused", _owner((43, 500)), 0)
    assert sorted(S.sweep_basetemps(tmp, proc=proc, now=NOW)) == sorted([dead, reused])
    assert live.exists() and not dead.exists() and not reused.exists()


def test_a_basetemp_stays_while_its_pytest_session_runs_after_its_suites_py_is_killed(tmp_path, monkeypatch):
    """vy-nebius-1, 18 Oct: two suites.py were killed (SIGKILL) while their pytest, in a session of its own, ran on for 8 and 37
    minutes; the next suites.py's sweep took their basetemps from under them. A process left in pytest's session (pytest itself,
    or an xdist worker it left) keeps the basetemp; once the session is empty it is swept."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    (tmp := tmp_path / "tmp").mkdir()
    d = _basetemp(tmp, "verity-suite-morphan", _owner((42, 500), sessions=[(60, 510)]), 0)
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {60: (510, 60)}), now=NOW) == [] and d.exists()
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {61: (520, 60)}), now=NOW) == [] and d.exists()
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {60: (900, 900)}), now=NOW) == [d] and not d.exists()


def test_a_basetemp_without_a_marker_is_swept_only_once_nothing_in_it_changed_for_a_day(tmp_path, monkeypatch):
    """An older suites.py's `verity-suite-<pid>-*`, the guard's `verity-suite-host-*`, or a marker that doesn't parse: the sweep
    can't tell who uses it, so the age of its newest entry decides, never a pid in its name."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    (tmp := tmp_path / "tmp").mkdir()
    proc = _proc(tmp_path / "proc", {})
    kept = [_basetemp(tmp, "verity-suite-999999-fresh", None, 1), _basetemp(tmp, "verity-suite-host-deep", None, 30, 23),
            _basetemp(tmp, "verity-suite-mbroken", {"boot": "boot-a"}, 2)]
    old = [_basetemp(tmp, "verity-suite-999999-old", None, 25), _basetemp(tmp, "verity-suite-host-old", None, 30),
           _basetemp(tmp, "verity-suite-mbroken-old", {"owners": "x"}, 25)]
    (tmp / "verity-suite-file").write_text("not a directory")
    assert sorted(S.sweep_basetemps(tmp, proc=proc, now=NOW)) == sorted(old)
    assert all(d.exists() for d in kept) and (tmp / "verity-suite-file").exists()


def test_the_marker_a_run_writes_names_this_process_and_pytests_session(tmp_path, monkeypatch):
    """`mark_basetemp` as `run` calls it, read back by the sweep: this process (by its start time) keeps the basetemp, and so
    does pytest's session once it is named; with both gone it is swept."""
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    (tmp := tmp_path / "tmp").mkdir()
    (d := tmp / "verity-suite-mrun").mkdir()
    me = os.getpid()
    S.mark_basetemp(d, proc=_proc(tmp_path / "proc", {me: (300, 300)}))
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {me: (300, 300)}), now=NOW) == []
    S.mark_basetemp(d, session=80, proc=_proc(tmp_path / "proc", {me: (300, 300), 80: (310, 80)}))
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {81: (320, 80)}), now=NOW) == []
    assert S.sweep_basetemps(tmp, proc=_proc(tmp_path / "proc", {me: (301, 1)}), now=NOW) == [d]


@pytest.mark.parametrize("listed", [False, True])
def test_markdown_inside_the_inputs_is_outside_them_unless_the_suite_lists_markdown(tmp_path, monkeypatch, listed):
    """A suite's keys leave `*.md` out of its directories unless it lists a `.md` (`suites.markdown`), so a test that reads one
    unlisted fails, even under the suite's own directory; one that lists it passes."""
    S, _ = _suite(tmp_path, monkeypatch, reads_outside=False)
    (tmp_path / "pkg" / "README.md").write_text("prose\n")
    (tmp_path / "pkg" / "tests" / "test_reads.py").write_text(
        "from pathlib import Path\n\ndef test_prose():\n    (Path(__file__).resolve().parents[1] / 'README.md').read_text()\n")
    suite = S.Suite("pkg", "pkg", (), ("pkg/README.md",) if listed else ())
    r = S.run(suite, ["pkg"], [], tmp_path / "logs")
    want = ("passed", {}) if listed else ("failed", {"pkg/README.md": "pkg/tests/test_reads.py"})
    assert (r.status, r.outside) == want, r.log.read_text()


@pytest.mark.parametrize("declared", ["nothing", "the directory", "its files"])
def test_a_nested_suites_files_are_outside_the_inputs_unless_declared(tmp_path, monkeypatch, declared):
    """`pkg/sub` is a suite of its own, which `pkg`'s key leaves out (`suites.dropped`): a test of `pkg` that reads one of its
    files, in its own process or a child's, fails `pkg` until `pkg` declares the directory or the files."""
    S, _ = _suite(tmp_path, monkeypatch, reads_outside=False)
    (tmp_path / "pkg" / "sub").mkdir()
    (tmp_path / "pkg" / "sub" / "pyproject.toml").write_text('[project]\nname = "sub"\n\n[tool.pytest.ini_options]\n')
    for f in ("data.txt", "other.txt"):
        (tmp_path / "pkg" / "sub" / f).write_text("the nested suite's\n")
    (tmp_path / "pkg" / "tests" / "test_reads.py").write_text(
        "import subprocess, sys\nfrom pathlib import Path\nSUB = Path(__file__).resolve().parents[1] / 'sub'\n\n"
        "def test_itself():\n    (SUB / 'data.txt').read_text()\n\n"
        "def test_child():\n    subprocess.run([sys.executable, '-c', 'import sys; open(sys.argv[1]).read()', str(SUB / 'other.txt')],"
        " check=True)\n")
    suite = S.Suite("pkg", "pkg", (), (), nested=("pkg/sub",))
    inputs = {"nothing": [], "the directory": ["pkg/sub"], "its files": ["pkg/sub/data.txt", "pkg/sub/other.txt"]}[declared]
    r = S.run(suite, ["pkg", *inputs], [], tmp_path / "logs")
    assert r.counts["passed"] == 2, r.log.read_text()
    if declared == "nothing":
        assert (r.status, r.outside) == ("failed", {"pkg/sub/data.txt": "pkg/tests/test_reads.py"}), r.log.read_text()
        assert r.child_outside == {"pkg/tests/test_reads.py": ["pkg/sub/other.txt"]}
    else:
        assert (r.status, r.outside, r.child_outside) == ("passed", {}, {}), r.log.read_text()


@pytest.mark.parametrize("workers", [1, 2])
def test_every_test_starts_without_the_machines_markers_caches_and_slot(tmp_path, monkeypatch, workers):
    """Host-dependent tests failed five full checks in 16 h, each fixed one test at a time: cpu-1's /etc/vy/runs-via-slot, node 1's
    real check caches, a slot's CHECK_SLOT. In every suite a test module sees, at import and in each test, every marker the
    guard names pointed at a path that doesn't exist, each cache an empty directory of its own and no CHECK_SLOT."""
    if workers > 1:
        pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    monkeypatch.delenv("VERITY_SUITE_REPORT", raising=False)
    G = _load(monkeypatch, "guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    for k in (*G.HOST_FILES, *G.HOST_CACHES, *G.HOST_SLOT):
        monkeypatch.setenv(k, str(tmp_path / "pkg" / "data.txt"))
    names = sorted({*G.HOST_FILES, *G.HOST_CACHES, *G.HOST_SLOT})
    (tmp_path / "pkg" / "tests" / "test_host.py").write_text(
        f"import os\nAT_IMPORT = {{k: os.environ.get(k) for k in {names!r}}}\n\n"
        "def _host(env):\n"
        f"    for k in {sorted(G.HOST_FILES)!r}:\n        assert not os.path.exists(env[k]), k\n"
        f"    for k in {sorted(G.HOST_CACHES)!r}:\n        assert os.listdir(env[k]) == [], k\n"
        f"    assert {list(G.HOST_SLOT)!r} and all(env.get(k) is None for k in {list(G.HOST_SLOT)!r})\n\n"
        "def test_at_import():\n    _host(AT_IMPORT)\n\n"
        "def test_in_the_test():\n    _host(os.environ)\n")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", workers)
    assert (r.status, r.counts["passed"]) == ("passed", 6), r.log.read_text()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("required", [False, True])
def test_under_a_recorded_check_a_skip_for_an_unreachable_store_fails(tmp_path, monkeypatch, workers, required):
    """A recorded check's Lean tests passed without running: their fixture skipped because the store was out of reach (#757's
    red team). Under VERITY_STORE_REQUIRED=1 that skip, from a fixture or a test, fails; any other skip stays a skip."""
    if workers > 1:
        pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    monkeypatch.delenv("VERITY_SUITE_REPORT", raising=False)   # load it inert: no second audit hook in this process
    G = _load(monkeypatch, "guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    (tmp_path / "pkg" / "tests" / "test_store.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef fixture():\n    pytest.skip(f'" + G.STORE_UNREACHABLE + ": no remote')\n\n"
        "def test_in_a_fixture(fixture):\n    pass\n\n"
        "def test_in_the_test():\n    pytest.skip('" + G.STORE_UNREACHABLE + " here')\n\n"
        "def test_without_a_gpu():\n    pytest.skip('no GPU')\n")
    if required:
        monkeypatch.setenv("VERITY_STORE_REQUIRED", "1")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", workers)
    log = r.log.read_text()
    assert r.counts["passed"] == 4, log
    if required:
        assert (r.status, r.counts["failed"], r.counts["error"], r.counts["skipped"]) == ("failed", 1, 1, 1), log
        assert "VERITY_STORE_REQUIRED=1 (a recorded check)" in log
    else:
        assert (r.status, r.counts["failed"], r.counts["error"], r.counts["skipped"]) == ("passed", 0, 0, 3), log


def _ours_and_theirs(tmp_path: Path) -> Path:
    """`mine`, a package in the suite's tree on its tests' `sys.path`, whose `needs` imports a distribution nobody installs."""
    mine = tmp_path / "pkg" / "tests" / "mine"
    mine.mkdir()
    (mine / "__init__.py").write_text("")
    (mine / "needs.py").write_text("import no_such_distribution_7f1c  # noqa: F401\n")
    return mine.parent


@pytest.mark.parametrize("workers", [1, 2])
def test_an_importorskip_of_a_module_of_ours_that_moved_fails(tmp_path, monkeypatch, workers):
    """pytest skips on the `ModuleNotFoundError` a moved module raises, so a test that importorskips a module of ours passed as a
    skip once it moved. The guard fails it, and still skips a missing third-party package, one a module of ours needs included."""
    if workers > 1:
        pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    (_ours_and_theirs(tmp_path) / "test_imports.py").write_text(
        "import pytest\n\n"
        "def test_moved():\n    pytest.importorskip('mine.moved', reason='not merged yet')\n\n"
        "def test_third_party():\n    pytest.importorskip('no_such_distribution_7f1c')\n\n"
        "def test_ours_needing_a_third_party():\n    pytest.importorskip('mine.needs')\n\n"
        "def test_ours_present():\n    assert pytest.importorskip('mine').__name__ == 'mine'\n")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", workers)
    log = r.log.read_text()
    assert (r.status, r.counts["passed"], r.counts["failed"], r.counts["skipped"]) == ("failed", 5, 1, 2), log
    assert "mine.moved is not importable, and mine is this checkout's" in log


def test_a_module_level_importorskip_of_a_module_of_ours_that_moved_fails_collection(tmp_path, monkeypatch):
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    (_ours_and_theirs(tmp_path) / "test_collected.py").write_text(
        "import pytest\n\npytest.importorskip('mine.gone')\n\n\ndef test_never_runs():\n    pass\n")
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", 1)
    log = r.log.read_text()
    assert r.status == "failed" and "mine.gone is not importable" in log, log


def test_a_suite_without_the_guards_report_fails(tmp_path, monkeypatch):
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    r = S.run(suite, ["pkg"], ["-p", "no:verity_suite_guard"], tmp_path / "logs")
    assert (r.status, r.reported) == ("failed", False)


def test_a_worker_that_sends_no_record_leaves_the_suite_unguarded(monkeypatch):
    monkeypatch.delenv("VERITY_SUITE_REPORT", raising=False)   # load it inert: no second audit hook in this process
    G = _load(monkeypatch, "guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    record = {"outside": {"a.txt": "pkg/tests/test_a.py"}, "watched": ["uv.lock"], "distributions": ["numpy"], "missing": ["torch"]}
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw0"), workeroutput={G._KEY: record}), None)
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw1")), error="crashed")
    assert (G._outside, G._watched, G._unguarded) == ({"a.txt": "pkg/tests/test_a.py"}, {"uv.lock"}, ["gw1"])
    assert G._record == {"distributions": {"numpy"}, "missing": {"torch"}}


@pytest.mark.parametrize("workers", [1, 2])
def test_the_guard_records_what_the_key_leaves_out(tmp_path, monkeypatch, workers):
    """Metadata files the tests read, distributions they import, and imports nothing installed provides: in any process."""
    if workers > 1:
        pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "pyproject.toml").write_text("[project]\nname = 'other'\n")
    (tmp_path / "pkg" / "tests" / "test_uses.py").write_text(
        "from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n\n"
        "def test_uses():\n    (ROOT / 'other' / 'pyproject.toml').read_text()\n    import iniconfig  # noqa: F401\n"
        "    try:\n        import surely_no_such_distribution  # noqa: F401\n    except ImportError:\n        pass\n")
    r = S.run(suite, ["pkg", "other/pyproject.toml"], [], tmp_path / "logs", workers, ["other/pyproject.toml"])
    assert r.status == "passed", r.log.read_text()
    assert r.watched == ["other/pyproject.toml"] and "iniconfig" in r.distributions and "pytest" in r.distributions
    assert "surely_no_such_distribution" in r.missing and "iniconfig" not in r.missing


def test_slots_go_to_the_suites_in_turn(tmp_path, monkeypatch):
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    slots, order = S.Slots(2), []
    waits = threading.Semaphore(0)

    class Counted(threading.Condition):
        def wait(self, timeout=None):
            waits.release()
            return super().wait(timeout)

    slots.cond = Counted()

    def take(turn: int, n: int) -> None:
        with slots.take(turn, n):
            order.append(turn)

    with slots.take(0, 2):                                     # a parallel suite holds every slot
        waiting = [threading.Thread(target=take, args=a) for a in ((2, 2), (1, 1))]
        for t in waiting:
            t.start()
        waits.acquire()
        waits.acquire()                                        # both are waiting for their turn
        assert order == []
    for t in waiting:
        t.join()
    assert order == [1, 2]
