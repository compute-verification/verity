"""The read guard in one process and across pytest-xdist workers: a file the tests open outside the suite's inputs fails the
suite whichever process opened it, and a suite nothing vouches for (no report, a worker without a record) fails too."""
from __future__ import annotations

import importlib.util
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
    assert r.outside == ["outside/by_conftest.txt", "outside/by_test.txt"]
    assert r.counts["passed"] == 4


def test_a_parallel_suite_that_reads_only_its_inputs_passes(tmp_path, monkeypatch):
    pytest.importorskip("xdist")
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    r = S.run(suite, ["pkg"], [], tmp_path / "logs", 2)
    assert (r.status, r.outside, r.unguarded, r.counts["passed"]) == ("passed", [], [], 4), r.log.read_text()


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


def test_a_suite_without_the_guards_report_fails(tmp_path, monkeypatch):
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    r = S.run(suite, ["pkg"], ["-p", "no:verity_suite_guard"], tmp_path / "logs")
    assert (r.status, r.reported) == ("failed", False)


def test_a_worker_that_sends_no_record_leaves_the_suite_unguarded(monkeypatch):
    monkeypatch.delenv("VERITY_SUITE_REPORT", raising=False)   # load it inert: no second audit hook in this process
    G = _load(monkeypatch, "guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    record = {"outside": ["a.txt"], "watched": ["uv.lock"], "distributions": ["numpy"], "missing": ["torch"]}
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw0"), workeroutput={G._KEY: record}), None)
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw1")), error="crashed")
    assert (G._outside, G._watched, G._unguarded) == ({"a.txt"}, {"uv.lock"}, ["gw1"])
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
