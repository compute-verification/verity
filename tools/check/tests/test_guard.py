"""The read guard in one process and across pytest-xdist workers: a file the tests open outside the suite's inputs fails the
suite whichever process opened it, and a suite nothing vouches for (no report, a worker without a record) fails too."""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
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


def test_a_suite_without_the_guards_report_fails(tmp_path, monkeypatch):
    S, suite = _suite(tmp_path, monkeypatch, reads_outside=False)
    r = S.run(suite, ["pkg"], ["-p", "no:verity_suite_guard"], tmp_path / "logs")
    assert (r.status, r.reported) == ("failed", False)


def test_a_worker_that_sends_no_record_leaves_the_suite_unguarded(monkeypatch):
    monkeypatch.delenv("VERITY_SUITE_REPORT", raising=False)   # load it inert: no second audit hook in this process
    G = _load(monkeypatch, "guard_under_test", HERE / "guard" / "verity_suite_guard.py")
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw0"), workeroutput={G._KEY: ["a.txt"]}), None)
    G.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw1")), error="crashed")
    assert (G._outside, G._unguarded) == ({"a.txt"}, ["gw1"])


def test_slots_go_to_the_suites_in_turn(tmp_path, monkeypatch):
    S = _load(monkeypatch, "suites_under_test", HERE / "suites.py")
    slots, order = S.Slots(2), []

    def take(turn: int, n: int) -> None:
        with slots.take(turn, n):
            order.append(turn)

    with slots.take(0, 2):                                     # a parallel suite holds every slot
        waiting = [threading.Thread(target=take, args=a) for a in ((2, 2), (1, 1))]
        for t in waiting:
            t.start()
        time.sleep(0.1)
        assert order == []
    for t in waiting:
        t.join(5)
    assert order == [1, 2]
