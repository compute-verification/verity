"""`pod_shim_preflight_reuse.apply` on `main`'s `research.queue`: the scenarios of `checked`'s preflight-reuse tests, run against
`main`'s module with the shim applied, on `main`'s own test fixtures (`tools/research/tests/test_queue.py`, with the autouse
isolation of its `conftest.py`: no credentials, a store and a git config of the test's own, no machine reached).

From a checkout of this branch (`main` plus this file and the shim at its root):
`PYTHONPATH=tools/research/src:. python3 -m pytest -q test_pod_shim_preflight_reuse.py`.  The fixtures are read from the
`tools/research/tests/` beside the `research` that PYTHONPATH gives."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import research
import research.queue as Q

TESTS = Path(research.__file__).resolve().parents[2] / "tests"
sys.path.insert(0, str(TESTS))
from test_queue import World, _git, _move_main, _sync_preflight  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_conftest = _load("research_tests_conftest", TESTS / "conftest.py")
_isolated_store, _isolated_git = _conftest._isolated_store, _conftest._isolated_git

SHIM = Path(__file__).with_name("pod_shim_preflight_reuse.py")
RELOCKED = {"pkg/lean-audit.json": json.dumps({"guarantees": {"T": "2"}, "reads": {}, "dependencies": {"M": "1"}})}


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


@pytest.fixture(autouse=True)
def shimmed():
    saved = {k: Q.Queue.__dict__[k] for k in ("_preflighted", "sync")}
    _load("pod_shim_preflight_reuse", SHIM).apply(Q)
    yield
    for k in ("_standing", "_preflights_reused"):
        delattr(Q.Queue, k)
    for k, v in saved.items():
        setattr(Q.Queue, k, v)


def test_the_shim_patches_mains_module_once_and_refuses_twice():
    assert Q.__file__.endswith("tools/research/src/research/queue.py") and hasattr(Q.Queue, "_standing")
    with pytest.raises(RuntimeError, match="already has _standing: this shim is obsolete"):
        _load("pod_shim_preflight_reuse", SHIM).apply(Q)


def test_an_unchanged_head_and_base_tree_reads_its_record_back_and_a_new_base_runs_a_refusal_again(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    refuse, calls = _sync_preflight({2: "refused: x"})
    q = world.queue()
    q.preflight = refuse
    q.sync()
    again = world.queue()
    again.preflight = refuse
    members, entries = again.sync()
    assert [m.pr for m in members] == [1] and entries[2].why == "refused: x" and [n for n, _ in calls] == [1, 2], world.logs
    _move_main(world, {"c.txt": "moved\n"})
    moved = world.queue()
    moved.preflight = refuse
    moved.sync()
    assert [n for n, _ in calls] == [1, 2, 2], world.logs


def test_a_rebuild_below_a_member_that_changes_nothing_its_change_touches_keeps_its_pass_and_runs_no_preflight(world):
    for n, f in ((1, "a.txt"), (2, "b.txt"), (3, "c.txt")):
        world.gh.put(n, world.change({f: f"{n}\n"}))
    q = world.queue()
    q.preflight, calls = _sync_preflight({})
    assert [m.pr for m in q.sync()[0]] == [1, 2, 3] and [n for n, _ in calls] == [1, 2, 3]
    world.gh.ready.discard((1, world.gh.prs[1].head))
    assert [m.pr for m in q.sync()[0]] == [2, 3] and [n for n, _ in calls] == [1, 2, 3], world.logs
    assert world.logs[-1].endswith("kept 0, merged 2, preflights' verdicts reused for 2"), world.logs[-1]
    _move_main(world, RELOCKED)
    assert [m.pr for m in q.sync()[0]] == [2, 3] and [n for n, _ in calls] == [1, 2, 3], world.logs
    assert world.logs[-1].endswith("kept 0, merged 2, preflights' verdicts reused for 2"), world.logs[-1]


def test_a_rebuild_that_changes_a_file_a_members_change_touches_preflights_that_member_again(world):
    _move_main(world, {"m.txt": "1\n2\n3\n4\n5\n6\n7\n"})
    world.gh.put(1, world.change({"m.txt": "one\n2\n3\n4\n5\n6\n7\n"}))
    world.gh.put(2, world.change({"m.txt": "1\n2\n3\n4\n5\n6\nseven\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    q = world.queue()
    q.preflight, calls = _sync_preflight({})
    assert [m.pr for m in q.sync()[0]] == [1, 2, 3] and [n for n, _ in calls] == [1, 2, 3]
    world.gh.ready.discard((1, world.gh.prs[1].head))
    assert [m.pr for m in q.sync()[0]] == [2, 3], world.logs
    assert calls[3:] == [(2, _git(world.origin, "rev-parse", "main"))], world.logs
    assert world.logs[-1].endswith("preflights' verdicts reused for 1"), world.logs[-1]


def test_a_pass_stands_across_no_base_that_changes_what_the_checks_run_or_deletes_a_path_nor_for_a_change_that_deletes_one(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": None}))
    q = world.queue()
    q.preflight, calls = _sync_preflight({})
    assert [m.pr for m in q.sync()[0]] == [1, 2] and [n for n, _ in calls] == [1, 2]
    _move_main(world, RELOCKED)
    assert [m.pr for m in q.sync()[0]] == [1, 2] and [n for n, _ in calls[2:]] == [2], world.logs
    _move_main(world, {"tests/test_pins.py": "def test_no_pin_is_stale():\n    pass\n"})
    assert [m.pr for m in q.sync()[0]] == [1, 2] and [n for n, _ in calls[3:]] == [1, 2], world.logs
    _move_main(world, {"c.txt": None})
    assert [m.pr for m in q.sync()[0]] == [1, 2] and [n for n, _ in calls[5:]] == [1, 2], world.logs


def test_a_new_head_is_preflighted_though_its_change_onto_next_is_the_old_heads_file_for_file(world):
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, old := world.change({"b.txt": "2\n"}))
    q = world.queue()
    q.preflight, calls = _sync_preflight({})
    assert [m.pr for m in q.sync()[0]] == [1, 2] and [n for n, _ in calls] == [1, 2]
    _move_main(world, {"c.txt": "moved\n"})
    world.gh.put(2, new := world.change({"b.txt": "2\n"}))
    members, _ = q.sync()
    assert new != old and [(m.pr, m.head) for m in members] == [(1, world.gh.prs[1].head), (2, new)], world.logs
    assert calls[2:] == [(2, members[0].commit)], world.logs


def test_a_waiting_clause_stays_after_the_reuse_clause_in_the_summary_line(world):
    now = [0.0]

    def slow(pr, base, commit, members):
        now[0] += Q.PREFLIGHT_BUDGET_S
        return None

    world.gh.put(1, world.change({"a.txt": "1\n"}))
    q = world.queue()
    q.preflight, q.clock = slow, lambda: now[0]
    q.sync()
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    world.gh.put(3, world.change({"c.txt": "3\n"}))
    _move_main(world, RELOCKED)
    assert [m.pr for m in q.sync()[0]] == [1, 2], world.logs
    assert world.logs[-1].endswith("preflights' verdicts reused for 1, 1 waiting off it for their preflight (#3)"), world.logs[-1]
