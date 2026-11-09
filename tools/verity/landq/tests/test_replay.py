"""The replay's pure parts: a trace's arrivals (times from its start, breaks drawn or from reverts) and the old path's
figures from the same trace."""
import pytest

from verity_landq import replay, sim

A = sim.Task("unit:a", frozenset({"a/"}), 10.0)
W = sim.World(tasks=(A,))


def row(pr: int, ready: str, main: str | None, *paths: str, reviewed: str | None = None, reverted: bool = False) -> dict:
    return {"pr": pr, "head": f"h{pr}", "ready": f"2000-01-01T{ready}:00+00:00",
            "reviewed": f"2000-01-01T{reviewed or ready}:00+00:00", "marked": True, "checked": f"2000-01-01T{ready}:00+00:00",
            "main": f"2000-01-01T{main}:00+00:00" if main else None, "reverted": reverted, "paths": list(paths)}


DOC = {"since": "2000-01-01T00:00:00+00:00", "until": "2000-01-01T10:00:00+00:00",
       "arrivals": [row(1, "01:00", "02:30", "a/x.py", reviewed="01:30"), row(2, "02:00", "07:00", "a/y.py", reverted=True),
                    row(3, "03:00", None, "docs/z.md")]}


def test_arrivals_count_minutes_from_the_start_and_a_revert_breaks_a_task_it_touches() -> None:
    got, _ = replay.arrivals(DOC, 0.0, 0, W)
    assert [(a.pr, a.ready, a.reviewed, a.breaks) for a in got] == [
        (1, 60.0, 90.0, frozenset()), (2, 120.0, 120.0, frozenset({"unit:a"})), (3, 180.0, 180.0, frozenset())]


def test_arrivals_draw_breaks_only_among_the_tasks_a_pr_touches() -> None:
    got, _ = replay.arrivals(DOC, 1.0, 0, W)
    assert [a.breaks for a in got] == [frozenset({"unit:a"}), frozenset({"unit:a"}), frozenset()]


def test_the_old_path_counts_a_revert_or_a_pr_off_main_as_never_on_main() -> None:
    old = replay.old_path(DOC)
    assert (old["prs"], old["landed"]) == (3, 1)
    assert old["p50_h"] == pytest.approx(1.5)
    assert old["within"] == pytest.approx(1 / 3)


def test_the_table_has_the_old_path_and_one_row_per_combination() -> None:
    s = sim.summary(sim.simulate(replay.arrivals(DOC, 0.0, 0, W)[0], W, replay.lq.Policy()))
    text = replay._table(replay.old_path(DOC), [{"batch": 4, "depth": 2, "nodes": 4, **s}])
    assert text.splitlines()[1].startswith("old path") and text.splitlines()[2].startswith("4  2  4")
