"""The replay's pure parts: a trace's arrivals (times from its start, breaks drawn or from reverts) and the old path's
figures from the same trace."""
import fake_vqueue as vq
import pytest

from verity_landq import replay, sim, snapshots

A = sim.Task("unit:a", frozenset({"a/"}), 10.0)
W = sim.World(tasks=(A,))
Q = snapshots.Driver(vq)


def simulate(trace: list[sim.Arrival]) -> sim.Report:
    return sim.simulate(trace, W, replay.lq.Policy(), step=Q.step, state=Q.start("main"))


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


def test_a_pr_that_landed_twice_arrives_as_two_changes_and_both_land() -> None:
    doc = dict(DOC, arrivals=[row(1, "01:00", "02:30", "a/x.py"), {**row(1, "01:10", "05:00", "a/y.py"), "head": "h1b"}])
    got, _ = replay.arrivals(doc, 0.0, 0, W)
    assert [(a.pr, a.head) for a in got] == [(1, "h1"), (1 + replay.REPEAT, "h1b")]
    assert all(x.main is not None for x in simulate(got).landings)


def test_the_old_path_reads_main_lag_from_the_review_and_from_the_landing_on_checked() -> None:
    old = replay.old_path(DOC)
    assert old["lag_max_h"] == old["checked_lag_max_h"] == pytest.approx(410 / 60)
    assert old["checked_from"] == "2000-01-01T01:00:00+00:00"
    late = dict(DOC, arrivals=[*DOC["arrivals"][:2], {**DOC["arrivals"][2], "checked": "2000-01-01T06:00:00+00:00"}])
    assert replay.old_path(late)["lag_max_h"] == pytest.approx(410 / 60)
    assert replay.old_path(late)["checked_lag_max_h"] == pytest.approx(290 / 60)


def test_the_table_has_the_old_path_and_one_row_per_combination() -> None:
    s = sim.summary(simulate(replay.arrivals(DOC, 0.0, 0, W)[0]))
    text = replay._table(replay.old_path(DOC), [{"batch": 4, "depth": 2, "nodes": 4, **s}])
    assert text.splitlines()[1].startswith("old path") and text.splitlines()[2].startswith("4  2  4")


def test_set_reads_snapshot_fields_and_the_queue_must_be_a_checkouts_vqueue() -> None:
    assert [replay._field(s) for s in ("lean=3", "pending_s=1.5e4", "now=.5", "x=a")] == [
        ("lean", 3), ("pending_s", 15000.0), ("now", 0.5), ("x", "a")]
    with pytest.raises(SystemExit, match="give `vqueue:SRC`"):
        replay.queue("model", {"lean": 3})
    with pytest.raises(SystemExit, match="research/vqueue.py"):
        replay.queue("vqueue:no-such-checkout")
