"""a5's session pool (`a5_pool`): an audit's sampled members' sessions and Lean's checks of them run side by side, each
admitted when its measured peak memory plus the headroom fits under the budget, each exactly once and never again. One job of
each kind is the members' order, one session at a time."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402
import a5_pool as POOL  # noqa: E402

S, L = POOL.SESSION, POOL.LEAN


def job(kind: str, key: str, need: float | None = 0.0) -> POOL.Job:
    return POOL.Job(kind, key, need, lambda: key)


def drain(sched: POOL.Schedule, order: list[str], lean_need: float | None = None) -> None:
    """Run the schedule to the end with no threads: the oldest running job finishes first, and a finished session's Lean
    check joins the pool (``lean_need``: None, no checks)."""
    while not sched.idle():
        order += [j.key for j in sched.start()]
        assert sched.running, "jobs wait and none runs"
        done = sched.running[0]
        sched.finish(done)
        if done.kind == S and lean_need is not None:
            sched.add(job(L, "lean-" + done.key, lean_need))


def options(*argv: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    a5.pool_options(ap)
    return ap.parse_args(list(argv))


def test_the_default_is_one_session_and_one_lean_check_at_a_time_with_no_budget():
    a = options()
    assert (a.session_jobs, a.lean_jobs, a.memory_budget_gb, a.pool_memory) == (1, 1, None, None)


def test_session_jobs_and_lean_jobs_take_more_than_one(tmp_path):
    f = tmp_path / "peaks.json"
    f.write_text(json.dumps({"session": {"LinearARow_v1@k_log=24": 16}}))
    a = options("--session-jobs", "16", "--lean-jobs", "4", "--memory-budget-gb", "430.5", "--pool-memory", str(f))
    assert (a.session_jobs, a.lean_jobs, a.memory_budget_gb, a.pool_memory) == (16, 4, 430.5, str(f))


@pytest.mark.parametrize("n", ["0", "-1", "two", "1.5"])
def test_a_pool_of_no_jobs_is_refused(n):
    with pytest.raises(argparse.ArgumentTypeError, match="at least 1 job"):
        POOL.jobs(n)


@pytest.mark.parametrize("gb", ["0", "-440", "lots", "nan", "inf", "4e2", ""])
def test_a_budget_that_is_no_positive_number_of_gb_is_refused_with_the_command_line(gb):
    with pytest.raises(argparse.ArgumentTypeError, match="positive number of GB"):
        POOL.budget(gb)
    assert POOL.budget("440") == 440.0 and POOL.budget(" 430.5 ") == 430.5


@pytest.mark.parametrize("text", [None, "{", "[]", '{"sessions": {}}', '{"session": {"Secret_v1{n=7}": 0}}', '{"lean": 9}',
                                  '{"session": {"Secret_v1{n=7}": "16"}}', '{"session": {"Secret_v1{n=7}": 16}, "Secret_v1{n=8}": {}}'])
def test_a_pool_memory_file_the_pool_cannot_read_is_refused_with_the_command_line(tmp_path, capsys, text):
    """An audit reads the file only after its draw (`a5.audit`), so one that is missing (``text`` None) or malformed is
    refused where the options are parsed, before any coin: by `POOL.peaks`, and so by every parser `a5.pool_options` fills.
    The refusal holds none of the file's keys: an exact file's are the drawn members' templates."""
    f = tmp_path / "peaks.json"
    if text is not None:
        f.write_text(text)
    with pytest.raises(argparse.ArgumentTypeError, match="no pool-memory file the pool can read") as refused:
        POOL.peaks(str(f))
    assert "Secret" not in str(refused.value) and refused.value.__cause__ is None and refused.value.__suppress_context__
    with pytest.raises(SystemExit) as e:
        options("--session-jobs", "4", "--pool-memory", str(f))
    err = capsys.readouterr().err
    assert e.value.code == 2 and "argument --pool-memory" in err and "Secret" not in err


def test_a_need_is_the_measured_peak_plus_ten_percent_by_template_then_size_class(tmp_path):
    f = tmp_path / "peaks.json"
    f.write_text(json.dumps({"session": {"LinearARow_v1@k_log=24": 16, "LinearOut_v1{N=3}": 120}, "lean": {"LinearARow_v1@k_log=24": 9.5}}))
    needs = POOL.Needs.read(f)
    assert needs.need(S, "LinearARow_v1{N=7}", 24) == pytest.approx(17.6)
    assert needs.need(S, "LinearOut_v1{N=3}", 27) == pytest.approx(132.0)        # the template itself, ahead of its size class
    assert needs.need(L, "LinearARow_v1{N=7}", 24) == pytest.approx(10.45)
    assert needs.need(S, "LinearARow_v1{N=7}", 25) is None and needs.need(L, "LinearOut_v1{N=3}", 27) is None
    assert POOL.Needs.read(None).need(S, "LinearARow_v1{N=7}", 24) is None


@pytest.mark.parametrize("doc", [{"sessions": {}}, {"session": {"A@k_log=1": 0}}, {"lean": {"A@k_log=1": "9"}}, {"lean": {"A": True}}])
def test_a_malformed_pool_memory_file_is_refused(tmp_path, doc):
    f = tmp_path / "peaks.json"
    f.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="pool-memory file"):
        POOL.Needs.read(f)


def test_a_side_by_side_prover_takes_its_memory_share_of_the_cores_and_at_least_an_even_share():
    assert POOL.threads(17.6, 430.0, 16, 128) == 8          # a small member: the even share
    assert POOL.threads(135.0, 430.0, 16, 128) == 40        # a large one: its share of the budget
    assert POOL.threads(470.0, 430.0, 16, 128) == 128       # over the budget, it runs alone: every core
    assert POOL.threads(None, 430.0, 16, 128) == 8 and POOL.threads(17.6, None, 4, 128) == 32
    assert POOL.threads(1.0, 430.0, 64, 16) == 1


def test_one_job_of_each_kind_and_no_budget_is_the_members_order_one_session_at_a_time():
    sched, order = POOL.Schedule({S: 1, L: 1}), []
    for i in range(5):
        sched.add(job(S, f"m{i}", None))
    drain(sched, order)
    assert order == [f"m{i}" for i in range(5)] and sched.peak_jobs == 1


def test_sessions_run_side_by_side_up_to_the_slots():
    sched = POOL.Schedule({S: 3, L: 1})
    for i in range(5):
        sched.add(job(S, f"m{i}", None))
    assert [j.key for j in sched.start()] == ["m0", "m1", "m2"] and sched.start() == []
    sched.finish(sched.running[1])
    assert [j.key for j in sched.start()] == ["m3"]


def test_admitted_jobs_never_exceed_the_budget():
    sched, order = POOL.Schedule({S: 8, L: 4}, budget_gb=100.0), []
    for i, need in enumerate([17.6, 30.8, 39.6, 17.6, 96.0, 30.8, 17.6, 39.6]):
        sched.add(job(S, f"m{i}", need))
    while not sched.idle():
        order += [j.key for j in sched.start()]
        assert sched.used_gb() <= 100.0
        done = sched.running[0]
        sched.finish(done)
        if done.kind == S:
            sched.add(job(L, "lean-" + done.key, 10.45))
    assert sorted(order) == sorted([f"m{i}" for i in range(8)] + [f"lean-m{i}" for i in range(8)])
    assert 1 < sched.peak_jobs and sched.peak_gb <= 100.0


def test_sessions_wait_largest_first_and_a_smaller_one_fills_the_room_left():
    sched = POOL.Schedule({S: 4, L: 1}, budget_gb=100.0)
    for key, need in [("small", 20.0), ("big", 70.0), ("mid", 40.0), ("small2", 20.0)]:
        sched.add(job(S, key, need))
    assert [j.key for j in sched.start()] == ["big", "small"]       # mid doesn't fit beside big; the next that fits does
    sched.finish(next(j for j in sched.running if j.key == "big"))
    assert [j.key for j in sched.start()] == ["mid", "small2"]


def test_lean_checks_are_admitted_ahead_of_waiting_sessions():
    sched = POOL.Schedule({S: 2, L: 2}, budget_gb=50.0)
    for i in range(4):
        sched.add(job(S, f"m{i}", 20.0))
    first = sched.start()
    assert [j.key for j in first] == ["m0", "m1"]
    sched.finish(first[0])
    sched.add(job(L, "lean-m0", 25.0))
    assert [j.key for j in sched.start()] == ["lean-m0"]       # 20 + 25 fits; m2 behind it would make 65


@pytest.mark.parametrize("need", [None, 470.0])
def test_a_job_with_no_measured_peak_or_over_the_budget_alone_runs_alone(need):
    sched = POOL.Schedule({S: 4, L: 2}, budget_gb=100.0)
    sched.add(job(S, "m0", 20.0))
    assert [j.key for j in sched.start()] == ["m0"]
    sched.add(job(S, "huge", need))
    sched.add(job(S, "m2", 20.0))
    assert sched.start() == []      # huge is first in line and waits for an empty pool; m2 doesn't pass it
    sched.finish(sched.running[0])
    assert [j.key for j in sched.start()] == ["huge"]
    sched.add(job(L, "lean-m0", 10.0))
    assert sched.start() == []      # nothing beside it, a Lean check included
    sched.finish(sched.running[0])
    assert [j.key for j in sched.start()] == ["lean-m0", "m2"]


def test_a_job_joins_the_pool_once():
    sched, j = POOL.Schedule({S: 2, L: 1}), job(S, "m0")
    sched.add(j)
    assert sched.start() == [j]
    sched.finish(j)
    with pytest.raises(RuntimeError, match="nothing is rerun"):
        sched.add(j)


def test_a_budget_is_positive():
    with pytest.raises(ValueError, match="positive"):
        POOL.Schedule({S: 1, L: 1}, budget_gb=0)


def pool(n: int, limits: dict[str, int], budget: float | None = None, need: float | None = 0.0, session=None, lean=None):
    """`POOL.run` on ``n`` sessions; -> its results, its record and each job's count of runs."""
    ran: dict[str, int] = {}
    lock = threading.Lock()

    def counted(key: str, body):
        def go():
            with lock:
                ran[key] = ran.get(key, 0) + 1
            return body(key) if body else key
        return go

    sessions = [POOL.Job(S, f"m{i}", need, counted(f"m{i}", session)) for i in range(n)]
    try:
        results, record = POOL.run(sessions, lambda s, out: POOL.Job(L, f"lean-{out}", need, counted(f"lean-{out}", lean)), limits, budget)
    except BaseException as e:      # noqa: BLE001 - the test reads it
        return e, None, ran
    return results, record, ran


def test_every_session_and_its_lean_check_run_exactly_once_and_results_keep_the_members_order():
    results, record, ran = pool(12, {S: 4, L: 2}, lean=lambda key: key.upper())
    assert results == [f"LEAN-M{i}" for i in range(12)]
    assert ran == {**{f"m{i}": 1 for i in range(12)}, **{f"lean-m{i}": 1 for i in range(12)}}
    assert record["session_jobs"] == 4 and record["lean_jobs"] == 2 and len(record["starts"]) == 24


def test_sessions_really_overlap_and_never_more_than_the_slots():
    """Four sessions each wait until all four have started: the pool must hold them at once. A fifth at a time would show in
    ``most``."""
    gate, lock, state = threading.Barrier(4), threading.Lock(), {"now": 0, "most": 0}

    def session(key: str) -> str:
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        gate.wait()
        with lock:
            state["now"] -= 1
        return key

    results, record, ran = pool(8, {S: 4, L: 1}, session=session)
    assert not isinstance(results, BaseException), results
    assert state["most"] == 4 and all(n == 1 for n in ran.values()) and len(ran) == 16


def test_one_job_of_each_kind_runs_the_sessions_in_the_members_order():
    order: list[str] = []
    results, record, ran = pool(6, {S: 1, L: 1}, session=lambda key: order.append(key) or key)
    assert order == [f"m{i}" for i in range(6)] and record["most_side_by_side"] <= 2


def test_a_session_that_raises_is_not_rerun_and_the_pool_raises_it():
    def session(key: str) -> str:
        if key == "m2":
            raise RuntimeError("prover timed out")
        return key

    err, record, ran = pool(6, {S: 1, L: 1}, session=session)
    assert isinstance(err, RuntimeError) and str(err) == "prover timed out"
    assert ran["m2"] == 1 and "lean-m2" not in ran
    assert not any(k in ran for k in ("m3", "m4", "m5")), "nothing starts after a failure"
    assert all(n == 1 for n in ran.values())


def test_a_failed_session_is_handed_to_its_checks_as_it_is_and_never_run_again():
    """A session that ends without a proof returns its record of that (a5's ``live`` with ``accepted`` False): the pool hands
    it on once, so the audit's verdicts show the failure."""
    seen: list[str] = []
    results, record, ran = pool(3, {S: 2, L: 2}, session=lambda key: f"{key}:failed" if key == "m1" else key,
                                lean=lambda key: seen.append(key) or key)
    assert sorted(seen) == ["lean-m0", "lean-m1:failed", "lean-m2"] and ran["m1"] == 1
    assert results == ["lean-m0", "lean-m1:failed", "lean-m2"]


def test_the_pool_records_the_jobs_with_no_measured_peak():
    results, record, ran = pool(2, {S: 2, L: 1}, budget=64.0, need=None)
    assert record["unmeasured"] == ["lean-m0", "lean-m1", "m0", "m1"] and record["most_side_by_side"] == 1
    assert record["memory_budget_gb"] == 64.0 and record["headroom"] == POOL.HEADROOM
