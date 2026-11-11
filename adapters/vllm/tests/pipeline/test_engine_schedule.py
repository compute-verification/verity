"""the required population is placed on the committed-step axis by the engine's schedule of record, not by the
declared arrive_steps: on a workload with a gap (r0's cap 2, r1 declared at step 7) the serving loop admits r1 as soon as the engine idles
and the bookkeeping iteration after a finish advances no committed step, so the declared axis drifts (measured: 8,850 of 180,540
identities 'unbound' = the drift, every launch exact).  Torch-free."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

# commit_delta imports torch/vllm at module import; lift engine_schedule out of the source without importing the module
_SRC = pathlib.Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / "commit.py"


def _engine_schedule():
    src = _SRC.read_text()
    i = src.index("def engine_schedule(")
    j = src.index("\ndef assert_pristine(")
    ns: dict = {}
    exec("from __future__ import annotations\n" + src[i:j], ns)          # noqa: S102 -- the function under test, verbatim
    return ns["engine_schedule"]


def _rows(schedule: dict[str, tuple[int, int]]) -> dict[int, list[tuple[str, int, int]]]:
    """schedule: request -> (first committed step, n steps) -> per-step rows in request order."""
    out: dict[int, list[tuple[str, int, int]]] = {}
    for r, (a, n) in schedule.items():
        for s in range(a, a + n):
            lo = sum(1 for x in out.get(s, []))
            out.setdefault(s, []).append((r, lo, lo + 1))
    return out


def test_h100_gap_workload_places_the_population_by_the_observed_schedule():
    es = _engine_schedule()
    declared = {"r0": 0, "r1": 7, "r2": 24, "r3": 26}
    caps = {"r0": 2, "r1": 19, "r2": 128, "r3": 119}
    observed = {"r0": (0, 2), "r1": (2, 19), "r2": (21, 128), "r3": (24, 119)}
    sch = es(_rows(observed), declared, caps)
    assert sch["use"] == "observed" and sch["why"] is None
    assert sch["observed_arrive_steps"] == {"r0": 0, "r1": 2, "r2": 21, "r3": 24}
    assert sch["all_equal_declared"] is False and sch["equal_declared"] == {"r0": True, "r1": False, "r2": False, "r3": False}
    assert sch["contiguous"] is True and sch["steps_equal_caps"] is True and sch["complete"] is True


def test_declared_schedule_stands_when_the_engine_matches_it():
    es = _engine_schedule()
    declared = {"r0": 0, "r1": 8}
    caps = {"r0": 117, "r1": 128}
    sch = es(_rows({"r0": (0, 117), "r1": (8, 128)}), declared, caps)
    assert sch["use"] == "observed" and sch["all_equal_declared"] is True and sch["observed_arrive_steps"] == declared


def test_single_request_or_no_rows_keeps_the_declaration():
    es = _engine_schedule()
    sch = es(None, {"r0": 0}, {"r0": 32})
    assert sch["use"] == "declared" and "declared arrive_steps" in sch["why"]


def test_a_request_never_scheduled_or_preempted_or_short_is_named_and_the_declaration_stands():
    es = _engine_schedule()
    declared = {"r0": 0, "r1": 7}
    caps = {"r0": 2, "r1": 19}
    sch = es(_rows({"r0": (0, 2)}), declared, caps)
    assert sch["use"] == "declared" and "never scheduled: ['r1']" in sch["why"]
    rows = _rows({"r0": (0, 2), "r1": (2, 19)})
    del rows[10]                                                    # r1 preempted at step 10
    sch = es(rows, declared, caps)
    assert sch["use"] == "declared" and "non-contiguous" in sch["why"]
    sch = es(_rows({"r0": (0, 2), "r1": (2, 18)}), declared, caps)  # r1 ran one step short of its cap
    # the expected count per request is its executed prefix of record (the cap on a length stop)
    assert sch["use"] == "declared" and "scheduled steps != expected (executed prefix of record): ['r1']" in sch["why"]
