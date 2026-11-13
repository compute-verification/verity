"""`verity_vllm.engine.chunks.attribute` on the fields it reads back: a timing record with no requests (absent, null,
empty) attributes nothing; the partial-prefill steps it totals and the loop's fields a reverted correction restores are the
ones it wrote itself a few lines before.  Hand-built collectors, no engine."""
from __future__ import annotations

from typing import Any

import pytest

from verity_vllm.engine import chunks

NOTHING: dict[str, Any] = {"source": "outputs", "steps_scheduled_tokens": None, "max_prefill_chunks": 0, "chunked_prefill_observed": False,
           "partial_prefill_steps_total": 0, "internal_request_ids": None, "consistent": True, "problems": [], "corrections": None}


def _collector(nonempty, new_tokens, computed=None):
    col = chunks.Collector(scheduler=object() if computed is not None else None)
    col.nonempty, col.new_tokens = list(nonempty), [dict(n) for n in new_tokens]
    if computed is not None:
        col.computed = [dict(c) for c in computed]
    return col


@pytest.mark.parametrize("timing", [{}, {"requests": None}, {"requests": {}}], ids=["absent", "null", "empty"])
def test_a_timing_record_without_requests_attributes_nothing(timing):
    out = chunks.attribute(timing, chunks.Collector(), {})
    assert out == NOTHING and timing["chunk_attribution"] is out


def test_the_partial_prefill_steps_of_every_request_are_totalled():
    # the outputs of iteration j come from the forward of iteration j - 1 (lag 1); two forwards serve no token before the first
    timing: dict[str, Any] = {"engine_steps": 4, "output_lag_steps": 1,
                              "requests": {"a": {"first_step": 0, "finish_step": 3, "n_generated": 2},
                                           "b": {"first_step": 1, "finish_step": 3, "n_generated": 2}}}
    col = _collector([False, True, True, True, True], [{}, {}, {}, {"a": 1, "b": 1}, {"a": 1, "b": 1}])
    out = chunks.attribute(timing, col, {"a": 9, "b": 9})
    assert timing["requests"]["a"]["partial_prefill_steps"] == [0, 1] and timing["requests"]["b"]["partial_prefill_steps"] == [1]
    assert out == {**NOTHING, "max_prefill_chunks": 3, "chunked_prefill_observed": True, "partial_prefill_steps_total": 3}


def test_a_reverted_correction_restores_the_three_fields_the_loop_wrote():
    # one request on a sync engine: its last forward is freed inside step(), so the schedule's deltas read a lag of 1 for a true 0
    loop = {"arrive_step_realized": 0, "first_step": 0, "finish_step": 2}
    r0 = {"arrive_step": 0, **loop, "finish_reason": "length", "admit_iter": 0, "finish_iter": 2, "n_generated": 3}
    timing: dict[str, Any] = {"engine_steps": 3, "output_lag_steps": 0, "requests": {"r0": r0}}
    col = _collector([True, True, True], [{"r0": 1}] * 3, [{"r0-01234567": (2, 2)}, {"r0-01234567": (3, 2)}, {}])
    out = chunks.attribute(timing, col, {"r0": 2})
    c = out["corrections"]
    assert c["applied"] is False and c["reason"].endswith("REVERTED: the corrected model failed the attribution's own consistency check")
    assert c["loop"] == {"engine_steps": 3, "output_lag_steps": 0, "requests": {"r0": loop}}
    assert c["corrected"] == {"engine_steps": 2, "output_lag_steps": 1, "requests": {"r0": {**loop, "finish_step": 1}}}
    assert (timing["engine_steps"], timing["output_lag_steps"]) == (3, 0)
    assert {k: timing["requests"]["r0"][k] for k in loop} == loop
    reverted = ("scheduler-state correction reverted: the corrected model (engine_steps 2, output_lag_steps 1) failed 1 "
                "consistency check(s); the loop's step fields stand")
    assert out["problems"] == [reverted, "r0: decode forwards [1].. != token-producing forwards [0, 1].."]
    assert not out["consistent"] and out["internal_request_ids"] == {"r0": "r0-01234567"}


def test_a_reverted_correction_takes_off_a_field_the_loop_never_wrote():
    # the loop left no `arrive_step_realized`; the correction writes one, and the revert removes it again
    r0 = {"first_step": 0, "finish_step": 2, "admit_iter": 0, "finish_iter": 2, "n_generated": 3}
    timing: dict[str, Any] = {"engine_steps": 3, "output_lag_steps": 0, "requests": {"r0": r0}}
    col = _collector([True, True, True], [{"r0": 1}] * 3, [{"r0-01234567": (2, 2)}, {"r0-01234567": (3, 2)}, {}])
    out = chunks.attribute(timing, col, {"r0": 2})
    assert out["corrections"]["applied"] is False and out["corrections"]["loop"]["requests"]["r0"]["arrive_step_realized"] is None
    assert "arrive_step_realized" not in timing["requests"]["r0"] and timing["requests"]["r0"]["finish_step"] == 2
