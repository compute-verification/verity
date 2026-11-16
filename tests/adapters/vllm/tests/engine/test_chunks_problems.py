"""`verity_vllm.engine.chunks.attribute` on records that do not agree with themselves: each problem it names, and the
branches that read a request the scheduler never advanced, an id the timing does not hold, or no finish iteration at all.
Hand-built collectors, no engine; every expectation is what the function gave before it was split."""
from __future__ import annotations

import copy
from typing import Any

from verity_vllm.engine import chunks

CHUNK = {"step": 0, "tokens": 2, "computed_before": 0, "prompt_complete": True}
CONTRADICTS = ("scheduler-state inference: derived output lag 1 contradicts the resolved async_scheduling LAG 0 (the record's lag is the "
               "resolved profile, never a schedule-delta inference) -- the loop's engine_steps {steps} / output_lag_steps 0 / finish_step "
               "values stand")
REVERTED = ("scheduler-state correction reverted: the corrected model (engine_steps {steps}, output_lag_steps {lag}) failed {n} "
            "consistency check(s); the loop's step fields stand")


def _attribute(timing: dict[str, Any], nonempty, new_tokens, computed=None, *, prompt_len, **kw) -> tuple[dict, dict]:
    col = chunks.Collector(scheduler=object() if computed is not None else None)
    col.nonempty, col.new_tokens = list(nonempty), [dict(n) for n in new_tokens]
    if computed is not None:
        col.computed = [dict(c) for c in computed]
    timing = copy.deepcopy(timing)
    return chunks.attribute(timing, col, prompt_len, **kw), timing


def _loop(first: int, finish: int, n: int, admit: int = 0, finish_iter: int | None = None) -> dict[str, Any]:
    rec = {"admit_iter": admit, "first_step": first, "finish_step": finish, "arrive_step_realized": first, "n_generated": n}
    return rec if finish_iter is None else {**rec, "finish_iter": finish_iter}


def test_requests_whose_output_lags_differ_are_a_named_problem_and_the_correction_is_reverted():
    timing = {"engine_steps": 3, "output_lag_steps": 1, "requests": {"a": _loop(0, 1, 2, finish_iter=3), "b": _loop(0, 2, 3, finish_iter=3)}}
    computed = [{"a": (2, 2), "b": (2, 2)}, {"a": (3, 2), "b": (3, 2)}, {"b": (4, 2)}, {}]
    out, after = _attribute(timing, [False, True, True, True], [{}, {"a": 1, "b": 1}, {"a": 1, "b": 1}, {"b": 1}], computed,
                            prompt_len={"a": 2, "b": 2})
    assert out["problems"] == [
        REVERTED.format(steps=3, lag=2, n=5),
        "per-request output lags differ: [1, 2]",
        "a: decode forwards [1].. != token-producing forwards [0]..",
        "a: finish_step 1 != the forward of the last served token 0",
        "b: decode forwards [1, 2].. != token-producing forwards [0, 1]..",
        "b: finish_step 2 != the forward of the last served token 1"]
    assert out["corrections"]["applied"] is False and out["corrections"]["corrected"]["output_lag_steps"] == 2
    assert (after["engine_steps"], after["output_lag_steps"]) == (3, 1)
    assert [after["requests"][r]["finish_step"] for r in "ab"] == [1, 2]
    assert out["steps_scheduled_tokens"] == [{"a": 2, "b": 2}, {"a": 1, "b": 1}, {"b": 1}]


def test_with_no_finish_iteration_the_derived_lag_is_the_iterations_less_the_forwards():
    timing = {"engine_steps": 2, "output_lag_steps": 0, "requests": {"a": {"first_step": 0, "finish_step": 1, "n_generated": 2}}}
    out, after = _attribute(timing, [True, True, False], [{"a": 1}, {"a": 1}, {}], [{"a": (2, 2)}, {"a": (3, 2)}, {}], prompt_len={"a": 2})
    c = out["corrections"]
    assert c["corrected"] == {"engine_steps": 2, "output_lag_steps": 1, "requests": {"a": {"finish_step": 1}}}
    assert c["loop"]["requests"] == {"a": {"arrive_step_realized": None, "first_step": 0, "finish_step": 1}} and c["applied"] is False
    assert out["problems"] == [
        REVERTED.format(steps=2, lag=1, n=2),
        "a: decode forwards [1].. != token-producing forwards [0]..",
        "a: finish_step 1 != the forward of the last served token 0"]
    assert (after["engine_steps"], after["output_lag_steps"]) == (2, 0) and "arrive_step_realized" not in after["requests"]["a"]
    assert after["requests"]["a"]["token_steps"] == [0, 0]


def test_a_request_the_scheduler_never_advanced_keeps_the_loops_finish_and_falls_back_to_the_outputs():
    timing = {"engine_steps": 2, "output_lag_steps": 0, "requests": {"a": _loop(0, 1, 2, finish_iter=1), "z": _loop(1, 1, 1, admit=1, finish_iter=1)}}
    out, after = _attribute(timing, [True, True], [{"a": 1}, {"a": 1, "z": 1}], [{"a": (2, 2)}, {"a": (3, 2)}], prompt_len={"a": 2, "z": 1})
    assert out["problems"] == ["z: no prefill chunk seen in the scheduler state; falling back to the outputs inference"]
    assert out["corrections"] is None and out["steps_scheduled_tokens"] == [{"a": 2}, {"a": 1}]
    a, z = after["requests"]["a"], after["requests"]["z"]
    assert (a["prefill_steps"], a["prefill_chunks"], a["token_steps"]) == ([0], [CHUNK], [0, 1])
    assert (z["prefill_steps"], z["prefill_chunks"], z["token_steps"], z["finish_step"]) == ([1], None, [1], 1)


def test_the_outputs_source_names_a_token_with_no_forward_and_a_count_that_is_not_the_generated_one():
    """A zero count and an id the timing does not hold add nothing; a request with no first step has no prefill."""
    timing = {"engine_steps": 1, "output_lag_steps": 0, "requests": {"a": {"first_step": 0, "finish_step": 0, "n_generated": 3},
                                                                    "q": {"n_generated": 0}}}
    out, after = _attribute(timing, [False, True], [{"a": 1, "zz": 4}, {"a": 0}], prompt_len={"a": 2})
    assert out["source"] == "outputs" and out["problems"] == [
        "iteration 0: 1 token(s) of a returned but no forward is attributed to iteration 0",
        "a: 1 token-producing forwards for 3 generated tokens"]
    assert set(after["requests"]) == {"a", "q"} and after["requests"]["a"]["token_steps"] == [0]
    assert after["requests"]["q"] == {"n_generated": 0, "prefill_steps": [], "prefill_chunks": None, "n_prefill_chunks": 0,
                                      "partial_prefill_steps": [], "first_token_step": None, "token_steps": []}


def test_an_id_the_timing_does_not_hold_is_scheduled_and_attributed_to_no_request():
    timing = {"engine_steps": 3, "output_lag_steps": 0, "requests": {"r0": _loop(0, 2, 3, finish_iter=2)}}
    computed = [{"r0-01234567": (2, 2), "other": (5, 5)}, {"r0-01234567": (3, 2), "other": (6, 5)}, {}]
    out, after = _attribute(timing, [True, True, True], [{"r0": 1}] * 3, computed, prompt_len={"r0": 2}, resolved_lag=0)
    assert out["problems"] == [CONTRADICTS.format(steps=3), "forward 2: a forward ran but the scheduler advanced no request",
                               "r0: decode forwards [1].. != token-producing forwards [1, 2].."]
    assert out["steps_scheduled_tokens"] == [{"r0": 2, "other": 5}, {"r0": 1, "other": 1}, {}]
    assert out["internal_request_ids"] == {"r0": "r0-01234567"} and set(after["requests"]) == {"r0"}
    assert out["corrections"]["applied"] is False and out["corrections"]["reason"].endswith(
        "; NOT APPLIED: the derived lag contradicts the resolved async_scheduling")
    assert (after["engine_steps"], after["output_lag_steps"], after["requests"]["r0"]["token_steps"]) == (3, 0, [0, 1, 2])


def test_a_schedule_that_advanced_where_the_outputs_show_no_forward_is_a_named_problem():
    timing = {"engine_steps": 2, "output_lag_steps": 0, "requests": {"r0": _loop(0, 1, 2, finish_iter=2)}}
    computed = [{"r0": (2, 2), "other": (5, 5)}, {"r0": (3, 2)}, {}]
    out, after = _attribute(timing, [False, True, True], [{}, {"r0": 1}, {"r0": 1}], computed, prompt_len={"r0": 2}, resolved_lag=0)
    assert out["problems"] == [
        CONTRADICTS.format(steps=2),
        "iteration 0: the scheduler advanced ['other', 'r0'] but no forward is attributed to this iteration",
        "forward 1: a forward ran but the scheduler advanced no request",
        "r0: no prefill chunk seen in the scheduler state; falling back to the outputs inference"]
    assert out["steps_scheduled_tokens"] == [{"r0": 1}, {}]
    r0 = after["requests"]["r0"]
    assert (r0["prefill_steps"], r0["prefill_chunks"], r0["token_steps"], r0["finish_step"]) == ([0], None, [0, 1], 1)


def test_a_last_prefill_chunk_that_is_not_the_first_tokens_forward_is_a_named_problem():
    timing = {"engine_steps": 3, "output_lag_steps": 0, "requests": {"a": _loop(0, 2, 2, finish_iter=2)}}
    out, after = _attribute(timing, [False, True, True], [{}, {"a": 1}, {"a": 1}], [{"a": (4, 4)}, {"a": (5, 4)}, {"a": (6, 4)}],
                            prompt_len={"a": 4})
    assert out["problems"] == ["a: last prefill chunk at forward 0 but the first served token came from forward 1",
                               "a: decode forwards [1, 2].. != token-producing forwards [2].."]
    assert out["corrections"] is None and after["requests"]["a"]["prefill_chunks"] == [{**CHUNK, "tokens": 4}]
    assert (after["requests"]["a"]["first_token_step"], after["requests"]["a"]["token_steps"]) == (1, [1, 2])
