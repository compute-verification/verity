"""The served advice the stand-in compiler takes (`pipeline.advice_windows`): the requests it compiles, and the advice it refuses by
name."""
from __future__ import annotations

import pytest
from verity_vllm.pipeline.advice_code import format as A

from verity_vllm.pipeline import advice_windows as AW


def _served(*steps, release) -> A.Advice:
    """I1 advice of q0 (3 prompt tokens, 2 kept outputs) and q1 (1 and 1), each step's rows given as (request, start, n)."""
    reqs = (A.RequestAdvice("q0", 3, 2, "length"), A.RequestAdvice("q1", 1, 1, "length"))
    rows = [[A.Row(q, "decode" if n == 1 and start >= reqs[q].prompt_len else "prefill", start, n) for q, start, n in step] for step in steps]
    return A.Advice(reqs, tuple(A.StepAdvice(tuple(r)) for r in rows), release)


def test_the_row_async_scheduling_runs_past_the_last_kept_output_is_no_program_row():
    advice = _served([(0, 0, 3)], [(0, 3, 1), (1, 0, 1)], [(0, 4, 1)], release=(0, 1, 1))
    assert AW.refusals(advice) == {}
    assert AW.requests_of(advice) == [{"request_id": "q0", "arrive_step": 0, "prompt_len": 3, "max_tokens": 2},
                                      {"request_id": "q1", "arrive_step": 1, "prompt_len": 1, "max_tokens": 1}]


@pytest.mark.parametrize("steps, release, why", [
    pytest.param(([(0, 0, 2)], [(0, 2, 1), (1, 0, 1)], [(0, 3, 1)]), (1, 1, 2), "a chunked prefill, 2 of its 3 prompt tokens at step 0",
                 id="chunked-prefill"),
    pytest.param(([(0, 0, 3)], [(1, 0, 1)], [(0, 3, 1)]), (0, 1, 2), r"its row at step 2 is \(start, n\) = \(3, 1\), not the next decode",
                 id="a-gap"),
])
def test_a_request_the_stand_in_cannot_compile_refuses_the_advice_by_name(steps, release, why):
    """Served advice the stand-in's Programs of record don't run (COMPILER's v0 refusals): refused naming the request and why, never
    compiled as if its prompt ran whole."""
    advice = _served(*steps, release=release)
    assert list(AW.refusals(advice)) == [0]
    with pytest.raises(ValueError, match=rf"compiles 1 of the advice's 2 requests \(q0: {why}\)"):
        AW.requests_of(advice)


def test_the_stand_in_advice_refuses_a_workload_that_leaves_a_step_idle():
    doc = {"requests": [{"prompt_token_ids": [1, 2], "max_tokens": 2, "arrive_step": 0}, {"prompt_token_ids": [3], "max_tokens": 1, "arrive_step": 3}]}
    with pytest.raises(ValueError, match=r"engine steps \[2\] idle"):
        AW.advice_of_workload(doc)
    doc["requests"][1]["arrive_step"] = 1
    assert [len(s.rows) for s in AW.advice_of_workload(doc).steps] == [1, 2]
