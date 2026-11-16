"""What `verity_vllm.engine.chunks.Collector` keeps of each iteration of the arrival loop: whether the step returned outputs,
each request's new tokens and, with a scheduler, each request's computed and prompt token counts.  The outputs and the
scheduler are stand-ins: no vLLM."""
from __future__ import annotations

import logging
from types import SimpleNamespace as NS
from typing import Any

import pytest

from verity_vllm.engine import chunks


def _out(rid: Any, n: int) -> NS:
    return NS(request_id=rid, outputs=[NS(token_ids=list(range(n)))])


def test_each_iteration_keeps_whether_it_returned_outputs_and_each_requests_new_tokens() -> None:
    c = chunks.Collector()
    nothing: Any = None
    for outs in ([], [_out("a", 1), _out(7, 2)], [_out("a", 3)], nothing, [_out("a", 3), _out(7, 5)]):
        c.on_step(outs)
    assert c.nonempty == [False, True, True, False, True]
    assert c.new_tokens == [{}, {"a": 1, "7": 2}, {"a": 2}, {}, {"a": 0, "7": 3}]
    assert c.computed == [] and c.errors == [] and c.source == "outputs"


def test_an_output_with_no_token_ids_counts_no_new_tokens_for_its_request_and_says_so_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    c = chunks.Collector()
    with caplog.at_level(logging.DEBUG, logger="verity_vllm"):
        c.on_step([_out("a", 1)])
        c.on_step([NS(request_id="empty", outputs=[]), NS(request_id="none", outputs=None), NS(request_id="bare"), _out("a", 2),
                   NS(request_id="odd", outputs=[NS(token_ids=None)])])
        c.on_step([_out("empty", 4), _out("a", 2)])
    assert c.nonempty == [True, True, True] and c.new_tokens == [{"a": 1}, {"a": 1}, {"empty": 4, "a": 0}] and c.errors == []
    mine = [r for r in caplog.records if r.name == "verity_vllm.engine.chunks"]
    said = [r.getMessage() for r in mine]
    assert len(said) == 4 and {r.levelno for r in mine} == {logging.DEBUG}
    assert all(line.startswith("iteration 1: an output with no token ids counts no new tokens: ") for line in said)
    assert [line.rsplit(": ", 1)[1].split("(")[0] for line in said] == ["IndexError", "TypeError", "AttributeError", "TypeError"]


def test_with_a_scheduler_each_iteration_keeps_each_requests_computed_and_prompt_tokens() -> None:
    requests: dict[Any, NS] = {"a-1f": NS(num_computed_tokens=5, num_prompt_tokens=9), 3: NS(num_computed_tokens="7"),
                               "waiting": NS(num_computed_tokens=None, num_prompt_tokens=4),
                               "f": NS(num_computed_tokens=2.0, num_prompt_tokens="6")}
    c = chunks.Collector(NS(requests=requests))
    c.on_step([])
    requests["a-1f"].num_computed_tokens = 9
    del requests[3]
    c.on_step([_out("a", 1)])
    assert c.computed == [{"a-1f": (5, 9), "3": (7, None), "f": (2, 6)}, {"a-1f": (9, 9), "f": (2, 6)}]
    assert all(type(v) is int for snap in c.computed for pair in snap.values() for v in pair if v is not None)
    assert c.nonempty == [False, True] and c.new_tokens == [{}, {"a": 1}]
    assert c.errors == [] and c.source == "scheduler-state"


def test_a_scheduler_that_cannot_be_read_is_an_error_line_and_what_was_read_before_it() -> None:
    class Broken:
        @property
        def requests(self) -> dict[str, Any]:
            raise RuntimeError("x" * 300)

    c = chunks.Collector(Broken())
    c.on_step([])
    c.on_step([_out("a", 1)])
    assert c.computed == [{}, {}] and c.new_tokens == [{}, {"a": 1}]
    assert len(c.errors) == 2 and all(len(e) == 200 for e in c.errors)
    assert c.errors[0].startswith("iteration 0: scheduler read failed: RuntimeError('xxx")
    assert c.errors[1].startswith("iteration 1: scheduler read failed: RuntimeError('xxx")

    part = chunks.Collector(NS(requests={"a": NS(num_computed_tokens=1), "b": NS(num_computed_tokens="x"), "c": NS(num_computed_tokens=3)}))
    part.on_step([])
    assert part.computed == [{"a": (1, None)}]
    assert len(part.errors) == 1 and part.errors[0].startswith("iteration 0: scheduler read failed: ValueError(")
