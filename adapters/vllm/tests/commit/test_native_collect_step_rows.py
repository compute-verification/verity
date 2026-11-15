"""The row order the windowed committer (`native_collect`) records of a step, `step_rows[step]`: the batch's requests in the
batch's order, each with the rows of the flattened token order it owns, on the CPU as in `test_native_collect_plan_cache`.

A request's rows are counted by the batch's own per-row token counts where it has a list of them, and by the scheduler's
`num_scheduled_tokens`, by request, where it has none. A batch whose counts are there and are no list of numbers is counted by
the scheduler's too, and the committer then says so in `step_rows_why`, the reason the binding's population records beside the
rows. The serving thread expands the rows itself or leaves a snapshot for the worker (`defer`): both give the same rows and the
same reason.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.commit import test_native_collect_plan_cache as PC

STEP = 4
#: the scheduler's tokens by request, which are not the batch's counts by row below, so the rows say whose were read
SCHEDULED = {"a": 1, "b": 1, "c": 1}
BY_SCHEDULER = [("b", 0, 1), ("a", 1, 2), ("c", 2, 3)]
BY_BATCH = [("b", 0, 2), ("a", 2, 5), ("c", 5, 9)]
DEFER = pytest.mark.parametrize("defer", [True, False], ids=["the-worker-expands", "the-serving-thread-expands"])


def rows(monkeypatch: pytest.MonkeyPatch, per_row: Any, *, defer: bool, **batch: Any) -> tuple[Any, list[tuple[str, int, int]] | None]:
    """A committer after it recorded step `STEP`'s rows from a batch of requests b, a, c with ``per_row`` as its counts."""
    com = PC.committer(monkeypatch)
    com._cur_sched = SimpleNamespace(num_scheduled_tokens=dict(SCHEDULED))
    com._cur_batch = SimpleNamespace(**({"req_ids": ["b", "a", "c"], "num_reqs": 3, "num_scheduled_tokens": per_row} | batch))
    job = SimpleNamespace(step=STEP)
    com._record_step_rows(job, defer=defer)
    assert hasattr(job, "rows_snapshot") is defer
    if defer:
        com._expand_step_rows(STEP, *job.rows_snapshot)
    return com, com.step_rows.get(STEP)


@DEFER
@pytest.mark.parametrize("per_row", [[2, 3, 4], (2, 3, 4), torch.tensor([2, 3, 4]), np.array([2, 3, 4])], ids=["list", "tuple", "tensor", "array"])
def test_a_steps_rows_are_the_batchs_requests_in_its_order_by_its_own_counts(monkeypatch, per_row, defer):
    com, got = rows(monkeypatch, per_row, defer=defer)
    assert got == BY_BATCH and com.step_rows_src == "gather_batch_req_state"
    assert getattr(com, "step_rows_why", None) is None


@DEFER
def test_a_row_with_no_request_or_no_token_owns_no_rows_and_rows_past_the_batchs_count_are_not_read(monkeypatch, defer):
    _com, got = rows(monkeypatch, [2, 3, 0, 4, 5], defer=defer, req_ids=["b", None, "a", "c", "d"], num_reqs=4)
    assert got == [("b", 0, 2), ("c", 2, 6)]


@DEFER
@pytest.mark.parametrize("per_row", [None, {"b": 2, "a": 3, "c": 4}], ids=["none", "by-request"])
def test_a_batch_with_no_list_of_counts_is_counted_by_the_schedulers_by_request(monkeypatch, per_row, defer):
    com, got = rows(monkeypatch, per_row, defer=defer)
    assert got == BY_SCHEDULER
    assert getattr(com, "step_rows_why", None) is None


@DEFER
@pytest.mark.parametrize("per_row, caught", [
    (7, "'int' object is not iterable"),
    (torch.tensor(7), "'int' object is not iterable|iteration over a 0-d tensor"),      # read as a number first by the snapshot, or as it is
    ([2, None, 4], r"int\(\) argument must be .*'NoneType'"),
], ids=["a-number", "a-tensor-of-no-dimension", "a-row-with-no-count"])
def test_counts_that_are_no_list_of_numbers_are_counted_by_the_schedulers_and_the_committer_says_so(monkeypatch, per_row, caught, defer):
    """The rows are the scheduler's, as they were; what is new is the reason, which names the step and what was caught."""
    com, got = rows(monkeypatch, per_row, defer=defer)
    assert got == BY_SCHEDULER
    assert re.fullmatch(rf"step {STEP}: its rows are counted by the scheduler's num_scheduled_tokens, by request: the batch's "
                        rf"is no list of numbers \(TypeError: ({caught})\)", com.step_rows_why), com.step_rows_why


@DEFER
def test_a_count_that_is_text_and_no_number_is_not_counted_by_the_schedulers(monkeypatch, defer):
    """Only a `TypeError` falls back: this one goes up to the step's end, which records it among the committer's `errors`."""
    with pytest.raises(ValueError, match="invalid literal for int"):
        rows(monkeypatch, [2, "three", 4], defer=defer)


def test_a_step_with_no_batch_records_no_rows_and_why(monkeypatch):
    com = PC.committer(monkeypatch)
    com._cur_sched = SimpleNamespace(num_scheduled_tokens=dict(SCHEDULED))
    com._record_step_rows(SimpleNamespace(step=STEP))
    assert not getattr(com, "step_rows", {})
    assert com.step_rows_why == "so=SimpleNamespace ib=None runner=tests.commit.test_native_collect_plan_cache._Runner"
