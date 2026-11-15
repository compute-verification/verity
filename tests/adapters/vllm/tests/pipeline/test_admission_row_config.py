"""A row's plan-time scale as the admission planner reads it from a workload (`telemetry.admission.row_config_from_workload`): the
workload's own facts, and what stands for each block a workload leaves out."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import verity_vllm.pipeline.telemetry.admission as A

Record = dict[str, Any]
ABSENT = object()
ROW = "llama32-1b__bf16__h100__tp1__b2__i64__o8__mixed__t1__eager"
WORKLOAD: Record = {
    "sweep": {"row_id": ROW, "batch": 2, "max_input": 64, "max_output": 8},
    "requests": [{"prompt_len": 60, "max_tokens": 8}, {"prompt_token_ids": [1, 2, 3], "max_tokens": 5}],
    "expected_engine_steps": 9,
    "engine": {"args": {"max_model_len": 100}},
}


def with_(doc: Record, field: str, value: object) -> Record:
    """`doc` with `field` left out (`ABSENT`) or holding `value`."""
    rest = {k: v for k, v in doc.items() if k != field}
    return rest if value is ABSENT else {**rest, field: value}


def scale(workload: Record | str, **more: Any) -> tuple[Any, ...]:
    """What the planner reads of the workload: (model, requests, prompt tokens, decode rows, steps, the longest context, the
    largest request's rows and context, notes)."""
    s = A.row_config_from_workload(workload, **more)
    return (s.model, s.requests, s.prompt_tokens, s.decode_rows, s.steps, s.max_context, s.largest_request_rows, s.largest_request_ctx,
            s.notes)


def test_a_workload_s_own_facts_are_the_row_s_scale() -> None:
    assert scale(WORKLOAD) == ("llama32-1b", 2, 63, 13, 9, 100, 8, 68, [])
    s = A.row_config_from_workload(WORKLOAD, pairs=2, replay_workers=4, host_limit_bytes=1 << 30)
    assert (s.pairs, s.replay_workers, s.host_limit_bytes) == (2, 4, 1 << 30)


def test_a_workload_file_is_read_and_names_its_row_where_its_sweep_does_not(tmp_path: Path) -> None:
    (tmp_path / f"{ROW}.json").write_text(json.dumps(with_(WORKLOAD, "sweep", ABSENT)))
    assert scale(str(tmp_path / f"{ROW}.json")) == ("llama32-1b", 2, 63, 13, 9, 100, 8, 68, [])


@pytest.mark.parametrize("sweep", [ABSENT, None, {}])
def test_a_workload_outside_a_sweep_takes_its_shape_from_its_requests_and_the_defaults(sweep: object) -> None:
    """No sweep: as many requests as the workload lists, and an input of 1,024 and an output of 128 tokens where a figure needs one."""
    assert scale(with_(WORKLOAD, "sweep", sweep), row_id=ROW) == ("llama32-1b", 2, 63, 13, 9, 100, 8, 68, [])
    bare = with_(with_(with_(WORKLOAD, "sweep", sweep), "expected_engine_steps", ABSENT), "engine", ABSENT)
    assert scale(bare, row_id=ROW) == ("llama32-1b", 2, 63, 13, 128, 1152, 8, 68, [])
    model, *rest = scale(with_(WORKLOAD, "sweep", sweep))
    assert (model, rest[:-1]) == ("", [2, 63, 13, 9, 100, 8, 68]), "a parsed workload with no row id names no model"


@pytest.mark.parametrize("requests", [ABSENT, None, []])
def test_a_workload_that_lists_no_request_takes_its_scale_from_its_sweep(requests: object) -> None:
    """No requests: the sweep's batch of its longest input and output, which is also its largest request."""
    assert scale(with_(WORKLOAD, "requests", requests)) == ("llama32-1b", 2, 128, 16, 9, 100, 8, 72, [])
    assert scale(with_(with_(WORKLOAD, "requests", requests), "sweep", {"row_id": ROW})) == ("llama32-1b", 1, 1024, 128, 9, 100, 128, 1152, [])


@pytest.mark.parametrize("ids", [ABSENT, None, []])
def test_a_request_that_gives_no_prompt_counts_no_prompt_token(ids: object) -> None:
    requests = [WORKLOAD["requests"][0], with_(WORKLOAD["requests"][1], "prompt_token_ids", ids)]
    assert scale(with_(WORKLOAD, "requests", requests)) == ("llama32-1b", 2, 60, 13, 9, 100, 8, 68, [])
    alone = [with_({"max_tokens": 5}, "prompt_token_ids", ids)]
    assert scale(with_(WORKLOAD, "requests", alone)) == ("llama32-1b", 2, 128, 5, 9, 100, 5, 5, []), "no prompt at all: the sweep's"


@pytest.mark.parametrize("engine", [ABSENT, None, {}, {"args": None}, {"args": {}}, {"args": {"max_model_len": None}}])
def test_a_workload_that_sets_no_model_length_reaches_its_longest_input_and_output(engine: object) -> None:
    assert scale(with_(WORKLOAD, "engine", engine)) == ("llama32-1b", 2, 63, 13, 9, 72, 8, 68, [])
