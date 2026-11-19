"""`verity_vllm.pipeline.match_capture`: the Match stage's capture and control arms, `main` whole, on the CPU.

Stood in, each because it needs a GPU, vLLM or a checkpoint: the workload's reading, the engine's build, its run of the
requests, its version facts and its scheduler's lag, the observer and its header, the acquisition's stage and its hooks, and
the protocols.  Each stand-in keeps what it was handed.  The engine's model holds real buffers, so the analytic tables' words
are the module's own.  The module's clock is one the stand-in run moves (`Clock`), and its timeline keeps its events, so every
number in a line is one the test states.

Held: the files `main` writes (`versions.json`, `spec_tokens.json`, `tokens.json`, `stats.json`,
`acquisition_observed.json`, the tables' words), what each stand-in was handed, every line it prints and how `main` ends: by
returning or by exit 3.  CPU only."""
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

import verity_vllm.pipeline.match_capture as M1

Record = dict[str, Any]
Lines = Callable[[], list[str]]

ROTARY = "model.layers.0.self_attn.rotary_emb.cos_sin_cache"
REQUESTS: list[Record] = [{"request_id": "r0", "prompt_token_ids": [11, 12, 13], "max_tokens": 4, "arrive_step": 0},
                          {"request_id": "r1", "prompt_token_ids": [21, 22], "max_tokens": 4}]
SAMPLING: Record = {"temperature": 0.0}
ACCEPTED = {"r0": [7, 8, 9, 10], "r1": [5, 2]}
SERVED = {"r0": [7, 8, 9, 10], "r1": [5, 2]}
STEPS: Record = {"r0": {"arrive_step_realized": 0, "first_step": 0, "finish_step": 3, "finish_reason": "length"},
                 "r1": {"arrive_step_realized": 1, "first_step": 1, "finish_step": 2, "finish_reason": "stop"}}
KWARGS: Record = {"model": "/weights", "tokenizer": "/weights", "enforce_eager": True, "max_num_seqs": 2}
SHOWN: Record = {"enforce_eager": True, "max_num_seqs": 2}
STATS: Record = {"events_written": 40, "steps": 5, "recorder": "pinned", "storage": "arena", "snapshot_bytes": 0, "errors": []}


def workload(**more: Any) -> Record:
    return {"requests": copy.deepcopy(REQUESTS), "sampling": dict(SAMPLING), "case": None, "accepted": copy.deepcopy(ACCEPTED),
            "row_id": "the-row", "serving_profile": "mixed", "context_class": "i256", "expected_engine_steps": 5} | more


def timing(wall_s: float, *, steps: Record | None = None, schedule: Record | None = None) -> Record:
    """A run's timing as `vllm_adapter.run_requests` returns it."""
    return {"wall_s": wall_s, "engine_steps": 5, "requests": copy.deepcopy(STEPS if steps is None else steps),
            "static_schedule": schedule}


def declared(digest: str, **more: Any) -> Record:
    """A run's static-schedule record, one the run kept."""
    return {"ok": True, "problems": [], "schedule": {"digest": digest, "engine_steps": 5}} | more


class Clock:
    """`match_capture`'s `time`: seconds the stand-ins move."""

    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now


class Model:
    def __init__(self, buffers: dict[str, torch.Tensor]) -> None:
        self.buffers = buffers

    def named_buffers(self) -> list[tuple[str, torch.Tensor]]:
        return list(self.buffers.items())










class Sink:
    """The occurrences the plan's hooks counted."""

    def to_json(self) -> Record:
        return {"sites": [{"site": "s0", "occurrences": 5}, {"site": "s1", "occurrences": 2}]}




@pytest.fixture
def said(capsys: pytest.CaptureFixture[str]) -> Lines:
    """The lines `match_capture` has printed since the last call."""
    return lambda: capsys.readouterr().out.splitlines()




def refused(run: Callable[[], Any]) -> None:
    """``run`` ends the process with exit status 3."""
    with pytest.raises(SystemExit) as e:
        run()
    assert e.value.code == 3




def row(rid: str, served: list[int], accepted: list[int] | None, equal: bool | None, *, finish: Record, executed: int | None,
        exhausted: bool, stopped: bool, lag: int | None) -> Record:
    """A request's row of `tokens.json`."""
    r = next(q for q in REQUESTS if q["request_id"] == rid)
    return {"request_id": rid, "prompt_token_ids": r["prompt_token_ids"], "max_tokens": 4, "generated_token_ids": served,
            "accepted_record_token_ids": accepted, "equals_accepted": equal, "prompt_len": len(r["prompt_token_ids"]),
            "n_generated": len(served), "arrive_step": int(r.get("arrive_step") or 0), **finish, "served_output_len": len(served),
            "executed_prefix_len": executed, "cap_exhausted": exhausted, "eos_stop_before_cap": stopped, "lag": lag}


ROW0 = row("r0", [7, 8, 9, 10], [7, 8, 9, 10], True, finish=STEPS["r0"], executed=4, exhausted=True, stopped=False, lag=1)
ROW1 = row("r1", [5, 2], [5, 2], True, finish=STEPS["r1"], executed=3, exhausted=False, stopped=True, lag=1)
BUILT = "[m1] engine built in 1.5s (case B0, org/model); kwargs={'enforce_eager': True, 'max_num_seqs': 2}"
TREE = "[m1] runtime tree 2 modules, 1 buffers -> runtime_tree.json"
WARMED = "[m1] warm-up (uninstrumented, 2 throwaway requests): 0.50s, 5 steps"
INSTALLED = "[m1] observer installed: 3 roots, profile_id=profile-1, reenter impls={'op': 'impl'}"
OBSERVED = "[m1] observed run 2.00s: 40 events, 5 steps, recorder=pinned, storage=arena, snapshot_bytes=0, errors=0"
SAID0 = ["[m1] r0: prompt_len=3 arrive=0 steps=0..3 (length) generated [7, 8, 9, 10]", "[m1]   accepted  [7, 8, 9, 10]  equal=True"]
SAID1 = ["[m1] r1: prompt_len=2 arrive=0 steps=1..2 (stop) generated [5, 2]", "[m1]   accepted  [5, 2]  equal=True"]












UNTIMED: Record = {"arrive_step_realized": None, "first_step": None, "finish_step": None, "finish_reason": None}


























def words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)










def test_the_steps_to_snapshot_are_the_option_s_numbers() -> None:
    assert M1.parse_steps(None) == frozenset() and M1.parse_steps("") == frozenset()
    assert M1.parse_steps("0,1, 8,") == frozenset({0, 1, 8})
