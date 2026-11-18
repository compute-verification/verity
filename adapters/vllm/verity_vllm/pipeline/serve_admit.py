"""serve-openai's admissions: what a request brings to the engine (`Admission`, and `Delta`, what goes back), and the order
and the step they are added in (the admission scheme's ``order``). ``arrival`` adds each request to the engine as soon
as the frontend has parsed it, in the API server's body-read order; ``release-order`` adds them in the network
attester's release order, so the advice needn't carry the permutation between the two; ``tick`` also fixes the step
each is added at, by the certified tick, so it needn't carry when each arrived either.

The orders ``release-order`` and ``tick`` are the network attester's (`network.prover.admission`: `ReleaseOrder`, and the
certified tick's `TickGate`); this module is what any admission order reads, and the step's rows it needs.

**A step's rows**. The advice numbers requests by their first row, so the order of the rows inside a step is a
choice the engine must not make. vLLM's scheduler (FCFS) lists a step's requests as its running ones in running order, then
the ones it newly schedules in waiting-queue order, which is the admission order; Model Runner V2 then lays the step out by
its own `sort_batch_req_ids` (vLLM d9105ea8: decode-length rows first, every other row by its row count, shortest first),
which put a later-admitted shorter prompt's first row ahead of an earlier one's. Under ``release-order``,
`keep_scheduler_rows` replaces that sort with `scheduler_order`: requests verifying draft tokens first (none without
speculative decoding), then every row in the scheduler's order. A step is then its running requests in running order (the
decodes, then the prefill a chunk continues, if any: under FCFS it is the latest scheduled), then its new requests' first
rows in admission order. vLLM sorts decodes first for the attention backends that split a batch into decodes and prefills
(``split_decodes_and_prefills``, the ones with a ``reorder_batch_threshold``: FlashInfer, MLA, Mamba); FLASH_ATTN, the
served backend, isn't one, so no kernel needs the decodes first, and `keep_scheduler_rows` refuses an engine with one.
Batch invariance keeps each row's values independent of its place.
"""
from __future__ import annotations

import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.engine.hooks import Hooks

@dataclass
class Delta:
    """One engine output of a request: the new token ids and text since the last, and its finish."""

    token_ids: list[int]
    text: str
    finish_reason: str | None = None
    stop_reason: Any = None
    prompt_token_ids: list[int] | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None or self.error is not None


@dataclass
class Admission:
    http_id: str
    engine_id: str
    prompt_ids: list[int]
    params: Any
    deliver: Callable[[Delta], None]
    brought: dict[str, Any]                  # the sampling fields it sent, as it sent them
    user: str | None = None
    release: tuple[str, int] | None = None   # (the network attester's run, the request's place in its release order)
    times: dict[str, float] = field(default_factory=dict)   # wall-clock stamps (telemetry): head, received, submitted, admitted
    ingress: dict[str, Any] = field(default_factory=dict, init=False)   # its ingress record, once the engine loop seeds and adds it


def engine_id_of(http_id: str) -> str:
    """vLLM's engine request id of a completion's one prompt: ``f"{request_id}-{i}"``."""
    return f"{http_id}-0"


def new_http_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


def scheduler_order(num_tokens_per_req: Mapping[str, int], draft_tokens: Mapping[str, Sequence[int]],
                    decode_query_len: int) -> list[str]:
    """Model Runner V2's ``sort_batch_req_ids`` under ``release-order``: the step's requests in the scheduler's order (its
    ``num_scheduled_tokens``), the ones verifying draft tokens first as vLLM keeps them; ``decode_query_len`` is unused."""
    return sorted(num_tokens_per_req, key=lambda r: not draft_tokens.get(r))


def keep_scheduler_rows(runner: Any, scheduler: Any) -> Hooks:
    """Lay each step of ``runner`` (Model Runner V2) out in ``scheduler``'s order (`scheduler_order`, the module docstring),
    until the returned hooks are uninstalled. Refused for a runner whose module has no ``sort_batch_req_ids`` (V1's), for
    one whose attention backends aren't built yet or one of which needs the decodes first (a ``reorder_batch_threshold``),
    and for a scheduler policy other than FCFS, whose waiting queue is not the admission order."""
    module = sys.modules.get(type(runner).__module__)
    if not callable(getattr(module, "sort_batch_req_ids", None)):
        raise RuntimeError(f"order release-order keeps the scheduler's row order through Model Runner V2's sort_batch_req_ids, "
                           f"which {type(runner).__module__} has none of: serve with VLLM_USE_V2_MODEL_RUNNER=1 (engine.env)")
    groups = getattr(runner, "attn_groups", None)
    if groups is None:
        raise RuntimeError("order release-order reads the runner's attention backends (attn_groups), which aren't built yet: "
                           "keep the scheduler's rows once the engine is up")
    split = sorted({type(b).__name__ for gs in groups for g in gs for b in getattr(g, "metadata_builders", ())
                    if getattr(b, "reorder_batch_threshold", None) is not None})
    if split:
        raise RuntimeError(f"order release-order keeps the scheduler's row order, but attention backend {', '.join(split)} "
                           f"splits each batch into decodes and prefills and needs the decodes first (reorder_batch_threshold)")
    policy = getattr(getattr(scheduler, "scheduler_config", None), "policy", None)
    if policy != "fcfs":
        raise RuntimeError(f"order release-order needs the FCFS scheduler, whose waiting queue is the admission order; "
                           f"this one's policy is {policy!r}")
    hooks = Hooks(tag="admit-rows")
    hooks.patch(module, "sort_batch_req_ids", scheduler_order)
    return hooks
