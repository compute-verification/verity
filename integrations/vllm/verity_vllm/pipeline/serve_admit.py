"""serve-openai's admissions: what a request brings to the engine (`Admission`, and `Delta`, what goes back), and the order
they are added in (``--admit``). ``arrival`` adds each request to the engine as soon as the frontend has parsed it, in the
API server's body-read order; ``release-order`` adds them in the network certifier's release order, so the advice needn't carry the
permutation between the two (3,669 bits on TRACE-CORPUS's heavy trace).

`ReleaseOrder` is a reorder buffer in front of the engine loop's ``submit``: it forwards request n of the release order
once the frontend holds every earlier-released request, each forwarded, refused or dropped, and holds it until then. There
is no hold to a bucket boundary (the boundary saves no step bits).

**What the network certifier sends.** Each request it releases carries ``x-verity-release: N``, N its 0-based index in the release
order of its run (the certifier's ``Proxy.requests_released`` before the release counts it): dense, each N once, every
released request numbered, a request a fault drops before its release unnumbered. With ``x-verity-release-run: ID`` (the
certifier run's id, hex) the numbering is per run: a new ID starts a new order at 0, and what the old run still held is
forwarded at once. The network certifier's shim (`benchmarks/network_traces/active_replay.LiveServer`) copies both onto the POST it
sends the endpoint; it doesn't today, so ``release-order`` holds nothing until it does.

What a request without the header gets: it is forwarded at once (``unordered``), as the served plan's probes are. A
malformed or repeated ordinal is refused (`ValueError`, the server's 400). A released request that never reaches the
frontend, or whose body doesn't arrive, would hold every later one: once a later one has been held ``hold_s`` behind it,
it is skipped and the held ones go on; if it comes after all it is forwarded at once (``late``). Both are counted
(`summary`), and so are the runs.

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

import asyncio
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.engine.hooks import Hooks

ADMIT = ("arrival", "release-order")
RELEASE_HEADER, RUN_HEADER = "x-verity-release", "x-verity-release-run"


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
    ingress: dict[str, Any]
    user: str | None = None
    release: tuple[str, int] | None = None   # (the network certifier's run, the request's place in its release order)
    times: dict[str, float] = field(default_factory=dict)   # wall-clock stamps (telemetry): head, received, submitted, admitted


def engine_id_of(http_id: str) -> str:
    """vLLM's engine request id of a completion's one prompt: ``f"{request_id}-{i}"``."""
    return f"{http_id}-0"


def new_http_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


class ReleaseOrder:
    """Forward each `Admission` to ``forward`` (the engine loop's ``submit``) in release order.
    `key` reads a request's headers as the frontend receives it; `submit` and `abort` stand in for the engine loop's;
    `passed` says a numbered request won't be submitted (refused, or its connection ended). All of it runs on the HTTP
    server's event loop. ``schedule(delay_s, fn)`` arms the skip (default: the running loop's ``call_later``)."""

    def __init__(self, forward: Callable[[Any], None], abort: Callable[[str], None], *, hold_s: float,
                 schedule: Callable[[float, Callable[[], None]], Any] | None = None) -> None:
        self.forward, self.abort_forward, self.hold_s, self._schedule = forward, abort, float(hold_s), schedule
        self.run: str | None = None
        self.runs: list[str] = []
        self.next = 0
        self.arrived: set[int] = set()           # this run's ordinals the frontend has received, at or past `next`
        self.held: dict[int, Any] = {}           # ordinal -> its admission, waiting for an earlier one
        self.gone: set[int] = set()              # ordinals past `next` that won't be submitted
        self.skipped: set[int] = set()
        self.armed: tuple[str | None, int] | None = None
        self.counts = {"ordered": 0, "unordered": 0, "late": 0, "stale": 0}
        self.max_held = 0

    def key(self, headers: Mapping[str, str]) -> tuple[str, int] | None:
        """The request's (run, ordinal) from its lower-cased headers, None without one; `ValueError` on a malformed or
        repeated ordinal. A new run starts here, forwarding what the old one still holds."""
        raw = headers.get(RELEASE_HEADER)
        if raw is None:
            return None
        if not raw.strip().isdigit():
            raise ValueError(f"{RELEASE_HEADER}: a non-negative integer, not {raw!r}")
        run, n = headers.get(RUN_HEADER, ""), int(raw)
        if run != self.run:
            if run in self.runs:
                self.counts["stale"] += 1
                return None
            self._start(run)
        if n in self.arrived or (n < self.next and n not in self.skipped):
            raise ValueError(f"{RELEASE_HEADER}: {n} came twice in run {run!r}")
        if n >= self.next:
            self.arrived.add(n)
        return run, n

    def submit(self, adm: Any) -> None:
        k = getattr(adm, "release", None)
        if k is None or k[0] != self.run:
            self.counts["unordered"] += 1
            return self.forward(adm)
        if k[1] < self.next:
            self.skipped.discard(k[1])
            self.counts["late"] += 1
            return self.forward(adm)
        self.held[k[1]] = adm
        self.max_held = max(self.max_held, len(self.held))
        self._flush()

    def abort(self, engine_id: str) -> None:
        for n, adm in self.held.items():
            if adm.engine_id == engine_id:
                del self.held[n]
                self.gone.add(n)
                return self._flush()
        self.abort_forward(engine_id)

    def passed(self, key: tuple[str, int] | None) -> None:
        if key is None or key[0] != self.run or key[1] < self.next or key[1] in self.held:
            return
        self.gone.add(key[1])
        self._flush()

    def _start(self, run: str) -> None:
        for n in sorted(self.held):
            self.forward(self.held[n])
            self.counts["ordered"] += 1
        self.run, self.next, self.armed = run, 0, None
        self.runs.append(run)
        self.arrived, self.held, self.gone, self.skipped = set(), {}, set(), set()

    def _flush(self) -> None:
        while True:
            n = self.next
            if n in self.held:
                self.forward(self.held.pop(n))
                self.counts["ordered"] += 1
            elif n in self.gone:
                self.gone.discard(n)
            else:
                break
            self.arrived.discard(n)
            self.next += 1
        if self.held and self.armed != (self.run, self.next):
            self.armed = (self.run, self.next)
            schedule = self._schedule or asyncio.get_running_loop().call_later
            schedule(self.hold_s, lambda at=self.armed: self._expire(at))

    def _expire(self, at: tuple[str | None, int]) -> None:
        """``hold_s`` after a later request began waiting on ordinal ``at``: skip it if it still hasn't come."""
        if (self.run, self.next) != at or not self.held:
            return
        self.skipped.add(self.next)
        self.gone.add(self.next)
        self._flush()

    def summary(self) -> dict[str, Any]:
        return {**self.counts, "skipped": len(self.skipped) + self.counts["late"], "skipped_unseen": sorted(self.skipped)[:20],
                "held_now": len(self.held), "max_held": self.max_held, "runs": len(self.runs), "hold_s": self.hold_s}


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
        raise RuntimeError(f"--admit release-order keeps the scheduler's row order through Model Runner V2's sort_batch_req_ids, "
                           f"which {type(runner).__module__} has none of: serve with VLLM_USE_V2_MODEL_RUNNER=1 (engine.env)")
    groups = getattr(runner, "attn_groups", None)
    if groups is None:
        raise RuntimeError("--admit release-order reads the runner's attention backends (attn_groups), which aren't built yet: "
                           "keep the scheduler's rows once the engine is up")
    split = sorted({type(b).__name__ for gs in groups for g in gs for b in getattr(g, "metadata_builders", ())
                    if getattr(b, "reorder_batch_threshold", None) is not None})
    if split:
        raise RuntimeError(f"--admit release-order keeps the scheduler's row order, but attention backend {', '.join(split)} "
                           f"splits each batch into decodes and prefills and needs the decodes first (reorder_batch_threshold)")
    policy = getattr(getattr(scheduler, "scheduler_config", None), "policy", None)
    if policy != "fcfs":
        raise RuntimeError(f"--admit release-order needs the FCFS scheduler, whose waiting queue is the admission order; "
                           f"this one's policy is {policy!r}")
    hooks = Hooks(tag="admit-rows")
    hooks.patch(module, "sort_batch_req_ids", scheduler_order)
    return hooks
