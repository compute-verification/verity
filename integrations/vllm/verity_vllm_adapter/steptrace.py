"""Scheduler step trace (contract §3, Production column: "the scheduler's step trace if exposed").

Wraps the in-process scheduler's ``schedule()`` and records, per engine step, what was
scheduled: new requests (with the prefix-cache hit expressed as the number of already
computed tokens at admission), cached/running requests, per-request scheduled token counts,
resumed-after-preemption and preempted request ids. Pure observation of the returned
``SchedulerOutput``; the object is not modified.
"""

from __future__ import annotations

from typing import Any

from verity_vllm_adapter.driver import external_id


class StepTrace:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._orig = None
        self._scheduler = None

    def install(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._orig = scheduler.schedule
        trace = self

        def schedule(*args: Any, **kwargs: Any) -> Any:
            out = trace._orig(*args, **kwargs)
            try:
                trace._record(out, scheduler)
            except Exception as e:  # noqa: BLE001 - never break the engine for a trace
                trace.steps.append({"error": repr(e)})
            return out

        scheduler.schedule = schedule

    def uninstall(self) -> None:
        if self._scheduler is not None and self._orig is not None:
            self._scheduler.schedule = self._orig
        self._scheduler = None
        self._orig = None

    def _record(self, out: Any, scheduler: Any) -> None:
        new_reqs = []
        for r in out.scheduled_new_reqs:
            new_reqs.append(
                {
                    "req_id": external_id(r.req_id),
                    "num_computed_tokens_at_admission": int(r.num_computed_tokens),
                    "prompt_len": len(r.prompt_token_ids) if r.prompt_token_ids is not None else None,
                    "num_blocks": [len(b) for b in r.block_ids],
                }
            )
        cached = out.scheduled_cached_reqs
        cached_reqs = []
        for i, rid in enumerate(cached.req_ids):
            cached_reqs.append(
                {
                    "req_id": external_id(rid),
                    "num_computed_tokens": int(cached.num_computed_tokens[i]),
                    "resumed": rid in cached.resumed_req_ids,
                }
            )
        step = {
            "step": len(self.steps),
            "num_reqs": len(out.num_scheduled_tokens),
            "total_num_scheduled_tokens": int(out.total_num_scheduled_tokens),
            "num_scheduled_tokens": {external_id(k): int(v) for k, v in out.num_scheduled_tokens.items()},
            "new_reqs": new_reqs,
            "cached_reqs": cached_reqs,
            "preempted_req_ids": sorted(external_id(r) for r in (out.preempted_req_ids or [])),
            "finished_req_ids": sorted(external_id(r) for r in (out.finished_req_ids or [])),
            "num_common_prefix_blocks": list(out.num_common_prefix_blocks or []),
        }
        kvm = getattr(scheduler, "kv_cache_manager", None)
        if kvm is not None and hasattr(kvm, "make_prefix_cache_stats"):
            try:
                st = kvm.make_prefix_cache_stats()
                if st is not None:
                    step["prefix_cache_stats"] = {
                        k: getattr(st, k) for k in ("requests", "queries", "hits", "preempted_requests", "preempted_queries", "preempted_hits") if hasattr(st, k)
                    }
            except Exception:  # noqa: BLE001
                pass
        self.steps.append(step)

    def summary(self) -> dict[str, Any]:
        steps = [s for s in self.steps if "error" not in s]
        per_req_steps: dict[str, int] = {}
        per_req_prefill_chunks: dict[str, int] = {}
        for s in steps:
            for rid, n in s["num_scheduled_tokens"].items():
                per_req_steps[rid] = per_req_steps.get(rid, 0) + 1
                if n > 1:
                    per_req_prefill_chunks[rid] = per_req_prefill_chunks.get(rid, 0) + 1
        return {
            "engine_steps": len(steps),
            "peak_concurrency_achieved": max((s["num_reqs"] for s in steps), default=0),
            "preemptions": sum(len(s["preempted_req_ids"]) for s in steps),
            "resumed": sum(1 for s in steps for c in s["cached_reqs"] if c["resumed"]),
            "prefix_cache_hits_at_admission": sum(
                1 for s in steps for r in s["new_reqs"] if r["num_computed_tokens_at_admission"] > 0
            ),
            "prefill_chunks_per_request": per_req_prefill_chunks,
            "steps_per_request": per_req_steps,
            "max_tokens_in_a_step": max((s["total_num_scheduled_tokens"] for s in steps), default=0),
        }
