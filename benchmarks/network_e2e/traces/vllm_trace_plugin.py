"""A vLLM general plugin (entry point group ``vllm.general_plugins``) that records the V1 scheduler's every step as JSON lines.

vLLM loads general plugins in every process it starts (the API server, the engine core, the workers), so `register` runs in
the engine core process too, where the scheduler lives, whatever the multiprocessing start method. `corpus.py` makes the
entry point visible by writing a ``.dist-info`` beside this module on the server's PYTHONPATH.

`register` reads two variables:

* ``VERITY_INT8_PARTS`` (``linears``): imports `verity_vllm.engine.int8_ranks`, which registers the int8_exact linear method
  (``--quantization verity_int8``) in this process, as it does in a tensor-parallel engine's ranks.
* ``VERITY_TRACE_DIR``: wraps `Scheduler.add_request`, `Scheduler.schedule`, `Scheduler.update_from_output` and
  `Scheduler._free_request` (the `AsyncScheduler` inherits them) so that the engine core appends to four files there,
  named with its pid:

  - ``arrivals.PID.jsonl``, one line per request the engine core hands the scheduler, in that order (the order its FCFS
    waiting queue takes them in): the time, vLLM's ``arrival_time`` stamp (set in the API server), the scheduler's
    ``current_step`` then and the queue lengths;
  - ``steps.PID.jsonl``, one line per ``schedule()`` call that scheduled a token or reported a preempted or finished request:
    the step's new and cached requests with their ``num_computed_tokens`` (the first position scheduled) and
    ``num_scheduled_tokens``, the prefix-cache hit of a request's first scheduling, the preempted and finished ids, and
    the scheduler's ``current_step`` (``sched_step``, which `vllm_adversarial_plugin`'s records carry too);
  - ``outputs.PID.jsonl``, one line per recorded step once its model output is back: the model runner's batch order and the
    tokens it sampled for each row;
  - ``requests.PID.jsonl``, one line per request when the scheduler frees it: arrival time, prompt and output ids, finish
    reason, preemptions and prefix-cache hit.

A request id is the engine's (``cmpl-{X-Request-Id}-0-{8 hex}``) cut back to the client's X-Request-Id (`client_id`). An
error inside the hook is counted and written to ``hook_errors.PID.txt``; it never reaches the engine. The API server's
side of a request's arrival (its accept, headers and body) is `vllm_accept_stamp`'s.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path

TRACE_DIR = "VERITY_TRACE_DIR"
INT8_PARTS = "VERITY_INT8_PARTS"
_ENGINE_ID = re.compile(r"^(?:cmpl|chatcmpl)-(.+)-\d+-[0-9a-f]{8}$")
MAX_ERRORS = 20


def client_id(engine_id: str) -> str:
    """The client's X-Request-Id inside the engine's request id, else the engine's id unchanged."""
    m = _ENGINE_ID.match(engine_id)
    return m.group(1) if m else engine_id


class Tracer:
    """The record files of one scheduler, opened when it first gets a request or a step."""

    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        self.out = out
        self.files = {k: open(out / f"{k}.{pid}.jsonl", "a", encoding="utf-8")
                      for k in ("arrivals", "steps", "outputs", "requests")}
        self.errors_path = out / f"hook_errors.{pid}.txt"
        self.errors = 0
        self.k = 0
        self.empty = 0
        self.pending: dict[int, int] = {}
        #: each live request's prefix-cache hit: its num_computed_tokens when first scheduled (the Request's prefill_stats is
        #: unset by the time it is freed)
        self.hits: dict[str, int] = {}

    def write(self, kind: str, rec: dict) -> None:
        f = self.files[kind]
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()

    def error(self, where: str) -> None:
        self.errors += 1
        if self.errors <= MAX_ERRORS:
            with open(self.errors_path, "a", encoding="utf-8") as f:
                f.write(f"--- {where} (error {self.errors})\n{traceback.format_exc()}\n")

    def arrived(self, sched, req) -> None:
        self.write("arrivals", {"id": client_id(req.request_id), "t": time.time(), "arrival": req.arrival_time,
                                "sched_step": getattr(sched, "current_step", None), "waiting": len(sched.waiting),
                                "running": len(sched.running)})

    def step(self, sched, so) -> None:
        nst = so.num_scheduled_tokens
        preempted = sorted(so.preempted_req_ids or ())
        finished = sorted(so.finished_req_ids or ())
        if not nst and not preempted and not finished:
            self.empty += 1
            return
        k, self.k = self.k, self.k + 1
        if nst:
            self.pending[id(so)] = k
        new = []
        for r in so.scheduled_new_reqs:
            req = sched.requests.get(r.req_id)
            resumed = bool(req is not None and req.num_preemptions > 0)
            row = {"id": client_id(r.req_id), "num_computed_tokens": int(r.num_computed_tokens),
                   "num_scheduled_tokens": int(nst.get(r.req_id, 0)), "prompt_len": int(r.prompt_len), "resumed": resumed}
            if req is not None:
                row["num_output_tokens"] = int(req.num_output_tokens)
            if not resumed:
                row["cache_hit"] = int(r.num_computed_tokens)
                self.hits.setdefault(r.req_id, row["cache_hit"])
            new.append(row)
        cached = []
        c = so.scheduled_cached_reqs
        resumed_ids = getattr(c, "resumed_req_ids", None) or ()
        for i, rid in enumerate(c.req_ids):
            req = sched.requests.get(rid)
            row = {"id": client_id(rid), "num_computed_tokens": int(c.num_computed_tokens[i]),
                   "num_scheduled_tokens": int(nst.get(rid, 0)), "num_output_tokens": int(c.num_output_tokens[i]),
                   "resumed": rid in resumed_ids}
            if req is not None:
                row["prompt_len"] = int(req.num_prompt_tokens)
            cached.append(row)
        self.write("steps", {"step": k, "sched_step": getattr(sched, "current_step", None), "t": time.time(), "new": new,
                             "cached": cached,
                             "total_num_scheduled_tokens": int(so.total_num_scheduled_tokens),
                             "preempted": [client_id(r) for r in preempted], "finished": [client_id(r) for r in finished],
                             "waiting": len(sched.waiting), "running": len(sched.running), "empty_before": self.empty})
        self.empty = 0

    def update(self, so, mro) -> None:
        k = self.pending.pop(id(so), None)
        if k is None or mro is None:
            return
        sampled = getattr(mro, "sampled_token_ids", None) or []
        self.write("outputs", {"step": k, "t": time.time(), "batch": [client_id(r) for r in mro.req_ids],
                               "sampled": [[int(x) for x in s] for s in sampled]})

    def freed(self, req) -> None:
        reason = req.get_finished_reason()
        self.write("requests", {
            "id": client_id(req.request_id), "engine_id": req.request_id, "arrival": req.arrival_time, "t_free": time.time(),
            "prompt_ids": list(req.prompt_token_ids or ()), "output_ids": [int(x) for x in req.output_token_ids],
            "finish_reason": reason.name.lower() if reason is not None else None,
            "stop_reason": req.stop_reason if isinstance(req.stop_reason, (int, str)) or req.stop_reason is None else str(req.stop_reason),
            "status": req.status.name, "num_preemptions": int(req.num_preemptions), "max_tokens": int(req.max_tokens),
            "cache_hit": self.hits.pop(req.request_id, None)})


def _tracer(sched) -> Tracer:
    t = getattr(sched, "_verity_tracer", None)
    if t is None:
        t = sched._verity_tracer = Tracer(Path(os.environ[TRACE_DIR]))
    return t


def install(cls) -> None:
    """Wrap `cls`'s add_request, schedule, update_from_output and _free_request (once)."""
    if getattr(cls, "_verity_traced", False):
        return
    add, schedule, update, free = cls.add_request, cls.schedule, cls.update_from_output, cls._free_request

    def traced_add(self, request, *args, **kwargs):
        t = _tracer(self)
        try:
            t.arrived(self, request)
        except Exception:  # noqa: BLE001 -- the engine must serve on
            t.error("add_request")
        return add(self, request, *args, **kwargs)

    def traced_schedule(self, *args, **kwargs):
        so = schedule(self, *args, **kwargs)
        t = _tracer(self)
        try:
            t.step(self, so)
        except Exception:  # noqa: BLE001 -- the engine must serve on
            t.error("schedule")
        return so

    def traced_update(self, scheduler_output, model_runner_output, *args, **kwargs):
        out = update(self, scheduler_output, model_runner_output, *args, **kwargs)
        t = _tracer(self)
        try:
            t.update(scheduler_output, model_runner_output)
        except Exception:  # noqa: BLE001
            t.error("update_from_output")
        return out

    def traced_free(self, request, *args, **kwargs):
        t = _tracer(self)
        try:
            t.freed(request)
        except Exception:  # noqa: BLE001
            t.error("_free_request")
        return free(self, request, *args, **kwargs)

    cls.add_request, cls.schedule = traced_add, traced_schedule
    cls.update_from_output, cls._free_request = traced_update, traced_free
    cls._verity_traced = True


def register() -> None:
    if os.environ.get(INT8_PARTS):
        import verity_vllm.engine.int8_ranks  # noqa: F401  (registers what VERITY_INT8_PARTS names, on import)
    if os.environ.get(TRACE_DIR):
        from vllm.v1.core.sched.scheduler import Scheduler
        install(Scheduler)
