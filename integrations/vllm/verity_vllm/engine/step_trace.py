"""The scheduler's step trace: what vLLM V1's in-process scheduler decided at every executed step, as data.

A served run's batches, positions, chunked prefills, preemptions, cache hits and finishes are decided by the scheduler while
serving, so nothing declared before the run predicts them.  `StepTrace` records them from the in-process engine core
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, `engine.env`), through `engine.hooks`, without changing what the engine does:

- ``scheduler.add_request``: a request's arrival, its prompt length and sampling parameters, and the step it arrived before;
- ``scheduler.schedule``: the step's index among the steps that schedule tokens (an empty schedule runs no forward);
- ``scheduler.update_from_output(scheduler_output, model_runner_output)``: the step's rows in the model runner's order
  (``model_runner_output.req_ids``: the order the step's tokens are laid out in, Model Runner V2's ``sort_batch_req_ids``), each
  row's first position (``num_computed_tokens`` before the step) and length (``num_scheduled_tokens``), the requests first scheduled
  or resumed this step and the prefix-cache tokens a new request starts from, the requests preempted, the tokens each request
  emitted, and the requests that finished with their reason;
- ``scheduler.finish_requests``: a finish from outside the scheduler (an abort, a stop string the frontend matched).

Under async scheduling the scheduler schedules step s+1 before step s's output is processed; a step's record is still keyed by
the schedule that made it, so rows that compute a token of a request that finished at the step before are recorded as run.

The trace is one JSON document per line (`SCHEMA`): a ``header`` (the scheduler's configuration), then ``arrive``, ``step`` and
``finish`` records in the order they happened, and on a server whose Commit runs in windows, each window's ``open`` and ``close``
(`StepTrace.window`).  `load` reads it back.  `verity_vllm.pipeline.advice` turns a trace and the run's
Commit record into the compiler's advice.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from verity_vllm.engine.hooks import Hooks

SCHEMA = "verity-vllm/step-trace/v1"
#: the trace's file in a `verity-vllm serve-openai` run, beside the committer's step_trace.jsonl
SERVED = "advice_trace.jsonl"
#: `FinishReason` (vllm/v1/engine/__init__.py) by value
FINISH = {0: "stop", 1: "length", 2: "abort", 3: "error", 4: "repetition"}
#: the sampling parameters a request carries into the engine core, as `SamplingParams` names them
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "min_tokens", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "stop_token_ids", "stop", "ignore_eos", "logit_bias", "n")


def scheduler_of(engine: Any) -> Any:
    """The in-process V1 scheduler of a `vllm.LLM`, a `verity_vllm.LLM` or an `LLMEngine` (or the scheduler itself); refuses by name a
    multiprocess engine core, whose scheduler lives in another process."""
    if hasattr(engine, "update_from_output"):
        return engine
    eng = getattr(engine, "vllm", engine)
    eng = getattr(eng, "llm_engine", eng)
    core = getattr(eng, "engine_core", None)
    inner = getattr(core, "engine_core", None) or core
    sched = getattr(inner, "scheduler", None)
    if sched is None or not hasattr(sched, "update_from_output"):
        raise RuntimeError(f"{type(engine).__name__}: no in-process V1 scheduler (engine core {type(core).__name__}); the step trace needs "
                           f"VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine.env.ENGINE)")
    return sched


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_plain(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)]
    return repr(v)


def scheduler_config(sched: Any) -> dict[str, Any]:
    """What of the scheduler's configuration decides its schedules (``max_model_len`` the scheduler's own where its
    scheduler config no longer holds it, as in vLLM 0.11 on)."""
    sc = getattr(sched, "scheduler_config", None)
    cc = getattr(sched, "cache_config", None)
    pick = lambda o, names: {n: _plain(getattr(o, n)) for n in names if o is not None and hasattr(o, n)}   # noqa: E731
    out = {"scheduler": type(sched).__name__,
           **pick(sc, ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "enable_chunked_prefill", "long_prefill_token_threshold",
                       "max_num_partial_prefills", "policy", "async_scheduling")),
           **pick(cc, ("block_size", "enable_prefix_caching", "num_gpu_blocks"))}
    if "max_model_len" not in out:
        mc = getattr(getattr(sched, "vllm_config", None), "model_config", None)
        n = getattr(sched, "max_model_len", None) or getattr(mc, "max_model_len", None)
        if n is not None:
            out["max_model_len"] = int(n)
    return out


def step_geometry(so: Any) -> tuple[dict[str, int], list[str], list[str], dict[str, int]]:
    """From a `SchedulerOutput`: each scheduled request's first position this step, the requests first scheduled, those resumed
    from preemption, and the prefix-cache tokens each new request starts from (`num_computed_tokens` of a new request > 0)."""
    start: dict[str, int] = {}
    new: list[str] = []
    hits: dict[str, int] = {}
    for r in getattr(so, "scheduled_new_reqs", None) or ():
        start[r.req_id] = int(r.num_computed_tokens)
        new.append(r.req_id)
        if int(r.num_computed_tokens):
            hits[r.req_id] = int(r.num_computed_tokens)
    cached = getattr(so, "scheduled_cached_reqs", None)
    resumed: list[str] = []
    if cached is not None:
        for i, rid in enumerate(cached.req_ids):
            start[rid] = int(cached.num_computed_tokens[i])
        resumed = sorted(getattr(cached, "resumed_req_ids", None) or ())
    return start, new, resumed, hits


class StepTrace:
    """The step trace of one in-process engine, installed with `install` and written as it happens to `path` (one line a record,
    flushed at every step) and kept in `records`.  `uninstall` puts the scheduler back."""

    def __init__(self, path: str | Path | None = None, *, request_id: Callable[[str], str] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.request_id = request_id or (lambda rid: rid)
        self.clock = clock                      # when given, each arrive and step record carries its reading `t`
        self.records: list[dict[str, Any]] = []
        self.hooks = Hooks(tag="step-trace")
        self.scheduled = 0                      # steps scheduled with tokens so far: the index the next such step gets
        self.executed = 0
        self._pending: dict[int, int] = {}      # id(SchedulerOutput) -> its step
        self._fh = None
        self.sched: Any = None

    def _emit(self, rec: dict[str, Any]) -> None:
        if self.clock is not None and rec["kind"] in ("arrive", "step"):
            rec["t"] = self.clock()
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            if rec["kind"] in ("step", "header", "window"):
                self._fh.flush()

    def window(self, window: int, event: str) -> None:
        """A Commit window's ``open`` or ``close`` (`pipeline.serve_commit.CommitWindows`): the steps it holds are those scheduled
        from its open's ``before_step`` up to its close's."""
        self._emit({"kind": "window", "window": int(window), "event": event, "before_step": self.scheduled})

    def install(self, engine: Any) -> StepTrace:
        sched = self.sched = scheduler_of(engine)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a")
        self._emit({"kind": "header", "schema": SCHEMA, "config": scheduler_config(sched)})
        me = self

        def add_request(orig):
            def wrapped(request, *a, **kw):
                fresh = request.request_id not in sched.requests
                out = orig(request, *a, **kw)
                if fresh:
                    sp = getattr(request, "sampling_params", None)
                    me._emit({"kind": "arrive", "request": me.request_id(request.request_id), "before_step": me.scheduled,
                              "prompt_len": int(getattr(request, "num_prompt_tokens", 0) or len(request.prompt_token_ids or ())),
                              "eos_token_id": _plain(getattr(request, "eos_token_id", None)),
                              "sampling": {n: _plain(getattr(sp, n)) for n in SAMPLING if sp is not None and hasattr(sp, n)}})
                return out
            return wrapped

        def schedule(orig):
            def wrapped(*a, **kw):
                so = orig(*a, **kw)
                if int(getattr(so, "total_num_scheduled_tokens", 0) or 0) > 0:
                    me._pending[id(so)] = me.scheduled
                    me.scheduled += 1
                return so
            return wrapped

        def update_from_output(orig):
            def wrapped(so, mro, *a, **kw):
                outs = orig(so, mro, *a, **kw)
                step = me._pending.pop(id(so), None)
                if step is not None:
                    me._record_step(step, so, mro, outs)
                return outs
            return wrapped

        def finish_requests(orig):
            def wrapped(request_ids, finished_status, *a, **kw):
                done = orig(request_ids, finished_status, *a, **kw)
                for r in done or ():
                    reason = getattr(r, "get_finished_reason", lambda: None)()
                    me._emit({"kind": "finish", "request": me.request_id(r.request_id), "after_step": me.scheduled - 1,
                              "reason": FINISH.get(int(reason), str(reason)) if reason is not None else str(finished_status),
                              "stop_reason": _plain(getattr(r, "stop_reason", None)), "outside": True})
                return done
            return wrapped

        self.hooks.wrap(sched, "add_request", add_request)
        self.hooks.wrap(sched, "schedule", schedule)
        self.hooks.wrap(sched, "update_from_output", update_from_output)
        self.hooks.wrap(sched, "finish_requests", finish_requests)
        return self

    def _record_step(self, step: int, so: Any, mro: Any, outs: Any) -> None:
        nst = dict(getattr(so, "num_scheduled_tokens", None) or {})
        start, new, resumed, hits = step_geometry(so)
        runner = [r for r in (getattr(mro, "req_ids", None) or ()) if nst.get(r)]
        order = "runner" if sorted(runner) == sorted(nst) else "scheduler"
        ids = runner if order == "runner" else list(nst)
        emitted: dict[str, int] = {}
        finished: dict[str, dict[str, Any]] = {}
        for eco in (outs or {}).values():
            for o in getattr(eco, "outputs", None) or ():
                rid = self.request_id(o.request_id)
                if o.new_token_ids:
                    emitted[rid] = len(o.new_token_ids)
                if o.finish_reason is not None:
                    finished[rid] = {"reason": FINISH.get(int(o.finish_reason), str(o.finish_reason)), "stop_reason": _plain(o.stop_reason)}
        rid = self.request_id
        self._emit({"kind": "step", "step": step, "T": int(so.total_num_scheduled_tokens), "order": order,
                    "rows": [[rid(r), start.get(r), int(nst[r])] for r in ids],
                    "new": [rid(r) for r in new], "resumed": [rid(r) for r in resumed], "cache_hit": {rid(r): k for r, k in hits.items()},
                    "preempted": sorted(rid(r) for r in (getattr(so, "preempted_req_ids", None) or ())),
                    "emitted": emitted, "finished": finished})
        self.executed += 1

    def uninstall(self) -> None:
        self.hooks.uninstall()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def load(path: str | Path) -> list[dict[str, Any]]:
    """A step trace's records, refused unless it opens with a `SCHEMA` header."""
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not recs or recs[0].get("kind") != "header" or recs[0].get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} step trace (its first record is {recs[0] if recs else None!r})")
    return recs


def steps(recs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The trace's step records in step order, refused unless the steps are 0..S-1 each once."""
    out = sorted((r for r in recs if r["kind"] == "step"), key=lambda r: r["step"])
    if [r["step"] for r in out] != list(range(len(out))):
        raise ValueError(f"the trace's steps are {[r['step'] for r in out][:12]}…, not 0..{len(out) - 1} each once")
    return out
