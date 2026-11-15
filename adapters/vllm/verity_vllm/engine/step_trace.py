"""The scheduler's step trace: what vLLM V1's in-process scheduler decided at every executed step, as data.

A served run's batches, positions, chunked prefills, preemptions, cache hits and finishes are decided by the scheduler while
serving, so nothing declared before the run predicts them.  `StepTrace` records them from the in-process engine core
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, `engine.env`), through `engine.hooks`, without changing what the engine does:

- ``scheduler.add_request``: a request's arrival, its prompt length and sampling parameters, and the step it arrived before.
  Where the scheduler's last schedule scheduled nothing (every request it held was held back, as a request at its token
  limit is until its last step's output is back, or it held none), the arrival also says after how many such empty
  schedules it came (``after_empty``). Under async scheduling the engine processes the step in flight after an empty
  schedule, so such a request is first scheduled once that step has finished its requests and cached their last blocks;
  put back before the empty schedule, it is scheduled while that step is in flight, and one whose prompt goes on from a
  request that step finished hits a block less;
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

import atexit
import json
import queue
import threading
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import IO, Any

from verity_vllm.engine import handle
from verity_vllm.engine.hooks import Hooks

__all__ = ["SCHEMA", "SERVED", "StepTrace", "load", "scheduler_config", "step_geometry", "steps"]

#: `records.schema_ids.STEP_TRACE_SCHEMA`'s string, written here: suites outside the adapter list the files this module
#: imports one by one (`benchmarks/network_traces` and `benchmarks/serve_tax`, their `[tool.verity.tests] inputs`)
SCHEMA = "verity-vllm/step-trace/v1"
#: the trace's file in a `verity-vllm serve-openai` run, beside the committer's step_trace.jsonl
SERVED = "advice_trace.jsonl"
#: `FinishReason` (vllm/v1/engine/__init__.py) by value
FINISH = {0: "stop", 1: "length", 2: "abort", 3: "error", 4: "repetition"}
#: the sampling parameters a request carries into the engine core, as `SamplingParams` names them
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "min_tokens", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "stop_token_ids", "stop", "ignore_eos", "logit_bias", "n")
#: the speculative decoding an engine is built with, as vLLM's ``SpeculativeConfig`` names it: the settings that decide which
#: draft tokens a step verifies, which it accepts and which prefix-cache blocks a request hits. The draft model's path is the
#: engine's own, so it is left out. A name the engine's vLLM does not have is skipped: the lookahead block hashes, the
#: draft head's quantization and FLy's acceptance are later vLLM's.
SPECULATIVE = ("method", "num_speculative_tokens", "num_speculative_tokens_per_batch_size", "draft_sample_method",
               "rejection_sample_method", "synthetic_acceptance_rates", "index_share_for_mtp_iteration", "parallel_drafting",
               "disable_padded_drafter_batch", "disable_eagle_block_drop", "use_local_argmax_reduction",
               "enable_adaptive_verification", "disable_lookahead_block_hashes", "draft_lm_head_quantization",
               "fly_window_size", "fly_entropy_threshold", "fly_entropy_top_k")


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
    scheduler config no longer holds it, as in vLLM 0.11 on), and what an engine at vLLM's defaults does not have: the
    KV cache's format where it is not ``auto`` (``kv_cache_dtype``, as the engine's arguments name it), and under
    speculative decoding its ``speculative`` settings (`SPECULATIVE`, those that are set): an engine built with them
    as its ``kv_cache_dtype`` and ``speculative_config`` stores, drafts and accepts as this one does. Where the launch
    left the context limit to vLLM (``max_model_len`` -1, the command line's ``auto``), ``max_model_len_auto`` says so:
    vLLM then builds the model at the checkpoint's own limit and fits ``max_model_len`` to the KV pool afterwards, so
    an engine launched at the fitted number is built at another length than this one."""
    sc = getattr(sched, "scheduler_config", None)
    cc: Any = getattr(sched, "cache_config", None)
    pick = lambda o, names: {n: _plain(getattr(o, n)) for n in names if o is not None and hasattr(o, n)}   # noqa: E731
    out = {"scheduler": type(sched).__name__,
           **pick(sc, ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "enable_chunked_prefill", "long_prefill_token_threshold",
                       "max_num_partial_prefills", "policy", "async_scheduling")),
           **pick(cc, ("block_size", "enable_prefix_caching", "num_gpu_blocks"))}
    mc = getattr(getattr(sched, "vllm_config", None), "model_config", None)
    if "max_model_len" not in out:
        n = getattr(sched, "max_model_len", None) or getattr(mc, "max_model_len", None)
        if n is not None:
            out["max_model_len"] = int(n)
    if getattr(mc, "original_max_model_len", None) == -1:
        out["max_model_len_auto"] = True
    if getattr(cc, "cache_dtype", "auto") != "auto":
        out["kv_cache_dtype"] = _plain(cc.cache_dtype)
    spec = getattr(getattr(sched, "vllm_config", None), "speculative_config", None)
    if spec is not None:
        out["speculative"] = {n: v for n, v in pick(spec, SPECULATIVE).items() if v is not None}
    return out


def step_geometry(so: Any, preempted: Collection[str] = ()) -> tuple[dict[str, int], list[str], list[str], dict[str, int]]:
    """From a `SchedulerOutput`: each scheduled request's first position this step, the requests first scheduled, those resumed
    from preemption, and the prefix-cache tokens each new request starts from (`num_computed_tokens` of a new request > 0).
    A vLLM whose scheduler sends a resumed request again in full names it among the new requests and nowhere as resumed
    (0.34): it is resumed where ``preempted``, the requests preempted and not scheduled since, holds it."""
    start: dict[str, int] = {}
    new: list[str] = []
    resumed: list[str] = []
    hits: dict[str, int] = {}
    for r in getattr(so, "scheduled_new_reqs", None) or ():
        start[r.req_id] = int(r.num_computed_tokens)
        if r.req_id in preempted:
            resumed.append(r.req_id)
            continue
        new.append(r.req_id)
        if int(r.num_computed_tokens):
            hits[r.req_id] = int(r.num_computed_tokens)
    cached = getattr(so, "scheduled_cached_reqs", None)
    if cached is not None:
        for i, rid in enumerate(cached.req_ids):
            start[rid] = int(cached.num_computed_tokens[i])
        resumed += getattr(cached, "resumed_req_ids", None) or ()
    return start, new, sorted(resumed), hits


class StepTrace:
    """The step trace of one in-process engine, installed with `install` and written as it happens to `path` (one line a record,
    flushed at every step) and kept in `records`.  `uninstall` puts the scheduler back.  ``writer``: the file is written by a
    thread of its own, in the same order and bytes, so the engine thread never waits on its disk (a sealer's writes can hold an
    append there for hundreds of ms); `drain` waits until a reader of the file sees every record so far, and a failed write is
    raised on the engine thread at its next record."""

    def __init__(self, path: str | Path | None = None, *, request_id: Callable[[str], str] | None = None,
                 clock: Callable[[], float] | None = None, writer: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.request_id = request_id or (lambda rid: rid)
        self.clock = clock                      # when given, each arrive and step record carries its reading `t`
        self.records: list[dict[str, Any]] = []
        self.hooks = Hooks(tag="step-trace")
        self.scheduled = 0                      # steps scheduled with tokens so far: the index the next such step gets
        self.empty = 0                          # schedules that scheduled nothing since the last such step
        self.executed = 0
        self._pending: dict[int, int] = {}      # id(SchedulerOutput) -> its step
        self._preempted: set[str] = set()       # the requests preempted and not scheduled since
        self._fh: IO[str] | None = None
        self.sched: Any = None
        self.writer = writer
        self._lines: queue.SimpleQueue | None = None        # the writer thread's: (line, flush), a drain's Event, None to stop
        self._thread: threading.Thread | None = None
        self._failed: OSError | ValueError | None = None

    def _emit(self, rec: dict[str, Any]) -> None:
        if self.clock is not None and rec["kind"] in ("arrive", "step"):
            rec["t"] = self.clock()
        self.records.append(rec)
        if self._fh is None:
            return
        line, flush = json.dumps(rec, separators=(",", ":")) + "\n", rec["kind"] in ("step", "header", "window")
        if (lines := self._lines) is None:
            self._fh.write(line)
            if flush:
                self._fh.flush()
            return
        if self._failed is not None:
            raise self._failed
        lines.put((line, flush))

    def _write(self) -> None:
        """The writer thread: each line as `_emit` would write it, and a drain's event set once the lines before it are flushed.
        After a failed write it writes nothing more, but still sets every drain's event."""
        lines, fh = self._lines, self._fh
        assert lines is not None and fh is not None     # `install` sets both before it starts this thread, and
        while (item := lines.get()) is not None:        # `uninstall` joins it before it clears them
            line, flush = item if isinstance(item, tuple) else ("", True)
            if self._failed is None:
                try:
                    fh.write(line)
                    if flush:
                        fh.flush()
                except (OSError, ValueError) as e:
                    self._failed = e
            if isinstance(item, threading.Event):
                item.set()

    def drain(self) -> None:
        """Every record so far in the file and flushed, for a reader of it (the writer's lines written first)."""
        if self._lines is None:
            if self._fh is not None:
                self._fh.flush()
            return
        done = threading.Event()
        self._lines.put(done)
        done.wait()

    def window(self, window: int, event: str) -> None:
        """A Commit window's ``open`` or ``close`` (`pipeline.serve_commit.CommitWindows`): the steps it holds are those scheduled
        from its open's ``before_step`` up to its close's."""
        self._emit({"kind": "window", "window": int(window), "event": event, "before_step": self.scheduled})

    def install(self, engine: Any) -> StepTrace:
        """Wraps the in-process scheduler of ``engine`` (`handle.scheduler`), or ``engine`` itself where it is the scheduler."""
        sched = self.sched = engine if hasattr(engine, "update_from_output") else handle.scheduler(engine)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a")
            if self.writer:
                self._lines = queue.SimpleQueue()
                self._thread = threading.Thread(target=self._write, name="step-trace-writer", daemon=True)
                self._thread.start()
                atexit.register(self.drain)             # a server that exits without `uninstall`: its file still whole
        self._emit({"kind": "header", "schema": SCHEMA, "config": scheduler_config(sched)})
        me = self

        def add_request(orig):
            def wrapped(request, *a, **kw):
                fresh = request.request_id not in sched.requests
                after = {"after_empty": me.empty} if me.empty else {}
                out = orig(request, *a, **kw)
                if fresh:
                    sp = getattr(request, "sampling_params", None)
                    me._emit({"kind": "arrive", "request": me.request_id(request.request_id), "before_step": me.scheduled, **after,
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
                    me.empty = 0
                else:
                    me.empty += 1
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
                    me._preempted.discard(r.request_id)
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
        start, new, resumed, hits = step_geometry(so, self._preempted)
        preempted = set(getattr(so, "preempted_req_ids", None) or ())
        self._preempted = (self._preempted - set(resumed)) | preempted
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
        ext = self.request_id
        self._emit({"kind": "step", "step": step, "T": int(so.total_num_scheduled_tokens), "order": order,
                    "rows": [[ext(r), start.get(r), int(nst[r])] for r in ids],
                    "new": [ext(r) for r in new], "resumed": [ext(r) for r in resumed], "cache_hit": {ext(r): k for r, k in hits.items()},
                    "preempted": sorted(ext(r) for r in preempted),
                    "emitted": emitted, "finished": finished})
        self.executed += 1

    def uninstall(self) -> None:
        self.hooks.uninstall()
        if self._lines is not None:
            self._lines.put(None)
            thread = self._thread
            assert thread is not None                       # `install` sets it with `_lines`
            thread.join()
            self._lines = self._thread = None
            atexit.unregister(self.drain)
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
