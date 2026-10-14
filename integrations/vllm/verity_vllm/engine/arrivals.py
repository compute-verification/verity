"""verity_vllm/engine/arrivals.py — step-indexed request arrivals (coverage campaign D87, serving profile B8-mixed).  Torch-free.

`run_requests_arrivals(llm, requests, params_for, on_step=...)` is `verity_vllm.engine.engine_driver.run_requests` with one addition:
request `r` is added to the engine right before engine step `r.arrive_step` (0 = before the first step, the historical
behaviour).  Arrivals are keyed to the engine-step counter, never to wall time, so the observer run, the control run and the
non-interference variants schedule the same batch at every step -- dynamic batch membership becomes a declared, reproducible
fact of the workload.  If every earlier request has finished before the next arrival's step, the next arrival is admitted at
once (the engine would otherwise idle); the realized arrival step is what `timing["requests"]` records.

`timing["requests"][rid]` = {arrive_step (planned), arrive_step_realized, first_step (the engine step that ran the request's
prefill), finish_step, finish_reason, n_generated}.  With every request at step 0 the loop is the driver's loop.

Step indexing.  An "engine step" is a model forward pass -- the unit the observer, the census and the gates count.  vLLM V1's
`engine.step()` returns the *previous* forward's sampled tokens (one-iteration output lag; the non-interference hooks variant
documents the same), so `step()` iterations are not forwards: the first iteration returns nothing and one drain iteration after the
last forward returns the finish.  `engine_steps` is the number of forwards; `first_step` / `finish_step` / `arrive_step_realized` are
forward indices; the raw iteration numbers are kept as `*_iter` and `step_iterations` for the record.

Which iterations ran a forward (2026-09-17, lead's request after the chunked owner's finding).  Until this window the loop MEASURED
the lag as (first non-empty output iteration - first admission iteration) and inferred "forward at iteration j <=> iteration j + lag
returned outputs".  Both break under chunked prefill: a partial-prefill step is a forward that returns NOTHING (the engine discards the
sample of a chunk that leaves the prompt incomplete), so with a 3-chunk first prompt the measured lag was 3 instead of 1 and two
forwards went uncounted (15 for 17).  The loop now derives forwards in this order and records the source in `timing`:

  1. `output_lag_source = "scheduler"` -- the in-process V1 scheduler is reachable (`llm_engine.engine_core[.engine_core].scheduler`
     with a `requests` dict, or `scheduler=` given): after every `step()` the per-request `num_computed_tokens` is read (a pure read,
     as the chunked owner's `chunks.Collector`); an iteration whose schedule advanced some request ran a forward, and the
     lag is (finish iteration - last advancing iteration) of each request, which must agree across requests.  Exact, including
     chunks, waiting requests and idle iterations (an idle iteration is neither a forward nor a drain: `iterations - forwards` is
     NOT the lag when arrivals leave gaps).
  2. `"clamped"` / `"measured"` -- no scheduler: the measured lag is clamped to the engine's structural lag (vLLM V1 = 1, V0 = 0;
     an engine of unknown kind keeps the measured value), and a forward is attributed to every iteration after which the engine
     still had unfinished requests (lag >= 1: the only iterations without a forward are those that merely drained a finish), or to
     every iteration (lag 0: the loop never calls `step()` on an idle engine).  A partial-prefill chunk therefore counts as the
     forward it is, while an idle drain iteration does not.
`timing["output_lag_measured"]` keeps the old measurement.  Census / non-interference forward counts are untouched (they count kernels
and hooks directly); existing records are not reinterpreted.
"""
from __future__ import annotations

import time
from typing import Any, Callable


def scheduler_of(llm: Any) -> Any | None:
    """The in-process vLLM V1 scheduler of an offline `LLM` (`llm_engine.engine_core[.engine_core].scheduler`) when it exposes a
    `requests` dict, else None (multiprocess engine core, V0, a stub).  Pure attribute reads; nothing is patched."""
    try:
        eng = llm.llm_engine
        core = getattr(eng, "engine_core", None)
        inner = getattr(core, "engine_core", None) or core
        sched = getattr(inner, "scheduler", None)
        return sched if isinstance(getattr(sched, "requests", None), dict) else None
    except Exception:  # noqa: BLE001
        return None


def structural_output_lag(eng: Any) -> int | None:
    """The output lag the engine's architecture implies: 1 for vLLM V1 (`engine_core`), 0 for V0 (`model_executor`, no core), None
    when the engine is of no known kind (the measured lag is then kept)."""
    if getattr(eng, "engine_core", None) is not None:
        return 1
    if getattr(eng, "model_executor", None) is not None:
        return 0
    return None


def _snapshot_computed(sched: Any) -> dict[str, int] | None:
    try:
        out: dict[str, int] = {}
        for rid, req in list(sched.requests.items()):
            nc = getattr(req, "num_computed_tokens", None)
            if nc is not None:
                out[str(rid)] = int(nc)
        return out
    except Exception:  # noqa: BLE001
        return None


def forwards_from_schedule(computed: list[dict[str, int]]) -> tuple[list[bool], dict[str, int]]:
    """Per iteration: did this iteration's schedule advance some request's `num_computed_tokens` (=> a forward ran)?  Also the last
    iteration that advanced each request (its final forward): the finish output returned `lag` iterations later measures the lag
    per request, independent of idle iterations (which are neither forwards nor drains, so `iterations - forwards` is not the lag)."""
    ran: list[bool] = []
    last_adv: dict[str, int] = {}
    prev: dict[str, int] = {}
    for j, snap in enumerate(computed):
        adv = False
        for rid, nc in snap.items():
            if nc > prev.get(rid, 0):
                adv = True
                last_adv[rid] = j
        prev.update(snap)
        ran.append(adv)
    return ran, last_adv


def run_requests_arrivals(llm: Any, requests: list[Any], params_for: Callable[[Any], Any], *,
                          on_step: Callable[[list[Any]], None] | None = None, scheduler: Any | None = None,
                          clock: Callable[[], float] = time.perf_counter,
                          request_timing: bool = False) -> tuple[dict[str, Any], dict]:
    """Drive a step-indexed schedule and report throughput and per-request latency.

    ``clock`` is injectable so callers and tests can measure the same admission,
    first-token and completion boundaries without depending on wall-clock time.
    ``request_timing`` adds those durations to each request record; it is off by
    default so existing deterministic schedule records remain unchanged.
    """
    eng = llm.llm_engine
    sched = scheduler if scheduler is not None else scheduler_of(llm)
    reqs = list(requests)
    pending = sorted(range(len(reqs)), key=lambda i: (int(getattr(reqs[i], "arrive_step", 0) or 0), i))
    t0 = clock()
    rec: dict[str, dict] = {r.request_id: {"arrive_step": int(getattr(r, "arrive_step", 0) or 0), "arrive_step_realized": None,
                                           "first_step": None, "finish_step": None, "finish_reason": None} for r in reqs}
    finished: dict[str, Any] = {}
    steps = 0                     # step() iterations
    t_add = 0.0
    nonempty: list[bool] = []     # per iteration: did step() return outputs?
    unfinished_after: list[bool] = []   # per iteration: did the engine still report unfinished requests right after step()?
    computed: list[dict[str, int]] = []  # per iteration (scheduler reachable): {rid: num_computed_tokens after this iteration's schedule}
    admit_iter: dict[str, int] = {}
    finish_iter: dict[str, int] = {}
    admit_at: dict[str, float] = {}
    first_at: dict[str, float] = {}

    def _admit(i: int) -> None:
        nonlocal t_add
        r = reqs[i]
        ta = clock()
        eng.add_request(request_id=r.request_id, prompt={"prompt_token_ids": list(r.prompt_token_ids)}, params=params_for(r))
        admit_at[r.request_id] = ta
        t_add += clock() - ta
        admit_iter[r.request_id] = steps

    while pending or eng.has_unfinished_requests():
        while pending and int(getattr(reqs[pending[0]], "arrive_step", 0) or 0) <= steps:
            _admit(pending.pop(0))
        if not eng.has_unfinished_requests():
            _admit(pending.pop(0))                 # idle gap: bring the next arrival forward (recorded as its realized step)
            continue
        outs = eng.step()
        nonempty.append(bool(outs))
        if sched is not None:
            snap = _snapshot_computed(sched)
            if snap is None:
                sched = None                        # a failed read: the schedule source is abandoned for this run, not half-used
            else:
                computed.append(snap)
        unfinished_after.append(bool(eng.has_unfinished_requests()))
        if on_step is not None:
            on_step(outs)
        for o in outs:
            if o.request_id in rec and o.request_id not in first_at and getattr(o, "outputs", None):
                first_at[o.request_id] = clock()
            if getattr(o, "finished", False) and o.request_id in rec:
                finished[o.request_id] = o
                finish_iter[o.request_id] = steps
                fr = [getattr(c, "finish_reason", None) for c in o.outputs]
                rec[o.request_id]["finish_reason"] = (str(fr[0]) if fr[0] is not None else None) if len(fr) == 1 else [str(x) for x in fr]
        steps += 1
    dt = clock() - t0
    missing = [r for r in rec if r not in finished]
    if missing:
        raise RuntimeError(f"{len(missing)} requests did not finish: {missing[:3]}")
    # iterations -> forward indices (module docstring "Which iterations ran a forward")
    n_it = len(nonempty)
    first_admit = min(admit_iter.values()) if admit_iter else 0
    first_out = next((i for i, ne in enumerate(nonempty) if ne), None)
    measured = 0 if first_out is None else max(0, first_out - first_admit)
    lag_source = None
    if sched is not None and len(computed) == n_it:
        ran, last_adv = forwards_from_schedule(computed)
        lags = {finish_iter[rid] - last_adv[rid] for rid in finish_iter if rid in last_adv}
        if len(lags) == 1 and 0 <= min(lags) <= measured:   # every request's finish came the same number of iterations after its last forward
            lag = min(lags)
            lag_source = "scheduler"
    if lag_source is None:
        structural = structural_output_lag(eng)
        lag = measured if structural is None else min(measured, structural)
        lag_source = "measured" if lag == measured else "clamped"
        ran = list(unfinished_after) if lag >= 1 else [True] * n_it
    fwd_index = [0] * (n_it + 1)                   # fwd_index[i] = forwards before iteration i
    for j in range(len(nonempty)):
        fwd_index[j + 1] = fwd_index[j] + (1 if ran[j] else 0)
    n_forwards = fwd_index[len(nonempty)]
    for rid in rec:
        ai = admit_iter[rid]
        fi = finish_iter[rid]
        rec[rid]["arrive_step_realized"] = fwd_index[min(ai, len(nonempty))]
        rec[rid]["first_step"] = fwd_index[min(ai, len(nonempty))]
        rec[rid]["finish_step"] = max(0, fwd_index[min(max(fi - lag, 0), len(nonempty))])   # forward that sampled the last token
        rec[rid]["admit_iter"] = ai
        rec[rid]["finish_iter"] = fi
        if request_timing:
            first = first_at.get(rid)
            rec[rid]["arrival_s"] = admit_at[rid] - t0
            rec[rid]["first_output_s"] = None if first is None else first - t0
            rec[rid]["time_to_first_token_s"] = None if first is None else first - admit_at[rid]
    gen = sum(len(o.outputs[0].token_ids) if o.outputs else 0 for o in finished.values())
    prompt = sum(len(r.prompt_token_ids) for r in reqs)
    for rid, o in finished.items():
        rec[rid]["n_generated"] = len(o.outputs[0].token_ids) if o.outputs else 0
    timing = {"wall_s": dt, "add_requests_s": t_add, "engine_steps": n_forwards, "step_iterations": steps, "output_lag_steps": lag,
              "output_lag_source": lag_source, "output_lag_measured": measured,
              "generated_tokens": gen, "prompt_tokens": prompt,
              "generated_tokens_per_s": gen / dt if dt else None, "total_tokens_per_s": (gen + prompt) / dt if dt else None,
              "arrivals": True, "requests": rec}
    return finished, timing


def admission_order(requests: list[dict]) -> list[dict]:
    """The same rule for callers that step the engine themselves (noninterference's hooks variant): requests sorted by
    `arrive_step` (stable), each to be admitted right before that engine step."""
    return sorted(requests, key=lambda r: int(r.get("arrive_step") or 0))
