"""verity_vllm/engine/chunks.py — which engine steps were a request's prefill chunks, and which step produced each served token.
Torch-free.

vLLM V1 prefills a prompt in chunks when `max_num_batched_tokens` is smaller than the prompt: each chunk is one engine step (a
forward pass with partial-prefill attention), the engine samples a token at every chunk and DISCARDS the samples of the chunks
that leave the prompt incomplete (`gpu_model_runner`: `discard_request_mask = seq_len < num_tokens`); the first served token is the
sample of the chunk that completes the prompt, and `step()` returns NO output for a request at a partial-prefill step
("EngineCore returns no partial prefill outputs").  The realized record of a run (`timing["requests"][rid]`, D87) so far carries
`first_step` / `finish_step` only; this module adds, per request,

    prefill_steps          the forwards that scheduled prompt tokens of the request (its chunks), in order
    prefill_chunks         [{step, tokens, computed_before, prompt_complete}] -- `tokens` known from the scheduler source only
    n_prefill_chunks       len(prefill_steps)  (1 = unchunked prefill)
    partial_prefill_steps  the chunks after which the prompt was still incomplete (their sample is discarded by the engine)
    first_token_step       the forward whose sample became generated token 0 (== the last prefill chunk)
    token_steps            the forward that produced generated token i, for every served token

and, run-wide, `timing["chunk_attribution"]` = {source, steps_scheduled_tokens (per forward: {rid: tokens}), max_prefill_chunks,
chunked_prefill_observed, partial_prefill_steps_total, consistent, problems, corrections}.  Additive: no existing key changes
its meaning; see "Correction" for the one case in which existing VALUES are repaired.

Two sources.  (1) "scheduler-state": the in-process V1 scheduler (`llm.llm_engine.engine_core[.engine_core].scheduler`) keeps
`requests[rid].num_computed_tokens`, which `schedule()` advances by the tokens it schedules *before* the forward runs; reading it
after every `step()` iteration (a pure read -- nothing is patched, nothing touches the device) gives the tokens each iteration's
forward scheduled per request, hence the exact chunk sizes, and a request that WAITS (budget exhausted by another prefill) shows
a delta of 0 and is correctly not credited with a chunk.  (2) "outputs": when the scheduler is unreachable (multiprocess engine
core, a stub), the chunks are inferred as the forwards from `first_step` up to the forward that produced the first served token
-- exact for a request that is scheduled at every step from its admission (B1; every request of an unchunked cell), an
over-count for a request that waited.  The source is recorded.  Prefix caching (the prefix owner's cells): the scheduler sets a
waiting request's `num_computed_tokens` to its cache hits before adding the scheduled tokens, so the first chunk's `tokens`
then counts the cached positions too (`computed_before` stays 0); the chunk STEPS are unaffected.

Iterations vs forwards.  As in `arrivals.py`, `step()` returns the previous forward's sample on this vLLM (lag 1): the forward of
iteration j is forward index `fwd_index[j]`, and the outputs returned at iteration j were produced by the forward of iteration
j - lag.  `arrivals.run_requests_arrivals` MEASURES the lag as (first non-empty iteration) - (first admission iteration), which is
right whenever the first forward serves a token.  Under chunked prefill with no other request decoding it is not: a 155-token
prompt at budget 64 returns nothing at iterations 0, 1, 2 (lag + two partial chunks), the measured lag is 3 instead of 1, and the
arrival loop then reports 15 forwards for 17 (the control run's `timing.engine_steps` is the gates' step expectation, so G1/G2
would fail for a harness reason).  **Correction:** with the scheduler-state source the forwards are known exactly (an iteration
whose schedule advanced some request ran a forward) and so is the lag (per request: finish iteration - the last iteration that
advanced it; NOT `iterations - forwards`, which an idle gap between arrivals inflates), and when that disagrees with the arrival
loop's inference `attribute()` repairs `timing["engine_steps"]`, `timing["output_lag_steps"]` and every request's
`arrive_step_realized` / `first_step` (the forward index at admission, the loop's meaning) / `finish_step` (the forward that
scheduled its last token), keeping the loop's values under `timing["chunk_attribution"]["corrections"]`.  Where `arrivals.py` derives the
forwards from the scheduler itself, the correction is a no-op and `corrections` stays None.

Request ids.  The pinned vLLM runs the scheduler under an internal id f"{submitted}-{8 hex}" (`input_processor.assign_request_id`)
and returns outputs under the submitted id; `external_id` maps the scheduler's ids back (run `0917T213719-880581` showed the
mismatch: `steps_scheduled_tokens` under the internal id, "no prefill chunk seen in the scheduler state" per request).  The map
is recorded as `internal_request_ids` = {submitted: internal}.

`Collector.on_step` is the `on_step` callback of `arrivals.run_requests_arrivals`; `attribute()` turns its per-iteration records +
the arrival loop's timing into the keys above.
"""
from __future__ import annotations

import re
from typing import Any


def scheduler_of(llm: Any) -> Any | None:
    """The in-process V1 scheduler of an offline `LLM` (`llm_engine.engine_core[.engine_core].scheduler`, as
    `verity_vllm.engine.handle.scheduler`), or None when the path does not exist or exposes no `requests` dict."""
    try:
        eng = llm.llm_engine
        core = getattr(eng, "engine_core", None)
        inner = getattr(core, "engine_core", None) or core
        sched = getattr(inner, "scheduler", None)
        return sched if isinstance(getattr(sched, "requests", None), dict) else None
    except Exception:  # noqa: BLE001
        return None


class Collector:
    """Per-iteration facts for `attribute()`: whether `step()` returned outputs, the new tokens per request (child 0), and --
    when a scheduler is given -- {rid: (num_computed_tokens after this iteration's schedule, num_prompt_tokens)}."""

    def __init__(self, scheduler: Any | None = None) -> None:
        self.scheduler = scheduler
        self.nonempty: list[bool] = []
        self.new_tokens: list[dict[str, int]] = []
        self.computed: list[dict[str, tuple[int, int | None]]] = []
        self._len: dict[str, int] = {}
        self.errors: list[str] = []

    def on_step(self, outs: list[Any]) -> None:
        self.nonempty.append(bool(outs))
        new: dict[str, int] = {}
        for o in outs or []:
            try:
                n = len(o.outputs[0].token_ids)
            except Exception:  # noqa: BLE001
                continue
            rid = str(o.request_id)
            new[rid] = n - self._len.get(rid, 0)
            self._len[rid] = n
        self.new_tokens.append(new)
        if self.scheduler is not None:
            snap: dict[str, tuple[int, int | None]] = {}
            try:
                for rid, req in list(self.scheduler.requests.items()):
                    nc = getattr(req, "num_computed_tokens", None)
                    if nc is None:
                        continue
                    npt = getattr(req, "num_prompt_tokens", None)
                    snap[str(rid)] = (int(nc), None if npt is None else int(npt))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"iteration {len(self.nonempty) - 1}: scheduler read failed: {e!r}"[:200])
            self.computed.append(snap)

    @property
    def source(self) -> str:
        return "scheduler-state" if self.scheduler is not None else "outputs"


def _forward_map(ran: list[bool]) -> list[int]:
    """fwd_index[j] = forwards before iteration j."""
    fwd = [0] * (len(ran) + 1)
    for j, r in enumerate(ran):
        fwd[j + 1] = fwd[j] + (1 if r else 0)
    return fwd


def _ran_from_outputs(nonempty: list[bool], lag: int) -> list[bool]:
    """`arrivals.run_requests_arrivals`' rule: forward ran at iteration j <=> iteration j + lag returned outputs."""
    n = len(nonempty)
    return [(j + lag < n and bool(nonempty[j + lag])) for j in range(n)]


def _deltas(computed: list[dict[str, tuple[int, int | None]]]) -> list[dict[str, tuple[int, int, int | None]]]:
    """Per iteration: {rid: (tokens scheduled by this iteration's forward, computed_before, prompt_len)} (positive deltas only)."""
    out: list[dict[str, tuple[int, int, int | None]]] = []
    prev: dict[str, int] = {}
    for snap in computed:
        d: dict[str, tuple[int, int, int | None]] = {}
        for rid, (nc, npt) in snap.items():
            t = nc - prev.get(rid, 0)
            if t > 0:
                d[rid] = (t, nc - t, npt)
            prev[rid] = nc
        out.append(d)
    return out


def external_id(rid: str, known: set[str]) -> str:
    """The id we submitted for a scheduler-side request id.  vLLM (`v1/engine/input_processor.py::assign_request_id`, pinned wheel
    0.28.1rc1) runs the scheduler under an INTERNAL id f"{external}-{8 hex}" (unless VLLM_DISABLE_REQUEST_ID_RANDOMIZATION) while
    `step()` returns outputs under the external id; the scheduler's `Request` does not carry the external id, so it is recovered
    by stripping the 8-hex suffix when the stripped id is one we know (a parent id, or a parallel-sampling child f"{i}_{parent}").
    Unknown ids are returned unchanged (never guessed)."""
    if rid in known:
        return rid
    if len(rid) > 9 and rid[-9] == "-" and all(c in "0123456789abcdef" for c in rid[-8:]):
        base = rid[:-9]
        m = _CHILD_RE.match(base)
        if base in known or (m is not None and m.group(2) in known):
            return base
    return rid


_CHILD_RE = re.compile(r"^(\d+)_(.+)$")


def attribute(timing: dict, collector: Collector, prompt_len: dict[str, int], *, resolved_lag: int | None = None) -> dict:
    """Add the chunk keys to `timing["requests"][rid]` (in place, additive) and return `timing["chunk_attribution"]`.

    `resolved_lag` = the engine's RESOLVED async_scheduling as LAG (1 / 0; None = unattested), the record's lag of
    record (tokens.json `lag.value`).  The scheduler-state inference below derives a lag from schedule deltas (finish iteration - last
    advancing iteration); under async_scheduling=False the finishing request is freed inside step() BEFORE the post-step snapshot, so its
    last forward is invisible, last_adv one early, and the derived lag reads 1 for a true 0: applied, it would rewrite a sync record's
    finish_step / engine_steps / output_lag_steps one early, and the global match's G6 would fail the record against its own tokens.  The
    loop's step fields STAND (a) when the derived lag contradicts `resolved_lag`, and (b) when the corrected model fails this function's own
    consistency check; the contradiction is a named problem and `corrections.applied` is False -- never a silent rewrite, never a lag
    inferred from the trace as the record's lag."""
    recs: dict[str, dict] = timing.get("requests") or {}
    nonempty = list(collector.nonempty)
    n_it = len(nonempty)
    problems: list[str] = list(collector.errors)
    corrections: dict[str, Any] | None = None

    lag = int(timing.get("output_lag_steps") or 0)
    ran = _ran_from_outputs(nonempty, lag)
    use_sched = collector.scheduler is not None and len(collector.computed) == n_it
    deltas = _deltas(collector.computed) if use_sched else []
    # scheduler ids -> the ids we submitted (vLLM's internal f"{id}-{8 hex}" randomization; see `external_id`)
    known = set(recs) | set(prompt_len)
    internal_ids: dict[str, str] = {}
    for snap in collector.computed:
        for rid in snap:
            ext = external_id(rid, known)
            if ext != rid:
                internal_ids[ext] = rid
    if internal_ids:
        deltas = [{external_id(rid, known): v for rid, v in d.items()} for d in deltas]
    if use_sched:
        # the forwards are known exactly: an iteration whose schedule advanced some request ran one.  The lag is NOT
        # `iterations - forwards` (an idle gap -- a finished batch drained before the next arrival is due -- adds a no-forward
        # iteration that is not a lag): it is, per request, finish iteration - last iteration that advanced
        # it (the forward that sampled its last token), the same value for every request on one engine.
        ran_s = [bool(d) for d in deltas]
        n_fwd = sum(ran_s)
        last_adv: dict[str, int] = {}
        for j, d in enumerate(deltas):
            for rid in d:
                last_adv[rid] = j
        lags = sorted(int(rec["finish_iter"]) - last_adv[rid] for rid, rec in recs.items()
                      if rec.get("finish_iter") is not None and rid in last_adv)
        if lags:
            lag_s = max(0, lags[len(lags) // 2])
            if lags[0] != lags[-1]:
                problems.append(f"per-request output lags differ: {sorted(set(lags))}")
        else:
            lag_s = max(0, n_it - n_fwd)
        loop_steps = timing.get("engine_steps")
        fwd_s = _forward_map(ran_s)
        # the exact per-request values (the loop's meanings: first_step = forward index at admission, finish_step = the forward
        # that sampled the last token); a correction is recorded only when a VALUE the loop wrote differs (a loop that
        # derives the forwards from the scheduler itself agrees -> corrections None)
        exact: dict[str, dict[str, int]] = {}
        for rid, rec in recs.items():
            e: dict[str, int] = {}
            ai = rec.get("admit_iter")
            fi = rec.get("finish_iter")
            if ai is not None:
                e["arrive_step_realized"] = e["first_step"] = fwd_s[min(int(ai), n_it)]
            if rid in last_adv:
                e["finish_step"] = fwd_s[last_adv[rid]]
            elif fi is not None:
                e["finish_step"] = max(0, fwd_s[min(max(int(fi) - lag_s, 0), n_it)])
            exact[rid] = e
        differs = lag_s != lag or (loop_steps is not None and int(loop_steps) != n_fwd) or any(
            rec.get(k) != v for rid, rec in recs.items() for k, v in exact[rid].items())
        if differs:
            corrections = {"reason": "the arrival loop infers the output lag from the first returned output; a partial-prefill step "
                                     "returns nothing, so its inference breaks when the first forward serves no token (chunked prefill "
                                     "with no request decoding beside it)",
                           "loop": {"engine_steps": loop_steps, "output_lag_steps": lag,
                                    "requests": {rid: {k: rec.get(k) for k in ("arrive_step_realized", "first_step", "finish_step")} for rid, rec in recs.items()}},
                           "corrected": {"engine_steps": n_fwd, "output_lag_steps": lag_s, "requests": exact}, "applied": False}
            if resolved_lag is not None and lag_s != int(resolved_lag):
                # the derived lag contradicts the RESOLVED profile: the snapshots missed a forward (sync frees the
                # finishing request inside step()); the loop's fields stand, the outputs inference is the forward map, the contradiction is named
                problems.append(f"scheduler-state inference: derived output lag {lag_s} contradicts the resolved async_scheduling LAG "
                                f"{int(resolved_lag)} (the record's lag is the resolved profile, never a schedule-delta inference) -- "
                                f"the loop's engine_steps {loop_steps} / output_lag_steps {lag} / finish_step values stand")
                corrections["reason"] += "; NOT APPLIED: the derived lag contradicts the resolved async_scheduling"
                use_sched_map = False
            else:
                timing["engine_steps"] = n_fwd
                timing["output_lag_steps"] = lag_s
                for rid, rec in recs.items():
                    rec.update(exact[rid])
                corrections["applied"] = True
                use_sched_map = True
        else:
            use_sched_map = True
        if use_sched_map:
            ran, lag = ran_s, lag_s
    fwd = _forward_map(ran)

    # served tokens -> producing forward (outputs returned at iteration j came from the forward of iteration j - lag)
    token_steps: dict[str, list[int]] = {rid: [] for rid in recs}
    for j, new in enumerate(collector.new_tokens):
        for rid, k in new.items():
            if k <= 0 or rid not in token_steps:
                continue
            src = max(j - lag, 0)
            step = fwd[src] if ran[src] else max(fwd[src] - 1, 0)
            if not ran[src]:
                problems.append(f"iteration {j}: {k} token(s) of {rid} returned but no forward is attributed to iteration {src}")
            token_steps[rid].extend([step] * k)

    # scheduler-state source: the tokens each forward scheduled per request -> chunks / decode forwards
    chunks: dict[str, list[dict]] = {rid: [] for rid in recs}
    decode_steps: dict[str, list[int]] = {rid: [] for rid in recs}
    sched_steps: list[dict[str, int]] = []
    if use_sched:
        for j, d in enumerate(deltas):
            if not ran[j]:
                if d:
                    problems.append(f"iteration {j}: the scheduler advanced {sorted(d)} but no forward is attributed to this iteration")
                continue
            step = fwd[j]
            sched_steps.append({rid: int(t) for rid, (t, _, _) in d.items()})
            for rid, (t, before, npt) in d.items():
                if rid not in recs:
                    continue
                plen = int(npt) if npt is not None else int(prompt_len.get(rid, 0))
                if before < plen:
                    chunks[rid].append({"step": step, "tokens": int(t), "computed_before": int(before), "prompt_complete": before + t >= plen})
                else:
                    decode_steps[rid].append(step)
            if not d:
                problems.append(f"forward {step}: a forward ran but the scheduler advanced no request")

    for rid, rec in recs.items():
        ts = token_steps.get(rid, [])
        first_tok = ts[0] if ts else None
        if use_sched and chunks[rid]:
            pre: list[dict] | None = chunks[rid]
            steps_pre = [c["step"] for c in pre]
            partial = [c["step"] for c in pre if not c["prompt_complete"]]
            if first_tok is not None and steps_pre[-1] != first_tok:
                problems.append(f"{rid}: last prefill chunk at forward {steps_pre[-1]} but the first served token came from forward {first_tok}")
            if decode_steps[rid] and ts[1:] != decode_steps[rid]:
                problems.append(f"{rid}: decode forwards {decode_steps[rid][:6]}.. != token-producing forwards {ts[1:7]}..")
        else:
            if use_sched:
                problems.append(f"{rid}: no prefill chunk seen in the scheduler state; falling back to the outputs inference")
            first = rec.get("first_step")
            if first is None or first_tok is None:
                steps_pre, partial, pre = [], [], None
            else:
                steps_pre = list(range(int(first), int(first_tok) + 1))
                partial = steps_pre[:-1]
                pre = None
        rec["prefill_steps"] = steps_pre
        rec["prefill_chunks"] = pre
        rec["n_prefill_chunks"] = len(steps_pre)
        rec["partial_prefill_steps"] = partial
        rec["first_token_step"] = first_tok
        rec["token_steps"] = ts
        n_gen = rec.get("n_generated")
        if n_gen is not None and len(ts) != int(n_gen):
            problems.append(f"{rid}: {len(ts)} token-producing forwards for {n_gen} generated tokens")
        if ts and rec.get("finish_step") is not None and ts[-1] != int(rec["finish_step"]):
            problems.append(f"{rid}: finish_step {rec['finish_step']} != the forward of the last served token {ts[-1]}")
    # a correction that fails THIS function's own consistency check is not applied: the loop's step fields are restored
    # (timing.engine_steps / output_lag_steps, each request's arrive_step_realized / first_step / finish_step), `corrections.applied` is False
    # and the revert is a named problem -- the derived chunk keys above stay as the diagnostic they are, under consistent: False
    if corrections is not None and corrections.get("applied") and problems:
        loop_blk = corrections["loop"]
        if loop_blk.get("engine_steps") is not None:
            timing["engine_steps"] = loop_blk["engine_steps"]
        timing["output_lag_steps"] = loop_blk["output_lag_steps"]
        for rid, rec in recs.items():
            for k, v in (loop_blk["requests"].get(rid) or {}).items():
                if v is None:
                    rec.pop(k, None)
                else:
                    rec[k] = v
        corrections["applied"] = False
        corrections["reason"] += "; REVERTED: the corrected model failed the attribution's own consistency check"
        problems.insert(0, f"scheduler-state correction reverted: the corrected model (engine_steps {corrections['corrected']['engine_steps']}, "
                           f"output_lag_steps {corrections['corrected']['output_lag_steps']}) failed {len(problems)} consistency check(s); the loop's "
                           f"step fields stand")
    n_chunks = [int(r.get("n_prefill_chunks") or 0) for r in recs.values()]
    out = {"source": "scheduler-state" if use_sched else ("outputs" if collector.scheduler is None else "outputs (scheduler snapshots incomplete)"),
           "steps_scheduled_tokens": sched_steps if use_sched else None,
           "max_prefill_chunks": max(n_chunks, default=0),
           "chunked_prefill_observed": any(n > 1 for n in n_chunks),
           "partial_prefill_steps_total": sum(len(r.get("partial_prefill_steps") or []) for r in recs.values()),
           "internal_request_ids": internal_ids or None,
           "consistent": not problems, "problems": problems[:32], "corrections": corrections}
    timing["chunk_attribution"] = out
    return out
