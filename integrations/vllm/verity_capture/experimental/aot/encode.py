"""EncodeHints(φ, record) -> h: the oracle side.  Predict-then-correct against the observed execution record.

The observed *structural* record is the step stream of the capture log — `StepStart` (token rows after vLLM's input
preparation, `seq_lens`, `logits_rows`) and `RunEnd` — nothing else is read (no tensor values, no sampled ids, no views).
`observe_schedule(log)` turns it into the same `StepPlan` list the predictor produces; `encode_hints(φ, observed)` then
finds the residual:

    1. served lengths per request (observed served samples ≠ `max_tokens`)            -> hints.served
    2. re-predict; for the first step whose row layout / sampled rows differ, record a `StepRows` override; repeat
       (each override re-synchronises the simulation, so one pass in step order converges)  -> hints.steps
    3. samples the engine discarded where the predictor served them                       -> hints.discarded
    4. assert predict_schedule(φ, h) == observed (else raise: the vocabulary cannot express this run — a result)

Everything the predictor already gets right costs nothing; the returned `Hints` is exactly the correction.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

from .hints import DiscardedSample, Hints, ServedLength, StepRows
from .phi import Phi
from .predict import PREDICTOR_ID, StepPlan, predict_schedule

_STEP_EVENTS = ('"$e":"StepStart"', '"$e":"RunEnd"', '"$e":"RunHeader"', '"$e": "StepStart"', '"$e": "RunEnd"', '"$e": "RunHeader"')


@dataclass
class Observed:
    plans: list[StepPlan]
    request_ids: list[str]
    ended: bool = False
    notes: list[str] = field(default_factory=list)


def observe_schedule(log_path: str | Path, phi: Phi) -> Observed:
    """The observed structural schedule of a capture log (StepStart stream only; request ids mapped to φ's request indices)."""
    reqs = phi.logical_requests()
    idx = {r["request_id"]: i for i, r in enumerate(reqs)}
    # n > 1: the header may record the fork (`workload.children`, written by the adapter); it must agree with the φ-derived rule
    # (child id `{index}_{parent}`, seed + index) — a disagreement is a φ-rule violation, never silently remapped.
    for c in (phi.workload.get("children") or []):
        j = idx.get(c["request_id"])
        if j is None or reqs[j]["_parent"] != next(i for i, r in enumerate(phi.requests) if r["request_id"] == c["parent"]) \
                or ("seed" in c and c["seed"] is not None and int(c["seed"]) != reqs[j].get("seed")):
            raise ValueError(f"RunHeader.workload.children entry {c!r} does not match the derived fork of φ (parallel_sampling rule)")
    plens = {i: len(r["prompt_token_ids"]) for i, r in enumerate(reqs)}
    p = Path(log_path)
    op = gzip.open if p.suffix == ".gz" else open
    plans: list[StepPlan] = []
    ended = False
    notes: list[str] = []
    # A sample is served iff the request's whole KNOWN sequence is computed after the step (vLLM: the sampled token is appended only
    # when num_computed_tokens >= request.num_tokens; the runner's prefill_len = len(all_token_ids) at (re)admission).  The known
    # sequence is the prompt plus every token served so far — so a chunk of a preemption RECOMPUTE that ends past the prompt but
    # short of the generated tokens is a discarded partial chunk, exactly like a partial prompt chunk (MEASURED pre6 step 123: the
    # older rule `seq_len >= prompt_len` called it served and made the record inexpressible).  The in-flight token of a preempted
    # request is still delivered (scheduler.py `_preempt_request`: stale output is appended, not dropped), so counting served samples
    # of all earlier steps is right for a resumed request too.
    served_so_far: dict[int, int] = {i: 0 for i in range(len(reqs))}
    with op(p, "rt") as f:
        for line in f:
            head = line[:40]
            if not any(tag in head for tag in _STEP_EVENTS):
                continue
            d = json.loads(line)
            e = d.get("$e")
            if e == "RunHeader":
                if d.get("profile_id") != phi.profile_id:
                    raise ValueError(f"log profile_id {d.get('profile_id')!r} != φ {phi.profile_id!r}")
                continue
            if e == "RunEnd":
                ended = True
                continue
            rows = [(idx[r], int(pos)) for r, pos in d["rows"]]
            lrows = [(idx[r], int(pos)) for r, pos in d["logits_rows"]]
            seq = {idx[r]: int(v) for r, v in (d.get("seq_lens") or {}).items()}
            served = [seq.get(r, plens[r]) >= plens[r] + served_so_far[r] for r, _ in lrows]
            for (r, _), s in zip(lrows, served):
                if s:
                    served_so_far[r] += 1
            kinds = {"prefill" if pos < plens[r] else "decode" for r, pos in rows}
            plans.append(StepPlan(int(d["step"]), rows, lrows, served, kinds.pop() if len(kinds) == 1 else "mixed"))
    if not plans:
        raise ValueError(f"{log_path}: no StepStart events")
    return Observed(plans, [r["request_id"] for r in reqs], ended, notes)


def _same(a: StepPlan, b: StepPlan) -> bool:
    return a.step == b.step and a.rows == b.rows and a.logits_rows == b.logits_rows and a.served == b.served


def encode_hints(phi: Phi, observed: Observed, *, max_rounds: int | None = None, predictor: str = PREDICTOR_ID) -> tuple[Hints, dict]:
    """The residual hints for `observed` under the predictor `predictor` (default: the current one), and a report of what was corrected."""
    obs = observed.plans
    hints = Hints(predictor=predictor)
    report: dict = {"observed_steps": len(obs), "corrections": []}
    # 1. served lengths
    lreqs = phi.logical_requests()
    served_obs: dict[int, int] = {i: 0 for i in range(len(lreqs))}
    for p in obs:
        for (r, _), s in zip(p.logits_rows, p.served):
            if s:
                served_obs[r] += 1
    for i, r in enumerate(lreqs):
        if served_obs[i] != int(r.get("max_tokens") or 0):
            hints.served.append(ServedLength(i, served_obs[i]))
            report["corrections"].append({"kind": "served", "request": i, "observed": served_obs[i], "predicted": int(r.get("max_tokens") or 0)})
    # 2. step layout overrides, first deviation first
    rounds = 0
    limit = max_rounds if max_rounds is not None else len(obs) + 2
    while True:
        pred = predict_schedule(phi, hints)
        dev = None
        for k in range(max(len(pred), len(obs))):
            if k >= len(obs):
                dev = ("extra-predicted", k)
                break
            if k >= len(pred) or not _same(pred[k], obs[k]):
                dev = ("differs" if k < len(pred) else "missing-predicted", k)
                break
        if dev is None:
            break
        rounds += 1
        if rounds > limit:
            raise RuntimeError(f"encode_hints: no convergence after {rounds} rounds (last deviation {dev})")
        kind, k = dev
        if kind == "extra-predicted":
            # the predictor runs steps the engine did not: only possible if a request was served fewer tokens than
            # recorded — which step 1 already fixed — or a scheduling artefact; report precisely and stop.
            raise RuntimeError(f"encode_hints: predictor produced step {k} ({pred[k].phase}, rows={len(pred[k].rows)}) beyond the "
                               f"observed {len(obs)} steps; served hints cannot express this")
        o = obs[k]
        default_lr = o.default_logits_rows()
        ov = StepRows(o.step, o.runs(), None if o.logits_rows == default_lr else tuple(o.logits_rows))
        hints.steps.append(ov)
        entry = {"kind": "step", "step": o.step, "why": kind, "observed_rows": len(o.rows), "observed_runs": [r.__dict__ for r in o.runs()]}
        if k < len(pred):
            entry["predicted_rows"] = len(pred[k].rows)
            entry["predicted_runs"] = [r.__dict__ for r in pred[k].runs()]
            entry["predicted_phase"], entry["observed_phase"] = pred[k].phase, o.phase
        report["corrections"].append(entry)
        # 3. discarded samples the layout does not explain (the served rule is shared; a deviation here is a real fact)
        pred2 = predict_schedule(phi, hints)
        if k < len(pred2) and pred2[k].rows == o.rows and pred2[k].logits_rows == o.logits_rows and pred2[k].served != o.served:
            for (r, pos), ps, os_ in zip(o.logits_rows, pred2[k].served, o.served):
                if ps and not os_:
                    hints.discarded.append(DiscardedSample(r, o.step, pos))
                    report["corrections"].append({"kind": "discarded", "request": r, "step": o.step, "position": pos})
                elif os_ and not ps:
                    raise RuntimeError(f"encode_hints: step {o.step} serves (request {r}, pos {pos}) which the rule says is a partial prefill")
    final = predict_schedule(phi, hints)
    if len(final) != len(obs) or any(not _same(a, b) for a, b in zip(final, obs)):
        raise RuntimeError("encode_hints: internal — corrected prediction still differs from the observation")
    report["hints_empty"] = hints.empty
    report["n_served_hints"], report["n_step_hints"], report["n_discarded_hints"] = len(hints.served), len(hints.steps), len(hints.discarded)
    report["predicted_steps"] = len(final)
    report["rounds"] = rounds
    return hints, report
