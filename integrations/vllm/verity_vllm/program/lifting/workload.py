"""The lifted Program where it meets the rest of the integration, beside its Definitions (`verity_vllm.program.registry.lifted`):
GP-01's workload composed over lifted components (`lift_component`, `lifted_workload`, `workload_program`), the post-EOS
lag a run header attests and a Build declares (`lag_from_header`, `declared_lag_of_row`), and the use document of a lifted
VU-partition (`lifted_vu_use_document`)."""
from __future__ import annotations

import re
from typing import Any, Sequence

from verity_vllm.program.registry.lifted import (BOUNDARY_PREFIX, LIFTED_RULES_VERSION, N_OUT_HISTOGRAM_MAX_SETS, USE_KIND_LIFTED,
                                                 W_MAX, W_MAX_LIFTED, W_MAX_LIFTED_STATUS, check_lag, lift_request_component,
                                                 lift_type, width_summary_a44)


LAG_FIELD = "RunHeader.profile.engine_kwargs.async_scheduling"  # the ONE attested field GP-01's eos_lag_rows and this LAG both read
LAG_DECLARED_FIELDS = ("lag.value", "stop_rule.lag", "stop_rule.eos_lag_rows")  # workload_program.json: moe M-0015 names, then GP-01's pre-M-0015 field
LAG_STATIC_FIELD = "workload_fn"  # the digest-bound source: the `lag=<n>` static of `Workload_v1{requests, shared, lag}` in the Program's own spec id
_LAG_STATIC_RE = re.compile(r"^Workload_v1\{.*,lag=(\d+)\}$", re.S)  # the LAST binding of the record, at the top level (rev F-r17-02: the static
# lives in the spec-id string, not in a bindings dict of the Root wrapper)


def _lag_rule(engine_block: Any) -> int | None:
    """The ONE implementation of the lag rule (R17-1 / SPEC v0.5 A2.2; coord M-0017 "by reference to moe's unit, not a second
    implementation"): GP-01's `eos_lag_rows_declared` on an engine block (`async_scheduling` True → 1, False → 0, anything else → None).
    When moe's `TargetProfile.lag()` lands, this is the one call that moves."""
    from verity_vllm.pipeline.global_program import eos_lag_rows_declared
    return eos_lag_rows_declared(engine_block)


def lag_from_header(header: dict) -> tuple[int, str]:
    """R17-1 (SPEC v0.5 A2.2; RULING 21 before it): the record's ATTESTATION of the lag — the RESOLVED `SchedulerConfig.async_scheduling`
    in `profile.engine_kwargs` (True → 1, False → 0), read through the same function GP-01 / GM-01 read (`_lag_rule`), from nowhere
    else; an absent or non-boolean attestation is refused by name (never a default of 0 or 1).  Returns `(lag, field)`; the field
    string goes into the report.  This is the attested side; the DECLARED side is `declared_lag_of_row` and the two must agree (R0)."""
    try:
        ek = header["profile"]["engine_kwargs"]
        if not isinstance(ek, dict) or "async_scheduling" not in ek:
            raise KeyError("async_scheduling")
    except (KeyError, TypeError):
        raise ValueError(f"{LAG_FIELD} is not attested in the run header: the post-EOS lag is a declared static (R17-1) and is never "
                         f"defaulted — refuse the record (attestation is exec/sys's int-3 unit, RULING 21)") from None
    lag = _lag_rule(ek)
    if lag is None:
        raise ValueError(f"{LAG_FIELD} = {ek.get('async_scheduling')!r} is not the resolved boolean (the CLI default `None` is not an attestation)")
    return check_lag(lag), LAG_FIELD


def declared_lag_of_row(workload_program: dict | None) -> tuple[int | None, str | None]:
    """R17-1 (SPEC v0.5 A2.1): the lag the Build DECLARED as a static of the Program of record, read from the row's
    `workload_program.json` — moe's `lag.value` / `stop_rule.lag` (M-0015), else GP-01's `stop_rule.eos_lag_rows` (the field the
    pre-M-0015 Build wrote from `applicability.engine.async_scheduling`).  Returns `(lag, field)`; `(None, None)` when the Build
    declared none — then the lifted tool's R0 is never ok (a Program with no lag declared is not a lifted Program of record).
    Nothing is read from a trace and nothing is defaulted.

    Sources, all of one artifact and all required to agree: the DIGEST-BOUND static `lag=<n>` in the Program's spec id (`workload_fn` =
    `Workload_v1{requests=…,shared=…,lag=n}`; rev F-r17-02: the static lives in the spec-id string, the Root wrapper's bindings are `{}`),
    and the sidecar fields `lag.value` / `stop_rule.lag` / `stop_rule.eos_lag_rows` moe's GP-01 writes from the same `wl.lag`.  A sidecar
    edited against the digest (or a digest against the sidecar) is "the declaration disagrees with itself", refused by name — the field
    reported first is the digest-bound one when present."""
    if not isinstance(workload_program, dict):
        return None, None
    found: list[tuple[int, str]] = []
    fn = workload_program.get(LAG_STATIC_FIELD)
    if isinstance(fn, str):
        m = _LAG_STATIC_RE.match(fn.strip())
        if m:
            found.append((check_lag(int(m.group(1))), LAG_STATIC_FIELD + ":lag="))
    for field in LAG_DECLARED_FIELDS:
        cur: Any = workload_program
        for part in field.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, bool):  # a bare boolean is the knob, not the lag: run it through the one rule
            cur = _lag_rule({"async_scheduling": cur})
        if isinstance(cur, int) and not isinstance(cur, bool):
            found.append((check_lag(cur), field))
    if not found:
        return None, None
    if len({v for v, _ in found}) > 1:  # one artifact, two values: not a declaration (moe M-0047: lag.value == stop_rule.lag == eos_lag_rows)
        raise ValueError("the Build's declaration disagrees with itself: " + ", ".join(f"{f}={v}" for v, f in found) + " — refused by name")
    return found[0]


def lifted_vu_use_document(query: Any, program: Any, verdict: Any, *, w_max_lifted: int = W_MAX_LIFTED, utc: str | None = None,
                           fam: Any = None, histogram_max_sets: int = N_OUT_HISTOGRAM_MAX_SETS) -> tuple[dict, dict]:
    """SPEC v0.5 A5: the (query document, use document) of a LIFTED VU-partition use — `query_artifact.vu_use_document` with the use
    configuration `{"kind": "vu-partition-lifted", "w_max": w_max_lifted, "rules_version": "lifted-v1", "w_max_active_reference": 32}`
    and (revb M-0042, B1's condition) a `width_summary` {w_max_lifted, max_seen (the validator's), n_out histogram (the boundary
    module's; `fam` = the evaluated family, else the query is evaluated here)}.  The query document is unchanged in shape (no kind, no
    w_max: it binds the query to Lift(P)'s digest).  `gates.gate_G8` refuses this document by name (`use.kind != "vu-partition"`): a
    lifted verdict never reads as an active-reference G8 pass."""
    from verity_vllm.query import query_artifact as QA
    from verity.ir.query_ast import evaluate
    qdoc, use = QA.vu_use_document(query, program, verdict, w_max=int(w_max_lifted), utc=utc)
    use["use"] = {"kind": USE_KIND_LIFTED, "w_max": int(w_max_lifted), "rules_version": LIFTED_RULES_VERSION,
                  "w_max_active_reference": W_MAX, "w_max_lifted_status": W_MAX_LIFTED_STATUS}
    fam = evaluate(program, query) if fam is None else fam
    use["width_summary"] = width_summary_a44(program, fam, getattr(verdict, "width", verdict), w_max_lifted=w_max_lifted, histogram_max_sets=histogram_max_sets)
    return qdoc, use


def lift_component(component: Any, eos: Sequence[int], *, max_tokens: int, lag: int, boundary_prefix: str = BOUNDARY_PREFIX) -> Any:
    """GP-01's `RequestComponent` → the same component, lifted (F-pad-14 on its `fn`; PAD-03 modulo the guard nodes).  Same parameter
    GROUPS (the shared `weights` group stays shared: one lifted weights Input for the whole workload), lifted parameter types, the same
    `input_gates` (a leaf is a gate whatever its width), `body_gates` = the unlifted + exactly the guard gates, and the step segmentation
    recomputed on the lifted body from its sampling-event nodes (a step's guard nodes precede the step's first token reader, so they
    belong to the step they gate).  The `program_digest`/`program_fn` identity is the UNLIFTED component's (that is what the lift is of)."""
    import hashlib
    import json
    from verity_vllm.program import workload as W
    lifted = lift_request_component(component.fn, eos, max_tokens=max_tokens, lag=lag, boundary_prefix=boundary_prefix)
    body = lifted.body
    nidx = lifted.node_map  # type: ignore[attr-defined]
    segs, lo = [], 0
    for b in (nidx[b] for b in lifted.boundaries):  # type: ignore[attr-defined]
        segs.append((lo, b + 1))
        lo = b + 1
    if lo != len(body.nodes):
        # F-r17b-43: the nodes after the LAST sampling event are the last slot's Continuation_v1 controller (`slot=<cap>/is_eos`,
        # `slot=<cap>/out`: its EOS test and designated output) -- they belong to the step that produced that token, not to a
        # (cap+1)-th step; GP-01's chronology requires exactly `cap` step segments per component (`component_steps_equal_cap`)
        segs[-1] = (segs[-1][0], len(body.nodes))
    step_gates = tuple((body.offsets[a], body.offsets[b]) for a, b in segs)
    params = tuple((n, lift_type(t)) for n, t in component.params)  # the component's OWN parameter names (grouping), lifted types
    return W.RequestComponent(lifted, component.program_digest, component.program_fn, params, component.input_gates, body.gates,
                              len(body.nodes), hashlib.sha256(json.dumps(body.offsets).encode()).hexdigest(), component.groups, tuple(segs), step_gates)


def lifted_workload(requests: Sequence[Any], eos: Sequence[int], *, lag: int, boundary_prefix: str = BOUNDARY_PREFIX, shared: str = "weights",
                    manifest_digest: str | None = None, ordering: str = "manifest") -> Any:
    """The lifted fixed WORKLOAD Program (MIXED BATCH, Program side): GP-01's `WorkloadProgram` compose — the same code, unchanged — over
    the lifted components of the declared requests (`WorkloadRequest`s from the manifest: id, index, prompt_len, cap, arrive_step).  The
    population is `Σ_r cap_r` sampling events fixed by the manifest; one shared lifted weights Input; the address map and the chronology are
    GP-01's own functions applied to the lifted Program (per request: the unlifted rows + that request's guard rows, PAD-03).  Every request
    carries its own guard on its own tokens and NOTHING crosses between requests (the compose has no cross-request edge; the S-migration
    count is a root-level value of the committed tokens, `live_count_hook`).  Nothing about a stop enters: the same object for every transcript."""
    from verity_vllm.program import workload as W
    check_lag(lag)
    reqs = []
    for r in requests:
        lc = lift_component(r.component, eos, max_tokens=r.cap, lag=lag, boundary_prefix=boundary_prefix)
        reqs.append(W.WorkloadRequest(r.request_id, r.index, r.prompt_len, r.cap, r.arrive_step, lc, r.T))
    return workload_program(reqs, lag, shared=shared, manifest_digest=manifest_digest, ordering=ordering)


def workload_program(reqs: Sequence[Any], lag: int, **kw: Any) -> Any:
    """GP-01's compose, with the declared LAG passed as the `Workload_v1` static when the tree carries moe's R17-1 unit (lane/r17-moe
    a96a2071b: `WorkloadProgram(lag=)` required, the lag in the workload digest — DP-pad-02 (b)); on a tree before it the compose has
    no lag parameter and none is passed.  The signature is INSPECTED, never guessed: the same call composes the same Program the Build did."""
    import inspect
    from verity_vllm.program import workload as W
    if "lag" in inspect.signature(W.WorkloadProgram.__init__).parameters:
        return W.WorkloadProgram(list(reqs), lag=check_lag(lag), **kw)
    return W.WorkloadProgram(list(reqs), **kw)


__all__ = ["lifted_vu_use_document", "lift_component", "lifted_workload", "LAG_FIELD", "LAG_DECLARED_FIELDS", "LAG_STATIC_FIELD",
           "lag_from_header", "declared_lag_of_row"]
