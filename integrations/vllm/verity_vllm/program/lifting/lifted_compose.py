"""The lifted fixed WORKLOAD Program (MIXED BATCH): GP-01's compose (`program.workload`), unchanged, over the lifted
components of the declared requests (`registry.lifted.lift_request_component`)."""

from __future__ import annotations

from typing import Any, Sequence

from verity_vllm.program.registry.lifted import BOUNDARY_PREFIX, check_lag, lift_request_component, lift_type


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
