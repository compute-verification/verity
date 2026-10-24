"""Batch invariance of the request-major compiler (CIR-87): the templates a registration's compilations call, for `circuit-check
--batch`, and the rule `steps.compile_steps` holds every compilation to (`check_compiled`).

A request's outputs may depend on its own certified inputs (its prompt, its seed) and on what the registration fixes (weights,
tables, constants), never on its batch-mates. The batch reaches a Program two ways an owner-labelled check alone wouldn't see:

- a request-owned Input whose value the batch picks: the seeded top-p sampler's split count, S = SplitsFor_v1(live rows,
  num_SMs) ("splits": "derived"), which changes the top-p keep set at top_p < 1;
- a launch static: a kind bound from the launch's shape (RMSNorm's BLOCK from its rows off batch invariance, FA3's tile from
  max_seqlen_q, FA2's guard `MASKED_FROM` from the launch's causal flag on cc 12.0).

So Inputs are labelled by provenance (`OWN_GROUPS` own, the registration's shared, any other request-local group `BATCH`), and a
role's launch statics are held to (b) across the launch shapes the registration allows (`templates`).

`check_compiled` refuses (`BatchDependent`), deterministically:

- R3: a derived split count. At top_p = 1 the keep set is every lane for any S, so the compiler emits the constant form
  ("splits": SplitsFor_v1(1, num_SMs)); at top_p < 1 serving must pin S first.
- R2: a request whose unit (LP, T, policy, seed, id), rows or discarded rows differ when its advice is compiled alone (its rows
  renumbered, its own release buckets): planning reads no batch-mate, whatever path a later change adds.
- R1: GP-01 wiring: a RowRef or a component naming another request's Program, or a request's outputs whose cone (`check.batch.
  separable` over `workload_labels`) reads another request's Inputs or a batch-provenance one (a `prefix` behind no checked
  source included).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from functools import cache
from typing import Any, Callable, Mapping

from verity_vllm.check.batch import BATCH, SHARED, Fields, Template, own, separable

from .build import Unexpressible, _Specs, build_program, model_shape
from .common import _ptypes, kind
from .phi import Phi

#: the request-local Input groups a request's own certified body fixes; any other request-local group is the batch's
OWN_GROUPS = ("prompt", "seed")
#: the RTX PRO 6000 (sm_120) as `target_profile` reads it
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}


class BatchDependent(Unexpressible):
    """A compilation in which a request's outputs could depend on its batch-mates."""


# --- the templates ---------------------------------------------------------------------------------------------------------------


def v0_phi() -> Phi:
    """The frozen v0 registration: Llama-3.2-1B, bf16, bi-eager on the RTX PRO 6000."""
    import json

    from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
    cfg = json.loads((HF_CONFIG_DIR / "LLAMA32_1B.config.json").read_text())
    return Phi(model={"repo": "meta-llama/Llama-3.2-1B", "revision": "registered", "hf_config": cfg}, workload={"requests": []},
               target=dict(RTXPRO6000), execution="bi-eager")


def workload_labels(wl: Any) -> tuple[tuple[Any, ...], Fields]:
    """GP-01's parameters by holder: its shared Input batches `SHARED`; each request's own groups (`OWN_GROUPS`) held by its index;
    any other request-local group (`splits`, `prefix`, `schedule`, `peers_*`, or one unknown here) `BATCH`. Its return: each
    request's outputs held by its index."""
    labels: list[Any] = [SHARED] * len(wl.shared_by_request[0])
    for r, sh in zip(wl.requests, wl.shared_by_request):
        groups = r.component.groups
        if len(groups) != len(r.component.params):
            raise BatchDependent(f"request {r.index}: its component names no group for some Input (provenance unknown)")
        labels += [own(r.index) if g in OWN_GROUPS else BATCH for j, g in enumerate(groups) if j not in set(sh)]
    return tuple(labels), Fields(tuple(own(r.index) for r in wl.requests))


def _workload(phi: Phi, shapes: list[tuple[int, int]], policy: dict) -> Any:
    from verity_vllm.pipeline.global_program import request_id_of
    from verity_vllm.program import workload as W

    from .score import request_component
    canon: dict = {}
    pq = Phi.from_json({**phi.to_json(), "sampling": policy})
    comps = {s: request_component(build_program(pq, "request", LP=s[0], T=s[1]), canon) for s in dict.fromkeys(shapes)}
    reqs = [W.WorkloadRequest(request_id_of(i), i, LP, T + 1, i, comps[(LP, T)], T=T) for i, (LP, T) in enumerate(shapes)]
    return W.WorkloadProgram(reqs, manifest_digest=None, lag=int(phi.target_profile().lag()))


def _row_labels(fn: Any, *labels: Any) -> tuple[Any, ...]:
    """A per-row call's labels, ``labels`` for its leading operands and `SHARED` for the rest."""
    n = len(_ptypes(fn))
    return tuple(labels) + (SHARED,) * (n - len(labels))


def _grid(lo: int, hi: int, *marks: int) -> list[int]:
    return sorted({x for x in (lo, lo + 1, *marks, hi - 1, hi) if lo <= x <= hi})


def templates(phi: Phi | None = None, *, max_tokens: int = 2048, max_seqs: int = 64, max_model_len: int = 2048,
              composition: tuple[tuple[int, int], ...] = ((4, 2), (3, 1))) -> list[Template]:
    """Every template a request-major compilation of ``phi`` (default `v0_phi`) calls, each with its labels, the launch shapes
    the registration allows (launches of up to ``max_tokens`` rows, ``max_seqs`` sampled rows, ``max_model_len`` positions) and its
    declared verdicts; then GP-01 over ``composition``'s (LP, T) requests, greedy and under a derived split count."""
    phi = phi or v0_phi()
    tp, C = phi.target_profile(), model_shape(phi.model["hf_config"])
    if not tp.batch_invariant:
        raise Unexpressible(f"{phi.execution!r}: the request-major compiler serves batch-invariant targets only")
    specs = cache(lambda M: _Specs(C, tp, M=M))
    S1 = specs(1)
    rows = _grid(1, max_tokens, 2, 64, 128, 256)
    by_rows = tuple(((), m) for m in rows)
    out: list[Template] = []

    def per_row(name: str, role: str, labels: tuple[Any, ...], **kw: Any) -> None:
        fn = getattr(S1, role)
        out.append(Template(f"{name} {fn.id}", fn, _row_labels(fn, *labels), own("r"),
                            bind_at=lambda _req, M: getattr(specs(M), role), shapes=by_rows, **kw))

    per_row("embedding", "embedding", (own("r"),))
    per_row("rmsnorm", "norm0", (own("r"),))
    per_row("rmsnorm (residual)", "fused", (own("r"), own("r")))
    for role in ("qkv", "o", "gate_up", "down", "logits"):
        per_row(f"gemm {role}", role, (own("r"),))
    per_row("rope q", "rope_q", (own("r"),))
    per_row("rope k", "rope_k", (own("r"),))
    per_row("silu_mul", "silu", (own("r"),))

    BN = S1.TG.attention_geometry(tp, C.D, 1, C.NH, C.KVH)["BN"]
    Ps = _grid(1, max_model_len, BN, BN + 1, 2 * BN, 2 * BN + 1)
    LPs = _grid(1, min(max_tokens, max_model_len), BN, BN + 1)
    attn_shapes = [((P, 0, 1), sq) for P in Ps for sq in _grid(1, max_tokens, 2, BN + 1)]
    attn_shapes += [((i + 1, i, LP), sq) for LP in LPs for i in sorted({0, LP - 1}) for sq in _grid(LP, max_tokens)]
    fn = S1.attention(BN + 1, 0, 1, 1)
    out.append(Template(f"attention {fn.id.split('{')[0]} (q, own K/V history)", fn, (own("r"), own("r"), own("r")), own("r"),
                        bind_at=lambda req, sq: S1.attention(req[0], req[1], req[2], sq), shapes=tuple(attn_shapes),
                        expect=(True, False),
                        why="FA2's guard static MASKED_FROM reads the launch's causal flag (max_seqlen_q > 1) on cc 12.0: a decode "
                            "row past one key block binds MASKED_FROM=1 alone and 0 beside a prefill (liveness, `fa2-causal-guard`)"))

    V = C.V
    fn = kind("token_select", V=V)
    out.append(Template(f"token select (greedy) {fn.id}", fn, (own("r"),), own("r"), bind_at=lambda _req, n: fn,
                        shapes=tuple(((), n) for n in range(1, max_seqs + 1))))
    # (logits, top_p, seed, pos, temperature, S)
    fn = kind("token_select_stochastic", V=V, CONSTRUCTION=tp.effective_sampler_construction())
    out.append(Template(f"token select (top-p, derived S) {fn.id}", fn, (own("r"), SHARED, own("r"), SHARED, SHARED, BATCH), own("r"),
                        expect=(False, None), why="its S operand is SplitsFor_v1(live rows): the batch picks the nucleus at top_p < 1"))
    from verity_vllm.program.registry.topp_geometry import splits_for_v1
    at = lambda n: kind("token_select_stochastic_split", V=V, S=splits_for_v1(n, tp.num_sms))  # noqa: E731
    fn = at(1)
    sampled = tuple(((), n) for n in range(1, max_seqs + 1))
    out.append(Template(f"token select (top-p, S = SplitsFor_v1(live rows)) {fn.id.split('{')[0]}", fn,
                        (own("r"), SHARED, own("r"), SHARED, SHARED, SHARED), own("r"), bind_at=lambda _req, n: at(n), shapes=sampled,
                        expect=(True, False), why="binding S per step makes it a launch static: five Definitions over 1-64 sampled rows"))
    out.append(Template(f"token select (top-p, registered S) {fn.id}", fn, (own("r"), SHARED, own("r"), SHARED, SHARED, SHARED),
                        own("r"), bind_at=lambda _req, n: fn, shapes=sampled))

    greedy = {"kind": "greedy"}
    derived = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
    for name, policy, expect, why in (("greedy", greedy, (True, None), ""),
                                      ("top-p, derived S", derived, (False, None), "each request's `splits` Input is the batch's")):
        wl = _workload(phi, list(composition), policy)
        labels, ret = workload_labels(wl)
        out.append(Template(f"GP-01 over {len(composition)} requests ({name})", wl.fn, labels, ret, expect=expect, why=why))
    return out


# --- the compile-unit rule -------------------------------------------------------------------------------------------------------


def _solo(advice: Any, r: int) -> Any:
    """``advice`` with request ``r`` alone: its record, its rows renumbered to request 0 in their steps (steps without it dropped),
    and its own release buckets in order. A cache hit sourced from another request has no solo form."""
    I1 = sys.modules[type(advice).__module__]
    release = tuple(b for o, b in zip(I1.outputs(advice), advice.release) if o.request == r)
    steps = []
    for s, step in enumerate(advice.steps):
        for row in step.rows:
            if row.request != r:
                continue
            if any(src.request != r for src in row.sources):
                raise BatchDependent(f"request {r}: step {s}'s cache hit reads KV another request computed (v0 has no solo form for it)")
            steps.append(type(step)((replace(row, request=0, sources=tuple(replace(src, request=0) for src in row.sources)),)))
    return type(advice)((advice.requests[r],), tuple(steps), release)


def _unit(u: Any) -> tuple:
    return (u.LP, u.T, u.policy, u.seed, u.id)


def check_compiled(registered: Any, advice: Any, ingress: Mapping[str, Any], compiled: Any,
                   plan: Callable[..., Any]) -> None:
    """Refuse (`BatchDependent`) ``compiled`` (`steps.Compiled` of ``advice``, I1 v0's `Advice`, and ``ingress``) where a request's
    outputs could depend on its batch-mates: R3, R2 and R1 of the module docstring, cheapest first. ``plan`` is `steps.plan_steps`.
    Returns None when every request's Programs read only its own certified inputs and the registration."""
    units = compiled.requests
    for u in units:
        if u is not None and u.policy.get("kind") == "stochastic" and u.policy.get("splits") == "derived":
            raise BatchDependent(f"request {u.request}: its top-p sampler's split count is derived from the step's live rows (S = "
                                 f"SplitsFor_v1), so its batch picks its nucleus; at top_p = 1 compile the constant form (splits = "
                                 f"SplitsFor_v1(1, num_SMs)), at top_p < 1 serving must pin S first")
    for s, rows in enumerate(compiled.steps):
        for rr in rows:
            u = units[rr.request]
            if u is None or rr.digest != u.digest:
                raise BatchDependent(f"step {s}: a row of request {rr.request} names Program {rr.digest[:16]}…, not its own")
    for r, u in enumerate(units):
        if u is None:
            continue
        q = advice.requests[r]
        try:
            alone = plan(registered, _solo(advice, r), {q.id: ingress[q.id]})
        except BatchDependent:
            raise
        except Unexpressible as e:
            raise BatchDependent(f"request {r} compiles in its batch, and alone it's refused: {e}") from None
        a = alone.requests[0]
        if _unit(a) != _unit(u):
            raise BatchDependent(f"request {r}: compiled alone it's (LP, T, policy, seed, id) = {_unit(a)}, in its batch {_unit(u)}")
        mine = [(rr.start, rr.n, rr.samples) for rows in compiled.steps for rr in rows if rr.request == r]
        theirs = [(start, n, k) for rows in alone.steps for _, start, n, k in rows]
        if mine != theirs:
            raise BatchDependent(f"request {r}: its rows (start, n, samples) are {theirs} alone, {mine} in its batch")
        mine = [(start, n) for _, req, start, n in compiled.discarded if req == r]
        theirs = [(start, n) for _, _, start, n in alone.discarded]
        if mine != theirs:
            raise BatchDependent(f"request {r}: its discarded rows are {theirs} alone, {mine} in its batch")
    wl = compiled.workload
    for w in wl.requests:
        u = units[w.index]
        if u is None or w.component.program_digest != u.digest:
            raise BatchDependent(f"GP-01's request {w.index} calls Program {w.component.program_digest[:16]}…, not its own")
    labels, ret = workload_labels(wl)
    if crossings := separable(wl.fn, labels, ret):
        raise BatchDependent(f"GP-01: {crossings[0]}" + (f" (+{len(crossings) - 1} more)" if len(crossings) > 1 else ""))
