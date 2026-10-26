"""Batch invariance of the request-major compiler: the templates a registration's compilations call, for `circuit-check
--batch`, and the rule `steps.compile_steps` holds every compilation to (`check_compiled`).

A request's outputs may depend on its own certified inputs (its prompt, its seed) and on what the registration fixes (weights,
tables, constants), never on its batch-mates. The batch reaches a Program two ways an owner-labelled check alone wouldn't see:

- a request-owned Input whose value the batch picks: the seeded top-p sampler's split count in its derived form, S =
  SplitsFor_v1(live rows, num_SMs) ("splits": "derived"), which changes the top-p keep set at top_p < 1.  Serving pins S to the
  registered split count instead (`topp_geometry.registered_splits`, `engine.topp_pin`), a constant of the registration;
- a launch static: a kind bound from the launch's shape (RMSNorm's BLOCK from its rows off batch invariance, FA3's tile from
  max_seqlen_q, FA2's guard `MASKED_FROM` from the launch's causal flag on cc 12.0).

So Inputs are labelled by provenance (`OWN_GROUPS` own, the registration's shared, any other request-local group `BATCH`), and a
role's launch statics are held to (b) across the launch shapes the registration allows (`templates`).

`check_compiled` refuses (`BatchDependent`), deterministically:

- R3: a derived split count. The served engine runs every top-p launch at the registered split count, so the compiler emits the
  constant form ("splits": registered_splits(num_SMs) = SplitsFor_v1(1, num_SMs)) at any top_p.
- R2: a request whose unit (LP, T, policy, seed, id, C), rows or discarded rows differ when its advice is compiled alone (its rows
  renumbered, its own release buckets): planning reads no batch-mate, whatever path a later change adds. A prefix-cache hit on
  another request's KV is kept when its tokens are checked equal (through each run, the source's certified prompt equals the
  request's), since that KV is then a function of the request's own prompt, given (a) and (b) on the units that computed it.
  Alone, a stand-in computes it from the request's own tokens. Any other hit on another request's KV is refused.
- R1: GP-01 wiring: a RowRef or a component naming another request's Program, or a request's outputs whose cone (`check.batch.
  separable` over `workload_labels`) reads another request's Inputs or a batch-provenance one (a `prefix` behind no checked
  hit included). A tensor-parallel rank's `peers_*` are its request's own: this rests on the node binding each to the same
  request's partial on the other rank, which the cross-rank transfer must hold request by request.
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


class BatchDependent(Unexpressible):
    """A compilation in which a request's outputs could depend on its batch-mates."""


# --- the templates ---------------------------------------------------------------------------------------------------------------


def registered_phi(role: str, target: str, execution: str = "bi-eager") -> Phi:
    """A registration of ``role``'s model, its config as the engine profile ships it (`find_hf_config`), on the device class the
    row-id token ``target`` names (`config.TARGETS`, `config.TARGET_NUM_SMS`)."""
    import json

    from verity_vllm.config import TARGET_NUM_SMS, TARGETS
    from verity_vllm.engine.profiles.dense_generic import find_hf_config
    found = find_hf_config(role)
    if found is None or target not in TARGET_NUM_SMS:
        raise Unexpressible(f"{role!r} on {target!r}: no shipped config, or a device class whose SM count varies")
    cfg = json.loads(found[0].read_text())
    return Phi(model={"repo": (found[2] or {}).get("repo", role), "revision": "registered", "hf_config": cfg},
               workload={"requests": []}, target={"compute_capability": list(TARGETS[target]), "num_sms": TARGET_NUM_SMS[target]},
               execution=execution)


def workload_labels(wl: Any, checked: frozenset[int] = frozenset()) -> tuple[tuple[Any, ...], Fields]:
    """GP-01's parameters by holder: its shared Input batches `SHARED`; each request's own groups (`OWN_GROUPS`) held by its index,
    and its `prefix` too when its hit's tokens are checked equal (``checked``, R2); a tensor-parallel rank's `peers_<k>` held by its
    index too, since the node binds each to the same request's partial of collective k on the other rank (`tp_links.json`; GP-01's
    `request[<id>].peers_<k>`), a value of that request's own inputs on every rank; any other request-local group (`splits`, an
    unchecked `prefix`, `schedule`, or one unknown here) `BATCH`. Its return: each request's outputs held by its index."""
    labels: list[Any] = [SHARED] * len(wl.shared_by_request[0])
    for r, sh in zip(wl.requests, wl.shared_by_request):
        groups = r.component.groups
        if len(groups) != len(r.component.params):
            raise BatchDependent(f"request {r.index}: its component names no group for some Input (provenance unknown)")
        mine = OWN_GROUPS + ("prefix",) * (r.index in checked)
        labels += [own(r.index) if g in mine or g.startswith("peers_") else BATCH for j, g in enumerate(groups) if j not in set(sh)]
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


def templates(phi: Phi, *, max_tokens: int = 2048, max_seqs: int = 64, max_model_len: int = 2048,
              composition: tuple[tuple[int, int], ...] = ((4, 2), (3, 1))) -> list[Template]:
    """Every template a request-major compilation of ``phi`` calls, each with its labels, the launch shapes the registration
    allows (launches of up to ``max_tokens`` rows, ``max_seqs`` sampled rows, ``max_model_len`` positions) and its declared
    verdicts; then GP-01 over ``composition``'s (LP, T) requests, greedy, at the registered split count, and under a derived one."""
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
    from verity.ml.definitions.registry.topp_geometry import registered_splits, splits_for_v1
    at = lambda n: kind("token_select_stochastic_split", V=V, S=splits_for_v1(n, tp.num_sms))  # noqa: E731
    S_reg = registered_splits(tp.num_sms)
    fn = kind("token_select_stochastic_split", V=V, S=S_reg)
    sampled = tuple(((), n) for n in range(1, max_seqs + 1))
    out.append(Template(f"token select (top-p, S = SplitsFor_v1(live rows)) {fn.id.split('{')[0]}", at(1),
                        (own("r"), SHARED, own("r"), SHARED, SHARED, SHARED), own("r"), bind_at=lambda _req, n: at(n), shapes=sampled,
                        expect=(True, False), why="binding S per step makes it a launch static: five Definitions over 1-64 sampled rows"))
    out.append(Template(f"token select (top-p, registered S) {fn.id}", fn, (own("r"), SHARED, own("r"), SHARED, SHARED, SHARED),
                        own("r"), bind_at=lambda _req, n: fn, shapes=sampled))

    greedy = {"kind": "greedy"}
    topp = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True}
    for name, policy, expect, why in (("greedy", greedy, (True, None), ""),
                                      ("top-p, registered S", {**topp, "splits": S_reg}, (True, None), ""),
                                      ("top-p, derived S", {**topp, "splits": "derived"}, (False, None),
                                       "each request's `splits` Input is the batch's")):
        wl = _workload(phi, list(composition), policy)
        labels, ret = workload_labels(wl)
        out.append(Template(f"GP-01 over {len(composition)} requests ({name})", wl.fn, labels, ret, expect=expect, why=why))
    return out


# --- the compile-unit rule -------------------------------------------------------------------------------------------------------


def _prompt(body: Any) -> tuple[int, ...] | None:
    """A certified body's prompt as token ids; None for a body that carries none (ingress v0's bytes)."""
    p = body.get("prompt") if isinstance(body, Mapping) else None
    return tuple(p) if isinstance(p, (list, tuple)) and all(type(t) is int for t in p) else None


def _check_hit(advice: Any, ingress: Mapping[str, Any], s: int, row: Any) -> None:
    """Refuse ``row``'s cache hit unless each run on another request's KV holds the request's own tokens: through the run, the
    source's certified prompt equals the hitting request's (the KV at a position reads every token up to it)."""
    r, at = row.request, 0
    mine = _prompt(ingress.get(advice.requests[r].id))
    for src in row.sources:
        end = at + src.n
        if src.request != r:
            theirs = _prompt(ingress.get(advice.requests[src.request].id))
            if mine is None or theirs is None:
                raise BatchDependent(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV, and their tokens can't be "
                                     f"checked equal: this rule reads a certified prompt as token ids, and one of the two bodies has none")
            if end > min(len(mine), len(theirs)) or mine[:end] != theirs[:end]:
                raise BatchDependent(f"request {r}: step {s}'s cache hit reads request {src.request}'s KV at positions {at} … {end - 1}, "
                                     f"and their tokens aren't its own (the certified prompts differ there, or one ends first)")
        at = end


def _solo(advice: Any, ingress: Mapping[str, Any], r: int) -> tuple[Any, dict, int]:
    """``advice`` and ``ingress`` with request ``r`` alone, and r's index there: its record and body, each of its rows in a step of
    its own, its own release buckets in order, and the state a Commit window carries it in with (`pipeline.window_advice`). A hit
    on another request's KV (`_check_hit`) holds the KV of r's first C tokens, so alone a stand-in computes it: a greedy request of
    those C tokens with one output, admitted just before r in r's first step. Such a hit on a later row (a resume, or any row of a
    carried request) has no solo form."""
    I1 = sys.modules[type(advice).__module__]
    q = advice.requests[r]
    carried = getattr(advice, "carried", ())
    rows = [(s, row) for s, step in enumerate(advice.steps) for row in step.rows if row.request == r]
    for j, (s, row) in enumerate(rows):
        if any(src.request != r for src in row.sources):
            if j or r < len(carried):
                raise BatchDependent(f"request {r}: step {s} resumes it on KV another request computed (v0 has no solo form for it)")
            _check_hit(advice, ingress, s, row)
    k = int(any(src.request != r for src in rows[0][1].sources))
    release = tuple(b for o, b in zip(I1.outputs(advice), advice.release) if o.request == r)
    steps = [I1.StepAdvice((replace(row, request=k, sources=tuple(replace(src, request=k) for src in row.sources)),)) for _, row in rows]
    requests, bodies = (q,), {q.id: ingress[q.id]}
    if k:
        C, stand = rows[0][1].cache_hit, f"{q.id} (prefix)"
        requests, release = (I1.RequestAdvice(stand, C, 1, "length"), q), (0, *release)
        bodies[stand] = {"prompt": list(_prompt(ingress[q.id])[:C]), "max_tokens": 1, "temperature": 0}
        steps[0] = I1.StepAdvice((I1.Row(0, "prefill", 0, C), replace(rows[0][1], request=1, sources=(I1.Source(0, C),))))
    if r < len(carried):
        return I1.Advice(requests, tuple(steps), release, carried=(carried[r],)), bodies, k
    return I1.Advice(requests, tuple(steps), release), bodies, k


def _unit(u: Any) -> tuple:
    return (u.LP, u.T, u.policy, u.seed, u.id, u.C)


def check_compiled(registered: Any, advice: Any, ingress: Mapping[str, Any], compiled: Any,
                   plan: Callable[..., Any]) -> None:
    """Refuse (`BatchDependent`) ``compiled`` (`steps.Compiled` of ``advice``, a Commit window's `Advice`, and ``ingress``) where a request's
    outputs could depend on its batch-mates: R3, R2 and R1 of the module docstring, cheapest first. ``plan`` is `steps.plan_steps`.
    Returns None when every request's Programs read only its own certified inputs and the registration."""
    units = compiled.requests
    for u in units:
        if u is not None and u.policy.get("kind") == "stochastic" and u.policy.get("splits") == "derived":
            raise BatchDependent(f"request {u.request}: its top-p sampler's split count is derived from the step's live rows (S = "
                                 f"SplitsFor_v1), so its batch picks its nucleus; compile the registered split count the served engine "
                                 f"pins (splits = registered_splits(num_SMs) = SplitsFor_v1(1, num_SMs))")
    for s, rows in enumerate(compiled.steps):
        for rr in rows:
            u = units[rr.request]
            if u is None or rr.digest != u.digest:
                raise BatchDependent(f"step {s}: a row of request {rr.request} names Program {rr.digest[:16]}…, not its own")
    checked: set[int] = set()
    for r, u in enumerate(units):
        if u is None:
            continue
        try:
            solo, bodies, k = _solo(advice, ingress, r)
            alone = plan(registered, solo, bodies)
        except BatchDependent:
            raise
        except Unexpressible as e:
            raise BatchDependent(f"request {r} compiles in its batch, and alone it's refused: {e}") from None
        if k:
            checked.add(r)
        a = alone.requests[k]
        if _unit(a) != _unit(u):
            raise BatchDependent(f"request {r}: compiled alone it's (LP, T, policy, seed, id, C) = {_unit(a)}, in its batch {_unit(u)}")
        mine = [(rr.start, rr.n, rr.samples) for rows in compiled.steps for rr in rows if rr.request == r]
        theirs = [(start, n, m) for rows in alone.steps for req, start, n, m in rows if req == k]
        if mine != theirs:
            raise BatchDependent(f"request {r}: its rows (start, n, samples) are {theirs} alone, {mine} in its batch")
        mine = [(start, n) for _, req, start, n in compiled.discarded if req == r]
        theirs = [(start, n) for _, req, start, n in alone.discarded if req == k]
        if mine != theirs:
            raise BatchDependent(f"request {r}: its discarded rows are {theirs} alone, {mine} in its batch")
    wl = compiled.workload
    for w in wl.requests:
        u = units[w.index]
        if u is None or w.component.program_digest != u.digest:
            raise BatchDependent(f"GP-01's request {w.index} calls Program {w.component.program_digest[:16]}…, not its own")
    labels, ret = workload_labels(wl, frozenset(checked))
    if crossings := separable(wl.fn, labels, ret):
        raise BatchDependent(f"GP-01: {crossings[0]}" + (f" (+{len(crossings) - 1} more)" if len(crossings) > 1 else ""))
