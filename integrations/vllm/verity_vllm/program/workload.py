"""GP-01 — the workload Program: ONE rooted Program over a declared multi-request workload.

The request-wrapper derivation (`cb_a.derive_step --wrapper request`, the X-09 unit) is used as a COMPONENT: each derived
request Program (its descriptor) becomes one `WorkloadRequest_v1{program_digest, program_fn}` specialization whose body is the
derived root body with the prescribed-input gates turned back into parameters. The workload Program is then

    Workload_v1{requests=(...), shared=<param index>}(weights, request[r0].p0, request[r1].p0, ...) -> Tuple(out_r0, out_r1, ...)

whose root lowers to: the shared prescribed inputs (weights) ONCE, every request's request-specific prescribed inputs, then ONE
`call` node per request (hierarchical: the request subcircuit is a Definition instance, not inlined; nothing is flattened, no
scheduler is written). The workload digest commits to every request's own Program digest (a static of its specialization).

Canonical address space (deterministic, documented in `address_map()`): the root circuit's canonical gate layout. Request-local
identity (request Program root gate / X-09 instance index) -> global identity by a fixed affine rule per request (see
`address_map`). Two requests of the same shape are two instances of the same specialization at two different addresses.

Chronology (`chronology()`): DECLARED arrival schedule in engine-step units + BOUNDED output interfaces (cap tokens per request);
programmed order/dependencies only — observed elapsed time is not part of the Program. The builder never reads execution
outputs: shapes come from the manifest (prompt length, cap), never from served lengths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.codec import _spec_id, decode_program, encode_program, program_digest
from verity.ml.circuits.defs import Body, CompositeDefinition, Node, SpecializedDefinition
from verity.ml.circuits.constants import REGISTRATION
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Refs, Strided, coll_concat, concat
from verity.ml.circuits.types import Tuple as TupleT, Type
from verity_vllm.program.sampling_event import is_sampling_event

SCHEMA = "verity/workload-program/v1"


def _is_input_node(n: Node) -> bool:
    return n.form == "batch" and is_input(n.fn)


def _remap(r: Refs, K: int, logs: dict[int, tuple[PartLog, PartLog]] | None = None) -> Refs:
    """Root-body refs -> request-body refs: node k < K (an Input batch) becomes parameter k; node k >= K becomes node k-K.

    `logs` (one per `K`) maps a `PartLog` a `Concat` was cut from to the log of its remapped parts, so a key/value sequence
    one row longer than the call before remaps one part."""
    if logs is not None and isinstance(r, Concat) and r.prefix is not None:
        log, j = r.prefix
        hit = logs.get(id(log))
        if hit is None or hit[0] is not log:
            hit = logs[id(log)] = (log, PartLog())
        sub = hit[1]
        for p in log.parts[len(sub):j]:
            sub.append(_remap(p, K))
        return sub.raw_cut(j, [_remap(p, K) for p in r.parts[j:]])
    if isinstance(r, Affine):
        if r.space != "n":
            raise ValueError(f"unexpected ref space {r.space!r} in a root body")
        return Affine("p", r.idx, r.base, r.count) if r.idx < K else Affine("n", r.idx - K, r.base, r.count)
    if isinstance(r, Strided):
        if r.space != "n":
            raise ValueError(f"unexpected ref space {r.space!r} in a root body")
        sp, idx = ("p", r.idx) if r.idx < K else ("n", r.idx - K)
        return Strided(sp, idx, r.base, r.inner, r.outer_stride, r.inner_stride, r.count)
    if isinstance(r, Concat):
        return Concat([_remap(p, K) for p in r.parts])
    if isinstance(r, Explicit):
        items = []
        for space, idx, leaf in r.items:
            if space != "n":
                raise ValueError(f"unexpected ref space {space!r} in a root body")
            items.append(("p", idx, leaf) if idx < K else ("n", idx - K, leaf))
        return Explicit(tuple(items))
    raise TypeError(type(r))


def _type_leaves(s: str) -> int:
    """Leaf count of a type from its repr (`Value<w>`, `Array<n,T>`, `(T, T)`, `{k: T, ...}`) — the form instances.json records."""
    pos = 0

    def parse() -> int:
        nonlocal pos
        if s.startswith("Value<", pos):
            pos = s.index(">", pos) + 1
            return 1
        if s.startswith("Array<", pos):
            pos += len("Array<")
            j = s.index(",", pos)
            n = int(s[pos:j])
            pos = j + 1
            k = parse()
            assert s[pos] == ">", s[pos:pos + 20]
            pos += 1
            return n * k
        if s[pos] == "(":
            pos += 1
            total = 0
            while s[pos] != ")":
                total += parse()
                if s.startswith(", ", pos):
                    pos += 2
            pos += 1
            return total
        if s[pos] == "{":
            pos += 1
            total = 0
            while s[pos] != "}":
                pos = s.index(": ", pos) + 2
                total += parse()
                if s.startswith(", ", pos):
                    pos += 2
            pos += 1
            return total
        raise ValueError(f"unparsable type at {pos}: {s[pos:pos + 40]!r}")

    n = parse()
    if pos != len(s):
        raise ValueError(f"trailing type text: {s[pos:pos + 40]!r}")
    return n


@dataclass
class RequestComponent:
    """One derived request Program turned into a callable specialization (the component)."""

    fn: SpecializedDefinition
    program_digest: str
    program_fn: str
    params: tuple[tuple[str, Type], ...]
    input_gates: int          # gates of the request Program's own root that are Input gates (all params)
    body_gates: int           # gates of the request body (= request root gates - input_gates)
    n_nodes: int              # X-09 instance count (body nodes)
    local_offsets_sha256: str  # sha256 of the request body's node offsets (the request-local instance -> gate rule)
    groups: tuple[str, ...] = ()  # original request-wrapper parameter each component parameter (Input batch) belongs to
    step_nodes: tuple[tuple[int, int], ...] = ()  # request-relative step s -> body node range [lo, hi) (segments ending at TokenSelect_v1, X-09's unit)
    step_gates: tuple[tuple[int, int], ...] = ()  # the same in component-local gates
    bindings: tuple[str, ...] = ()  # each component parameter's binding time (`verity.ml.circuits.constants`: registration or run)

    def to_json(self) -> dict:
        return {"specialization": _spec_id(self.fn), "program_digest": self.program_digest, "program_fn": self.program_fn,
                "n_params": len(self.params), "param_groups": {g: self.groups.count(g) for g in dict.fromkeys(self.groups)},
                "input_gates": self.input_gates, "body_gates": self.body_gates,
                "n_nodes": self.n_nodes, "local_offsets_sha256": self.local_offsets_sha256, "steps": len(self.step_nodes),
                "step_nodes": [list(x) for x in self.step_nodes], "step_gates": [list(x) for x in self.step_gates]}


_CANON: dict[str, tuple[Any, dict]] = {}  # specialization id -> (the canonical decoded object, its v0 encoding); the DEFAULT canon
# the canon is the set of definitions ONE compose shares; a process composing several workloads (the
# reviewer's fixture runner, a test module) must not compare a later compose against an earlier one's definitions -> callers pass their
# own `canon` dict per compose (global_program.build does); the module-level dict serves callers that pass none (single-compose processes).


def _canonicalize(dp, canon: dict[str, tuple[Any, dict]] | None = None) -> None:
    """Share definitions across components. Each descriptor decodes into FRESH objects, so a definition used by two request
    Programs (every layer's Gemm/RmsNorm/... specialization) would be two objects under one id: `encode_program` then proves
    them equal by re-encoding both for every pair (O(components²) encodes — 20+ min at B=8). Here each decoded definition is
    encoded ONCE (v0, self-contained), compared with the canonical one under its id (fail-closed: a different body under the
    same id is `IdentityConflict`, never first-wins), and every reference to it (node callees, function-valued statics) is
    re-pointed to the canonical object. Primitives come from the registry and are already shared."""
    from verity.ml.circuits.codec import _encode_definition
    from verity.ml.circuits.defs import IdentityConflict, PrimitiveDefinition

    if canon is None:
        canon = _CANON
    subst: dict[str, Any] = {}
    # pass 1: an encoding names callees by id, so it does not depend on object identity — compare first, re-point after
    for fid, fn in dp.defs.items():
        if isinstance(fn, PrimitiveDefinition):
            continue
        if fn is dp.root:
            # the request Program's ROOT is never shared: its body becomes the component's
            # (`WorkloadRequest{program_digest, program_fn}`, identity = the whole descriptor's digest) and nothing calls it. Its id is
            # derive's `Root[Derived[vLLMRequest[...]]{SRC=...}]` whose SRC static names class/mode/profile/INPUT SHAPES/config -- not T --
            # so two requests of one manifest with the same prompt length and different caps (LP11: T24/T36) carry ONE root id over two
            # bodies, and comparing roots under that id would refuse a valid workload (E_IDENTITY) although their digests differ. Callees stay
            # under the fail-closed rule below; the root's identity is the component digest.
            continue
        enc = _encode_definition(fn)
        c = canon.get(fid)
        if c is None:
            canon[fid] = (fn, enc)
        elif c[1] != enc:
            raise IdentityConflict(f"E_IDENTITY: request Programs disagree on the definition {fid!r}")
        elif c[0] is not fn:
            subst[fid] = c[0]
    if not subst:
        return
    # pass 2: every node callee / function-valued static of this decode now points at the canonical object
    for fn in list(dp.defs.values()) + [dp.root]:
        if not isinstance(fn, PrimitiveDefinition):
            _repoint(fn, subst)


def _repoint(fn, subst: dict[str, Any]) -> None:
    """Re-point `fn`'s node callees and function-valued statics at their canonical objects (`subst`, by `_spec_id`), in
    place. Each substitute must have the id it replaces (`_canonical`): a complete specialization's id is formatted once
    from its bindings, so a static swapped for another id would leave it stale."""
    if fn._body is None:
        return
    for n in fn.body.nodes:
        c = _canonical(n.fn, subst)
        if c is not n.fn:
            n.fn = c
    for k, v in list(fn.bindings.items()):
        fn.bindings[k] = _subst_static(v, subst)


def _canonical(v, subst):
    """`v`'s substitute in `subst`, or `v` itself; refused (IdentityConflict) before it is applied when its id is not `v`'s."""
    from verity.ml.circuits.defs import IdentityConflict
    key = _spec_id(v)
    c = subst.get(key, v)
    if c is not v and _spec_id(c) != key:
        raise IdentityConflict(f"E_IDENTITY: {key!r} would be re-pointed to a definition of another id, {_spec_id(c)!r}")
    return c


def _subst_static(v, subst):
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    if isinstance(v, SpecializedDefinition):
        return _canonical(v, subst)
    if isinstance(v, PrimitiveDefinition):
        return v
    if isinstance(v, tuple):
        return tuple(_subst_static(x, subst) for x in v)
    if isinstance(v, list):
        return [_subst_static(x, subst) for x in v]
    if isinstance(v, dict):
        return {k: _subst_static(x, subst) for k, x in v.items()}
    return v


_REQUEST_DEF = CompositeDefinition("WorkloadRequest", 1, ("program_digest", "program_fn"), lambda S: ((), None), lambda *_: None,
                                   doc="a derived request Program (descriptor) as a callable component: its root body with the Input "
                                       "gates as parameters; identity = the request Program's digest + program function",
                                   conformance="exact-model-tested", register=False)
_REQUEST_CACHE: dict[tuple, SpecializedDefinition] = {}


def request_component(desc: dict, registry, wrapper_params: list[tuple[str, str]] | None = None,
                      canon: dict[str, tuple[Any, dict]] | None = None, digest: str | None = None) -> RequestComponent:
    """Decode a request Program descriptor (derive_step `descriptor.json.gz`) into a callable specialization.

    The derived root is `[Input batch per prescribed leaf group..., body nodes...]` (program.py lowering): a scalar/array
    prescribed parameter is ONE Input batch, a record-typed one (the `weights` record) is one Input batch PER FIELD, in field
    order. Component parameter j is the j-th Input batch (type `Array<n_j, Value<w_j>>`). `wrapper_params` = the request
    wrapper's own parameters `[(name, type repr)]` (derive_step `instances.json.gz` `params`): the Input batches are grouped
    back onto them by leaf count, so `groups[j]` names the wrapper parameter batch j belongs to (the shared one is `weights`).
    Without it every batch is its own group (`p<j>`) and nothing can be shared.  `digest`: `program_digest(desc)`, when the caller has it."""
    digest = digest or program_digest(desc)
    key = (("program_digest", digest), ("program_fn", desc["program_fn"]))
    cached = _REQUEST_CACHE.get(key)
    dp = decode_program(desc, registry)
    _canonicalize(dp, canon)   # `canon`: this compose's shared-definition set (None -> the process default)
    body = dp.root.body
    K = 0
    while K < len(body.nodes) and _is_input_node(body.nodes[K]):
        K += 1
    if any(_is_input_node(n) for n in body.nodes[K:]):
        raise ValueError("Input gates after the first body node: not a derive_step root")
    groups: list[str] = []
    if wrapper_params:
        j = 0
        for name, trepr in wrapper_params:
            want = _type_leaves(trepr)
            got = 0
            while got < want:
                if j >= K:
                    raise ValueError(f"parameter {name!r} needs {want} prescribed leaves but the root has only {body.offsets[K]}")
                got += body.nodes[j].out.leaves
                groups.append(name)
                j += 1
            if got != want:
                raise ValueError(f"Input batches do not tile parameter {name!r}: {got} leaves vs {want}")
        if j != K:
            raise ValueError(f"{K - j} Input batches beyond the declared parameters")
    else:
        groups = [f"p{j}" for j in range(K)]
    counts: dict[str, int] = {}
    names = []
    for g in groups:
        names.append(g if groups.count(g) == 1 else f"{g}[{counts.get(g, 0)}]")
        counts[g] = counts.get(g, 0) + 1
    params = tuple((names[j], body.nodes[j].out) for j in range(K))
    logs: dict = {}
    nodes = [Node(n.form, n.fn, tuple(Coll(a.type, _remap(a.refs, K, logs)) for a in n.args), n.out, axes=n.axes, n=n.n, n_xs=n.n_xs,
                  name=n.name) for n in body.nodes[K:]]
    offsets = [0]
    for n in nodes:
        offsets.append(offsets[-1] + n.gates)
    ret = _remap(body.ret, K)
    if cached is None:
        sp = SpecializedDefinition(_REQUEST_DEF, {"program_digest": digest, "program_fn": desc["program_fn"]}, key)
        sp.set_body(Body(params, body.ret_type, nodes, ret, offsets))
        _REQUEST_CACHE[key] = sp
    else:
        sp = cached
        if sp.body.gates != offsets[-1] or len(sp.body.nodes) != len(nodes):
            raise ValueError(f"E_IDENTITY: two different bodies for {sp.id}")
    input_gates = body.offsets[K]
    # request-relative steps = segments ending at a sampling-event instance (inclusive) — X-09's segmentation, on the Program side;
    # the sampling-event vocabulary is `verity_vllm.program.sampling_event`'s (greedy and seeded Gumbel selects alike), read nowhere else
    bounds, lo = [], 0
    for i, n in enumerate(nodes):
        if is_sampling_event(_spec_id(n.fn)):
            bounds.append((lo, i + 1))
            lo = i + 1
    if lo != len(nodes):
        bounds.append((lo, len(nodes)))  # trailing instances after the last TokenSelect (none for the request wrapper; kept, never dropped)
    step_gates = tuple((offsets[a], offsets[b]) for a, b in bounds)
    return RequestComponent(sp, digest, desc["program_fn"], params, input_gates, offsets[-1], len(nodes),
                            hashlib.sha256(json.dumps(offsets).encode()).hexdigest(), tuple(groups), tuple(bounds), step_gates,
                            tuple(body.nodes[j].fn.binding for j in range(K)))


@dataclass
class WorkloadRequest:
    """A declared request of the workload: manifest facts (never served lengths) + its component."""

    request_id: str
    index: int
    prompt_len: int
    cap: int              # max_tokens: the bounded output interface (cap tokens; the wrapper's T = cap - 1 decode steps)
    arrive_step: int      # declared arrival, engine-step units
    component: RequestComponent
    T: int | None = None  # decode steps derived (cap - 1); checked against the component's program_fn when it names T
    # a prefix-cache hit (`program.prefix_plan`): {"rows": C, "producers": [the request that computed each cached block]}; the
    # component reads rows 0..C-1 of every layer's KV cache as its `prefix` input, which Match binds to the producers' rows
    prefix: dict | None = None

    def static(self) -> dict:
        return {"request_id": self.request_id, "index": self.index, "prompt_len": self.prompt_len, "cap": self.cap,
                "arrive_step": self.arrive_step, "fn": self.component.fn, **({"prefix": self.prefix} if self.prefix else {})}


# request-LOCAL wrapper parameter groups: a component's own inputs, positions may differ per component (TP rank Programs carry
# `peers_<k>` = the received collective values, one per site x step, so K differs per (LP, T) shape); a stochastic
# wrapper's `seed` and `splits` are its request's own, like the prompt, and so are a warm request's cached `prefix` and a TP rank
# Program's `schedule` (its MoE align calls' keys, `CollectiveBus.schedule_key`)
_REQUEST_LOCAL_GROUPS = ("prompt", "seed", "splits", "prefix", "schedule")
_REQUEST_LOCAL_PREFIXES = ("peers",)


def _is_request_local(group: str) -> bool:
    return group in _REQUEST_LOCAL_GROUPS or any(group == p or group.startswith(p + "_") or group.startswith(p + "[") for p in _REQUEST_LOCAL_PREFIXES)


def _shared_positions(S) -> list[tuple[int, ...]]:
    """The shared group's local parameter positions per request: the static `shared` is ONE tuple when every component puts the
    group at the same positions (TP1: the historical form, digests unchanged) or one tuple per request when request-local inputs
    before it shift them (TP rank Programs); the m-th shared position of every request is the m-th shared Input batch."""
    sh = tuple(S.shared)
    if sh and isinstance(sh[0], (tuple, list)):
        return [tuple(int(j) for j in x) for x in sh]
    return [tuple(int(j) for j in sh)] * len(S.requests)


def _workload_signature(S):
    """params = the shared group's Input batches ONCE (`shared.<name>`, from request 0 — all requests agree by construction),
    then every request's non-shared batches (`request[<id>].<name>`), manifest order; ret = tuple of the requests' outputs."""
    reqs = S.requests
    pos = _shared_positions(S)
    p0 = reqs[0]["fn"].signature()[0]
    params = [(f"shared.{p0[j][0]}", p0[j][1]) for j in pos[0]]
    rets = []
    for r, sh in zip(reqs, pos):
        ps, ret = r["fn"].signature()
        shared = set(sh)
        for j, (pn, pt) in enumerate(ps):
            if j not in shared:
                params.append((f"request[{r['request_id']}].{pn}", pt))
        rets.append(ret)
    return tuple(params), TupleT(rets)


def _workload_body(B, S, *args):
    reqs = S.requests
    pos = _shared_positions(S)
    k = len(pos[0])
    outs = []
    for r, sh in zip(reqs, pos):
        ps, _ = r["fn"].signature()
        shared = {j: args[m] for m, j in enumerate(sh)}   # the m-th shared position of THIS request -> the m-th shared Input batch
        a = []
        for j in range(len(ps)):
            if j in shared:
                a.append(shared[j])
            else:
                a.append(args[k])
                k += 1
        outs.append(B.call(r["fn"], *a, name=f"request[{r['request_id']}]"))
    _, ret = _workload_signature(S)
    return coll_concat(ret, outs)


# `lag` = LAG, the rows the declared scheduler computes for a request AFTER its EOS sample (vLLM async scheduling:
# 1; synchronous: 0) -- a REQUIRED specialization parameter of the workload Program, coupled to the serving profile's resolved
# async-scheduling configuration (TargetProfile.lag()).  It does not change the wiring (the lag step is one of the request component's
# bounded cap steps; its VUs are required like every executed step's) but it changes what the Program DESIGNATES: under LAG = 1 the
# step after an EOS stop is active internal computation whose sampled token is not a served output (served_output_len = a_r,
# executed_prefix_len = a_r + LAG), inactivity starts after it.  Different value -> different Program id and digest.
LAG_VALUES = (0, 1)

WORKLOAD = CompositeDefinition("Workload", 1, ("requests", "shared", "lag"), _workload_signature, _workload_body,
                               doc="the declared workload: shared prescribed inputs once + one call per declared request "
                                   "(request Programs as components, manifest order); output = tuple of the requests' bounded "
                                   "output interfaces; `lag` = the declared scheduler's post-EOS rows (R17-1: served output length "
                                   "a_r, executed prefix a_r + lag on an EOS stop before the cap)", conformance="exact-model-tested", register=False)


def served_lengths_rule(lag: int) -> dict:
    """The two lengths the Program keeps distinct for every request, stated once (GM-01 G6 and the finalizer carry the same
    names): `served_output_len` = a_r, the tokens designated as output (the events up to and including the FIRST declared EOS, or all
    `cap` events on a length stop); `executed_prefix_len` = the steps that ran = a_r + LAG after an EOS stop before the cap (the LAG
    step(s) are active internal computation: their VUs are required, their sampled tokens are not output), a_r on a length stop (the
    scheduler knows the cap ahead: no post-EOS row; an EOS sampled AT the cap is a length stop).  Cap exhaustion is the separate
    condition a_r == cap.  Internal capacity: the request component is bounded at `cap` events and an EOS stop needs a_r <= cap - 1, so
    a_r + LAG <= cap for LAG <= 1 -- the declared lag step always has a programmed step to run in (`capacity_covers_lag`)."""
    return {"served_output_len": "a_r = output events up to and including the first declared EOS id (eos_token_ids), else cap",
            "executed_prefix_len": f"a_r + {int(lag)} after an EOS stop before the cap (LAG step(s): active internal computation, sampled token NOT served); "
                                   f"a_r on a length stop (a_r == cap; an EOS sampled at the cap is a length stop, no post-EOS row)",
            "cap_exhaustion": "a_r == cap (length stop): specified separately from the EOS stop; never widened by the lag",
            "eos_set": "stop_rule.eos_token_ids, NOT widened by the post-EOS row's discarded sample",
            "inactivity": f"positions >= a_r + {int(lag)} of an EOS-stopped request are programmed-but-unexecuted (bot in the lifted baseline)",
            "capacity_covers_lag": "cap events per request; a_r <= cap - 1 on an EOS stop, so a_r + LAG <= cap for LAG in {0, 1}",
            "rule": "R17-1 / DP-pad-02 (b); one definition for Build (this Program), GM-01 G6 and the finalizer"}


def eos_lag_rows_declared(engine: dict | None) -> int | None:
    """The post-EOS rows a declared engine block computes: `async_scheduling` True -> 1, False -> 0, undeclared or non-boolean ->
    None.  GP-01 always declares it (`workload_lag`), so None here means a Program from before the lag was declared: GM-01 G6 then
    fails by name on any EOS stop (nothing is read from the record as the rule) and labels a record without an EOS stop
    `lag.exercised false`."""
    if not isinstance(engine, dict):
        return None
    v = engine.get("async_scheduling")
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return 1 if v.strip().lower() == "true" else 0
    return None


class WorkloadProgram:
    def __init__(self, requests: list[WorkloadRequest], shared: str = "weights", manifest_digest: str | None = None,
                 ordering: str = "manifest", *, lag: int, theta_registered: bool = False) -> None:
        """`theta_registered`: bind the model's weights (`theta`) at registration, as `RegisteredInput<w>_v1`, whatever the
        components bind them as. The inference-only check reads only registered inputs as θ (nci-check's `theta-not-registered`). Off, the
        weights are bound as the components bind them, so every workload digest built before it stands."""
        if not requests:
            raise ValueError("a workload needs at least one request")
        # LAG is REQUIRED (a keyword with no default: omitting it is a TypeError, never a silent 0 or 1) and in {0, 1}
        if isinstance(lag, bool) or not isinstance(lag, int) or lag not in LAG_VALUES:
            raise ValueError(f"lag must be one of {LAG_VALUES} (the declared scheduler's post-EOS rows, TargetProfile.lag()); got {lag!r}")
        ids = [r.request_id for r in requests]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate request_id in the workload: {ids}")
        # the shared groups: `weights`, and every group a component registers (its `values`, its tables: what the program brings, bound
        # once at registration) -- the same Input batches, same count, names and types, in every request component
        reg = [frozenset(g for g, b in zip(r.component.groups, r.component.bindings) if b == REGISTRATION) for r in requests]
        if any(x != reg[0] for x in reg[1:]):
            raise ValueError(f"the request components register different groups {sorted({tuple(sorted(x)) for x in reg})}: not one program; refused")
        idx = [tuple(j for j, g in enumerate(r.component.groups) if g == shared or g in reg[0]) for r in requests]
        if not idx[0]:
            raise ValueError(f"no prescribed parameter group {shared!r} in the request components (groups: {sorted(set(requests[0].component.groups))})")
        sig = [tuple(r.component.params[j] for j in ix) for r, ix in zip(requests, idx)]
        if any(s != sig[0] for s in sig[1:]):
            raise ValueError(f"the shared group {shared!r} differs across requests (count/name/type): not one model / one weight set; refused")
        uniform = len(set(idx)) == 1
        if not uniform:
            # the shared group may sit at different positions per component ONLY when what precedes/follows it is
            # request-local by construction (prompt, seed, splits, schedule, peers_*: a TP rank Program's received collective values, K per shape); any other
            # group at a shifting position is still refused -- it would mean the components disagree on what the model's inputs are
            foreign = sorted({g for r in requests for g in r.component.groups if g != shared and not _is_request_local(g)})
            if foreign:
                raise ValueError(f"the shared group {shared!r} sits at different parameter positions across requests: {sorted(set(idx))} "
                                 f"and the components carry non-request-local groups {foreign}; refused")
        shared_idx = idx[0]
        for r in requests:
            if r.T is not None and r.T != r.cap - 1:
                raise ValueError(f"request {r.request_id}: T={r.T} is not cap-1={r.cap - 1}: the Program must be built from the declared "
                                 "cap, never from a served length")
        self.requests = list(requests)
        self.shared_group = shared
        self.theta_registered = bool(theta_registered)
        self.registered = tuple(f"shared.{requests[0].component.params[j][0]}" for j in shared_idx
                                if requests[0].component.groups[j] in reg[0]
                                or self.theta_registered and requests[0].component.groups[j] == shared)   # the root parameters bound at registration
        self.theta = tuple(f"shared.{requests[0].component.params[j][0]}" for j in shared_idx
                           if requests[0].component.groups[j] == shared)        # the model's weights (`Program(theta=)`)
        self.shared = shared_idx                       # request 0's positions = the shared Input batches' order
        self.shared_by_request = tuple(idx)            # per request (== shared_idx everywhere in the uniform/TP1 case)
        self.shared_positions_uniform = uniform
        self.manifest_digest = manifest_digest
        self.ordering = ordering
        self.lag = int(lag)
        # the static keeps the historical ONE-tuple form when uniform (every existing workload digest unchanged) and becomes one
        # tuple per request only when the positions differ (a distinct, self-describing encoding); `lag` is the third static:
        # every workload digest differs per declared value (and from the digest of every Program that declared no lag)
        self.fn = WORKLOAD.bind(requests=tuple(r.static() for r in self.requests), shared=shared_idx if uniform else tuple(idx), lag=self.lag)
        self.program = Program(self.fn, annotations={"schema": SCHEMA, "manifest_digest": manifest_digest}, registered=self.registered,
                               theta=self.theta)
        self.descriptor = encode_program(self.program)  # encoded ONCE (every component definition is re-encoded: the dominant cost)
        self.digest = program_digest(self.descriptor)
        # sanity: hierarchical — one call node per request, in manifest order, after the Input nodes
        body = self.program.root.body
        self._K = sum(1 for nd in body.nodes if _is_input_node(nd))
        calls = body.nodes[self._K:]
        assert len(calls) == len(self.requests) and all(nd.form == "call" for nd in calls), "workload root is not [inputs..., call per request]"
        for r, nd in zip(self.requests, calls):
            assert nd.fn is r.component.fn
            if nd.fn.gates != r.component.body_gates:
                raise AssertionError("component gate count changed under composition")

    # -- canonical address space --------------------------------------------------------------------
    def address_map(self) -> dict:
        """request-local identity -> global identity, deterministically.

        Global identity of a value = its gate index in the root circuit's canonical layout (`Circuit(root)`), and of a
        computation instance = its hierarchical path `root/node[k]/...`. For request r with root call node k_r at gate offset
        G_r:  request-local body gate g (the request Program's own root layout, g >= I_r = its input gates) -> G_r + (g - I_r);
        request-local X-09 instance i (row i of instances.jsonl / body node i) -> path (k_r, i), gate base G_r + off_r[i];
        request-local prescribed-input leaf (param j, leaf l) -> the global Input batch of that parameter: the shared parameter
        maps every request onto ONE interval (weights once), request-specific parameters onto their own interval."""
        body = self.program.root.body
        params, _ = self.fn.signature()
        S = len(self.shared)
        inputs = []
        for j in range(self._K):
            owner = "shared" if j < S else params[j][0].split("].", 1)[0][len("request["):]
            inputs.append({"root_node": j, "param": params[j][0], "type": str(params[j][1]), "gates": [body.offsets[j], body.offsets[j + 1]], "owner": owner})
        shared_gates = [body.offsets[0], body.offsets[S]]
        reqs = []
        pi = S
        for r_i, r in enumerate(self.requests):
            k = self._K + r_i
            G = body.offsets[k]
            c = r.component
            local_inputs = []
            sh_r = self.shared_by_request[r_i]
            for j, (pn, pt) in enumerate(c.params):
                if j in sh_r:
                    s = sh_r.index(j)
                    local_inputs.append({"local_param": j, "name": pn, "global": params[s][0], "root_node": s, "gates": [body.offsets[s], body.offsets[s + 1]]})
                else:
                    local_inputs.append({"local_param": j, "name": pn, "global": params[pi][0], "root_node": pi, "gates": [body.offsets[pi], body.offsets[pi + 1]]})
                    pi += 1
            reqs.append({"request_id": r.request_id, "index": r.index, "root_node": k, "specialization": _spec_id(c.fn),
                         "program_digest": c.program_digest, "gates": [G, G + c.body_gates], "local_input_gates": c.input_gates,
                         "n_instances": c.n_nodes, "local_offsets_sha256": c.local_offsets_sha256,
                         "rule": {"body_gate": f"global = {G} + (local_root_gate - {c.input_gates})",
                                  "instance": f"path (root/node[{k}], i); gate base = {G} + local_offsets[i]",
                                  "shared_inputs": f"local params {list(sh_r)} ({self.shared_group}) -> root nodes 0..{S - 1}, gates {shared_gates}"},
                         "inputs": local_inputs})
        return {"schema": SCHEMA + "/address-map", "workload_digest": self.digest, "gates": body.gates, "input_gates": body.offsets[self._K],
                "shared": {"group": self.shared_group, "root_nodes": [0, S], "gates": shared_gates, "n_batches": S,
                           **({} if self.shared_positions_uniform else {"component_params_by_request": [list(x) for x in self.shared_by_request]})},
                "convention": "gate index in the canonical layout of the workload root (verity.ml.circuits.layout.Circuit); hierarchical path "
                              "root/node[k]/node[i]... for instances; Input gates first in root-parameter order (the shared group once, "
                              "then each request's own), then one call node per request in manifest order",
                "inputs": inputs, "requests": reqs}

    # -- chronology: declared, programmed, bounded --------------------------------------------------
    def chronology(self) -> dict:
        """What the Program says about time. Unit: engine step (integer, one scheduler iteration of the engine). Declared:
        arrival step per request (manifest `arrive_step`). Programmed: request r's step s (0 = its prefill, s = 1..T its
        decode steps, T = cap - 1) is programmed for engine step `arrive_step + s` under the stated convention; the only
        dependencies are inside the request (its own KV state / previous token) — no cross-request data dependency is
        programmed. Bounded output interface: request r emits at most `cap` tokens, one per programmed step. Observed elapsed
        time / observed step indices are NOT here: they are the execution's, and the checker compares them to this."""
        rows = []
        for r in self.requests:
            T = r.cap - 1
            rows.append({"request_id": r.request_id, "index": r.index, "arrive_step": r.arrive_step, "prompt_len": r.prompt_len, "cap": r.cap,
                         "programmed_steps": {"prefill": r.arrive_step, "decode": [r.arrive_step + 1, r.arrive_step + T] if T > 0 else None,
                                              "count": T + 1},
                         "output_interface": {"kind": "bounded", "tokens_max": r.cap, "event": "(request_id, s) -> programmed engine step arrive_step + s"},
                         # the two lengths, bounded from the declaration alone: served <= cap; executed <= cap (the lag step fits)
                         "lengths": {"served_output_len_max": r.cap, "executed_prefix_len_max": r.cap,
                                     "eos_stop_executed_prefix_max": min(r.cap, max(r.cap - 1, 0) + self.lag), "lag": self.lag, "capacity_events": r.cap},
                         "input_interface": {"prompt_tokens": r.prompt_len, "available_at": r.arrive_step},
                         **({"prefix": {**r.prefix, "rule": "rows 0..rows-1 of every layer's KV cache are the producers' rows (served from the "
                                                            "prefix cache, never recomputed); the prefill computes rows..prompt_len-1"}}
                            if r.prefix else {})})
        last = max(r.arrive_step + r.cap - 1 for r in self.requests)
        # the programmed instance set of every engine step: the union over the requests programmed for it of their step-s
        # segment (component node range + global gate range) — well-defined from the declaration alone, no observation read
        body = self.program.root.body
        engine_steps = []
        for e in range(last + 1):
            members = []
            for r_i, r in enumerate(self.requests):
                s = e - r.arrive_step
                if 0 <= s <= r.cap - 1:
                    c = r.component
                    G = body.offsets[self._K + r_i]
                    seg = c.step_nodes[s] if s < len(c.step_nodes) else None
                    members.append({"request_id": r.request_id, "s": s, "phase": "prefill" if s == 0 else "decode",
                                    "nodes": list(seg) if seg else None,
                                    "gates": [G + c.step_gates[s][0], G + c.step_gates[s][1]] if seg else None})
            engine_steps.append({"e": e, "members": members, "co_scheduled": len(members) > 1,
                                 "mixed_prefill_decode": len({m["phase"] for m in members}) > 1})
        steps_ok = all(len(r.component.step_nodes) == r.cap for r in self.requests)
        return {"schema": SCHEMA + "/chronology", "time_unit": "engine step (integer scheduler iteration; not wall-clock)",
                "engine_steps": engine_steps,
                "engine_step_semantics": "members = the requests programmed for engine step e (a_r <= e <= a_r + cap_r - 1), each with its "
                                         "request-relative step s = e - a_r and that step's instance segment (component body nodes [lo,hi), "
                                         "segments end at TokenSelect_v1 — X-09's unit) and global gate range; the step's programmed instance "
                                         "set is the union; inside one engine step NO sequence is declared: the step's instances form a DAG "
                                         "multiset per (request, s) whose canonical addresses are structural (component node index = the "
                                         "address-map convention), never an order of events",
                "component_steps_equal_cap": steps_ok,
                # chronology = the engine-step counter on every event (a) + dependency order (b);
                # within one engine step (c) nothing is sequenced -- the address map's (arrive_step, manifest index) key is an ADDRESS, not
                # a declared order of the requests' computations, and the checker must not read either the record's or the Program's
                # within-step layout as an event sequence.
                "convention": "continuous scheduling: a request arriving at step a runs its prefill at step a and decode step s at step a+s, "
                              "no preemption, no chunked prefill; requests co-scheduled in one engine step are addressed by (arrive_step, "
                              "manifest index) — an addressing key only, not a declared order of their computations (none exists inside a "
                              "step); there is no programmed data dependency between requests",
                "within_step": {"declared_order": None,
                                "criterion": "DAG multiset per (request, s) with structural canonical addresses (rev-pm M-0099 (c)); the "
                                             "fold's emission sequence is an encoding of the record, recorded by the checker as layout"},
                "programmed_horizon_steps": last + 1, "requests": rows,
                "state_dependencies": "request-local: each request's KV cache and previous-token dependencies live inside its component "
                                      "(the request wrapper); the one cross-request state is a declared prefix-cache hit (`requests[].prefix`): "
                                      "the producers' KV rows, read as the warm request's `prefix` input",
                "observed_time": "not represented: observed step indices (fold StepStart) and elapsed time belong to the execution record; "
                                 "agreement with the programmed schedule is the global checker's (gmatch) check, not a builder claim",
                "future_dependent_events": {"eos_stop": "prescribed: the Program computes all cap steps of every request; an execution that "
                                                        "stops a request early at EOS leaves the tail programmed-but-unexecuted; the builder does "
                                                        "not rebuild from served lengths (rules §2.6)",
                                            # the declared scheduler's post-EOS rows are part of the executed prefix, not of the served output
                                            "post_eos_lag": f"LAG = {self.lag} declared: after an EOS stop before the cap exactly {self.lag} more programmed "
                                                            f"step(s) of the request RUN (active internal computation, fed the EOS; sampled token not served); "
                                                            f"inactivity starts at a_r + {self.lag}"},
                "lag": self.lag, "lengths": served_lengths_rule(self.lag)}

    def population(self) -> dict:
        """G-population, Program side: the global gate population = the shared/request prescribed inputs + the
        request subcircuits' populations, and nothing else; the component intervals partition the non-input gates exactly, every
        gate has one global address, and the only genuinely global computation is the (gate-free) output tuple wiring."""
        body = self.program.root.body
        K = self._K
        comp = [(body.offsets[K + i], body.offsets[K + i + 1]) for i in range(len(self.requests))]
        contiguous = all(comp[i][1] == comp[i + 1][0] for i in range(len(comp) - 1)) and (not comp or comp[0][0] == body.offsets[K]) \
            and (not comp or comp[-1][1] == body.gates)
        return {"root_nodes": len(body.nodes), "input_batches": K, "component_calls": len(self.requests), "other_root_nodes": len(body.nodes) - K - len(self.requests),
                "gates": body.gates, "input_gates": body.offsets[K], "component_gates": sum(b - a for a, b in comp),
                "sum_equals_total": body.offsets[K] + sum(b - a for a, b in comp) == body.gates, "components_partition_non_input_gates": contiguous,
                "component_intervals": comp, "shared_input_gates": [body.offsets[0], body.offsets[len(self.shared)]],
                "global_control_computation": "none with gates: the output interface is the tuple of the components' outputs (reference wiring only)",
                "unique_addresses": "every gate index in [0, gates) belongs to exactly one root node (canonical layout)"}

    def artifact(self) -> dict:
        params, ret = self.fn.signature()
        return {"schema": SCHEMA, "workload_digest": self.digest, "workload_fn": _spec_id(self.fn), "gates": self.program.gates, "population": self.population(),
                # the declared LAG static (also inside workload_fn / the digest) and the lengths rule it fixes
                "lag": self.lag, "lengths": served_lengths_rule(self.lag),
                "input_gates": self.program.input_gates, "n_root_nodes": len(self.program.root.body.nodes), "manifest_digest": self.manifest_digest,
                "shared": {"group": self.shared_group, "component_params": list(self.shared),
                           **({"registered": list(self.registered)} if self.registered else {}),
                           **({"theta_registered": True} if self.theta_registered else {}),
                           **({} if self.shared_positions_uniform else {"component_params_by_request": [list(x) for x in self.shared_by_request]})},
                "shared_param_index": list(self.shared),  # (compat: gmatch's from_gprog labels the shared parameter with it)
                "shared_param_index_by_request": None if self.shared_positions_uniform else [list(x) for x in self.shared_by_request],
                "params": [[n, str(t)] for n, t in params[:len(self.shared)]] + [[f"... {len(params) - len(self.shared)} request-specific batches", ""]],
                "n_params": len(params), "ret": str(ret),
                "requests": [{"request_id": r.request_id, "index": r.index, "prompt_len": r.prompt_len, "cap": r.cap, "arrive_step": r.arrive_step,
                              "component": r.component.to_json(), **({"prefix": r.prefix} if r.prefix else {})} for r in self.requests],
                "representation": "hierarchical: Program(Workload_v1{requests, shared}); one call node per request Program component; the "
                                  "request subcircuits are Definitions (not flattened); no scheduler; builder inputs = manifest + request descriptors",
                "padding_scope": "active-reference only: inactive padded interiors are outside this Program's claims; "
                                 "cross-request cache reuse is only the declared prefix-cache hits (`requests[].prefix`)",
                "x09_inside": "each component is the X-09 request-wrapper Program (its digest is a static of the component); X-09 per-request "
                              "equality is evidence inside the global check, not a substitute for it",
                "address_map": self.address_map(), "chronology": self.chronology()}
