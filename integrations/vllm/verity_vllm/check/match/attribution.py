"""Attribution of every fold instance to its request, and G2 (requests_observed): every declared request executed, nothing extra."""
from __future__ import annotations

import collections
import dataclasses

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.check.match.match_run import MatchRun, NamedCheck
from verity_vllm.program import launch_units as LU


@dataclasses.dataclass(frozen=True)
class Attribution:
    """Every fold instance's owner: `per_req` = request -> its instances (a weight-only instance sits in the leg of every request
    with rows in its engine step), the instances no request owns, and the fold's shared weights input nodes.  A shared launch's unit
    (meta `launch`, `e<step>.<j>`) is no request's: `launches` = its instances, `launch_members` = its members' (slot, pairs)."""
    weights_nodes: frozenset[int]
    weights_nodes_error: str | None
    meta_of: dict[int, dict]
    per_req: dict[str, list[int]]
    unattributed: list[int]
    shared_idx: list[tuple[int, int]]
    shared_gi: frozenset[int]
    peer_index: dict
    launches: dict[str, list[int]]
    launch_members: dict[str, dict[str, tuple[int, int]]]


def attribute_instances(run: MatchRun, hdr_reqs: list[dict]) -> Attribution:
    """Attribute every fold instance to its request: the fold's meta channel, else the rows of its engine step."""
    B, steps = run.fold, run.steps
    n_in = B.n_input_nodes
    # [R16 global, F-fp8-GM1 / M-0823] the shared weights = the Input nodes no request owns (one per leaf field of a mixed-width Record:
    # FP8 435; one node for a homogeneous bf16 Record) -- never `n_input_nodes - 1`.  [R15 stoch] the request-owned inputs are the prompts
    # (one per header request) + the seeds (compare.Prog.seed_index).  A layout that does not hold is a G2 failure by name, not a guess.
    _peer_index_all = dict(getattr(B, "peer_index", {}))   # [R17 D90] TP rank fold: peers node -> request index (one per request)
    weights_nodes_error = None
    try:
        weights_nodes = frozenset(BD.request_inputs(B, hdr_reqs)[3])
    except ValueError as e:
        weights_nodes, weights_nodes_error = frozenset(), str(e)   # G5 fails by name below; every weight read is then a cross-request read
    meta_of = {o["node"] + n_in: o for o in B.oracle_rows}
    per_req: dict[str, list[int]] = collections.OrderedDict()
    unattributed = []
    # [R19, F-r19-int-15] weight-only instances (every node operand a shared weight or an earlier weight-only instance: gemma2's RMSNorm
    # scale `1 + w`, 11235 on #57 B8 1024/128) carry no request's data.  They are SHARED -- owned by no request, read like a weight --
    # and belong to the leg of every request with rows in their engine step (BD.shared_legs: the request's own derive computes the node
    # inside its forward at that step).  The same rule X-09 (BD.decompose) applies, after the meta channel and before the row-count rule,
    # which at B>1 refused them ("rows=1 != tokens|logits") and at B=1 handed them to the one request by coincidence.
    shared_nodes: set[int] = set()            # node ids of the weight-only instances (an operand source `is_weight_only` accepts)
    shared_idx: list[tuple[int, int]] = []    # (instance index, engine step)
    _compact = getattr(B, "compact", False)
    shared_inputs = weights_nodes | SI.values_nodes(B)
    launches: dict[str, list[int]] = {}
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        if o and o.get("launch"):
            launches.setdefault(o["launch"], []).append(i)
            continue
        rid = o.get("request") if o else None
        if rid is None and o is not None:
            if BD.is_weight_only(r, shared_inputs, shared_nodes, _compact):
                shared_nodes.add(r["k"])
                shared_idx.append((i, int(o["step"])))
                continue
            st = steps.get(o.get("step"))
            if st is not None:
                rid, _how = BD.attribute(o, {"rows": st["rows"], "logits_rows": st["logits_rows"]})
        if rid is None:
            unattributed.append(i)
        else:
            per_req.setdefault(rid, []).append(i)
    shared_gi: frozenset[int] = frozenset(i for i, _ in shared_idx)
    if shared_idx:
        per_req = collections.OrderedDict(BD.shared_legs(per_req, shared_idx, steps))
    ids = [r["request_id"] for r in hdr_reqs]
    launch_members = {u: {ids[i]: (slot, pairs) for i, (_w, _off, slot, pairs) in m.items()} for u, m in getattr(B, "linked", {}).items()}
    return Attribution(weights_nodes=weights_nodes, weights_nodes_error=weights_nodes_error, meta_of=meta_of, per_req=per_req,
                       unattributed=unattributed, shared_idx=shared_idx, shared_gi=shared_gi, peer_index=_peer_index_all,
                       launches=launches, launch_members=launch_members)


def requests_observed(run: MatchRun, att: Attribution, g_ids: list[str]) -> dict[str, list[int]]:
    """G2: every declared request has rows and instances; none is left unattributed or undeclared.  Returns request -> the engine
    steps with its rows."""
    res, steps, G, B = run.res, run.steps, run.declared, run.fold
    per_req, unattributed, shared_idx, shared_gi = att.per_req, att.unattributed, att.shared_idx, att.shared_gi
    c = NamedCheck(res, "G2", "requests_observed: every declared request has rows + output events + instances; 0 unattributed; no undeclared request; "
                              "the fold's launch units == the Program's")
    present: dict[str, list[int]] = collections.defaultdict(list)
    for s in sorted(steps):
        for rid in {r for r, _ in steps[s]["rows"]}:
            present[rid].append(s)
    for rid in g_ids:
        if rid not in present:
            c.fail(f"{rid}: declared but never scheduled (no StepStart rows)")
        if rid not in per_req:
            c.fail(f"{rid}: declared but no fold instance is attributed to it")
    for rid in set(present) | set(per_req):
        if rid not in G:
            c.fail(f"{rid}: executed / in the fold but not declared")
    if unattributed:
        c.fail(f"{len(unattributed)} fold instances attributed to no request")
    _launch_units(c, run, att)
    c.info.update({"n_instances": len(B.inst), "attributed": sum(len(v) for v in per_req.values()), "unattributed": len(unattributed),
                   "engine_steps": len(steps), "launch_units": len(att.launches), "launch_instances": sum(map(len, att.launches.values())),
                   # [F-r19-int-15] weight-only instances: owned by no request, in the leg of every request with rows in their step
                   "weight_only_shared": len(shared_idx),
                   "weight_only_specs": dict(collections.Counter(B.inst[i]["spec"].split("{")[0] for i in shared_gi).most_common(6))})
    c.done()
    return present


def _launch_units(c: NamedCheck, run: MatchRun, att: Attribution) -> None:
    """Each launch unit of the fold is one of the workload Program's (`launch[e<e>.<j>]`, its address map's `launches`): the same
    members at the same slots and pairs, and its instances the unit's gates; and each of the Program's ran."""
    declared = {f"e{u['e']}.{u['j']}": u for u in run.gp.get("launches") or ()}
    for name in sorted(att.launches.keys() - declared.keys()):
        c.fail(f"fold launch unit {name}: the workload Program has no launch[{name}]")
    for name in sorted(declared.keys() - att.launches.keys()):
        c.fail(f"launch[{name}]: no fold instance runs the workload Program's launch unit")
    for name in sorted(att.launches.keys() & declared.keys()):
        u = declared[name]
        want = {m["request_id"]: (m["slot"], m["pairs"]) for m in u["members"]}
        got = att.launch_members.get(name, {})
        if got != want:
            c.fail(f"launch[{name}]: the fold's members (request: slot, pairs) {got} != the Program's {want}")
        specs, unit = collections.Counter(run.fold.inst[i]["spec"] for i in att.launches[name]), LU.unit_specs(u)
        if specs != unit:
            c.fail(f"launch[{name}]: the fold's instances are not the unit's gates: extra {dict((specs - unit).most_common(4))}, "
                   f"missing {dict((unit - specs).most_common(4))}")
