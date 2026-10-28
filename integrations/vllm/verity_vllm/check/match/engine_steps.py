"""The engine-step checks: G7 (vu_population: the global query's population and boundary probes) and G8 (within_step_order)."""
from __future__ import annotations

import collections
import re

from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.check.match.attribution import Attribution
from verity_vllm.check.match.match_run import MatchRun, NamedCheck
from verity_vllm.check.match.per_request import RequestFacts, perm_eligible, perm_match
from verity_vllm.program.sampling_event import GREEDY_SELECTS


def vu_population(run: MatchRun, att: Attribution, reqs: RequestFacts) -> dict[int, list[int]]:
    """G7: the population the global query exposes == the attributed fold; probes on both sides of every request boundary inside an
    engine step resolve to their owner.  Returns engine step -> its fold instances."""
    res, B, fast = run.res, run.fold, run.fast
    per_req, meta_of = att.per_req, att.meta_of
    glob2loc, engine_step_of, comps, exec_pop, unexercised = reqs.glob2loc, reqs.engine_step_of, reqs.comps, reqs.exec_pop, reqs.unexercised
    c = NamedCheck(res, "G7", "vu_population: global query population (components over executed steps) == attributed fold; boundary probes resolve")
    attributed = sum(len(v) for v in per_req.values())
    if exec_pop != attributed:
        c.fail(f"global query exposes {exec_pop} VUs over the executed steps but the fold holds {attributed} attributed instances")
    probes = failures = 0
    loc2glob = {v: k for k, v in glob2loc.items()}
    by_step: dict[int, list[int]] = collections.defaultdict(list)
    for gi, s in engine_step_of.items():
        by_step[s].append(gi)
    for s in sorted(by_step):
        gis = sorted(by_step[s])
        owners = []
        block_last: list[int] = []
        if fast:   # same blocks; the last attributed instance of each block kept while building them (O(1) per boundary below)
            owners, block_last = GMF.g7_owner_blocks(gis, glob2loc)
        else:
            for gi in gis:
                o = glob2loc.get(gi)
                if o is None:
                    continue
                if not owners or owners[-1][0] != o[0]:
                    owners.append((o[0], gi))
        active = {o for o, _ in owners}
        if len(active) < 2:
            continue
        # every boundary between consecutive owner blocks in emission order: probe the last VU before and the first after
        for bi, ((oa, ga), (ob, gb)) in enumerate(zip(owners, owners[1:])):
            before = block_last[bi] if fast else max(g for g in gis if g < gb and glob2loc.get(g, (None,))[0] == oa)
            for gi in (before, gb):
                probes += 1
                loc = glob2loc[gi]
                rid, ls, lj = loc
                back = loc2glob.get(loc)
                A = comps.get(rid)
                ok = back == gi and rid in per_req and A is not None and ls < len(A.steps) and lj < len(A.steps[ls]) \
                    and A.inst[A.steps[ls][lj]]["spec"] == B.inst[gi]["spec"] and (meta_of.get(B.inst[gi]["k"]) or {}).get("request") in (None, rid)
                if not ok:
                    failures += 1
                    if failures <= 10:
                        c.fail(f"probe engine step {s} node {B.inst[gi]['k']}: address {loc} does not resolve to its owner/spec")
    c.info.update({"population_query": exec_pop, "population_fold": attributed, "unexercised_declared": unexercised,
                   "boundary_probes": probes, "boundary_probe_failures": failures})
    c.done()
    return by_step


def within_step_order(run: MatchRun, att: Attribution, reqs: RequestFacts, by_step: dict[int, list[int]]) -> None:
    """G8: the order of one request's own instances inside an engine step is its component's; the interleaving of different
    requests' rows is the runtime's batch layout, recorded and never re-sorted."""
    res, steps, G, within_step, ops = run.res, run.steps, run.declared, run.within_step, run.ops
    per_req = att.per_req
    glob2loc, shared_loc, proj, comps = reqs.glob2loc, reqs.shared_loc, reqs.proj, reqs.comps
    engine_step_of, dep_viol, perm_facts = reqs.engine_step_of, reqs.dep_viol, reqs.perm_facts
    c = NamedCheck(res, "G8", "within_step_order: each request's own instances in its component order inside every engine step; cross-request interleaving = recorded runtime batch layout"
                  if within_step == "sequence" else
                  "within_step_order: emission order inside an engine step recorded as batch layout (canonical addresses are structural); "
                  "dependency order is G3's (a consumer before its producer is DAG-unequal)")
    layout = []
    nonmono_steps = {}
    for s in sorted(by_step):
        gis = sorted(by_step[s])
        seq = [glob2loc.get(gi) for gi in gis]
        reqs_in_rows = list(dict.fromkeys(r for r, _ in steps.get(s, {}).get("rows", [])))
        emission_reqs = list(dict.fromkeys(o[0] for o in seq if o))
        last_j: dict[tuple[str, int], int] = {}
        nonmono = 0
        for o in seq:
            if o is None:
                continue
            rid, ls, lj = o
            k = (rid, ls)
            if k in last_j and lj < last_j[k]:
                nonmono += 1
            last_j[k] = lj
        if nonmono:
            nonmono_steps[s] = nonmono
            if within_step == "sequence":
                c.fail(f"engine step {s}: {nonmono} instances emitted out of their request-local order")
        if len(reqs_in_rows) > 1:
            layout.append({"engine_step": s, "batch_rows_order": reqs_in_rows[:8], "emission_first_seen": emission_reqs[:8],
                           "phase": "mixed" if any(p < G[r]["prompt_len"] for r, p in steps[s]["rows"] if r in G) and
                           any(p >= G[r]["prompt_len"] for r, p in steps[s]["rows"] if r in G) else "uniform",
                           **({"emitted_out_of_canonical_order": nonmono} if nonmono else {})})
    _segmentation(c, res, proj, per_req, engine_step_of)
    # a request whose step is a permutation of its component's (same DAG multiset, different sequence) -- characterise it: under
    # `sequence` it is the G3 failure (first_divergence); under `dag` it is the recorded layout of every step with a permutation
    perms = []
    perm_targets = []
    # dependency order inside the fold's emission: every producer before its consumer.  Under `sequence` a violation is already a G3
    # sequence/wiring failure; under `dag` G8 must FAIL on it explicitly (every producer is emitted before its consumer -- the
    # recorded permutation is a layout only while the dependency order holds).
    # `dep_viol`: collected per request by `leg` (the same scan over every request's projection -- P.canon(i) of every instance of every
    # local step, a node source whose producer is emitted at or after its consumer -- in request order, right where the canonical
    # sources were resolved); G8 names it here
    if dep_viol:
        c.fail(f"{len(dep_viol)} consumer-before-producer emissions inside a request step (first: {dep_viol[0]}) -- not a layout permutation")
    c.info["dependency_order_violations"] = len(dep_viol)
    for ent in res["requests"]:
        rid = ent["request_id"]
        if rid not in proj or rid not in comps:
            continue
        if within_step == "sequence":
            if ent.get("executed_steps_equal") is False:
                perm_targets.append((ent, rid, (ent.get("first_divergence") or {}).get("step")))
        else:
            A_, P_ = comps[rid], proj[rid]
            for s_ in range(min(len(P_.steps), len(A_.steps))):
                if any((glob2loc.get(per_req[rid][il]) or shared_loc.get(per_req[rid][il], {}).get(rid) or (None, None, None))[2] != pos
                       for pos, il in enumerate(P_.steps[s_])):
                    perm_targets.append((ent, rid, s_))
    for ent, rid, s in perm_targets[:32]:
        A, P = comps[rid], proj[rid]
        if not perm_eligible(A, P, s):
            continue
        # Merkle (order-insensitive) matching of the step's instances: B instance -> A index with the same DAG hash -- the leg's
        # matching when it ran (computed where the hashes were), else this process's hashes on both sides
        pi = (perm_facts.get(rid) or {}).get(s)
        if pi is None:
            pi = perm_match(A, P, s, ops.dag_hashes(A), ops.dag_hashes(P))
        same_dag = all(x is not None for x in pi)
        moved = [(j, x) for j, x in enumerate(pi) if x != j]
        es = engine_step_of.get(per_req[rid][P.steps[s][0]])
        # which computations were displaced: the specs of the A instances that moved by more than the induced shift, with their
        # Attention T (= query position + 1) against the positions of the OTHER requests' rows in the same engine step
        offs = collections.Counter(x - j for j, x in moved if x is not None)
        big = [(j, x) for j, x in moved if x is not None and abs(x - j) > 2]
        displaced = collections.Counter(A.inst[A.steps[s][x]]["spec"] for _, x in big)
        t_re = re.compile(r"\{T=(\d+)")
        disp_pos = sorted({int(m.group(1)) - 1 for _, x in big for m in [t_re.search(A.inst[A.steps[s][x]]["spec"])] if m})
        others = sorted({p for r_, p in steps.get(es, {}).get("rows", []) if r_ != rid}) if es is not None else []
        if within_step == "sequence":
            statement = ("the request's own instances of this step are exactly its component's computations (Merkle-identical DAG multiset) "
                         "emitted in a different order: the displaced instances are moved to the head of each layer's attention list; the "
                         "request-local order is prescribed by the component Program, the emission order of a prefill co-scheduled with other "
                         "requests' rows is a property of the executed kernel/fold, not derivable from the declared workload -> refused, "
                         "not re-sorted (M-ARRIVAL-ORDER)" if same_dag else
                         "the request's instances of this step are NOT a permutation of the component's (different DAG): a structural difference")
        else:
            statement = ("emission order of the request's instances inside this engine step differs from its component's canonical layout "
                         "(fold encoding: AttentionScope groups the batch's rows by distinct T in first-occurrence order); the instances, their "
                         "wiring and dependency order are the component's (Merkle-identical), each is addressed by its canonical component index; "
                         "recorded as batch layout, not re-sorted" if same_dag else
                         "NOT a permutation of the component's step (different DAG): reported by G3")
        perms.append({"request_id": rid, "local_step": s, "engine_step": es, "n": len(pi), "moved": len(moved), "same_dag_multiset": same_dag,
                      "unmatched": sum(1 for x in pi if x is None), "offset_histogram": dict(offs.most_common(6)),
                      "displaced_specs": dict(displaced.most_common(6)), "displaced_query_positions": disp_pos[:16],
                      "coscheduled_other_positions": others[:16], "permutation_head": pi[:8], "statement": statement})
        if within_step == "sequence":
            c.fail(f"{rid}: local step {s} (engine step {es}) is a permutation of its component's step ({len(moved)}/{len(pi)} moved; displaced query "
                   f"positions {disp_pos[:6]} vs co-scheduled others' positions {others[:6]})" if same_dag
                   else f"{rid}: local step {s} differs structurally from its component ({sum(1 for x in pi if x is None)} unmatched)")
    c.info.update({"shared_steps": len(layout), "batch_layout": layout[:6] + ([{"...": len(layout) - 6}] if len(layout) > 6 else []),
                   "steps_with_emission_out_of_canonical_order": len(nonmono_steps),
                   "order_source": "request-local canonical order: the component Program; cross-request row interleaving and (under dag) the "
                                   "emission order: runtime batch layout / fold encoding (StepStart.rows), recorded not prescribed",
                   "permutations": perms})
    c.done()


def _segmentation(c: NamedCheck, res: dict, proj: dict, per_req: dict[str, list[int]], engine_step_of: dict[int, int]) -> None:
    """A request's local steps, segmented at its TokenSelects, against the engine steps its fold rows record."""
    # a request's local steps are segmented at ITS TokenSelects (compare.Prog._segment, the rc3 rule); the fold rows
    # carry the engine step of every instance.  The two must agree: a local step lies inside ONE engine step, and one engine step holds
    # at most one TokenSelect-terminated local step of the request (its sampler is the last of its instances there), in increasing
    # engine-step order.  A mis-emitted sampler is named here (it would otherwise re-segment the request and surface as unrelated counts).
    seg_bad = []
    for ent in res["requests"]:
        rid = ent["request_id"]
        if rid not in proj:
            continue
        P_, gl = proj[rid], per_req[rid]
        closed: dict[int, int] = {}      # engine step -> local step whose TokenSelect closed it
        prev_es = None
        for s_, loc in enumerate(P_.steps):
            ess = sorted({engine_step_of[gl[i]] for i in loc if gl[i] in engine_step_of})
            has_ts = bool(loc) and P_.inst[loc[-1]]["spec"].startswith(GREEDY_SELECTS)
            if len(ess) > 1:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess, "problem": "the local step spans engine steps"})
            elif ess and prev_es is not None and ess[0] < prev_es:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess, "problem": f"engine step order decreases (after {prev_es})"})
            elif ess and has_ts and ess[0] in closed:
                seg_bad.append({"request_id": rid, "local_step": s_, "engine_steps": ess,
                                "problem": f"second TokenSelect of the request in engine step {ess[0]} (local step {closed[ess[0]]} already ended there)"})
            if ess and has_ts:
                closed.setdefault(ess[-1], s_)
            if ess:
                prev_es = ess[-1]
    if seg_bad:
        c.fail(f"{len(seg_bad)} request-local step(s) whose TokenSelect segmentation disagrees with the fold rows' engine steps "
               f"(first: {seg_bad[0]}) -- a mis-emitted sampler, not a layout")
    c.info["segmentation"] = {"rule": "request-local steps = the request's instances segmented at its TokenSelects (compare.Prog._segment); "
                                      "cross-checked against the engine step the fold rows record for every instance",
                              "disagreements": seg_bad[:6]}
