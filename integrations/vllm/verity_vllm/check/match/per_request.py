"""The per-request checks G3 (subcircuits_x09), G4 (identity_map) and G5 (shared_state): one leg per request over read-only
shared objects, reconciled in request order."""
from __future__ import annotations

import collections
import dataclasses
import functools

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.check.match import instance_sequence as SEQ
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import projection as PJ
from verity_vllm.check.match.attribution import Attribution
from verity_vllm.check.match.match_run import MatchRun, NamedCheck
from verity_vllm.program.compact import n_max_producer as _n_max_producer, n_producers as _n_producers
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event

# A leg can run in a forked worker (`GMF.request_legs`), so each problem it returns names its check by role, never by the NamedCheck.
SUBCIRCUITS, IDENTITY_MAP, SHARED_STATE = "subcircuits_x09", "identity_map", "shared_state"


@dataclasses.dataclass(frozen=True)
class RequestFacts:
    """What the per-request checks established for the engine-step checks: the identity map, the population, the Projections and
    components, and the dependency-order and Merkle-matching facts of every request."""
    glob2loc: dict[int, tuple[str, int, int]]
    shared_loc: dict[int, dict[str, tuple[str, int, int]]]
    exec_pop: int
    unexercised: int
    proj: dict[str, PJ.Projection]
    comps: dict[str, SEQ.RootView]
    engine_step_of: dict[int, int]
    dep_viol: list[dict]
    perm_facts: dict[str, dict]


def perm_match(A: SEQ.RootView, P: SEQ.RootView, s: int, ha: dict, hb: dict) -> list:
    """G8's Merkle (order-insensitive) matching of one request step: fold instance -> component index with the same DAG hash, chosen
    in emission order inside a hash class; None where no component instance is left with that hash.  `ha` / `hb` must be
    dag_hashes(A) / dag_hashes(P) of ONE process (the fast path's ids are per process)."""
    ma = collections.defaultdict(list)
    for j, i in enumerate(A.steps[s]):
        ma[ha[i]].append(j)
    pi = []
    for i in P.steps[s]:
        js = ma.get(hb[i])
        pi.append(js.pop(0) if js else None)
    return pi


def perm_eligible(A: SEQ.RootView, P: SEQ.RootView, s) -> bool:
    return not (s is None or s >= len(P.steps) or s >= len(A.steps) or len(A.steps[s]) != len(P.steps[s]))


def check_requests(run: MatchRun, att: Attribution, param_of: dict) -> RequestFacts:
    """G3 + G4 + G5, and the G7/G8 material: per request."""
    res, B, within_step, fast, _loaded = run.res, run.fold, run.within_step, run.fast, run.loaded
    per_req, meta_of, weights_nodes, weights_nodes_error, shared_gi = (att.per_req, att.meta_of, att.weights_nodes, att.weights_nodes_error,
                                                                       att.shared_gi)
    c3 = NamedCheck(res, "G3", "subcircuits_x09: each request's projection canonical-equal to its component over the executed steps (eos-truncated labelled)")
    c4 = NamedCheck(res, "G4", "identity_map: (request, local step, local index) <-> global node is a bijection over the attributed fold")
    c5 = NamedCheck(res, "G5", "shared_state: one weights input for all requests (no overlapping fields); node operands same-request, producer step <= consumer step; "
                               "a warm request's cached rows (the engine's first scheduled position) and their producers are the Program's declared prefix-cache hit")
    glob2loc: dict[int, tuple[str, int, int]] = {}
    shared_loc: dict[int, dict[str, tuple[str, int, int]]] = {}   # weight-only instance -> {request: its local identity}
    loc_count = 0
    exec_pop = 0
    unexercised = 0
    A_cache: dict[str, SEQ.RootView] = _loaded.setdefault("components", {})  # decoded components shared with the alternate-criterion run
    proj: dict[str, PJ.Projection] = {}
    comps: dict[str, SEQ.RootView] = {}
    xstep_specs: collections.Counter = collections.Counter()
    weight_readers: set[str] = set()
    if getattr(B, "oracle_wmap_overlaps", None):
        c5.fail(f"weights field map has {len(B.oracle_wmap_overlaps)} overlapping ranges (ambiguous shared-value identity)")
    engine_step_of: dict[int, int] = {}  # global instance index -> engine step
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        if o is not None and o.get("step") is not None:
            engine_step_of[i] = int(o["step"])
    # Safe parallelization: the per-request work below is ONE function of the request's declaration, its
    # attributed instances (established above), and the shared READ-ONLY fold + component Programs: `leg(rid)` returns facts (the request
    # entry, the check problems in the order it met them, the local->component address map, population contributions, the request's
    # dependency-order facts for G8) and writes nothing shared.  `apply(out)` -- always in the parent, always in request order -- does the
    # global reconciliation: the identity map over EVERY request (two requests claiming one instance is named here), the population sums,
    # the problems appended to G3/G4/G5 in the same order the inline form appended them.  The checker of record runs leg+apply inline per
    # request.  MATCH_IMPL=fast runs the legs first (GMF.request_legs: forked workers over the parent's objects, copy-on-write -- never a
    # copy of the trace per process; a failed worker raises, never a silent fallback), then apply in request order: same function, same
    # inputs, same facts.  State-linked steps and cross-rank dependencies are NOT assumed independent: nothing here compares across
    # requests; every cross-request fact (shared inputs G5, interfaces G7, layout G8, population G7) stays in the single parent pass.
    # `pre` is keyed by POSITION in gp["requests"], not by request_id: a Program that lists one request twice (G1 names it) still gets one
    # entry per listed request, exactly as the inline loop produced -- the record's document is the same document.
    pre = _prepare_requests(run, att, param_of, proj, comps, A_cache)
    leg = functools.partial(_request_leg, run.steps, B, fast, run.ops, within_step, weights_nodes, engine_step_of, pre)

    checks_by_role = {SUBCIRCUITS: c3, IDENTITY_MAP: c4, SHARED_STATE: c5}
    dep_viol: list[dict] = []   # every request's dependency-order violations, in request order (G8 names the first)
    perm_facts: dict[str, dict] = {}   # rid -> {local step: G8's Merkle matching}, computed by the leg in ITS process

    def apply(out: dict) -> None:
        """Global reconciliation of one request's facts, in request order: problems to their checks in the order met, the identity map
        over all requests (an instance with two local identities is named), the population sums, the request entry on the record."""
        nonlocal loc_count, exec_pop, unexercised
        rid, ent = out["rid"], out["ent"]
        for role, msg in out["fails"]:
            checks_by_role[role].fail(msg)
        if out["weight_reader"]:
            weight_readers.add(rid)
        xstep_specs.update(out["xstep"])
        dep_viol.extend(out["dep_viol"])
        if out.get("perm"):
            perm_facts[rid] = out["perm"]
        if out["P_steps"] is None:
            res["requests"].append(ent)
            return
        idxs, local_addr, A_len = per_req[rid], out["local_addr"], out["A_steps_len"]
        # identity map: local (s, j) <-> global node
        for s, loc in enumerate(out["P_steps"]):
            if s < len(A_len) and A_len[s] != len(loc):
                c4.fail(f"{rid}: local step {s} has {len(loc)} executed instances but the component has {A_len[s]}")
            for pos, il in enumerate(loc):
                gi = idxs[il]
                j = local_addr[s][pos] if s < len(local_addr) and pos < len(local_addr[s]) else None
                if j is None:
                    if within_step == "dag":
                        c4.fail(f"{rid}: fold instance {gi} (local step {s}, emission {pos}) has no structurally identical component instance: no canonical address")
                    j = pos  # emission position: keeps the map total for reporting (the step is already a G3 failure)
                key = (rid, s, j)
                if gi in shared_gi:
                    # a weight-only instance has one local identity PER request whose leg holds it (never a bijection
                    # with the global node): kept apart from glob2loc, so G7's owner blocks and G8's layout skip it like an unowned node
                    shared_loc.setdefault(gi, {})[rid] = key
                    loc_count += 1
                    continue
                if gi in glob2loc:
                    c4.fail(f"global instance {gi} carries two local identities {glob2loc[gi]} and {key}")
                glob2loc[gi] = key
                loc_count += 1
        exec_pop += out["exec_pop"]
        unexercised += out["unexercised"]
        res["requests"].append(ent)

    order = list(range(len(pre)))
    if fast:
        n_workers = GMF.workers(sum(1 for k in order if pre[k][2] is not None))
        GMF.stats["request_workers"] = max(GMF.stats.get("request_workers", 0), n_workers)   # telemetry (impl block), never the verdict
        legs = GMF.request_legs(order, leg, n_workers, weight=lambda k: len(pre[k][2] or ()))
        for out in legs:
            apply(out)
    else:
        for k in order:
            apply(leg(k))
    n_distinct = len(set().union(*per_req.values())) if per_req else 0   # every attributed fold instance once (a shared one sits in several legs)
    if len(glob2loc) + len(shared_loc) != n_distinct:
        c4.fail(f"identity map covers {len(glob2loc) + len(shared_loc)} of {n_distinct} attributed instances (not total)")
    c4.info.update({"mapped": len(glob2loc), "attributed": sum(len(v) for v in per_req.values()), "distinct": n_distinct,
                    "weight_only_shared": len(shared_loc), "weight_only_local_identities": sum(len(v) for v in shared_loc.values())})
    c3.done()
    c4.done()
    missing_w = [rid for rid in per_req if rid not in weight_readers]
    if missing_w:
        c5.fail(f"requests that never read the shared weights input: {missing_w}")
    if weights_nodes_error:
        c5.fail(f"shared weights input nodes could not be identified from the fold's parameter layout: {weights_nodes_error}")
    c5.info.update({"weights_param": (min(weights_nodes) if weights_nodes else None),   # first weights Input node (== the node for bf16)
                    # the weights as a node range: one node (homogeneous bf16 Record) or one per leaf field (FP8)
                    "weights_input_nodes": {"n": len(weights_nodes), "first": min(weights_nodes), "last": max(weights_nodes)} if weights_nodes else None,
                    "weight_fields": len(getattr(B, "oracle_wmap", []) or []),
                    "cross_step_producer_classes": dict(xstep_specs.most_common(8))})
    c5.done()
    return RequestFacts(glob2loc=glob2loc, shared_loc=shared_loc, exec_pop=exec_pop, unexercised=unexercised, proj=proj, comps=comps,
                        engine_step_of=engine_step_of, dep_viol=dep_viol, perm_facts=perm_facts)


def _prepare_requests(run: MatchRun, att: Attribution, param_of: dict, proj: dict, comps: dict, A_cache: dict) -> list[tuple]:
    """Per position in gp["requests"]: (g, ent, idxs, P, A | None, cd), the Projection and component built in the parent (filling
    `proj`, `comps` and `A_cache`) and read by the leg."""
    gp, B, fast, ops, _loaded = run.gp, run.fold, run.fast, run.ops, run.loaded
    per_req, weights_nodes, _peer_index_all = att.per_req, att.weights_nodes, att.peer_index
    pre: list[tuple] = []   # position -> (g, ent, idxs, P, A | None, cd): Projections + components built in the parent, shared read-only
    for g in gp["requests"]:
        rid = g["request_id"]
        ent = {"request_id": rid, "workload_request_id": g.get("workload_request_id"), "index": g["index"], "LP": g["prompt_len"], "max_tokens": g["max_tokens"],
               "arrive_step_declared": g["arrive_step"]}
        idxs = per_req.get(rid)
        if not idxs:
            ent["error"] = "no attributed instances"
            pre.append((g, ent, None, None, None, None))
            continue
        pidx = param_of.get(rid)
        # the request's seed parameter (`seed_<i>`, fold._header) is its own input, not a cross-request read
        # (`Input64_v1` nodes in request order, compare.RootView.seed_index: the i-th seed node is request index i's)
        seed_index = dict(getattr(B, "seed_index", {}))
        my_seed = next((k for k, i in seed_index.items() if i == g["index"]), None)
        my_peers = next((k for k, i in _peer_index_all.items() if i == g["index"]), None)   # the request's own peers node
        if fast:   # one Projection per request per process: X-09's (same construction key) / the verdict-of-record run's for the alternate
            P = GMF.projection(_loaded, rid, B, f"oracle[{rid}]", idxs, pidx, weights_nodes, my_seed, peer_node=my_peers)
        else:
            P = ops.projection(B, f"oracle[{rid}]", idxs, tok_node=pidx, weights_node=weights_nodes, seed_node=my_seed, peer_node=my_peers)
        P._all_seed_nodes = set(seed_index)
        P.cached_rows = PX.cached_rows(run.header, run.steps, rid)
        proj[rid] = P
        cd = (g.get("component") or {}).get("dir")
        A = None
        if cd:
            A = A_cache.get(cd)
            if A is None:
                A = A_cache[cd] = ops.load_derived(cd, f"component[{rid}]")
            comps[rid] = A
        pre.append((g, ent, idxs, P, A, cd))
    return pre


def _request_leg(steps: dict, B: SEQ.RootView, fast: bool, ops: BD.Ops, within_step: str, weights_nodes: frozenset[int],
                 engine_step_of: dict[int, int], pre: list[tuple], k: int) -> dict:
    """The request's own facts: canonical operand sources (P.canon), chronology of its producers, comparison with its component
    (compare_steps; under `dag` the Merkle matching per executed step -> canonical addresses).  Reads shared objects, writes none."""
    g, ent, idxs, P, A, cd = pre[k]
    rid = g["request_id"]
    out = {"rid": rid, "ent": ent, "fails": [], "P_steps": None, "local_addr": None, "A_steps_len": None, "exec_pop": 0, "unexercised": 0,
           "weight_reader": False, "xstep": collections.Counter(), "dep_viol": [], "perm": {}}
    fails = out["fails"]
    if idxs is None:
        return out
    for i_local in range(len(P.inst)):
        P.canon(i_local)  # resolves every operand source once: fills P.xreq (cross-request reads) before the checks below
    n_ts = sum(1 for r in P.inst if _is_sampling_event(r["spec"]))   # any sampling-event family
    LP = g["prompt_len"]
    prefill_steps = sorted(s for s in steps if any(r == rid and pos < LP for r, pos in steps[s]["rows"]))
    chunks = max(1, len(prefill_steps))
    T_obs = n_ts - chunks
    ent.update({"n_instances": len(idxs), "local_steps": len(P.steps), "token_selects": n_ts, "prefill_chunks": chunks, "T_observed": T_obs})
    why, hit = PX.check_hits(g, P)
    if P.xreq:
        why.insert(0, f"{rid}: {sum(P.xreq.values())} cross-request operand reads {dict(P.xreq)}")
    fails.extend((SHARED_STATE, w) for w in why)
    if hit:
        ent["prefix"] = hit
    # G8's dependency-order facts of THIS request (every producer emitted before its consumer inside a request
    # step) -- the same scan G8 made over every request's projection, done here, AFTER the G5 count above:
    # every canon() call of the record counts cross-request sources again, and the record's count is the first pass's
    for s_, loc in enumerate(P.steps):
        for pos, i in enumerate(loc):
            for a in P.canon(i, False):
                for t in a:
                    if t[0] != "N":
                        continue
                    mx = _n_max_producer(P.steps, t)   # O(1) for a progression whose producers are monotone; all runs otherwise
                    if mx >= i:
                        for prod in _n_producers(P.steps, t):   # one record per violating RUN, as the record emits
                            if prod >= i:
                                out["dep_viol"].append({"request_id": rid, "local_step": s_, "emission": pos, "consumer": i, "producer": prod})
    # producer chronology: every node operand's producer is this request's instance in the same or an earlier engine step
    late = 0
    compact = getattr(B, "compact", False)
    if fast and not compact:   # run form: the same count with node -> producer step resolved once per fold
        late, _xs, _wr = GMF.producer_chronology(P, idxs, B, engine_step_of, weights_nodes)
        out["xstep"].update(_xs)
        if _wr:
            out["weight_reader"] = True
    else:
      for i_local, r in enumerate(P.inst):
        gi = idxs[i_local]
        cs = engine_step_of.get(gi)
        for a in r["args"]:
            for run in a:   # record: (space, idx, lo, hi), one per run; compact: (space, (idx0, lo, hi), (dI, dL, dH), count) = `count` runs
                if run[0] != "n":
                    continue
                if compact:
                    dI, n_runs = run[2][0], run[3]
                    idxs_ = (run[1][0],) if dI == 0 else range(run[1][0], run[1][0] + n_runs * dI, dI)
                    w = n_runs if dI == 0 else 1
                else:
                    idxs_, w = (run[1],), 1
                for idx in idxs_:
                    if idx in B.inputs:
                        if idx in weights_nodes:
                            out["weight_reader"] = True
                        continue
                    j = B.k2i.get(idx)
                    ps = engine_step_of.get(j) if j is not None else None
                    if ps is not None and cs is not None and ps > cs:
                        late += w
                    if ps is not None and cs is not None and ps < cs:
                        out["xstep"][r["spec"].split("{")[0]] += w
    if late:
        fails.append((SHARED_STATE, f"{rid}: {late} operands produced in a LATER engine step than their consumer"))
    return _compare_component(ops, within_step, fast, g, ent, P, A, cd, LP, out)


def addressed_instances(A: SEQ.RootView) -> int:
    """The component's instances as GP-01's address map counts them (`n_instances` = its root nodes): the compared sequence plus the
    single-request splits constants `RootView._bind_splits` removed from it (one `Const32[S]` per top-p select; a batched component reads S
    from an Input and has none)."""
    return len(A.inst) + int(getattr(A, "n_splits_consts_bound", 0))


def _compare_component(ops: BD.Ops, within_step: str, fast: bool, g: dict, ent: dict, P: PJ.Projection, A: SEQ.RootView | None,
                       cd: str | None, LP: int, out: dict) -> dict:
    """The request's projection against its component over the executed steps: the leg's G3/G4 facts and the canonical address of
    every executed fold instance."""
    rid, fails = g["request_id"], out["fails"]
    if not cd:
        fails.append((SUBCIRCUITS, f"{rid}: no component Program in the global Program for shape (LP {LP}, T_bound {g.get('T_bound')})"))
        return out
    ent["component"] = {"dir": cd, "digest": A.digest, "steps": len(A.steps), "instances": len(A.inst),
                        "LP": g["component"].get("LP"), "T": g["component"].get("T")}
    want_d = g["component"].get("program_digest")
    if want_d and want_d != A.digest:
        fails.append((IDENTITY_MAP, f"{rid}: the global Program's component digest {want_d[:16]} != the loaded component's {A.digest[:16]}"))
    ga = g.get("global_address")
    if ga:
        ent["global_address"] = ga
        n_addressed = addressed_instances(A)
        if ga.get("n_instances") is not None and int(ga["n_instances"]) != n_addressed:
            fails.append((IDENTITY_MAP, f"{rid}: address map declares {ga['n_instances']} instances for the request, the component has {n_addressed} "
                                f"({len(A.inst)} compared + {n_addressed - len(A.inst)} bound splits constants)"))
    n_exec = len(P.steps)
    if n_exec > len(A.steps):
        fails.append((SUBCIRCUITS, f"{rid}: executed {n_exec} local steps but the bounded component has {len(A.steps)} (output interface exceeded)"))
    cmp = ops.compare_steps(A, P, unfold_a=False)
    exec_steps = cmp["steps"][:n_exec]
    bad_seq = [s for s in exec_steps if not s["spec_seq_equal"] or (s.get("wiring") or {}).get("operands_different")]
    tail = cmp["steps"][n_exec:]
    ent["eos_truncated"] = bool(tail)
    ent["unexecuted_steps"] = len(tail)
    out["unexercised"] = sum(len(A.steps[s]) for s in range(n_exec, len(A.steps)))
    ent["sequence_equal_steps"] = n_exec - len(bad_seq)
    ent["operands"] = {"equal": sum((s.get("wiring") or {}).get("operands_equal", 0) for s in exec_steps),
                       "different": sum((s.get("wiring") or {}).get("operands_different", 0) for s in exec_steps)}
    # canonical address of every executed fold instance: (rid, s, j) with j the COMPONENT's index of the corresponding instance.
    #   sequence: j = emission position (X-09 sequence criterion: the fold's order IS the component's order, else G3 fails);
    #   dag:      j = the index of the structurally identical component instance (equal Merkle DAG hash: spec, form, out type,
    #             operands with node sources named by their producer's hash) -- the emission order is never an address.
    local_addr: list[list[int | None]] = []
    if within_step == "dag":
        ha, hb = ops.dag_hashes(A), ops.dag_hashes(P)
        bad_dag, dag_info = [], []
        for s in range(min(n_exec, len(A.steps))):
            by_hash: dict[str, list[int]] = collections.defaultdict(list)
            for j, ia in enumerate(A.steps[s]):
                by_hash[ha[ia]].append(j)
            classes_gt1 = sum(1 for v in by_hash.values() if len(v) > 1)
            addr, unmatched = [], []
            for il in P.steps[s]:
                js = by_hash.get(hb[il])
                if js:
                    addr.append(js.pop(0))
                else:
                    addr.append(None)
                    unmatched.append(il)
            leftover = sum(len(v) for v in by_hash.values())
            permuted = sum(1 for j, x in enumerate(addr) if x is not None and x != j)
            if unmatched or leftover:
                spec = lambda Q, i: Q.inst[i]["spec"].split("{")[0]  # noqa: E731
                bad_dag.append({"step": s, "n_a": len(A.steps[s]), "n_b": len(P.steps[s]), "unmatched_fold": len(unmatched), "unmatched_component": leftover,
                                "unmatched_fold_specs": dict(collections.Counter(spec(P, i) for i in unmatched).most_common(6)),
                                "unmatched_component_specs": dict(collections.Counter(spec(A, A.steps[s][j]) for v in by_hash.values() for j in v).most_common(6)),
                                "forward_refs_in_fold": sum(1 for i in unmatched for a in P.canon(i, False) for t in a
                                                            for prod in _n_producers(P.steps, t) if prod >= i)})
            dag_info.append({"step": s, "permuted": permuted, "duplicate_hash_classes": classes_gt1,
                             "duplicate_hash_class_sizes": dict(collections.Counter(len(v) for v in by_hash.values() if len(v) > 1))})
            local_addr.append(addr)
        for s in range(len(A.steps), n_exec):   # executed beyond the bounded component: no address (G3 already fails)
            local_addr.append([None] * len(P.steps[s]))
        bad = bad_dag
        ent["executed_steps_equal"] = not bad
        ent["n_divergent_steps"] = len(bad)
        ent["criterion"] = "dag (order-insensitive Merkle multiset per request step; X-09's chunked-prefill criterion) over executed steps" + \
                           (" of the bounded component (eos-truncated)" if tail else "")
        sizes: collections.Counter = collections.Counter()
        for d in dag_info:
            sizes.update({int(k): v for k, v in d["duplicate_hash_class_sizes"].items()})
        # hash classes of multiplicity > 1 (structurally identical instances inside one request step, e.g. the same op over two equal
        # KV blocks): the bijection is chosen in emission order inside the class -- reported with the class sizes
        ent["within_step"] = {"permuted_instances": sum(d["permuted"] for d in dag_info), "steps_with_permutation": sum(1 for d in dag_info if d["permuted"]),
                              "duplicate_hash_classes": sum(d["duplicate_hash_classes"] for d in dag_info),
                              "duplicate_hash_class_sizes": {str(k): v for k, v in sorted(sizes.items())},
                              "sequence_also_equal": not bad_seq}
        if bad:
            ent["first_divergence"] = bad[0]
            fails.append((SUBCIRCUITS, f"{rid}: {len(bad)} of {n_exec} executed steps not DAG-equal to the component (first: local step {bad[0]['step']}, "
                                f"{bad[0]['unmatched_fold']} fold / {bad[0]['unmatched_component']} component instances unmatched)"))
        if fast:
            # G8's per-step Merkle matching of every step G8 may name (its address differs from the emission position
            # anywhere), computed HERE with this process's hash ids -- the leg hands back the matching (facts), never hash dicts:
            # the fast path's dag ids are per process and a worker's must not meet the parent's or another worker's
            for s, addr in enumerate(local_addr):
                if any(x is None or x != pos for pos, x in enumerate(addr)) and perm_eligible(A, P, s):
                    out["perm"][s] = perm_match(A, P, s, ha, hb)
    else:
        bad = bad_seq
        ent["executed_steps_equal"] = not bad
        ent["n_divergent_steps"] = len(bad)
        ent["criterion"] = "sequence (X-09) over executed steps" + (" of the bounded component (eos-truncated)" if tail else "")
        if bad:
            ent["first_divergence"] = {"step": bad[0]["step"], "n_a": bad[0]["n_a"], "n_b": bad[0]["n_b"], "spec_seq_equal": bad[0]["spec_seq_equal"],
                                       "wiring_examples": (bad[0].get("wiring") or {}).get("examples", [])[:3]}
            fails.append((SUBCIRCUITS, f"{rid}: {len(bad)} of {n_exec} executed steps not canonical-equal (first: local step {bad[0]['step']})"))
            if fast and perm_eligible(A, P, bad[0]["step"]):   # G8 names the first divergence: its matching, computed here
                s0 = bad[0]["step"]
                out["perm"][s0] = perm_match(A, P, s0, ops.dag_hashes(A), ops.dag_hashes(P))
        local_addr = [list(range(len(loc))) for loc in P.steps]
    sv = cmp.get("splits_values")
    why = ["cross-request reads"] if P.xreq else []
    if sv and not sv["equal"]:   # a single-request component's own Const32[S]: neither criterion above sees its value
        ent["splits_values"] = sv
        why.append(f"the component's constant S differs from the fold's at {len(sv['mismatches'])} event(s) "
                   f"(first [event, component S, fold S] = {sv['mismatches'][0]})")
    if why:
        fails.append((SUBCIRCUITS, f"{rid}: " + "; ".join(why)))
    out.update({"P_steps": P.steps, "local_addr": local_addr, "A_steps_len": [len(s) for s in A.steps],
                "exec_pop": sum(len(A.steps[s]) for s in range(min(n_exec, len(A.steps))))})
    return out
