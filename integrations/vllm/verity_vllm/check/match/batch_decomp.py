"""Batch decomposition (M-BATCH-DECOMP, X-09): Match at B>1 by projecting the batched eager fold onto its requests.

Under batch invariance a batched kernel launch (one GEMM with M = Σ tokens of the co-scheduled requests, one fused norm over
the step's token rows, ...) is the token-row-wise union of the per-request Calls the padded per-request Programs describe. The
fold already emits ONE Call per token row (`fold.py`, "Per-row emission"), so the batched execution IS a sequence of
per-token-row Calls; what Match lacked was the occurrence resolution of each Call to its request and a per-request comparison.

Occurrence resolution (glossary; the code still says 'attribution': the fold's own naming rule, `fold._name` /
`patterns.attribution_for`): a Call with `rows == num_tokens` of its engine step is token row `row` -> `StepStart.rows[row] =
(request, position)`; with `rows == len(logits_rows)` it is logits row `row` -> `StepStart.logits_rows[row]`. When both counts
coincide (a pure decode step) the two resolutions must
name the same request, else the Call is AMBIGUOUS. Any other row count is UNATTRIBUTABLE. Both refuse (fail-closed):
no Call is dropped, guessed or shared.

Projection: request r's Calls in fold order, segmented at r's own `TokenSelect` Calls into r's steps (prefill,
then one per decode). Operand sources are canonicalised exactly as `compare.RootView.canon` does, relative to the projection:
`(TOK, lo, hi)` for r's OWN prompt parameter, `(W, field, lo, hi)` for weights, `(N, step_r, index_r, lo, hi)` for a node
produced by one of r's Calls. A read of ANOTHER request's prompt parameter or of a node resolved to another request
is a cross-request dependency (`XREQ`): it is counted, reported with examples, and fails the sweep row — token-row independence is
exactly what BI guarantees, so such a read is either a non-row-separable op (must be listed explicitly) or a defect.

Comparison: each projection against the derived request-wrapper Program for the request's OWN shape `(LP_r, T_r)` —
`LP_r` = its prompt length (RunHeader), `T_r` = its TokenSelect count − 1 (decode steps actually served) — with
`compare.compare_steps` (spec sequence per step, 1:1 wiring, canonical sha256). One derivation per distinct `(LP, T)`,
reusable across requests and rows. The verdict is PASS iff every request is canonical-equal AND every Call is
resolved AND no XREQ AND every needed derivation is present. Missing derivations are listed (`missing_derived`) and
exit 4 so the driver (`row_pod.sh`) can derive them and rerun; refusal/mismatch exits 3.

    verity-vllm batch-decomp --oracle <match dir: program.json + instances.jsonl + capture/log.jsonl.gz>
        --derived <dir with descriptor.json.gz + derive-report.json + artifact.json> [--derived <dir> ...] --out decomp.json

PROTOCOL SUMMARY (Match stage, B>1: the per-request decomposition of the batched fold -- X-09; occurrence resolution at B>1)
  Purpose: resolve every Call of the batched eager fold to the request that produced it, project each request's Calls into
    its own steps, and compare that projection canonically against the Build's request-wrapper Program of the request's OWN shape.
  Inputs: --oracle <match dir> (program.json + instances.jsonl = the fold; capture/log.jsonl.gz = RunHeader / StepStart rows);
    --derived <build_request*> dirs (descriptor.json.gz, derive-report.json, artifact.json), one per distinct (LP, T).
  Outputs: --out match_decomp.json {verdict PASS|FAIL|MISSING_DERIVED, requests[{canonical_equal, LP, T, steps, totals}],
    attribution{unattributed}, xreq_total, missing_derived, within_step_criterion, reasons}.  Exit 0 PASS, 4 missing derived
    shapes (row_pod derives them and reruns once), 3 refusal / mismatch.
  Invariants: no Call is dropped, guessed or shared -- an AMBIGUOUS or UNATTRIBUTABLE token-row count refuses; a read of another
    request's prompt or node is XREQ and fails the sweep row; a request is compared at (LP_r, T_r) = (its prompt length, its served decode
    steps), never at the row envelope; the operand counts are the positional comparison, the verdict follows --within-step (dag of
    record; sequence recorded beside it).  Inside GM-01 (MATCH_PIPELINE=shared) the same decompose() writes this document via --decomp-out.
"""

from __future__ import annotations

import collections
import json
import os
import time
from dataclasses import dataclass

from verity_vllm import settings
from verity_vllm.check.match import compiler_form as CF
from verity_vllm.check.match import prefix_reads as PX
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.observe.capture_log import read_steps, request_params
from verity_vllm.check.match.dag_form import dag_hashes, live_cone
from verity_vllm.check.match.decomp_profile import (FORCE_DAG_ENV, PROFILE_ENV, _memo_stats, _install_profile_counters, _n_runs_of, _NoPhase,
                                                    _Phase, _profile_on)
from verity_vllm.check.match.derived_index import derived_shape
from verity_vllm.check.match.input_layout import request_inputs, weights_input_nodes  # noqa: F401  (weights_input_nodes re-exported: tests)
from verity_vllm.check.match.occurrence import SHARED, attribute, shared_legs
from verity_vllm.check.match.projection import Projection, engine_steps_of
from verity_vllm.check.program_ops import Ops
from verity_vllm.config import option
from verity_vllm.program import prefix_plan as PP
from verity_vllm.program.sampling_event import is_sampling_event as _is_sampling_event  #
# Imported from here by a test an open PR edits (tests/pipeline/test_global_program_regress.py); it moves to dag_form after that lands.
from verity_vllm.check.match.dag_form import dag_multiset  # noqa: F401


# the within-step criterion of record: `dag` accepts a request whose projection
# is the same live-cone DAG multiset as its Program although the fold's emission SEQUENCE differs (a co-scheduled prefill displaced
# by a running decode, M-ARRIVAL-ORDER); `sequence` is the r14-v1 criterion (dag accepted for a chunked prefill only). Both facts are
# recorded per request either way (`sequence_equal`, `dag_multiset_equal`); the global checker (GM-01) applies the per-step form and gates.
WITHIN_STEP_CRITERIA = ("sequence", "dag")
DEFAULT_WITHIN_STEP = "dag"


def record_ops() -> Ops:   # the checker of record's operations: X-09's default, GM-01's unless MATCH_IMPL=fast
    return Ops("record", C.load_oracle, C.load_derived, Projection, C.compare_steps, dag_hashes, live_cone)


def decompose(oracle_dir: str, derived_dirs: list[str], log_path: str | None = None, within_step: str = DEFAULT_WITHIN_STEP,
              oracle: "C.RootView | None" = None, components: "dict[str, C.RootView] | None" = None, ops: Ops | None = None) -> dict:
    if within_step not in WITHIN_STEP_CRITERIA:
        raise ValueError(f"within_step must be one of {WITHIN_STEP_CRITERIA}, got {within_step!r}")
    ops = ops or record_ops()
    t0 = time.time()
    profiling = _profile_on()
    force_dag = (settings.get(FORCE_DAG_ENV) or "").strip() == "1"
    run_prof: dict = {"schema": "cb-a/x09-profile/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir,
                      "force_dag": force_dag, "pid": os.getpid(), "phases": {}, "requests": []}
    if profiling:
        _install_profile_counters()
    log_path = log_path or os.path.join(oracle_dir, "capture", "log.jsonl.gz")
    header, steps = read_steps(log_path)
    reqs, param_of, parent_of = request_params(header)
    # the caller (global_match.check) may pass the fold it has already decoded: the fold RootView is read-only
    # after load_oracle and at B16 1024/128 it is ~40 GB of Python objects -- decoding it a second time in the same process doubled
    # GM-01's peak RSS (87 GB on cp-dev1) for nothing
    with _Phase(run_prof, "fold_load") if profiling else _NoPhase():
        B = oracle if oracle is not None else ops.load_oracle(oracle_dir, "oracle")
        if oracle is None:
            CF.fold_in_form(B, derived_dirs)
    if profiling:
        run_prof["fold_preloaded"] = oracle is not None
        run_prof["fold_instances"] = len(B.inst)
        run_prof["fold_compact"] = bool(getattr(B, "compact", False))
    n_in = B.n_input_nodes
    _attr_phase = _Phase(run_prof, "attribution") if profiling else _NoPhase()
    _attr_phase.__enter__()
    # the request-owned parameters after the prompts: a stochastic fold's `seed_<i>` per seeded request (every request of a stochastic
    # row is seeded, derive_step.sampler_geometry refuses otherwise), a TP rank fold's `peers_<i>` per request and a profile's witness
    # parameters; the shared weights are the Input nodes no request owns -- ONE node for a homogeneous bf16 Record, one per leaf field
    # for a mixed-width one (FP8: 435) -- never `n_input_nodes - 1`; refused by name when the layout does not hold
    seed_node_of, peer_node_of, witness_owners, weights_nodes = request_inputs(B, reqs)
    meta_of = {o["node"] + n_in: o for o in B.oracle_rows}

    # attribution of every instance, in node order (producers precede consumers). Channels, in order:
    #   meta     -- the fold recorded `request` on the instance (fold._compute_emit, X-09+);
    #   dataflow -- every request-owned operand the instance reads (its own prompt parameter / a node attributed earlier) names ONE request
    #               (row independence: an instance over one request's data is that request's; several requests -> non-separable, refused);
    #   weight-only -- every node operand is a shared weight or an earlier weight-only instance (and there is at least
    #               one): the instance carries no request's data -- gemma2's RMSNorm scale `1 + w` (AddScalarBf16 N=2304, rows=1, per norm
    #               per step; 11235 on #57 B8 1024/128) is the first such node in the table (llama/mistral/olmoe/qwen norms multiply by w
    #               directly).  It is SHARED: owned by no request, read like a weight, and projected into the leg of every request that has
    #               rows in its engine step -- the request's own derive computes the same node inside its forward.  Before this rule the
    #               row-count channel refused it ("rows=1 != tokens|logits") and every norm reading it counted as a cross-request read.
    #   rows     -- no request-owned operand at all: the fold's row-count rule (token rows / logits rows).
    # A per-position Attention group (rows=|positions sharing T|) is the case only dataflow gets right on pre-X-09 folds.
    shared_idx: list[tuple[int, int]] = []   # (instance index, engine step) of every weight-only instance (owner SHARED, module level)
    owners_of_param: dict[int, set[str]] = collections.defaultdict(set)
    # request-owned inputs: its prompt, seed, peers (a collective instance reading request r's peers node is r's), witnesses
    # and run values
    for rid, k in [*param_of.items(), *seed_node_of.items(), *peer_node_of.items(), *witness_owners, *SI.run_owners(B, reqs)]:
        owners_of_param[k].add(rid)
    shared_inputs = weights_nodes | SI.values_nodes(B)
    owner_of_node: dict[int, str] = {}
    per_req: dict[str, list[int]] = collections.OrderedDict()
    unattributed: list[dict] = []
    how = collections.Counter()
    disagree = 0
    compact = getattr(B, "compact", False)
    if compact:
        from verity_vllm.program.compact import idx_values as _idx_values
    for i, r in enumerate(B.inst):
        o = meta_of.get(r["k"])
        if o is None:
            unattributed.append({"i": i, "k": r["k"], "spec": r["spec"], "why": "no instances.jsonl row"})
            continue
        rid, h = o.get("request"), "meta"
        if rid is None:
            deps: set = set()
            reads_weight = False        # a shared weight node or an earlier weight-only instance among the operands
            for a in r["args"]:
                for run in a:   # record (space, idx, lo, hi) | compact (space, (idx0, lo, hi), (dI, dL, dH), count): space / idx only
                    if run[0] != "n":
                        continue
                    for idx in (_idx_values(run) if compact else (run[1],)):
                        if idx in B.inputs:
                            if idx not in shared_inputs:
                                deps |= owners_of_param.get(idx, {("?", "param", idx)})
                            else:
                                reads_weight = True
                        else:
                            own = owner_of_node.get(idx, ("?", "node", idx))
                            if own == SHARED:
                                reads_weight = True
                            else:
                                deps.add(own)
            unknown = {d for d in deps if isinstance(d, tuple)}
            owners = deps - unknown
            if unknown:
                h = "reads unattributed " + ", ".join(f"{d[1]} {d[2]}" for d in sorted(unknown))[:200]
            elif len(owners) == 1:
                rid, h = next(iter(owners)), "dataflow"
            elif len(owners) > 1:
                h = "reads several requests: " + ", ".join(sorted(owners))[:200]
            elif reads_weight:
                rid, h = SHARED, "weight-only"
            else:
                st = steps.get(o["step"])
                if st is None:
                    h = f"no StepStart for step {o['step']}"
                else:
                    rid, h = attribute(o, st)
                    h = "rows:" + h
        how[h.split(":")[0] if h.startswith("rows") else h.split(" ")[0]] += 1
        if rid is None:
            unattributed.append({"i": i, "k": r["k"], "spec": r["spec"], "step": o["step"], "row": o["row"], "rows": o["rows"], "why": h})
            continue
        if rid == SHARED:
            owner_of_node[r["k"]] = SHARED
            shared_idx.append((i, o["step"]))
            continue
        # cross-check against the row-count rule whenever it applies unambiguously (diagnostic; the dataflow channel is authoritative)
        st = steps.get(o["step"])
        if st is not None and h != "rows":
            rr, hh = attribute(o, st)
            if rr is not None and rr != rid and hh in ("token", "logits"):
                disagree += 1
        owner_of_node[r["k"]] = rid
        per_req.setdefault(rid, []).append(i)
    _attr_phase.__exit__(None, None, None)

    # derived index by (LP, T[, C])
    derived: dict[tuple[int, ...], str] = {}
    bad_derived = []
    for d in derived_dirs:
        sh = derived_shape(d)
        if sh is None:
            bad_derived.append(d)
        else:
            derived.setdefault(sh, d)
    A_cache: dict[tuple[int, ...], C.RootView] = {}
    components = components if components is not None else {}  # decoded component Progs keyed by dir, shared with the caller

    res = {"schema": "cb-a/batch-decomp/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": oracle_dir,
           "log": log_path, "n_requests_header": len(reqs), "n_output_sequences": len(per_req), "engine_steps": len(steps),
           "oracle": {"digest": B.digest, "instances": len(B.inst), "n_input_nodes": n_in, "fold_form": getattr(B, "fold_form", None),
                      # the shared weights as a node RANGE (contiguous after the request-owned parameters)
                      "weights_input_nodes": {"n": len(weights_nodes), "first": min(weights_nodes), "last": max(weights_nodes),
                                              "request_owned_inputs": len(reqs) + len(seed_node_of)},
                      "weight_map": {"n_ranges": len(B.oracle_wmap), "n_overlaps": len(B.oracle_wmap_overlaps)}},
           "attribution": {"how": dict(how), "unattributed": len(unattributed), "unattributed_examples": unattributed[:12],
                           "attributed": sum(len(v) for v in per_req.values()), "row_rule_disagreements": disagree,
                           # weight-only instances: owned by no request, projected into every participating leg
                           "weight_only_shared": len(shared_idx),
                           "weight_only_specs": dict(collections.Counter(B.inst[i]["spec"].split("{")[0] for i, _ in shared_idx).most_common(6))},
           "derived_available": {PP.shape_name(s): d for s, d in sorted(derived.items())}, "derived_unreadable": bad_derived,
           "requests": [], "missing_derived": []}
    missing: set[tuple[int, ...]] = set()
    all_equal = True

    def shape_of(rid: str, idxs: list[int]):
        """(LP, T[, C]) of a request from the fold alone (the leg recomputes the same; here to preload its component before the legs fork)."""
        pidx = param_of.get(rid)
        if pidx is None:
            return None
        LP = len(reqs[pidx]["prompt_token_ids"])
        n_ts = sum(1 for i in idxs if _is_sampling_event(B.inst[i]["spec"]))
        chunks = max(1, sum(1 for s, st in steps.items() if any(r == rid and pos < LP for r, pos in st.get("rows", []))))
        return PP.shape(LP, n_ts - chunks, PX.cached_rows(header, steps, rid))

    def leg(item) -> tuple:
        """X-09 for ONE request: its Projection of the fold, the comparison with its component (sequence + DAG criteria),
        its cross-request reads.  Reads the shared fold / components / attribution, writes nothing shared: the record's loop body,
        returning (entry, missing shape | None) instead of appending.  Runs in this process, or on a forked worker (fast path)."""
        rid, idxs = item
        pidx = param_of.get(rid)
        ent = {"request_id": rid, "parent": parent_of.get(rid), "prompt_param": pidx, "n_instances": len(idxs)}
        prof: dict = {"request_id": rid, "n_instances": len(idxs), "pid": os.getpid(), "t_start": time.time()} if profiling else None
        if prof is not None:
            ent["_profile"] = prof
        if pidx is None:
            ent["error"] = "request not in RunHeader.workload (no prompt parameter)"
            return ent, None
        LP = len(reqs[pidx]["prompt_token_ids"])
        with _Phase(prof, "projection_build") if prof is not None else _NoPhase():
            P = ops.projection(B, f"oracle[{rid}]", idxs, tok_node=pidx, weights_node=weights_nodes, seed_node=seed_node_of.get(parent_of.get(rid, rid)),
                               peer_node=peer_node_of.get(parent_of.get(rid, rid)))
            P._all_seed_nodes = set(seed_node_of.values())
            P.cached_rows = PX.cached_rows(header, steps, rid)
        with _Phase(prof, "shape") if prof is not None else _NoPhase():
            n_ts = sum(1 for r in P.inst if _is_sampling_event(r["spec"]))
            # prefill chunks: engine steps in which the request had PROMPT rows (position < LP). vLLM v1 samples (and discards) a token
            # after every partial chunk, so the fold holds one TokenSelect per chunk + one per served decode step: T = n_ts - chunks.
            prefill_steps = sorted(s for s, st in steps.items() if any(r == rid and pos < LP for r, pos in st.get("rows", [])))
            chunks = max(1, len(prefill_steps))
            T = n_ts - chunks
            es = engine_steps_of(P, meta_of)
            sh = PP.shape(LP, T, P.cached_rows)
        ent.update({"LP": LP, "T": T, **({"C": P.cached_rows} if P.cached_rows else {}), "token_selects": n_ts, "steps": len(P.steps), "engine_steps_per_step": es[:4] + (["..."] if len(es) > 4 else []),
                    "prefill_chunks": chunks, "prefill_engine_steps": prefill_steps[:8],
                    "chunked_steps": sum(1 for e in es if len(e) != 1), "engine_step_range": [min(min(e) for e in es if e), max(max(e) for e in es if e)] if es else None})
        if prof is not None:
            prof.update({"LP": LP, "T": T, "steps": len(P.steps), "prefill_chunks": chunks,
                         "p_runs_total": sum(_n_runs_of(a) for r in P.inst for a in r["args"]) if len(P.inst) <= 2_000_000 else None})
        if n_ts == 0:
            ent["error"] = "no TokenSelect attributed to the request"
            return ent, None
        d = derived.get(sh)
        if d is None:
            ent["derived"] = None
            return ent, sh
        with _Phase(prof, "component_load") if prof is not None else _NoPhase():
            A = A_cache.get(sh)
            if A is None:
                A = components.get(d)
                if A is None:
                    A = components[d] = ops.load_derived(d, f"derived[{PP.shape_name(sh)}]")
                A_cache[sh] = A
        if prof is not None:
            prof.update({"a_instances": len(A.inst), "a_steps": len(A.steps),
                         "a_runs_total": sum(_n_runs_of(a) for r in A.inst for a in r["args"]) if len(A.inst) <= 2_000_000 else None})
        with _Phase(prof, "compare_steps") if prof is not None else _NoPhase():
            cmp = ops.compare_steps(A, P, unfold_a=False)
        ent["derived"] = d
        ent["derived_digest"] = A.digest
        ent["xreq"] = {"counts": dict(P.xreq), "examples": P.xreq_examples}
        seq_equal = bool(cmp["canonical_sha256"]["equal"]) and not P.xreq
        ent["sequence_equal"] = seq_equal
        ent["canonical_sha256"] = cmp["canonical_sha256"]
        ent["criterion"] = "sequence"
        chunked = chunks > 1
        if prof is not None:
            prof.update({"sequence_equal": seq_equal, "xreq_total": sum(P.xreq.values()),
                         "steps_spec_equal": cmp["totals"].get("steps_spec_equal", 0), "cmp_steps": len(cmp["steps"])})
        if (not seq_equal and not P.xreq) or (force_dag and not P.xreq):
            # Order-insensitive DAG form over the LIVE cone (see dag_multiset / live_cone). It is the ACCEPTED criterion only for a
            # request whose prefill the engine chunked: the fold is then chunk-major and carries one discarded sample per partial
            # chunk, so the sequence criterion cannot hold by construction. Otherwise it is reported as a diagnostic only.
            with _Phase(prof, "live_cone_A") if prof is not None else _NoPhase():
                la = ops.live_cone(A)
            with _Phase(prof, "live_cone_P") if prof is not None else _NoPhase():
                lb = ops.live_cone(P)
            with _Phase(prof, "dag_hashes_A") if prof is not None else _NoPhase():
                ha = ops.dag_hashes(A, la)
            with _Phase(prof, "dag_hashes_P") if prof is not None else _NoPhase():
                hb = ops.dag_hashes(P, lb)
            with _Phase(prof, "multiset_diff") if prof is not None else _NoPhase():
                ma, mb = collections.Counter(ha.values()), collections.Counter(hb.values())
                only_a, only_b = ma - mb, mb - ma
                diff = only_a + only_b
                spec = lambda Q, i: Q.inst[i]["spec"].split("{")[0]  # noqa: E731
                ent["dag_multiset_equal"] = not diff
                ent["dag_multiset"] = {
                    "a_live": len(la), "a_dead": len(A.inst) - len(la), "b_live": len(lb), "b_dead": len(P.inst) - len(lb),
                    "a_dead_specs": dict(collections.Counter(spec(A, i) for i in range(len(A.inst)) if i not in la)),
                    "b_dead_specs": dict(collections.Counter(spec(P, i) for i in range(len(P.inst)) if i not in lb)),
                    "only_a": sum(only_a.values()), "only_b": sum(only_b.values()),
                    "only_a_specs": dict(collections.Counter(spec(A, i) for i, h in ha.items() if h in only_a).most_common(6)),
                    "only_b_specs": dict(collections.Counter(spec(P, i) for i, h in hb.items() if h in only_b).most_common(6))}
            if prof is not None:
                prof.update({"a_live": len(la), "b_live": len(lb), "dag_only_a": sum(only_a.values()), "dag_only_b": sum(only_b.values())})
            if not seq_equal and not diff and chunked:
                ent["criterion"] = "dag-multiset over the live cone (chunked prefill)"
            elif not seq_equal and not diff and within_step == "dag":
                ent["criterion"] = "dag-multiset over the live cone (within-step criterion of record; sequence differs)"
        ent["canonical_equal"] = (seq_equal or (ent.get("dag_multiset_equal") is True and (chunked or within_step == "dag"))) and (cmp.get("splits_values") or {}).get("equal", True)
        ent["totals"] = cmp["totals"]
        ent["a_steps"], ent["b_steps"] = cmp["a"]["steps"], cmp["b"]["steps"]
        bad = [s for s in cmp["steps"] if not s["spec_seq_equal"] or (s.get("wiring") or {}).get("operands_different")]
        ent["first_divergences"] = [{"step": s["step"], "n_a": s["n_a"], "n_b": s["n_b"], "div": s.get("first_divergences_T_normalised", [])[:2],
                                     "wiring_examples": (s.get("wiring") or {}).get("examples", [])[:3]} for s in bad[:3]]
        ent["n_divergent_steps"] = len(bad)
        if prof is not None:
            prof["seconds"] = round(time.time() - prof["t_start"], 3)
        return ent, None

    if shared_idx:
        # a weight-only instance of engine step s belongs to the projection of every request with rows in s (the
        # request's own derive computes that node inside its forward at that step); a request with no rows in s does not see it.
        # Legs stay in request order; each leg's indices stay in node order (producers precede consumers).  GM-01 builds the same legs.
        items = list(shared_legs(per_req, shared_idx, steps).items())
    else:
        items = list(per_req.items())
    if ops.legs is not None:
        # the per-request comparisons are separable once attribution is done (above, single and exact): the
        # fast path runs the legs on forked workers over the SAME fold / components (copy-on-write) -- every component is loaded here
        # first so no worker loads its own copy; the aggregation below is the record's loop, in request order
        to_load: list[tuple[str, str]] = []
        for rid, idxs in items:
            sh = shape_of(rid, idxs)
            d = derived.get(sh) if sh is not None else None
            if d is not None and d not in components and all(d != x[0] for x in to_load):
                to_load.append((d, f"derived[{PP.shape_name(sh)}]"))
        if to_load:
            # the components are built on the workers too, one per leg (2 s to build vs 0.1 s to pickle a component back: the build is
            # the descriptor walk), then held by the parent so the legs below share them copy-on-write; a build failure raises, as the
            # record's load in the loop does
            components.update(zip((d for d, _ in to_load), ops.legs(to_load, lambda it: ops.load_derived(*it), lambda it: 1)))
        with _Phase(run_prof, "legs") if profiling else _NoPhase():
            outs = ops.legs(items, leg, lambda it: len(it[1]))
    else:
        with _Phase(run_prof, "legs") if profiling else _NoPhase():
            outs = [leg(it) for it in items]
    for ent, miss in outs:
        if profiling:
            p = ent.pop("_profile", None)   # the sidecar's, never the document's
            if p is not None:
                run_prof["requests"].append(p)
        else:
            ent.pop("_profile", None)
        if miss is not None:
            missing.add(miss)
        elif ent.get("error"):
            all_equal = False
        elif ent.get("derived") is not None:
            all_equal = all_equal and ent["canonical_equal"]
        res["requests"].append(ent)
    res["within_step_criterion"] = within_step
    res["missing_derived"] = [list(s) for s in sorted(missing)]
    res["xreq_total"] = sum(sum(e.get("xreq", {}).get("counts", {}).values()) for e in res["requests"])
    complete = not unattributed and not missing and not bad_derived
    res["verdict"] = "PASS" if (complete and all_equal and per_req) else ("MISSING_DERIVED" if missing and not unattributed else "FAIL")
    res["reasons"] = ([f"{len(unattributed)} unattributed instances"] if unattributed else []) + \
                     ([f"missing derived Programs for {res['missing_derived']}"] if missing else []) + \
                     ([f"{len(bad_derived)} derived dir(s) unreadable: {[os.path.basename(x) for x in bad_derived[:3]]}"] if bad_derived else []) + \
                     ([f"{res['xreq_total']} cross-request operand reads"] if res["xreq_total"] else []) + \
                     [f"{e['request_id']}: not canonical-equal" for e in res["requests"] if e.get("canonical_equal") is False and not e.get("xreq", {}).get("counts")] + \
                     [f"{e['request_id']}: {e['error']}" for e in res["requests"] if e.get("error")]
    res["seconds"] = round(time.time() - t0, 1)
    if profiling:
        run_prof["seconds"] = res["seconds"]
        run_prof["verdict"] = res["verdict"]
        run_prof["x09_legs"] = "forked" if ops.legs is not None else "inline"
        st = _memo_stats()
        if st is not None:
            run_prof["fast_stats_final"] = {k: int(v) for k, v in sorted(st.items())}
        try:
            import resource
            run_prof["max_rss_mib_parent"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
            run_prof["max_rss_mib_children"] = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)
        except Exception as e:   # noqa: BLE001 - telemetry only
            run_prof["max_rss_error"] = repr(e)[:200]
        out = os.environ[PROFILE_ENV]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(run_prof, open(out, "w"), indent=1, default=str)
    return res


@dataclass(frozen=True, kw_only=True)
class Options:
    oracle: str = option("--oracle", help="match dir: program.json + instances.jsonl (+ capture/log.jsonl.gz)")
    derived: list[str] = option("--derived", action="append", default=[],
                                help="derived request-wrapper dir (repeatable; shape read from artifact.json)")
    log: str | None = option("--log", default=None, help="capture log (default <oracle>/capture/log.jsonl.gz)")
    out: str = option("--out")
    within_step: str = option("--within-step", choices=WITHIN_STEP_CRITERIA, default=DEFAULT_WITHIN_STEP,
                              help="within-step criterion of record: dag (live-cone DAG multiset accepted for every request) | sequence (r14-v1)")


def main(a: Options) -> int:
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level cyclic-GC thresholds; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    res = decompose(a.oracle, a.derived, a.log, within_step=a.within_step)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    brief = {"verdict": res["verdict"], "within_step": res["within_step_criterion"], "reasons": res["reasons"],
             "requests": [{k: e.get(k) for k in ("request_id", "LP", "T", "n_instances", "steps", "canonical_equal", "sequence_equal", "dag_multiset_equal", "criterion", "chunked_steps")}
                                                                                 for e in res["requests"]],
             "attribution": {k: v for k, v in res["attribution"].items() if k != "unattributed_examples"}, "missing_derived": res["missing_derived"],
             "xreq_total": res["xreq_total"], "seconds": res["seconds"]}
    print(json.dumps(brief, indent=1, default=str))
    return 0 if res["verdict"] == "PASS" else (4 if res["verdict"] == "MISSING_DERIVED" else 3)
