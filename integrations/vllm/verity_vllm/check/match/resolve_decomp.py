"""verity_vllm.check.match.resolve_decomp -- the Match stage's X-09 decomposition of a fold record, driven by `observe.iter_observations` + `resolve`
(v2 Phase 4b), in the shape `cb_a.batch_decomp.decompose` writes (`match_decomp.json`: per request `canonical_sha256{a,b}`, `derived_digest`,
`xreq`, `n_instances`; `attribution.how` / `unattributed` / `row_rule_disagreements`) so the two documents compare field by field.

WHAT IS v2 HERE: the attribution of every value to a request (`resolve.attribute_record`: the stream of observations through
`resolve` / `validate` / `LiveState.advance`; named faults counted).  WHAT IS WRAPPED, NOT REWRITTEN: the fold decode (`compare.load_oracle`
-- the files of record hashed as read), the request projection (`batch_decomp.Projection` over the resolved value ids; weight-only values
joined by `batch_decomp.shared_legs`), the component comparison and the canonical hash (`compare.compare_steps`: `canonical_sha256.a` =
the derived component, `.b` = the record's projection), the shape rule (LP from the header, T = sampling events - prefill chunks) and the
weights / seed / peers / witness input layout (`batch_decomp.request_inputs`).  Nothing here writes into the
record; the result is returned (a caller may dump it beside its own outputs).

Cost = v1's: the fold is decoded once (`load_oracle`, multi-GB at B>=8: a pod), each request's projection is compared with its component.
"""
from __future__ import annotations

import collections
import os
import time
from typing import Any

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import program_compare as C

from verity_vllm.observe import contract as OB
from verity_vllm.check.match import resolve as RS

SCHEMA = "verity-vllm/resolve-decomp/v1"


def decompose(match_dir: str, derived_dirs: list[str], *, oracle: "C.RootView | None" = None, log_path: str | None = None,
              components: dict[str, "C.RootView"] | None = None, max_fault_examples: int = 24) -> dict[str, Any]:
    """The record's decomposition with v2's attribution.  `derived_dirs`: the row's request components (`build_request*/`), matched to a
    request by (LP, T) as BD.decompose does.  Returns the document (never writes)."""
    t0 = time.time()
    match_dir = OB.match_dir(match_dir)
    log_path = log_path or os.path.join(match_dir, "capture", "log.jsonl.gz")
    hdr = OB.read_header(match_dir)
    if os.path.isfile(log_path):
        header, steps = BD.read_steps(log_path)
        reqs, param_of, parent_of = BD.request_params(header)
        lp_of = {r["request_id"]: len(r.get("prompt_token_ids") or ()) for r in reqs}
        log_state = "capture log read (RunHeader + StepStart)"
    else:
        # the capture log is not beside the record (archived: `capture.MOVED.txt`): the request table is the fold's own
        # header line (index = prompt parameter, prompt_len); the rows channel and the meta-vs-rows cross-check are then unavailable and
        # SAID so -- every row of a current record is stamped (meta), so attribution itself does not read the log
        header, steps = None, None
        reqs = [{"request_id": str(r["request_id"]), "index": int(r["index"])} for r in sorted(hdr.requests, key=lambda r: int(r["index"]))]
        param_of = {r["request_id"]: r["index"] for r in reqs}
        parent_of = {}
        lp_of = {str(r["request_id"]): int(r["prompt_len"]) for r in hdr.requests}
        log_state = f"capture log ABSENT ({log_path}): request table from the instances header; rows channel + row-rule cross-check not available"
    # --- v2: attribution by occurrence resolution over the observation stream (O(1) live state, the record never loaded)
    t_a = time.time()
    att = RS.attribute_record(match_dir, header=hdr, run_header=header, steps=steps, max_fault_examples=max_fault_examples)
    t_attr = time.time() - t_a
    # --- v1, wrapped: the fold decode, the projections, the component comparison
    t_l = time.time()
    B = oracle if oracle is not None else C.load_oracle(match_dir, "oracle")
    t_load = time.time() - t_l
    n_in = B.n_input_nodes
    if att.n_input_nodes != n_in:
        raise ValueError(f"{match_dir}: instances header / program.json disagree on n_input_nodes ({att.n_input_nodes} vs {n_in})")
    seed_node_of, peer_node_of, _witnesses, weights_nodes = BD.request_inputs(B, reqs)
    k2i = {r["k"]: i for i, r in enumerate(B.inst)}
    meta_of = {o["node"] + n_in: o for o in B.oracle_rows}

    def to_i(nodes: list[int]) -> list[int]:
        out = []
        for k in nodes:
            i = k2i.get(k)
            if i is None:
                raise ValueError(f"value id {k} of the observation stream is not an instance of the decoded fold")
            out.append(i)
        return out

    legs_nodes = att.legs(steps)                       # node ids per request, weight-only values joined (BD.shared_legs)
    per_req_i = collections.OrderedDict((rid, to_i(ks)) for rid, ks in att.per_req.items())
    shared_i = [(k2i[k], s) for k, s in att.shared_nodes if k in k2i]
    legs = collections.OrderedDict((rid, to_i(ks)) for rid, ks in legs_nodes.items())

    derived: dict[tuple[int, int], str] = {}
    bad_derived: list[str] = []
    for d in derived_dirs:
        sh = BD.derived_shape(d)
        if sh is None:
            bad_derived.append(d)
        else:
            derived.setdefault(sh, d)
    components = components if components is not None else {}
    res: dict[str, Any] = {
        "schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "oracle_dir": match_dir, "log": log_path, "log_state": log_state,
        "n_requests_header": len(reqs), "n_output_sequences": len(att.per_req), "engine_steps": len(steps) if steps else len(att.present_by_step),
        "oracle": {"digest": B.digest, "instances": len(B.inst), "n_input_nodes": n_in,
                   "weights_input_nodes": {"n": len(weights_nodes), "first": min(weights_nodes), "last": max(weights_nodes),
                                           "request_owned_inputs": len(reqs) + len(seed_node_of)},
                   "weight_map": {"n_ranges": len(B.oracle_wmap), "n_overlaps": len(B.oracle_wmap_overlaps)}},
        "fold_load": dict(C.LOAD_NOTES.get(match_dir) or {}),
        "attribution": {"how": dict(att.how), "unattributed": len(att.unattributed), "unattributed_examples": att.unattributed[:12],
                        "attributed": sum(len(v) for v in att.per_req.values()), "row_rule_disagreements": att.disagreements,
                        "weight_only_shared": len(att.shared_nodes),
                        "weight_only_specs": dict(collections.Counter(B.inst[i]["spec"].split("{")[0] for i, _ in shared_i).most_common(6)),
                        "faults": dict(att.faults), "fault_examples": att.fault_examples, "path": "observe.iter_observations -> resolve.resolve"},
        "derived_available": {f"LP{lp}_T{t}": d for (lp, t), d in sorted(derived.items())}, "derived_unreadable": bad_derived,
        "requests": [], "missing_derived": []}
    all_equal = True
    missing: set[tuple[int, int]] = set()
    for rid, idxs in legs.items():
        pidx = param_of.get(rid)
        ent: dict[str, Any] = {"request_id": rid, "parent": parent_of.get(rid), "prompt_param": pidx, "n_instances": len(idxs),
                               "n_values_own": len(per_req_i.get(rid, ()))}
        if pidx is None:
            ent["error"] = "request not in RunHeader.workload (no prompt parameter)"
            res["requests"].append(ent)
            all_equal = False
            continue
        LP = lp_of[parent_of.get(rid, rid)]
        P = BD.Projection(B, f"oracle[{rid}]", idxs, tok_node=pidx, weights_node=weights_nodes, seed_node=seed_node_of.get(parent_of.get(rid, rid)),
                          peer_node=peer_node_of.get(parent_of.get(rid, rid)))
        P._all_seed_nodes = set(seed_node_of.values())
        n_ts = sum(1 for r in P.inst if RS.is_sampling_event(r["spec"]))
        # prefill chunks: engine steps in which the request had PROMPT rows (position < LP) -- the log's StepStart rows when read, else the
        # stream's own statement (the resolved request-relative step 0 occurrences)
        if steps:
            prefill_steps = sorted(s for s, st in steps.items() if any(r == rid and pos < LP for r, pos in st.get("rows", [])))
        else:
            prefill_steps = att.prefill_steps(rid)
        chunks = max(1, len(prefill_steps))
        T = n_ts - chunks
        es = BD.engine_steps_of(P, meta_of)
        ent.update({"LP": LP, "T": T, "token_selects": n_ts, "steps": len(P.steps), "prefill_chunks": chunks, "prefill_engine_steps": prefill_steps[:8],
                    "chunked_steps": sum(1 for e in es if len(e) != 1),
                    "engine_step_range": [min(min(e) for e in es if e), max(max(e) for e in es if e)] if es else None})
        if n_ts == 0:
            ent["error"] = "no sampling event attributed to the request"
            res["requests"].append(ent)
            all_equal = False
            continue
        d = derived.get((LP, T))
        if d is None:
            ent["derived"] = None
            missing.add((LP, T))
            res["requests"].append(ent)
            all_equal = False
            continue
        A = components.get(d)
        if A is None:
            A = components[d] = C.load_derived(d, f"derived[LP{LP}_T{T}]")
        cmp = C.compare_steps(A, P, unfold_a=False)
        ent["derived"] = d
        ent["derived_digest"] = A.digest
        ent["xreq"] = {"counts": dict(P.xreq), "examples": P.xreq_examples}
        seq_equal = bool(cmp["canonical_sha256"]["equal"]) and not P.xreq
        ent["sequence_equal"] = seq_equal
        ent["canonical_sha256"] = cmp["canonical_sha256"]
        ent["criterion"] = "sequence"
        ent["canonical_equal"] = seq_equal
        ent["totals"] = cmp["totals"]
        ent["a_steps"], ent["b_steps"] = cmp["a"]["steps"], cmp["b"]["steps"]
        bad = [s for s in cmp["steps"] if not s["spec_seq_equal"] or (s.get("wiring") or {}).get("operands_different")]
        ent["first_divergences"] = [{"step": s["step"], "n_a": s["n_a"], "n_b": s["n_b"], "div": s.get("first_divergences_T_normalised", [])[:2],
                                     "wiring_examples": (s.get("wiring") or {}).get("examples", [])[:3]} for s in bad[:3]]
        ent["n_divergent_steps"] = len(bad)
        all_equal = all_equal and seq_equal
        res["requests"].append(ent)
    res["missing_derived"] = [f"LP{lp}_T{t}" for lp, t in sorted(missing)]
    res["xreq_total"] = sum(sum(e.get("xreq", {}).get("counts", {}).values()) for e in res["requests"])
    res["verdict"] = "PASS" if all_equal and not att.unattributed and not missing else "FAIL"
    res["reasons"] = ([f"{len(att.unattributed)} unattributed value(s)"] if att.unattributed else []) + \
                     ([f"missing derived: {res['missing_derived']}"] if missing else []) + \
                     [f"{e['request_id']}: {e.get('error') or 'sequence differs'}" for e in res["requests"] if e.get("error") or e.get("sequence_equal") is False]
    res["seconds"] = round(time.time() - t0, 1)
    res["timing"] = {"attribution_s": round(t_attr, 1), "fold_load_s": round(t_load, 1)}
    return res
