"""The Match's gates, each a pure function `gate(run: RunArtifacts) -> GateResult` entered in `gate_base`'s registry under
its id, and the reference-integrity rule R1 (`fold_integrity`).  `gates`' docstring states what each gate checks; `gates`
hands every name on, and other code reads them through it."""

from __future__ import annotations

import re
from typing import Any

from verity_vllm.check.gate_base import (
    CHAIN_MIN_SLICES,
    MIN_DECODE_TOKEN_STEPS,
    QUERY_USE_KIND,
    REF_INTEGRITY_RULE,
    TIER_A_K,
    ArtifactError,
    GateResult,
    RunArtifacts,
    _gate,
    _get,
    _int,
    _result,
    _Skip,
    replay_b_key,
)
from verity_vllm.check.verdict_names import (
    CANONICALITY,
    COMPOSITION,
    HOLDOUT,
    HOLDOUT_GATES,
    OBSERVATION,
    PRIMITIVE,
    ROOT_CLOSURE,
    SEMANTIC_CLOSURE,
    VU_PARTITION,
)
from verity_vllm.records.schema_ids import QUERY_USE_SCHEMA

# ---------------------------------------------------------------------------------------------------------
# G1 — observation fidelity
# ---------------------------------------------------------------------------------------------------------


@_gate(OBSERVATION)
def gate_observation(run: RunArtifacts) -> GateResult:
    """Observation fidelity: `noninterference.json` PASS; tokens equal across the (a) observer, (b) hooks and (c) bare
    variants AND equal to the clean control run's tokens (computed here from `generated_token_ids`, never from a
    workload-authored accepted list); every boundary hash equal; `steps == max_tokens`; `layers == expected_layers`
    (from the file; null fails); AND the census reconciliation has residue 0, 0 ambiguous claims and 0 launch-count
    mismatches."""
    ni = run.need("noninterference")
    cr = run.need("census_reconcile")
    ev = [run.rel("noninterference"), run.rel("census_reconcile")]
    problems: list[str] = []
    detail: dict[str, Any] = {}
    # -- non-interference
    if not ni.get("PASS"):
        problems.append("noninterference PASS is false")
    n_cmp = _int(_get(ni, "hash_comparisons", "noninterference"), "hash_comparisons")
    n_eq = _int(_get(ni, "hash_equal", "noninterference"), "hash_equal")
    if n_cmp <= 0:
        problems.append("no boundary hash comparisons")
    elif n_eq != n_cmp:
        problems.append(f"boundary hashes {n_eq}/{n_cmp} equal")
    mismatches, recorded = ni.get("mismatches"), ni.get("problems")   # a report that lists neither counts none
    if mismatches:
        problems.append(f"{len(mismatches)} hash mismatches")
    if recorded:
        problems.append(f"{len(recorded)} problems recorded: {recorded[:2]}")
    toks = ni.get("tokens")   # each variant's generated ids by request; a variant the report leaves out is `missing` below
    stated = {k: toks.get(k) if toks else None for k in ("a_observer", "b_hooks", "c_bare")}
    variants = {k: {str(r): [int(t) for t in v] for r, v in by_request.items()} if by_request else {} for k, by_request in stated.items()}
    missing = [k for k, v in variants.items() if not v]
    if missing:
        problems.append(f"tokens missing for variants {missing}")
    control, control_src = run.control_tokens()
    if control_src == "control/tokens.json":
        ev.append(run.rel("control_tokens"))
    if control is None:
        problems.append("no control tokens (control/tokens.json and c_bare both absent)")
    else:
        for k, v in variants.items():
            if v and v != control:
                problems.append(f"{k} tokens differ from the control run's ({control_src})")
    if variants["a_observer"] and variants["b_hooks"] and variants["a_observer"] != variants["b_hooks"]:
        problems.append("(a) and (b) tokens differ")
    steps = ni.get("steps")
    if not steps:
        steps = []   # a report that states no step compared none
    rows = [s for s in steps if isinstance(s, dict) and "error" not in s]
    if len(rows) != len(steps):
        problems.append("a step is present in only one variant")
    bad_sampled = [r.get("step") for r in rows if r.get("sampled_eq") is not True]
    if bad_sampled:
        problems.append(f"sampled ids differ between (a) and (b) at steps {bad_sampled[:8]}")
    exp = run.expected_steps()
    if exp is None:
        exp = ni.get("expected_steps")
    if exp is None:
        problems.append("expected step count unknown (no control run step count, no workload max_tokens)")
    elif len(steps) != int(exp):
        problems.append(f"{len(steps)} steps, expected {exp} (control run engine steps, else workload max_tokens)")
    if ni.get("expected_steps") is not None and exp is not None and int(ni["expected_steps"]) != int(exp):
        problems.append(f"noninterference.expected_steps={ni['expected_steps']} disagrees with the workload's {exp}")
    exp_layers = ni.get("expected_layers")
    if exp_layers is None:
        problems.append("expected_layers is null (noninterference run without --case/--expected-layers)")
    else:
        exp_layers = int(exp_layers)
        if ni.get("layers") != exp_layers:
            problems.append(f"layers={ni.get('layers')!r} != expected_layers={exp_layers}")
        layer_counts = sorted({(r.get("layers_a"), r.get("layers_b")) for r in rows})
        if rows and layer_counts != [(exp_layers, exp_layers)]:
            problems.append(f"per-step layer counts {layer_counts} != expected_layers={exp_layers} in both variants")
        try:
            C = run.C
        except ArtifactError as e:
            C = None
            problems.append(f"expected_layers={exp_layers} not checked against the profile's NL: {e}")
        if C and C.get("NL") is not None and int(C["NL"]) != exp_layers:
            problems.append(f"expected_layers={exp_layers} disagrees with the profile's NL={C['NL']}")
    detail["noninterference"] = {"PASS": ni.get("PASS"), "hash_comparisons": n_cmp, "hash_equal": n_eq, "steps": len(steps),
                                 "expected_steps": exp, "layers": ni.get("layers"), "expected_layers": ni.get("expected_layers"),
                                 "control_source": control_src, "tokens_equal_control": {k: (v == control) if control and v else None for k, v in variants.items()},
                                 "n_mismatches": len(mismatches) if mismatches else 0, "n_problems": len(recorded) if recorded else 0}
    # -- census
    t = _get(cr, "totals", "census_reconcile")
    residue = _int(_get(t, "residue_kernels", "census_reconcile.totals"), "residue_kernels")
    ambiguous = _int(_get(t, "ambiguous", "census_reconcile.totals"), "ambiguous")
    if residue:
        problems.append(f"census residue {residue} kernels {t.get('residue_names')}")
    if ambiguous:
        problems.append(f"census: {ambiguous} ambiguous compute claims")
    if not cr.get("steps_aligned"):
        problems.append(f"census steps ({cr.get('census_steps')}) not aligned with log steps ({cr.get('log_steps')})")
    census_steps = cr.get("steps")   # a reconciliation that lists no step has none without a log step
    step_errors = [s for s in census_steps if isinstance(s, dict) and "error" in s] if census_steps else []
    if step_errors:
        problems.append(f"{len(step_errors)} census steps without a resolved log step")
    counts = _int(_get(t, "count_mismatches", "census_reconcile.totals (a reconciliation without the launch-count gate: "
                                              "rerun `verity-vllm census --reconcile`)"), "count_mismatches")
    if counts:
        problems.append(f"census: {counts} launch counts differ from their declaring events {t.get('count_mismatch_kernels')}")
    detail["census"] = {"census_kernels": t.get("census_kernels"), "distinct_kernel_names": t.get("distinct_kernel_names"),
                        "residue_kernels": residue, "residue_names": t.get("residue_names"), "ambiguous": ambiguous,
                        "steps_aligned": cr.get("steps_aligned"), "residue_zero": cr.get("residue_zero"),
                        "count_mismatches": counts, "count_exempt": t.get("count_exempt")}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G2 — semantic closure
# ---------------------------------------------------------------------------------------------------------


def _ambiguity(res: dict, run: RunArtifacts) -> tuple[int | None, str, str | None]:
    """(count, source, evidence key): the resolver report's ambiguity field when it has one, else the census reconcile's."""
    for path in ("totals.ambiguous", "ambiguous", "primary_report.ambiguous", "totals.ambiguous_matches", "ambiguous_matches"):
        cur: Any = res
        ok = True
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            n = len(cur) if isinstance(cur, list) else cur
            return _int(n, f"resolution.{path}"), f"resolution.{path}", None
    cr = run.json("census_reconcile")
    if isinstance(cr, dict) and isinstance(cr.get("totals"), dict) and "ambiguous" in cr["totals"]:
        return _int(cr["totals"]["ambiguous"], "census_reconcile.totals.ambiguous"), "census_reconcile.totals.ambiguous (the resolver report has no ambiguity field)", "census_reconcile"
    return None, "unavailable", None


@_gate(SEMANTIC_CLOSURE)
def gate_semantic_closure(run: RunArtifacts) -> GateResult:
    """Semantic closure: the resolution report has 0 Unsupported (and 0 ResolutionErrors), the fold 0 Unresolved (and no
    other fold error: any fold error leaves an instance unemitted), and 0 ambiguous pattern matches (the census
    reconciliation's ambiguity count stands in when the resolver report has no ambiguity field)."""
    res = run.need("resolution")
    fs = run.need("fold_summary")
    ev = [run.rel("resolution"), run.rel("fold_summary")]
    problems: list[str] = []
    unsupported = _int(_get(res, "totals.unsupported", "resolution"), "resolution.totals.unsupported")
    res_errors = _int(_get(res, "totals.errors", "resolution"), "resolution.totals.errors")
    fold_unsupported = _int(_get(fs, "unsupported", "fold_summary"), "fold_summary.unsupported")
    fold_unresolved = _int(_get(fs, "unresolved", "fold_summary"), "fold_summary.unresolved")
    fold_errors = _int(_get(fs, "errors", "fold_summary"), "fold_summary.errors")
    if unsupported:
        report = res.get("primary_report")   # the resolver's own report names the ops; without it the count stands alone
        ops = report.get("unsupported_ops") if report else None
        problems.append(f"{unsupported} Unsupported resolutions ({list(ops)[:4] if ops else []})")
    if res_errors:
        problems.append(f"{res_errors} steps with a ResolutionError")
    if fold_unsupported:
        problems.append(f"fold: {fold_unsupported} Unsupported")
    if fold_unresolved:
        problems.append(f"fold: {fold_unresolved} Unresolved reads")
    if fold_errors:
        problems.append(f"fold: {fold_errors} errors {fs.get('error_classes')}")
    steps = _int(_get(res, "totals.steps", "resolution"), "resolution.totals.steps")
    exp = run.expected_steps()
    if exp is not None and steps != exp:
        problems.append(f"resolution covers {steps} steps, the workload takes {exp}")
    if res.get("header_mismatches"):
        problems.append(f"log header disagrees with the profile's C: {res['header_mismatches']}")
    amb, amb_src, amb_key = _ambiguity(res, run)
    if amb_key:
        ev.append(run.rel(amb_key))
    if amb is None:
        raise _Skip("no ambiguity count: the resolver report has no ambiguity field and census_reconcile.json is missing", ev,
                    {"unsupported": unsupported, "fold_unresolved": fold_unresolved})
    if amb:
        problems.append(f"{amb} ambiguous pattern matches ({amb_src})")
    stated = res.get("totals")
    totals = stated if stated else {}
    detail = {"unsupported": unsupported, "resolution_errors": res_errors, "fold_unsupported": fold_unsupported,
              "fold_unresolved": fold_unresolved, "fold_errors": fold_errors, "fold_error_classes": fs.get("error_classes"),
              "ambiguous": amb, "ambiguity_source": amb_src, "steps": steps, "expected_steps": exp,
              "computes": totals.get("computes"), "compute_rows": totals.get("compute_rows"),
              "n_instances": fs.get("n_instances"), "header_mismatches": res.get("header_mismatches")}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G3 — root closure
# ---------------------------------------------------------------------------------------------------------


@_gate(ROOT_CLOSURE)
def gate_root_closure(run: RunArtifacts) -> GateResult:
    """Root closure: 0 unknown roots (fold summary and resolution report), 0 opaque / prescribed fallbacks, and
    `root_policy.json` status pass with `new_root_classes == []` and no failed root validation."""
    fs = run.need("fold_summary")
    ev = [run.rel("fold_summary")]
    problems: list[str] = []
    stated = fs.get("unknown_roots")   # a fold summary that lists no unknown root has none
    unknown = list(stated) if stated else []
    if unknown:
        problems.append(f"fold: {len(unknown)} unknown roots: {unknown[:4]}")
    res = run.json("resolution")
    if isinstance(res, dict):
        ev.append(run.rel("resolution"))
        roots_report = res.get("roots")   # the resolver's root classes; a report without them names no unknown root
        named = roots_report.get("unknown") if roots_report else None
        ru = list(named) if named else []
        if ru:
            problems.append(f"resolution: {len(ru)} unknown roots: {ru[:4]}")
    opaque = _int(fs.get("opaque_fallbacks", 0), "fold_summary.opaque_fallbacks")
    if opaque:
        problems.append(f"fold: {opaque} opaque / prescribed fallbacks")
    rp = run.need("root_policy")
    ev.append(run.rel("root_policy"))
    if rp.get("status") != "pass":
        problems.append(f"root_policy status {rp.get('status')!r}" + (f": {rp['reason']}" if rp.get("reason") else ""))
    stated = rp.get("new_root_classes")   # a root policy report lists what it found; a field it leaves out found none
    new_classes = list(stated) if stated else []
    if new_classes:
        problems.append(f"{len(new_classes)} new root classes: {new_classes[:4]}")
    classes, roots, claims = rp.get("classes"), rp.get("roots"), rp.get("alias_claims")
    if not classes:
        classes = {}
    if not roots:
        roots = []
    if not claims:
        claims = []
    failed_total = sum(int((c or {}).get("failed") or 0) for c in classes.values() if isinstance(c, dict))
    if failed_total:
        problems.append(f"{failed_total} root validations failed")
    bad_roots = [r for r in roots if isinstance(r, dict) and r.get("status") not in ("pass", "validated", "ok")]
    if bad_roots:
        problems.append(f"{len(bad_roots)} roots not validated: {[r.get('name') for r in bad_roots[:3]]}")
    opaque_roots = [r.get("name") for r in roots if isinstance(r, dict) and str(r.get("class", "")).lower() in ("opaque", "prescribed")]
    if opaque_roots:
        problems.append(f"{len(opaque_roots)} opaque/prescribed roots in root_policy: {opaque_roots[:3]}")
    bad_alias = [a for a in claims if isinstance(a, dict) and a.get("status") not in ("pass", "proven", "ok")]
    if bad_alias:
        problems.append(f"{len(bad_alias)} alias claims unproven: {[a.get('name') for a in bad_alias[:3]]}")
    detail = {"unknown_roots": unknown, "opaque_fallbacks": opaque, "unbound_roots": fs.get("unbound_roots"),
              "root_policy_status": rp.get("status"), "new_root_classes": new_classes,
              "classes": {k: dict(v) if isinstance(v, dict) else v for k, v in classes.items()},
              "roots": len(roots), "roots_failed": len(bad_roots), "alias_claims": len(claims),
              "alias_unproven": len(bad_alias)}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G4 — primitive exactness (tier a replay + quarantine differential tests)
# ---------------------------------------------------------------------------------------------------------


def quarantine_specs(res: dict | None) -> list[str]:
    """Specializations the resolution report attributes to a `quarantine` registry module (heuristic over the fields the
    resolver may expose: `quarantine_specs`, `spec_modules`/`by_spec_module`, or a label mentioning `quarantine`)."""
    if not isinstance(res, dict):
        return []
    out: list[str] = []
    q = res.get("quarantine_specs")
    if isinstance(q, list):
        out += [str(x) for x in q]
    report, by_spec = res.get("primary_report"), res.get("by_spec")   # a report without either names no module and counts no specialization
    for key in ("spec_modules", "by_spec_module", "registries"):
        m = res.get(key) or (report.get(key) if report else None)
        if isinstance(m, dict):
            out += [str(s) for s, mod in m.items() if "quarantine" in str(mod)]
    for s in list(by_spec.keys()) if by_spec else []:
        if "quarantine" in str(s).lower():
            out.append(str(s))
    return sorted(set(out))


@_gate(PRIMITIVE)
def gate_primitive(run: RunArtifacts) -> GateResult:
    """Primitive exactness: the tier-a replay passes with `k >= TIER_A_K` and no uncovered specialization; when the run
    uses newly registered definitions, `quarantine.json` must show every differential test passed (passed == total > 0)
    and `negative_matcher_tests: pass` — missing while the resolution report lists quarantine specs is a failure."""
    rep = run.need("replay_a")
    ev = [run.rel("replay_a")]
    problems: list[str] = []
    status, k = rep.get("status"), rep.get("k")
    stated, cov = rep.get("uncovered_specs"), rep.get("coverage")   # a sample report that lists neither left no specialization out and covered none
    uncovered = list(stated) if stated else []
    if not cov:
        cov = {}
    if status != "pass":
        problems.append(f"tier a status {status!r}" + (f": {rep['reason']}" if rep.get("reason") else ""))
    if not isinstance(k, int) or isinstance(k, bool):
        problems.append(f"tier a k={k!r} is not an integer")
    elif k < TIER_A_K:
        problems.append(f"tier a k={k} < TIER_A_K={TIER_A_K}")
    if uncovered:
        problems.append(f"{len(uncovered)} uncovered specializations: {uncovered[:4]}")
    fails = {s: c for s, c in cov.items() if isinstance(c, dict) and (c.get("fail") or 0)}
    if fails:
        problems.append(f"instances failed in {len(fails)} specializations: {sorted(fails)[:4]}")
    never = sorted(s for s, c in cov.items() if isinstance(c, dict) and int(c.get("evaluated") or 0) == 0 and s not in uncovered)
    if never:
        problems.append(f"{len(never)} specializations evaluated 0 times: {never[:4]}")
    n_pass = sum(int(c.get("pass") or 0) for c in cov.values() if isinstance(c, dict))
    n_fail = sum(int(c.get("fail") or 0) for c in cov.values() if isinstance(c, dict))
    n_skip = sum(int(c.get("skipped") or 0) for c in cov.values() if isinstance(c, dict))
    detail: dict[str, Any] = {"status": status, "k": k, "seed": rep.get("seed"), "specs": len(cov), "covered": len(cov) - len(uncovered),
                              "uncovered_specs": uncovered, "pass": n_pass, "fail": n_fail, "skipped": n_skip,
                              "seconds": rep.get("seconds"), "max_rss_gb": rep.get("max_rss_gb")}
    # -- quarantine (new registered definitions)
    res = run.json("resolution")
    qspecs = quarantine_specs(res)
    q = run.json("quarantine")
    detail["quarantine_specs"] = qspecs
    if q is None:
        detail["quarantine"] = "absent"
        if qspecs:
            ev.append(run.rel("quarantine"))
            problems.append(f"quarantine.json missing while the run uses quarantine specs {qspecs[:3]}")
    else:
        ev.append(run.rel("quarantine"))
        diff = q.get("differential")   # a quarantine report without its differential count ran no test: a problem below
        passed, total = (diff.get("passed"), diff.get("total")) if diff else (None, None)
        neg = q.get("negative_matcher_tests")
        detail["quarantine"] = {"differential": {"passed": passed, "total": total}, "negative_matcher_tests": neg, "status": q.get("status")}
        if not isinstance(passed, int) or not isinstance(total, int) or total <= 0:
            problems.append(f"quarantine differential tests: {passed}/{total} (need total > 0)")
        elif passed != total:
            problems.append(f"quarantine differential tests {passed}/{total} passed")
        if neg != "pass":
            problems.append(f"quarantine negative_matcher_tests = {neg!r}")
        if q.get("status") not in (None, "pass"):
            problems.append(f"quarantine status {q.get('status')!r}")
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G5 — composition exactness (tier b replay of step 0 and every decode snapshot step)
# ---------------------------------------------------------------------------------------------------------

@_gate(COMPOSITION)
def gate_composition(run: RunArtifacts) -> GateResult:
    """Composition exactness (D82): exhaustive provenance/structure, sampled numerical semantics.

    (a) `wiring.json` — every operand read of every instance is reproduced by the independent last-writer map
        (`n_mismatch == 0`, `uncovered == 0`, `reads > 0`), the run is on one stream, and no journaled write lies
        outside the schema-declared write set of its dispatcher op.
    (b) `replay_a.json` — status pass (the verifier-seeded stratified sample: every (specialization, regime) stratum
        has its required passing evaluations; bodies are G4's view of the same report), and `tokens`: every served
        token at every snapshot step reproduced from its observed logits, with step 0 and >= MIN_DECODE_TOKEN_STEPS
        decode steps present.  This proves TokenSelect given the logits; the logits themselves are covered by (a)+(b).
    (c) `replay_chain.json` — >= CHAIN_MIN_SLICES connected slices replayed with computed intermediate values, all
        pass, each with >= 1 chained edge.
    Full replay (`replay_b_step*.json`) is diagnostic: summarized in `detail.full_replay` when present, never gating."""
    problems: list[str] = []
    ev: list[str] = []
    detail: dict[str, Any] = {}
    # (a) exhaustive provenance
    w = run.need("wiring")
    ev.append(run.rel("wiring"))
    st, sc = w.get("streams"), w.get("schema")
    if not st:
        st = {}   # no stream census: a problem below
    if not sc:
        sc = {}   # no schema census: no write is counted outside it
    detail["provenance"] = {"status": w.get("status"), "reads": w.get("reads"), "writes": w.get("writes"), "instances": w.get("instances"),
                            "n_mismatch": w.get("n_mismatch"), "uncovered": w.get("uncovered"), "single_stream": st.get("single_stream"),
                            "streams": st.get("distinct_streams"), "writes_outside_schema": sc.get("writes_outside_schema"),
                            "schema_roots": sc.get("schema_roots"), "triton_roots": sc.get("triton_roots"),
                            "declared_unwritten": sc.get("declared_unwritten"), "seconds": w.get("seconds")}
    if w.get("status") != "pass":
        failures = w.get("failures")   # a failed report may name no failure
        problems.append(f"wiring status {w.get('status')!r}: {'; '.join(failures) if failures else ''}")
    if (w.get("n_mismatch") or 0) > 0:
        problems.append(f"{w['n_mismatch']} operand read(s) not reproduced by the independent provenance map (first: {(w.get('mismatches') or [{}])[0]})")
    if (w.get("uncovered") or 0) > 0:
        problems.append(f"{w['uncovered']} read(s) touch bytes with no journaled producer")
    if not isinstance(w.get("reads"), int) or w.get("reads", 0) <= 0:
        problems.append("wiring: no operand reads checked")
    if st and st.get("single_stream") is False:
        problems.append(f"events on several streams {st.get('distinct_streams')}: log order is not execution order")
    if not st:
        problems.append("wiring: stream census absent (run provenance with the raw log)")
    if sc and (sc.get("writes_outside_schema") or 0) > 0:
        problems.append(f"{sc['writes_outside_schema']} journaled write(s) outside the schema-declared write set")
    # (b) sampled semantics + served tokens
    rep = run.need("replay_a")
    ev.append(run.rel("replay_a"))
    toks = rep.get("tokens")   # the served-token reproduction; a report without it is a problem below
    detail["sample"] = {"status": rep.get("status"), "k": rep.get("k"), "seed": rep.get("seed"), "strata": rep.get("strata"),
                        "specs": rep.get("n_specs"), "specs_covered": rep.get("n_specs_covered"), "summary": rep.get("summary")}
    if rep.get("status") != "pass":
        problems.append(f"tier a status {rep.get('status')!r}" + (f": {rep['status_reason']}" if rep.get("status_reason") else ""))
    if not toks:
        problems.append("tier a report has no `tokens` section (served-token reproduction)")
    else:
        checked, unequal = toks.get("steps"), toks.get("mismatch")   # a section that lists neither checked no step and found no unequal token
        steps = [int(x) for x in checked] if checked else []
        decode = [x for x in steps if x > 0]
        detail["tokens"] = {"steps": steps, "evaluated": toks.get("evaluated"), "equal": toks.get("equal"), "skipped": toks.get("skipped"),
                            "mismatch": unequal[:5] if unequal else [], "all_rows_equal": toks.get("all_rows_equal")}
        if 0 not in steps:
            problems.append("served tokens: step 0 (prefill) not checked")
        if len(decode) < MIN_DECODE_TOKEN_STEPS:
            problems.append(f"served tokens checked at {len(decode)} decode step(s) < MIN_DECODE_TOKEN_STEPS={MIN_DECODE_TOKEN_STEPS}")
        if unequal:
            problems.append(f"{len(unequal)} served token(s) not reproduced from the logits boundary: {unequal[:3]}")
        if toks.get("skipped"):
            problems.append(f"{toks['skipped']} TokenSelect instance(s) skipped (missing logits snapshot?)")
        if not toks.get("evaluated"):
            problems.append("no TokenSelect instance evaluated")
    # (c) chained slices
    ch = run.need("replay_chain")
    ev.append(run.rel("replay_chain"))
    slices = ch.get("slices")
    if not slices:
        slices = []   # a chain report that lists no slice replayed none; its own count is what the floor below reads
    detail["chain"] = {"status": ch.get("status"), "n_slices": ch.get("n_slices"), "n_edges_chained": ch.get("n_edges_chained"),
                       "slices": [{"step": sl.get("step"), "status": sl.get("status"), "edges": sl.get("edges_chained"),
                                   "specs": [str(x).split("{")[0] for x in specs] if (specs := sl.get("specs")) else []} for sl in slices],
                       "seconds": ch.get("seconds")}
    if ch.get("status") != "pass":
        problems.append(f"chain status {ch.get('status')!r}" + (f": {ch.get('status_reason') or ch.get('reason')}" if (ch.get('status_reason') or ch.get('reason')) else ""))
    if (ch.get("n_slices") or 0) < CHAIN_MIN_SLICES:
        problems.append(f"{ch.get('n_slices') or 0} chained slice(s) < CHAIN_MIN_SLICES={CHAIN_MIN_SLICES}")
    if ch.get("n_slices") and any(int(sl.get("edges_chained") or 0) < 1 for sl in slices):
        problems.append("a chained slice chained no producer->consumer edge")
    # diagnostic: full replay when present
    full: dict[str, Any] = {}
    for p in sorted(run.run_dir.glob("replay_b_step*.json")):
        m = re.search(r"step(\d+)\.json$", p.name)
        if not m:
            continue
        rb = run.json(replay_b_key(int(m.group(1))))
        if isinstance(rb, dict):
            served = rb.get("tokens")   # a full replay that reproduced no token states no count
            full[m.group(1)] = {"status": rb.get("status"), "partial": bool(rb.get("partial")), "n_instances": rb.get("n_instances"),
                                "n_diverged": rb.get("n_diverged"), "tokens_equal": served.get("equal") if served else None,
                                "n_from_earlier_steps": rb.get("n_from_earlier_steps")}
    detail["full_replay"] = full or "not run (diagnostic only)"
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G6 — canonicality + regression
# ---------------------------------------------------------------------------------------------------------


@_gate(CANONICALITY)
def gate_canonicality(run: RunArtifacts) -> GateResult:
    """Canonicality + regression: two independent folds (separate processes) of the same log give one digest, and
    `golden.json` (`verity_vllm.properties.golden --check`) passes with every entry's digest and attribution sha as expected.
    Equal digests across two captures of one workload are NOT required (recorded as detail when available)."""
    a = run.need("fold_summary")
    b = run.need("fold_summary_2")
    ev = [run.rel("fold_summary"), run.rel("fold_summary_2")]
    problems: list[str] = []
    da, db = a.get("digest"), b.get("digest")
    if not da or not db:
        problems.append("a fold ran without Program() (no digest)" if (a.get("with_program") is False or b.get("with_program") is False) else "empty digest")
    elif da != db:
        problems.append(f"digests differ: {str(da)[:16]}… vs {str(db)[:16]}…")
    if a.get("n_instances") != b.get("n_instances"):
        problems.append(f"instance counts differ: {a.get('n_instances')} vs {b.get('n_instances')}")
    detail: dict[str, Any] = {"digest": da, "digest_2": db, "digests_equal": bool(da) and da == db, "n_instances": a.get("n_instances"),
                              "spec": a.get("spec"), "seconds": [a.get("seconds"), b.get("seconds")], "max_rss_gb": [a.get("max_rss_gb"), b.get("max_rss_gb")]}
    other = run.json("fold_summary_other")
    if isinstance(other, dict):
        ev.append(run.rel("fold_summary_other"))
        detail["other_capture"] = {"digest": other.get("digest"), "log": other.get("log"), "equal": bool(da) and other.get("digest") == da}
    g = run.need("golden")
    ev.append(run.rel("golden"))
    if g.get("status") != "pass":
        problems.append(f"golden status {g.get('status')!r}" + (f": {g['reason']}" if g.get("reason") else ""))
    entries = g.get("entries")
    if not entries:
        entries = []   # a golden report that lists no entry changed none
    bad = [e for e in entries if isinstance(e, dict) and (e.get("status") != "pass" or (e.get("expected_digest") is not None and e.get("expected_digest") != e.get("got_digest"))
                                                        or (e.get("expected_attribution_sha") is not None and e.get("expected_attribution_sha") != e.get("got_attribution_sha")))]
    if bad:
        problems.append(f"{len(bad)} golden entries changed: {[e.get('name') for e in bad[:4]]}")
    detail["golden"] = {"status": g.get("status"), "entries": len(entries), "changed": [e.get("name") for e in bad]}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G7 — holdout generalization
# ---------------------------------------------------------------------------------------------------------


@_gate(HOLDOUT)
def gate_holdout(run: RunArtifacts) -> GateResult:
    """Holdout generalization: `holdouts.json` (written by the orchestrator) is present with at least one holdout and
    G1..G6 pass in every holdout."""
    h = run.json("holdouts")
    ev = [run.rel("holdouts")]
    if h is None:
        raise _Skip("holdouts.json missing (the orchestrator has not run the holdout workloads)", ev)
    holdouts = h.get("holdouts")
    problems: list[str] = []
    if not holdouts:
        holdouts = []
        problems.append("no holdouts")
    if h.get("status") not in (None, "pass"):
        problems.append(f"holdouts status {h.get('status')!r}")
    per = {}
    for ho in holdouts:
        if not isinstance(ho, dict):
            problems.append(f"malformed holdout entry {ho!r}")
            continue
        gates = ho.get("gates")
        if not gates:
            gates = {}   # a holdout that states no gate passes none
        statuses = {g: (gates.get(g, {}).get("status") if isinstance(gates.get(g), dict) else gates.get(g)) for g in HOLDOUT_GATES}
        failing = sorted(g for g, s in statuses.items() if s != "pass")
        per[str(ho.get("name"))] = {"workload": ho.get("workload"), "run_dir": ho.get("run_dir"), "gates": statuses, "pass": not failing}
        if failing:
            problems.append(f"holdout {ho.get('name')}: {failing} not passing")
    detail = {"status": h.get("status"), "n_holdouts": len(holdouts), "holdouts": per}
    return _result(problems, ev, detail)


# ---------------------------------------------------------------------------------------------------------
# G8 — VU partition (structural; outside the frozen G1..G7 set)
# ---------------------------------------------------------------------------------------------------------


@_gate(VU_PARTITION)
def gate_vu_partition(run: RunArtifacts) -> GateResult:
    """Proof-block partition: `query_use.json` (`verity-ir/query-use/v1`, written by the partition track: use configuration + verdict).
    Pass iff `verdict.partition.status == "exact"` and `verdict.width.status == "ok"` (no violators, no unsupported); fail with
    the partition codes / width violators otherwise.  Absent document -> `skipped` ("no query-use document"); a document that
    is not this schema, or lacks the verdict fields, -> `error`.  Reads only fields of the frozen document shape (the freeze
    doc); never touches G1..G7."""
    doc = run.json("query_use")
    ev = [run.rel("query_use")]
    if doc is None:
        raise _Skip("no query-use document", ev)
    what = run.rel("query_use")
    if not isinstance(doc, dict):
        raise ArtifactError(f"{what}: not a JSON object")
    if doc.get("schema") != QUERY_USE_SCHEMA:
        raise ArtifactError(f"{what}: schema {doc.get('schema')!r} != {QUERY_USE_SCHEMA!r}")
    use = doc.get("use")
    if not isinstance(use, dict):
        use = {}
    v = _get(doc, "verdict", what)
    part = _get(v, "partition", f"{what}.verdict")
    width = _get(v, "width", f"{what}.verdict")
    p_status = _get(part, "status", f"{what}.verdict.partition")
    w_status = _get(width, "status", f"{what}.verdict.width")
    listed = part.get("codes"), width.get("violators"), width.get("unsupported")   # a verdict lists what it found; a field it leaves out found none
    codes = [str(c) for c in listed[0]] if listed[0] else []
    violators = list(listed[1]) if listed[1] else []
    unsupported = list(listed[2]) if listed[2] else []
    w_max = width.get("w_max", use.get("w_max"))
    v_status = v.get("status")                      # "ok" | "fail" | "not-ready" | "unsupported" (qpart INTERFACE NOTE 21:45Z)
    # Two classes of finding (lead's decision 22:22Z).  FAIL: a wrong partition (overlap / gap), a width violation, `not-ready` (a
    # Definition / gate wider than w_max inside the partition), the evaluator's own `fail`, a foreign document.  UNSUPPORTED: the
    # evaluator could not decide (partition / width / status `unsupported`, member classes with undecidable w_out) -- neither pass nor
    # fail: G8 is `unsupported`, the run verdict INCOMPLETE, never REJECTED on that basis.  A fail-class finding dominates.
    problems: list[str] = []
    undecided: list[str] = []
    if use.get("kind") != QUERY_USE_KIND:
        problems.append(f"use.kind {use.get('kind')!r} != {QUERY_USE_KIND!r}")
    if p_status == "unsupported":
        undecided.append("partition 'unsupported'" + (f" {codes[:6]}" if codes else ""))
    elif p_status != "exact":
        problems.append(f"partition {p_status!r}" + (f" {codes[:6]}" if codes else ""))
    if w_status == "unsupported":
        undecided.append("width 'unsupported'")
    elif w_status != "ok":
        problems.append(f"width {w_status!r}")
    if violators:
        problems.append(f"{len(violators)} member class(es) with w_out > w_max={w_max}: {violators[:4]}")
    if unsupported:
        undecided.append(f"{len(unsupported)} member class(es) with undecidable w_out: {unsupported[:4]}")
    if v_status == "unsupported":
        undecided.append("verdict.status 'unsupported'")
    elif v_status != "ok":                          # `fail` / `not-ready` (and anything unknown): surfaced verbatim, not a pass
        problems.append(f"verdict.status {v_status!r}")
    # binding: the companion query document (`RUN_DIR/query.json`, qpart 22:14Z) must carry the same binding_id and the run's digest
    binding: dict[str, Any] = {"checked": False}
    qdoc = run.json("query_doc")
    if isinstance(qdoc, dict):
        ev.append(run.rel("query_doc"))
        fs = run.json("fold_summary")
        digest = fs.get("digest") if isinstance(fs, dict) else None
        binding = {"checked": True, "query_binding_id": qdoc.get("binding_id"), "query_program_digest": qdoc.get("program_digest"),
                   "run_digest": digest, "query_id": qdoc.get("query_id")}
        if qdoc.get("binding_id") != doc.get("binding_id"):
            problems.append("binding: query.json binding_id != query_use.json binding_id")
        if digest is not None and qdoc.get("program_digest") != digest:
            problems.append("binding: query.json program_digest != the run's fold digest")
    rd = v.get("readiness_32")
    if not isinstance(rd, dict):
        rd = {}
    over_32 = rd.get("violators")
    detail: dict[str, Any] = {
        "schema": doc.get("schema"), "binding_id": doc.get("binding_id"), "use": {"kind": use.get("kind"), "w_max": use.get("w_max")},
        "status": v_status,                          # the document's verdict.status, verbatim
        "partition": {"status": p_status, "codes": codes[:32]},
        "width": {"status": w_status, "w_max": w_max, "max_seen": width.get("max_seen"), "n_violators": len(violators),
                  "violators": violators[:16], "n_unsupported": len(unsupported), "unsupported": unsupported[:16]},
        "count": v.get("count"), "verdict_status": v_status,
        "readiness_32": {"ok": rd.get("ok"), "max_w": rd.get("max_w"), "n_violators": len(over_32) if over_32 else 0} if rd else None,
        "evaluator": doc.get("evaluator"),
        "binding": binding, "binding_checked": binding["checked"],
        "undecided": undecided,
    }
    if problems:
        return _result(problems, ev, detail)
    if undecided:
        return GateResult("", "", "unsupported", ev, detail, "unsupported (" + "; ".join(undecided) + ")")
    return _result([], ev, detail)


# ---------------------------------------------------------------------------------------------------------
# R1 — reference integrity, fail-closed (outside the frozen G1..G7 set)
# ---------------------------------------------------------------------------------------------------------

_INTEGRITY_COUNTS = ("errors", "unresolved", "unsupported")


def _count_field(d: dict, key: str, what: str, problems: list[str], unverified: list[str]) -> int | None:
    """An integer count from a summary; a missing / non-integer / negative value is *unverified* evidence (never zero)."""
    if key not in d:
        unverified.append(f"{what}.{key} missing")
        return None
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        unverified.append(f"{what}.{key} = {v!r} is not a non-negative integer")
        return None
    if v > 0:
        problems.append(f"{what}.{key} = {v}")
    return v


def fold_integrity(fold_summary: Any, fold_summary_2: Any = None, resolution: Any = None) -> tuple[str, list[str], dict]:
    """The pure R1 rule over the parsed artifacts: `("pass" | "fail" | "unverified", problems, detail)`.

    * `fail`       — any fold error / Unresolved read / Unsupported resolution (either fold summary), or any ResolutionError /
                     Unsupported in the resolution report's totals: the Program is NOT eligible as an accepted reference;
    * `unverified` — no fold summary, a count that is missing or not an integer, or inconsistent evidence (`error_classes`
                     histogram or `per_step` error sum disagreeing with `errors`): missing evidence is never read as zero;
    * `pass`       — every count present, integer and zero.
    A `fail` finding dominates `unverified`.  `fold_summary_2` / `resolution` are optional inputs (checked when present)."""
    problems: list[str] = []
    unverified: list[str] = []
    detail: dict[str, Any] = {"rule": dict(REF_INTEGRITY_RULE), "fold_summary": None, "fold_summary_2": None, "resolution": None}
    for what, fs in (("fold_summary", fold_summary), ("fold_summary_2", fold_summary_2)):
        if fs is None:
            if what == "fold_summary":
                unverified.append("fold_summary.json missing: no fold error evidence")
            continue
        if not isinstance(fs, dict):
            unverified.append(f"{what}.json is not a JSON object")
            continue
        counts = {k: _count_field(fs, k, what, problems, unverified) for k in _INTEGRITY_COUNTS}
        classes = fs.get("error_classes")
        n_err = counts["errors"]
        if isinstance(classes, dict) and n_err is not None:
            try:
                total = sum(int(v) for v in classes.values())
            except (TypeError, ValueError):
                unverified.append(f"{what}.error_classes has non-integer counts")
            else:
                if total != n_err:
                    unverified.append(f"{what}.error_classes sums to {total}, errors = {n_err}")
        per_step = fs.get("per_step")
        if isinstance(per_step, list) and per_step and n_err is not None:
            vals = [s.get("errors") for s in per_step if isinstance(s, dict)]
            ints = [v for v in vals if isinstance(v, int) and not isinstance(v, bool)]
            if len(ints) == len(vals) == len(per_step) and sum(ints) != n_err:
                unverified.append(f"{what}.per_step errors sum to {sum(ints)}, errors = {n_err}")
        detail[what] = {**counts, "error_classes": classes if isinstance(classes, dict) else None, "digest": fs.get("digest"),
                        "with_program": fs.get("with_program"), "n_instances": fs.get("n_instances")}
    if resolution is not None:
        if not isinstance(resolution, dict) or not isinstance(resolution.get("totals"), dict):
            unverified.append("resolution.json has no `totals` object")
        else:
            t = resolution["totals"]
            rc = {k: _count_field(t, k, "resolution.totals", problems, unverified) for k in ("errors", "unsupported")}
            detail["resolution"] = rc
    status = "fail" if problems else ("unverified" if unverified else "pass")
    detail["unverified"] = unverified
    return status, problems, detail


@_gate("R1")
def gate_R1(run: RunArtifacts) -> GateResult:
    """Reference integrity, fail-closed (`REF_INTEGRITY_RULE`): `fold_integrity` over `fold_summary.json`, `fold_summary_2.json`
    (when present) and `resolution.json` (when present).  Never `skipped`: a missing / unreadable summary is `unverified`."""
    ev = [run.rel("fold_summary")]
    arts: dict[str, Any] = {}
    unreadable: list[str] = []
    for key in ("fold_summary", "fold_summary_2", "resolution"):
        try:
            arts[key] = run.json(key)
        except ArtifactError as e:
            arts[key] = None
            unreadable.append(f"{run.rel(key)}: {e}")
        if key != "fold_summary" and arts[key] is not None:
            ev.append(run.rel(key))
    status, problems, detail = fold_integrity(arts["fold_summary"], arts["fold_summary_2"], arts["resolution"])
    if unreadable:
        detail["unreadable"] = unreadable
        if status == "pass":
            status = "unverified"
    if status == "fail":
        reason = "fold/construction errors > 0: " + "; ".join(problems) + f" [rule {REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}]"
    elif status == "unverified":
        reason = "unverified (" + "; ".join(detail["unverified"] + unreadable) + f") [rule {REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}]"
    else:
        reason = None
    return GateResult("", "", status, ev, detail, reason)
