"""X-03 (R12): fail-closed Commit verdict for the sweep harness (`commit_delta`).  Dependency-free so tests and report tooling can import it.

REL-02 (R13 rel, interim for reviewer finding R-14): `expected_tensors` counts acquisitions the collector ATTEMPTED, so a required producer
that never fires (Hopper FA3 with the FA2 tap loaded, a tap left off the command line, a dropped value family) contributed 0 to both sides
and passed.  Until the expected manifest comes from the Program/query (lane p0b), the verdict also takes the committer's coverage
declaration and a set of REQUIRED class ids: every required class must be declared, and every declared class that binds leaves
(`bound-as-leaf` / `registered-input`) must have `tensors > 0`."""
from __future__ import annotations

# The interim required-value manifest for the native committer + FA2 hidden tap on an FA2 dense model (the R12 canary rows' `required_classes`
# minus the two derived-from-bound norm classes, which are protocol clauses with no leaves and are checked for presence only when declared).
# Lanes override with `commit_delta --required-classes a,b,c` (recorded in verdict.json) when the model's producers genuinely differ
# (e.g. a native FA3 tap class on Hopper) -- never to make a missing producer pass.
REQUIRED_CLASSES_DEFAULT = ("input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights", "fa2_hidden_m1_stream")
LEAF_HOWS = ("bound-as-leaf", "registered-input")
HIDDEN_CLASS_BY_FA = {2: "fa2_hidden_m1_stream", 3: "fa3_hidden_m1_stream"}


def required_classes_default(flash_attn_version: int | None) -> tuple[str, ...]:
    """R13 hopper: the default manifest keyed by the engine's attention implementation (the TargetProfile fact `effective_flash_attn_version`,
    the same fact that selects the Bet A attention Definition and the Match pattern): the attention hidden class is the FA2 tap's under FA2
    and the FA3 tap's (`fa3_hidden_m1_stream`, hidden_source.HiddenM1SourceFA3) under FA3.  None -> the FA2 default (pre-R13 rows)."""
    hidden = HIDDEN_CLASS_BY_FA.get(int(flash_attn_version) if flash_attn_version is not None else 2, "fa2_hidden_m1_stream")
    return tuple(hidden if c == "fa2_hidden_m1_stream" else c for c in REQUIRED_CLASSES_DEFAULT)

def json_short(v: dict, n: int = 160) -> str:
    import json

    return json.dumps({k: x for k, x in v.items() if x and k != "step"}, default=str)[:n]



def class_coverage(declaration: dict | None) -> dict[str, dict]:
    """{class id: {"how", "tensors", "bytes"}} from a coverage declaration (see commit_integ.api.declaration)."""
    out: dict[str, dict] = {}
    for c in (declaration or {}).get("classes") or []:
        out[c.get("class")] = {"how": c.get("how"), "tensors": c.get("tensors"), "bytes": c.get("bytes")}
    return out


def required_classes_check(declaration: dict | None, required: tuple[str, ...] | list[str] | None) -> tuple[bool, str]:
    """R-14 interim: (ok, reason).  Every class in `required` must be declared; every declared leaf-binding class must have tensors > 0."""
    cov = class_coverage(declaration)
    if declaration is None:
        return False, "coverage declaration missing: cannot check the required classes (R-14)"
    for cls in required or ():
        if cls not in cov:
            return False, f"required class {cls!r} not declared: its producer never fired / was not attached (R-14 omitted producer)"
    for cls, d in cov.items():
        if d["how"] in LEAF_HOWS:
            n = d.get("tensors")
            if n is None or int(n) <= 0:
                return False, f"class {cls!r} declared {d['how']} with tensors={n}: producer never fired (R-14 omitted producer)"
    return True, "PASS"


def binding_check(bnd: dict, *, min_openings: int = 64) -> tuple[bool, str]:
    """R14 BIND-01 (TABLE-RULES r14-v1 Commit): (1) the binding map was built and its context (roots, layout) is the run's; (2) every identity of
    the collector-independent expected population has a binding (a duplicate / alias / extra copy never counts for a missing one --
    F-rev-01 drop `/1` + bind `/0` twice fails here); (3) openings drawn from the EXPECTED population all had a binding and all verified
    (retain=none: the values come from replay and are checked by openings_after_release)."""
    if bnd.get("error"):
        return False, f"binding map not built: {bnd['error']}"
    ctx = bnd.get("context") or {}
    if not ctx.get("ok"):
        return False, f"binding-map context: {ctx.get('why')}"
    cov = bnd.get("coverage") or {}
    if cov.get("error"):                                                         # [MAN-02] incomplete manifest (unmodelled families) -> fail closed, named
        return False, f"identity coverage: {cov['error']}"
    if not cov.get("ok"):
        if int(cov.get("missing_n") or 0) == 0 and int(cov.get("ambiguous_n") or 0) > 0:
            # [R16 rev F-r16-08 / M-0077] every required identity is bound, but not UNAMBIGUOUSLY: two leaf entries under one identity
            amb = (cov.get("ambiguous") or [{}])[0]
            return False, (f"identity coverage: {cov.get('ambiguous_n')} of {cov.get('expected_n')} required identities have more than one binding "
                           f"(C1 requires one unambiguous binding; first: step {amb.get('step')} {amb.get('op_path')}/{amb.get('output_member')} inv {amb.get('invocation')} "
                           f"req {amb.get('request_id')} rank {amb.get('rank')} bound {amb.get('bindings_n')} x at {amb.get('bound_at')})")
        miss = (cov.get("missing") or [{}])[0]
        return False, (f"identity coverage: {cov.get('missing_n')} of {cov.get('expected_n')} required identities have no binding "
                       f"(first: step {miss.get('step')} {miss.get('op_path')}/{miss.get('output_member')} inv {miss.get('invocation')} req {miss.get('request_id')} rank {miss.get('rank')})")
    po = bnd.get("openings_population") or {}
    if int(po.get("unbound_n") or 0) > 0:
        return False, f"{po['unbound_n']} population-drawn openings had no binding (first: {json_short((po.get('unbound') or [{}])[0])})"
    if not po.get("skipped"):
        if int(po.get("opened") or 0) < min(min_openings, int(po.get("n") or 0)):
            return False, f"population openings opened {po.get('opened')} < {min(min_openings, int(po.get('n') or 0))}"
        if int(po.get("verified") or 0) != int(po.get("opened") or 0):
            return False, f"population openings verified {po.get('verified')} != opened {po.get('opened')}"
    elif int(po.get("verified") or 0) != int(po.get("opened") or 0):
        # [R17 fp8] retain=host + exclude: the host share opened here must verify; the deferred share is opened by replay (openings_after_release)
        return False, f"population openings (host share) verified {po.get('verified')} != opened {po.get('opened')}"
    return True, "PASS"


_WHY_BY_NAME = ("no-evaluator", "state-linked", "plane-fed", "interior-composed-v2")     # == sampled_replay.WHY_CLASSES_BY_NAME


def _why_class_of(why: str | None) -> str:
    try:
        from verity_capture.commit.sampled_replay import why_class
        return why_class(why)
    except Exception:  # noqa: BLE001
        return "other"


_POPULATION_GAP_CLASSES = ("plane-fed", "interior-composed-v2")     # the MAN-06f / MAN-06g population moves (rev-cb M-0751 (b))
# [R16 rev F-r16-07 / M-0068; report M-0059 (2)] which by-name classes form (B) CAN cover: form (B) compares the Match record's dispatcher-level
# module outputs (`instance_outputs`), so an interior-composed VU (MAN-06g: its module's returned tensor IS an instance output) is covered by a
# complete (B); a PLANE-FED VU (MAN-06f) is not -- its committed members are the moe/ block-stream planes, and oracle_compare has NO oracle
# family for `moe/` (NO_ORACLE_FAMILY_PREFIXES: counted 'no oracle family', excluded from compared).  Neither form value-checks those planes
# (form (A) has no row->plane mapping in this replay), so a plane-fed gap FAILs by name whatever form (B) says; C1 + C3 still hold on them.
_FORM_B_COVERABLE_CLASSES = ("no-evaluator", "state-linked", "interior-composed-v2")
_PLANE_FED_NEITHER_FORM = ("have committed moe/ plane members value-checked by NEITHER form (form (B) has no oracle family for moe/ planes "
                           "[oracle_compare.NO_ORACLE_FAMILY_PREFIXES]; form (A) has no row->plane mapping in this replay [MAN-06f, v2])")


def _form_b_complete_over_values(oc: dict | None) -> str | None:
    """[COORD M-0721 (1); rev-cb M-0740 G3] None when form (B) -- the Match-oracle compare -- is COMPLETE over the values: result True, 0
    mismatch, complete (every step snapshotted), compared > 0, attribution complete, and the module outputs (`instance_outputs`, the family
    the gap strata produce) compared > 0 with none left not-compared (weights / hidden_stream have no oracle family and never stand in).
    Else the named reason it is not."""
    if not isinstance(oc, dict):
        return "form (B) absent (no --match-dir oracle compare on the record)"
    if oc.get("result") is not True:
        return f"form (B) result {oc.get('result')} ({oc.get('grade') or 'no grade'})" + (f": {str(oc.get('why'))[:120]}" if oc.get("why") else "")
    if int(oc.get("mismatch_n") or 0) != 0:
        return f"form (B) {oc.get('mismatch_n')} mismatch"
    if oc.get("complete") is not True:
        return f"form (B) not complete ({oc.get('grade') or 'partial'}: steps not snapshotted {oc.get('steps_not_snapshotted') or '?'} -- Match with MATCH_SNAP_STEPS=all)"
    if int(oc.get("compared") or 0) <= 0:
        return "form (B) compared 0"
    if (oc.get("attribution") or {}).get("complete", True) is False:
        return "form (B) request attribution not complete"
    cbf, ncf = oc.get("compared_by_family"), oc.get("not_compared_by_family")
    if isinstance(cbf, dict) and (int(cbf.get("instance_outputs") or 0) <= 0 or int((ncf or {}).get("instance_outputs") or 0) > 0):
        return f"form (B) instance outputs compared {cbf.get('instance_outputs') or 0}, not compared {(ncf or {}).get('instance_outputs') or 0}"
    return None


_NO_EVALUATOR_PREFIX = "no registered evaluator for "      # == the text sampled_replay.population() counts a non-NOT_YET family under


def _not_yet_texts() -> set[str]:
    """The replay's stated applicability domain as the exact texts `population.not_evaluable` carries for it (sampled_replay.NOT_YET values:
    RoPE_v1 no evaluator yet; Attention_v3 / AttentionHead_v1 state-linked) -- read from the module, never re-typed here."""
    try:
        from verity_capture.commit.sampled_replay import NOT_YET
        return {str(v) for v in NOT_YET.values()}
    except Exception:  # noqa: BLE001
        return set()


def _is_token_select(fam: str) -> bool:
    try:
        from verity_capture.commit.required_manifest import is_token_select
        return bool(is_token_select(fam))
    except Exception:  # noqa: BLE001
        return str(fam).split("_v", 1)[0].endswith("TokenSelect")


def _no_evaluator_gaps(not_ev: dict) -> tuple[dict[str, int], dict[str, int]]:
    """[R16 rev F-r16-20 / M-0400; pin F-r16-20b / M-0431] ({selector family: n}, {other family: n}) among the population's `no registered
    evaluator for <fam>` exclusions whose family is NOT in NOT_YET.  Selectors (TokenSelect families, e.g. GumbelTopPTokenSelect_v1 on a
    stochastic row): the sampled id is THE value a stochastic row adds and is reproducible from the committed logits only with the seed AND
    the top-p split count S (reqval M-0305 / REQ-03).  Others (Hopper: Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1} = every linear, Attention_v2
    = every attention row -- targets.gemm_spec / attention_spec bind them on a cc 9.0 step; neither is in EVALUATORS nor NOT_YET): their
    module outputs ARE the oracle's `instance_outputs`, so a complete form (B) covers them exactly as it covers the interior-composed class.
    Both carry the SAME class label ('no-evaluator') as the RoPE_v1 applicability domain and must not hide behind it."""
    sel: dict[str, int] = {}
    other: dict[str, int] = {}
    domain = _not_yet_texts()
    for k, v in not_ev.items():
        k, n = str(k), int(v or 0)
        if n <= 0 or k in domain or not k.startswith(_NO_EVALUATOR_PREFIX) or _why_class_of(k) != "no-evaluator":
            continue
        fam = k[len(_NO_EVALUATOR_PREFIX):].strip()
        if not fam:
            continue
        d = sel if _is_token_select(fam) else other
        d[fam] = d.get(fam, 0) + n
    return sel, other


def _selector_no_evaluator_gaps(not_ev: dict) -> dict[str, int]:
    return _no_evaluator_gaps(not_ev)[0]


def _form_b_selector_coverage(oc: dict | None, n_sel: int) -> tuple[str | None, str]:
    """[R16 rev F-r16-20] The FAMILY-TARGETED form (B) test for selector outputs: the oracle family `sampled_token_ids` (the committed id ==
    the Match capture's SampleEnd id for that (step, request); every step, no snapshot needed) must be compared for EVERY (step, request)
    pair with 0 mismatch and the request attribution complete.  Not `_form_b_complete_over_values`: a stochastic row's oracle is value-partial
    on `instance_outputs` (the selector's own instance output has no fold instance) and that partiality is not about the sampled ids.
    Returns (None, 'compared K/K') when covered, else (the named reason, detail)."""
    if not isinstance(oc, dict):
        return "form (B) absent (no --match-dir oracle compare on the record)", ""
    if oc.get("result") is False:
        return f"form (B) result False ({oc.get('grade') or 'no grade'})" + (f": {str(oc.get('why'))[:120]}" if oc.get("why") else ""), ""
    if int(oc.get("mismatch_n") or 0) != 0:
        return f"form (B) {oc.get('mismatch_n')} mismatch", ""
    att = oc.get("attribution") or {}
    if att.get("complete") is not True:
        return "form (B) request attribution not complete", ""
    cbf, ncf = oc.get("compared_by_family") or {}, oc.get("not_compared_by_family") or {}
    n_cmp, n_not, pairs = int(cbf.get("sampled_token_ids") or 0), int(ncf.get("sampled_token_ids") or 0), int(att.get("pairs") or 0)
    detail = f"sampled_token_ids compared {n_cmp}/{pairs or n_sel} (step, request) pairs, not compared {n_not}, 0 mismatch"
    if n_cmp <= 0 or n_not > 0 or n_cmp < n_sel or (pairs > 0 and n_cmp != pairs):
        return f"form (B) does not cover the selector outputs ({detail}; {n_sel} selector VUs)", detail
    return None, detail


def _query_population_scope(sr: dict | None) -> tuple[str | None, str | None]:
    """[R17-3 / reqval C2 population audit] The population of record is Q(P) -- the required manifest -- never the replay's own enumeration
    of the Program rows.  `sampled_replay` reconciles its reach against Q(P) identity by identity and carries the result on the record
    (`population.query_total` / `sampled_strata` / `uncovered_by_name` / `query_checks`).  Returns (scope note for the PASS text, None) --
    the scope is ALWAYS stated on a PASS so no row reports 'qualified' over a narrowed population silently -- or (None, FAIL text) when the
    reconciliation itself fails: a VU whose address is not a required identity, a required identity no Program row produces, a block height
    that disagrees, a rank set that disagrees, a required family with no stated mechanism.  A replay record WITHOUT a manifest to
    reconcile against (`query_checks.ok` None) carries a note naming that; a record predating the fields is graded as before (no note --
    the rows of record before R17-3 keep their text).  The grade rules above stand either way."""
    if not isinstance(sr, dict):
        return None, None
    pop = sr.get("population") if isinstance(sr.get("population"), dict) else {}
    qt, chk = pop.get("query_total"), pop.get("query_checks")
    if not isinstance(qt, dict) or not isinstance(chk, dict):
        return None, None
    # [R17 reqval task 3] the one-release engine cross-check on the manifest of record: DISAGREED fails the row by name (the build refuses to
    # write such a manifest; a record carrying one anyway -- another path, an edited header -- does not PASS over it)
    cc = qt.get("cross_check")
    if isinstance(cc, dict) and cc.get("agree") is False:
        return None, ("required manifest engines DISAGREE (R17 task 3 cross-check: the traversal of record and the inventory beside it must agree "
                      f"row for row for one release) -- {str(cc.get('why') or cc)[:300]}")
    if chk.get("ok") is False:
        parts = []
        for k in ("vus_outside_query", "identities_without_rows", "rows_disagree"):
            c = chk.get(k) or {}
            if int(c.get("n") or 0) > 0:
                first = (c.get("first") or [{}])[0]
                parts.append(f"{k} x{c['n']} (first {first.get('request_id')} step {first.get('step')} {first.get('op_path')}/{first.get('member')})")
        rk = chk.get("ranks") or {}
        if rk.get("equal") is False:
            parts.append(f"ranks replay {rk.get('replay')} != manifest {rk.get('manifest')}")
        if chk.get("accounted") is False:
            parts.append("reached + uncovered != |Q(P)|")
        if chk.get("unstated_families"):
            parts.append(f"required families with no stated mechanism {chk['unstated_families']}")
        if chk.get("missing_acquisition_families"):
            parts.append(f"required families with NO acquisition mechanism {chk['missing_acquisition_families']}")
        # [R17-26/27, F-moe-r17-05, R4 c2] the newer checks are named too -- never 'see population.query_checks' for a check that has a name
        ep = chk.get("executed_prefix_rule") or {}
        if ep.get("faults"):
            parts.append(f"executed-prefix rule faults x{len(ep['faults'])} (first {str(ep['faults'][0])[:160]})")
        for k, label in (("evaluated_past_stop", "identities evaluated PAST a request's executed prefix"),
                         ("manifest_cut_before_classification", "manifest cut before the not-executed classification")):
            c = ep.get(k) or {}
            if int(c.get("n") or 0) > 0:
                parts.append(f"{label} x{c['n']} (first {str((c.get('first') or [''])[0])[:160]})")
        ua = chk.get("identities_unreachable_by_alias") or {}
        if int(ua.get("n") or 0) > 0:                                            # [F-r17b-38] the CAUSE of outside + without-rows, named first
            f0 = (ua.get("first") or [{}])[0]
            parts.insert(0, f"identities unreachable by alias x{ua['n']} (invocation >= 1 with no op_path_aliases entry; table has "
                            f"{ua.get('alias_entries')} entries; first {f0.get('op_path')} invocation {f0.get('invocation')} rank {f0.get('rank')}; F-r17b-38)")
        cd = chk.get("controller_rows_disagree") or {}
        if int(cd.get("n") or 0) > 0:
            f0 = (cd.get("first") or [{}])[0]
            parts.append(f"controller rows: header {f0.get('header_rows')} != request Programs' sum {f0.get('program_rows_sum')} "
                         f"{f0.get('program_rows_by_request')} (F-r17b-36 scope: sum over request Programs)")
        return None, ("C2 population does not reconcile with Q(P) (R17-3): the replay's enumeration of the Program rows and the required manifest "
                      f"disagree on what the Program produces -- {'; '.join(parts) or 'see population.query_checks'}")
    scope = str(pop.get("scope") or "")
    qid = qt.get("query_id") or "Q_module (pre-R17-10 header)"                   # [R17-10] the query of record, by id, on every PASS
    if chk.get("ok") is None:
        return f"C2 population scope: {scope} -- NOT reconciled against Q(P) (no required manifest supplied to the replay); the reach is the replay's own count", None
    return f"C2 population scope (R17-3; query of record {qid}): {scope}", None


def _complete_replay_population_gap(sr: dict | None, oc: dict | None) -> tuple[str | None, str | None]:
    """[R16 rev F-r16-02 / M-0016; COORD M-0721 (1)] The C2 rule when the sampled replay is COMPLETE but its POPULATION excluded VUs BY NAME
    under a MAN-06f / MAN-06g class (`population.not_evaluable`: VUs fed by MoE block-stream planes; VUs fed by an interior producer without an
    evaluator, e.g. Qwen2.5's BiasAdd over Gemm_v2).  Those VUs HAVE committed members the replay did not check -- the same gap the PARTIAL
    rule (`_partial_replay_named_gap`) carries on a PASS only when form (B) is complete over the values; moving the gap into the population
    must not lift that condition.  Returns (gap note for the PASS text, None) when form (B) is complete; (None, FAIL text) when it is not;
    (None, None) when the population carries no such gap.  The NOT_YET exclusions (no-evaluator: RoPE_v1; state-linked: Attention_v3) are
    the replay's stated applicability domain (R15 standard) and are not gated here.

    [R16 rev F-r16-20 / M-0400] a SELECTOR family with no registered evaluator (GumbelTopPTokenSelect_v1: not in NOT_YET, not in any ruling's
    domain) is a form-(B)-REQUIRED gap with a family-targeted test (`_form_b_selector_coverage`: every (step, request)'s sampled id compared
    by the oracle, 0 mismatch, attribution complete) -> PASS carrying the note, else FAIL by name: without it a stochastic Commit-only rerun
    (no --match-dir) PASSes with the sampled ids value-checked by NEITHER form (C1 / C3 bind them; nothing computes them)."""
    if not (isinstance(sr, dict) and sr.get("result") is True):
        return None, None
    not_ev = (sr.get("population") or {}).get("not_evaluable") or {}
    if not isinstance(not_ev, dict):
        return None, None
    gaps = {str(k): int(v) for k, v in not_ev.items() if int(v or 0) > 0 and _why_class_of(str(k)) in _POPULATION_GAP_CLASSES}
    sel, other = _no_evaluator_gaps(not_ev)
    # [pin F-r16-20b / M-0431] a non-selector family with no registered evaluator outside NOT_YET (Hopper Gemm_v2 / Attention_v2) joins the
    # gaps form (B) must cover over instance_outputs -- the direct output gap gets the SAME rule as the composed one (BiasAdd over Gemm_v2)
    for fam, cnt in other.items():
        gaps[f"{_NO_EVALUATOR_PREFIX}{fam}"] = cnt
    if not gaps and not sel:
        return None, None
    n = sum(gaps.values())
    cls = sorted({_why_class_of(k) for k in gaps})
    tags = sorted({t for k in gaps for t in ("MAN-06f", "MAN-06g") if t in k})
    reasons = " | ".join(f"{v} x {k.split(' [', 1)[0][:110]}" for k, v in sorted(gaps.items(), key=lambda kv: -kv[1]))
    missing = _form_b_complete_over_values(oc)
    # [R16 rev F-r16-07 / M-0068] plane-fed VUs are covered by neither form: FAIL by name, whatever form (B) says
    plane = {k: v for k, v in gaps.items() if _why_class_of(k) == "plane-fed"}
    if plane:
        n_p = sum(plane.values())
        rest = {k: v for k, v in gaps.items() if k not in plane}
        reasons_p = " | ".join(f"{v} x {k.split(' [', 1)[0][:110]}" for k, v in sorted(plane.items(), key=lambda kv: -kv[1]))
        tail = ""
        if rest:
            tail = (f"; the other {sum(rest.values())} VUs not evaluable by name ({', '.join(sorted({_why_class_of(k) for k in rest}))}) "
                    + ("are covered by the complete form (B)" if missing is None else f"are not covered either: {missing}"))
        return None, (f"C2 of record not established: {n_p} plane-fed VUs (MAN-06f) {_PLANE_FED_NEITHER_FORM} -- {reasons_p}{tail}; "
                      "C1 + C3 hold on those planes, C2 needs form (A) over the planes (MAN-06f v3 row->plane mapping) or a bit-exact plane xcheck on record")
    # [R16 rev F-r16-20 / M-0400] selector outputs with no evaluator: form (B) must value-check every sampled id (family-targeted), else FAIL
    sel_note = None
    if sel:
        n_s, fams_s = sum(sel.values()), ", ".join(f"{f} x{c}" for f, c in sorted(sel.items()))
        reason_s, detail_s = _form_b_selector_coverage(oc, n_s)
        if reason_s is not None:
            return None, (f"C2 of record not established (F-r16-20): {n_s} selector output VUs ({fams_s}) not replayed (no registered evaluator; "
                          f"the top-p split count S is not a committed identity until REQ-03) and {reason_s} -- the committed sampled ids are "
                          "value-checked by NEITHER form; C1 + C3 bind them, nothing computes them")
        sel_note = (f"selector outputs x{n_s} ({fams_s}) not replayed (no evaluator; S not committed until REQ-03), value-checked by form (B) "
                    f"[{detail_s}]")
    if not gaps:
        return sel_note, None
    if missing is not None:
        return None, (f"C2 of record not established (COORD M-0721 (1)): {n} VUs not evaluable by name ({', '.join(cls)}{'; ' + ', '.join(tags) if tags else ''}) "
                      f"have committed members the complete sampled replay did not check and {missing} -- {reasons}")
    cbf = oc.get("compared_by_family") if isinstance(oc, dict) else None
    return (f"C2 of record by form (B) [oracle compare complete: {oc.get('compared')} compared, 0 mismatch"
            f"{', instance outputs ' + str(cbf.get('instance_outputs')) if isinstance(cbf, dict) else ''}] over {n} VUs not evaluable by name "
            f"({', '.join(cls)}{'; ' + ', '.join(tags) if tags else ''}): {reasons}" + (f"; {sel_note}" if sel_note else "")), None


_REPLAY_SEED_ROOT_FORM = "run root (first 8 bytes, after commitment)"       # == commit_delta's [O6] default seed_source
_REPLAY_SEED_CHALLENGE_PREFIX = "verifier challenge "                        # == sampled_replay.challenge_seed's source text


def _replay_seed_of_record(sr: dict, run_root_hex: str | None) -> tuple[bool | None, str]:
    """[R16 rev F-r16-23 / M-0579] Is the sampled replay's SEED post-commitment?  The replay's adversarial meaning (C3: the picks are unknown
    before the bytes are committed) rests on the seed's source, which the record carries as a LABEL (`sample.seed_source`).  This reads it as
    a FACT: the root form is recomputed from the record's run root (sample.seed == int.from_bytes(run_root[:8])); the verifier-challenge form is
    post-commitment by construction (sha256(challenge|root)); `--replay-seed (targeted)` is PROVER-CHOSEN before the run -> a MECHANISM run,
    not a C2 of record.  Returns (True, how) / (False, why) / (None, why) when the record carries no seed source (legacy: not graded here)."""
    smp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    src = sr.get("seed_source") if sr.get("seed_source") is not None else smp.get("seed_source")
    seed = sr.get("seed") if sr.get("seed") is not None else smp.get("seed")
    if src is None:
        return None, "sampled replay seed source absent from the record"
    src = str(src)
    if src.startswith("--replay-seed"):
        return False, "sampled replay seed prover-chosen (--replay-seed): the picks were knowable before commitment -- a mechanism run, not a C2 of record"
    if src == _REPLAY_SEED_ROOT_FORM:
        if not run_root_hex or seed is None:
            return False, f"sampled replay seed source says '{src}' but the record carries no run root / seed to recompute it from"
        try:
            expect = int.from_bytes(bytes.fromhex(str(run_root_hex))[:8], "big")
        except ValueError:
            return False, f"sampled replay seed: run root {str(run_root_hex)[:16]}... is not hex"
        if int(seed) != expect:
            return False, f"sampled replay seed {seed} != the run root's first 8 bytes {expect} although seed_source says post-commitment root form"
        return True, "seed == run_root[:8] (recomputed)"
    if src.startswith(_REPLAY_SEED_CHALLENGE_PREFIX):
        # [rev M-0600] the challenge form is a FACT only with the challenge on the record: recompute sha256(challenge|run_root)[:8] and the
        # source text (sampled_replay.challenge_seed); a label without its challenge / another seed is the prover-chosen class (unverifiable)
        chal = smp.get("challenge")
        if chal is None:
            return False, f"sampled replay seed source claims a verifier challenge but the challenge is not on the record (sample.challenge): unverifiable"
        if not run_root_hex or seed is None:
            return False, f"sampled replay seed source says '{src[:40]}...' but the record carries no run root / seed to recompute it from"
        import hashlib
        try:
            root = bytes.fromhex(str(run_root_hex))
        except ValueError:
            return False, f"sampled replay seed: run root {str(run_root_hex)[:16]}... is not hex"
        h = hashlib.sha256(str(chal).encode("utf-8") + b"|" + root).digest()
        expect = int.from_bytes(h[:8], "big")
        want_src = f"{_REPLAY_SEED_CHALLENGE_PREFIX}{str(chal)[:16]!r} bound to the run root (sha256(challenge|root)[:8], post-commitment)"
        if int(seed) != expect:
            return False, f"sampled replay seed {seed} != sha256(challenge|run_root)[:8] = {expect} recomputed from the recorded challenge"
        if src != want_src:
            return False, f"sampled replay seed source text does not name the recorded challenge ({str(chal)[:16]!r})"
        return True, "seed == sha256(challenge|run_root)[:8] (recomputed from the recorded challenge)"
    return False, f"sampled replay seed source {src!r} is not a post-commitment form (run root / verifier challenge)"


def _partial_replay_named_gap(sr: dict | None, oc: dict | None) -> tuple[str | None, str | None]:
    """[COORD M-0721 (1); moe M-0715] The C2 rule when the sampled replay is PARTIAL.  Returns (NAMED GAP text, None) when the record still
    establishes C2 -- a partial replay with 0 mismatch whose every unevaluated stratum is named (family | module | reason) under a class form
    (B) can cover, AND form (B) (the Match-oracle compare) complete over the values with 0 mismatch and attribution complete; (None, FAIL text)
    when a named stratum is PLANE-FED (R16 rev F-r16-07: form (B) has no oracle family for the moe/ planes those VUs commit, so no complete
    (B) covers them); else (None, None) (the caller fails closed with the generic partial text).  A partial replay with an UNNAMED gap (an
    unevaluated count with no strata / reasons) never qualifies: that is what O1 forbids."""
    if not (isinstance(sr, dict) and sr.get("result") is None and str(sr.get("grade") or "") == "partial"):
        return None, None
    samp = sr.get("sample") or {}
    if int(samp.get("mismatch_n") or sr.get("mismatch_n") or 0) != 0 or int(samp.get("evaluated") or 0) <= 0:
        return None, None
    unev = sr.get("strata_unevaluated") or []
    if not unev or int(samp.get("not_evaluated_n") or 0) <= 0:
        return None, None
    # [rev-cb M-0740 G1] decide PER STRATUM from the uncapped `unevaluated_strata` (never from the capped not_evaluated[:12] sample): every
    # stratum in strata_unevaluated must carry an entry with spec, op_path and a reason; a stratum without one is an unnamed gap -> None
    by = {str(u.get("stratum")): u for u in (sr.get("unevaluated_strata") or []) if isinstance(u, dict)}
    if any(k not in by or not (by[k].get("spec") and by[k].get("op_path") and by[k].get("why")) for k in unev):
        return None, None
    # [rev-cb M-0740 G2] only BY-NAME classes qualify (no evaluator for the family / state-linked / plane-fed / interior-composed v2); an
    # evaluator fault, a row->member mapping fault, a missing committed input, a device fault never do: a fault is a run that did not
    # evaluate, not a gap on the record.  The class is read from the record when stamped, else derived from the reason text.
    classes = {k: sorted(set(by[k].get("classes") or [by[k].get("why_class") or _why_class_of(by[k].get("why"))])) for k in unev}
    bad = {k: c for k, c in classes.items() if any(x not in _WHY_BY_NAME for x in c)}
    if bad:
        return None, None
    # [R16 rev F-r16-07 / M-0068] a plane-fed stratum is covered by neither form -> FAIL by name (form (B) cannot lift it)
    plane = [k for k, c in classes.items() if "plane-fed" in c]
    if plane:
        n_p = sum(int(by[k].get("n") or 0) for k in plane)
        fams_p = sorted({str(by[k].get("spec") or "").split("{", 1)[0] for k in plane})
        whys_p = sorted({str(by[k].get("why") or "").split(" [", 1)[0][:90] for k in plane})[:3]
        return None, (f"C2 of record not established: {len(plane)} plane-fed strata ({n_p} VUs; MAN-06f) {_PLANE_FED_NEITHER_FORM}: "
                      f"{', '.join(fams_p)} -- {' | '.join(whys_p)}; sampled replay partial [{samp.get('evaluated')} evaluated, 0 mismatch, "
                      f"{len(unev)} strata named not evaluated]; C1 + C3 hold on those planes, C2 needs form (A) over the planes (MAN-06f v3) "
                      "or a bit-exact plane xcheck on record")
    if _form_b_complete_over_values(oc) is not None:
        return None, None
    cbf = oc.get("compared_by_family")
    fams = sorted({str(by[k].get("spec") or "").split("{", 1)[0] for k in unev})
    whys = sorted({str(by[k].get("why") or "").split(" [", 1)[0][:90] for k in unev})[:3]
    cls = sorted({x for c in classes.values() for x in c})
    return (f"C2 of record by form (B) [oracle compare complete: {oc.get('compared')} compared, 0 mismatch"
            f"{', instance outputs ' + str(cbf.get('instance_outputs')) if isinstance(cbf, dict) else ''}]; sampled replay partial "
            f"[{samp.get('evaluated')} evaluated, 0 mismatch, {len(unev)} strata named not evaluated ({', '.join(cls)}): {', '.join(fams)} -- "
            f"{' | '.join(whys)}]"), None


def _b2_attribution_note(bnd: dict, oc) -> str | None:
    """[R16 rev M-0243 / M-0259; rev-cb F-cb-23 / M-0216] at B>=2 the per-(step, request) ATTRIBUTION of every committed value rests on form (B):
    an equal-shape request swap relabels every row of a shared step together, so the committed record stays self-consistent and the sampled
    replay passes by construction; boundary linkage sees it only at the first swapped boundary and only when the two requests' tokens differ
    there.  Form (B) is not made mandatory here (R16 pin RULES; the verdict stays what the record establishes) -- but a PASS at B>=2 without a
    complete request attribution on the row's own record NAMES the gap, so no reader has to infer it from an absent block: (None) at B=1, or when
    the oracle compare's attribution is complete; else the note for the PASS text.  Value-partial by snapshot gaps (M-0217) is not this: the
    attribution block is complete there."""
    try:
        n_req = int((bnd.get("population") or {}).get("requests") or 0)
    except (TypeError, ValueError):
        return None
    if n_req < 2:
        return None
    if not isinstance(oc, dict):
        why = "form (B) not armed on this attempt (no --match-dir oracle compare: no Match record)"
    else:
        att = oc.get("attribution") if isinstance(oc.get("attribution"), dict) else {}
        if att.get("complete") is True:
            return None
        why = (f"form (B) request attribution not complete (pairs {att.get('pairs')}, unverified {att.get('unverified_n')}, "
               f"ambiguous steps {len(att.get('ambiguous_steps') or [])})")
    return (f"B={n_req}: per-(step,request) attribution not verified on this record -- {why}; the sampled replay + linkage cannot see a "
            "whole-step equal-shape request swap in general (F-cb-23, rev M-0243) -- a gap by name, not a verified attribution")


def commit_verdict(commit: dict, tokens_equal: bool | None, *, min_openings: int = 64, declaration: dict | None = None,
                   required_classes: tuple[str, ...] | list[str] | None = None) -> tuple[bool, str]:
    """X-03 fail-closed Commit verdict for ONE instrumented run.  PASS only if ALL hold: the required-value manifest's attempted
    acquisitions equal the bound ones (`expected_tensors == committed_tensors`, both > 0), no collector/arena/hook error was recorded,
    every opening checked verified and at least `min_openings` were checked, control vs instrumented tokens are equal, and (REL-02, when a
    `declaration` is given) every required class is declared and every declared leaf-binding class bound > 0 tensors.  Returns
    (ok, reason) with the FIRST failing reason; never 'PASS with errors recorded'."""
    ex = commit.get("extra") or {}
    c2_notes: list[str] = []                               # [COORD M-0721 (1)] named gaps carried on a PASS (form (B) covered a partial replay)
    n_exp, n_com = ex.get("expected_tensors"), ex.get("committed_tensors")
    errs_n = ex.get("errors_n")
    if errs_n is None:
        errs_n = len(ex.get("errors") or [])
    ops = commit.get("openings") or {}
    n_open, n_ok = int(ops.get("n") or 0), int(ops.get("verified") or 0)
    if n_exp is None or n_com is None:
        return False, "reference integrity counters missing (expected_tensors/committed_tensors): committer does not report them"
    if int(commit.get("fa2_zero_rows") or 0) > 0:
        return False, f"{commit['fa2_zero_rows']} FA2 hidden rows bound as all-zero bytes (tap staged after it was zeroed: ordering fault)"
    if int(n_exp) <= 0:
        return False, "expected_tensors == 0: the required-value manifest bound nothing"
    if int(n_exp) != int(n_com):
        return False, f"expected_tensors {n_exp} != committed_tensors {n_com} (refused/lost acquisitions -> incomplete root)"
    bnd = commit.get("binding")
    if bnd is not None:                                    # R14 BIND-01 (rules r14-v1): identity COVERAGE of the expected population is the gate
        ok, why = binding_check(bnd, min_openings=min_openings)
        if not ok:
            return False, why
    cov = ex.get("coverage") or commit.get("coverage")
    if cov is not None and not cov.get("ok", False):     # R14 MAN-01: identities REQUIRED by the Program/query manifest, not the collector's own attempts
        fam = cov.get("families_missing_entirely") or []
        first = (cov.get("missing") or [{}])[0]
        if cov.get("error") and not cov.get("missing_n"):   # [R17-1 / R17-27] a named coverage error with nothing uncovered (prefix, reconciliation): the name IS the reason
            return False, f"manifest coverage: {cov['error']}"
        return False, (f"manifest coverage: {cov.get('missing_n')} required identities uncovered ({cov.get('missing_by_family')}); "
                       + (f"whole families missing: {fam}; " if fam else "") + f"first: step {first.get('step')} {first.get('op_path')}/{first.get('output_member')} inv {first.get('invocation')} want {first.get('want_range')} got {first.get('committed_numel')}")
    idc = ex.get("identity_check")
    if idc is not None and not idc.get("ok", False):     # R13: identities, not counts -- missing-one-and-duplicating-another fails here
        v = (idc.get("violations") or [{}])[0]
        return False, f"identity check: {idc.get('violations_n')} step(s) differ from the expected manifest (first: step {v.get('step')} {json_short(v)})"
    if int(errs_n) > 0:
        first = (ex.get("errors") or ["?"])[0]
        return False, f"{errs_n} collector error(s): {str(first)[:160]}"
    if declaration is not None or required_classes is not None:
        ok, why = required_classes_check(declaration, required_classes if required_classes is not None else REQUIRED_CLASSES_DEFAULT)
        if not ok:
            return False, why
    oar = commit.get("openings_after_release")
    # [R17 fp8] a bounded-staging record run must not carry a learned-only step (a placeholder root from the learn-only warm-up path)
    _bs = ex.get("bounded_staging") if isinstance(ex.get("bounded_staging"), dict) else None
    if _bs and _bs.get("learned_only_steps"):
        return False, f"bounded staging: {len(_bs['learned_only_steps'])} learned-only step(s) {list(_bs['learned_only_steps'])[:8]} on a record run (no root)"
    if _bs and _bs.get("learn_only"):
        return False, "bounded staging: the record run was finalized in learn-only mode"
    if ops.get("skipped") and oar is None:    # retain=none: the raw bytes never existed off-device -> the ONLY opening path is replay (after release)
        return False, f"openings skipped ({ops.get('skipped')}) and no openings-after-release (replay) record"
    if not ops.get("skipped"):
        if n_open < min_openings:
            return False, f"openings checked {n_open} < {min_openings}"
        if n_ok != n_open:
            return False, f"openings verified {n_ok} != checked {n_open}"
    elif n_ok != n_open:                      # [R17 fp8] retain=host + exclude: the host share of the in-run openings must still verify
        return False, f"openings (host share) verified {n_ok} != checked {n_open}"
    if tokens_equal is not True:
        return False, "control vs instrumented tokens differ"
    vc = commit.get("value_correspondence")
    if isinstance(vc, dict) and vc.get("result") is False:
        # [R15 COORD M-0111 (2)] LIVE C2: the committed bytes of an identity differ from the Match oracle's value (or the reference root):
        # a relabel / same-name byte swap / late read that C1 (coverage) and C3 (openings) cannot see.  Named, fail closed.
        # [rev-cb M-0301 R1] name the FAILING component's own cause: the oracle's `why` only when the ORACLE failed (a partial-coverage note
        # on a None oracle result is not the cause of a root-compare failure); else the composite's `why` (root_compare / replay), else generic.
        oc = vc.get("oracle_compare") or {}
        oc_top = commit.get("oracle_compare") if isinstance(commit.get("oracle_compare"), dict) else {}
        comps = vc.get("components") or {}
        failing = [(k, c) for k, c in comps.items() if isinstance(c, dict) and c.get("result") is False]
        if oc.get("result") is False and oc.get("why"):
            return False, oc["why"]
        if oc_top.get("result") is False and (oc_top.get("why") or oc.get("why")):
            return False, oc_top.get("why") or oc.get("why")
        if failing:
            return False, "; ".join(f"{k}: {c.get('why') or 'FAIL'}" for k, c in failing)
        return False, (vc.get("why") or f"value correspondence FAIL ({vc.get('method')})")
    oc = commit.get("oracle_compare")
    if isinstance(oc, dict) and oc.get("result") is False:
        return False, oc.get("why") or f"C2 oracle mismatch: {oc.get('mismatch_n')} identities differ from the Match oracle"
    # [DECISIONS-2 §4 C2 / M-0356 (1); rev-cb M-0411 O1] C2 OF RECORD = sampled exact VU replay from the committed bytes (registered relation
    # evaluators on the Program's dataflow) + exhaustive ordered boundary linkage (declared prompt -> input ids at the arrival step; sampled
    # id[s-1] -> input id[s]).  Required at EVERY B for a row-procedure record (one carrying the binding block): a mismatch / an unlinked
    # boundary is a named FAIL; ABSENCE or a partial replay is a named FAIL too (never an unsound PASS on C1 + C3 alone, F-cb-02 / M-0106).
    # The linkage is not a B>=2 device: at B=1 the arrival-step check ties the Commit to the workload of record (the wl_swapped canary) and
    # the decode chain is what a cross-step relabel breaks.  The full Match-oracle compare is a diagnostic: a mismatch still fails (above),
    # a complete attribution no longer substitutes for the replay.  Records without a binding block (unit fixtures of other checks) keep
    # the legacy rule.  TP2 (tp2_commit grades the rank block before attaching its binding) is graded by tp's path until the rank replay lands.
    sr = commit.get("sampled_replay")
    lk = commit.get("boundary_linkage")
    # [rel M-0621] a producer that assembles its C2 under value_correspondence.components (compiled S3's kernel replay) is graded by the
    # SAME gate: the components are read where the top level is absent, and only when the component says it IS a sampled exact replay
    # (method / schema); a component of another method never stands in for the replay of record.
    _comps = (commit.get("value_correspondence") or {}).get("components") if isinstance(commit.get("value_correspondence"), dict) else None
    if not isinstance(sr, dict) and isinstance(_comps, dict) and isinstance(_comps.get("sampled_replay"), dict):
        _c = _comps["sampled_replay"]
        if "sampled-exact-replay" in str(_c.get("method") or _c.get("schema") or _c.get("reference") or ""):
            sr = _c
    if not isinstance(lk, dict) and isinstance(_comps, dict) and isinstance(_comps.get("boundary_linkage"), dict):
        lk = _comps["boundary_linkage"]
    if isinstance(sr, dict) and sr.get("result") is False:
        return False, f"C2 sampled replay: {sr.get('why') or 'a sampled VU does not replay from its committed inputs'}"
    if isinstance(lk, dict) and lk.get("result") is False:
        return False, f"C2 boundary linkage: {lk.get('why') or 'an (engine step, request) boundary is not linked'} ({lk.get('unlinked_n')} unlinked)"
    # [MAN-09; rev-cb M-0610, coord M-0614 (1)] WEIGHTS OF RECORD: the registered inputs must be authenticated by their canonical reference --
    # the pinned checkpoint shards -- not by re-hashing the committer's own parameters (O5 is self-consistency).  Every served parameter the
    # Program names is compared by digest with the checkpoint-composed tensor of record: a difference (rev-cb flipweight) or a missing field
    # is a named FAIL; on a row-procedure record the pin is REQUIRED (absent / not evaluated = 'weights unpinned', a named FAIL, like O1).
    wp = commit.get("weights_pin")
    if isinstance(wp, dict) and wp.get("result") is False:
        return False, f"weights of record: {wp.get('why') or 'served weights differ from the checkpoint of record'}"
    if isinstance(bnd, dict) and (bnd.get("population") or bnd.get("coverage")):
        _oc = oc if isinstance(oc, dict) else ((vc.get("oracle_compare") if isinstance(vc, dict) else None) or
                                               (_comps.get("oracle_compare") if isinstance(_comps, dict) else None))
        # [R16 rev F-r16-23 / M-0579] the replay's seed must be POST-COMMITMENT as a fact, not a label: a prover-chosen --replay-seed (or a
        # root-form label whose seed is not the run root's) makes the replay a mechanism run -- C2 of record then rests on form (B) alone
        # (M-0721 (1)) and the PASS names it; without a complete form (B) the Commit FAILS by name.  Legacy records without a seed source
        # are graded by the finalizer (report), not here.
        _mech = False
        if isinstance(sr, dict) and sr.get("result") is not False:
            _sok, _swhy = _replay_seed_of_record(sr, commit.get("run_root"))
            if _sok is False:
                _missing_b = _form_b_complete_over_values(_oc)
                if _missing_b is not None:
                    return False, f"C2 of record not established (F-r16-23): {_swhy}; and form (B) does not stand in: {_missing_b}"
                # form (B) never covers plane-fed VUs (F-r16-07) -- the population / stratum gaps are still named whatever the seed
                _gap0, _fail0 = _complete_replay_population_gap(sr, _oc) if sr.get("result") is True else _partial_replay_named_gap(sr, _oc)
                if _fail0 is not None:
                    return False, _fail0
                if sr.get("result") is not True and _gap0 is None:
                    # [R16 rev F-r16-26 / rel M-0686] the F-r16-13 rule runs BEFORE the mechanism lift: a PARTIAL replay with no NAMED gap (an
                    # unevaluated pick of FAULT class inside an evaluated stratum -- 'a fault is a run that did not evaluate, not a gap') fails
                    # closed whatever seeded the replay; form (B) stands in for a named gap, never for a fault
                    sr_why = sr.get("why") or sr.get("grade")
                    return False, (f"C2 of record not established (DECISIONS-2 §4): sampled replay {sr.get('grade')} -- {sr_why}; "
                                   f"the row's Commit is C1 + C3 only (F-r16-13, whatever seeded the replay: {_swhy})")
                c2_notes.append(f"{_swhy}; C2 of record by the complete form (B) alone")
                _mech = True
        if _mech:
            pass
        elif not (isinstance(sr, dict) and sr.get("result") is True):
            # [COORD M-0721 (1), moe M-0715] C2 of record holds when EITHER form is complete (M-0481 'any ONE suffices'): a PARTIAL replay
            # with 0 MISMATCH whose every unevaluated stratum is NAMED (family | module | reason) is a GAP on the record, not a failure,
            # when form (B) -- the Match-oracle compare -- is complete (0 mismatch, attribution complete) over the values.  A replay
            # mismatch always fails (above); the absence of both forms still fails (O1 stands).
            _gap, _fail = _partial_replay_named_gap(sr, _oc)
            if _fail is not None:                                    # [R16 rev F-r16-07] a plane-fed stratum: neither form covers it, named
                return False, _fail
            if _gap is None:
                sr_why = (sr.get("why") or sr.get("grade")) if isinstance(sr, dict) else "not run (no --program-dir / --programs-root)"
                return False, (f"C2 of record not established (DECISIONS-2 §4): sampled replay {sr.get('grade') if isinstance(sr, dict) else None} -- {sr_why}; "
                               "the row's Commit is C1 + C3 only")
            c2_notes.append(_gap)
        else:
            # [R16 rev F-r16-02 / M-0016] a COMPLETE replay whose population excluded VUs BY NAME under MAN-06f / MAN-06g (plane-fed,
            # interior-composed v2) carries the same gap the partial rule does: those VUs have committed members no form checked unless
            # form (B) is complete over the values -- the population move (rev-cb M-0751 (b)) keeps the M-0721 (1) condition, it does not lift it.
            _gap, _fail = _complete_replay_population_gap(sr, _oc)
            if _fail is not None:
                return False, _fail
            if _gap is not None:
                c2_notes.append(_gap)
        if not (isinstance(lk, dict) and lk.get("result") is True):
            lk_why = lk.get("why") if isinstance(lk, dict) else "not run"
            return False, f"C2 of record not established (DECISIONS-2 §4): boundary linkage {lk.get('result') if isinstance(lk, dict) else None} -- {lk_why}"
        if not (isinstance(wp, dict) and wp.get("result") is True):
            wp_why = wp.get("why") if isinstance(wp, dict) else "not run (no --weights-of-record / --checkpoint-dir)"
            return False, (f"weights unpinned (MAN-09): the served weights were not compared with the checkpoint of record -- {wp_why}; "
                           "the committed weights root is self-consistency, not authentication")
        _att = _b2_attribution_note(bnd, _oc)
        if _att is not None:
            c2_notes.append(_att)
        # [R17-3 / reqval] the population of record is Q(P): the replay's reach reconciled against the required manifest is stated on
        # every PASS (query_total / uncovered_by_name), and a reconciliation failure is a named FAIL -- never a silent narrowing
        _scope, _scope_fail = _query_population_scope(sr)
        if _scope_fail is not None:
            return False, _scope_fail
        if _scope is not None:
            c2_notes.append(_scope)
    if oar is not None:                      # R13 P0: openings after the raw device staging was released must verify against the ORIGINAL roots
        n_a, ok_a = int(oar.get("n") or 0), int(oar.get("verified") or 0)
        if n_a < min_openings:
            return False, f"openings after release checked {n_a} < {min_openings}"
        if ok_a != n_a:
            return False, f"openings after release verified {ok_a} != checked {n_a} (source {oar.get('source')})"
        if "replay" in str(oar.get("source") or "") and oar.get("replay_root_equal") is not True:   # replay | host+replay (R17-5)
            return False, "replay run root differs from the original run root (replay-to-open is not reproducing the committed bytes)"
        if "replay" in str(oar.get("source") or "") and int(oar.get("replay_errors_n") or 0) > 0:
            return False, f"replay run recorded {oar.get('replay_errors_n')} collector error(s)"
        if int(oar.get("retained_device_after") or 0) > 0:
            return False, f"raw/level bytes still on the device after release: {oar.get('retained_device_after')}"
        # [R17 fp8, R17-5] the challenge / population picks the in-run stages could not open from host (excluded families) were handed to
        # this stage: every one of them must have been opened here by replay, and verified (ok_a == n_a above covers the verification)
        _po = (bnd or {}).get("openings_population") or {}
        _def_a, _def_b = int(ops.get("deferred_replay_n") or 0), int(_po.get("deferred_replay_n") or 0)
        if _def_a or _def_b:
            # [rev F-r17b-13] by POSITION when the record carries them: the union of both stages' deferred (step, leaf) sets must be a subset
            # of the after-release stage's received set; count-only records (older) fall back to received >= max(a, b)
            _pa = [tuple(p) for p in (ops.get("deferred_positions") or [])]
            _pb = [tuple(p) for p in (_po.get("deferred_positions") or [])]
            _got = {tuple(p) for p in (oar.get("deferred_received_positions") or [])}
            # [revb S-r17b-03] the count-only branch is for records written BEFORE positions existed: a writer that knows positions always
            # writes the keys (empty lists included), so their presence gates the branch -- it cannot be reached by omitting the lists
            _positional_writer = any(k in d for d, k in ((ops, "deferred_positions"), (_po, "deferred_positions"), (oar, "deferred_received_positions")))
            if _pa or _pb or _got or _positional_writer:
                if len(_pa) != _def_a or len(_pb) != _def_b:
                    return False, (f"deferred picks recorded by count ({_def_a} challenge, {_def_b} population) but their positions are not on the "
                                   f"record ({len(_pa)}, {len(_pb)}) -- the after-release stage cannot be checked against them")
                _miss = sorted((set(_pa) | set(_pb)) - _got)
                if _miss:
                    return False, (f"{_def_a} challenge + {_def_b} population pick(s) deferred to replay; {len(_miss)} position(s) not received by "
                                   f"openings after release (first: {_miss[:4]}) -- every deferred pick must be opened by replay")
            elif int(oar.get("deferred_received_n") or 0) < max(_def_a, _def_b):
                return False, (f"{_def_a} challenge + {_def_b} population pick(s) deferred to replay, openings after release received "
                               f"{oar.get('deferred_received_n')} (every deferred pick must be opened by replay)")
    elif int(ops.get("deferred_replay_n") or 0) or int(((bnd or {}).get("openings_population") or {}).get("deferred_replay_n") or 0):
        return False, "picks deferred to replay (excluded families) and no openings-after-release record"
    if c2_notes:
        return True, "PASS (" + "; ".join(c2_notes) + ")"
    return True, "PASS"
