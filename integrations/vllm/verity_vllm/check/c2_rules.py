"""The C2 rules of one instrumented run's Commit record (local replay, boundary linkage, weights of record) and its execution
extent, each a `CheckResult`, with the helpers the decision (`verdict.commit_verdict`) shares: form (B) coverage of the replay's named
gaps, the seed of record, the population scope against Q(P).  The cuBLAS route pin's rule reads the section `check.cublas_route_pin`
stamps on the record.

The rules decide from the structured codes the sampled replay stamps on its record (`replay_codes`: the class / family / domain of
every not-evaluable reason, the classes of every unevaluated stratum, the seed form), never from its message texts."""
from __future__ import annotations

from typing import Any

from verity_vllm.check.commit_rules import _binding_of, _d, _first, _is_row_procedure, _leak, _linkage_of, _mk, _oracle_of, _replay_of
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.check.result import FAIL, INSUFFICIENT, NOT_RUN, PASS, CheckResult
from verity_vllm.query.manifest.format import is_token_select
from verity_vllm.records.fields import RecordError
from verity_vllm.records.query_checks import QueryChecks

QUERY_CHECKS = "the sampled replay's population.query_checks"

_POPULATION_GAP_CLASSES = ("plane-fed", "interior-composed-v2")     # the population's by-name classes
# which by-name classes form (B) CAN cover: form (B) compares the Match record's dispatcher-level
# module outputs (`instance_outputs`), so an interior-composed VU (its module's returned tensor IS an instance output) is covered by a
# complete (B); a PLANE-FED VU is not -- its committed members are the moe/ block-stream planes, and oracle_compare has NO oracle
# family for `moe/` (NO_ORACLE_FAMILY_PREFIXES: counted 'no oracle family', excluded from compared).  Neither form value-checks those planes
# (form (A) has no row->plane mapping in this replay), so a plane-fed gap FAILs by name whatever form (B) says; C1 + C3 still hold on them.
_PLANE_FED_NEITHER_FORM = ("have committed moe/ plane members value-checked by NEITHER form (form (B) has no oracle family for moe/ planes "
                           "[oracle_compare.NO_ORACLE_FAMILY_PREFIXES]; form (A) has no row->plane mapping in this replay)")


def _form_b_complete_over_values(oc: dict | None) -> str | None:
    """None when form (B) -- the Match-oracle compare -- is COMPLETE over the values: result True, 0
    mismatch, complete (every step snapshotted), compared > 0, attribution complete, and the module outputs (`instance_outputs`, the family
    the gap strata produce) compared > 0 with none left not-compared (weights / hidden_stream have no oracle family and never stand in).
    Else the named reason it is not; a form (B) that is never complete says why in its `scope` (a TP Commit's, `pipeline.tp.c2.tp_form_b`)."""
    if not isinstance(oc, dict):
        return "form (B) absent (no --match-dir oracle compare on the record)"
    if oc.get("result") is not True:
        return f"form (B) result {oc.get('result')} ({oc.get('grade') or 'no grade'})" + (f": {str(oc.get('why'))[:120]}" if oc.get("why") else "")
    if int(oc.get("mismatch_n") or 0) != 0:
        return f"form (B) {oc.get('mismatch_n')} mismatch"
    if oc.get("complete") is not True and oc.get("scope"):
        return f"form (B) not complete over the values ({oc['scope']})"
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


def _no_evaluator_reasons(not_ev: dict, codes: dict[str, dict]) -> list[tuple[str, str, int]]:
    """(reason, family, n) for every counted `no-evaluator` reason whose family is outside the replay's stated domain (NOT_YET)."""
    out = []
    for k, v in not_ev.items():
        c, n = codes.get(str(k)) or {}, int(v or 0)
        if n > 0 and c.get("class") == "no-evaluator" and not c.get("domain") and c.get("family"):
            out.append((str(k), str(c["family"]), n))
    return out


def _no_evaluator_gaps(not_ev: dict, codes: dict[str, dict] | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """({selector family: n}, {other family: n}) among the population's no-evaluator
    exclusions whose family is NOT in NOT_YET (`codes`: the reasons' structured codes, default decoded from `not_ev`).  Selectors
    (TokenSelect families, e.g. GumbelTopPTokenSelect_v1 on a stochastic row): the sampled id is THE value a stochastic row adds and is
    reproducible from the committed logits only with the seed AND the top-p split count S (REQ-03).  Others (Hopper:
    Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1} = every linear, Attention_v2 = every attention row -- targets.gemm_spec / attention_spec bind
    them on a cc 9.0 step; neither is in EVALUATORS nor NOT_YET): their module outputs ARE the oracle's `instance_outputs`, so a complete
    form (B) covers them exactly as it covers the interior-composed class.  Both carry the SAME class ('no-evaluator') as the RoPE_v1
    applicability domain and must not hide behind it."""
    codes = RC.not_evaluable_codes({"not_evaluable": not_ev}) if codes is None else codes
    sel: dict[str, int] = {}
    other: dict[str, int] = {}
    for _k, fam, n in _no_evaluator_reasons(not_ev, codes):
        d = sel if is_token_select(fam) else other
        d[fam] = d.get(fam, 0) + n
    return sel, other


def _form_b_selector_coverage(oc: dict | None, n_sel: int) -> tuple[str | None, str]:
    """The FAMILY-TARGETED form (B) test for selector outputs: the oracle family `sampled_token_ids` (the committed id ==
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
    """The population of record is Q(P) -- the required manifest -- never the replay's own enumeration
    of the Program rows.  `sampled_replay` reconciles its reach against Q(P) identity by identity and carries the result on the record
    (`population.query_total` / `sampled_strata` / `uncovered_by_name` / `query_checks`).  Returns (scope note for the PASS text, None) --
    the scope is ALWAYS stated on a PASS so no row reports 'qualified' over a narrowed population silently -- or (None, FAIL text) when the
    reconciliation itself fails: a VU whose address is not a required identity, a required identity no Program row produces, a block height
    that disagrees, a rank set that disagrees, a required family with no stated mechanism.  A replay record WITHOUT a manifest to
    reconcile against (`query_checks.ok` None) carries a note naming that; a record predating the fields is graded as before (no note --
    rows recorded before the fields keep their text).  A `query_checks` section missing a field its writers all write FAILs, naming the
    field (`records.query_checks`).  The grade rules above stand either way."""
    if not isinstance(sr, dict):
        return None, None
    pop = sr.get("population") if isinstance(sr.get("population"), dict) else {}
    qt, chk = pop.get("query_total"), pop.get("query_checks")
    if not isinstance(qt, dict) or not isinstance(chk, dict):
        return None, None
    # the one-release engine cross-check on the manifest of record: DISAGREED fails the row by name (the build refuses to
    # write such a manifest; a record carrying one anyway -- another path, an edited header -- does not PASS over it)
    cc = qt.get("cross_check")
    if isinstance(cc, dict) and cc.get("agree") is False:
        return None, ("required manifest engines DISAGREE (the cross-check: the traversal of record and the inventory beside it must agree "
                      f"row for row for one release) -- {str(cc.get('why') or cc)[:300]}")
    try:
        c = QueryChecks.of(chk, QUERY_CHECKS)
    except RecordError as e:
        return None, f"C2 population reconciliation refused, nothing graded ({e})"
    if c.ok is False:
        parts = []
        if c.manifest_refused is not None:
            parts.append(f"required manifest refused, nothing reconciled ({c.manifest_refused.get('why')})")
        for k, f in (("vus_outside_query", c.vus_outside_query), ("identities_without_rows", c.identities_without_rows), ("rows_disagree", c.rows_disagree)):
            if f.n > 0:
                first = f.first[0] if f.first else {}
                parts.append(f"{k} x{f.n} (first {first.get('request_id')} step {first.get('step')} {first.get('op_path')}/{first.get('member')})")
        if c.ranks.equal is False:
            parts.append(f"ranks replay {c.ranks.replay} != manifest {c.ranks.manifest}")
        if c.accounted is False:
            parts.append("reached + uncovered != |Q(P)|")
        if c.unstated_families:
            parts.append(f"required families with no stated mechanism {c.unstated_families}")
        if c.missing_acquisition_families:
            parts.append(f"required families with NO acquisition mechanism {c.missing_acquisition_families}")
        # the newer checks are named too -- never 'see population.query_checks' for a check that has a name
        ep = c.executed_prefix_rule
        if ep is not None and ep.faults:
            parts.append(f"executed-prefix rule faults x{len(ep.faults)} (first {str(ep.faults[0])[:160]})")
        if ep is not None:
            for label, f in (("identities evaluated PAST a request's executed prefix", ep.evaluated_past_stop),
                             ("manifest cut before the not-executed classification", ep.manifest_cut_before_classification)):
                if f.n > 0:
                    parts.append(f"{label} x{f.n} (first {str(f.first[0] if f.first else '')[:160]})")
        ua = c.identities_unreachable_by_alias
        if ua is not None and ua.n > 0:                                          # the CAUSE of outside + without-rows, named first
            f0 = ua.first[0] if ua.first else {}
            parts.insert(0, f"identities unreachable by alias x{ua.n} (invocation >= 1 with no op_path_aliases entry; table has "
                            f"{ua.alias_entries} entries; first {f0.get('op_path')} invocation {f0.get('invocation')} rank {f0.get('rank')})")
        cd = c.controller_rows_disagree
        if cd is not None and cd.n > 0:
            f0 = cd.first[0] if cd.first else {}
            parts.append(f"controller rows: header {f0.get('header_rows')} != request Programs' sum {f0.get('program_rows_sum')} "
                         f"{f0.get('program_rows_by_request')} (scope: sum over request Programs)")
        return None, ("C2 population does not reconcile with Q(P): the replay's enumeration of the Program rows and the required manifest "
                      f"disagree on what the Program produces -- {'; '.join(parts) or 'see population.query_checks'}")
    scope = str(pop.get("scope") or "")
    qid = qt.get("query_id") or "Q_module (older header)"                   # the query of record, by id, on every PASS
    if c.ok is None:
        return f"C2 population scope: {scope} -- NOT reconciled against Q(P) (no required manifest supplied to the replay); the reach is the replay's own count", None
    return f"C2 population scope (query of record {qid}): {scope}", None


def _query_checks(sr: dict) -> QueryChecks | None:
    """The replay's `population.query_checks` as its record; None when the record carries none, or one `_query_population_scope`
    refuses (that refusal is the FAIL)."""
    chk = _d(sr.get("population")).get("query_checks")
    try:
        return QueryChecks.of(chk, QUERY_CHECKS) if isinstance(chk, dict) else None
    except RecordError:
        return None


def _faulted_prefix_facts(sr: dict) -> list[dict]:
    """The executed-prefix facts of record (`executed_prefix_of_record.facts_of_record`, what `sampled_replay.executed_prefix_of_record`
    was called with) of each request the replay's executed-prefix rule faulted, in the rule's order.  A fault's cause is read from these
    fields -- `account_missing` (a Match account leg not beside the Commit, the fault named first), `linked` -- never from its text."""
    facts = _d(_d(sr.get("executed_prefix_of_record")).get("facts_of_record"))
    c = _query_checks(sr)
    rule = c.executed_prefix_rule if c is not None else None
    return [_d(facts.get(str(f.get("request_id")))) for f in (rule.faults if rule is not None else ())]


def _scope_fail_account_missing(sr: dict) -> bool:
    """`_query_population_scope`'s FAIL is a missing input: the reconciliation's FAIL (not the engine cross-check's) whose first
    executed-prefix fault is a request missing a Match account leg."""
    pop, c, pf = _d(sr.get("population")), _query_checks(sr), _faulted_prefix_facts(sr)
    return (c is not None and c.ok is False and _d(_d(pop.get("query_total")).get("cross_check")).get("agree") is not False
            and bool(pf) and bool(pf[0].get("account_missing")))


def _complete_replay_population_gap(sr: dict | None, oc: dict | None) -> tuple[str | None, str | None]:
    """The C2 rule when the sampled replay is COMPLETE but its POPULATION excluded VUs BY NAME
    under a plane-fed or interior-composed class (`population.not_evaluable`: VUs fed by MoE block-stream planes; VUs fed by an interior producer without an
    evaluator, e.g. Qwen2.5's BiasAdd over Gemm_v2).  Those VUs HAVE committed members the replay did not check -- the same gap the PARTIAL
    rule (`_partial_replay_named_gap`) carries on a PASS only when form (B) is complete over the values; moving the gap into the population
    must not lift that condition.  Returns (gap note for the PASS text, None) when form (B) is complete; (None, FAIL text) when it is not;
    (None, None) when the population carries no such gap.  The NOT_YET exclusions (no-evaluator: RoPE_v1; state-linked: Attention_v3) are
    the replay's stated applicability domain and are not gated here.

    A SELECTOR family with no registered evaluator (GumbelTopPTokenSelect_v1: not in NOT_YET, not in any exclusion's
    domain) is a form-(B)-REQUIRED gap with a family-targeted test (`_form_b_selector_coverage`: every (step, request)'s sampled id compared
    by the oracle, 0 mismatch, attribution complete) -> PASS carrying the note, else FAIL by name: without it a stochastic Commit-only rerun
    (no --match-dir) PASSes with the sampled ids value-checked by NEITHER form (C1 / C3 bind them; nothing computes them)."""
    if not (isinstance(sr, dict) and sr.get("result") is True):
        return None, None
    pop = sr.get("population") or {}
    not_ev = pop.get("not_evaluable") or {}
    if not isinstance(not_ev, dict):
        return None, None
    codes = RC.not_evaluable_codes(pop)

    def cls_of(k: str) -> str | None:
        return (codes.get(k) or {}).get("class")

    gaps = {str(k): int(v) for k, v in not_ev.items() if int(v or 0) > 0 and cls_of(str(k)) in _POPULATION_GAP_CLASSES}
    sel = _no_evaluator_gaps(not_ev, codes)[0]
    # a non-selector family with no registered evaluator outside NOT_YET (Hopper Gemm_v2 / Attention_v2) joins the
    # gaps form (B) must cover over instance_outputs -- the direct output gap gets the SAME rule as the composed one (BiasAdd over Gemm_v2)
    for k, fam, cnt in _no_evaluator_reasons(not_ev, codes):
        if not is_token_select(fam):
            gaps[k] = cnt
    if not gaps and not sel:
        return None, None
    n = sum(gaps.values())
    cls = sorted({cls_of(k) for k in gaps})
    tags = sorted({t for k in gaps for t in (codes.get(k) or {}).get("tags") or ()})
    reasons = " | ".join(f"{v} x {k.split(' [', 1)[0][:110]}" for k, v in sorted(gaps.items(), key=lambda kv: -kv[1]))
    missing = _form_b_complete_over_values(oc)
    # plane-fed VUs are covered by neither form: FAIL by name, whatever form (B) says
    plane = {k: v for k, v in gaps.items() if cls_of(k) == "plane-fed"}
    if plane:
        n_p = sum(plane.values())
        rest = {k: v for k, v in gaps.items() if k not in plane}
        reasons_p = " | ".join(f"{v} x {k.split(' [', 1)[0][:110]}" for k, v in sorted(plane.items(), key=lambda kv: -kv[1]))
        tail = ""
        if rest:
            tail = (f"; the other {sum(rest.values())} VUs not evaluable by name ({', '.join(sorted({cls_of(k) for k in rest}))}) "
                    + ("are covered by the complete form (B)" if missing is None else f"are not covered either: {missing}"))
        return None, (f"C2 of record not established: {n_p} plane-fed VUs {_PLANE_FED_NEITHER_FORM} -- {reasons_p}{tail}; "
                      "C1 + C3 hold on those planes, C2 needs form (A) over the planes (a row->plane mapping) or a bit-exact plane xcheck on record")
    # selector outputs with no evaluator: form (B) must value-check every sampled id (family-targeted), else FAIL
    sel_note = None
    if sel:
        n_s, fams_s = sum(sel.values()), ", ".join(f"{f} x{c}" for f, c in sorted(sel.items()))
        reason_s, detail_s = _form_b_selector_coverage(oc, n_s)
        if reason_s is not None:
            return None, (f"C2 of record not established: {n_s} selector output VUs ({fams_s}) not replayed (no registered evaluator; "
                          f"the top-p split count S is not a committed identity until REQ-03) and {reason_s} -- the committed sampled ids are "
                          "value-checked by NEITHER form; C1 + C3 bind them, nothing computes them")
        sel_note = (f"selector outputs x{n_s} ({fams_s}) not replayed (no evaluator; S not committed until REQ-03), value-checked by form (B) "
                    f"[{detail_s}]")
    if not gaps:
        return sel_note, None
    if missing is not None:
        return None, (f"C2 of record not established: {n} VUs not evaluable by name ({', '.join(cls)}{'; ' + ', '.join(tags) if tags else ''}) "
                      f"have committed members the complete sampled replay did not check and {missing} -- {reasons}")
    cbf = oc.get("compared_by_family") if isinstance(oc, dict) else None
    return (f"C2 of record by form (B) [oracle compare complete: {oc.get('compared')} compared, 0 mismatch"
            f"{', instance outputs ' + str(cbf.get('instance_outputs')) if isinstance(cbf, dict) else ''}] over {n} VUs not evaluable by name "
            f"({', '.join(cls)}{'; ' + ', '.join(tags) if tags else ''}): {reasons}" + (f"; {sel_note}" if sel_note else "")), None


def _replay_seed_of_record(sr: dict, run_root_hex: str | None) -> tuple[bool | None, str]:
    """Is the sampled replay's SEED post-commitment?  The replay's adversarial meaning (C3: the picks are unknown
    before the bytes are committed) rests on the seed's form (`sample.seed_form`, `replay_codes.seed_form`).  This reads it as a FACT: the
    root form is recomputed from the record's run root (sample.seed == int.from_bytes(run_root[:8])); the verifier-challenge form is
    post-commitment by construction (sha256(challenge|root)); a `--replay-seed` seed is PROVER-CHOSEN before the run -> a MECHANISM run,
    not a C2 of record.  Returns (True, how) / (False, why) / (None, why) when the record carries no seed source (legacy: not graded here)."""
    smp = sr.get("sample") if isinstance(sr.get("sample"), dict) else {}
    src = str(sr.get("seed_source") if sr.get("seed_source") is not None else smp.get("seed_source"))
    seed = sr.get("seed") if sr.get("seed") is not None else smp.get("seed")
    form = RC.seed_form(sr)
    if form is None:
        return None, "sampled replay seed source absent from the record"
    if form == "prover-chosen":
        return False, "sampled replay seed prover-chosen (--replay-seed): the picks were knowable before commitment -- a mechanism run, not a C2 of record"
    if form == "run-root":
        if not run_root_hex or seed is None:
            return False, f"sampled replay seed source says '{src}' but the record carries no run root / seed to recompute it from"
        try:
            expect = int.from_bytes(bytes.fromhex(str(run_root_hex))[:8], "big")
        except ValueError:
            return False, f"sampled replay seed: run root {str(run_root_hex)[:16]}... is not hex"
        if int(seed) != expect:
            return False, f"sampled replay seed {seed} != the run root's first 8 bytes {expect} although seed_source says post-commitment root form"
        return True, "seed == run_root[:8] (recomputed)"
    if form == "challenge":
        # the challenge form is a FACT only with the challenge on the record: recompute sha256(challenge|run_root)[:8] and the
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
        if int(seed) != expect:
            return False, f"sampled replay seed {seed} != sha256(challenge|run_root)[:8] = {expect} recomputed from the recorded challenge"
        if src != RC.challenge_seed_source(str(chal)):
            return False, f"sampled replay seed source text does not name the recorded challenge ({str(chal)[:16]!r})"
        return True, "seed == sha256(challenge|run_root)[:8] (recomputed from the recorded challenge)"
    return False, f"sampled replay seed source {src!r} is not a post-commitment form (run root / verifier challenge)"


def _partial_replay_named_gap(sr: dict | None, oc: dict | None) -> tuple[str | None, str | None]:
    """The C2 rule when the sampled replay is PARTIAL.  Returns (NAMED GAP text, None) when the record still
    establishes C2 -- a partial replay with 0 mismatch whose every unevaluated stratum is named (family | module | reason) under a class form
    (B) can cover, AND form (B) (the Match-oracle compare) complete over the values with 0 mismatch and attribution complete; (None, FAIL text)
    when a named stratum is PLANE-FED (form (B) has no oracle family for the moe/ planes those VUs commit, so no complete
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
    # decide PER STRATUM from the uncapped `unevaluated_strata` (never from the capped not_evaluated[:12] sample): every
    # stratum in strata_unevaluated must carry an entry with spec, op_path and a reason; a stratum without one is an unnamed gap -> None
    by = {str(u.get("stratum")): u for u in (sr.get("unevaluated_strata") or []) if isinstance(u, dict)}
    if any(k not in by or not (by[k].get("spec") and by[k].get("op_path") and by[k].get("why")) for k in unev):
        return None, None
    # only BY-NAME classes qualify (no evaluator for the family / state-linked / plane-fed / interior-composed v2); an
    # evaluator fault, a row->member mapping fault, a missing committed input, a device fault never do: a fault is a run that did not
    # evaluate, not a gap on the record.
    classes = {k: RC.stratum_classes(by[k]) for k in unev}
    bad = {k: c for k, c in classes.items() if any(x not in RC.WHY_CLASSES_BY_NAME for x in c)}
    if bad:
        return None, None
    # a plane-fed stratum is covered by neither form -> FAIL by name (form (B) cannot lift it)
    plane = [k for k, c in classes.items() if "plane-fed" in c]
    if plane:
        n_p = sum(int(by[k].get("n") or 0) for k in plane)
        fams_p = sorted({str(by[k].get("spec") or "").split("{", 1)[0] for k in plane})
        whys_p = sorted({str(by[k].get("why") or "").split(" [", 1)[0][:90] for k in plane})[:3]
        return None, (f"C2 of record not established: {len(plane)} plane-fed strata ({n_p} VUs) {_PLANE_FED_NEITHER_FORM}: "
                      f"{', '.join(fams_p)} -- {' | '.join(whys_p)}; sampled replay partial [{samp.get('evaluated')} evaluated, 0 mismatch, "
                      f"{len(unev)} strata named not evaluated]; C1 + C3 hold on those planes, C2 needs form (A) over the planes "
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


def _requests_n(commit: dict) -> int:
    """`binding.population.requests`, the batch the C2 attribution is about; 0 when the binding counts none (a failed one, a sealed
    window's, or a TP rank's, which lists its requests' ids)."""
    b = _binding_of(commit)
    pop = None if b is None else b.population
    return 0 if pop is None or pop.requests is None else pop.requests


def _batch_attribution_note(n_req: int, oc) -> str | None:
    """At B>=2 (`n_req`, `_requests_n`) the per-(step, request) ATTRIBUTION of every committed value rests on form (B):
    an equal-shape request swap relabels every row of a shared step together, so the committed record stays self-consistent and the sampled
    replay passes by construction; boundary linkage sees it only at the first swapped boundary and only when the two requests' tokens differ
    there.  Form (B) is not made mandatory here (the verdict stays what the record establishes) -- but a PASS at B>=2 without a
    complete request attribution on the workload's own record NAMES the gap, so no reader has to infer it from an absent section: (None) at B=1, or when
    the oracle compare's attribution is complete; else the note for the PASS text.  Value-partial by snapshot gaps is not this: the
    attribution section is complete there."""
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
            "whole-step equal-shape request swap in general -- a gap by name, not a verified attribution")


def _replay_scope(sr: dict | None) -> dict[str, Any]:
    if not isinstance(sr, dict):
        return {}
    smp = _d(sr.get("sample"))
    pop = _d(sr.get("population"))
    picks = _d(smp.get("picks_by_request_step"))
    positions = sum(len(_d(v)) for v in picks.values()) if picks else None
    qt, qc = _d(pop.get("query_total")), _query_checks(sr)
    return {"result": sr.get("result"), "grade": sr.get("grade"), "sample_size": smp.get("evaluated"), "picked": smp.get("picked"), "equal": smp.get("equal"),
            "mismatch_n": smp.get("mismatch_n", sr.get("mismatch_n")), "not_evaluated_n": smp.get("not_evaluated_n"), "per_stratum": smp.get("per_stratum"),
            "positions": positions, "requests": len(picks) if picks else (len(pop.get("requests") or []) if isinstance(pop.get("requests"), (list, dict)) else None),
            "strata": len(sr.get("strata") or []) or sr.get("strata_n") or pop.get("strata_n"), "strata_unevaluated": len(sr.get("strata_unevaluated") or []),
            "population_vus": pop.get("vus"), "population_by_family": pop.get("by_family"), "not_evaluable": pop.get("not_evaluable") or None,
            "query_total": qt.get("identities") if qt else None, "query_id": qt.get("query_id") if qt else None, "query_checks_ok": qc.ok if qc is not None else None}


def _replay_inputs(sr: dict | None, ctx: dict) -> dict[str, Any]:
    if not isinstance(sr, dict):
        return {}
    smp = _d(sr.get("sample"))
    out = {"sample_seed": sr.get("seed") if sr.get("seed") is not None else smp.get("seed"), "seed_source": sr.get("seed_source") if sr.get("seed_source") is not None else smp.get("seed_source"),
           "artifact": sr.get("artifact")}
    for k, v in (ctx.get("file_digests") or {}).items():
        if "sampled_replay" in k:
            out[k] = v
    return {k: v for k, v in out.items() if v is not None}


C_SR = ("local_replay/sampled_replay", "C2 form (A)",
        "A post-commitment stratified sample of required values replays exactly (bit-for-bit) from the committed bytes under the Program's registered relations.",
        ["sampled_replay.result", "sampled_replay.grade", "sampled_replay.why", "sampled_replay.sample", "sampled_replay_p*.json", "replay_cache"])
C_SEED = ("local_replay/seed_of_record", "C2 seed of record",
          "The replay's seed is post-commitment as a FACT: recomputed from the run root (or the recorded verifier challenge), never prover-chosen.",
          ["sampled_replay.sample.seed", "sampled_replay.sample.seed_source", "sampled_replay.sample.challenge"])
C_GAP = ("local_replay/population_gaps", "C2 population gaps by name",
         "Every VU the replay left unevaluated is named under a class form (B) covers, and form (B) is complete over the values where it must stand in; plane-fed VUs never pass by name.",
         ["sampled_replay.population.not_evaluable", "sampled_replay.strata_unevaluated", "sampled_replay.unevaluated_strata", "oracle_compare"])
C_SCOPE = ("local_replay/population_scope", "C2 population against Q(P)",
           "The replay's population is the required-value manifest Q(P) of record, reconciled identity by identity (reach + uncovered-by-name == |Q(P)|; engines agree).",
           ["sampled_replay.population.query_total", "sampled_replay.population.query_checks", "sampled_replay.population.scope", "sampled_replay.population.uncovered_by_name"])
C_ATT = ("local_replay/attribution", "C2 per-(step, request) attribution",
         "At B >= 2 the per-(step, request) attribution of every committed value is verified by a complete form (B); else the gap is named on the PASS.",
         ["binding.population.requests", "oracle_compare.attribution"])
C_LK = ("boundary_linkage/ordered_linkage", "C2 boundary linkage",
        "Every (engine step, request) boundary is linked: declared prompt -> input ids at the arrival step; sampled id[s-1] -> input id[s]; none unlinked.",
        ["boundary_linkage.result", "boundary_linkage.unlinked_n", "boundary_linkage.pairs", "boundary_linkage.horizon", "linkage"])
C_WP = ("checkpoint_binding/weights_of_record", "MAN-09",
        "Every served parameter the Program names equals by digest the checkpoint-composed tensor of record; every shard is pinned by its manifest sha256.",
        ["weights_pin.result", "weights_pin.why", "weights_pin.compared", "weights_pin.mismatched", "weights_pin.missing", "weights_pin.shards"])


def c2_rules(commit: dict, ctx: dict) -> list[CheckResult]:
    """The C2-of-record section of commit_verdict (lines `sr = commit.get("sampled_replay")` .. `_scope`), evaluated rule by rule with
    the SAME nesting (a branch v1 never took is not taken here), emitting: local_replay/sampled_replay, /seed_of_record, /population_gaps,
    /population_scope, /attribution, /cublas_route_pin; boundary_linkage/ordered_linkage; checkpoint_binding/weights_of_record."""
    sr, lk, wp = _replay_of(commit), _linkage_of(commit), commit.get("weights_pin")
    row_proc = _is_row_procedure(commit)
    oc = _oracle_of(commit)
    out: dict[str, CheckResult] = {}
    sr_scope, sr_inputs = _replay_scope(sr), _replay_inputs(sr, ctx)
    sr_notes = [_leak("sampled_replay.EVALUATORS / NOT_YET", "which families the replay can evaluate is a table by family name (the stated applicability domain)")]
    rc = ctx.get("replay_cache")
    if rc:
        sr_scope["replay_cache"] = {k: rc.get(k) for k in ("distinct_replays", "reused_pairs")}

    def put(spec: tuple, outcome: str, *, reason: str | None = None, differences: list | None = None, notes: list | None = None, scope: dict | None = None, inputs: dict | None = None) -> None:
        nm, rule, claim, compat = spec
        out[nm] = _mk(nm, rule, claim, list(compat), commit, ctx, outcome=outcome, reason=reason, differences=differences or [], notes=notes or [], scope=scope or {}, inputs=inputs or {})

    lk_scope = {k: _d(lk).get(k) for k in ("pairs", "unlinked_n", "linked", "prescribed", "embed_op_path")} if isinstance(lk, dict) else {}
    if isinstance(lk, dict):
        lk_scope["requests"] = len(_d(lk.get("horizon"))) or None
        lk_scope["prompt_linked_n"] = sum(1 for v in _d(lk.get("prompt_linked")).values() if v) or None
    wp_scope = {k: (len(_d(wp).get(k)) if isinstance(_d(wp).get(k), (list, dict)) else _d(wp).get(k)) for k in ("compared", "mismatched", "missing", "unresolved", "unpinned", "shards", "all_shards_pinned", "shards_ok")} if isinstance(wp, dict) else {}
    wp_inputs = {"weights_program_digest": _d(wp).get("program_digest"), "root_of_record": _d(wp).get("root_of_record"), "committed_weights_root": _d(wp).get("committed_weights_root"),
                 "checkpoint_entry": (str(_d(wp).get("checkpoint_entry"))[:120] if _d(wp).get("checkpoint_entry") else None)} if isinstance(wp, dict) else {}
    wp_inputs = {k: v for k, v in wp_inputs.items() if v is not None}
    wp_notes = [_leak("weights_of_record.FUSED / EXPERTS / EXPERTS_MIXTRAL", "the checkpoint composition of fused / expert tensors is keyed by known parameter-name tables")]

    # --- the unconditional FAILs (any record) -------------------------------------------------------------------------------------------
    if isinstance(sr, dict) and sr.get("result") is False:
        why = sr.get("why") or "a sampled VU does not replay from its committed inputs"
        put(C_SR, FAIL, reason=f"C2 sampled replay: {why}", scope=sr_scope, inputs=sr_inputs, notes=sr_notes,
            differences=[{"name": "replay_mismatch", "mismatch_n": _d(sr.get("sample")).get("mismatch_n", sr.get("mismatch_n")),
                          "first": _first(sr.get("mismatch")), "why": str(why)[:300]}])
    if isinstance(lk, dict) and lk.get("result") is False:
        put(C_LK, FAIL, reason=f"C2 boundary linkage: {lk.get('why') or 'an (engine step, request) boundary is not linked'} ({lk.get('unlinked_n')} unlinked)", scope=lk_scope,
            differences=[{"name": "boundaries_unlinked", "n": lk.get("unlinked_n"), "first": _first(lk.get("unlinked"))}])
    if isinstance(wp, dict) and wp.get("result") is False:
        put(C_WP, FAIL, reason=f"weights of record: {wp.get('why') or 'served weights differ from the checkpoint of record'}", scope=wp_scope, inputs=wp_inputs, notes=wp_notes,
            differences=[{"name": "weights_differ", "mismatched": (wp.get("mismatched") or [])[:8], "missing": (wp.get("missing") or [])[:8], "unresolved": (wp.get("unresolved") or [])[:8]}])

    if row_proc:
        _oc = oc
        mech = False
        if isinstance(sr, dict) and sr.get("result") is not False:
            sok, swhy = _replay_seed_of_record(sr, commit.get("run_root"))
            if sok is False:
                missing_b = _form_b_complete_over_values(_oc)
                if missing_b is not None:
                    put(C_SEED, FAIL, reason=f"C2 of record not established: {swhy}; and form (B) does not stand in: {missing_b}", inputs=sr_inputs,
                        differences=[{"name": "seed_not_post_commitment", "why": swhy, "form_b": missing_b}])
                else:
                    gap0, fail0 = _complete_replay_population_gap(sr, _oc) if sr.get("result") is True else _partial_replay_named_gap(sr, _oc)
                    if fail0 is not None:
                        put(C_GAP, FAIL, reason=fail0, differences=[{"name": "population_gap_uncovered", "why": fail0[:400]}])
                    elif sr.get("result") is not True and gap0 is None:
                        sr_why = sr.get("why") or sr.get("grade")
                        put(C_SR, FAIL, reason=(f"C2 of record not established: sampled replay {sr.get('grade')} -- {sr_why}; "
                                               f"the row's Commit is C1 + C3 only (whatever seeded the replay: {swhy})"), scope=sr_scope, inputs=sr_inputs, notes=sr_notes,
                            differences=[{"name": "replay_partial_unnamed_gap", "grade": sr.get("grade"), "why": str(sr_why)[:300]}])
                    else:
                        put(C_SEED, PASS, notes=[f"{swhy}; C2 of record by the complete form (B) alone (mechanism run)"], inputs=sr_inputs,
                            differences=[{"name": "seed_prover_chosen", "why": swhy}])
                        if gap0 is not None:
                            put(C_GAP, PASS, notes=[gap0])
                        mech = True
            elif sok is True:
                put(C_SEED, PASS, notes=[swhy], inputs=sr_inputs, scope={"seed_source": sr_inputs.get("seed_source")})
            else:
                put(C_SEED, NOT_RUN, notes=[swhy], inputs=sr_inputs)
        if mech:
            pass
        elif not (isinstance(sr, dict) and sr.get("result") is True):
            gap, fail = _partial_replay_named_gap(sr, _oc)
            if fail is not None:
                put(C_GAP, FAIL, reason=fail, differences=[{"name": "population_gap_uncovered", "why": fail[:400]}])
            elif gap is None:
                if "local_replay/sampled_replay" not in out:            # (a result-False replay was named above)
                    if not isinstance(sr, dict):
                        put(C_SR, INSUFFICIENT, reason="C2 of record not established: sampled replay None -- not run (no --program-dir / --programs-root); the row's Commit is C1 + C3 only",
                            notes=sr_notes, differences=[{"name": "replay_not_run"}])
                    else:
                        sr_why = sr.get("why") or sr.get("grade")
                        put(C_SR, FAIL, reason=f"C2 of record not established: sampled replay {sr.get('grade')} -- {sr_why}; the row's Commit is C1 + C3 only",
                            scope=sr_scope, inputs=sr_inputs, notes=sr_notes, differences=[{"name": "replay_partial_unnamed_gap", "grade": sr.get("grade"), "why": str(sr_why)[:300]}])
            else:
                put(C_GAP, PASS, notes=[gap])
                if "local_replay/sampled_replay" not in out:
                    put(C_SR, PASS, scope=sr_scope, inputs=sr_inputs, notes=sr_notes + ["partial replay, 0 mismatch; every unevaluated stratum named and covered by form (B)"])
        else:
            gap, fail = _complete_replay_population_gap(sr, _oc)
            if fail is not None:
                put(C_GAP, FAIL, reason=fail, differences=[{"name": "population_gap_uncovered", "why": fail[:400]}])
            else:
                put(C_GAP, PASS, notes=[gap] if gap else [])
        if not (isinstance(lk, dict) and lk.get("result") is True) and "boundary_linkage/ordered_linkage" not in out:
            lk_why = lk.get("why") if isinstance(lk, dict) else "not run"
            put(C_LK, INSUFFICIENT if not isinstance(lk, dict) else FAIL,
                reason=f"C2 of record not established: boundary linkage {lk.get('result') if isinstance(lk, dict) else None} -- {lk_why}", scope=lk_scope,
                differences=[{"name": "linkage_not_run" if not isinstance(lk, dict) else "linkage_not_established", "why": lk_why}])
        if not (isinstance(wp, dict) and wp.get("result") is True) and "checkpoint_binding/weights_of_record" not in out:
            wp_why = wp.get("why") if isinstance(wp, dict) else "not run (no --weights-of-record / --checkpoint-dir)"
            put(C_WP, INSUFFICIENT, reason=(f"weights unpinned: the served weights were not compared with the checkpoint of record -- {wp_why}; "
                                            "the committed weights root is self-consistency, not authentication"), scope=wp_scope, inputs=wp_inputs, notes=wp_notes,
                differences=[{"name": "weights_unpinned", "why": wp_why}])
        n_req = _requests_n(commit)
        att = _batch_attribution_note(n_req, _oc)
        if att is not None:
            put(C_ATT, PASS, notes=[att], scope={"requests": n_req, "attribution_complete": False}, differences=[{"name": "attribution_unverified", "requests": n_req}])
        elif n_req >= 2:
            put(C_ATT, PASS, scope={"requests": n_req, "attribution_complete": True})
        else:
            put(C_ATT, NOT_RUN, scope={"requests": n_req})
        scope_note, scope_fail = _query_population_scope(sr)
        if scope_fail is not None:
            acct = _scope_fail_account_missing(sr)
            put(C_SCOPE, INSUFFICIENT if acct else FAIL, reason=scope_fail, scope={k: sr_scope.get(k) for k in ("query_total", "query_id", "query_checks_ok")},
                differences=[{"name": "account_missing" if acct else "population_not_reconciled", "why": scope_fail[:400]}])
        elif scope_note is not None:
            put(C_SCOPE, PASS, notes=[scope_note], scope={k: sr_scope.get(k) for k in ("query_total", "query_id", "query_checks_ok")})
    # --- what was not graded is NOT_RUN / PASS-by-record -------------------------------------------------------------------------------
    if "local_replay/sampled_replay" not in out:
        if isinstance(sr, dict) and sr.get("result") is True:
            put(C_SR, PASS, scope=sr_scope, inputs=sr_inputs, notes=sr_notes)
        elif isinstance(sr, dict):
            put(C_SR, PASS if not row_proc else FAIL, scope=sr_scope, inputs=sr_inputs, notes=sr_notes + (["legacy record (no binding block): a partial replay is not graded"] if not row_proc else []))
        else:
            put(C_SR, NOT_RUN, notes=sr_notes)
    if "boundary_linkage/ordered_linkage" not in out:
        put(C_LK, PASS if isinstance(lk, dict) and lk.get("result") is True else NOT_RUN, scope=lk_scope)
    if "checkpoint_binding/weights_of_record" not in out:
        put(C_WP, PASS if isinstance(wp, dict) and wp.get("result") is True else NOT_RUN, scope=wp_scope, inputs=wp_inputs, notes=wp_notes)
    for spec in (C_SEED, C_GAP, C_SCOPE, C_ATT):
        if spec[0] not in out:
            put(spec, NOT_RUN)
    order = [C_SR[0], C_SEED[0], C_GAP[0], C_SCOPE[0], C_ATT[0], C_LK[0], C_WP[0]]
    return [out[n] for n in order] + [rule_route_pin(commit, ctx)]


C_ROUTE = ("local_replay/cublas_route_pin", "cuBLAS route pin",
           "A Program that binds a Definition only the cuBLAS route pin binds was served under the pin's scan's conditions, each measured in the serving process.",
           ["cublas_route_pin.ok", "cublas_route_pin.why", "cublas_route_pin.departures", "cublas_route_pin.families", "cublas_route_pin.pin"])


def rule_route_pin(commit: dict, ctx: dict) -> CheckResult:
    """v1: `cublas_route_pin.ok is False` -> FAIL by its `why`, on any record (`check.cublas_route_pin.stamp`); NOT_RUN when no section is
    stamped (the Program binds none of the pin's Definitions)."""
    rp = commit.get("cublas_route_pin")
    if not isinstance(rp, dict):
        return _mk(*C_ROUTE, commit, ctx, outcome=NOT_RUN)
    scope = {k: rp.get(k) for k in ("families", "pin")}
    if rp.get("ok") is not False:
        return _mk(*C_ROUTE, commit, ctx, outcome=PASS, scope=scope)
    return _mk(*C_ROUTE, commit, ctx, outcome=FAIL, reason=rp.get("why") or "cuBLAS route pin: served off the pin's scan's conditions", scope=scope,
               differences=[{"name": "served_off_route_pin", "departures": rp.get("departures") or []}])


def execution_extent_rules(commit: dict, ctx: dict) -> list[CheckResult]:
    """The executed prefix of record + its Match-account link + rows past the stop, read from binding.coverage (executed_prefix.py's outputs).
    v1 gated these THROUGH the coverage / replay results (a faulted class fails the replay's population scope by name); here each is its own
    check, and a MISSING ACCOUNT LEG (tokens.json / global_match.json not beside the Commit) is INSUFFICIENT_EVIDENCE, not FAIL."""
    b = _binding_of(commit)
    cov = None if b is None else b.coverage
    epr, link, clink, past = ((None,) * 4 if cov is None else
                              (cov.executed_prefix_of_record, cov.match_account_link, cov.commit_account_link, cov.rows_past_the_stop))
    # no Match record beside the Commit (a config run): the Commit's token record is the account; with both, both hold where it places a prefix
    by_commit = clink is not None and link is not None and not link.match_dir
    with_commit = clink is not None and (by_commit or clink.required)
    sr = _replay_of(commit) or {}
    facts = _d(_d(sr.get("executed_prefix_of_record")).get("facts_of_record"))
    C_EPR = ("execution_extent/executed_prefix_of_record", "executed prefix of record",
             "Each request's executed prefix is the cap on a length stop, or served + the Program's declared LAG on an EOS stop whose last served token is a declared EOS id; never defaulted.",
             ["binding.coverage.executed_prefix_of_record", "sampled_replay.executed_prefix_of_record", "executed_prefix_facts", "coverage.executed_prefix_of_record"])
    C_LINK = ("execution_extent/match_account_link", "executed-prefix conditions (a), (b)",
              "The Commit's served lengths equal the Match record's (tokens.json) and GM-01's G6 account of the same attempt -- or, with no Match stage, every served "
              "token of the Commit's token record opens from the run root to the committed sampled id -- and the LAG is attested on every leg.",
              ["binding.coverage.match_account_link", "binding.coverage.commit_account_link", "executed_prefix_facts.linked", "executed_prefix_facts.lag_attested",
               "account_missing"])
    C_PAST = ("execution_extent/rows_past_the_stop", "executed-prefix condition (c)",
              "No committed row lies at a position >= a request's executed prefix (execution past the stop).",
              ["binding.coverage.rows_past_the_stop", "executed_prefix_facts.rows_past_stop"])
    out = []
    inputs_extra = {}
    tj, g6 = (None, None) if link is None else (link.tokens_json, link.global_match)
    if tj is not None and tj.sha256:
        inputs_extra["tokens_json_sha256"] = tj.sha256
    if g6 is not None and g6.sha256:
        inputs_extra["global_match_sha256"] = g6.sha256
    if with_commit and clink.record_sha256:
        inputs_extra["commit_tokens_sha256"] = clink.record_sha256
    if epr is None:
        out.append(_mk(*C_EPR, commit, ctx, outcome=NOT_RUN))
    else:
        scope = {"requests": len(epr.requests), "lag": epr.lag, "eos_of_record": None if epr.eos_of_record is None else list(epr.eos_of_record),
                 "served_known": epr.served_known, "any_eos_stop": epr.any_eos_stop,
                 "lengths": {r: [v.served_output_len, v.executed_prefix_len, v.cap] for r, v in epr.requests.items()}}
        if epr.ok:
            out.append(_mk(*C_EPR, commit, ctx, outcome=PASS, scope=scope))
        else:
            probs = list(epr.problems)
            out.append(_mk(*C_EPR, commit, ctx, outcome=FAIL, scope=scope, differences=[{"name": "executed_prefix_problems", "n": len(probs), "first": probs[:4]}],
                           reason=f"executed prefix of record: {probs[0] if probs else 'not ok'}"))
    account_missing = sorted({leg for f in facts.values() if isinstance(f, dict) for leg in (f.get("account_missing") or [])})
    if not account_missing and link is not None and not by_commit:
        account_missing = [leg for leg, present in (("tokens.json", tj is not None), ("global_match.json", g6 is not None)) if not present]
    # v1 gated the link THROUGH the replay's not-executed class: it faults (-> population_scope FAIL) only when applied, i.e. when a request's
    # prefix is reduced from its cap (`link.required`) or the class names the fault.  Outside that the link is recorded, not gating.
    acct = clink if by_commit else link
    gating = (acct is not None and acct.required) or any(f.get("account_missing") or f.get("linked") is not True for f in _faulted_prefix_facts(sr))
    if link is None:
        out.append(_mk(*C_LINK, commit, ctx, outcome=NOT_RUN))
    else:
        scope = {"required": link.required, "gating": gating, "lag_attested": link.lag_attested, "tokens_json": None if tj is None else tj.path,
                 "global_match_verdict": None if g6 is None else g6.verdict, "g6_status": None if g6 is None else g6.g6_status,
                 "requests_linked": (len(g6.requests_linked) if g6 is not None and g6.requests_linked else None),
                 "facts_linked": {r: f.get("linked") for r, f in facts.items() if isinstance(f, dict)} or None, "account_missing": account_missing or None}
        if with_commit:
            scope |= {"account": "commit token record" if by_commit else "Match record + commit token record", "commit_tokens": clink.record_path,
                      "commit_required": clink.required, "commit_lag_attested": clink.lag_attested}
        probs = ([] if by_commit else list(link.problems)) + (list(clink.problems) if with_commit else [])
        linked = (by_commit or link.ok) and (not with_commit or clink.ok is True)
        if account_missing and gating:
            out.append(_mk(*C_LINK, commit, ctx, outcome=INSUFFICIENT, scope=scope, inputs=inputs_extra,
                           differences=[{"name": "account_missing", "legs": account_missing}],
                           reason=f"Match account leg(s) NOT beside the Commit: {', '.join(account_missing)} -- a missing input, not a verdict on the Program"))
        elif account_missing:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra,
                           notes=[f"Match account leg(s) not beside the Commit: {', '.join(account_missing)}; not gating (no request stopped before its cap)"]))
        elif linked:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra))
        elif gating:
            out.append(_mk(*C_LINK, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs_extra,
                           differences=[{"name": "account_not_linked", "n": len(probs), "first": probs[:4]}],
                           reason=f"{'commit token record' if by_commit else 'Match account'} link: {probs[0] if probs else 'not ok'}"))
        else:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra,
                           notes=[f"link problems recorded, not gating (no request stopped before its cap): {'; '.join(p[:120] for p in probs[:3])}"]))
    if past is None:
        out.append(_mk(*C_PAST, commit, ctx, outcome=NOT_RUN))
    else:
        scope = {"entries_n": past.entries_n, "padded_record": past.padded_record, "padding_entries_n": past.padding_entries_n}
        if past.ok and not past.entries_n and not past.error:
            out.append(_mk(*C_PAST, commit, ctx, outcome=PASS, scope=scope))
        else:
            out.append(_mk(*C_PAST, commit, ctx, outcome=FAIL, scope=scope,
                           differences=[{"name": "rows_past_the_stop", "n": past.entries_n, "by_request": past.steps_past_by_request, "error": past.error}],
                           reason=f"{past.entries_n} committed row(s) past a request's executed prefix"))
    return out
