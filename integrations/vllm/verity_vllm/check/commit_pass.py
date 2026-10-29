"""v1's fail-closed Commit decision for ONE instrumented run (`commit_verdict`: `(ok, reason)` with the FIRST failing reason), the
predicate `verdict.run_checks` evaluates rule by rule."""

from __future__ import annotations

from verity_vllm.check import c2_rules as C2
from verity_vllm.check import commit_rules as CR
from verity_vllm.check.commit_rules import REQUIRED_CLASSES_DEFAULT


def commit_verdict(commit: dict, tokens_equal: bool | None, *, min_openings: int = 64, declaration: dict | None = None,
                   required_classes: tuple[str, ...] | list[str] | None = None) -> tuple[bool, str]:
    """X-03 fail-closed Commit verdict for ONE instrumented run.  PASS only if ALL hold: the required-value manifest's attempted
    acquisitions equal the bound ones (`expected_tensors == committed_tensors`, both > 0), no collector/arena/hook error was recorded,
    every opening checked verified and at least `min_openings` were checked, control vs instrumented tokens are equal, and (REL-02, when a
    `declaration` is given) every required class is declared and every declared leaf-binding class bound > 0 tensors.  Returns
    (ok, reason) with the FIRST failing reason; never 'PASS with errors recorded'."""
    c2_notes: list[str] = []                               # named gaps carried on a PASS (form (B) covered a partial replay)
    fail = (_c1_verdict(commit, tokens_equal, min_openings, declaration, required_classes) or _c2_verdict(commit, c2_notes)
            or _after_release_verdict(commit, min_openings))
    if fail is not None:
        return fail
    if c2_notes:
        return True, "PASS (" + "; ".join(c2_notes) + ")"
    return True, "PASS"


def _c1_verdict(commit: dict, tokens_equal: bool | None, min_openings: int, declaration: dict | None,
                required_classes: tuple[str, ...] | list[str] | None) -> tuple[bool, str] | None:
    """Reference integrity, identity coverage, collector errors, required classes, bounded staging, openings, tokens."""
    ex = commit.get("extra") or {}
    n_exp, n_com = ex.get("expected_tensors"), ex.get("committed_tensors")
    errs_n = ex.get("errors_n")
    if errs_n is None:
        errs_n = len(ex.get("errors") or [])
    ops = commit.get("openings") or {}
    n_open, n_ok = int(ops.get("n") or 0), int(ops.get("verified") or 0)
    if ex.get("collector_failure"):                        # a step the worker abandoned: the counts and the root below follow from it
        return False, f"collector failed: {str(ex['collector_failure'])[:200]}"
    if n_exp is None or n_com is None:
        return False, "reference integrity counters missing (expected_tensors/committed_tensors): committer does not report them"
    if int(commit.get("fa2_zero_rows") or 0) > 0:
        return False, f"{commit['fa2_zero_rows']} FA2 hidden rows bound as all-zero bytes (tap staged after it was zeroed: ordering fault)"
    if int(n_exp) <= 0:
        return False, "expected_tensors == 0: the required-value manifest bound nothing"
    if int(n_exp) != int(n_com):
        return False, f"expected_tensors {n_exp} != committed_tensors {n_com} (refused/lost acquisitions -> incomplete root)"
    bnd = commit.get("binding")
    if bnd is not None:                                    # BIND-01: identity COVERAGE of the expected population is the gate
        ok, why = CR.binding_check(bnd, min_openings=min_openings)
        if not ok:
            return False, why
    cov = ex.get("coverage") or commit.get("coverage")
    if cov is not None and not cov.get("ok", False):     # Identities REQUIRED by the Program/query manifest, not the collector's own attempts
        fam = cov.get("families_missing_entirely") or []
        first = (cov.get("missing") or [{}])[0]
        if cov.get("error") and not cov.get("missing_n"):   # a named coverage error with nothing uncovered (prefix, reconciliation): the name IS the reason
            return False, f"manifest coverage: {cov['error']}"
        return False, (f"manifest coverage: {cov.get('missing_n')} required identities uncovered ({cov.get('missing_by_family')}); "
                       + (f"whole families missing: {fam}; " if fam else "") + f"first: step {first.get('step')} {first.get('op_path')}/{first.get('output_member')} inv {first.get('invocation')} want {first.get('want_range')} got {first.get('committed_numel')}")
    idc = ex.get("identity_check")
    if idc is not None and not idc.get("ok", False):     # Identities, not counts -- missing-one-and-duplicating-another fails here
        v = (idc.get("violations") or [{}])[0]
        return False, f"identity check: {idc.get('violations_n')} step(s) differ from the expected manifest (first: step {v.get('step')} {CR.json_short(v)})"
    if int(errs_n) > 0:
        first = (ex.get("errors") or ["?"])[0]
        return False, f"{errs_n} collector error(s): {str(first)[:160]}"
    if declaration is not None or required_classes is not None:
        ok, why = CR.required_classes_check(declaration, required_classes if required_classes is not None else REQUIRED_CLASSES_DEFAULT)
        if not ok:
            return False, why
    oar = commit.get("openings_after_release")
    # a bounded-staging record run must not carry a learned-only step (a placeholder root from the learn-only warm-up path)
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
    elif n_ok != n_open:                      # retain=host + exclude: the host share of the in-run openings must still verify
        return False, f"openings (host share) verified {n_ok} != checked {n_open}"
    if tokens_equal is not True:
        return False, "control vs instrumented tokens differ"
    return None


def _c2_verdict(commit: dict, c2_notes: list[str]) -> tuple[bool, str] | None:
    """Value correspondence, the C2 of record (sampled replay, boundary linkage, weights of record) and its population scope."""
    bnd = commit.get("binding")
    vc = commit.get("value_correspondence")
    if isinstance(vc, dict) and vc.get("result") is False:
        # LIVE C2: the committed bytes of an identity differ from the Match oracle's value (or the reference root):
        # a relabel / same-name byte swap / late read that C1 (coverage) and C3 (openings) cannot see.  Named, fail closed.
        # name the FAILING component's own cause: the oracle's `why` only when the ORACLE failed (a partial-coverage note
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
    # C2 OF RECORD = sampled exact VU replay from the committed bytes (registered relation
    # evaluators on the Program's dataflow) + exhaustive ordered boundary linkage (declared prompt -> input ids at the arrival step; sampled
    # id[s-1] -> input id[s]).  Required at EVERY B for a row-procedure record (one carrying the binding block): a mismatch / an unlinked
    # boundary is a named FAIL; ABSENCE or a partial replay is a named FAIL too (never an unsound PASS on C1 + C3 alone).
    # The linkage is not a B>=2 device: at B=1 the arrival-step check ties the Commit to the workload of record (the wl_swapped canary) and
    # the decode chain is what a cross-step relabel breaks.  The full Match-oracle compare is a diagnostic: a mismatch still fails (above),
    # a complete attribution no longer substitutes for the replay.  Records without a binding block (unit fixtures of other checks) keep
    # the legacy rule.  TP2 (tp2_commit grades the rank block before attaching its binding) is graded by tp's path until the rank replay lands.
    sr = commit.get("sampled_replay")
    lk = commit.get("boundary_linkage")
    # a producer that assembles its C2 under value_correspondence.components (compiled S3's kernel replay) is graded by the
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
    # WEIGHTS OF RECORD: the registered inputs must be authenticated by their canonical reference --
    # the pinned checkpoint shards -- not by re-hashing the committer's own parameters (O5 is self-consistency).  Every served parameter the
    # Program names is compared by digest with the checkpoint-composed tensor of record: a difference (a flipped weight) or a missing field
    # is a named FAIL; on a row-procedure record the pin is REQUIRED (absent / not evaluated = 'weights unpinned', a named FAIL, like O1).
    wp = commit.get("weights_pin")
    if isinstance(wp, dict) and wp.get("result") is False:
        return False, f"weights of record: {wp.get('why') or 'served weights differ from the checkpoint of record'}"
    rp = commit.get("cublas_route_pin")
    if isinstance(rp, dict) and rp.get("ok") is False:
        return False, rp.get("why") or "cuBLAS route pin: served off the pin's scan's conditions"
    if isinstance(bnd, dict) and (bnd.get("population") or bnd.get("coverage")):
        _oc = oc if isinstance(oc, dict) else ((vc.get("oracle_compare") if isinstance(vc, dict) else None) or
                                               (_comps.get("oracle_compare") if isinstance(_comps, dict) else None))
        # the replay's seed must be POST-COMMITMENT as a fact, not a label: a prover-chosen --replay-seed (or a
        # root-form label whose seed is not the run root's) makes the replay a mechanism run -- C2 of record then rests on form (B) alone
        # and the PASS names it; without a complete form (B) the Commit FAILS by name.  Legacy records without a seed source
        # are graded by the finalizer (report), not here.
        _mech = False
        if isinstance(sr, dict) and sr.get("result") is not False:
            _sok, _swhy = C2._replay_seed_of_record(sr, commit.get("run_root"))
            if _sok is False:
                _missing_b = C2._form_b_complete_over_values(_oc)
                if _missing_b is not None:
                    return False, f"C2 of record not established: {_swhy}; and form (B) does not stand in: {_missing_b}"
                # form (B) never covers plane-fed VUs -- the population / stratum gaps are still named whatever the seed
                _gap0, _fail0 = C2._complete_replay_population_gap(sr, _oc) if sr.get("result") is True else C2._partial_replay_named_gap(sr, _oc)
                if _fail0 is not None:
                    return False, _fail0
                if sr.get("result") is not True and _gap0 is None:
                    # the named-gap rule runs BEFORE the mechanism lift: a PARTIAL replay with no NAMED gap (an
                    # unevaluated pick of FAULT class inside an evaluated stratum -- 'a fault is a run that did not evaluate, not a gap') fails
                    # closed whatever seeded the replay; form (B) stands in for a named gap, never for a fault
                    sr_why = sr.get("why") or sr.get("grade")
                    return False, (f"C2 of record not established: sampled replay {sr.get('grade')} -- {sr_why}; "
                                   f"the row's Commit is C1 + C3 only (whatever seeded the replay: {_swhy})")
                c2_notes.append(f"{_swhy}; C2 of record by the complete form (B) alone")
                _mech = True
        if _mech:
            pass
        elif not (isinstance(sr, dict) and sr.get("result") is True):
            # C2 of record holds when EITHER form is complete (any ONE suffices): a PARTIAL replay
            # with 0 MISMATCH whose every unevaluated stratum is NAMED (family | module | reason) is a GAP on the record, not a failure,
            # when form (B) -- the Match-oracle compare -- is complete (0 mismatch, attribution complete) over the values.  A replay
            # mismatch always fails (above); the absence of both forms still fails (O1 stands).
            _gap, _fail = C2._partial_replay_named_gap(sr, _oc)
            if _fail is not None:                                    # a plane-fed stratum: neither form covers it, named
                return False, _fail
            if _gap is None:
                sr_why = (sr.get("why") or sr.get("grade")) if isinstance(sr, dict) else "not run (no --program-dir / --programs-root)"
                return False, (f"C2 of record not established: sampled replay {sr.get('grade') if isinstance(sr, dict) else None} -- {sr_why}; "
                               "the row's Commit is C1 + C3 only")
            c2_notes.append(_gap)
        else:
            # a COMPLETE replay whose population excluded VUs BY NAME (plane-fed,
            # interior-composed v2) carries the same gap the partial rule does: those VUs have committed members no form checked unless
            # form (B) is complete over the values -- the population move keeps that condition, it does not lift it.
            _gap, _fail = C2._complete_replay_population_gap(sr, _oc)
            if _fail is not None:
                return False, _fail
            if _gap is not None:
                c2_notes.append(_gap)
        if not (isinstance(lk, dict) and lk.get("result") is True):
            lk_why = lk.get("why") if isinstance(lk, dict) else "not run"
            return False, f"C2 of record not established: boundary linkage {lk.get('result') if isinstance(lk, dict) else None} -- {lk_why}"
        if not (isinstance(wp, dict) and wp.get("result") is True):
            wp_why = wp.get("why") if isinstance(wp, dict) else "not run (no --weights-of-record / --checkpoint-dir)"
            return False, (f"weights unpinned: the served weights were not compared with the checkpoint of record -- {wp_why}; "
                           "the committed weights root is self-consistency, not authentication")
        _att = C2._batch_attribution_note(C2._requests_n(commit), _oc)
        if _att is not None:
            c2_notes.append(_att)
        # the population of record is Q(P): the replay's reach reconciled against the required manifest is stated on
        # every PASS (query_total / uncovered_by_name), and a reconciliation failure is a named FAIL -- never a silent narrowing
        _scope, _scope_fail = C2._query_population_scope(sr)
        if _scope_fail is not None:
            return False, _scope_fail
        if _scope is not None:
            c2_notes.append(_scope)
    return None


def _after_release_verdict(commit: dict, min_openings: int) -> tuple[bool, str] | None:
    """Openings after the raw device staging was released, and every pick deferred to replay."""
    ops = commit.get("openings") or {}
    oar = commit.get("openings_after_release")
    bnd = commit.get("binding")
    if oar is not None:                      # Openings after the raw device staging was released must verify against the ORIGINAL roots
        n_a, ok_a = int(oar.get("n") or 0), int(oar.get("verified") or 0)
        if n_a < min_openings:
            return False, f"openings after release checked {n_a} < {min_openings}"
        if ok_a != n_a:
            return False, f"openings after release verified {ok_a} != checked {n_a} (source {oar.get('source')})"
        if "replay" in str(oar.get("source") or "") and oar.get("replay_root_equal") is not True:   # replay | host+replay
            return False, "replay run root differs from the original run root (replay-to-open is not reproducing the committed bytes)"
        if "replay" in str(oar.get("source") or "") and int(oar.get("replay_errors_n") or 0) > 0:
            return False, f"replay run recorded {oar.get('replay_errors_n')} collector error(s)"
        if int(oar.get("retained_device_after") or 0) > 0:
            return False, f"raw/level bytes still on the device after release: {oar.get('retained_device_after')}"
        # the challenge / population picks the in-run stages could not open from host (excluded families) were handed to
        # this stage: every one of them must have been opened here by replay, and verified (ok_a == n_a above covers the verification)
        _po = (bnd or {}).get("openings_population") or {}
        _def_a, _def_b = int(ops.get("deferred_replay_n") or 0), int(_po.get("deferred_replay_n") or 0)
        if _def_a or _def_b:
            # by POSITION when the record carries them: the union of both stages' deferred (step, leaf) sets must be a subset
            # of the after-release stage's received set; count-only records (older) fall back to received >= max(a, b)
            _pa = [tuple(p) for p in (ops.get("deferred_positions") or [])]
            _pb = [tuple(p) for p in (_po.get("deferred_positions") or [])]
            _got = {tuple(p) for p in (oar.get("deferred_received_positions") or [])}
            # the count-only branch is for records written BEFORE positions existed: a writer that knows positions always
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
    return None
