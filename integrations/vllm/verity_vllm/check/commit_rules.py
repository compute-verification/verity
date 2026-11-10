"""The rules of one instrumented run's Commit record, each a `CheckResult`: runtime match, required-value coverage and program
source identity, with the C1 helpers the decision (`verdict.commit_verdict`) shares.

`expected_tensors` counts the acquisitions the collector attempted, so a required producer that never fires contributes 0 to both
sides.  The required classes close that gap: every required class must be declared by the committer's coverage declaration, and
every declared class that binds leaves (`bound-as-leaf` / `registered-input`) must have `tensors > 0`."""
from __future__ import annotations

from typing import Any, Mapping

from verity_vllm.check.codes import VerificationCode as VC

from verity_vllm.check.result import FAIL, INSUFFICIENT, NOT_RUN, PASS, CheckResult, code_of
from verity_vllm.records.commit_binding import CommitBinding
from verity_vllm.records.fields import RecordError

BINDING = "the run's Commit block"

# The interim required-value manifest for the native committer + FA2 hidden tap on an FA2 dense model: the canary rows' `required_classes`
# minus the two derived-from-bound norm classes (protocol clauses with no leaves, checked for presence only when declared).  When a model's
# producers differ (FA3's tap on Hopper), `commit_delta --required-classes` overrides it (kept in verdict.json), never to hide a missing one.
REQUIRED_CLASSES_DEFAULT = ("input_token_ids", "instance_outputs_native", "sampled_token_ids", "weights", "fa2_hidden_m1_stream")
LEAF_HOWS = ("bound-as-leaf", "registered-input")
HIDDEN_CLASS_BY_FA = {2: "fa2_hidden_m1_stream", 3: "fa3_hidden_m1_stream"}


def required_classes_default(flash_attn_version: int | None) -> tuple[str, ...]:
    """The default manifest keyed by the engine's attention implementation (the TargetProfile fact `effective_flash_attn_version`,
    the same fact that selects the Bet A attention Definition and the Match pattern): the attention hidden class is the FA2 tap's under FA2
    and the FA3 tap's (`fa3_hidden_m1_stream`, hidden_source.HiddenM1SourceFA3) under FA3.  None -> the FA2 default (rows recorded without the fact)."""
    hidden = HIDDEN_CLASS_BY_FA.get(int(flash_attn_version) if flash_attn_version is not None else 2, "fa2_hidden_m1_stream")
    return tuple(hidden if c == "fa2_hidden_m1_stream" else c for c in REQUIRED_CLASSES_DEFAULT)

def hidden_classes_required(manifest: Mapping[str, Any] | None) -> bool | None:
    """Whether the required-value manifest names an attention hidden-stream identity (FA2's or FA3's tap); None without a manifest.
    A Program whose attention is its own Calls (the exact-int8 profile's) names none, and its Commit attaches no hidden tap."""
    if manifest is None:
        return None
    families = (manifest.get("populations") or {}).get("by_family") or {}
    return any(int(families.get(f, 0) or 0) > 0 for f in HIDDEN_CLASS_BY_FA.values())


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


def binding_check(bnd: CommitBinding | Mapping[str, Any], *, min_openings: int = 64) -> tuple[bool, str]:
    """BIND-01: (1) the binding map was built and its context (roots, layout) is the run's; (2) every identity of
    the collector-independent expected population has a binding (a duplicate / alias / extra copy never counts for a missing one --
    dropping `/1` and binding `/0` twice fails here); (3) openings sampled from the EXPECTED population all had a binding and all verified
    (retain=none: the values come from replay and are checked by openings_after_release)."""
    try:
        b = CommitBinding.of(bnd, BINDING)
    except RecordError as e:
        return False, f"binding record refused: {e}"
    if b.error:
        return False, f"binding map not built: {b.error}"
    if not b.context_ok:
        return False, f"binding-map context: {b.context_why}"
    cov = b.coverage
    if cov is None:
        return False, "identity coverage: not recorded"
    if cov.error:                                                                # incomplete manifest (unmodelled families) -> fail closed, named
        return False, f"identity coverage: {cov.error}"
    if not cov.ok:
        if cov.missing_n == 0 and (cov.ambiguous_n or 0) > 0:
            # every required identity is bound, but not UNAMBIGUOUSLY: two leaf entries under one identity
            amb = cov.ambiguous[0] if cov.ambiguous else {}
            return False, (f"identity coverage: {cov.ambiguous_n} of {cov.expected_n} required identities have more than one binding "
                           f"(C1 requires one unambiguous binding; first: step {amb.get('step')} {amb.get('op_path')}/{amb.get('output_member')} inv {amb.get('invocation')} "
                           f"req {amb.get('request_id')} rank {amb.get('rank')} bound {amb.get('bindings_n')} x at {amb.get('bound_at')})")
        miss = cov.missing[0] if cov.missing else {}
        return False, (f"identity coverage: {cov.missing_n} of {cov.expected_n} required identities have no binding "
                       f"(first: step {miss.get('step')} {miss.get('op_path')}/{miss.get('output_member')} inv {miss.get('invocation')} req {miss.get('request_id')} rank {miss.get('rank')})")
    po = b.entry.get("openings_population") or {}
    if int(po.get("unbound_n") or 0) > 0:
        return False, f"{po['unbound_n']} population-drawn openings had no binding (first: {json_short((po.get('unbound') or [{}])[0])})"
    if not po.get("skipped"):
        if int(po.get("opened") or 0) < min(min_openings, int(po.get("n") or 0)):
            return False, f"population openings opened {po.get('opened')} < {min(min_openings, int(po.get('n') or 0))}"
        if int(po.get("verified") or 0) != int(po.get("opened") or 0):
            return False, f"population openings verified {po.get('verified')} != opened {po.get('opened')}"
    elif int(po.get("verified") or 0) != int(po.get("opened") or 0):
        # retain=host + exclude: the host share opened here must verify; the deferred share is opened by replay (openings_after_release)
        return False, f"population openings (host share) verified {po.get('verified')} != opened {po.get('opened')}"
    return True, "PASS"


#: what a FAIL of each rule finds against its claim; a group's code is its first failing rule's
REJECT: dict[str, VC] = {name: code for code, names in (
    (VC.PUBLIC_IO_MISMATCH, ("runtime_match/tokens_equal", "boundary_linkage/ordered_linkage")),
    (VC.INVALID_VALUE, ("runtime_match/fa2_zero_rows",)),
    (VC.INVALID_COMMITMENT, ("runtime_match/collector_errors", "runtime_match/bounded_staging", "program_source_identity/binding_context")),
    (VC.INVALID_OPENING, ("runtime_match/openings", "runtime_match/openings_after_release", "runtime_match/openings_population")),
    (VC.CHECK_MISMATCH, ("runtime_match/oracle_compare",)),
    (VC.INVALID_COMPILED_RESULT, ("runtime_match/compiled_check",)),
    (VC.RELATION_REJECTED, ("local_replay/sampled_replay",)),
    (VC.CHALLENGE_MISMATCH, ("local_replay/seed_of_record",)),
    (VC.COVERAGE_MISMATCH, ("runtime_match/reference_integrity", "local_replay/population_gaps", "local_replay/population_scope", "local_replay/attribution",
                            "execution_extent/rows_past_the_stop", "required_value_coverage/identity_coverage", "required_value_coverage/manifest_coverage",
                            "required_value_coverage/required_classes")),
    (VC.EXPECTATION_MISMATCH, ("checkpoint_binding/weights_of_record", "local_replay/cublas_route_pin", "execution_extent/executed_prefix_of_record", "execution_extent/match_account_link",
                               "required_value_coverage/manifest_of_record_verified", "program_source_identity/identity_layout")),
) for name in names}


def reject_of(d: Mapping[str, Any]) -> VC | None:
    """The code a FAIL of the check document `d` stands for: its rule's, or on a group its first failing rule's."""
    name = str(d.get("name"))
    if "/" not in name:
        name = next((r for r, o in _d(_d(d.get("scope")).get("rules")).items() if o == FAIL), name)
    return REJECT.get(name)


# ---------------------------------------------------------------------------------------------------------------------- helpers

def _d(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _first(seq: Any) -> Any:
    return (seq or [None])[0] if isinstance(seq, list) else None


def _components(commit: dict) -> dict:
    vc = commit.get("value_correspondence")
    return _d(_d(vc).get("components")) if isinstance(vc, dict) else {}


def _replay_of(commit: dict) -> dict | None:
    """The sampled replay block commit_verdict grades: the top level, else a `components.sampled_replay` that says it IS a sampled exact replay."""
    sr = commit.get("sampled_replay")
    if isinstance(sr, dict):
        return sr
    c = _components(commit).get("sampled_replay")
    if isinstance(c, dict) and "sampled-exact-replay" in str(c.get("method") or c.get("schema") or c.get("reference") or ""):
        return c
    return None


def _linkage_of(commit: dict) -> dict | None:
    lk = commit.get("boundary_linkage")
    if isinstance(lk, dict):
        return lk
    c = _components(commit).get("boundary_linkage")
    return c if isinstance(c, dict) else None


def _oracle_of(commit: dict) -> dict | None:
    oc = commit.get("oracle_compare")
    if isinstance(oc, dict):
        return oc
    vc = commit.get("value_correspondence")
    if isinstance(vc, dict) and isinstance(vc.get("oracle_compare"), dict):
        return vc["oracle_compare"]
    c = _components(commit).get("oracle_compare")
    return c if isinstance(c, dict) else None


def _is_row_procedure(commit: dict) -> bool:
    bnd = commit.get("binding")
    return isinstance(bnd, dict) and bool(bnd.get("population") or bnd.get("coverage"))


def _binding_of(commit: dict) -> CommitBinding | None:
    """The run's binding map as `records.commit_binding` types it; None when the run has none."""
    b = commit.get("binding")
    return None if b is None else CommitBinding.of(b, BINDING)


def _base_inputs(commit: dict, ctx: dict) -> dict[str, Any]:
    out = {"program_digest": ctx.get("program_digest") or _d(commit.get("binding")).get("program_digest"),
           "manifest_digest": ctx.get("manifest_digest") or _d(commit.get("coverage")).get("manifest_digest"), "run_root": commit.get("run_root")}
    return {k: v for k, v in out.items() if v is not None}


def _leak(table: str, why: str) -> str:
    return f"structural_leak: {table} ({why})"


# ---------------------------------------------------------------------------------------------------------------------- the rules
# Every function mirrors ONE predicate of verdict.commit_verdict (cited by its line's comment tag); none adds a condition.  `ctx`
# carries the record-level facts a rule needs beyond the run's commit block (program / manifest digests, file digests, manifest_verify).

def _mk(name: str, rule: str, claim: str, compat: list[str], commit: dict, ctx: dict, **kw: Any) -> CheckResult:
    inputs = dict(_base_inputs(commit, ctx))
    inputs.update(kw.pop("inputs", {}) or {})
    code = code_of(kw.pop("outcome", NOT_RUN), REJECT[name])
    return CheckResult(name=name, code=code, claim=claim, inputs=inputs, compat=compat, rule=rule, **kw)


def rule_tokens_equal(commit: dict, tokens_equal: Any, ctx: dict) -> CheckResult:
    """v1: `if tokens_equal is not True: FAIL 'control vs instrumented tokens differ'`."""
    if tokens_equal is True:
        out, diffs, reason = PASS, [], None
    elif tokens_equal is False:
        out, diffs, reason = FAIL, [{"name": "tokens_differ", "control_vs_instrumented": False}], "control vs instrumented tokens differ"
    else:
        out, diffs, reason = INSUFFICIENT, [{"name": "tokens_equal_unknown", "value": tokens_equal}], "control vs instrumented tokens differ (not recorded)"
    return _mk("runtime_match/tokens_equal", "X-03 tokens", "The instrumented engine served exactly the control engine's token ids for every request.",
               ["tokens_equal_ref", "commit_summary.tokens_equal"], commit, ctx, outcome=out, differences=diffs, reason=reason,
               scope={"tokens_equal": tokens_equal})


def rule_reference_integrity(commit: dict, ctx: dict) -> CheckResult:
    """v1: counters missing -> FAIL; expected <= 0 -> FAIL; expected != committed -> FAIL."""
    ex = _d(commit.get("extra"))
    n_exp, n_com = ex.get("expected_tensors"), ex.get("committed_tensors")
    if n_exp is None or n_com is None:
        out, diffs, reason = INSUFFICIENT, [{"name": "counters_missing", "expected_tensors": n_exp, "committed_tensors": n_com}], \
            "reference integrity counters missing (expected_tensors/committed_tensors): committer does not report them"
    elif _int(n_exp) <= 0:
        out, diffs, reason = FAIL, [{"name": "nothing_bound", "expected_tensors": _int(n_exp)}], "expected_tensors == 0: the required-value manifest bound nothing"
    elif _int(n_exp) != _int(n_com):
        out, diffs, reason = FAIL, [{"name": "acquisitions_lost", "expected_tensors": _int(n_exp), "committed_tensors": _int(n_com)}], \
            f"expected_tensors {n_exp} != committed_tensors {n_com} (refused/lost acquisitions -> incomplete root)"
    else:
        out, diffs, reason = PASS, [], None
    return _mk("runtime_match/reference_integrity", "X-03 counters",
               "Every value acquisition the collector attempted was bound (attempted == committed, both > 0).",
               ["extra.expected_tensors", "extra.committed_tensors"], commit, ctx, outcome=out, differences=diffs, reason=reason,
               scope={"attempted": n_exp, "committed": n_com})


def rule_fa2_zero_rows(commit: dict, ctx: dict) -> CheckResult:
    z = commit.get("fa2_zero_rows")
    if z is None:
        out, diffs, reason = NOT_RUN, [], None
    elif _int(z) > 0:
        out, diffs, reason = FAIL, [{"name": "fa2_zero_rows", "rows": _int(z)}], f"{z} FA2 hidden rows bound as all-zero bytes (tap staged after it was zeroed: ordering fault)"
    else:
        out, diffs, reason = PASS, [], None
    return _mk("runtime_match/fa2_zero_rows", "FA2 tap ordering", "No attention hidden-value row was bound as all-zero bytes (the tap was staged before it was zeroed).",
               ["fa2_zero_rows"], commit, ctx, outcome=out, differences=diffs, reason=reason, scope={"rows_checked": z is not None})


def rule_collector_errors(commit: dict, ctx: dict) -> CheckResult:
    ex = _d(commit.get("extra"))
    errs_n = ex.get("errors_n")
    if errs_n is None:
        errs_n = len(ex.get("errors") or []) if "errors" in ex else commit.get("errors_n")
    if errs_n is None:
        errs_n = 0
    failure = ex.get("collector_failure")
    if failure:
        out, diffs, reason = FAIL, [{"name": "collector_failure", "first": str(failure)[:200]}], f"collector failed: {str(failure)[:200]}"
    elif _int(errs_n) > 0:
        first = _first(ex.get("errors")) or "?"
        out, diffs, reason = FAIL, [{"name": "collector_errors", "n": _int(errs_n), "first": str(first)[:160]}], f"{errs_n} collector error(s): {str(first)[:160]}"
    else:
        out, diffs, reason = PASS, [], None
    return _mk("runtime_match/collector_errors", "X-03 errors", "No collector, arena or hook error was recorded during the instrumented run.",
               ["extra.errors_n", "extra.errors", "extra.collector_failure"], commit, ctx, outcome=out, differences=diffs, reason=reason, scope={"errors_n": _int(errs_n)})


def rule_bounded_staging(commit: dict, ctx: dict) -> CheckResult:
    bs = _d(commit.get("extra")).get("bounded_staging")
    if not isinstance(bs, dict):
        return _mk("runtime_match/bounded_staging", "fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=NOT_RUN)
    if bs.get("learned_only_steps"):
        steps = list(bs["learned_only_steps"])
        return _mk("runtime_match/bounded_staging", "fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=FAIL, differences=[{"name": "learned_only_steps", "n": len(steps), "first": steps[:8]}],
                   reason=f"bounded staging: {len(steps)} learned-only step(s) {steps[:8]} on a record run (no root)")
    if bs.get("learn_only"):
        return _mk("runtime_match/bounded_staging", "fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=FAIL, differences=[{"name": "learn_only", "value": True}],
                   reason="bounded staging: the record run was finalized in learn-only mode")
    return _mk("runtime_match/bounded_staging", "fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
               ["extra.bounded_staging"], commit, ctx, outcome=PASS, scope={k: bs.get(k) for k in ("learn_only", "learned_only_steps", "mode") if k in bs})


def rule_openings(commit: dict, ctx: dict, min_openings: int) -> CheckResult:
    """v1: skipped and no after-release record -> FAIL; not skipped: n < min -> FAIL, verified != n -> FAIL; skipped (host share): verified != n -> FAIL."""
    ops = _d(commit.get("openings"))
    oar = commit.get("openings_after_release")
    n_open, n_ok = _int(ops.get("n")), _int(ops.get("verified"))
    claim = "Every challenge-drawn opening of the committed bytes verified against the run root, and at least min_openings were drawn (C3)."
    compat = ["openings.n", "openings.verified", "openings.skipped", "openings.deferred_replay_n", "commit_summary.openings"]
    scope = {"sample_size": n_open, "verified": n_ok, "min_openings": min_openings, "skipped": ops.get("skipped"), "challenge_n": ops.get("challenge_n"),
             "deferred_replay_n": ops.get("deferred_replay_n")}
    if ops.get("skipped") and oar is None:
        return _mk("runtime_match/openings", "C3 openings", claim, compat, commit, ctx, outcome=INSUFFICIENT, scope=scope,
                   differences=[{"name": "openings_skipped_no_replay", "skipped": ops.get("skipped")}],
                   reason=f"openings skipped ({ops.get('skipped')}) and no openings-after-release (replay) record")
    if not ops.get("skipped"):
        if n_open < min_openings:
            return _mk("runtime_match/openings", "C3 openings", claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                       differences=[{"name": "openings_too_few", "checked": n_open, "min": min_openings}], reason=f"openings checked {n_open} < {min_openings}")
        if n_ok != n_open:
            return _mk("runtime_match/openings", "C3 openings", claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                       differences=[{"name": "openings_unverified", "checked": n_open, "verified": n_ok, "failed": (ops.get("failed") or [])[:8]}],
                       reason=f"openings verified {n_ok} != checked {n_open}")
    elif n_ok != n_open:
        return _mk("runtime_match/openings", "C3 openings", claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                   differences=[{"name": "openings_unverified", "checked": n_open, "verified": n_ok, "share": "host"}],
                   reason=f"openings (host share) verified {n_ok} != checked {n_open}")
    return _mk("runtime_match/openings", "C3 openings", claim, compat, commit, ctx, outcome=PASS, scope=scope)


def rule_openings_after_release(commit: dict, ctx: dict, min_openings: int) -> CheckResult:
    """v1's `if oar is not None:` block + the `elif deferred picks and no record` FAIL."""
    ops = _d(commit.get("openings"))
    oar = commit.get("openings_after_release")
    bnd = _d(commit.get("binding"))
    po = _d(bnd.get("openings_population"))
    claim = ("After the raw device staging was released, every opening (the challenge's and every pick the in-run stages deferred to replay) "
             "verified against the ORIGINAL roots, the replay reproduced the run root, and no raw byte stayed on the device.")
    compat = ["openings_after_release.n", "openings_after_release.verified", "openings_after_release.source", "openings_after_release.replay_root_equal",
              "openings_after_release.replay_errors_n", "openings_after_release.retained_device_after", "openings_after_release.deferred_received_positions"]
    name, rule = "runtime_match/openings_after_release", "openings after release"
    def_a, def_b = _int(ops.get("deferred_replay_n")), _int(po.get("deferred_replay_n"))
    if oar is None:
        if def_a or def_b:
            return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT, differences=[{"name": "deferred_picks_unopened", "challenge": def_a, "population": def_b}],
                       reason="picks deferred to replay (excluded families) and no openings-after-release record", scope={"deferred": def_a + def_b})
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    oar = _d(oar)
    n_a, ok_a = _int(oar.get("n")), _int(oar.get("verified"))
    scope = {"sample_size": n_a, "verified": ok_a, "source": oar.get("source"), "replay_root_equal": oar.get("replay_root_equal"),
             "retained_device_after": oar.get("retained_device_after"), "deferred_received_n": oar.get("deferred_received_n"), "forced_n": oar.get("forced_n")}

    def fail(diff: dict, reason: str) -> CheckResult:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, differences=[diff], reason=reason, scope=scope)

    if n_a < min_openings:
        return fail({"name": "openings_too_few", "checked": n_a, "min": min_openings}, f"openings after release checked {n_a} < {min_openings}")
    if ok_a != n_a:
        return fail({"name": "openings_unverified", "checked": n_a, "verified": ok_a, "source": oar.get("source")},
                    f"openings after release verified {ok_a} != checked {n_a} (source {oar.get('source')})")
    src = str(oar.get("source") or "")
    if "replay" in src and oar.get("replay_root_equal") is not True:
        return fail({"name": "replay_root_differs", "replay_root_equal": oar.get("replay_root_equal")},
                    "replay run root differs from the original run root (replay-to-open is not reproducing the committed bytes)")
    if "replay" in src and _int(oar.get("replay_errors_n")) > 0:
        return fail({"name": "replay_collector_errors", "n": _int(oar.get("replay_errors_n"))}, f"replay run recorded {oar.get('replay_errors_n')} collector error(s)")
    if _int(oar.get("retained_device_after")) > 0:
        return fail({"name": "raw_bytes_retained_on_device", "bytes": _int(oar.get("retained_device_after"))},
                    f"raw/level bytes still on the device after release: {oar.get('retained_device_after')}")
    if def_a or def_b:
        pa = [tuple(p) for p in (ops.get("deferred_positions") or [])]
        pb = [tuple(p) for p in (po.get("deferred_positions") or [])]
        got = {tuple(p) for p in (oar.get("deferred_received_positions") or [])}
        positional_writer = any(k in d for d, k in ((ops, "deferred_positions"), (po, "deferred_positions"), (oar, "deferred_received_positions")))
        if pa or pb or got or positional_writer:
            if len(pa) != def_a or len(pb) != def_b:
                return fail({"name": "deferred_positions_missing", "challenge": def_a, "population": def_b, "positions": [len(pa), len(pb)]},
                            f"deferred picks recorded by count ({def_a} challenge, {def_b} population) but their positions are not on the record ({len(pa)}, {len(pb)}) -- "
                            "the after-release stage cannot be checked against them")
            miss = sorted((set(pa) | set(pb)) - got)
            if miss:
                return fail({"name": "deferred_picks_not_received", "n": len(miss), "first": [list(m) for m in miss[:4]]},
                            f"{def_a} challenge + {def_b} population pick(s) deferred to replay; {len(miss)} position(s) not received by openings after release "
                            f"(first: {miss[:4]}) -- every deferred pick must be opened by replay")
        elif _int(oar.get("deferred_received_n")) < max(def_a, def_b):
            return fail({"name": "deferred_picks_not_received", "expected": max(def_a, def_b), "received": _int(oar.get("deferred_received_n"))},
                        f"{def_a} challenge + {def_b} population pick(s) deferred to replay, openings after release received {oar.get('deferred_received_n')} "
                        "(every deferred pick must be opened by replay)")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope)


def rule_oracle_compare(commit: dict, ctx: dict) -> CheckResult:
    """v1: value_correspondence.result False -> FAIL (the failing component's own cause); oracle_compare.result False -> FAIL."""
    vc = commit.get("value_correspondence")
    oc = commit.get("oracle_compare") if isinstance(commit.get("oracle_compare"), dict) else None
    claim = ("No committed value differs from the Match oracle's (form (B): the Match capture's instance-output snapshots and sampled ids of "
             "the same attempt) where the oracle looked; scope states how much it looked at.")
    compat = ["oracle_compare.result", "oracle_compare.mismatch_n", "value_correspondence.result", "value_correspondence.components"]
    name, rule = "runtime_match/oracle_compare", "C2 form (B)"
    _oc = _oracle_of(commit) or {}
    att = _d(_oc.get("attribution"))
    scope = {"compared": _oc.get("compared"), "mismatch_n": _oc.get("mismatch_n"), "complete": _oc.get("complete"), "grade": _oc.get("grade"),
             "attribution_complete": att.get("complete"), "attribution_pairs": att.get("pairs"), "compared_by_family": _oc.get("compared_by_family"),
             "method": (_d(vc).get("method") if isinstance(vc, dict) else None)}
    if isinstance(vc, dict) and vc.get("result") is False:
        oc_in = _d(vc.get("oracle_compare"))
        oc_top = oc or {}
        comps = _d(vc.get("components"))
        failing = [(k, c) for k, c in comps.items() if isinstance(c, dict) and c.get("result") is False]
        if oc_in.get("result") is False and oc_in.get("why"):
            why = oc_in["why"]
        elif oc_top.get("result") is False and (oc_top.get("why") or oc_in.get("why")):
            why = oc_top.get("why") or oc_in.get("why")
        elif failing:
            why = "; ".join(f"{k}: {c.get('why') or 'FAIL'}" for k, c in failing)
        else:
            why = vc.get("why") or f"value correspondence FAIL ({vc.get('method')})"
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, reason=str(why),
                   differences=[{"name": "value_correspondence_fail", "components_failing": [k for k, _ in failing], "mismatch_n": _oc.get("mismatch_n"),
                                 "first_mismatch": _first(_oc.get("mismatch"))}])
    if oc is not None and oc.get("result") is False:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                   reason=oc.get("why") or f"C2 oracle mismatch: {oc.get('mismatch_n')} identities differ from the Match oracle",
                   differences=[{"name": "oracle_mismatch", "mismatch_n": oc.get("mismatch_n"), "first_mismatch": _first(oc.get("mismatch"))}])
    if not _oc and not isinstance(vc, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN, scope=scope)
    notes = []
    if _oc and _oc.get("result") is not True:
        notes.append(f"form (B) {_oc.get('grade') or 'partial'}: {str(_oc.get('why') or '')[:200]}")
    notes.append(_leak("oracle_compare.NO_ORACLE_FAMILY_PREFIXES", "kernel-interior streams excluded from the oracle by name prefix"))
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, notes=notes)


def rule_compiled_check(commit: dict, ctx: dict) -> CheckResult:
    """commit_delta's two post-verdict compiled conditions (S3): kernel replay exact; ring refused nothing."""
    vc = _d(commit.get("value_correspondence"))
    ckc = vc.get("compiled_check")
    cs = _d(_d(commit.get("compiled_source")).get("stats"))
    claim = "Under compiled acquisition every kernel launch replays exactly and the compiled ring refused / mis-shaped no acquisition (S3)."
    compat = ["value_correspondence.compiled_check", "compiled_source.stats"]
    name, rule = "runtime_match/compiled_check", "compiled kernel check"
    if not isinstance(ckc, dict) and not cs:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    if isinstance(ckc, dict) and not ckc.get("ok"):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope={k: ckc.get(k) for k in ("launches_exact", "launches_checked", "launches_missing", "linkage", "equal", "checked")},
                   differences=[{"name": "compiled_launches_inexact", **{k: ckc.get(k) for k in ("launches_exact", "launches_checked", "launches_missing", "equal", "checked", "error")}}],
                   reason=f"compiled C2 FAIL: {ckc.get('launches_exact')}/{ckc.get('launches_checked')} launches exact, missing {ckc.get('launches_missing')}, linkage {ckc.get('linkage')} {ckc.get('error', '')}")
    if cs and (cs.get("refused_capture_alloc") or cs.get("shape_mismatch") or cs.get("oversize_in_capture")):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=dict(cs),
                   differences=[{"name": "compiled_ring_refused", **{k: cs.get(k) for k in ("refused_capture_alloc", "shape_mismatch", "oversize_in_capture")}}],
                   reason=f"compiled ring refused acquisitions: refused_capture_alloc {cs.get('refused_capture_alloc')} shape_mismatch {cs.get('shape_mismatch')} oversize_in_capture {cs.get('oversize_in_capture')}")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope={**{k: ckc.get(k) for k in ("launches_exact", "launches_checked") if isinstance(ckc, dict)}, **cs})


def rule_binding_context(commit: dict, ctx: dict) -> CheckResult:
    """binding_check (1): the map was built (no error) and its context (roots, layout) is the run's."""
    b = _binding_of(commit)
    claim = "The binding map was built for this run: its Program digest, layout and roots are the run's own (BIND-01 (1))."
    compat = ["binding.error", "binding.context.ok", "binding.context.why", "binding.program_digest", "binding.map_digest"]
    name, rule = "program_source_identity/binding_context", "BIND-01 (1)"
    if b is None:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    bnd = b.entry
    scope = {"map_digest": bnd.get("map_digest"), "map_entries_n": bnd.get("map_entries_n"), "program_digest_source": bnd.get("program_digest_source"),
             "layout": bnd.get("layout"), "rules_version": bnd.get("rules_version")}
    inputs = {"binding_program_digest": b.program_digest, "root_program_digest": bnd.get("root_program_digest"), "map_digest": bnd.get("map_digest")}
    if b.error:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, differences=[{"name": "binding_map_not_built", "error": b.error[:300]}],
                   reason=f"binding map not built: {b.error}")
    if not b.context_ok:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, differences=[{"name": "binding_context_mismatch", "why": b.context_why}],
                   reason=f"binding-map context: {b.context_why}")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs)


def rule_identity_coverage(commit: dict, ctx: dict) -> CheckResult:
    """binding_check (2): identity coverage of the expected population (error -> refusal; missing; ambiguous)."""
    b = _binding_of(commit)
    claim = "Every identity of the required-value population has exactly one committed binding (a duplicate never covers a missing one)."
    compat = ["binding.coverage.ok", "binding.coverage.error", "binding.coverage.missing_n", "binding.coverage.ambiguous_n", "binding.coverage.expected_n", "binding.coverage.bound_n"]
    name, rule = "required_value_coverage/identity_coverage", "BIND-01 (2) / MAN-02"
    if b is None or b.coverage is None:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    cov = b.coverage
    pop = {} if b.population is None else b.population.entry
    scope = {"expected_n": cov.expected_n, "bound_n": cov.bound_n, "requests": pop.get("requests"), "steps": pop.get("steps"),
             "population_source": str(pop.get("source") or "")[:120], "extra_n": cov.entry.get("extra_n"), "physical_copies_n": cov.entry.get("physical_copies_n")}
    notes = []
    if "required-manifest" not in str(pop.get("source") or ""):
        notes.append(_leak("binding.MEMBER_RULES", "the expected population's members come from a module-class table, not the Program"))
    if b.error:                              # already FAIL under binding_context; the coverage was never computed
        return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT, scope=scope, notes=notes,
                   differences=[{"name": "binding_map_not_built", "error": b.error[:300]}], reason=f"binding map not built: {b.error}")
    if cov.error:
        err = cov.error
        unm = cov.entry.get("unmodelled")
        diffs = [{"name": "manifest_incomplete", "error": err[:400], "unmodelled": dict(list(_d(unm).items())[:8]) if unm else None}]
        if "unmodelled" in err:
            notes.append(_leak("required_manifest.members_for", "a Program family is INTERIOR because the vocabulary names no member for it -- by name"))
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes, differences=diffs, reason=f"identity coverage: {err}")
    if not cov.ok:
        if cov.missing_n == 0 and (cov.ambiguous_n or 0) > 0:
            amb = cov.ambiguous[0] if cov.ambiguous else {}
            return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes,
                       differences=[{"name": "identities_ambiguous", "n": cov.ambiguous_n, "expected_n": cov.expected_n, "first": amb}],
                       reason=(f"identity coverage: {cov.ambiguous_n} of {cov.expected_n} required identities have more than one binding "
                               f"(C1 requires one unambiguous binding; first: step {amb.get('step')} {amb.get('op_path')}/{amb.get('output_member')} inv {amb.get('invocation')} "
                               f"req {amb.get('request_id')} rank {amb.get('rank')} bound {amb.get('bindings_n')} x at {amb.get('bound_at')})"))
        miss = cov.missing[0] if cov.missing else {}
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes,
                   differences=[{"name": "identities_unbound", "n": cov.missing_n, "expected_n": cov.expected_n, "first": miss}],
                   reason=(f"identity coverage: {cov.missing_n} of {cov.expected_n} required identities have no binding "
                           f"(first: step {miss.get('step')} {miss.get('op_path')}/{miss.get('output_member')} inv {miss.get('invocation')} req {miss.get('request_id')} rank {miss.get('rank')})"))
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, notes=notes)


def rule_openings_population(commit: dict, ctx: dict, min_openings: int) -> CheckResult:
    """binding_check (3): openings sampled from the EXPECTED population all had a binding and all verified."""
    b = _binding_of(commit)
    claim = "Every opening drawn from the required-value population had a binding and verified (BIND-01 (3); retain=none: the deferred share opens by replay)."
    compat = ["binding.openings_population.n", "binding.openings_population.unbound_n", "binding.openings_population.opened", "binding.openings_population.verified", "binding.openings_population.skipped"]
    name, rule = "runtime_match/openings_population", "BIND-01 (3)"
    if b is None:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    if b.error or not b.context_ok or (b.coverage is not None and (b.coverage.error or not b.coverage.ok)):
        # v1 returned from binding_check before reaching (3): the population openings were not graded on this run
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN, notes=["not graded: binding_context / identity_coverage failed first (v1 binding_check returns at the first failing part)"])
    po = _d(b.entry.get("openings_population"))
    scope = {"sample_size": po.get("n"), "bound": po.get("bound"), "opened": po.get("opened"), "verified": po.get("verified"), "skipped": po.get("skipped"),
             "families": po.get("families"), "deferred_replay_n": po.get("deferred_replay_n"), "late_step_in_sample": po.get("late_step_in_sample")}
    if _int(po.get("unbound_n")) > 0:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                   differences=[{"name": "population_openings_unbound", "n": _int(po.get("unbound_n")), "first": _first(po.get("unbound"))}],
                   reason=f"{po['unbound_n']} population-drawn openings had no binding (first: {json_short(_d(_first(po.get('unbound'))))})")
    want = min(min_openings, _int(po.get("n")))
    if not po.get("skipped"):
        if _int(po.get("opened")) < want:
            return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, differences=[{"name": "population_openings_too_few", "opened": _int(po.get("opened")), "min": want}],
                       reason=f"population openings opened {po.get('opened')} < {want}")
        if _int(po.get("verified")) != _int(po.get("opened")):
            return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                       differences=[{"name": "population_openings_unverified", "opened": _int(po.get("opened")), "verified": _int(po.get("verified"))}],
                       reason=f"population openings verified {po.get('verified')} != opened {po.get('opened')}")
    elif _int(po.get("verified")) != _int(po.get("opened")):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                   differences=[{"name": "population_openings_unverified", "opened": _int(po.get("opened")), "verified": _int(po.get("verified")), "share": "host"}],
                   reason=f"population openings (host share) verified {po.get('verified')} != opened {po.get('opened')}")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope)


def rule_manifest_coverage(commit: dict, ctx: dict) -> CheckResult:
    """v1: `cov = ex.coverage or commit.coverage; if cov is not None and not cov.ok: FAIL`."""
    ex = _d(commit.get("extra"))
    cov = ex.get("coverage") or commit.get("coverage")
    claim = "Every identity the required-value manifest names (Program x query, per request / step / member / element range) was committed with its full extent."
    compat = ["coverage.ok", "coverage.missing_n", "coverage.missing_by_family", "coverage.families_missing_entirely", "coverage.error", "coverage.manifest_digest", "commit_summary.coverage"]
    name, rule = "required_value_coverage/manifest_coverage", "MAN-01"
    if not isinstance(cov, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    scope = {"checked": cov.get("checked"), "checked_by_family": cov.get("checked_by_family"), "steps_committed_n": len(cov.get("steps_committed") or []),
             "composition_mismatch_n": cov.get("composition_mismatch_n"), "committed_extras_n": cov.get("committed_extras_n"), "duplicate_names_n": cov.get("duplicate_names_n"),
             "engine_step_basis": cov.get("engine_step_basis")}
    inputs = {"manifest_digest": cov.get("manifest_digest")}
    mor = _d(cov.get("manifest_of_record"))
    if mor:
        scope["manifest_of_record"] = {k: mor.get(k) for k in ("identities_n", "required_n", "not_executed_n", "edited")}
    # the manifest this grades against was built by required_manifest.members_for: a Program family with no vocabulary member is INTERIOR by name
    notes = [_leak("required_manifest.members_for", "a Program family is INTERIOR because the vocabulary names no member for it -- by name")]
    if not cov.get("ok", False):
        fam = cov.get("families_missing_entirely") or []
        first = _d(_first(cov.get("missing")))
        if cov.get("error") and not cov.get("missing_n"):
            return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, notes=notes, differences=[{"name": "coverage_error", "error": str(cov["error"])[:400]}],
                       reason=f"manifest coverage: {cov['error']}")
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, notes=notes,
                   differences=[{"name": "identities_uncovered", "missing_n": cov.get("missing_n"), "missing_by_family": cov.get("missing_by_family"),
                                 "families_missing_entirely": fam[:8], "first": first, "error": (str(cov.get("error"))[:300] if cov.get("error") else None)}],
                   reason=(f"manifest coverage: {cov.get('missing_n')} required identities uncovered ({cov.get('missing_by_family')}); "
                           + (f"whole families missing: {fam}; " if fam else "")
                           + f"first: step {first.get('step')} {first.get('op_path')}/{first.get('output_member')} inv {first.get('invocation')} want {first.get('want_range')} got {first.get('committed_numel')}"))
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs, notes=notes)


def rule_identity_layout(commit: dict, ctx: dict) -> CheckResult:
    """v1: `idc = ex.identity_check; if idc is not None and not idc.ok: FAIL` (X-03 on identities).  The rule body is native_host's
    (`NativeHostCommitter.identity_check`); this only names its findings."""
    idc = _d(commit.get("extra")).get("identity_check")
    claim = ("Every committed step's identity set is the expected manifest's: every expected module produced >= 1 tensor and none came from outside it, "
             "every attention layer carries the same positive count of hidden rows, exactly one sampled-ids tensor, the same name multiset as step 0 (X-03).")
    compat = ["extra.identity_check.ok", "extra.identity_check.violations", "extra.identity_check.violations_n", "extra.identity_check.steps_checked"]
    name, rule = "program_source_identity/identity_layout", "X-03 identities"
    notes = [_leak("native_host.ACQUIRE_CLASSES", "the expected module set is selected by module class name, not by the Program")]
    if not isinstance(idc, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN, notes=notes)
    man = _d(idc.get("manifest"))
    scope = {"steps_checked": idc.get("steps_checked"), "modules": man.get("modules"), "layers": man.get("attention_layers"), "fa2_hidden_per_layer": man.get("fa2_hidden_per_layer"),
             "sampler_ids": man.get("sampler_ids"), "moe_layers": man.get("moe_layers"), "tp_sites": man.get("tp_sites"), "tp_rank": man.get("tp_rank"),
             "padding": idc.get("padding")}
    if idc.get("ok") is None:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT, scope=scope, notes=notes,
                   differences=[{"name": "identity_check_skipped", "skipped": idc.get("skipped")}], reason=f"identity check: skipped ({idc.get('skipped')})")
    if not idc.get("ok"):
        viols = [v for v in (idc.get("violations") or []) if isinstance(v, dict)]
        diffs: list[dict[str, Any]] = []
        for key in ("fa2_layers_bad", "missing_modules", "unknown_modules", "moe_layers_bad", "tp_sites_bad", "multiplicity_vs_step0", "call_boundaries_bad",
                    "routes_bad"):
            hits = [v for v in viols if v.get(key)]
            if hits:
                diffs.append({"name": key, "steps": [v.get("step") for v in hits][:16], "steps_n": len(hits), "first": {"step": hits[0].get("step"), key: hits[0][key]}})
        sampler = [v for v in viols if man.get("sampler_ids") and v.get("sampler_ids") != man.get("sampler_ids")]
        if sampler:
            diffs.append({"name": "sampler_ids", "steps": [v.get("step") for v in sampler][:16], "steps_n": len(sampler), "first": {"step": sampler[0].get("step"), "sampler_ids": sampler[0].get("sampler_ids")}})
        v0 = _d(_first(viols))
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope={**scope, "violations_n": idc.get("violations_n")}, notes=notes, differences=diffs,
                   reason=f"identity check: {idc.get('violations_n')} step(s) differ from the expected manifest (first: step {v0.get('step')} {json_short(v0)})")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, notes=notes)


def rule_required_classes(commit: dict, ctx: dict, declaration: dict | None, required_classes: Any) -> CheckResult:
    """v1: `if declaration is not None or required_classes is not None: required_classes_check(declaration, required or DEFAULT)` (REL-02 / R-14)."""
    claim = "Every required value class is declared by the committer and every declared leaf-binding class bound > 0 tensors (REL-02: an omitted producer cannot pass)."
    compat = ["required_classes", "required_classes_source", "classes", "declaration_*.json"]
    name, rule = "required_value_coverage/required_classes", "REL-02 / R-14"
    src = str(ctx.get("required_classes_source") or "")
    notes = []
    if required_classes is None or "default" in src or "fallback" in src or src == "cli":
        notes.append(_leak("commit_verdict.REQUIRED_CLASSES_DEFAULT", "the required class set is a table keyed by the attention implementation, not the Program/query manifest"))
    if declaration is None and required_classes is None:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN, notes=notes)
    req = tuple(required_classes) if required_classes is not None else REQUIRED_CLASSES_DEFAULT
    ok, why = required_classes_check(declaration, req)
    cov = class_coverage(declaration)
    scope = {"required": list(req), "declared": {k: {"how": v.get("how"), "tensors": v.get("tensors")} for k, v in cov.items()}, "source": src or None}
    inputs = {"required_classes": list(req)}
    if not ok:
        missing = [c for c in req if c not in cov]
        empty = [c for c, d in cov.items() if d.get("how") in LEAF_HOWS and (d.get("tensors") is None or _int(d.get("tensors")) <= 0)]
        diff = {"name": "declaration_missing"} if declaration is None else {"name": "required_class_undeclared", "classes": missing} if missing else {"name": "class_bound_nothing", "classes": empty}
        return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT if declaration is None else FAIL, scope=scope, inputs=inputs, notes=notes, differences=[diff], reason=why)
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs, notes=notes)


def rule_manifest_verified(commit: dict, ctx: dict) -> CheckResult:
    """row_pod.sh: `required_manifest verify` rebuilt the manifest from this attempt's Program x workload == the one used."""
    mv = ctx.get("manifest_verify")
    claim = "The required-value manifest the Commit used is byte-identical (rows digest) to the one rebuilt from this attempt's Program x workload."
    compat = ["manifest_verify.ok", "manifest_verify.used_digest", "manifest_verify.rebuilt_digest", "stages.txt manifest_verify="]
    name, rule = "required_value_coverage/manifest_of_record_verified", "EXEC-05 / MAN-05"
    if not isinstance(mv, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    scope = {k: mv.get(k) for k in ("rebuilt_engine", "rebuilt_identities", "used_identities", "rebuilt_complete") if k in mv}
    inputs = {"used_digest": mv.get("used_digest"), "rebuilt_digest": mv.get("rebuilt_digest"), "used_program_digest": mv.get("used_program_digest"), "rebuilt_program_digest": mv.get("rebuilt_program_digest")}
    if mv.get("ok") is True:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs)
    return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs,
               differences=[{"name": "manifest_of_record_differs", "used_digest": mv.get("used_digest"), "rebuilt_digest": mv.get("rebuilt_digest"), "why": mv.get("why") or mv.get("reason")}],
               reason=f"manifest of record VERIFY FAILED: used {str(mv.get('used_digest'))[:16]} != rebuilt {str(mv.get('rebuilt_digest'))[:16]}")
