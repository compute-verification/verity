"""The Commit VERDICT as typed records (`vllm-verdict/v1`): every check the Commit decides, by its human name, with what it read, what
it established and what it found -- ONE document (`<row>/verdict.json`) beside the files it is derived from.

    python -m verity_vllm.check.verdict from-record <row_dir> [--out PATH]      rebuild the record from a row's evidence
    python -m verity_vllm.check.verdict mapping                                  the old -> new check-name table (markdown)

WHY.  The v1 Commit records its decision three times over: `commit/verdict.json` (`commit_pass`, one `first_fail_reason` string: the FIRST
rule that failed, the others silent), `commit/runs.jsonl` (per instrumented run, every rule's own block under `commit.*`, the check
names being the field names they grew under -- `weights_pin`, `linkage`, `coverage`, `identity_check`) and `commit_summary.json`, whose
`pass` degenerated to `rc == 0 and tokens_equal` once `openings` / `coverage` stopped being carried on runs.jsonl (F-r19-int-21).  The
verdict of record was `commit/verdict.json` + `commit/sampled_replay_p*.json`, read together by whoever knew to.

WHAT.  A `Verdict` = the seven named checks (groups) and the R*/X* rules under them, each a `CheckResult`:

    name        `<group>` for the group, `<group>/<rule>` for a rule                    (human names: see GROUPS)
    claim       one sentence: what PASS establishes
    inputs      digests / ids of what the check read (program_digest, manifest_digest, run root, file sha256s, the sample seed)
    scope       what was actually established (sample_size, positions, layers, requests, steps ...) -- never more than the record shows
    outcome     PASS | FAIL | INSUFFICIENT_EVIDENCE | NOT_RUN
    differences named, machine-readable findings ({"name": ..., ...}; e.g. `fa2_layers_bad`, `identities_unbound`, `account_missing`)
    compat      the v1 field name(s) the check was read from (the old -> new mapping is `OLD_TO_NEW`, tested for completeness)
    rule        the R*/X* id of the rule (None on a group)
    notes       prose the PASS carries (named gaps, `structural_leak: <table>` where the rule rests on a name table, not on the Program)

The decisions are v1's, unchanged: every rule (`check.commit_rules`, `check.c2_rules`) is `commit_verdict`'s own predicate, evaluated
one by one instead of first-failure-wins, so `Verdict.outcome == PASS` iff v1's `commit_pass` (the record's own `commit_pass` is read
back and a disagreement is named on the record, `compat.reconstruction_mismatch`, never silently resolved).  What changes is the
representation: a missing INPUT (no weights pin supplied, a Match account leg moved away, no sampled replay run) is INSUFFICIENT_EVIDENCE,
a finding against the claim is FAIL; both are "not PASS", as before.

OUTCOME RULE.  A group = the worst of its rules (FAIL > INSUFFICIENT_EVIDENCE > PASS; NOT_RUN only when every rule is NOT_RUN); the
verdict = the worst of its groups, NOT_RUN when the Commit never ran.  A rule v1 skipped when its block was absent (legacy records,
unit fixtures) is NOT_RUN and does not weigh; a rule v1 FAILED for an absent block (a row-procedure record without a weights pin) is
INSUFFICIENT_EVIDENCE and does weigh.

Glossary (v2): Definition, Call, Value, Program, verification query, VU, boundary / required values; runtime correspondence, runtime
occurrence, value acquisition, value observation, occurrence resolution, input provenance, execution evidence, check, verdict,
required-value manifest.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.check import c2_rules as C2
from verity_vllm.check import commit_rules as CR
from verity_vllm.check.commit_rules import _d, _int
from verity_vllm.check.result import FAIL, INSUFFICIENT, NOT_RUN, OUTCOMES, PASS, CheckResult, worst

SCHEMA = "vllm-verdict/v1"

#: the seven named checks (groups), their claims, and the v1 names they were read under
GROUPS: dict[str, dict[str, Any]] = {
    "runtime_match": {
        "claim": ("The committed record is the runtime occurrence's: the instrumented engine served the control's tokens, every attempted "
                  "value acquisition was bound, the committed bytes open against the run root, and no committed value differs from the Match "
                  "oracle where the oracle looked."),
        "compat": ["tokens_equal", "openings", "openings_after_release", "oracle_compare", "value_correspondence", "expected_tensors", "committed_tensors", "errors_n"]},
    "local_replay": {
        "claim": ("A post-commitment sample of required values replays exactly from the committed bytes under the Program's relations "
                  "(C2 form (A)), over the required-value manifest Q(P) of record, with every unevaluated stratum named."),
        "compat": ["sampled_replay", "sampled_replay_p*.json", "replay_cache"]},
    "boundary_linkage": {
        "claim": ("Every (engine step, request) boundary is linked: the declared prompt to the input ids at the arrival step, and each "
                  "sampled id to the next step's input id (C2 ordered boundary linkage)."),
        "compat": ["boundary_linkage", "linkage"]},
    "checkpoint_binding": {
        "claim": ("Every served parameter the Program names equals, by digest, the checkpoint-composed tensor of record (MAN-09), "
                  "with the shards pinned by their manifest sha256."),
        "compat": ["weights_pin"]},
    "execution_extent": {
        "claim": ("Each request's executed prefix of record (served length; the cap on a length stop, served + LAG on an EOS stop) is "
                  "derived from the declared stop rule, linked to the Match account of the same attempt, and no committed row lies past it."),
        "compat": ["executed_prefix", "executed_prefix_of_record", "executed_prefix_facts", "match_account_link", "rows_past_the_stop", "account_missing"]},
    "required_value_coverage": {
        "claim": ("Every identity of the required-value manifest Q(P) has exactly one committed binding, every required value class was "
                  "acquired, and the manifest used is the one rebuilt from this attempt's Program x workload."),
        "compat": ["coverage", "binding.coverage", "required_classes", "required_classes_source", "classes", "manifest_verify"]},
    "program_source_identity": {
        "claim": ("The committed identities are the Program of record's: every step's committed identity set matches the expected manifest "
                  "(X-03: modules, one attention row per layer, one sampled-ids tensor, the same multiset every step) and the binding map "
                  "was built for this Program, layout and run roots."),
        "compat": ["identity", "identity_check", "binding.context"]},
}

#: the v1 name -> the v2 check.  `claim` is the check's; `where` names the v1 document the fact lived in.  Tested for completeness
#: against the reference records (every field of `commit/verdict.json` and of runs.jsonl's `commit.*` block is named here).
OLD_TO_NEW: list[dict[str, str]] = [
    {"old": "commit_pass", "where": "commit/verdict.json, runs.jsonl commit", "new": "verdict.outcome (PASS iff commit_pass)"},
    {"old": "first_fail_reason", "where": "commit/verdict.json", "new": "the FAIL / INSUFFICIENT_EVIDENCE checks' `reason` (every one, not the first); compat.first_fail_reason"},
    {"old": "commit_fail_reason", "where": "runs.jsonl commit", "new": "the FAIL checks' `reason` per run (differences carry `pair`)"},
    {"old": "runs_instrumented", "where": "commit/verdict.json", "new": "verdict.runs.instrumented"},
    {"old": "runs_failed", "where": "commit/verdict.json", "new": "verdict.runs.failed"},
    {"old": "required_classes", "where": "commit/verdict.json", "new": "required_value_coverage/required_classes inputs.required_classes"},
    {"old": "required_classes_source", "where": "commit/verdict.json", "new": "required_value_coverage/required_classes scope.source"},
    {"old": "required_manifest", "where": "commit/verdict.json", "new": "verdict.program_digest / manifest_digest / query_id; required_value_coverage/manifest_coverage"},
    {"old": "moe_source", "where": "commit/verdict.json", "new": "verdict.execution.moe_source"},
    {"old": "negatives", "where": "commit/verdict.json", "new": "verdict.compat.negatives (armed NEGATIVE-TEST flags, by name)"},
    {"old": "run_roots", "where": "commit/verdict.json", "new": "verdict.run_roots; every check's inputs.run_root"},
    {"old": "replay_cache", "where": "commit/verdict.json", "new": "local_replay/sampled_replay scope.replay_cache"},
    {"old": "host_peak", "where": "commit/verdict.json", "new": "verdict.execution.host_peak (a measurement, not a decision)"},
    {"old": "only_arm", "where": "commit/verdict.json (control-only runs)", "new": "verdict.outcome NOT_RUN (no instrumented run)"},
    {"old": "compilation", "where": "commit/verdict.json (control-only runs)", "new": "verdict.execution.compilation"},
    {"old": "tokens_equal_ref", "where": "runs.jsonl", "new": "runtime_match/tokens_equal"},
    {"old": "expected_tensors", "where": "runs.jsonl commit.extra", "new": "runtime_match/reference_integrity"},
    {"old": "committed_tensors", "where": "runs.jsonl commit.extra", "new": "runtime_match/reference_integrity"},
    {"old": "errors_n", "where": "runs.jsonl commit.extra", "new": "runtime_match/collector_errors"},
    {"old": "fa2_zero_rows", "where": "runs.jsonl commit", "new": "runtime_match/fa2_zero_rows"},
    {"old": "openings", "where": "runs.jsonl commit", "new": "runtime_match/openings"},
    {"old": "openings_after_release", "where": "runs.jsonl commit", "new": "runtime_match/openings_after_release"},
    {"old": "bounded_staging", "where": "runs.jsonl commit.extra", "new": "runtime_match/bounded_staging"},
    {"old": "oracle_compare", "where": "runs.jsonl commit", "new": "runtime_match/oracle_compare (form (B))"},
    {"old": "value_correspondence", "where": "runs.jsonl commit", "new": "runtime_match/oracle_compare (the composite's result and components)"},
    {"old": "compiled_check", "where": "runs.jsonl commit.value_correspondence", "new": "runtime_match/compiled_check"},
    {"old": "compiled_source", "where": "runs.jsonl commit", "new": "runtime_match/compiled_check"},
    {"old": "binding.context", "where": "runs.jsonl commit.binding", "new": "program_source_identity/binding_context"},
    {"old": "binding.coverage", "where": "runs.jsonl commit.binding", "new": "required_value_coverage/identity_coverage"},
    {"old": "binding.openings_population", "where": "runs.jsonl commit.binding", "new": "runtime_match/openings_population"},
    {"old": "binding.population", "where": "runs.jsonl commit.binding", "new": "required_value_coverage/identity_coverage scope; local_replay/attribution scope.requests"},
    {"old": "binding.coverage.executed_prefix_of_record", "where": "runs.jsonl commit.binding.coverage", "new": "execution_extent/executed_prefix_of_record"},
    {"old": "binding.coverage.match_account_link", "where": "runs.jsonl commit.binding.coverage", "new": "execution_extent/match_account_link"},
    {"old": "binding.coverage.rows_past_the_stop", "where": "runs.jsonl commit.binding.coverage", "new": "execution_extent/rows_past_the_stop"},
    {"old": "coverage", "where": "runs.jsonl commit (+ commit.extra.coverage)", "new": "required_value_coverage/manifest_coverage"},
    {"old": "identity_check", "where": "runs.jsonl commit.extra", "new": "program_source_identity/identity_layout (X-03)"},
    {"old": "classes", "where": "runs.jsonl commit", "new": "required_value_coverage/required_classes (the declaration read back)"},
    {"old": "sampled_replay", "where": "runs.jsonl commit; commit/sampled_replay_p*.json", "new": "local_replay/sampled_replay, /seed_of_record, /population_gaps, /population_scope"},
    {"old": "boundary_linkage", "where": "runs.jsonl commit; commit/sampled_replay_p*.json", "new": "boundary_linkage/ordered_linkage"},
    {"old": "linkage", "where": "prose", "new": "boundary_linkage"},
    {"old": "weights_pin", "where": "runs.jsonl commit", "new": "checkpoint_binding/weights_of_record"},
    {"old": "executed_prefix", "where": "prose", "new": "execution_extent"},
    {"old": "identity", "where": "prose", "new": "program_source_identity"},
    {"old": "account_missing", "where": "sampled_replay executed_prefix_of_record.facts_of_record", "new": "execution_extent/match_account_link differences[account_missing] -> INSUFFICIENT_EVIDENCE"},
    {"old": "run_root", "where": "runs.jsonl commit", "new": "every check's inputs.run_root"},
    {"old": "rules_version", "where": "runs.jsonl commit", "new": "verdict.execution.rules_version"},
    {"old": "manifest_verify.ok", "where": "commit/manifest_verify.json", "new": "required_value_coverage/manifest_of_record_verified"},
    {"old": "commit_summary.pass", "where": "commit_summary.json", "new": "compat.commit_summary_pass (deprecated, F-r19-int-21: rc == 0 and tokens_equal)"},
    {"old": "commit_summary.rc", "where": "commit_summary.json", "new": "compat.rc; verdict.outcome is not PASS when rc != 0"},
    {"old": "commit_summary.openings", "where": "commit_summary.json", "new": "deprecated (always None since runs.jsonl stopped carrying it); runtime_match/openings is the fact"},
    {"old": "commit_summary.coverage", "where": "commit_summary.json", "new": "deprecated (always None); required_value_coverage/manifest_coverage is the fact"},
    {"old": "commit_summary.tokens_equal", "where": "commit_summary.json", "new": "runtime_match/tokens_equal"},
    {"old": "commit_summary.n_runs / n_control / n_treatment", "where": "commit_summary.json", "new": "verdict.runs"},
    {"old": "stages.txt commit PASS|FAIL", "where": "row_pod.sh stage line", "new": "verdict.outcome"},
    {"old": "stages.txt manifest_verify=", "where": "row_pod.sh stage line", "new": "required_value_coverage/manifest_of_record_verified"},
    {"old": "manifest_verify.rebuilt_digest / manifest_verify.used_digest", "where": "commit/manifest_verify.json", "new": "required_value_coverage/manifest_of_record_verified (differences)"},
    {"old": "binding.error / binding.map_digest / binding.program_digest", "where": "runs.jsonl commit.binding", "new": "program_source_identity/binding_context (inputs / differences)"},
    {"old": "extra.errors", "where": "runs.jsonl commit.extra", "new": "runtime_match/collector_errors (differences.first)"},
    {"old": "declaration_*.json", "where": "commit/ (commit_integ.api.declaration)", "new": "required_value_coverage/required_classes (inputs)"},
    {"old": "sampled_replay_p*.json", "where": "commit/", "new": "local_replay/* (inputs: file sha256s under verdict.inputs)"},
]

#: fields of runs.jsonl's `commit` block that are MEASUREMENTS / EXECUTION LABELS, not verdict facts (listed so the completeness test can
#: tell "not a decision" from "lost")
MEASUREMENT_FIELDS = frozenset({
    "bytes_bound", "finalize_s", "leaves", "persistent_bytes", "root_ready_delay_s", "spans_bytes", "spans_calls", "spans_total_s", "steps",
    "t_ready_minus_last_step_s", "temporary_bytes_peak", "tensors", "weight_registration", "extra",
})


# ---------------------------------------------------------------------------------------------------------------------- the records

@dataclass
class Verdict:
    checks: list[CheckResult]
    outcome: str
    program_digest: str | None = None
    manifest_digest: str | None = None
    query_id: str | None = None
    correspondence_digest: str | None = None          # the Build's `runtime-correspondence/v1` digest (build_summary.json / result.json); None on v1-frontend Builds
    correspondence_digest_source: str = "absent"      # `build_summary.json:<wrapper>` | `result.json:<wrapper>` | `absent` -- never a stand-in hash
    correspondence_digests: dict[str, str] = field(default_factory=dict)   # every wrapper's (step, request, per-shape) digest the Build recorded
    schema: str = SCHEMA
    row: str | None = None
    run_roots: list[str] = field(default_factory=list)
    runs: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)       # file -> sha256 of every document the verdict was derived from
    execution: dict[str, Any] = field(default_factory=dict)    # labels / measurements carried over, not decisions
    compat: dict[str, Any] = field(default_factory=dict)       # the v1 decision fields, read back
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"verdict outcome {self.outcome!r} not in {OUTCOMES}")
        for c in self.checks:
            if c.group not in GROUPS:
                raise ValueError(f"{c.name}: group {c.group!r} not one of {tuple(GROUPS)}")

    def check(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)

    def groups(self) -> dict[str, str]:
        return {c.name: c.outcome for c in self.checks if c.rule is None and "/" not in c.name}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [c.to_dict() for c in self.checks]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Verdict":
        if d.get("schema") not in (None, SCHEMA):
            raise ValueError(f"not a {SCHEMA} document: schema {d.get('schema')!r}")
        kw = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d and k != "checks"}
        return cls(checks=[CheckResult.from_dict(c, CR.reject_of(c)) for c in d.get("checks") or []], **kw)  # type: ignore[arg-type]

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=False, default=str) + "\n"

    def write(self, path: str | os.PathLike) -> Path:
        p = Path(path)
        p.write_text(self.dumps())
        return p

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Verdict":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------------------------------------------------- helpers

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- one run

def run_checks(commit: dict, tokens_equal: Any, *, min_openings: int = 64, declaration: dict | None = None, required_classes: Any = None,
               ctx: dict | None = None) -> list[CheckResult]:
    """Every rule of `commit_verdict` (+ commit_delta's compiled conditions, row_pod's manifest verify) for ONE instrumented run,
    each with its own outcome.  `ctx`: record-level facts (program_digest, manifest_digest, required_classes_source, manifest_verify, file_digests,
    replay_cache)."""
    ctx = dict(ctx or {})
    checks = [CR.rule_tokens_equal(commit, tokens_equal, ctx), CR.rule_reference_integrity(commit, ctx), CR.rule_fa2_zero_rows(commit, ctx), CR.rule_collector_errors(commit, ctx),
              CR.rule_openings(commit, ctx, min_openings), CR.rule_openings_after_release(commit, ctx, min_openings), CR.rule_openings_population(commit, ctx, min_openings),
              CR.rule_bounded_staging(commit, ctx), CR.rule_oracle_compare(commit, ctx), CR.rule_compiled_check(commit, ctx)]
    checks += C2.c2_rules(commit, ctx)
    checks += C2.execution_extent_rules(commit, ctx)
    checks += [CR.rule_identity_coverage(commit, ctx), CR.rule_manifest_coverage(commit, ctx), CR.rule_required_classes(commit, ctx, declaration, required_classes), CR.rule_manifest_verified(commit, ctx)]
    checks += [CR.rule_identity_layout(commit, ctx), CR.rule_binding_context(commit, ctx)]
    return checks


def run_outcome(checks: list[CheckResult]) -> str:
    """The outcome of one run's rule set (= v1's commit_pass when no rule is FAIL / INSUFFICIENT_EVIDENCE)."""
    return worst([c.outcome for c in checks if c.outcome != NOT_RUN]) if any(c.outcome != NOT_RUN for c in checks) else NOT_RUN


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
        ok, why = CR.binding_check(bnd, min_openings=min_openings)
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
        return False, f"identity check: {idc.get('violations_n')} step(s) differ from the expected manifest (first: step {v.get('step')} {CR.json_short(v)})"
    if int(errs_n) > 0:
        first = (ex.get("errors") or ["?"])[0]
        return False, f"{errs_n} collector error(s): {str(first)[:160]}"
    if declaration is not None or required_classes is not None:
        ok, why = CR.required_classes_check(declaration, required_classes if required_classes is not None else CR.REQUIRED_CLASSES_DEFAULT)
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
    return None


def _c2_verdict(commit: dict, c2_notes: list[str]) -> tuple[bool, str] | None:
    """Value correspondence, the C2 of record (sampled replay, boundary linkage, weights of record) and its population scope."""
    bnd = commit.get("binding")
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
            _sok, _swhy = C2._replay_seed_of_record(sr, commit.get("run_root"))
            if _sok is False:
                _missing_b = C2._form_b_complete_over_values(_oc)
                if _missing_b is not None:
                    return False, f"C2 of record not established (F-r16-23): {_swhy}; and form (B) does not stand in: {_missing_b}"
                # form (B) never covers plane-fed VUs (F-r16-07) -- the population / stratum gaps are still named whatever the seed
                _gap0, _fail0 = C2._complete_replay_population_gap(sr, _oc) if sr.get("result") is True else C2._partial_replay_named_gap(sr, _oc)
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
            _gap, _fail = C2._partial_replay_named_gap(sr, _oc)
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
            _gap, _fail = C2._complete_replay_population_gap(sr, _oc)
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
        _att = C2._batch_attribution_note(bnd, _oc)
        if _att is not None:
            c2_notes.append(_att)
        # [R17-3 / reqval] the population of record is Q(P): the replay's reach reconciled against the required manifest is stated on
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
    return None


# ---------------------------------------------------------------------------------------------------------------------- aggregation

def aggregate_runs(per_run: list[tuple[int | None, list[CheckResult]]]) -> list[CheckResult]:
    """Rule-level checks across the instrumented runs: the worst outcome; scope from the first run + `runs`; differences tagged by `pair`
    from the worst runs; inputs = the first run's (+ every distinct run root)."""
    by_name: dict[str, list[tuple[int | None, CheckResult]]] = {}
    for pair, checks in per_run:
        for c in checks:
            by_name.setdefault(c.name, []).append((pair, c))
    out = []
    for name, items in by_name.items():
        outcomes = [c.outcome for _, c in items]
        o = worst([x for x in outcomes if x != NOT_RUN]) if any(x != NOT_RUN for x in outcomes) else NOT_RUN
        first = items[0][1]
        worst_items = [(p, c) for p, c in items if c.outcome == o]
        diffs = []
        for p, c in worst_items:
            for d in c.differences:
                diffs.append({**d, "pair": p} if p is not None else dict(d))
        notes = []
        for _, c in items:
            for n in c.notes:
                if n not in notes:
                    notes.append(n)
        roots = sorted({str(c.inputs.get("run_root")) for _, c in items if c.inputs.get("run_root")})
        inputs = dict(first.inputs)
        if len(roots) > 1:
            inputs["run_roots"] = roots
        scope = dict(first.scope)
        scope["runs"] = [p for p, _ in items]
        scope["outcome_by_run"] = {str(p): c.outcome for p, c in items}
        reason = next((c.reason for _, c in worst_items if c.reason), None)
        out.append(CheckResult(name=name, code=worst_items[0][1].code, claim=first.claim, inputs=inputs, scope=scope, differences=diffs,
                               compat=list(first.compat), group=first.group, rule=first.rule, notes=notes, reason=reason))
    return out


def group_checks(rules: list[CheckResult]) -> list[CheckResult]:
    """One CheckResult per group, aggregating its rules (the worst outcome; NOT_RUN only when every rule is NOT_RUN)."""
    out = []
    for g, spec in GROUPS.items():
        mine = [r for r in rules if r.group == g]
        outcomes = [r.outcome for r in mine if r.outcome != NOT_RUN]
        o = worst(outcomes) if outcomes else NOT_RUN
        inputs: dict[str, Any] = {}
        for r in mine:
            for k, v in r.inputs.items():
                inputs.setdefault(k, v)
        compat = list(spec["compat"])
        for r in mine:
            for c in r.compat:
                if c not in compat:
                    compat.append(c)
        diffs = [{**d, "rule": r.name} for r in mine if r.outcome == o and o in (FAIL, INSUFFICIENT) for d in r.differences]
        notes = [n for r in mine for n in r.notes if n.startswith("structural_leak")]
        notes = list(dict.fromkeys(notes))
        reasons = [r.reason for r in mine if r.outcome == o and r.reason and o in (FAIL, INSUFFICIENT)]
        out.append(CheckResult(name=g, code=next((r.code for r in mine if r.outcome == o), None), claim=spec["claim"], inputs=inputs,
                               scope={"rules": {r.name: r.outcome for r in mine}}, differences=diffs, compat=compat, group=g, rule=None, notes=notes,
                               reason=(reasons[0] if reasons else None)))
    return out


# ---------------------------------------------------------------------------------------------------------------------- building

#: the Build wrapper whose correspondence digest is THE row's `correspondence_digest` (the step Program is the one every row Builds; the
#: request / per-shape digests are carried beside it in `correspondence_digests`)
CORRESPONDENCE_WRAPPER = "step"


def correspondence_of_build(build_summary: dict | None, *, result_jsons: Mapping[str, dict] | None = None) -> tuple[str | None, str, dict[str, str]]:
    """(correspondence_digest, source, {wrapper: digest}) from the Build's own record: `build_summary.json` (row_pod.sh's build.summary
    lifts `result.json:program.correspondence_digest` per wrapper), else the wrappers' `result.json` directly.  A Build made with the
    v1 frontend records none -> (None, "absent", {}): never a stand-in hash (the row's global_match.json digest is an INPUT, `inputs`)."""
    digests: dict[str, str] = {}
    source = "absent"
    for w, v in (build_summary or {}).items():
        if isinstance(v, dict) and v.get("correspondence_digest"):
            digests[w] = str(v["correspondence_digest"])
    if digests:
        source = "build_summary.json"
    else:
        for w, r in (result_jsons or {}).items():
            d = ((r or {}).get("program") or {}).get("correspondence_digest")
            if d:
                digests[w] = str(d)
        if digests:
            source = "result.json"
    if not digests:
        return None, "absent", {}
    w = CORRESPONDENCE_WRAPPER if CORRESPONDENCE_WRAPPER in digests else sorted(digests)[0]
    return digests[w], f"{source}:{w}", dict(sorted(digests.items()))


EXECUTION_KEYS = ("committer", "collector_variant", "retention", "engine_settings", "collector", "cmd")


def execution_of_record(row_dir: str | os.PathLike) -> dict:
    """The execution configuration the Commit ran with, read from ITS OWN evidence (never typed): `cmd` = the commit_delta command line
    (first line of commit.log, written by row_pod.sh before the run), `committer` (commit/summary.json), `collector` = the instrumented
    runs' own collector record (runs.jsonl commit.binding.collector: variant, retention, window_mb, window_slots, arena_mb, layout, gpu_tree),
    `collector_variant` / `retention` ('windowed' iff window_mb > 0 else 'whole-block'; retention from the record, else --retain in cmd),
    `engine_settings` = --engine-arg k=v pairs + --max-num-seqs / --openings / --pairs / --open-after-release / --required-manifest /
    --expect-root from the command line.  [P5] this is what `commit_summary.json` used to carry (F-r19-int-21: that file is no longer
    written; the harness keeps a reader for old records); verdict.json `execution` is the record."""
    d = Path(row_dir)
    out: dict = {}
    s = json.load(open(d / "commit" / "summary.json")) if (d / "commit" / "summary.json").is_file() else {}
    runs: list[dict] = []
    if (d / "commit" / "runs.jsonl").is_file():
        with open(d / "commit" / "runs.jsonl") as f:
            runs = [json.loads(ln) for ln in f if ln.strip()]
    trt = [r for r in runs if not (r.get("arm") == "control" or r.get("committer") in (None, "control"))]
    cmd = None
    if (d / "commit.log").is_file():
        first = open(d / "commit.log", encoding="utf-8", errors="replace").readline().strip()
        if first.startswith("$ "):
            cmd = first[2:]
    out["cmd"] = cmd
    out["committer"] = s.get("committer")
    coll = None
    for r in trt:
        b = (r.get("commit") or {}).get("binding") if isinstance(r.get("commit"), dict) else None
        if isinstance(b, dict) and isinstance(b.get("collector"), dict):
            coll = b["collector"]
            break
    out["collector"] = coll
    if coll is not None:
        out["collector_variant"] = "windowed" if int(coll.get("window_mb") or 0) > 0 else "whole-block"
        out["retention"] = str(coll.get("retention")).lower() if coll.get("retention") is not None else None
    elif cmd:
        out["collector_variant"] = "windowed" if re.search(r"--window-mb\s+(?!0\b)\d+", cmd) else "whole-block"
        m = re.search(r"--retain\s+(host|device|none|sample)", cmd)
        out["retention"] = m.group(1) if m else None
    else:
        out["collector_variant"] = None
        out["retention"] = None
    es: dict = {}
    if cmd:
        for k, v in re.findall(r"--engine-arg\s+([A-Za-z_]+)=([^\s]+)", cmd):
            try:
                es[k] = json.loads(v)
            except Exception:  # noqa: BLE001
                es[k] = v
        for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--openings", "openings"), ("--pairs", "pairs")):
            m = re.search(flag + r"\s+(\d+)", cmd)
            if m:
                es[key] = int(m.group(1))
        es["open_after_release"] = "--open-after-release" in cmd
        es["required_manifest"] = bool(re.search(r"--required-manifest\s+\S+", cmd))
        m = re.search(r"--expect-root\s+([0-9a-f]{64})", cmd)
        if m:
            es["expect_root"] = m.group(1)
    out["engine_settings"] = es or None
    return out


def build(rows: list[dict], verdict_doc: dict | None, *, summary: dict | None = None, manifest_verify: dict | None = None, file_digests: dict[str, str] | None = None,
          row: str | None = None, min_openings: int | None = None, declaration: dict | None = None, rc: int | None = None,
          build_summary: dict | None = None, build_results: Mapping[str, dict] | None = None, execution: dict | None = None) -> Verdict:
    """The Verdict from the Commit's own documents: `rows` = runs.jsonl entries (control + instrumented), `verdict_doc` = commit/verdict.json,
    `manifest_verify` = commit/manifest_verify.json, `execution` = `execution_of_record(row_dir)` (the Commit's own configuration record);
    `summary` = a pre-P5 record's commit_summary.json (old rows only: its execution fields stand in for `execution`, its `rc` for a missing `rc`).
    `declaration` defaults to the run's own `commit.classes`.  `build_summary` (build_summary.json) / `build_results` ({wrapper: result.json})
    give the Build's `runtime-correspondence/v1` digest."""
    vd = _d(verdict_doc)
    if execution is None:
        execution = {k: v for k, v in _d(summary).items() if k in EXECUTION_KEYS}
    elif summary:
        # a pre-P5 row whose evidence is partial here (e.g. commit.log not beside the record): its recorded summary fills what the
        # evidence did not state -- the record's own statement, never a stand-in
        execution = {**{k: v for k, v in _d(summary).items() if k in EXECUTION_KEYS}, **{k: v for k, v in execution.items() if v is not None}}
    rm = _d(vd.get("required_manifest"))
    program_digest = rm.get("program_digest")
    manifest_digest = rm.get("manifest_digest")
    query_id = _d(rm.get("query")).get("query_id")
    treatment = [r for r in rows if r.get("arm") != "control" and r.get("committer") not in (None, "control") and isinstance(r.get("commit"), dict)]
    if not treatment:
        treatment = [r for r in rows if r.get("arm") not in (None, "control") and isinstance(r.get("commit"), dict)]
    if min_openings is None:
        es = _d(execution.get("engine_settings"))
        min_openings = min(64, _int(es.get("openings"), 64)) if es.get("openings") is not None else 64
    ctx = {"program_digest": program_digest, "manifest_digest": manifest_digest, "required_classes_source": vd.get("required_classes_source"),
           "manifest_verify": manifest_verify, "file_digests": dict(file_digests or {}), "replay_cache": vd.get("replay_cache")}
    required_classes = vd.get("required_classes")
    per_run: list[tuple[int | None, list[CheckResult]]] = []
    recorded: list[tuple[int | None, bool | None, str | None]] = []
    mismatches = []
    for r in treatment:
        c = r["commit"]
        if not program_digest:
            ctx["program_digest"] = _d(c.get("binding")).get("program_digest") or _d(c.get("weights_pin")).get("program_digest")
        if not manifest_digest:
            ctx["manifest_digest"] = _d(c.get("coverage")).get("manifest_digest")
        decl = declaration
        if decl is None and isinstance(c.get("classes"), dict):
            decl = {"classes": [{"class": k, **_d(v)} for k, v in c["classes"].items()]}
        checks = run_checks(c, r.get("tokens_equal_ref", r.get("tokens_equal")), min_openings=min_openings, declaration=decl, required_classes=required_classes, ctx=ctx)
        pair = r.get("pair")
        per_run.append((pair, checks))
        rec = c.get("commit_pass")
        recorded.append((pair, rec, c.get("commit_fail_reason")))
        if rec is not None and (run_outcome(checks) == PASS) != bool(rec):
            mismatches.append({"pair": pair, "recorded_commit_pass": bool(rec), "reconstructed": run_outcome(checks), "recorded_reason": c.get("commit_fail_reason"),
                               "reconstructed_failing": [x.name for x in checks if x.outcome in (FAIL, INSUFFICIENT)]})
    rules = aggregate_runs(per_run)
    groups = group_checks(rules)
    # commit_delta's decision (commit_pass) covers every rule EXCEPT manifest_of_record_verified, which row_pod.sh grades after the Commit
    # (EXEC-05: a verify FAIL is a Commit FAIL); reconcile the two scopes separately, then take the worst
    mv_rule = next((r for r in rules if r.name == "required_value_coverage/manifest_of_record_verified"), None)
    delta_rules = [r for r in rules if r is not mv_rule]
    graded = [r.outcome for r in delta_rules if r.outcome != NOT_RUN]
    outcome = worst(graded) if graded else NOT_RUN
    notes: list[str] = []
    if not treatment:
        outcome = NOT_RUN
        notes.append("no instrumented run on the record (control arm only, or the Commit never ran)")
    rc_v = rc if rc is not None else (_d(summary).get("rc") if summary else None)
    if rc_v is not None and _int(rc_v) != 0 and outcome == PASS:
        outcome = INSUFFICIENT
        notes.append(f"commit_delta exited rc={rc_v}: the stage did not close although every graded rule passed (crash / timeout / verify)")
    commit_pass = vd.get("commit_pass")
    if commit_pass is None and treatment:
        commit_pass = all(bool(r["commit"].get("commit_pass")) for r in treatment) if all(r["commit"].get("commit_pass") is not None for r in treatment) else None
    if commit_pass is not None and treatment and (outcome == PASS) != bool(commit_pass) and not (rc_v is not None and _int(rc_v) != 0):
        notes.append(f"RECONSTRUCTION MISMATCH: recorded commit_pass={commit_pass} but the rule-by-rule outcome is {outcome}; the recorded decision stands")
        outcome = PASS if commit_pass else (FAIL if outcome == PASS else outcome)
    if mv_rule is not None and mv_rule.outcome != NOT_RUN and treatment:
        outcome = worst([outcome, mv_rule.outcome]) if outcome != NOT_RUN else mv_rule.outcome
    sums = _d(summary)
    execution = {k: execution.get(k) for k in EXECUTION_KEYS if k in execution}
    if vd.get("moe_source") is not None:
        execution["moe_source"] = vd.get("moe_source")
    if vd.get("host_peak") is not None:
        execution["host_peak"] = vd.get("host_peak")
    if vd.get("compilation") is not None:
        execution["compilation"] = vd.get("compilation")
    if treatment:
        execution["rules_version"] = treatment[0]["commit"].get("rules_version")
    compat = {"commit_pass": commit_pass, "first_fail_reason": vd.get("first_fail_reason"), "runs_failed": vd.get("runs_failed"),
              "rc": rc_v, "commit_summary_pass": sums.get("pass") if summary else None,
              "commit_summary_deprecated": "F-r19-int-21 / P5: commit_summary.json is no longer written (its pass was rc == 0 and tokens_equal); this verdict is the record",
              "per_run": [{"pair": p, "commit_pass": rec, "commit_fail_reason": why} for p, rec, why in recorded],
              "negatives": vd.get("negatives"), "reconstruction_mismatch": mismatches or None, "only_arm": vd.get("only_arm")}
    runs_ = {"instrumented": len(treatment), "failed": sum(1 for _, rec, _ in recorded if rec is False), "control": sum(1 for r in rows if r.get("arm") == "control"),
             "not_pass_by_reconstruction": sum(1 for _, ch in per_run if run_outcome(ch) != PASS), "pairs": [p for p, _ in per_run]}
    cdig, csrc, cdigs = correspondence_of_build(build_summary, result_jsons=build_results)
    return Verdict(checks=groups + rules, outcome=outcome, program_digest=ctx.get("program_digest"), manifest_digest=ctx.get("manifest_digest"), query_id=query_id,
                   correspondence_digest=cdig, correspondence_digest_source=csrc, correspondence_digests=cdigs,
                   row=row, run_roots=list(dict.fromkeys(str(x) for x in (vd.get("run_roots") or [r["commit"].get("run_root") for r in treatment]) if x)),
                   runs=runs_, inputs=dict(file_digests or {}), execution=execution, compat=compat, notes=notes)


RECORD_FILES = ("commit/verdict.json", "commit/runs.jsonl", "commit_summary.json", "commit/manifest_verify.json", "commit/summary.json", "commit.log", "global_match.json",
                "match_summary.json", "build_summary.json")


def from_record(row_dir: str | os.PathLike, *, rc: int | None = None, stage_ok: str | None = None) -> Verdict:
    """Rebuild the Verdict from a row dir's evidence (`commit/runs.jsonl`, `commit/verdict.json`, `commit/summary.json` + `commit.log`
    (`execution_of_record`), `commit/manifest_verify.json`, `commit/sampled_replay_p*.json` for their digests; a pre-P5 row's
    `commit_summary.json` is read when present).  A row whose Commit never ran gives NOT_RUN with the missing files named.
    `rc` = commit_delta's exit code when the caller (row_pod.sh) knows it; `stage_ok` = a stage decision made OUTSIDE this verdict (pre-P5
    rows: the commit_summary word), recorded under `compat.stage_ok` and compared (a disagreement is a note, never a changed decision).
    Since P5 the verdict's `outcome` IS the Commit stage's decision (row_pod.sh reads it back)."""
    d = Path(row_dir)
    digests: dict[str, str] = {}
    for rel in RECORD_FILES + tuple(sorted(os.path.relpath(p, d) for p in glob.glob(str(d / "commit" / "sampled_replay_p*.json")))):
        p = d / rel
        if p.is_file():
            digests[rel] = _sha256(p)
    rows: list[dict] = []
    if (d / "commit" / "runs.jsonl").is_file():
        with open(d / "commit" / "runs.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    vd = json.load(open(d / "commit" / "verdict.json")) if (d / "commit" / "verdict.json").is_file() else None
    summ = json.load(open(d / "commit_summary.json")) if (d / "commit_summary.json").is_file() else None
    mv = json.load(open(d / "commit" / "manifest_verify.json")) if (d / "commit" / "manifest_verify.json").is_file() else None
    bs = json.load(open(d / "build_summary.json")) if (d / "build_summary.json").is_file() else None
    results: dict[str, dict] = {}
    if bs is None:                                    # no build.summary on the row: the wrappers' own result.json (build_step/, build_request*/)
        for p in sorted(glob.glob(str(d / "build_*" / "result.json"))):
            w = os.path.basename(os.path.dirname(p))
            try:
                results[{"build_step": "step", "build_request": "request"}.get(w, w)] = json.load(open(p))
            except (OSError, ValueError):
                continue
    v = build(rows, vd, summary=summ, manifest_verify=mv, file_digests=digests, row=d.name, rc=rc, build_summary=bs, build_results=results,
              execution=execution_of_record(d))
    missing = [rel for rel in ("commit/verdict.json", "commit/runs.jsonl") if rel not in digests]
    if missing:
        v.notes.append(f"record incomplete: {', '.join(missing)} not on the row (Commit never closed / never ran)")
        if v.outcome == PASS:
            v.outcome = INSUFFICIENT
    if (d / "stages.txt").is_file():
        stage = [ln for ln in (d / "stages.txt").read_text().splitlines() if ln.startswith("commit ")]
        if stage:
            v.compat["stage_line"] = stage[-1][:200]
            v.compat["stage_outcome"] = stage[-1].split()[1]
    if stage_ok is not None:
        v.compat["stage_ok"] = stage_ok
        if (stage_ok.upper() == PASS) != (v.outcome == PASS):
            v.notes.append(f"STAGE MISMATCH: row_pod.sh's Commit stage says {stage_ok} but the rule-by-rule outcome is {v.outcome}; the stage decision stands (F-r19-int-21)")
    return v


def mapping_markdown() -> str:
    lines = ["| old (v1 field) | where | new (v2 check / field) |", "|---|---|---|"]
    for m in OLD_TO_NEW:
        lines.append(f"| `{m['old']}` | {m['where']} | {m['new']} |")
    lines += ["", "| check | claim |", "|---|---|"]
    for g, spec in GROUPS.items():
        lines.append(f"| `{g}` | {spec['claim']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity_vllm.check.verdict", description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("from-record", help="rebuild <row_dir>/verdict.json from the row's Commit evidence")
    fr.add_argument("row_dir")
    fr.add_argument("--out", default=None, help="where to write (default <row_dir>/verdict.json)")
    fr.add_argument("--rc", type=int, default=None, help="commit_delta's exit code (row_pod.sh passes it; a pre-P5 row's commit_summary.json supplies it otherwise)")
    fr.add_argument("--stage-ok", default=None, help="a stage decision made outside this verdict, if any (recorded as compat.stage_ok; a disagreement is a note)")
    fr.add_argument("--quiet", action="store_true")
    sub.add_parser("mapping", help="print the old -> new check-name table (markdown)")
    a = ap.parse_args(argv)
    if a.cmd == "mapping":
        print(mapping_markdown(), end="")
        return 0
    v = from_record(a.row_dir, rc=a.rc, stage_ok=a.stage_ok)
    out = Path(a.out) if a.out else Path(a.row_dir) / "verdict.json"
    v.write(out)
    if not a.quiet:
        gs = " ".join(f"{k}={v_}" for k, v_ in v.groups().items())
        print(f"{v.outcome} {out} [{gs}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
