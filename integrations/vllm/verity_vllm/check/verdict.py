"""The Commit VERDICT as typed records (`vllm-verdict/v1`): every check the Commit decides, by its human name, with what it read, what
it established and what it found -- ONE document (`<row>/verdict.json`) beside the files it is derived from.

    verity-vllm verdict from-record <row_dir> [--out PATH]      rebuild the record from a row's evidence
    verity-vllm verdict mapping                                  the old -> new check-name table (markdown)

WHY.  The v1 Commit records its decision three times over: `commit/verdict.json` (`commit_pass`, one `first_fail_reason` string: the FIRST
rule that failed, the others silent), `commit/runs.jsonl` (per instrumented run, every rule's own block under `commit.*`, the check
names being the field names they grew under -- `weights_pin`, `linkage`, `coverage`, `identity_check`) and `commit_summary.json`, whose
`pass` degenerated to `rc == 0 and tokens_equal` once `openings` / `coverage` stopped being carried on runs.jsonl (F-r19-int-21).  The
verdict of record was `commit/verdict.json` + `commit/sampled_replay_p*.json`, read together by whoever knew to.

WHAT.  A `Verdict` = the seven named checks (groups) and the R*/X* rules under them, each a `CheckVerdict`:

    name        `<group>` for the group, `<group>/<rule>` for a rule                    (human names: see GROUPS)
    claim       one sentence: what PASS establishes
    inputs      digests / ids of what the check read (program_digest, manifest_digest, run root, file sha256s, the sample seed)
    scope       what was actually established (sample_size, positions, layers, requests, steps ...) -- never more than the record shows
    outcome     PASS | FAIL | INSUFFICIENT_EVIDENCE | NOT_RUN
    differences named, machine-readable findings ({"name": ..., ...}; e.g. `fa2_layers_bad`, `identities_unbound`, `account_missing`)
    compat      the v1 field name(s) the check was read from (the old -> new mapping is `OLD_TO_NEW`, tested for completeness)
    rule        the R*/X* id of the rule (None on a group)
    notes       prose the PASS carries (named gaps, `structural_leak: <table>` where the rule rests on a name table, not on the Program)

The decisions are v1's, unchanged: every rule here is `verity_vllm.check.commit_verdict.commit_verdict`'s own predicate, evaluated
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

import glob
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.check import commit_verdict as CV
from verity_vllm.config import option

SCHEMA = "vllm-verdict/v1"
PASS, FAIL, INSUFFICIENT, NOT_RUN = "PASS", "FAIL", "INSUFFICIENT_EVIDENCE", "NOT_RUN"
OUTCOMES = (PASS, FAIL, INSUFFICIENT, NOT_RUN)
_SEVERITY = {FAIL: 3, INSUFFICIENT: 2, PASS: 1, NOT_RUN: 0}

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
class CheckVerdict:
    name: str
    claim: str
    inputs: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    outcome: str = NOT_RUN
    differences: list[dict[str, Any]] = field(default_factory=list)
    compat: list[str] = field(default_factory=list)
    group: str | None = None
    rule: str | None = None
    notes: list[str] = field(default_factory=list)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"{self.name}: outcome {self.outcome!r} not in {OUTCOMES}")
        if self.group is None:
            self.group = self.name.split("/", 1)[0]
        if self.group not in GROUPS:
            raise ValueError(f"{self.name}: group {self.group!r} not one of {tuple(GROUPS)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckVerdict":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})  # type: ignore[arg-type]


@dataclass
class Verdict:
    checks: list[CheckVerdict]
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

    def check(self, name: str) -> CheckVerdict | None:
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
        return cls(checks=[CheckVerdict.from_dict(c) for c in d.get("checks") or []], **kw)  # type: ignore[arg-type]

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


def worst(outcomes: list[str]) -> str:
    """FAIL > INSUFFICIENT_EVIDENCE > PASS > NOT_RUN; an empty list is NOT_RUN."""
    if not outcomes:
        return NOT_RUN
    return max(outcomes, key=lambda o: _SEVERITY[o])


# ---------------------------------------------------------------------------------------------------------------------- helpers

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _base_inputs(commit: dict, ctx: dict) -> dict[str, Any]:
    out = {"program_digest": ctx.get("program_digest") or _d(commit.get("binding")).get("program_digest"),
           "manifest_digest": ctx.get("manifest_digest") or _d(commit.get("coverage")).get("manifest_digest"), "run_root": commit.get("run_root")}
    return {k: v for k, v in out.items() if v is not None}


def _leak(table: str, why: str) -> str:
    return f"structural_leak: {table} ({why})"


# ---------------------------------------------------------------------------------------------------------------------- the rules
# Every function mirrors ONE predicate of commit_verdict.commit_verdict (cited by its line's comment tag); none adds a condition.  `ctx`
# carries the record-level facts a rule needs beyond the run's commit block (program / manifest digests, file digests, manifest_verify).

def _mk(name: str, rule: str, claim: str, compat: list[str], commit: dict, ctx: dict, **kw: Any) -> CheckVerdict:
    inputs = dict(_base_inputs(commit, ctx))
    inputs.update(kw.pop("inputs", {}) or {})
    return CheckVerdict(name=name, claim=claim, inputs=inputs, compat=compat, rule=rule, **kw)


def rule_tokens_equal(commit: dict, tokens_equal: Any, ctx: dict) -> CheckVerdict:
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


def rule_reference_integrity(commit: dict, ctx: dict) -> CheckVerdict:
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
    return _mk("runtime_match/reference_integrity", "X-03 (R12) counters",
               "Every value acquisition the collector attempted was bound (attempted == committed, both > 0).",
               ["extra.expected_tensors", "extra.committed_tensors"], commit, ctx, outcome=out, differences=diffs, reason=reason,
               scope={"attempted": n_exp, "committed": n_com})


def rule_fa2_zero_rows(commit: dict, ctx: dict) -> CheckVerdict:
    z = commit.get("fa2_zero_rows")
    if z is None:
        out, diffs, reason = NOT_RUN, [], None
    elif _int(z) > 0:
        out, diffs, reason = FAIL, [{"name": "fa2_zero_rows", "rows": _int(z)}], f"{z} FA2 hidden rows bound as all-zero bytes (tap staged after it was zeroed: ordering fault)"
    else:
        out, diffs, reason = PASS, [], None
    return _mk("runtime_match/fa2_zero_rows", "R13 p0b tap ordering", "No attention hidden-value row was bound as all-zero bytes (the tap was staged before it was zeroed).",
               ["fa2_zero_rows"], commit, ctx, outcome=out, differences=diffs, reason=reason, scope={"rows_checked": z is not None})


def rule_collector_errors(commit: dict, ctx: dict) -> CheckVerdict:
    ex = _d(commit.get("extra"))
    errs_n = ex.get("errors_n")
    if errs_n is None:
        errs_n = len(ex.get("errors") or []) if "errors" in ex else commit.get("errors_n")
    if errs_n is None:
        errs_n = 0
    if _int(errs_n) > 0:
        first = _first(ex.get("errors")) or "?"
        out, diffs, reason = FAIL, [{"name": "collector_errors", "n": _int(errs_n), "first": str(first)[:160]}], f"{errs_n} collector error(s): {str(first)[:160]}"
    else:
        out, diffs, reason = PASS, [], None
    return _mk("runtime_match/collector_errors", "X-03 errors", "No collector, arena or hook error was recorded during the instrumented run.",
               ["extra.errors_n", "extra.errors"], commit, ctx, outcome=out, differences=diffs, reason=reason, scope={"errors_n": _int(errs_n)})


def rule_bounded_staging(commit: dict, ctx: dict) -> CheckVerdict:
    bs = _d(commit.get("extra")).get("bounded_staging")
    if not isinstance(bs, dict):
        return _mk("runtime_match/bounded_staging", "R17 fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=NOT_RUN)
    if bs.get("learned_only_steps"):
        steps = list(bs["learned_only_steps"])
        return _mk("runtime_match/bounded_staging", "R17 fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=FAIL, differences=[{"name": "learned_only_steps", "n": len(steps), "first": steps[:8]}],
                   reason=f"bounded staging: {len(steps)} learned-only step(s) {steps[:8]} on a record run (no root)")
    if bs.get("learn_only"):
        return _mk("runtime_match/bounded_staging", "R17 fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
                   ["extra.bounded_staging"], commit, ctx, outcome=FAIL, differences=[{"name": "learn_only", "value": True}],
                   reason="bounded staging: the record run was finalized in learn-only mode")
    return _mk("runtime_match/bounded_staging", "R17 fp8 bounded staging", "A bounded-staging record run carries no learned-only step and was not finalized in learn-only mode.",
               ["extra.bounded_staging"], commit, ctx, outcome=PASS, scope={k: bs.get(k) for k in ("learn_only", "learned_only_steps", "mode") if k in bs})


def rule_openings(commit: dict, ctx: dict, min_openings: int) -> CheckVerdict:
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


def rule_openings_after_release(commit: dict, ctx: dict, min_openings: int) -> CheckVerdict:
    """v1's `if oar is not None:` block + the `elif deferred picks and no record` FAIL."""
    ops = _d(commit.get("openings"))
    oar = commit.get("openings_after_release")
    bnd = _d(commit.get("binding"))
    po = _d(bnd.get("openings_population"))
    claim = ("After the raw device staging was released, every opening (the challenge's and every pick the in-run stages deferred to replay) "
             "verified against the ORIGINAL roots, the replay reproduced the run root, and no raw byte stayed on the device.")
    compat = ["openings_after_release.n", "openings_after_release.verified", "openings_after_release.source", "openings_after_release.replay_root_equal",
              "openings_after_release.replay_errors_n", "openings_after_release.retained_device_after", "openings_after_release.deferred_received_positions"]
    name, rule = "runtime_match/openings_after_release", "R13 P0 / R17-5 openings after release"
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

    def fail(diff: dict, reason: str) -> CheckVerdict:
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


def rule_oracle_compare(commit: dict, ctx: dict) -> CheckVerdict:
    """v1: value_correspondence.result False -> FAIL (the failing component's own cause); oracle_compare.result False -> FAIL."""
    vc = commit.get("value_correspondence")
    oc = commit.get("oracle_compare") if isinstance(commit.get("oracle_compare"), dict) else None
    claim = ("No committed value differs from the Match oracle's (form (B): the Match capture's instance-output snapshots and sampled ids of "
             "the same attempt) where the oracle looked; scope states how much it looked at.")
    compat = ["oracle_compare.result", "oracle_compare.mismatch_n", "value_correspondence.result", "value_correspondence.components"]
    name, rule = "runtime_match/oracle_compare", "C2 form (B) / R15 COORD M-0111 (2)"
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


def rule_compiled_check(commit: dict, ctx: dict) -> CheckVerdict:
    """commit_delta's two post-verdict compiled conditions (R15 compiled S3): kernel replay exact; ring refused nothing."""
    vc = _d(commit.get("value_correspondence"))
    ckc = vc.get("compiled_check")
    cs = _d(_d(commit.get("compiled_source")).get("stats"))
    claim = "Under compiled acquisition every kernel launch replays exactly and the compiled ring refused / mis-shaped no acquisition (S3)."
    compat = ["value_correspondence.compiled_check", "compiled_source.stats"]
    name, rule = "runtime_match/compiled_check", "R15 compiled S3"
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


def rule_binding_context(commit: dict, ctx: dict) -> CheckVerdict:
    """binding_check (1): the map was built (no error) and its context (roots, layout) is the run's."""
    bnd = commit.get("binding")
    claim = "The binding map was built for this run: its Program digest, layout and roots are the run's own (BIND-01 (1))."
    compat = ["binding.error", "binding.context.ok", "binding.context.why", "binding.program_digest", "binding.map_digest"]
    name, rule = "program_source_identity/binding_context", "R14 BIND-01 (1)"
    if not isinstance(bnd, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    scope = {"map_digest": bnd.get("map_digest"), "map_entries_n": bnd.get("map_entries_n"), "program_digest_source": bnd.get("program_digest_source"),
             "layout": bnd.get("layout"), "rules_version": bnd.get("rules_version")}
    inputs = {"binding_program_digest": bnd.get("program_digest"), "root_program_digest": bnd.get("root_program_digest"), "map_digest": bnd.get("map_digest")}
    if bnd.get("error"):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, differences=[{"name": "binding_map_not_built", "error": str(bnd["error"])[:300]}],
                   reason=f"binding map not built: {bnd['error']}")
    c = _d(bnd.get("context"))
    if not c.get("ok"):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs, differences=[{"name": "binding_context_mismatch", "why": c.get("why")}],
                   reason=f"binding-map context: {c.get('why')}")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs)


def rule_identity_coverage(commit: dict, ctx: dict) -> CheckVerdict:
    """binding_check (2): identity coverage of the expected population (error -> MAN-02 refusal; missing; ambiguous)."""
    bnd = commit.get("binding")
    claim = "Every identity of the required-value population has exactly one committed binding (a duplicate never covers a missing one; BIND-01 (2) / MAN-02)."
    compat = ["binding.coverage.ok", "binding.coverage.error", "binding.coverage.missing_n", "binding.coverage.ambiguous_n", "binding.coverage.expected_n", "binding.coverage.bound_n"]
    name, rule = "required_value_coverage/identity_coverage", "R14 BIND-01 (2) / MAN-02"
    if not isinstance(bnd, dict) or not isinstance(bnd.get("coverage"), dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    cov = bnd["coverage"]
    pop = _d(bnd.get("population"))
    scope = {"expected_n": cov.get("expected_n"), "bound_n": cov.get("bound_n"), "requests": pop.get("requests"), "steps": pop.get("steps"),
             "population_source": str(pop.get("source") or "")[:120], "extra_n": cov.get("extra_n"), "physical_copies_n": cov.get("physical_copies_n")}
    notes = []
    if "required-manifest" not in str(pop.get("source") or ""):
        notes.append(_leak("binding.MEMBER_RULES", "the expected population's members come from a module-class table, not the Program"))
    if bnd.get("error"):                     # already FAIL under binding_context; the coverage was never computed
        return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT, scope=scope, notes=notes,
                   differences=[{"name": "binding_map_not_built", "error": str(bnd["error"])[:300]}], reason=f"binding map not built: {bnd['error']}")
    if cov.get("error"):
        err = str(cov["error"])
        diffs = [{"name": "manifest_incomplete", "error": err[:400], "unmodelled": dict(list(_d(cov.get("unmodelled")).items())[:8]) if cov.get("unmodelled") else None}]
        if "unmodelled" in err:
            notes.append(_leak("required_manifest.members_for", "a Program family is INTERIOR because the vocabulary names no member for it -- by name, F-r19-int-20"))
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes, differences=diffs, reason=f"identity coverage: {err}")
    if not cov.get("ok"):
        if _int(cov.get("missing_n")) == 0 and _int(cov.get("ambiguous_n")) > 0:
            amb = _d(_first(cov.get("ambiguous")))
            return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes,
                       differences=[{"name": "identities_ambiguous", "n": _int(cov.get("ambiguous_n")), "expected_n": cov.get("expected_n"), "first": amb}],
                       reason=(f"identity coverage: {cov.get('ambiguous_n')} of {cov.get('expected_n')} required identities have more than one binding "
                               f"(C1 requires one unambiguous binding; first: step {amb.get('step')} {amb.get('op_path')}/{amb.get('output_member')} inv {amb.get('invocation')} "
                               f"req {amb.get('request_id')} rank {amb.get('rank')} bound {amb.get('bindings_n')} x at {amb.get('bound_at')})"))
        miss = _d(_first(cov.get("missing")))
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, notes=notes,
                   differences=[{"name": "identities_unbound", "n": _int(cov.get("missing_n")), "expected_n": cov.get("expected_n"), "first": miss}],
                   reason=(f"identity coverage: {cov.get('missing_n')} of {cov.get('expected_n')} required identities have no binding "
                           f"(first: step {miss.get('step')} {miss.get('op_path')}/{miss.get('output_member')} inv {miss.get('invocation')} req {miss.get('request_id')} rank {miss.get('rank')})"))
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, notes=notes)


def rule_openings_population(commit: dict, ctx: dict, min_openings: int) -> CheckVerdict:
    """binding_check (3): openings drawn from the EXPECTED population all had a binding and all verified."""
    bnd = commit.get("binding")
    claim = "Every opening drawn from the required-value population had a binding and verified (BIND-01 (3); retain=none: the deferred share opens by replay)."
    compat = ["binding.openings_population.n", "binding.openings_population.unbound_n", "binding.openings_population.opened", "binding.openings_population.verified", "binding.openings_population.skipped"]
    name, rule = "runtime_match/openings_population", "R14 BIND-01 (3)"
    if not isinstance(bnd, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    if bnd.get("error") or not _d(bnd.get("context")).get("ok") or _d(bnd.get("coverage")).get("error") or not _d(bnd.get("coverage")).get("ok", True):
        # v1 returned from binding_check before reaching (3): the population openings were not graded on this run
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN, notes=["not graded: binding_context / identity_coverage failed first (v1 binding_check returns at the first failing part)"])
    po = _d(bnd.get("openings_population"))
    scope = {"sample_size": po.get("n"), "bound": po.get("bound"), "opened": po.get("opened"), "verified": po.get("verified"), "skipped": po.get("skipped"),
             "families": po.get("families"), "deferred_replay_n": po.get("deferred_replay_n"), "late_step_in_sample": po.get("late_step_in_sample")}
    if _int(po.get("unbound_n")) > 0:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope,
                   differences=[{"name": "population_openings_unbound", "n": _int(po.get("unbound_n")), "first": _first(po.get("unbound"))}],
                   reason=f"{po['unbound_n']} population-drawn openings had no binding (first: {CV.json_short(_d(_first(po.get('unbound'))))})")
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


def rule_manifest_coverage(commit: dict, ctx: dict) -> CheckVerdict:
    """v1: `cov = ex.coverage or commit.coverage; if cov is not None and not cov.ok: FAIL` (R14 MAN-01)."""
    ex = _d(commit.get("extra"))
    cov = ex.get("coverage") or commit.get("coverage")
    claim = "Every identity the required-value manifest names (Program x query, per request / step / member / element range) was committed with its full extent (MAN-01)."
    compat = ["coverage.ok", "coverage.missing_n", "coverage.missing_by_family", "coverage.families_missing_entirely", "coverage.error", "coverage.manifest_digest", "commit_summary.coverage"]
    name, rule = "required_value_coverage/manifest_coverage", "R14 MAN-01"
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
    notes = [_leak("required_manifest.members_for", "a Program family is INTERIOR because the vocabulary names no member for it -- by name, F-r19-int-20")]
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


def rule_identity_layout(commit: dict, ctx: dict) -> CheckVerdict:
    """v1: `idc = ex.identity_check; if idc is not None and not idc.ok: FAIL` (X-03 on identities, R13).  The rule body is native_host's
    (`NativeHostCommitter.identity_check`, owned by lane/vllm-x03); this only names its findings."""
    idc = _d(commit.get("extra")).get("identity_check")
    claim = ("Every committed step's identity set is the expected manifest's: every expected module produced >= 1 tensor and none came from outside it, "
             "every attention layer carries the same positive count of hidden rows, exactly one sampled-ids tensor, the same name multiset as step 0 (X-03).")
    compat = ["extra.identity_check.ok", "extra.identity_check.violations", "extra.identity_check.violations_n", "extra.identity_check.steps_checked"]
    name, rule = "program_source_identity/identity_layout", "X-03 (R13) identities"
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
        for key in ("fa2_layers_bad", "missing_modules", "unknown_modules", "moe_layers_bad", "tp_sites_bad", "multiplicity_vs_step0"):
            hits = [v for v in viols if v.get(key)]
            if hits:
                diffs.append({"name": key, "steps": [v.get("step") for v in hits][:16], "steps_n": len(hits), "first": {"step": hits[0].get("step"), key: hits[0][key]}})
        sampler = [v for v in viols if man.get("sampler_ids") and v.get("sampler_ids") != man.get("sampler_ids")]
        if sampler:
            diffs.append({"name": "sampler_ids", "steps": [v.get("step") for v in sampler][:16], "steps_n": len(sampler), "first": {"step": sampler[0].get("step"), "sampler_ids": sampler[0].get("sampler_ids")}})
        v0 = _d(_first(viols))
        return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope={**scope, "violations_n": idc.get("violations_n")}, notes=notes, differences=diffs,
                   reason=f"identity check: {idc.get('violations_n')} step(s) differ from the expected manifest (first: step {v0.get('step')} {CV.json_short(v0)})")
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, notes=notes)


def rule_required_classes(commit: dict, ctx: dict, declaration: dict | None, required_classes: Any) -> CheckVerdict:
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
    req = tuple(required_classes) if required_classes is not None else CV.REQUIRED_CLASSES_DEFAULT
    ok, why = CV.required_classes_check(declaration, req)
    cov = CV.class_coverage(declaration)
    scope = {"required": list(req), "declared": {k: {"how": v.get("how"), "tensors": v.get("tensors")} for k, v in cov.items()}, "source": src or None}
    inputs = {"required_classes": list(req)}
    if not ok:
        missing = [c for c in req if c not in cov]
        empty = [c for c, d in cov.items() if d.get("how") in CV.LEAF_HOWS and (d.get("tensors") is None or _int(d.get("tensors")) <= 0)]
        diff = {"name": "declaration_missing"} if declaration is None else {"name": "required_class_undeclared", "classes": missing} if missing else {"name": "class_bound_nothing", "classes": empty}
        return _mk(name, rule, claim, compat, commit, ctx, outcome=INSUFFICIENT if declaration is None else FAIL, scope=scope, inputs=inputs, notes=notes, differences=[diff], reason=why)
    return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs, notes=notes)


def rule_manifest_verified(commit: dict, ctx: dict) -> CheckVerdict:
    """row_pod.sh EXEC-05 / MAN-05: `required_manifest verify` rebuilt the manifest from this attempt's Program x workload == the one used."""
    mv = ctx.get("manifest_verify")
    claim = "The required-value manifest the Commit used is byte-identical (rows digest) to the one rebuilt from this attempt's Program x workload (EXEC-05 / MAN-05)."
    compat = ["manifest_verify.ok", "manifest_verify.used_digest", "manifest_verify.rebuilt_digest", "stages.txt manifest_verify="]
    name, rule = "required_value_coverage/manifest_of_record_verified", "R15 EXEC-05 / MAN-05"
    if not isinstance(mv, dict):
        return _mk(name, rule, claim, compat, commit, ctx, outcome=NOT_RUN)
    scope = {k: mv.get(k) for k in ("rebuilt_engine", "rebuilt_identities", "used_identities", "rebuilt_complete") if k in mv}
    inputs = {"used_digest": mv.get("used_digest"), "rebuilt_digest": mv.get("rebuilt_digest"), "used_program_digest": mv.get("used_program_digest"), "rebuilt_program_digest": mv.get("rebuilt_program_digest")}
    if mv.get("ok") is True:
        return _mk(name, rule, claim, compat, commit, ctx, outcome=PASS, scope=scope, inputs=inputs)
    return _mk(name, rule, claim, compat, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs,
               differences=[{"name": "manifest_of_record_differs", "used_digest": mv.get("used_digest"), "rebuilt_digest": mv.get("rebuilt_digest"), "why": mv.get("why") or mv.get("reason")}],
               reason=f"manifest of record VERIFY FAILED: used {str(mv.get('used_digest'))[:16]} != rebuilt {str(mv.get('rebuilt_digest'))[:16]}")


def _replay_scope(sr: dict | None) -> dict[str, Any]:
    if not isinstance(sr, dict):
        return {}
    smp = _d(sr.get("sample"))
    pop = _d(sr.get("population"))
    picks = _d(smp.get("picks_by_request_step"))
    positions = sum(len(_d(v)) for v in picks.values()) if picks else None
    qt = _d(pop.get("query_total"))
    return {"result": sr.get("result"), "grade": sr.get("grade"), "sample_size": smp.get("evaluated"), "picked": smp.get("picked"), "equal": smp.get("equal"),
            "mismatch_n": smp.get("mismatch_n", sr.get("mismatch_n")), "not_evaluated_n": smp.get("not_evaluated_n"), "per_stratum": smp.get("per_stratum"),
            "positions": positions, "requests": len(picks) if picks else (len(pop.get("requests") or []) if isinstance(pop.get("requests"), (list, dict)) else None),
            "strata": len(sr.get("strata") or []) or sr.get("strata_n") or pop.get("strata_n"), "strata_unevaluated": len(sr.get("strata_unevaluated") or []),
            "population_vus": pop.get("vus"), "population_by_family": pop.get("by_family"), "not_evaluable": pop.get("not_evaluable") or None,
            "query_total": qt.get("total") if qt else None, "query_id": qt.get("query_id") if qt else None, "query_checks_ok": _d(pop.get("query_checks")).get("ok")}


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


def c2_rules(commit: dict, ctx: dict) -> list[CheckVerdict]:
    """The C2-of-record block of commit_verdict (DECISIONS-2 §4; lines `sr = commit.get("sampled_replay")` .. `_scope`), evaluated rule by rule with
    the SAME nesting (a branch v1 never took is not taken here), emitting: local_replay/sampled_replay, /seed_of_record, /population_gaps,
    /population_scope, /attribution; boundary_linkage/ordered_linkage; checkpoint_binding/weights_of_record."""
    sr, lk, wp = _replay_of(commit), _linkage_of(commit), commit.get("weights_pin")
    row_proc = _is_row_procedure(commit)
    bnd = _d(commit.get("binding"))
    oc = _oracle_of(commit)
    C_SR = ("local_replay/sampled_replay", "DECISIONS-2 §4 C2 form (A) / F-cb-02",
            "A post-commitment stratified sample of required values replays exactly (bit-for-bit) from the committed bytes under the Program's registered relations.",
            ["sampled_replay.result", "sampled_replay.grade", "sampled_replay.why", "sampled_replay.sample", "sampled_replay_p*.json", "replay_cache"])
    C_SEED = ("local_replay/seed_of_record", "F-r16-23 / M-0579",
              "The replay's seed is post-commitment as a FACT: recomputed from the run root (or the recorded verifier challenge), never prover-chosen.",
              ["sampled_replay.sample.seed", "sampled_replay.sample.seed_source", "sampled_replay.sample.challenge"])
    C_GAP = ("local_replay/population_gaps", "COORD M-0721 (1) / F-r16-02 / F-r16-07 / F-r16-20",
             "Every VU the replay left unevaluated is named under a class form (B) covers, and form (B) is complete over the values where it must stand in; plane-fed VUs never pass by name.",
             ["sampled_replay.population.not_evaluable", "sampled_replay.strata_unevaluated", "sampled_replay.unevaluated_strata", "oracle_compare"])
    C_SCOPE = ("local_replay/population_scope", "R17-3 / R17-10 reqval",
               "The replay's population is the required-value manifest Q(P) of record, reconciled identity by identity (reach + uncovered-by-name == |Q(P)|; engines agree).",
               ["sampled_replay.population.query_total", "sampled_replay.population.query_checks", "sampled_replay.population.scope", "sampled_replay.population.uncovered_by_name"])
    C_ATT = ("local_replay/attribution", "F-cb-23 / rev M-0243",
             "At B >= 2 the per-(step, request) attribution of every committed value is verified by a complete form (B); else the gap is named on the PASS.",
             ["binding.population.requests", "oracle_compare.attribution"])
    C_LK = ("boundary_linkage/ordered_linkage", "DECISIONS-2 §4 C2 boundary linkage",
            "Every (engine step, request) boundary is linked: declared prompt -> input ids at the arrival step; sampled id[s-1] -> input id[s]; none unlinked.",
            ["boundary_linkage.result", "boundary_linkage.unlinked_n", "boundary_linkage.pairs", "boundary_linkage.horizon", "linkage"])
    C_WP = ("checkpoint_binding/weights_of_record", "MAN-09 / rev-cb M-0610",
            "Every served parameter the Program names equals by digest the checkpoint-composed tensor of record; every shard is pinned by its manifest sha256.",
            ["weights_pin.result", "weights_pin.why", "weights_pin.compared", "weights_pin.mismatched", "weights_pin.missing", "weights_pin.shards"])
    out: dict[str, CheckVerdict] = {}
    sr_scope, sr_inputs = _replay_scope(sr), _replay_inputs(sr, ctx)
    sr_notes = [_leak("sampled_replay.EVALUATORS / NOT_YET", "which families the replay can evaluate is a table by family name (the stated applicability domain)")]
    rc = ctx.get("replay_cache")
    if rc:
        sr_scope["replay_cache"] = {k: rc.get(k) for k in ("distinct_replays", "reused_pairs")}

    def put(spec: tuple, outcome: str, *, reason: str | None = None, differences: list | None = None, notes: list | None = None, scope: dict | None = None, inputs: dict | None = None) -> None:
        nm, rule, claim, compat = spec
        out[nm] = _mk(nm, rule, claim, compat, commit, ctx, outcome=outcome, reason=reason, differences=differences or [], notes=notes or [], scope=scope or {}, inputs=inputs or {})

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
            sok, swhy = CV._replay_seed_of_record(sr, commit.get("run_root"))
            if sok is False:
                missing_b = CV._form_b_complete_over_values(_oc)
                if missing_b is not None:
                    put(C_SEED, FAIL, reason=f"C2 of record not established (F-r16-23): {swhy}; and form (B) does not stand in: {missing_b}", inputs=sr_inputs,
                        differences=[{"name": "seed_not_post_commitment", "why": swhy, "form_b": missing_b}])
                else:
                    gap0, fail0 = CV._complete_replay_population_gap(sr, _oc) if sr.get("result") is True else CV._partial_replay_named_gap(sr, _oc)
                    if fail0 is not None:
                        put(C_GAP, FAIL, reason=fail0, differences=[{"name": "population_gap_uncovered", "why": fail0[:400]}])
                    elif sr.get("result") is not True and gap0 is None:
                        sr_why = sr.get("why") or sr.get("grade")
                        put(C_SR, FAIL, reason=(f"C2 of record not established (DECISIONS-2 §4): sampled replay {sr.get('grade')} -- {sr_why}; "
                                               f"the row's Commit is C1 + C3 only (F-r16-13, whatever seeded the replay: {swhy})"), scope=sr_scope, inputs=sr_inputs, notes=sr_notes,
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
            gap, fail = CV._partial_replay_named_gap(sr, _oc)
            if fail is not None:
                put(C_GAP, FAIL, reason=fail, differences=[{"name": "population_gap_uncovered", "why": fail[:400]}])
            elif gap is None:
                if "local_replay/sampled_replay" not in out:            # (a result-False replay was named above)
                    if not isinstance(sr, dict):
                        put(C_SR, INSUFFICIENT, reason="C2 of record not established (DECISIONS-2 §4): sampled replay None -- not run (no --program-dir / --programs-root); the row's Commit is C1 + C3 only",
                            notes=sr_notes, differences=[{"name": "replay_not_run"}])
                    else:
                        sr_why = sr.get("why") or sr.get("grade")
                        put(C_SR, FAIL, reason=f"C2 of record not established (DECISIONS-2 §4): sampled replay {sr.get('grade')} -- {sr_why}; the row's Commit is C1 + C3 only",
                            scope=sr_scope, inputs=sr_inputs, notes=sr_notes, differences=[{"name": "replay_partial_unnamed_gap", "grade": sr.get("grade"), "why": str(sr_why)[:300]}])
            else:
                put(C_GAP, PASS, notes=[gap])
                if "local_replay/sampled_replay" not in out:
                    put(C_SR, PASS, scope=sr_scope, inputs=sr_inputs, notes=sr_notes + ["partial replay, 0 mismatch; every unevaluated stratum named and covered by form (B)"])
        else:
            gap, fail = CV._complete_replay_population_gap(sr, _oc)
            if fail is not None:
                put(C_GAP, FAIL, reason=fail, differences=[{"name": "population_gap_uncovered", "why": fail[:400]}])
            else:
                put(C_GAP, PASS, notes=[gap] if gap else [])
        if not (isinstance(lk, dict) and lk.get("result") is True) and "boundary_linkage/ordered_linkage" not in out:
            lk_why = lk.get("why") if isinstance(lk, dict) else "not run"
            put(C_LK, INSUFFICIENT if not isinstance(lk, dict) else FAIL,
                reason=f"C2 of record not established (DECISIONS-2 §4): boundary linkage {lk.get('result') if isinstance(lk, dict) else None} -- {lk_why}", scope=lk_scope,
                differences=[{"name": "linkage_not_run" if not isinstance(lk, dict) else "linkage_not_established", "why": lk_why}])
        if not (isinstance(wp, dict) and wp.get("result") is True) and "checkpoint_binding/weights_of_record" not in out:
            wp_why = wp.get("why") if isinstance(wp, dict) else "not run (no --weights-of-record / --checkpoint-dir)"
            put(C_WP, INSUFFICIENT, reason=(f"weights unpinned (MAN-09): the served weights were not compared with the checkpoint of record -- {wp_why}; "
                                            "the committed weights root is self-consistency, not authentication"), scope=wp_scope, inputs=wp_inputs, notes=wp_notes,
                differences=[{"name": "weights_unpinned", "why": wp_why}])
        att = CV._b2_attribution_note(bnd, _oc)
        n_req = _int(_d(bnd.get("population")).get("requests"))
        if att is not None:
            put(C_ATT, PASS, notes=[att], scope={"requests": n_req, "attribution_complete": False}, differences=[{"name": "attribution_unverified", "requests": n_req}])
        elif n_req >= 2:
            put(C_ATT, PASS, scope={"requests": n_req, "attribution_complete": True})
        else:
            put(C_ATT, NOT_RUN, scope={"requests": n_req})
        scope_note, scope_fail = CV._query_population_scope(sr)
        if scope_fail is not None:
            acct = CV._scope_fail_account_missing(sr)
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
    return [out[n] for n in order]


def execution_extent_rules(commit: dict, ctx: dict) -> list[CheckVerdict]:
    """The executed prefix of record + its Match-account link + rows past the stop, read from binding.coverage (executed_prefix.py's outputs).
    v1 gated these THROUGH the coverage / replay results (a faulted class fails the replay's population scope by name); here each is its own
    check, and a MISSING ACCOUNT LEG (F-r19-int-19: tokens.json / global_match.json not beside the Commit) is INSUFFICIENT_EVIDENCE, not FAIL."""
    cov = _d(_d(commit.get("binding")).get("coverage"))
    epr = cov.get("executed_prefix_of_record")
    link = cov.get("match_account_link")
    past = cov.get("rows_past_the_stop")
    sr = _replay_of(commit) or {}
    facts = _d(_d(sr.get("executed_prefix_of_record")).get("facts_of_record"))
    C_EPR = ("execution_extent/executed_prefix_of_record", "R17-1 / F-moe-r17-03",
             "Each request's executed prefix is the cap on a length stop, or served + the Program's declared LAG on an EOS stop whose last served token is a declared EOS id; never defaulted.",
             ["binding.coverage.executed_prefix_of_record", "sampled_replay.executed_prefix_of_record", "executed_prefix_facts", "coverage.executed_prefix_of_record"])
    C_LINK = ("execution_extent/match_account_link", "COORD R17-27 (a)(b)",
              "The Commit's served lengths equal the Match record's (tokens.json) and GM-01's G6 account of the same attempt, and the LAG is attested on every leg.",
              ["binding.coverage.match_account_link", "executed_prefix_facts.linked", "executed_prefix_facts.lag_attested", "account_missing"])
    C_PAST = ("execution_extent/rows_past_the_stop", "COORD R17-27 (c)",
              "No committed row lies at a position >= a request's executed prefix (execution past the stop).",
              ["binding.coverage.rows_past_the_stop", "executed_prefix_facts.rows_past_stop"])
    out = []
    inputs_extra = {}
    tj, g6 = _d(_d(link).get("tokens_json")), _d(_d(link).get("global_match"))
    if tj.get("sha256"):
        inputs_extra["tokens_json_sha256"] = tj.get("sha256")
    if g6.get("sha256"):
        inputs_extra["global_match_sha256"] = g6.get("sha256")
    if not isinstance(epr, dict):
        out.append(_mk(*C_EPR, commit, ctx, outcome=NOT_RUN))
    else:
        reqs = _d(epr.get("requests"))
        scope = {"requests": len(reqs), "lag": epr.get("lag"), "eos_of_record": epr.get("eos_of_record"), "served_known": epr.get("served_known"), "any_eos_stop": epr.get("any_eos_stop"),
                 "lengths": {r: [v.get("served_output_len"), v.get("executed_prefix_len"), v.get("cap")] for r, v in reqs.items() if isinstance(v, dict)}}
        if epr.get("ok") is True:
            out.append(_mk(*C_EPR, commit, ctx, outcome=PASS, scope=scope))
        else:
            probs = list(epr.get("problems") or [])
            out.append(_mk(*C_EPR, commit, ctx, outcome=FAIL, scope=scope, differences=[{"name": "executed_prefix_problems", "n": len(probs), "first": probs[:4]}],
                           reason=f"executed prefix of record: {probs[0] if probs else 'not ok'}"))
    account_missing = sorted({leg for f in facts.values() if isinstance(f, dict) for leg in (f.get("account_missing") or [])})
    if not account_missing and isinstance(link, dict):
        account_missing = [leg for leg, present in (("tokens.json", bool(tj)), ("global_match.json", bool(g6))) if not present]
    # v1 gated the link THROUGH the replay's not-executed class: it faults (-> population_scope FAIL) only when applied, i.e. when a request's
    # prefix is reduced from its cap (`link.required`) or the class names the fault.  Outside that the link is recorded, not gating.
    gating = bool(_d(link).get("required")) or any(f.get("account_missing") or f.get("linked") is not True for f in CV._faulted_prefix_facts(sr))
    if not isinstance(link, dict):
        out.append(_mk(*C_LINK, commit, ctx, outcome=NOT_RUN))
    else:
        scope = {"required": link.get("required"), "gating": gating, "lag_attested": link.get("lag_attested"), "tokens_json": tj.get("path"), "global_match_verdict": g6.get("verdict"),
                 "g6_status": g6.get("g6_status"), "requests_linked": len(g6.get("requests_linked") or []) or None,
                 "facts_linked": {r: f.get("linked") for r, f in facts.items() if isinstance(f, dict)} or None, "account_missing": account_missing or None}
        probs = list(link.get("problems") or [])
        if account_missing and gating:
            out.append(_mk(*C_LINK, commit, ctx, outcome=INSUFFICIENT, scope=scope, inputs=inputs_extra,
                           differences=[{"name": "account_missing", "legs": account_missing}],
                           reason=f"Match account leg(s) NOT beside the Commit: {', '.join(account_missing)} -- a missing input, not a verdict on the Program (F-r19-int-19)"))
        elif account_missing:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra,
                           notes=[f"Match account leg(s) not beside the Commit: {', '.join(account_missing)}; not gating (no request stopped before its cap)"]))
        elif link.get("ok") is True:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra))
        elif gating:
            out.append(_mk(*C_LINK, commit, ctx, outcome=FAIL, scope=scope, inputs=inputs_extra,
                           differences=[{"name": "account_not_linked", "n": len(probs), "first": probs[:4]}], reason=f"Match account link: {probs[0] if probs else 'not ok'}"))
        else:
            out.append(_mk(*C_LINK, commit, ctx, outcome=PASS, scope=scope, inputs=inputs_extra,
                           notes=[f"link problems recorded, not gating (no request stopped before its cap): {'; '.join(p[:120] for p in probs[:3])}"]))
    if not isinstance(past, dict):
        out.append(_mk(*C_PAST, commit, ctx, outcome=NOT_RUN))
    else:
        scope = {"entries_n": past.get("entries_n"), "padded_record": past.get("padded_record"), "padding_entries_n": past.get("padding_entries_n")}
        if past.get("ok") is not False and not _int(past.get("entries_n")) and not past.get("error"):
            out.append(_mk(*C_PAST, commit, ctx, outcome=PASS, scope=scope))
        else:
            out.append(_mk(*C_PAST, commit, ctx, outcome=FAIL, scope=scope,
                           differences=[{"name": "rows_past_the_stop", "n": past.get("entries_n"), "by_request": past.get("steps_past_by_request"), "error": past.get("error")}],
                           reason=f"{past.get('entries_n')} committed row(s) past a request's executed prefix"))
    return out


def run_checks(commit: dict, tokens_equal: Any, *, min_openings: int = 64, declaration: dict | None = None, required_classes: Any = None,
               ctx: dict | None = None) -> list[CheckVerdict]:
    """Every rule of `commit_verdict.commit_verdict` (+ commit_delta's compiled conditions, row_pod's manifest verify) for ONE instrumented run,
    each with its own outcome.  `ctx`: record-level facts (program_digest, manifest_digest, required_classes_source, manifest_verify, file_digests,
    replay_cache)."""
    ctx = dict(ctx or {})
    checks = [rule_tokens_equal(commit, tokens_equal, ctx), rule_reference_integrity(commit, ctx), rule_fa2_zero_rows(commit, ctx), rule_collector_errors(commit, ctx),
              rule_openings(commit, ctx, min_openings), rule_openings_after_release(commit, ctx, min_openings), rule_openings_population(commit, ctx, min_openings),
              rule_bounded_staging(commit, ctx), rule_oracle_compare(commit, ctx), rule_compiled_check(commit, ctx)]
    checks += c2_rules(commit, ctx)
    checks += execution_extent_rules(commit, ctx)
    checks += [rule_identity_coverage(commit, ctx), rule_manifest_coverage(commit, ctx), rule_required_classes(commit, ctx, declaration, required_classes), rule_manifest_verified(commit, ctx)]
    checks += [rule_identity_layout(commit, ctx), rule_binding_context(commit, ctx)]
    return checks


def run_outcome(checks: list[CheckVerdict]) -> str:
    """The outcome of one run's rule set (= v1's commit_pass when no rule is FAIL / INSUFFICIENT_EVIDENCE)."""
    return worst([c.outcome for c in checks if c.outcome != NOT_RUN]) if any(c.outcome != NOT_RUN for c in checks) else NOT_RUN


# ---------------------------------------------------------------------------------------------------------------------- aggregation

def aggregate_runs(per_run: list[tuple[int | None, list[CheckVerdict]]]) -> list[CheckVerdict]:
    """Rule-level checks across the instrumented runs: the worst outcome; scope from the first run + `runs`; differences tagged by `pair`
    from the worst runs; inputs = the first run's (+ every distinct run root)."""
    by_name: dict[str, list[tuple[int | None, CheckVerdict]]] = {}
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
        out.append(CheckVerdict(name=name, claim=first.claim, inputs=inputs, scope=scope, outcome=o, differences=diffs, compat=list(first.compat),
                                group=first.group, rule=first.rule, notes=notes, reason=reason))
    return out


def group_checks(rules: list[CheckVerdict]) -> list[CheckVerdict]:
    """One CheckVerdict per group, aggregating its rules (the worst outcome; NOT_RUN only when every rule is NOT_RUN)."""
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
        out.append(CheckVerdict(name=g, claim=spec["claim"], inputs=inputs, scope={"rules": {r.name: r.outcome for r in mine}}, outcome=o, differences=diffs,
                                compat=compat, group=g, rule=None, notes=notes, reason=(reasons[0] if reasons else None)))
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
    per_run: list[tuple[int | None, list[CheckVerdict]]] = []
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


@dataclass(frozen=True, kw_only=True)
class FromRecordOptions:
    """rebuild <row_dir>/verdict.json from the row's Commit evidence"""

    row_dir: str = option("row_dir")
    out: str | None = option("--out", default=None, help="where to write (default <row_dir>/verdict.json)")
    rc: int | None = option("--rc", type=int, default=None,
                            help="commit_delta's exit code (row_pod.sh passes it; a pre-P5 row's commit_summary.json supplies it otherwise)")
    stage_ok: str | None = option("--stage-ok", default=None,
                                  help="a stage decision made outside this verdict, if any (recorded as compat.stage_ok; a disagreement is a note)")
    quiet: bool = option("--quiet", default=False, action="store_true")
    cmd: str = field(default="from-record", init=False)


@dataclass(frozen=True, kw_only=True)
class MappingOptions:
    """print the old -> new check-name table (markdown)"""

    cmd: str = field(default="mapping", init=False)


COMMANDS = {"from-record": FromRecordOptions, "mapping": MappingOptions}


def main(a: FromRecordOptions | MappingOptions) -> int:
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
