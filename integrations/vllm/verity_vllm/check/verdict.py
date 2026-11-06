"""The Commit VERDICT as typed records (`vllm-verdict/v1`): every check the Commit decides, by its human name, with what it read, what
it established and what it found -- ONE document (`<row>/verdict.json`) beside the files it is derived from.

    verity-vllm verdict from-record <row_dir> [--out PATH]      rebuild the record from a row's evidence
    verity-vllm verdict mapping                                  the old -> new check-name table (markdown)

WHY.  The v1 Commit records its decision three times over: `commit/verdict.json` (`commit_pass`, one `first_fail_reason` string: the FIRST
rule that failed, the others silent), `commit/runs.jsonl` (per instrumented run, every rule's own block under `commit.*`, the check
names being the field names they grew under -- `weights_pin`, `linkage`, `coverage`, `identity_check`) and `commit_summary.json`, whose
`pass` degenerated to `rc == 0 and tokens_equal` once `openings` / `coverage` stopped being carried on runs.jsonl.  The
verdict of record was `commit/verdict.json` + `commit/sampled_replay_p*.json`, read together by whoever knew to.

WHAT.  A `CommitChecks` = the seven named checks (groups) and the R*/X* rules under them, each a `CheckResult`:

    name        `<group>` for the group, `<group>/<rule>` for a rule                    (human names: see GROUPS)
    claim       one sentence: what PASS establishes
    inputs      digests / ids of what the check read (program_digest, manifest_digest, run root, file sha256s, the sample seed)
    scope       what was actually established (sample_size, positions, layers, requests, steps ...) -- never more than the record shows
    outcome     PASS | FAIL | INSUFFICIENT_EVIDENCE | NOT_RUN
    differences named, machine-readable findings ({"name": ..., ...}; e.g. `fa2_layers_bad`, `identities_unbound`, `account_missing`)
    compat      the v1 field name(s) the check was read from (the old -> new mapping is `verdict_names.OLD_TO_NEW`, tested for completeness)
    rule        the rule's check id, e.g. X-03 or BIND-01 (None on a group)
    notes       prose the PASS carries (named gaps, `structural_leak: <table>` where the rule rests on a name table, not on the Program)

The decisions are v1's, unchanged: every rule (`check.commit_rules`, `check.c2_rules`) is `commit_pass.commit_verdict`'s own predicate, evaluated
one by one instead of first-failure-wins, so `CommitChecks.outcome == PASS` iff v1's `commit_pass` (the record's own `commit_pass` is read
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity_vllm.check import c2_rules as C2
from verity_vllm.check import commit_rules as CR
from verity_vllm.check.commit_rules import _d, _int
from verity_vllm.check.result import FAIL, INSUFFICIENT, NOT_RUN, OUTCOMES, PASS, CheckResult, worst
from verity_vllm.check.verdict_names import GROUPS
from verity_vllm.properties.record import citations
from verity_vllm.records.build_summary import BuildSummary
from verity_vllm.records.commit_run import CommitRun
from verity_vllm.records.fields import RecordError
# Imported from here by modules open PRs edit (pipeline/commit.py, cpu_replay.py, replay_bundle.py, tp/commit.py and three tests); they
# move to commit_pass after those land.
from verity_vllm.check.commit_pass import _after_release_verdict, _c1_verdict, commit_verdict  # noqa: F401
from verity_vllm.check.commit_rules import REQUIRED_CLASSES_DEFAULT  # noqa: F401

SCHEMA = "vllm-verdict/v1"


# ---------------------------------------------------------------------------------------------------------------------- the records

@dataclass
class CommitChecks:
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
    properties: list[dict[str, Any]] = field(default_factory=list)   # citations of the run's property records; omitted when none

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
        d = {k: v for k, v in asdict(self).items() if v or k != "properties"}
        d["checks"] = [c.to_dict() for c in self.checks]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CommitChecks":
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
    def load(cls, path: str | os.PathLike) -> "CommitChecks":
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


def match_verdict(results: list[CheckResult], required: tuple[str, ...]) -> str:
    """A Match run's word over its gates: REJECTED when any gate FAILs; INCOMPLETE when a `required` gate is absent or did not run,
    or any gate is INSUFFICIENT_EVIDENCE; ACCEPTED otherwise.  A gate outside `required` that did not run does not weigh."""
    if any(r.outcome == FAIL for r in results):
        return "REJECTED"
    req = [r for r in results if r.name in required]
    if {r.name for r in req} != set(required) or any(r.outcome == NOT_RUN for r in req) or any(r.outcome == INSUFFICIENT for r in results):
        return "INCOMPLETE"
    return "ACCEPTED"


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


def correspondence_of_build(build_summary: BuildSummary | Mapping[str, Any] | None, *, result_jsons: Mapping[str, dict] | None = None
                            ) -> tuple[str | None, str, dict[str, str]]:
    """(correspondence_digest, source, {wrapper: digest}) from the Build's own record: `build_summary.json` (`records.build_summary`;
    its writer lifts `result.json:program.correspondence_digest` per wrapper), else the wrappers' `result.json` directly.  A Build made
    with the v1 frontend records none -> (None, "absent", {}): never a stand-in hash (the row's global_match.json digest is an INPUT,
    `inputs`)."""
    digests = {} if build_summary is None else BuildSummary.of(build_summary, "build_summary.json").correspondence_digests()
    source = "absent"
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
    --expect-root from the command line (open_after_release also when a run records its openings_after_release: the Commit's default).  This is what `commit_summary.json` used to carry (that file is no longer
    written; the harness keeps a reader for old records); verdict.json `execution` is the record."""
    d = Path(row_dir)
    out: dict = {}
    s = json.load(open(d / "commit" / "summary.json")) if (d / "commit" / "summary.json").is_file() else {}
    runs = CommitRun.load_lines(d / "commit" / "runs.jsonl") if (d / "commit" / "runs.jsonl").is_file() else []
    trt = [r for r in runs if not (r.arm == "control" or r.committer in (None, "control"))]
    cmd = None
    if (d / "commit.log").is_file():
        first = open(d / "commit.log", encoding="utf-8", errors="replace").readline().strip()
        if first.startswith("$ "):
            cmd = first[2:]
    out["cmd"] = cmd
    out["committer"] = s.get("committer")
    coll = next((r.commit.collector for r in trt if r.commit is not None and r.commit.collector is not None), None)
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
            except ValueError:
                es[k] = v
        for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--openings", "openings"), ("--pairs", "pairs")):
            m = re.search(flag + r"\s+(\d+)", cmd)
            if m:
                es[key] = int(m.group(1))
        es["open_after_release"] = "--open-after-release" in cmd or any(r.commit is not None and r.commit.openings_after_release is not None for r in trt)
        es["required_manifest"] = bool(re.search(r"--required-manifest\s+\S+", cmd))
        m = re.search(r"--expect-root\s+([0-9a-f]{64})", cmd)
        if m:
            es["expect_root"] = m.group(1)
    out["engine_settings"] = es or None
    return out


def build(rows: Sequence[CommitRun | Mapping[str, Any]], verdict_doc: dict | None, *, summary: dict | None = None, manifest_verify: dict | None = None,
          file_digests: dict[str, str] | None = None, row: str | None = None, min_openings: int | None = None, declaration: dict | None = None,
          rc: int | None = None, build_summary: BuildSummary | None = None, build_results: Mapping[str, dict] | None = None, execution: dict | None = None) -> CommitChecks:
    """The `CommitChecks` from the Commit's own documents: `rows` = runs.jsonl entries (control + instrumented; `records.commit_run`, a parsed
    line is read as one), `verdict_doc` = commit/verdict.json,
    `manifest_verify` = commit/manifest_verify.json, `execution` = `execution_of_record(row_dir)` (the Commit's own configuration record);
    `summary` = an older record's commit_summary.json (old rows only: its execution fields stand in for `execution`, its `rc` for a missing `rc`).
    `declaration` defaults to the run's own `commit.classes`.  `build_summary` (build_summary.json) / `build_results` ({wrapper: result.json})
    give the Build's `runtime-correspondence/v1` digest."""
    vd = _d(verdict_doc)
    if execution is None:
        execution = {k: v for k, v in _d(summary).items() if k in EXECUTION_KEYS}
    elif summary:
        # an older row whose evidence is partial here (e.g. commit.log not beside the record): its recorded summary fills what the
        # evidence did not state -- the record's own statement, never a stand-in
        execution = {**{k: v for k, v in _d(summary).items() if k in EXECUTION_KEYS}, **{k: v for k, v in execution.items() if v is not None}}
    rm = _d(vd.get("required_manifest"))
    program_digest = rm.get("program_digest")
    manifest_digest = rm.get("manifest_digest")
    query_id = _d(rm.get("query")).get("query_id")
    runs = [CommitRun.of(r, f"runs.jsonl[{i}]") for i, r in enumerate(rows)]
    treatment = [r for r in runs if r.arm != "control" and r.committer not in (None, "control") and r.commit is not None]
    if not treatment:
        treatment = [r for r in runs if r.arm != "control" and r.commit is not None]
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
        c = r.commit
        if not program_digest:
            ctx["program_digest"] = c.binding_program_digest or c.weights_pin_program_digest
        if not manifest_digest:
            ctx["manifest_digest"] = c.coverage_manifest_digest
        decl = declaration
        if decl is None and c.classes is not None:
            decl = {"classes": [{"class": k, **_d(v)} for k, v in c.classes.items()]}
        checks = run_checks(dict(c.entry), r.tokens_equal, min_openings=min_openings, declaration=decl, required_classes=required_classes, ctx=ctx)
        per_run.append((r.pair, checks))
        recorded.append((r.pair, c.commit_pass, c.commit_fail_reason))
        if c.commit_pass is not None and (run_outcome(checks) == PASS) != c.commit_pass:
            mismatches.append({"pair": r.pair, "recorded_commit_pass": c.commit_pass, "reconstructed": run_outcome(checks), "recorded_reason": c.commit_fail_reason,
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
        passes = [r.commit.commit_pass for r in treatment]
        commit_pass = all(passes) if None not in passes else None
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
        execution["rules_version"] = treatment[0].commit.rules_version
    compat = {"commit_pass": commit_pass, "first_fail_reason": vd.get("first_fail_reason"), "runs_failed": vd.get("runs_failed"),
              "rc": rc_v, "commit_summary_pass": sums.get("pass") if summary else None,
              "commit_summary_deprecated": "commit_summary.json is no longer written (its pass was rc == 0 and tokens_equal); this verdict is the record",
              "per_run": [{"pair": p, "commit_pass": rec, "commit_fail_reason": why} for p, rec, why in recorded],
              "negatives": vd.get("negatives"), "reconstruction_mismatch": mismatches or None, "only_arm": vd.get("only_arm")}
    runs_ = {"instrumented": len(treatment), "failed": sum(1 for _, rec, _ in recorded if rec is False), "control": sum(1 for r in runs if r.arm == "control"),
             "not_pass_by_reconstruction": sum(1 for _, ch in per_run if run_outcome(ch) != PASS), "pairs": [p for p, _ in per_run]}
    cdig, csrc, cdigs = correspondence_of_build(build_summary, result_jsons=build_results)
    return CommitChecks(checks=groups + rules, outcome=outcome, program_digest=ctx.get("program_digest"), manifest_digest=ctx.get("manifest_digest"), query_id=query_id,
                        correspondence_digest=cdig, correspondence_digest_source=csrc, correspondence_digests=cdigs,
                        row=row, run_roots=list(dict.fromkeys(str(x) for x in (vd.get("run_roots") or [r.commit.run_root for r in treatment]) if x)),
                        runs=runs_, inputs=dict(file_digests or {}), execution=execution, compat=compat, notes=notes)


RECORD_FILES = ("commit/verdict.json", "commit/runs.jsonl", "commit_summary.json", "commit/manifest_verify.json", "commit/summary.json", "commit.log", "global_match.json",
                "match_summary.json", "build_summary.json")


def from_record(row_dir: str | os.PathLike, *, rc: int | None = None, stage_ok: str | None = None) -> CommitChecks:
    """Rebuild the `CommitChecks` from a row dir's evidence (`commit/runs.jsonl`, `commit/verdict.json`, `commit/summary.json` + `commit.log`
    (`execution_of_record`), `commit/manifest_verify.json`, `commit/sampled_replay_p*.json` for their digests; an older row's
    `commit_summary.json` is read when present).  A row whose Commit never ran gives NOT_RUN with the missing files named.
    `rc` = commit_delta's exit code when the caller (row_pod.sh) knows it; `stage_ok` = a stage decision made OUTSIDE this verdict (older
    rows: the commit_summary word), recorded under `compat.stage_ok` and compared (a disagreement is a note, never a changed decision).
    The verdict's `outcome` IS the Commit stage's decision (row_pod.sh reads it back).  Every `<row>/properties/*.json` is cited
    under `properties` by digest; a record that does not hold, or whose digest does not recompute, is a note, never a changed decision.
    A `build_summary.json` its record (`records.build_summary`) refuses is a note too, and the wrappers' own `result.json` give the
    correspondence digest."""
    d = Path(row_dir)
    digests: dict[str, str] = {}
    for rel in RECORD_FILES + tuple(sorted(os.path.relpath(p, d) for p in glob.glob(str(d / "commit" / "sampled_replay_p*.json")))):
        p = d / rel
        if p.is_file():
            digests[rel] = _sha256(p)
    rows = CommitRun.load_lines(d / "commit" / "runs.jsonl") if (d / "commit" / "runs.jsonl").is_file() else []
    vd = json.load(open(d / "commit" / "verdict.json")) if (d / "commit" / "verdict.json").is_file() else None
    summ = json.load(open(d / "commit_summary.json")) if (d / "commit_summary.json").is_file() else None
    mv = json.load(open(d / "commit" / "manifest_verify.json")) if (d / "commit" / "manifest_verify.json").is_file() else None
    bs: BuildSummary | None = None
    refused = None
    if (d / "build_summary.json").is_file():
        try:
            bs = BuildSummary.load(d / "build_summary.json")
        except RecordError as e:
            refused = f"build_summary.json not read ({e}): the Build's correspondence digest is read from its wrappers' result.json"
    results: dict[str, dict] = {}
    unreadable: list[str] = []
    if bs is None:                                    # no build.summary on the row: the wrappers' own result.json (build_step/, build_request*/)
        for p in sorted(glob.glob(str(d / "build_*" / "result.json"))):
            w = os.path.basename(os.path.dirname(p))
            try:
                results[{"build_step": "step", "build_request": "request"}.get(w, w)] = json.load(open(p))
            except (OSError, ValueError) as e:
                unreadable.append(f"{os.path.relpath(p, d)} ({e!r:.200})")
    v = build(rows, vd, summary=summ, manifest_verify=mv, file_digests=digests, row=d.name, rc=rc, build_summary=bs, build_results=results,
              execution=execution_of_record(d))
    if refused:
        v.notes.append(refused)
    missing = [rel for rel in ("commit/verdict.json", "commit/runs.jsonl") if rel not in digests]
    if missing:
        v.notes.append(f"record incomplete: {', '.join(missing)} not on the row (Commit never closed / never ran)")
        if v.outcome == PASS:
            v.outcome = INSUFFICIENT
    if unreadable:
        v.notes.append(f"build record unreadable: {'; '.join(unreadable)}; the correspondence digest is read from the other wrappers' records only")
    if (d / "stages.txt").is_file():
        stage = [ln for ln in (d / "stages.txt").read_text().splitlines() if ln.startswith("commit ")]
        if stage:
            v.compat["stage_line"] = stage[-1][:200]
            v.compat["stage_outcome"] = stage[-1].split()[1]
    if stage_ok is not None:
        v.compat["stage_ok"] = stage_ok
        if (stage_ok.upper() == PASS) != (v.outcome == PASS):
            v.notes.append(f"STAGE MISMATCH: row_pod.sh's Commit stage says {stage_ok} but the rule-by-rule outcome is {v.outcome}; the stage decision stands")
    v.properties, notes = citations(d)
    v.notes += notes
    return v
