"""The Commit verdict's names: the seven named checks (`GROUPS`: each group's claim and the v1 fields it was read under) and the
v1 -> v2 name table (`OLD_TO_NEW`, rendered by `mapping_markdown`)."""

from __future__ import annotations

from typing import Any

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
        "claim": ("Every served parameter the Program names equals, by digest, the checkpoint-composed tensor of record, "
                  "with the shards pinned by their manifest sha256."),
        "compat": ["weights_pin"]},
    "execution_extent": {
        "claim": ("Each request's executed prefix of record (served length; the cap on a length stop, served + LAG on an EOS stop) is "
                  "derived from the declared stop rule, linked to the Match account of the same attempt (with no Match stage, to the Commit's token record bound to the commitment), and no committed row lies past it."),
        "compat": ["executed_prefix", "executed_prefix_of_record", "executed_prefix_facts", "match_account_link", "commit_account_link", "rows_past_the_stop", "account_missing"]},
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
    {"old": "cublas_route_pin", "where": "runs.jsonl commit (check.cublas_route_pin.stamp)", "new": "local_replay/cublas_route_pin"},
    {"old": "boundary_linkage", "where": "runs.jsonl commit; commit/sampled_replay_p*.json", "new": "boundary_linkage/ordered_linkage"},
    {"old": "linkage", "where": "prose", "new": "boundary_linkage"},
    {"old": "weights_pin", "where": "runs.jsonl commit", "new": "checkpoint_binding/weights_of_record"},
    {"old": "executed_prefix", "where": "prose", "new": "execution_extent"},
    {"old": "identity", "where": "prose", "new": "program_source_identity"},
    {"old": "account_missing", "where": "sampled_replay executed_prefix_of_record.facts_of_record", "new": "execution_extent/match_account_link differences[account_missing] -> INSUFFICIENT_EVIDENCE"},
    {"old": "run_root", "where": "runs.jsonl commit", "new": "every check's inputs.run_root"},
    {"old": "rules_version", "where": "runs.jsonl commit", "new": "verdict.execution.rules_version"},
    {"old": "manifest_verify.ok", "where": "commit/manifest_verify.json", "new": "required_value_coverage/manifest_of_record_verified"},
    {"old": "commit_summary.pass", "where": "commit_summary.json", "new": "compat.commit_summary_pass (deprecated: rc == 0 and tokens_equal)"},
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
    {"old": "extra.collector_failure", "where": "runs.jsonl commit.extra", "new": "runtime_match/collector_errors (differences[collector_failure])"},
    {"old": "declaration_*.json", "where": "commit/ (commit_integ.api.declaration)", "new": "required_value_coverage/required_classes (inputs)"},
    {"old": "sampled_replay_p*.json", "where": "commit/", "new": "local_replay/* (inputs: file sha256s under verdict.inputs)"},
]


def mapping_markdown() -> str:
    lines = ["| old (v1 field) | where | new (v2 check / field) |", "|---|---|---|"]
    for m in OLD_TO_NEW:
        lines.append(f"| `{m['old']}` | {m['where']} | {m['new']} |")
    lines += ["", "| check | claim |", "|---|---|"]
    for g, spec in GROUPS.items():
        lines.append(f"| `{g}` | {spec['claim']} |")
    return "\n".join(lines) + "\n"
