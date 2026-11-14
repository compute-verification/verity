# RULES.md — current operational defaults ({{policy.project}})

Seed authored by {{policy.seed_author}} {{policy.seed_utc}} from the rulings in force ({{policy.rulings}}). Under the runner brief (RUNNER-TELEMETRY-BRIEF.md §2) increment A makes the runner configuration the source and renders this file from it; until then this file is the statement of record and supersedes the message stream where they differ. Policy version: `{{policy.version}}`. Every attempt records the policy version it ran under.

## Entry point
- Every configuration job enters through `{{entry.runner}}` on the pod. Standalone commands are diagnostics only and produce no cell of record.
- The runner records the release tag and tree sha (`{{entry.release_record}}`), workload digest, stage list, admission readings, and `waited=` with the gate that held the job.

## Checker path
- `MATCH_IMPL={{checker.MATCH_IMPL}}` is the Match of record for every configuration class rev has signed ({{checker.fast_signed_ruling}}). The chain writes `{{checker.fast_document}}`.
- The record checker runs only as an {{checker.record_checker_route_md}}, never in a row's critical path and never beside a Commit in the same cgroup.
- X-09 runs inside the shared pipeline (`MATCH_PIPELINE={{checker.MATCH_PIPELINE}}` beside `MATCH_IMPL={{checker.MATCH_IMPL}}`); the standalone `batch_decomp` path is {{checker.standalone_batch_decomp_supported_md}}.
- A row completes on strict fast/record agreement or on a rev-signed class; an audit pending is marked `{{checker.audit_pending_label}}`, not a failure.

## Admission
- Host memory is admitted against the pod cgroup limit (`{{admission.memory_source_v2}}` / `{{admission.memory_source_v1}}`), read in {{admission.memory_units}}, never against `{{admission.memory_never}}`. `{{admission.need_var}}` is the stage's declared need; the gate compares against {{admission.headroom_basis}}, not usage including page cache.
- {{admission.host_heavy_per_cgroup_word}} host-memory-heavy stage ({{list:admission.host_heavy_stages}}) per cgroup at a time. A GPU that looks idle is not admission.
- GPU stages are exclusive per device (`GPU_GATE={{admission.GPU_GATE}}`).
- Disk: a job launches only if free space covers its declared output plus a {{admission.disk_reserve_gb}} GB reserve for logs, manifests, and failure records.
- No identical automatic retry after {{oxford_or:admission.identical_retry_after}}; a named next attempt is a decision.

## Queues
- Every pod has a queue; the next eligible job launches on success or named failure without a GO.
- Holds are files with reasons ({{codelist:queues.hold_markers}}) and are recorded as intervals.
- A pod whose queue is exhausted is terminated the same round ({{queues.exhausted_pod_rule}} hard rule).

## Releases
- No fleet-wide ship. A release ships to a named pod in an idle gap or a coordinated HOLD gap; no `RELEASING` marker stays on a pod with a running chain.
- Change-proportional gate: data-only declaration → {{releases.gate_data_only}}; tooling → {{releases.gate_tooling}}; native/collector change → {{releases.gate_native}}.
- A byte-identical approved candidate is tagged without rerun; a cherry-pick into a different candidate reruns the tests the compatibility check names.
- Near a deadline, new scope freezes; already-reviewed fixes may still release.
- Rows of record already complete stay byte-neutral under every ship.

## Evidence
- The row attempt directory on the pod is the bundle. Every completed stage is pulled to the lane worktree (`{{evidence.pull_tool}}`, `{{evidence.pull_flag}}` when a Build/Match pull exists) and mirrored to `{{evidence.mirror_root}}` before the row is posted complete.
- Excluded payloads are listed with sha256. Nothing is deleted from a pod until its pull is verified.
- The finalizer classifies from `stages.txt` + records + (when present) `failure.json`; the kill reason is never inferred from `rc=137` alone.

## Lanes
- A lane is a worktree + board identity + `STATUS`/timelog; a session is a shift. End a shift with a HANDOFF naming the exact next commands; resume by lane identity.
- One owner per lane. A lane silent {{lanes.silence_minutes}} minutes past a direct ask may be reconciled; adoption increments the ownership generation and is posted before any launch. A relaunched session that finds the original alive takes a new bounded role, never the same worktree.
- Every post to the board is by lane identity; the board is the event log.

## Fleet and budget
- Project `{{fleet.project}}`; ephemeral pods `{{fleet.ephemeral_prefix}}`; persistent `{{fleet.persistent_prefix}}`. Destructive operations act only on registered pod ids; foreign resources are read-only and excluded from accounting.
- Cap and deadline live on the deadman pod (`{{fleet.cap_file}}`, `{{fleet.rearm_tool}}` by PID file). The cap wins; it is never raised automatically. Stage timeout, pod lease, project cap, and the reporting schedule are distinct.
- Current: cap ${{fleet.cap_usd}}, deadline {{fleet.deadline_utc}}, pod-max {{fleet.pod_max_h}} h.

## Classification
- Rows report in {{count_word:classification.states}} states: {{slashlist:classification.states}}, plus the sub-state "{{classification.sub_state}}".
- GENUINELY BLOCKED requires a reproduced, quantitative failure under the best known implementation on a named resource class, with the kill reason and peak evidence recorded; a mechanism claim additionally requires a controlled experiment.
- Padded rows carry their actual population and query (`{{classification.padded_query}}`), labelled separately from active-reference rows; equal sample counts are an engineering-cost comparison, not an integrity claim.
