"""Shared runner and telemetry v0.1 (R17-49, RUNNER-TELEMETRY-BRIEF.md) -- increment A.

Project-neutral core: jobs, stages, processes, resources, evidence.  Stdlib only (Python 3.11+); pods have no guaranteed extras.

    job.json       identity of the attempt                         telemetry.job
    status.json    state machine of the attempt                    telemetry.job
    events.jsonl   spans / marks / holds / cancel markers          telemetry.events   (ONE append helper)
    resources.jsonl scoped resource samples (Level-0 sidecar)      telemetry.sample
    failure.json   deterministic classification                    telemetry.classify
    RULES.md       rendered from a policy file                     telemetry.policy

`research tele <cmd>` / `python3 -m research.telemetry <cmd>` -- see telemetry.cli.

Promoted from veritor zk/campaign/telemetry @ df7fbea3d (lane/r17-mprof, 2026-09-21); import paths adjusted, nothing else.
"""
from __future__ import annotations

SCHEMA = "telemetry/v0.1"
JOB_SCHEMA = "telemetry/job/v0.1"
STATUS_SCHEMA = "telemetry/status/v0.1"
EVENTS_SCHEMA = "telemetry/events/v0.1"
RESOURCES_SCHEMA = "telemetry/resources/v0.1"
FAILURE_SCHEMA = "telemetry/failure/v0.1"
COLLECTOR_VERSION = "0.1.3"   # 0.1.1: /proc/<pid>/oom_score sampled for tree + outsiders; TELE_PARENT_SPAN/TELE_ATTEMPT_DIR defaults
                              # 0.1.2: cgroup v1 memory.oom_control without an oom_kill field (kernel < 4.13) samples oom_kill=None, not 0
                              # 0.1.3: smaps_rollup is never read for a process over --pss-max-rss-gb (pss record: per-process
                              #   skipped + reason + status RSS, a `skipped` count beside `denied`), nor again sooner than 200x
                              #   its last read took (read_s); --pss-every 0 turns PSS off
CLASSIFIER_VERSION = "0.1.4"   # 0.1.1: shared-scope attribution window extended by half a sampler round (process record follows the cgroup record)
                               # 0.1.4: the run's queue scope (cluster-<run id>.scope) is its own cgroup; rule 6b CGROUP_OOM_ATTRIBUTED = SIGTERM
                               #   + oom_kill increment there (systemd stops the scope after the kill); OOM classes there name the cap (MemoryMax=NG)
                               # 0.1.3 (r20 issue 3): rule 7b CGROUP_OOM = rc != 0 without a signal + oom_kill increment (v2 memory.events or v1
                               #   memory.oom_control) + a `Killed` line in the job's log; v1 memory.failcnt growth recorded as corroboration.  A v1
                               #   cgroup with no oom_kill counter (kernel < 4.13) falls back to failcnt growth / under_oom + the Killed line, one
                               #   certainty step lower; log.killed is evidence whenever the line is present
                               # 0.1.2 (rev M-1748 d1-d4/d8 + S-r17-51 c1/c2): shared-scope attribution = a tree process vanished AND no outsider vanished
                               #   (oom_score / RSS ranking recorded as corroboration, never a veto); any class chosen beside SIGKILL + oom_kill delta is
                               #   ambiguous with both evidence sets under `competing` (a delivered + verified marker stays established); low mounts count
                               #   only when they hold the attempt or a declared output; no samples -> UNKNOWN_SIGNAL ambiguous; incidents carry vanished
                               #   pids + last oom_score + attribution_sample_age_s

HOLD_REASONS = ("dependency", "gpu_reservation", "host_memory_reservation", "release", "manual", "other")
SCOPES = ("process", "job_cgroup", "shared_pod_cgroup", "host", "gpu")
STATES = ("submitted", "admitted", "running", "preserving", "done", "failed", "cancelled")
CANCEL_ACTORS = ("deadman", "timeout", "guardian", "manual")
FAILURE_CLASSES = ("SUCCESS", "WORKLOAD_CHECK_FAILED", "CUDA_ALLOC_FAIL", "STORAGE_EXHAUSTED", "STAGE_TIMEOUT",
                   "CANCELLED_DEADMAN", "CANCELLED_GUARDIAN", "CANCELLED_MANUAL", "CGROUP_OOM_ATTRIBUTED",
                   "OOM_UNRESOLVED_ORIGIN", "UNKNOWN_SIGNAL", "UNKNOWN_EXIT")
