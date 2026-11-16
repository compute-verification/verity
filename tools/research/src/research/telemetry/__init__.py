"""Shared runner and telemetry: stdlib only (Python 3.11+)."""
from __future__ import annotations

SCHEMA = "telemetry/v0.1"
JOB_SCHEMA = "telemetry/job/v0.1"
STATUS_SCHEMA = "telemetry/status/v0.1"
EVENTS_SCHEMA = "telemetry/events/v0.1"
RESOURCES_SCHEMA = "telemetry/resources/v0.1"
FAILURE_SCHEMA = "telemetry/failure/v0.1"
COLLECTOR_VERSION = "0.1.3"
CLASSIFIER_VERSION = "0.1.4"

HOLD_REASONS = ("dependency", "gpu_reservation", "host_memory_reservation", "release", "manual", "other")
SCOPES = ("process", "job_cgroup", "shared_pod_cgroup", "host", "gpu")
STATES = ("submitted", "admitted", "running", "preserving", "done", "failed", "cancelled")
CANCEL_ACTORS = ("deadman", "timeout", "guardian", "manual")
FAILURE_CLASSES = ("SUCCESS", "WORKLOAD_CHECK_FAILED", "CUDA_ALLOC_FAIL", "STORAGE_EXHAUSTED", "STAGE_TIMEOUT",
                   "CANCELLED_DEADMAN", "CANCELLED_GUARDIAN", "CANCELLED_MANUAL", "CGROUP_OOM_ATTRIBUTED",
                   "OOM_UNRESOLVED_ORIGIN", "UNKNOWN_SIGNAL", "UNKNOWN_EXIT")


def end_state(rc: int, timed_out: bool = False) -> str:
    return "cancelled" if timed_out else ("done" if rc == 0 else "failed")
