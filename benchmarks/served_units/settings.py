"""window.py's settings: every environment variable it reads, declared once in `DECLARED` with its default and what it does,
and read only through `get`. A default of None is required (unset, `get` stops the job naming it); an empty default is
optional. Stdlib only: window.py runs under the node's own python3, outside the workspace."""
from __future__ import annotations

import os

DECLARED: dict[str, tuple[str | None, str]] = {
    "RESEARCH_RUN_DIR": ("/tmp/served-units-window", "the research run's directory (research's runner sets it): records go "
                                                     "under its out/"),
    "PATH": (None, "the groups' PATH, with the tree's gpu_lease.sh as gpu-lease where none is installed"),
    "FLOCK_WORK": (None, "where statements are staged and proved: under /workspace/tmp/ or /dev/shm/"),
    "WIN_CPUS": (None, "the most cores the job's affinity may hold"),
    "WIN_NICE": ("0", "the least niceness the job may run at"),
    "WIN_MEM_GB": ("", "host memory the job may hold, GB (unset: stage 600, prove 268)"),
    "WIN_DISK_GB": (None, "stage: the most FLOCK_WORK may hold, GB"),
    "WIN_FREE_GB": ("50", "stage: GB left free on FLOCK_WORK's filesystem"),
    "WIN_UNTIL": (None, "stage: the UTC time every staging is done by, e.g. 2026-11-10T01:30Z"),
    "WIN_SLOW": ("1.3", "stage: a staging's estimated seconds times this must end before WIN_UNTIL"),
    "WIN_BACKFILL_GB": ("8", "stage: once one waits for memory, only stagings of at most this many GB start ahead of it"),
    "WIN_BACKFILL_DISK_GB": ("0", "stage: once one waits for disk, only stagings of at most this many GB start ahead of it"),
    "WIN_MEM_PEER": ("", "stage: another job's pid, whose process tree's RSS counts against WIN_MEM_GB too"),
    "WIN_DEVICE": ("gpu", "gpu, or cpu to build and prove on the CPU (a rehearsal off the GPU node)"),
    "SM": ("", "the GPUs' compute capability the prover is built for (90 for H200, 120 for RTX PRO 6000); required on gpu"),
    "WIN_GPUS": ("8", "prove: the groups, one GPU each"),
    "WIN_DEV_GIB": ("45", "prove: device memory a group's prover may hold, GiB"),
    "WIN_ALSO": ("", "prove: colon-separated FLOCK_WORKs whose staged statements are proved too"),
    "WIN_SECONDS": ("", "prove: an earlier prove's out/seconds.tsv, whose seconds balance the groups"),
    "WIN_UNMEASURED_S": ("900", "prove: the estimate of a statement WIN_SECONDS doesn't hold finished, at least"),
    "WIN_KEEP_ALSO": ("", "prove: colon-separated globs under a statement's directory kept beside window.py's KEEP"),
    "WIN_KEEP": ("0", "prove: 1 keeps FLOCK_WORK when the job ends"),
    "WARM": ("1", "prove: prove_time.sh's warm-up runs a statement"),
    "RUNS": ("1", "prove: prove_time.sh's timed runs a statement"),
    "VCPUS": ("2", "prove: prove_time.sh's cores a session"),
    "LEASE_MIN": ("30", "prove: a group's gpu-lease, minutes"),
    "GPU_LEASE_WHO": ("servedcard", "prove: who the groups' GPU leases name"),
}


def get(name: str) -> str:
    if name not in DECLARED:
        raise KeyError(f"{name} is not declared in {__file__}")
    default, what = DECLARED[name]
    v = os.environ.get(name, default)
    if v is None:
        raise SystemExit(f"window.py: set {name}, {what}")
    return v
