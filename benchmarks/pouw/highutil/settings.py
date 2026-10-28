"""Every environment variable HIGHUTIL's period reads or sets, declared once (tests/lint/test_repo_env_reads.py). Stdlib only.

    OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS   set to 1 unless given: one BLAS thread a pool worker
    V31_LTW          v31.cu's own knob (run.sh's default 1), read here only for the driver's ready line
    GPU_LEASE_UUID   gpu-lease's GPU, for the clock poller's nvidia-smi -i
    HU_CLOCK         0: no clock poller (run.sh's nsys runs, where nsys would trace it too)
"""
from __future__ import annotations

import os

BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def one_blas_thread() -> None:
    for v in BLAS_THREADS:
        os.environ.setdefault(v, "1")


def ltw() -> str:
    return os.environ.get("V31_LTW", "0")


def lease_uuid() -> str | None:
    return os.environ.get("GPU_LEASE_UUID") or None


def poll_clock() -> bool:
    return os.environ.get("HU_CLOCK") != "0"
