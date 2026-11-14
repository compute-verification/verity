"""benchmarks/pouw/pearlplus_lean's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads,
declared once here, where a test's new read goes."""

from __future__ import annotations

import os

RECORDED = os.environ.get("POUW_PEARLPLUS_RECORDED", "")
"""`POUW_PEARLPLUS_RECORDED`: a directory holding a recorded H200 run's retained passes and the verdicts the Python recorded
on them (`passes/` and `verdicts/{run,control}.json`), which test_pearlplus_lean's `test_a_recorded_h200_run` holds the
Lean's verdict to, each Lean replay taking about 45 minutes on a CPU node; without it that test skips."""
