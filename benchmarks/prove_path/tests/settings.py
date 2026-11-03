"""benchmarks/prove_path's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared
once here, where a test's new read goes."""

from __future__ import annotations

import os

REMOTETIE = os.environ.get("VERITY_PROVE_PATH_REMOTETIE", "") == "1"
"""`VERITY_PROVE_PATH_REMOTETIE=1`: test_remotetie records, commits and writes its toy's window and runs the Lean verifiers on
it, which takes minutes and Lean's toolchain (`served_cases.build` runs ``lake build``); without it those tests skip."""
