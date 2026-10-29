"""benchmarks/pouw's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared once
here, where a test's new read goes."""

from __future__ import annotations

import os

V31_LIB = os.environ.get("VERITY_POUW_V31_LIB", "")
"""`VERITY_POUW_V31_LIB`: a libv31.so (accel_v31/v31lib.py's `build_lib`) built for the GPU in hand, which
test_accel_v31's GPU test runs on; without it that test skips."""
