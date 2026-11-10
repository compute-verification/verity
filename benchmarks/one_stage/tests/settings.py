"""benchmarks/one_stage's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads, declared
once here, where a test's new read goes."""

from __future__ import annotations

import os

TREE_FROM = os.environ.get("VERITY_ONE_STAGE_TREE_FROM", "") == "1"
"""`VERITY_ONE_STAGE_TREE_FROM=1`: test_a5_tree_relay stages RoPE's captured heads and serves them with ``FLOCK_CIRCUIT``'s
``serve --zk --tree-from`` through a trees session's relay; without it, or without the build, that test skips."""

FLOCK_CIRCUIT = os.environ.get("FLOCK_CIRCUIT", "")
"""`FLOCK_CIRCUIT`: a release build of ``flock-circuit`` with the features ``sha512,glue`` (`verity/ml/flock/check_build.sh`'s
checkout and patches, built with ``cargo build --release``)."""
