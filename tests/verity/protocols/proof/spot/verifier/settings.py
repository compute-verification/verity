"""verity/protocols/proof/spot/verifier's tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a
test reads, declared once here, where a test's new read goes."""

from __future__ import annotations

import os


def flock_circuit() -> str:
    """``FLOCK_CIRCUIT``: a build of `flock-circuit`, which `test_one_circuit_chain`'s prove stage runs; else empty."""
    return os.environ.get("FLOCK_CIRCUIT", "")
