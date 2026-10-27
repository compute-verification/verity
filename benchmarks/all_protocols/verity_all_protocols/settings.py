"""The environment variables step 0's table (`deployment`) sets for the legs and tools its parties run, declared once,
each with the default it gets when the caller set none (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os

#: the network certifier's parties relay honest traffic unless the caller asks for a tamper: `legs/certifier.py` reads
#: an unset ``CERTIFIER_TAMPER`` as ``extra``, its benchmark's self-test
CERTIFIER = {"CERTIFIER_TAMPER": ""}
#: the certifier's Rust checks: every core, and the stack their recursion needs
RUST = {"RAYON_NUM_THREADS": str(os.cpu_count() or 1), "RUST_MIN_STACK": str(64 << 20)}


def setdefault(values: dict[str, str]) -> None:
    """Each of ``values`` into this process's environment, for what it starts, unless the caller set it."""
    for name, value in values.items():
        os.environ.setdefault(name, value)
