"""The PoUS benchmarks' tests' settings registry (`tests/lint/test_repo_env_reads.py`): each variable a test reads,
declared once here, where a test's new read goes."""

from __future__ import annotations

import os


def smoke_out() -> str | None:
    """`SMOKE_OUT`, a file the honest smoke test appends its line to (the verdict, the protocol and the executable's
    digest) for a script that runs it; unset, the line goes nowhere."""
    return os.environ.get("SMOKE_OUT")
