"""Every environment variable `exact_kit` reads, declared once. Each is read when asked, never at import, and with the
standard library alone: a check runs where neither the harness nor `verity_config` is installed.
"""
from __future__ import annotations

import os
from typing import NamedTuple

#: what `exact_kit.checks` reads
__all__ = ["Run", "out_dir", "run"]


class Run(NamedTuple):
    """The `research run` a check runs under: its directory, its id and the commit it ran (None where unset)."""

    dir: str
    id: str
    source_sha: str | None


def out_dir() -> str | None:
    """``OUT``, the run's output directory, where a verdict goes when the script names no path; None where unset or
    empty."""
    return os.environ.get("OUT") or None


def run() -> Run | None:
    """``RESEARCH_RUN_DIR``, ``RESEARCH_RUN_ID`` and ``RESEARCH_SOURCE_SHA``, which `research run` sets; None outside a
    run, where either of the first two is unset or empty."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return None
    return Run(run_dir, run_id, os.environ.get("RESEARCH_SOURCE_SHA"))
