"""The environment variables `minutes.py` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os
from pathlib import Path


def run_dir() -> Path | None:
    """The `research run` this one is part of: its directory (``RESEARCH_RUN_DIR``), or None outside one."""
    return Path(d) if (d := os.environ.get("RESEARCH_RUN_DIR")) else None


def run_id() -> str | None:
    """That run's id (``RESEARCH_RUN_ID``)."""
    return os.environ.get("RESEARCH_RUN_ID") or None


def runs_root() -> Path:
    """Where `research fetch` copies runs' records: ``RESEARCH_RUNS``, else ``~/.research/runs``."""
    return Path(os.environ.get("RESEARCH_RUNS") or Path.home() / ".research" / "runs").expanduser()
