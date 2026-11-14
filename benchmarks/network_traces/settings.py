"""The environment variables `served_session.py` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong
here)."""
from __future__ import annotations

import os


def host(name: str) -> str | None:
    """The address the variable `name` holds, or None while it is unset or empty. ``site.json``'s gate names one,
    ``VY_CHALLENGER_1_HOST`` (the challenger host's), which no file here holds."""
    return os.environ.get(name) or None
