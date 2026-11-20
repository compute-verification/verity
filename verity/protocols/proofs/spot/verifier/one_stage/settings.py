"""The one-stage door's settings registry (`tests/lint/test_repo_env_reads.py`): each variable the door reads, declared once
here."""

from __future__ import annotations

import os

#: the environment variable that, at ``1``, puts the door on its templated statement; unset, it stays on the served one
TMPL_SWITCH = "VERITY_DOOR_TMPL"


def door_tmpl() -> bool:
    """Whether `TMPL_SWITCH` is ``1``, read at each call."""
    return os.environ.get(TMPL_SWITCH) == "1"
