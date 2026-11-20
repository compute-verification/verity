"""The one-stage door's settings registry (`tests/lint/test_repo_env_reads.py`): each variable the door reads, declared once
here."""

from __future__ import annotations

import os
from collections.abc import Mapping

#: the environment variable that, at ``1``, puts the door on its templated statement; unset, it stays on the served one
TMPL_SWITCH = "VERITY_DOOR_TMPL"


def door_tmpl() -> bool:
    """Whether `TMPL_SWITCH` is ``1``, read at each call."""
    return os.environ.get(TMPL_SWITCH) == "1"


#: the Lean runtime's switch: set, a panic (an index past the end among them) ends the process with exit 1 where it would
#: otherwise print its line, return a default value and carry on
ABORT_ON_PANIC = "LEAN_ABORT_ON_PANIC"


def lean_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a Lean executable whose answer can become a verdict starts in: ``environ`` (this process's when
    None) with `ABORT_ON_PANIC` set, whatever ``environ`` holds."""
    return {**(os.environ if environ is None else environ), ABORT_ON_PANIC: "1"}
