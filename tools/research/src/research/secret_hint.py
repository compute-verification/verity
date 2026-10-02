"""A Cursor secret this VM injects but this shell lacks: a long-lived shell on a cloud VM can lose them while a fresh tmux login
shell still has them, so a command that needs one says so instead of failing as if it were never configured."""
from __future__ import annotations

import os

FRESH_SHELL = "rerun from a fresh tmux login shell"


def lost(*names: str) -> str:
    """'' unless some of `names` are in $CLOUD_AGENT_INJECTED_SECRET_NAMES (a Cursor cloud VM's list) and unset here; then the hint."""
    injected = set(os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(","))
    gone = [n for n in names if n in injected and not os.environ.get(n)]
    return f"{', '.join(gone)} not in this shell though this VM injects {'it' if len(gone) == 1 else 'them'}: {FRESH_SHELL}" if gone else ""
