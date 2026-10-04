"""A Cursor secret this VM injects but this shell lacks: a long-lived shell on a cloud VM can lose them while a fresh tmux login
shell still has them, so `research` first takes them from the tmux server, where that shell gets them (`recover`), and a command
that still lacks one says so instead of failing as if it were never configured (`lost`)."""
from __future__ import annotations

import os
import subprocess

FRESH_SHELL = "rerun from a fresh tmux login shell"


def _injected() -> list[str]:
    return [n for n in os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(",") if n]


def recover() -> list[str]:
    """Set each secret in $CLOUD_AGENT_INJECTED_SECRET_NAMES that is unset here from the tmux server's global environment, and
    return the names set. A laptop, a pod or a node has no such list and runs nothing; a value is never printed. Nor does a
    process under a test (`$PYTEST_CURRENT_TEST`, which a test's subprocesses inherit): a secret a test unset is its setup."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return []
    got = []
    for n in (n for n in _injected() if not os.environ.get(n)):
        try:
            p = subprocess.run(["tmux", "show-environment", "-g", n], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            break
        # `NAME=value\n`; `-NAME` when tmux holds it removed, and an error when no server runs
        if p.returncode == 0 and p.stdout.startswith(f"{n}=") and (v := p.stdout[len(n) + 1:].removesuffix("\n")):
            os.environ[n] = v
            got.append(n)
    return got


def lost(*names: str) -> str:
    """'' unless some of `names` are in $CLOUD_AGENT_INJECTED_SECRET_NAMES (a Cursor cloud VM's list) and unset here; then the hint."""
    injected = set(_injected())
    gone = [n for n in names if n in injected and not os.environ.get(n)]
    return f"{', '.join(gone)} not in this shell though this VM injects {'it' if len(gone) == 1 else 'them'}: {FRESH_SHELL}" if gone else ""
