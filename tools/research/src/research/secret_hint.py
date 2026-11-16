from __future__ import annotations

import os
import re
import subprocess

FRESH_SHELL = "rerun from a fresh tmux login shell"
RECOVERED: list[str] = []
KEPT_SESSION = "vm-secrets"
START_KEPT = f"tmux new-session -d -s {KEPT_SESSION}"


def _injected() -> list[str]:
    return [n for n in os.environ.get("CLOUD_AGENT_INJECTED_SECRET_NAMES", "").split(",") if n]


def _tmux(*args: str) -> subprocess.CompletedProcess | None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    try:
        return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def recover() -> list[str]:
    got = []
    for n in (n for n in _injected() if not os.environ.get(n)):
        if (p := _tmux("show-environment", "-g", n)) is None:
            break
        if p.returncode == 0 and p.stdout.startswith(f"{n}=") and (v := p.stdout[len(n) + 1:].removesuffix("\n")):
            os.environ[n] = v
            got.append(n)
    RECOVERED.extend(got)
    return got


def server() -> tuple[list[str] | None, bool] | None:
    names = _injected()
    if not names or (held := _tmux("show-environment", "-g")) is None:
        return None
    if held.returncode:
        return None, False
    kept = _tmux("has-session", "-t", f"={KEPT_SESSION}")
    return [n for n in names if re.search(rf"^{re.escape(n)}=.", held.stdout, re.M)], kept is not None and kept.returncode == 0


def lost(*names: str) -> str:
    injected = set(_injected())
    gone = [n for n in names if n in injected and not os.environ.get(n)]
    return f"{', '.join(gone)} not in this shell though this VM injects {'it' if len(gone) == 1 else 'them'}: {FRESH_SHELL}" if gone else ""
