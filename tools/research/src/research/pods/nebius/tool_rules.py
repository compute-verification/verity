#!/usr/bin/env python3
"""What a shipped research package (the run recorder `research run --on` sends a machine) must do before the machine runs it.

research.remote's `launch-request` asks the machine's deployed copy of this file (deploy.toml installs it at
/usr/local/lib/vy-monitoring/tool_rules.py) about the package answering the request, and the launcher asks its own copy about a
--source tree's package before shipping it, so a lane's branch older than a rule is held to the rule the machine has. A package
older than that consult is caught only by a launcher that has it. Stdlib only: deploy.toml installs this file alone.
"""
from __future__ import annotations

from pathlib import Path

#: (file of the package, a string the fixed file has, what a package without it does, the change that fixed it). A package
#: without the file at all predates what the rule fixes.
RULES = (
    ("telemetry/pyspy.py", "ARGV_LINE", "its stall dumps print each process's whole command line, and custody publishes "
     "logs/stall-*.txt with the run record", "#7656"),
)


def refusal(pkg: Path | str) -> str | None:
    """Why the machine must not run the research package at `pkg`, or None."""
    for rel, marker, does, fix in RULES:
        f = Path(pkg) / rel
        if not f.exists():
            continue
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError) as e:
            return f"the research package at {pkg} can't be checked ({f}: {type(e).__name__}: {e}); nothing started"
        if marker not in text:
            return (f"the research package at {pkg} predates {fix} ({rel} has no {marker}): {does}. Merge origin/main into the "
                    f"branch and launch again; nothing started")
    return None
