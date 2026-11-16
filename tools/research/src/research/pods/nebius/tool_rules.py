#!/usr/bin/env python3
"""What a shipped research package must do before the machine runs it."""
from __future__ import annotations

from pathlib import Path

RULES = (
    ("telemetry/pyspy.py", "ARGV_LINE", "its stall dumps print each process's whole command line, and custody publishes "
     "logs/stall-*.txt with the run record", "#7656"),
)


def refusal(pkg: Path | str) -> str | None:
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
