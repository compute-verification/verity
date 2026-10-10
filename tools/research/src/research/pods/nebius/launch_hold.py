#!/usr/bin/python3 -I
"""vy-research-python: vy-nebius-1's interpreter for `research run --on` (machines.d `python`), installed by deploy.toml.

Every launcher runs `<python> -m research launch-request [--expect-tool ID] REQUEST` on the node with the research package it
shipped, so a launcher whose package predates research.remote's `launch_hold` would start a run during the disk guard's launch hold.
This refuses such a request itself, before the shipped package runs, while /etc/vy/launch-hold exists (pods/nebius/sky/
disk_guard.sh writes it) and none of its `except:` tokens names the run (its id, or a piece of the launcher's agent URL). It
answers as launch-request does when it refuses before the claim: a JSON line with `claimed: false`, and exit 3. A hold file that
can't be read or names no reason refuses too. Anything else runs as `/usr/bin/python3 ARGS...`. Stdlib only: deploy.toml installs
this file alone.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PYTHON = "/usr/bin/python3"
HOLD = Path(os.environ.get("LAUNCH_HOLD_FILE") or "/etc/vy/launch-hold")


def hold_reason(hold: Path, run_id: str, submitted_by: str | None) -> str | None:
    """Why HOLD refuses a new run, or None: its `reason:` line while it exists, unless one of its `except:` lines names the run (its
    id, or a piece of the launcher's agent URL). A file there that can't be read or names no reason refuses too: a guard never fails
    open. research.remote's launcher applies the same rule."""
    try:
        text = hold.read_text()
    except FileNotFoundError:
        return None
    except OSError as e:
        return f"this node holds new runs ({hold} can't be read: {e})"
    lines = [ln.partition(":") for ln in text.splitlines()]
    reason = next((v.strip() for k, _, v in lines if k.strip() == "reason" and v.strip()), None)
    exempt = [t for k, _, v in lines if k.strip() == "except" for t in v.split()]
    if reason and any(t == run_id or (submitted_by and t in submitted_by) for t in exempt):
        return None
    since = next((v.strip() for k, _, v in lines if k.strip() == "since"), None)
    return (f"this node holds new runs{f' since {since}' if since else ''}: {reason or f'{hold} names no reason'}. Send it to another "
            f"machine, or ask infra for an `except:` line in {hold}")


def held(hold: Path, request: Path) -> str | None:
    """hold_reason for the launch request file REQUEST (its run_id and submitted_by)."""
    try:
        req = json.loads(request.read_text())
    except (OSError, ValueError):
        req = {}
    return hold_reason(hold, str(req.get("run_id") or ""), str(req.get("submitted_by") or "") or None)


def main(argv: list[str]) -> int:
    if argv[:3] == ["-m", "research", "launch-request"] and len(argv) > 3:
        if why := held(HOLD, Path(argv[-1])):
            print(json.dumps({"claimed": False, "refused": why, "fault": "LaunchHold"}))
            return 3
    os.execv(PYTHON, [PYTHON, *argv])
    return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
