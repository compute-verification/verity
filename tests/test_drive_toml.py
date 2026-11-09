"""drive.toml parses strictly with this tree's research: a key it doesn't read (a typo, or a key landed before the code that
reads it) fails here, since every reader ignores such keys.  The low-risk lane keeps off every path that runs with the control
pod's or every agent's credentials once it is on `main`."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags, lowrisk

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALED = ("infra/control/post_start.sh", "tools/research/src/research/landing.py", "tools/agent-guard/hook.sh",
                ".cursor/hooks.json", "pyproject.toml", "uv.lock")


def test_drive_toml_parses_strictly():
    DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS, strict=True)


def test_the_low_risk_lane_lands_nothing_that_runs_with_credentials():
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS, strict=True)
    assert not (open_ := [p for p in CREDENTIALED if lowrisk([p], flags.lowrisk_excludes)]), \
        f"{DRIVE_FLAGS}'s lowrisk_excludes lets {open_} land on a ready mark alone"
