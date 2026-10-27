"""`tools/verity/check/queue.toml`: the merge queue's admission rules parse and name the hold label; no change waits on a
per-head grant.  `drive.toml`, drive's flags of record, parses as `research queue drive --config` reads it."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags, Rules

ROOT = Path(__file__).resolve().parents[4]


def test_the_rules_parse_and_name_the_hold_label():
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.hold_label == "hold"


def test_drives_flags_parse_and_put_no_pr_first_by_default():
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    assert flags.on and flags.first == ()


def test_drives_landing_gates_are_in_the_tree():
    """Drive runs each from `main`, and a gate that isn't there can't judge, which leaves every PR off `next`."""
    gates = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).landing_gates
    assert "tools/verity/check/gates/markdown_caps.py" in gates and all((ROOT / g).is_file() for g in gates)
