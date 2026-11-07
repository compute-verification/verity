"""`tools/verity/check/queue.toml`: the merge queue's admission rules parse and name the hold label; no change waits on a
per-head grant.  `drive.toml`, drive's flags of record, parses as `research queue drive --config` reads it."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags, Rules

ROOT = Path(__file__).resolve().parents[4]


def test_the_rules_parse_and_name_the_hold_label():
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.hold_label == "hold"


def test_drives_flags_parse_and_name_the_machines_its_checks_run_on():
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    assert flags.on


def test_drives_gates_are_in_the_tree_and_the_markdown_caps_only_warn():
    """Drive runs each from `main`, and a gate that isn't there can't judge, which leaves every PR off `next`.  A Markdown file
    over its cap is a warning: the caps' gate is a warn gate, never a landing gate."""
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    caps = "tools/verity/check/gates/markdown_caps.py"
    assert caps in flags.warn_gates and caps not in flags.landing_gates
    assert all((ROOT / g).is_file() for g in flags.landing_gates + flags.warn_gates)
