"""`tools/verity/check/queue.toml`: the merge queue's admission rules parse and name the hold label; no change waits on a
per-head grant.  `drive.toml`, the landing flags of record, parses as the verified queue reads it at `main`."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, DriveFlags, Rules

ROOT = Path(__file__).resolve().parents[4]


def test_the_rules_parse_and_name_the_hold_label():
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.hold_label == "hold"


def test_the_ready_mark_cutoff_is_set_once_in_the_rules():
    """The verified queue takes `marked_after` from the rules when drive.toml sets none, so it and every VM's queue admit the
    same marks only while the cutoff is set in the rules alone."""
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.marked_after
    assert DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).marked_after is None, \
        f"{DRIVE_FLAGS} sets marked_after: set the cutoff in tools/verity/check/queue.toml alone"


def test_the_flags_parse_and_name_the_machines_the_run_beside_launches_on():
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    assert flags.on


def test_the_gates_are_in_the_tree_and_the_markdown_caps_only_warn():
    """The queue runs each from `main`, and a gate that isn't there can't judge, which leaves every PR out.  A Markdown file
    over its cap is a warning: the caps' gate is a warn gate, never a landing gate."""
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    caps = "tools/verity/check/gates/markdown_caps.py"
    assert caps in flags.warn_gates and caps not in flags.landing_gates
    assert all((ROOT / g).is_file() for g in flags.landing_gates + flags.warn_gates)
