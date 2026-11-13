"""`tools/verity/check/queue.toml`: the merge queue's admission rules parse and name the hold label; no change waits on a
per-head grant.  `drive.toml`, drive's flags of record, parses as `research queue drive --config` reads it."""
from __future__ import annotations

from pathlib import Path

from research.queue import DRIVE_FLAGS, IDENTITY, DriveFlags, Rules

ROOT = Path(__file__).resolve().parents[4]


def test_the_rules_parse_and_name_the_hold_label():
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.hold_label == "hold"


def test_the_rules_name_no_author_so_the_queues_commits_here_are_its_own():
    """A repository's rules may name the author and committer of the queue's commits (`Rules.author`); these name none, so
    the queue makes its commits here as `IDENTITY`, the very object, and nothing about them changes."""
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.author is None and rules.identity is IDENTITY
    assert IDENTITY == {"GIT_AUTHOR_NAME": "research queue", "GIT_AUTHOR_EMAIL": "queue@research.invalid",
                        "GIT_COMMITTER_NAME": "research queue", "GIT_COMMITTER_EMAIL": "queue@research.invalid"}


def test_the_ready_mark_cutoff_is_set_once_in_the_rules():
    """Drive's tick takes `marked_after` from the rules when drive.toml sets none, so drive and every VM's queue admit the same
    marks only while the cutoff is set in the rules alone."""
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.marked_after
    assert DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).marked_after is None, \
        f"{DRIVE_FLAGS} sets marked_after: set the cutoff in tools/verity/check/queue.toml alone"


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


def test_a_red_ratchet_gate_step_never_reverts():
    """A red ratchet-gate step is a warn step's: drive rings its culprits' lanes, which fix it forward in their next PR, and
    reverts none of them."""
    assert "ratchet-gate" in DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).warn_steps
