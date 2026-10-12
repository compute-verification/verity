"""`tools/verity/check/queue.toml`: the merge queue's admission rules parse and name the hold label; no change waits on a
per-head grant (Daniel, 8 Oct)."""
from __future__ import annotations

from pathlib import Path

from research.queue import Rules

ROOT = Path(__file__).resolve().parents[4]


def test_the_rules_parse_and_name_the_hold_label():
    rules = Rules.parse((ROOT / "tools" / "verity" / "check" / "queue.toml").read_text(), "tools/verity/check/queue.toml")
    assert rules.hold_label == "hold"
