"""`tools/check/queue.toml`: the merge queue's admission rules parse, name only roles the vocabulary knows, and ask for the
grants the research coordinator applies by hand."""
from __future__ import annotations

from pathlib import Path

from research.merge import rule_touches
from research.queue import Rules
from research.store import vocab

ROOT = Path(__file__).resolve().parents[3]
RULES = Rules.parse((ROOT / "tools" / "check" / "queue.toml").read_text(), "tools/check/queue.toml")


def _roles(*changed: str) -> set[str]:
    return {r.role for r in RULES.grants if r.paths and rule_touches(list(r.paths), list(changed))}


def test_the_rules_parse_and_name_known_roles():
    assert {r.role for r in RULES.grants} <= set(vocab.GRANT_ROLES) and RULES.hold_label == "hold"
    assert not any(r.lean_audit for r in RULES.grants)      # a changed guarantee alerts Daniel and waits on nobody


def test_the_rules_ask_for_the_grants_applied_by_hand():
    assert _roles("integrations/vllm/verity_vllm/program/x.py") == {"vllm-coordinator"}
    assert _roles("backends/flock/verifier/lean/Flock/Tags.lean") == {"red-team"}
    assert _roles("backends/flock/README.md", "integrations/vllm/README.md") == set()
    assert _roles("backends/flock/python/tests/test_x.py") == set()                 # tests alone change no statement
    assert _roles("packages/verity/verity/ir/cut.py", "tools/research/src/research/queue.py") == set()
