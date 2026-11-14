"""`verity.claims`' table of each claim id's Lean premises (`LEAN`) is the Lean locks' `ids`, as
`tools/verity/lean/claim_ids.py` writes it. Read as text: `verity.claims` is a trusted place no root test imports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_claims_table_of_lean_premises_is_the_locks_ids() -> None:
    spec = importlib.util.spec_from_file_location("claim_ids", ROOT / "tools" / "verity" / "lean" / "claim_ids.py")
    claim_ids = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(claim_ids)
    text = claim_ids.CLAIMS.read_text()
    assert claim_ids.written(text, claim_ids.table(ROOT)) == text, (
        "verity/claims/__init__.py's LEAN differs from the Lean locks' `ids`: run "
        "`python3 tools/verity/lean/claim_ids.py --write` and commit it")
