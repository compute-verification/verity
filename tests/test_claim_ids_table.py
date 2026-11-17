"""The claim ids' table (`verity/commitments/claim_ids.json`) is the one `tools/verity/lean/claim_ids.py` writes from the
locks' premise entries."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repo_claim_ids_under_test", ROOT / "tools/verity/lean/claim_ids.py")
CI = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = CI
spec.loader.exec_module(CI)


def test_the_claim_ids_table_is_the_one_the_locks_give():
    ids, fails = CI.carried(ROOT)
    rows = json.loads(CI.TABLE.read_text(encoding="utf-8"))
    fails += CI.declared_failures(rows)
    assert not fails, "\n".join(fails)
    assert CI.TABLE.read_text(encoding="utf-8") == CI.text(CI.table(rows, ids)), (
        f"{CI.TABLE.relative_to(ROOT)} differs from the ids the locks carry: run `{CI.WRITE}` and commit it")
