"""scalar_route.py, run as written: the class-P escape shapes are enumerated and each is excluded below 2.03 issue slots
without tables (joint packing overflows past 2 rows, hashing is not the committed word, count-coefficient and bit-sliced are
measured above 2.03), and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "scalar_route.py"


def test_scalar_route(tmp_path):
    """The escape-shape search and its checkable arithmetic (concrete 2-row vs 3-row SWAR overflow, hash != committed word,
    the no-table floor cross-check): only running it checks the claims, and the mutations confirm a cheaper packing or a
    single-input word would be caught."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 3
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 16
