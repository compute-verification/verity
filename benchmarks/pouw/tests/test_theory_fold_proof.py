"""fold_proof.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_proof.py"


def test_fold_proof(tmp_path):
    """FOLD-PROOF's battery: FOLD-OPT at the sm_120 sheet's certificate as a class statement read from
    `Headline/Sm120.lean` and `TS/Sheet.lean`, NO-MIX at width 4 over every two-word form, the records' merge's
    predicate, the counting from FOLD-MIN-4, and the whole-block model that puts WIDTH-32 under the floor. The claims
    are computed, so only running the script checks them; about 1 s."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 6
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 29
