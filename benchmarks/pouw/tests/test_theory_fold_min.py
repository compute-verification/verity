"""fold_min.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_min.py"


def test_fold_min(tmp_path):
    """The battery: the IMAD-pipe program for a keyed round's constant XOR, bit-exact against the reference
    keyed round at random and at li's keys, priced at every headline certificate, and the B200 sheet's cheater LP
    re-solved with it open. The claims are computed, so only running the script checks them; about 2 s."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"]
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 14
