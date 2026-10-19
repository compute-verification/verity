"""isa_escape_fp.py, run as written: per device (sm_120, H200, B200), no tensor form is open to an elementwise route for
an FP8, FP4 or int8-on-FP word, the IMMA and H200 FP4 ties are found, k3/k3h/k4/k9 hold on every operand pair, premise
(a) holds on the Lean models' replays and fails just outside it, and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "isa_escape_fp.py"


def test_isa_escape_fp(tmp_path):
    """The verdicts, the constructions and the premise replays are claims only running the script checks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 6
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 70
