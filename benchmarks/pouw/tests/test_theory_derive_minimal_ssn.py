"""derive_minimal_ssn.py, run as written: at every coin bit p ≠ 24 each sm_90 integer family is refuted for ssn-1l's
noise register, p = 24's one LOP3 is the positive control, and every family's own function survives its certificate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "derive_minimal_ssn.py"


def test_derive_minimal_ssn(tmp_path):
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 12
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] == 3 + 32 * 13
