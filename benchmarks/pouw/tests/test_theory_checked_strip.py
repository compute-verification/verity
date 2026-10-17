"""checked_strip.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "checked_strip.py"


@pytest.mark.slow
def test_checked_strip(tmp_path):
    """The script's exact claims about CO-Dither (cross ranks at every instance, SAT tensor ranks, collapses, s32
    identities, priced strategies, the TS-model table): only running it checks them, since each is computed."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 2
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 40
