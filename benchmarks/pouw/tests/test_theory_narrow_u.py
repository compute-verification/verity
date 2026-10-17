"""narrow_u.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "narrow_u.py"


@pytest.mark.slow
def test_narrow_u(tmp_path):
    """co32c and Dither-C's cohalf Q Kᵀ with U on ±8, ±16, ±32 (the bench's co32c-uN and co32c-u16-cohalf arms):
    ranges, the served identity, freshness at the zero instance, F₂ cross ranks, the TS model's bit-sliced routes and
    the byte-field table; about 17 s, the cross ranks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 8
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 34
