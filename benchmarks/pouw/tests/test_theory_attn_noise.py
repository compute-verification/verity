"""attn_noise.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "attn_noise.py"


@pytest.mark.slow
def test_attn_noise(tmp_path):
    """Adopted P·V's and cohalf QKᵀ's F₂ cross ranks at B3's instances (zero, P = 0, V = 0, one-hot and
    uniform P, random), B3's two premises collapsing (one-sided noise, row dither), the grind's h, the in-proof noise
    compressions a unit and the grind bound a seed; about 8 s, the cross ranks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          check=False, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 2
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 9
