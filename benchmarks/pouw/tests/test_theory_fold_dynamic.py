"""fold_dynamic.py's quick tier, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_dynamic.py"


@pytest.mark.slow
def test_fold_dynamic_quick(tmp_path):
    """FOLD/DYNAMIC's claims are computed: that no arxm member's step is a program of 1 instruction at width 8 or 2 at
    width 6, single and two-step and at zero data; the families' symmetry counts the counting bound reads; and the decomp
    ranks. Only running the superoptimizer checks them, with its positive controls (rotadd at 1, the 2-instruction
    families at 2, degenerate members shorter) and the LEA.HI mutation. About 2 minutes on 4 cores, nearly all in
    the depth-2 searches."""
    verdict = tmp_path / "verdict.json"
    env = {**os.environ, "OUT": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path, env=env)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 2
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 30
