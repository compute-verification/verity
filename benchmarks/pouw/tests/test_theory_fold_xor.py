"""fold_xor.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_xor.py"


def test_fold_xor(tmp_path):
    """FOLD-XOR on the B200 sheet: a 78-instruction IMAD/IMAD.HI program is li's x ^ y (^ k) bit-exactly, and so is its
    SASS loop body with registers allocated, under the plain cheater's break-even (92.66) and over the lucky one's
    (62.54), so the design's 'no IMAD- or FP-pipe program under 93' is false while the 0.7052% headline stands. The
    claims are computed (the bit-exact runs, the re-solves of the cheater LP and the restricted mod-8 combine search),
    so only running the script checks them; about 16 s. Thirteen checks on a tree without #2706's design and fifteen
    with it (the rebuild and the skip constant checked against it)."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 5
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 13
