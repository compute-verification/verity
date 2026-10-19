"""nosum.py, run as written: every non-issue unit is enumerated and classified, the exactness/freshness physics holds,
and the machine model's floors price every side route >= 16 MACs/word; every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "nosum.py"


def test_nosum(tmp_path):
    """The NOSUM classification and its checkable physics (texture convex hull, ROP saturation, reducer sum, mover copy,
    the uniform-datapath exception) plus the floors cross-check: only running it checks the claims."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 3
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 19
