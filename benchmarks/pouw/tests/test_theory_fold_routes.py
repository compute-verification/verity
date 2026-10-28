"""fold_routes.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_routes.py"


def test_fold_routes(tmp_path):
    """The battery: the brief's fold routes on sm_120 (fx1), the H200 (li) and the B200 (li heavy), each bit-exact
    against `segment`'s reference fold or refuted, priced at its device's certificate, with the superoptimizer's
    single- and two-instruction searches. The claims are computed, so only running the script checks them; about 1 s."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"]
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 27
