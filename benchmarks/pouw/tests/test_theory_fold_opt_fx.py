"""fold_opt_fx.py, run as written: every check passes and every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "fold_opt_fx.py"


@pytest.mark.slow
def test_fold_opt_fx(tmp_path):
    """FOLD-OPT for the fx chains: fx1 (resp. fx2) needs >= 4 (resp. 5) issue slots a word and no cheaper fold on
    the binding joint-issue row, so mu_f = 0. The claims are computed (single-instruction and width-4 exhaustive
    searches over the modelled sm_120 ISA), so only running the script checks them. About 2 minutes, nearly all in the
    width-4 searches that neither output is a word of a two-instruction program, IMAD.WIDE's two words included, and
    that no two-instruction program writes both: the count argument needs them, and no cheaper check covers them. The
    search over every two-word form (`--wide3`, fold_wide3.c) is CPU minutes and runs outside the suite."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"] and doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] == 6
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 35
