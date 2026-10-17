"""fold_opt.py, run as written: the FOLD-OPT lower bound holds and every mutation is caught, and the script's duo step
matches receipts/receipt_folds.py byte for byte."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

THEORY = Path(__file__).resolve().parents[1] / "theory"
SCRIPT = THEORY / "fold_opt.py"
RECEIPTS = Path(__file__).resolve().parents[1] / "receipts"


def test_fold_step_matches_receipts():
    """fold_opt's inlined duo word is the one in receipt_folds.py (the reference of record)."""
    sys.path.insert(0, str(THEORY))
    sys.path.insert(0, str(RECEIPTS))
    try:
        import fold_opt as F
        import receipt_folds as R
    finally:
        sys.path.remove(str(THEORY))
        sys.path.remove(str(RECEIPTS))
    rng = F.lcg(0x1234)
    a = F.pick_key_ok_r9()
    for _ in range(200):
        y, z, d, k = next(rng), next(rng), next(rng), next(rng)
        for r in (5, 11, 17, 29):
            ry, rz = R.duo_word(y, z, d, k, a, r)
            assert F.duo_yp(y, z, d, k, a, r) == ry
            assert F.duo_zp(y, z, d, k, a, r) == rz


@pytest.mark.slow
def test_fold_opt(tmp_path):
    """The superoptimisation search and the structural/algebraic lower bound: neither output of the duo word is a single
    sm_120 integer instruction (XOR-then-multiply does not fuse; the width-4 search is exhaustive over all 256 LUTs), so
    the word needs >= 3 instructions; the chain is serial and admits no SWAR or homomorphism. Only running it checks the
    claims, and the mutations confirm the search catches a cheaper fold."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 3
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 15
