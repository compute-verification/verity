"""fold_super.py, run as written: every FOLD-OPT/FILL-OPT across-words break attempt closes (no program reproduces a fold
output below honest's lane-instruction count), and every mutation -- a fold that genuinely does have a shortcut -- is
caught. The script's fold models are held to receipts/receipt_folds.py byte for byte."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

THEORY = Path(__file__).resolve().parents[1] / "theory"
RECEIPTS = Path(__file__).resolve().parents[1] / "receipts"
EXACT_KIT = Path(__file__).resolve().parents[1] / "exact_kit"
SCRIPT = THEORY / "fold_super.py"


def _import_fold_super():
    for p in (str(THEORY), str(RECEIPTS), str(EXACT_KIT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fold_super as F
    import receipt_folds as R
    return F, R


def test_fold_models_match_receipts():
    """fold_super's li, fx1 and xin13 steps are the ones in receipt_folds.py (the reference of record)."""
    F, R = _import_fold_super()
    rng = F.lcg(0x1234)
    a0, a1 = F.pick_keys_ok_r9(2, 0xA11CE)
    ax = F.FO.pick_key_ok_r9() | 1
    k0, kx = 0x12345679, 0x5A5AC3C3
    for _ in range(300):
        y, z, d0, d1 = next(rng), next(rng), next(rng), next(rng)
        # duo word
        assert F.duo_word(y, z, d0, k0, a0, 5) == R.duo_word(y, z, d0, k0, a0, 5)
        # li half-step = one limb of receipt_folds.li_step
        assert F.li_half(y, d0, d1, k0, a0) == F.u32((y ^ d0 ^ k0) * a0 + d1)
        # fx1: the keyed IMAD then duo
        t = F.u32(d0 * ax)
        assert F.fx1_word(y, z, d0, kx, ax, k0, a0, 5) == R.duo_word(y, z, t, kx, a0, 5)
        # xin13 half-step = one limb of receipt_folds.xin13_step
        assert F.xin13_half(y, d0, d1, a0, 13) == F.u32((R.rotl(y, 13) ^ d0) * a0 + d1)


@pytest.mark.slow
def test_fold_super(tmp_path):
    """The across-words attack battery and the superopt hunt: every break attempt closes and every mutation is caught."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 4
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 30
