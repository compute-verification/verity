"""isa_escape.py, run as written: every committed-word candidate the red team names (warp/L2/TMA reductions, atomics, the
low-precision, 1-bit and FP64 MMA forms, the dot-product instructions, DPX, SIMD/SAD, texture, copies, packing tricks, the wide
multiply, the uniform datapath beside async tensor) is enumerated and closed, with the bit-exact decomposition and
exactness facts verified and the floors cross-checked; every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "isa_escape.py"


def test_isa_escape(tmp_path):
    """The ISA-escape reconciliation and its checkable arithmetic (reducers have no multiply, a cross-lane reduce as a
    fold costs 32 lane-instructions/word > duo's 3, SAD != dot product, the 64-plane b1 and 4-nibble s4 int8
    reconstructions, f16/bf16/f64 exactness vs E4M3/E3M2/E2M3/E2M1, every tensor form's int8 TOPS <= IMMA's, the
    floors): only running it checks the claims."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 4
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 50
