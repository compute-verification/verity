"""isa_escape_joint.py, run as written: Lemma J's slot-step identity holds at full depth on sm_120, H200 and B200, FPX's
19/9/21 open joint pairs get the recorded TIE / TIE-ON-M verdicts, premise (M) is graded per shallower shape, and every
mutation of a rate, a shape or (M) is caught. Its copies of other files' facts are held to them here."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1] / "theory"
SCRIPT = THEORY / "isa_escape_joint.py"


def _script():
    sys.path.insert(0, str(THEORY))
    try:
        spec = importlib.util.spec_from_file_location("isa_escape_joint", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(THEORY))


def test_isa_escape_joint(tmp_path):
    """The per-device verdicts and the mutations are claims only running the script checks."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 8
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 35


def test_measured_rows_are_m1s():
    """The (M) rows the script reads are tensor_kinds' forms, and the sm_120 forms M1 timed (KIND's first 19) include each
    measured shallower shape's SASS form."""
    s = _script()
    spec = importlib.util.spec_from_file_location("tensor_kinds", THEORY.parent / "tensor_kinds" / "tensor_kinds.py")
    tk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tk)
    sm120 = list(tk.KIND)[:list(tk.KIND).index("omma_nvf4_m16n8k64") + 1]
    timed = {"IMMA.16816": "imma_s8_m16n8k16", "IMMA.SP.16832": "imma_sp_s8_m16n8k32", "HMMA.1684.F32.TF32":
             "hmma_tf32_m16n8k4"}
    for sass, _, _ in s.MEASURED["sm_120"].values():
        assert timed[sass.split(",")[0]] in sm120, sass
    assert "hmma_f16_m16n8k8_f32" not in sm120 and "qmma_e4m3_m16n8k16_f32" not in sm120
