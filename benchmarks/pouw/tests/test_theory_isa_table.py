"""isa_table.py, run as written: the Lean instruction table (`Security/Definitions/Accounting/Isa/`) against DS-120's,
sm_90's and sm_100's censuses, floors.csv and isa_escape.py's candidates, every opcode of the CUDA Binary Utilities'
Hopper and Blackwell lists and of RT-ISA's SASS census placed in a class of its device's table or read as writing no
word, with none left out, and RT-MACHINE-2's triage agreeing; every mutation is caught."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "theory" / "isa_table.py"


def test_isa_table(tmp_path):
    """The Lean table parses to the classes the censuses and floors.csv name, the one traced class is MOV.64, every
    documented opcode and every base the corpus holds is placed or writes no word (an unclassed one fails; ARRIVES and
    F2FP are undocumented and placed), each opcode RT-MACHINE-2 triaged is treated as its class allows, and `class_of`
    and the corpus's word flags disagree only where they're named to: only running it against the Lean source and the
    data checks that they still agree."""
    verdict = tmp_path / "verdict.json"
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--mutations", "--verdict", str(verdict)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    doc = json.loads(verdict.read_text(encoding="utf-8"))
    assert doc["ok"]
    assert doc["counts"]["mutations"] == doc["counts"]["mutations_caught"] >= 15
    assert doc["counts"]["checks"] == doc["counts"]["checks_ok"] >= 29
