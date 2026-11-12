"""prof_summary.py's table of parts on a made-up bench.jsonl: each part's difference of arms, their sum, and an arm
the bench didn't time shown as nan."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prof_summary", HERE.parent / "prof_summary.py")
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P
_spec.loader.exec_module(P)

MS = {"plain": 1.0, "cplain": 1.1, "cduo": 1.3, "tK": 1.2, "formA": 0.1, "bsp": 0.2, "genU": 0.05, "td_bfus": 1.75}


def test_the_parts_are_differences_of_arms(tmp_path, capsys):
    row = {"shape": "o_proj", "m": 8192, "n": 4096, "k": 4096, "gpu": "a GPU", "lt_algo": [7],
           "arms": {k: {"ms": v, "mhz": 2000.0} for k, v in MS.items()}}
    (tmp_path / "bench.jsonl").write_text(json.dumps(row) + "\n")
    P.main(str(tmp_path))
    out = capsys.readouterr().out.splitlines()
    assert "shape o_proj m 8192 n 4096 k 4096, a GPU, plain cuBLASLt 1.0000 ms (algo 7)" in out
    assert "| cduo | 1.3000 | 1.300 | 2000 |" in out
    assert "**td_bfus 1.7500 ms = 1.750 x plain**; parts (each timed alone, hot):" in out
    assert "| custom mainloop, no receipts, against cuBLASLt (cplain - plain) | 0.1000 | 0.100 |" in out
    assert "| duo receipts at L = 32 in the mainloop (cduo - cplain) | 0.2000 | 0.200 |" in out
    assert "| the peel as a tail stage (tK - cplain) | 0.1000 | 0.100 |" in out
    assert "| F's expansion alone (genF) | nan | nan |" in out
    assert "| sum of mainloop + receipts + tail + formA + bsp (over plain) | 0.7000 | 0.700 |" in out
    assert "| td_bfus - plain | 0.7500 | 0.750 |" in out
    assert out[-2:] == ["## ncu (one call, --clock-control none)", ""]
