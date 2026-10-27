"""h200_step.py: the H200 step at M = 16,384 from `v31 bench`'s JSON lines, and its refusal of an incomplete step."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import h200_step


def _line(n: int, k: int, ms: float, lp: int = 512, t: int = 8, row: int = 1024, m: int = 16384) -> str:
    arms = {a: {"ms": ms, "mhz": 1980 if a == "ovl" else None} for a in h200_step.ARMS}
    return json.dumps({"shape": f"h200-{n}-{k}", "m": m, "n": n, "k": k, "lp": lp, "t": t, "row": row, "S": k // (lp * t),
                       "arms": arms, "gamma_call": 1.0, "gamma_ovl": 1.0, "gamma_gemm": 1.0})


def test_step_counts_each_call_of_52_layers_and_the_head(tmp_path):
    f = tmp_path / "bench-row1024.jsonl"
    f.write_text("\n".join(["noise"] + [_line(n, k, 2.0) for n, k in h200_step.CALLS]
                           + [_line(n, k, 5.0, t=16) for n, k in h200_step.CALLS]) + "\n")
    g = tmp_path / "bench-row1024-wrap4.jsonl"
    g.write_text("\n".join(_line(n, k, 3.0) for n, k in h200_step.CALLS) + "\n")
    st = h200_step.steps(h200_step.rows([str(f), str(g)]))
    assert set(st) == {(512, 8, 1024, "row1024"), (512, 16, 1024, "row1024"), (512, 8, 1024, "row1024-wrap4")}
    assert st[(512, 8, 1024, "row1024")]["s"]["ovl"] == pytest.approx(261 * 2e-3)
    assert st[(512, 8, 1024, "row1024-wrap4")]["s"]["ovl"] == pytest.approx(261 * 3e-3)
    assert st[(512, 16, 1024, "row1024")]["s"]["gemm"] == pytest.approx(261 * 5e-3)
    macs = 16384 * (104 * 16384**2 + 52 * 128 * 16384 + 104 * 65536 * 16384 + 256 * 16384)
    assert st[(512, 8, 1024, "row1024")]["macs"] == macs
    assert st[(512, 8, 1024, "row1024")]["mhz"] == (1980, 1980)


def test_a_missing_call_or_another_m_gives_no_step(tmp_path, capsys, monkeypatch):
    calls = list(h200_step.CALLS)
    f = tmp_path / "bench-row1024.jsonl"
    f.write_text("\n".join([_line(n, k, 1.0) for n, k in calls[1:]] + [_line(*calls[0], 1.0, m=8192)]) + "\n")
    assert h200_step.steps(h200_step.rows([str(f)])) == {}
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 1
    assert "summary: no complete step" in capsys.readouterr().out
