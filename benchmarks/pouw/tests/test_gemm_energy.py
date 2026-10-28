"""gemm_energy/summarize.py: pJ a MAC over idle and theta from energy.sh's windows, a row's median over its rounds, and
the excess over the floor split by the floor's arms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm_energy"))
import summarize

M, N, K = 16384, 28672, 4096


def _win(arm: str, r: int, calls: int, ms: float, mj: int, cmhz=None) -> dict:
    return {"arm": arm, "round": r, "calls": calls, "ms": ms, "mj": mj, "host_t0": 0.0, "host_t1": 0.0, "cmhz": cmhz,
            "nvml_mhz": 1980, "reasons": 4 if arm == "lt" else 0}


def _out(tmp_path: Path) -> Path:
    d = tmp_path / "1830"
    d.mkdir()
    fb = {"fill": True, "shape": "8B-up", "m": M, "n": N, "k": K, "arms": {},
          "power": [_win("gemm_d", r, 2000, 4000.0 + r, 2_600_000 + 1000 * r, 1830.0) for r in range(3)]
          + [_win("lt", 0, 1900, 4000.0, 2_700_000)]}
    (d / "fb-rp5-tst1.jsonl").write_text("GK noise\n" + json.dumps(fb) + "\n")
    t = [{"kind": "idle", "s": 8.0, "mj": 624000, "w": 78.0}, {"kind": "counter", "period_ms": 20.0},
         {"kind": "window", "shape": "8B-up", "m": M, "n": N, "k": K, **_win("int_mm", 0, 1900, 4000.0, 2_700_000)}]
    (d / "torch.jsonl").write_text("\n".join(json.dumps(x) for x in t) + "\n")
    return tmp_path


def test_a_window_is_its_energy_less_idle_over_its_macs():
    p = summarize.point({"calls": 2000, "ms": 4000.0, "mj": 2_600_000}, M, N, K, 78.0, 989.4e12)
    macs = 2000 * M * N * K
    assert p["pj_idle"] == pytest.approx(1e12 * (2600.0 - 78.0 * 4.0) / macs)
    assert p["pj_all"] == pytest.approx(1e12 * 2600.0 / macs)
    assert p["theta"] == pytest.approx(macs / 4.0 / 989.4e12)
    assert p["w"] == pytest.approx(650.0)


def test_rows_are_medians_over_rounds_with_their_spread(tmp_path):
    res = summarize.table(summarize.windows(_out(tmp_path)))
    by = {r["arm"]: r for r in res}
    assert set(by) == {"gemm_d", "lt", "int_mm"}
    g = by["gemm_d"]
    assert (g["rounds"], g["variant"], g["cmhz"]) == (3, "rp5-tst1", 1830.0)
    pts = [summarize.point(_win("gemm_d", r, 2000, 4000.0 + r, 2_600_000 + 1000 * r), M, N, K, 78.0, 989.4e12)
           for r in range(3)]
    assert g["pj_idle"] == pytest.approx(sorted(p["pj_idle"] for p in pts)[1])
    assert g["pj_spread"] == pytest.approx(max(p["pj_idle"] for p in pts) - min(p["pj_idle"] for p in pts))
    assert by["lt"]["reasons"] == 4 and by["lt"]["label"] == summarize.LABELS["lt"]
    assert by["int_mm"]["pj_idle"] == pytest.approx(by["lt"]["pj_idle"])


def test_the_excess_splits_into_the_floor_arms_steps_and_issue_overhead():
    fl = [["loop", 0.66], ["tma", 0.70], ["hbm", 0.80], ["epilogue", 0.85]]
    s = summarize.split({"pj_idle": 1.0}, fl)
    assert s["excess"] == pytest.approx(0.34) and s["excess_pct"] == pytest.approx(100 * 0.34 / 0.66)
    assert (s["step tma"], s["step hbm"], s["step epilogue"]) == pytest.approx((0.04, 0.10, 0.05))
    assert s["issue overhead"] == pytest.approx(0.15)
    assert s["step tma"] + s["step hbm"] + s["step epilogue"] + s["issue overhead"] == pytest.approx(s["excess"])


def test_main_writes_the_table_and_its_json(tmp_path, capsys):
    out = _out(tmp_path)
    (tmp_path / "floors.json").write_text(json.dumps({"1830": [["loop", 0.66], ["hbm", 0.8]]}))
    assert summarize.main([str(out), "--floors", str(tmp_path / "floors.json"), "--json", str(tmp_path / "t.json")]) == 0
    md = capsys.readouterr().out
    assert "locked at 1830 MHz" in md and "measured idle 78.0 W" in md and "counter period 20 ms" in md
    assert "Excess over loop (0.660 pJ)" in md
    assert len(json.loads((tmp_path / "t.json").read_text())["rows"]) == 3
