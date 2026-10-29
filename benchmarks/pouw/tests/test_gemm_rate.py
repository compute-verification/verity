"""gemm_rate/summarize.py: MACs per SM clock against 4,096 and theta from rate.sh's stamped windows, a row's median over
its rounds, the step over every shape, and the stamps' pairing by SM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm_rate"))
import summarize

M, N, K = 16384, 28672, 4096
SMS = 132


def _win(arm: str, r: int, calls: int, clocks: float, ns: float, cmhz=None) -> dict:
    return {"arm": arm, "round": r, "calls": calls, "ms": ns / 1e6, "sm_clocks": clocks, "span_ns": ns, "sms": SMS,
            "cmhz": cmhz, "nvml_mhz": 1980, "reasons": 4 if arm == "lt" else 0}


def _clocks(calls: int, m: int, n: int, k: int, of: float) -> float:
    return calls * m * n * k / (4096 * of)


def _out(tmp_path: Path, shapes=(("8B-up", M, N, K),)) -> Path:
    lines, torch_lines = [], []
    for name, m, n, k in shapes:
        ws = [_win("gemm_d", r, 800, _clocks(800, m, n, k, 0.90 + 0.01 * r), 2e9, 1830.0) for r in range(3)]
        ws.append(_win("lt", 0, 700, _clocks(700, m, n, k, 0.80), 2e9))
        lines.append(json.dumps({"fill": True, "shape": name, "m": m, "n": n, "k": k, "windows": ws}))
        torch_lines.append(json.dumps({"kind": "window", "shape": name, "m": m, "n": n, "k": k,
                                       **_win("int_mm", 0, 700, _clocks(700, m, n, k, 0.75), 2e9)}))
    (tmp_path / "fb-rp5-tst1.jsonl").write_text("GK noise\n" + "\n".join(lines) + "\n")
    (tmp_path / "torch.jsonl").write_text(json.dumps({"kind": "device", "sms": SMS}) + "\n" + "\n".join(torch_lines) + "\n")
    return tmp_path


def test_a_window_is_its_macs_over_its_sm_clocks():
    w = _win("gemm_d", 0, 800, 1.2e12, 2e9)
    p = summarize.point(w, M, N, K)
    macs = 800 * M * N * K
    assert p["of_4096"] == pytest.approx(macs / 1.2e12 / 4096)
    assert p["idle"] == pytest.approx(1 - p["of_4096"])
    assert p["theta"] == pytest.approx(macs / 2.0 / 989.4e12)
    assert p["sm_mhz"] == pytest.approx(1.2e12 / SMS / 2e9 * 1e3)
    assert p["ms_call"] == pytest.approx(2000 / 800)


def test_rows_are_medians_over_rounds_with_their_spread(tmp_path):
    res, refused = summarize.table(summarize.windows(_out(tmp_path)))
    by = {r["arm"]: r for r in res}
    assert set(by) == {"gemm_d", "lt", "int_mm"} and refused == []
    g = by["gemm_d"]
    assert (g["rounds"], g["variant"], g["cmhz"]) == (3, "rp5-tst1", 1830.0)
    assert g["of_4096"] == pytest.approx(0.91) and g["spread"] == pytest.approx(0.02)
    assert g["of_lt"] == pytest.approx(g["theta"] / by["lt"]["theta"])
    assert by["lt"]["reasons"] == 4 and by["lt"]["label"] == summarize.LABELS["lt"]
    assert [r["arm"] for r in res] == ["gemm_d", "lt", "int_mm"]


def test_the_step_sums_macs_and_clocks_over_every_shape(tmp_path):
    shapes = (("8B-up", M, N, K), ("8B-o", M, 4096, 4096))
    res, _ = summarize.table(summarize.windows(_out(tmp_path, shapes)))
    st = {r["arm"]: r for r in summarize.steps(res)}
    assert set(st) == {"gemm_d", "lt", "int_mm"}
    macs = sum(m * n * k for _, m, n, k in shapes)
    clocks = sum(m * n * k / (4096 * 0.91) for _, m, n, k in shapes)
    assert st["gemm_d"]["of_4096"] == pytest.approx(macs / clocks / 4096)
    assert st["gemm_d"]["idle"] == pytest.approx(1 - 0.91)
    assert summarize.steps([r for r in res if r["shape"] == "8B-up"]) == []


def test_stamps_pair_by_sm_and_refuse_a_missing_or_timed_out_sm():
    a = [(1000, 10, 1), (1010, 20, 0)]
    b = [(3010, 2020, 0), (3000, 1010, 1)]
    assert summarize.span(a, b) == {"sm_clocks": 1000 + 2000, "span_ns": 2000, "sms": 2}
    assert summarize.span(a, [(3000, 1010, 1), (3010, 2020, 2)])["sms"] == 0
    assert summarize.span([(1000, 10, -1), (1010, 20, 0)], b)["sms"] == 0
    assert summarize.span([(1000, 10, 0), (1010, 20, 0)], b)["sms"] == 0


def test_a_refused_window_is_named_and_left_out(tmp_path, capsys):
    out = _out(tmp_path)
    bad = {"kind": "window", "shape": "8B-up", "m": M, "n": N, "k": K, "arm": "int_mm", "round": 1, "calls": 700,
           "ms": 2000.0, "sm_clocks": 0, "span_ns": 0, "sms": 0}
    with (out / "torch.jsonl").open("a") as f:
        f.write(json.dumps(bad) + "\n")
    assert summarize.main([str(out), "--json", str(tmp_path / "t.json")]) == 0
    md = capsys.readouterr().out
    assert "Refused 1 window(s)" in md and "8B-up int_mm round 1" in md
    t = json.loads((tmp_path / "t.json").read_text())
    assert len(t["rows"]) == 3 and len(t["refused"]) == 1
