from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frontier as F  # noqa: E402


def test_units_are_the_step_model_a_window():
    m = {"token": {"roundings": 10.0, "gathers": 1.0}, "sample": {"roundings": 100.0},
         "coefficients": {"roundings": {"r0": 1.0, "b_q": 0.0, "b_kv": 2.0, "w0": 0.0, "w_start": 0.0}}, "calls": 4}
    w = {k: np.asarray([v]) for k, v in {"T": 3, "X": 1, "Q": 0, "K": 50, "n": 1, "n0": 0}.items()}
    assert F.units(m, w)[0] == 11 * 3 + 100 * 1 + 4 * (1 * 3 + 2 * 50)


def test_the_worst_window_is_the_most_units_over_T_seconds_over_T():
    u = np.asarray([1.0, 5.0, 5.0, 1.0, 0.0])
    assert F.worst_window(u, 2.0) == pytest.approx(5.0)
    assert F.worst_window(u, 1.0) == pytest.approx(5.0)
    assert F.worst_window(u, 5.0) == pytest.approx(12.0 / 5)
    # a trace shorter than T spends all it has in one window
    assert F.worst_window(u, 20.0) == pytest.approx(12.0 / 20)


def test_the_credited_share_interpolates_between_stocks_and_is_zero_past_the_grid():
    by = {"0.0": 0.9, "50.0": 0.5, "100.0": 0.1}
    assert F.share_at(by, 100.0, 25.0) == pytest.approx(0.7)
    assert F.share_at(by, 100.0, 150.0) == 0.0


def test_the_table_charges_a_longer_life_a_larger_stock():
    grid = {f"T={T:g},L={L}": {p: {"by_stock": {"0.0": 1.0, "1000000000000.0": 0.0}, "rekey_bytes_per_s": 1e6 * L,
                                   "append_bytes_per_s": 5e8, "credited_before_stock": 1.0, "held_bytes_mean": 4e10}
                                for p in ("rekey", "held", "expire")}
            for T, Ls in F.GRID for L in Ls}
    phase = {"frontier": grid, "busy_s": 50.0, "interval_s": [0.0, 100.0]}
    rows = F.table(phase, 1e12, 1e11, 1e4, 1e9)
    by = {(r["T"], r["L"]): r for r in rows}
    assert by[(60.0, 10)]["stock_GB[conjectured]"] == pytest.approx(10 * by[(60.0, 1)]["stock_GB[conjectured]"])
    assert by[(60.0, 1)]["law[conjectured]"] < by[(60.0, 1)]["law[best found]"]
    assert by[(60.0, 2)]["rekey_step_share"] == pytest.approx(2 * by[(60.0, 2)]["rekey_gpu_share"])
    assert by[(60.0, 2)]["rekey_total_gpu_share"] == pytest.approx(0.5 + 2e-3)
    # under the origin clock a re-key is a recompute: the re-prefill's time on top of the encode's
    o = F.table({**phase, "frontier_origin": grid}, 1e12, 1e11, 1e4, 1e9, "origin", 3e9)
    assert o[0]["rekey_gpu_share"] == pytest.approx(rows[0]["rekey_gpu_share"] * (1 + 1 / 3))
    # a re-key path at half the encode rate doubles the re-keys' share and leaves the append's
    h = F.table(phase, 1e12, 1e11, 1e4, 1e9, rekey_bps=5e8)
    assert h[0]["rekey_gpu_share"] == pytest.approx(2 * rows[0]["rekey_gpu_share"])
    assert h[0]["append_gpu_share"] == pytest.approx(rows[0]["append_gpu_share"])
    # a model with 2.5x the KV a token over 8 GPUs, on the run's schedule: held and the stock's reach scale, each
    # GPU encodes an eighth
    m = F.table(phase, 1e12, 1e11, 1e4, 1e9, kv_scale=2.5, gpus=8, law_pool=6e11)
    assert m[0]["held_GB"] == pytest.approx(2.5 * 40.0)
    assert m[0]["f_live[conjectured]"] == pytest.approx(rows[0]["f_live[conjectured]"] / 2.5)
    assert m[0]["append_gpu_share"] == pytest.approx(rows[0]["append_gpu_share"] * 2.5 / 8)
    assert m[0]["law[conjectured]"] == pytest.approx(1 - rows[0]["stock_GB[conjectured]"] * 1e9 / 6e11)


def test_the_table_has_twenty_seconds_at_two_windows_refuses_one_window_and_reads_each_windows_budget():
    grid = {f"T={T:g},L={L}": {p: {"by_stock": {"0.0": 1.0, "1000000000000.0": 0.0}, "rekey_bytes_per_s": 1e6 * L,
                                   "append_bytes_per_s": 5e8, "credited_before_stock": 1.0, "held_bytes_mean": 4e10}
                               for p in ("rekey", "held", "expire")}
            for T, Ls in F.GRID for L in Ls}
    phase = {"frontier": grid, "busy_s": 50.0, "interval_s": [0.0, 100.0]}
    rows = F.table(phase, 1e12, {60.0: 1e11, 20.0: 2e11}, 1e4, 1e9, rekey_bps=2e9, rekey_kvcu_bps=5e8)
    by = {(r["T"], r["L"]): r for r in rows}
    assert by[(20.0, 2)]["refused"] is None and by[(60.0, 2)]["refused"] is None
    assert all(r["refused"] for r in rows if r["L"] == 1) and all(not r["refused"] for r in rows if r["L"] > 1)
    # a row's stock is its own T's budget: twice the rate at a third of the life
    assert by[(20.0, 3)]["stock_GB[conjectured]"] == pytest.approx(
        by[(60.0, 1)]["stock_GB[conjectured]"] * 2, rel=1e-3)
    assert by[(20.0, 2)]["r_cap_engine"] == 2e11
    # the old re-key rate's columns beside the default's
    r = by[(20.0, 2)]
    assert r["rekey_gpu_share"] == pytest.approx(2e6 / 2e9)
    assert r["rekey_gpu_share_kvcu"] == pytest.approx(4 * r["rekey_gpu_share"])
    assert r["held_total_gpu_share_kvcu"] == pytest.approx(r["append_gpu_share"] + r["held_gpu_share_kvcu"])
    assert "rekey_gpu_share_kvcu" not in F.table(phase, 1e12, 1e11, 1e4, 1e9)[0]


def test_a_run_too_light_to_batch_a_big_prefill_reads_its_block_clock_and_takes_the_origin_clocks_rate_outright(
        tmp_path, monkeypatch):
    import json

    grid = {f"T={T:g},L={L}": {p: {"by_stock": {"0.0": 1.0, "1000000000000.0": 0.0}, "rekey_bytes_per_s": 1e6,
                                   "credited_before_stock": 1.0, "held_bytes_mean": 4e10}
                               for p in ("rekey", "held", "expire")}
            for T, Ls in F.GRID for L in Ls}
    win = {k: [1.0, 2.0] for k in ("T", "X", "Q", "K", "n", "n0")}
    phase = {"frontier": grid, "frontier_origin": grid, "busy_s": 50.0, "interval_s": [0.0, 100.0],
             "written_bytes_per_s": 5e8, "delayed": {}, "occupancy": {"held_bytes_mean": 4e10},
             "model_windows_1s": win, "model_totals": {k: 3.0 for k in win}, "prefill_rows_per_s": None}
    (tmp_path / "a.json").write_text(json.dumps({"serve": {"pool_bytes": 1e12, "kv_bytes_per_token": 131072},
                                                 "phases": [phase]}))
    model = {"token": {"r": 1.0}, "sample": {}, "coefficients": {}, "calls": 0}
    (tmp_path / "m.json").write_text(json.dumps({"models": {"m": {"l": model}}}))
    base = ["frontier.py", str(tmp_path / "a.json"), "--models", str(tmp_path / "m.json"), "--model-name", "m",
            "--lever", "l", "--engines", "1"]

    def run(*extra):
        out = tmp_path / "out.json"
        monkeypatch.setattr(sys, "argv", [*base, *extra, "--json", str(out)])
        assert F.main() == 0
        return json.loads(out.read_text())

    assert run()["recompute_Bps"] is None
    with pytest.raises(SystemExit, match="prefill-rows-per-s"):
        run("--clock", "origin")
    assert run("--clock", "origin", "--prefill-rows-per-s", "1e4")["recompute_Bps"] == pytest.approx(1e4 * 131072)


def test_main_reads_the_cap_from_the_worst_window_of_the_phase_or_the_run_and_records_prefix_caching(
        tmp_path, monkeypatch):
    import json

    grid = {f"T={T:g},L={L}": {p: {"by_stock": {"0.0": 1.0, "1000000000000.0": 0.0}, "rekey_bytes_per_s": 1e6,
                                   "credited_before_stock": 1.0, "held_bytes_mean": 4e10}
                               for p in ("rekey", "held", "expire")}
            for T, Ls in F.GRID for L in Ls}

    def phase(tokens):
        win = {k: [0.0] * len(tokens) for k in ("X", "Q", "K", "n", "n0")} | {"T": tokens}
        return {"frontier": grid, "busy_s": 50.0, "interval_s": [0.0, 100.0], "written_bytes_per_s": 5e8,
                "delayed": {}, "occupancy": {"held_bytes_mean": 4e10}, "model_windows_1s": win,
                "model_totals": {k: float(sum(v)) for k, v in win.items()}, "prefill_rows_per_s": None}

    # phase 0 bursts to 100 in one second of 60; phase 1 runs 30 a second for a minute
    (tmp_path / "a.json").write_text(json.dumps({
        "serve": {"pool_bytes": 1e12, "kv_bytes_per_token": 131072, "args": {"prefix_caching": "off"}},
        "phases": [phase([100.0] + [10.0] * 59), phase([30.0] * 60)]}))
    model = {"token": {"r": 1.0}, "sample": {}, "coefficients": {}, "calls": 0}
    (tmp_path / "m.json").write_text(json.dumps({"models": {"m": {"l": model}}}))
    base = ["frontier.py", str(tmp_path / "a.json"), "--models", str(tmp_path / "m.json"), "--model-name", "m",
            "--lever", "l", "--engines", "1", "--phase", "0"]

    def run(*extra):
        out = tmp_path / "out.json"
        monkeypatch.setattr(sys, "argv", [*base, *extra, "--json", str(out)])
        assert F.main() == 0
        return json.loads(out.read_text())

    peak = run()
    assert peak["prefix_caching"] == "off" and peak["r_cap"] == "peak"
    assert {r["r_cap_engine"] for r in peak["rows"]} == {100.0}
    w = run("--r-cap", "window")
    assert w["r_cap_trace_engine"]["window"] == {"60.0": 690 / 60, "20.0": 290 / 20}
    assert {(r["T"], r["r_cap_engine"]) for r in w["rows"]} == {(60.0, 690 / 60), (20.0, 290 / 20)}
    assert {r["r_cap_engine"] for r in run("--r-cap", "window-run")["rows"]} == {30.0}
    assert run("--r-cap-node", "7")["r_cap"] == "node"
