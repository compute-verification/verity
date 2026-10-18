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


def test_the_credited_share_interpolates_between_stocks_and_is_zero_past_the_grid():
    by = {"0.0": 0.9, "50.0": 0.5, "100.0": 0.1}
    assert F.share_at(by, 100.0, 25.0) == pytest.approx(0.7)
    assert F.share_at(by, 100.0, 150.0) == 0.0


def test_the_table_charges_a_longer_life_a_larger_stock():
    grid = {f"T={T:g},L={L}": {p: {"by_stock": {"0.0": 1.0, "1000000000000.0": 0.0}, "rekey_bytes_per_s": 1e6 * L,
                                   "credited_before_stock": 1.0, "held_bytes_mean": 4e10}
                                for p in ("rekey", "held", "expire")}
            for T, Ls in F.GRID for L in Ls}
    phase = {"frontier": grid, "busy_s": 50.0, "interval_s": [0.0, 100.0]}
    rows = F.table(phase, 1e12, 1e11, 1e4, 1e9)
    by = {(r["T"], r["L"]): r for r in rows}
    assert by[(60.0, 10)]["stock_GB[conjectured]"] == pytest.approx(10 * by[(60.0, 1)]["stock_GB[conjectured]"])
    assert by[(60.0, 1)]["law[conjectured]"] < by[(60.0, 1)]["law[best found]"]
    assert by[(60.0, 2)]["rekey_step_share"] == pytest.approx(2 * by[(60.0, 2)]["rekey_gpu_share"])
    # under the origin clock a re-key is a recompute: the re-prefill's time on top of the encode's
    o = F.table({**phase, "frontier_origin": grid}, 1e12, 1e11, 1e4, 1e9, "origin", 3e9)
    assert o[0]["rekey_gpu_share"] == pytest.approx(rows[0]["rekey_gpu_share"] * (1 + 1 / 3))
