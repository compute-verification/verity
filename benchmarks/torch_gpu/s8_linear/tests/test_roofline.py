"""roofline.py on synthetic sweeps whose ridge is known: a pure roofline time max(2 M W / P, bytes(M) / B) has its kink at
the card's ridge, and an arm at fixed fractions of P and B has its own ridge where those fractions put it."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as FX
import roofline as RF

P, B = 800.0, 1800.0
MS = [1, 16, 64, 128, 192, 256, 384, 512, 1024, 4096]


def _roof_us(n: int, k: int, m: int, p: float, b: float, out: int = 2) -> float:
    return max(2 * m * n * k / (p * 1e12), RF.bytes_of(n, k, m, out) / (b * 1e9)) * 1e6


def test_the_ridge_solves_compute_equals_traffic():
    n, k = 8192, 8192
    m = RF.ridge(n * k, lambda mm: RF.bytes_of(n, k, mm, 2), P, B)
    assert 2 * m * n * k / (P * 1e12) == pytest.approx(RF.bytes_of(n, k, m, 2) / (B * 1e9), rel=1e-12)
    assert 200 < m < 260
    assert RF.ridge(n * k, lambda mm: RF.bytes_of(n, k, mm, 2), 1e9, B) == math.inf


def test_interp_is_exact_on_a_power_law():
    ms = [16, 64, 256]
    ts = [3.0 * m**0.5 for m in ms]
    assert RF.interp(ms, ts, 128) == pytest.approx(3.0 * 128**0.5, rel=1e-12)


def test_an_arm_on_the_roofline_meets_the_peak_at_the_ridge():
    n, k = 4 * 8192, 8192
    c = RF.curve(n * k, lambda m: RF.bytes_of(n, k, m, 2), {m: _roof_us(n, k, m, P, B) for m in MS}, P, B)
    assert c["frac_peak"][-1] == pytest.approx(1.0) and c["frac_bw"][0] == pytest.approx(1.0)
    assert c["ridge_arm"] == pytest.approx(c["ridge_card"], rel=1e-9)
    assert 0.9 < c["at_ridge"]["card"]["frac_peak"] <= 1 + 1e-9


def test_an_arm_at_fixed_fractions_has_its_own_ridge():
    n, k = 8192, 4 * 8192
    us = {m: _roof_us(n, k, m, 0.5 * P, 0.9 * B) for m in MS}
    c = RF.curve(n * k, lambda m: RF.bytes_of(n, k, m, 2), us, P, B)
    want = RF.ridge(n * k, lambda m: RF.bytes_of(n, k, m, 2), 0.5 * P, 0.9 * B)
    assert c["ridge_arm"] == pytest.approx(want, rel=1e-9)
    assert c["ridge_arm"] < c["ridge_card"]
    assert max(c["frac_peak"]) == pytest.approx(0.5) and max(c["frac_bw"]) == pytest.approx(0.9)


def test_a_layer_is_its_linears_summed():
    lin = [(n, k) for _, n, k, _ in FX.MODELS["mqa16k"].linears]
    rows = [{"n": n, "k": k, "m": m, "us": {"exact": _roof_us(n, k, m, P, B), "lt": _roof_us(n, k, m, P, B, 4)}}
            for n, k in lin for m in MS]
    s = RF.summarize(rows, ("mqa16k", "mha8k"), P, B, {"exact": 2, "lt": 4})
    assert set(s["layers"]) == {"mqa16k"} and set(s["layers"]["mqa16k"]) == {"exact", "lt"}
    lay = s["layers"]["mqa16k"]["exact"]
    assert lay["us"][0] == pytest.approx(sum(_roof_us(n, k, 1, P, B) for n, k in lin))
    assert set(s["linears"]) == {f"{n},{k}" for n, k in lin} and len(lin) == 5
    assert lay["ridge_card"] == pytest.approx(RF.ridge(sum(n * k for n, k in lin), lambda m: sum(
        RF.bytes_of(n, k, m, 2) for n, k in lin), P, B))
