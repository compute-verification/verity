"""The served window's two modes (`served_window.py`): every rung covers the window's work, the fused rungs exceed
C-Flock's statement, proving alone picks the finest rung, the commitment moves the work optimum coarser as the pooled
period grows and along √(w β h C/(o Λ)), and leaves the exfiltration optimum where it was."""

from __future__ import annotations

import pytest

import served_window as S


@pytest.fixture(scope="module")
def out() -> dict:
    return S.run()


def _results(out: dict, **match) -> list[dict]:
    rows = [r for r in out["results"] if all(r[k] == v for k, v in match.items())]
    assert rows
    return rows


def test_every_rung_covers_the_windows_work(out):
    for name, rung in out["rungs"].items():
        assert sum(p["units"] * p["work"] for p in rung["parts"]) == pytest.approx(out["work_window"], rel=1e-9), name


def test_the_fused_rungs_exceed_the_statement_and_the_dot_level_rungs_fit(out):
    fits = {name: rung["fits"] for name, rung in out["rungs"].items()}
    assert not any(fits[n] for n in ("attn|mlp token", "layer token", "4 layers token", "tile 128x128"))
    assert all(fits[n] for n in fits if n.startswith("chunk") or n in ("coord", "tile 2x2", "tile 64x64"))


def test_proving_only_picks_the_finest_rung_at_every_scope(out):
    for r in _results(out, target="work", weight=S.PROVING_ONLY):
        assert r["best_lean"] == "chunk 32", (r["scope"], r["h"])


def test_the_commitment_moves_the_work_optimum_coarser_as_the_period_grows(out):
    order = list(out["rungs"])
    for h in S.readings():
        window, week = (_results(out, target="work", scope=s, h=h, weight=1.0)[0] for s in ("window", "week"))
        assert order.index(window["best_lean"]) > order.index("chunk 32")
        assert order.index(week["best_lean"]) > order.index(window["best_lean"])
        assert week["best_lean"] == week["best_planner"] == "coord"


def test_the_chunk_familys_optimum_tracks_the_root_law(out):
    """Between the hm96 row step's floor (one window) and the statement's steps (65,536 windows) the cheapest chunk is
    within 40% of √(w β h C/(o Λ)), or at an end of the 128-byte row step that holds it (the openings cost a block a
    step), and proving alone keeps the finest."""
    for r in out["sqrt_law"]:
        assert r["proving_only_L"] == 32
        if 16 <= r["windows"] <= 4096:
            assert 1 / 1.4 < r["best_L"] / r["sqrt_L"] < 1.4 or abs(r["best_L"] - r["sqrt_L"]) < 128, r


def test_the_leak_optimum_is_the_same_in_both_modes(out):
    for h in S.readings():
        only, both = (_results(out, target="leak", h=h, weight=w)[0] for w in (S.PROVING_ONLY, 1.0))
        assert only["best_lean"] == both["best_lean"]
        best = next(x for x in both["rows"] if x["rung"] == both["best_lean"])["lean"]
        assert best["commit"] < 1e-3 * best["total"]
