"""The served window's two modes (`served_window.py`): every rung covers the window's work, the fused rungs and wide tiles
exceed C-Flock's block and no fitting rung has a proof block past it, proving alone picks the finest rung, the commitment moves the work optimum coarser as the pooled
period grows and along √(w β h C/(o Λ)), and leaves the exfiltration optimum where it was; a code-based commitment
hashes 1/ρ the bytes and no row in circuit, and no rung's fit turns on hm96's slots."""

from __future__ import annotations

import pytest

import served_window as S
from inputs import K_MAX, MASK_LOG, block_k_log, cap


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


def test_the_fused_rungs_and_wide_tiles_exceed_c_flocks_block_and_the_dot_level_rungs_fit(out):
    fits = {name: rung["fits"] for name, rung in out["rungs"].items()}
    assert not any(fits[n] for n in ("attn|mlp token", "layer token", "4 layers token", "tile 16x16", "tile 64x64"))
    assert all(fits[n] for n in fits if n.startswith("chunk") or n in ("coord", "tile 2x2", "tile 4x4"))


def test_no_fitting_rung_has_a_proof_block_past_c_flocks_block(out):
    """Every proof block of a fitting rung sits in a C-Flock block of k_log <= `K_MAX` at the composer's occupancy
    (`inputs.cap`), its statement's share that cap; a query head's whole row doesn't, so attention is key blocks, the most
    keys that fit."""
    for name, rung in out["rungs"].items():
        if rung["fits"]:
            for p in rung["parts"]:
                assert p["k_log"] <= K_MAX, (name, p["name"])
                assert cap(round(p["gates"]), p["rows"]) == p["per_statement"] > 0, (name, p["name"])
    kb = S.attention_keys()
    positions = round(S.IN["tokens"].value / S.IN["sequences"].value)
    assert block_k_log(*_rounded(S.attention_block(positions, True))) > K_MAX
    assert block_k_log(*_rounded(S.attention_block(2 * kb, False))) > K_MAX >= block_k_log(
        *_rounded(S.attention_block(kb, False)))


def _rounded(block: tuple[float, tuple[float, ...]]) -> tuple[int, tuple[int, ...]]:
    gates, rows = block
    return round(gates), tuple(map(round, rows))


def test_proving_only_picks_the_finest_rung_at_every_scope(out):
    for r in _results(out, target="work", weight=S.PROVING_ONLY):
        assert r["best_lean"] == "chunk 32", (r["scope"], r["h"])


def test_the_commitment_moves_the_work_optimum_coarser_as_the_period_grows(out):
    """The scopes are the window and core's monitoring periods (`wipe_monitoring.PERIODS`); pooled over a longer one the
    work optimum is never finer, and from a week on it is coord in both views."""
    order = list(out["rungs"])
    assert list(S.scopes()) == ["window"] + [name for name, _ in S.wipe_monitoring.PERIODS]
    for h in S.readings():
        best = [order.index(_results(out, target="work", scope=s, h=h, scheme="hm96", weight=1.0)[0]["best_lean"])
                for s in S.scopes()]
        assert order.index("chunk 32") < best[0] < best[1] and best == sorted(best), (h, best)
        for s in list(S.scopes())[1:]:
            r = _results(out, target="work", scope=s, h=h, scheme="hm96", weight=1.0)[0]
            assert r["best_lean"] == r["best_planner"] == "coord", (s, h)


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
        only, both = (_results(out, target="leak", h=h, scheme="hm96", weight=w)[0] for w in (S.PROVING_ONLY, 1.0))
        assert only["best_lean"] == both["best_lean"]
        best = next(x for x in both["rows"] if x["rung"] == both["best_lean"])["lean"]
        assert best["commit"] < 1e-3 * best["total"]


def test_a_code_commitment_hashes_1_over_rho_the_bytes_and_no_row_in_circuit(out):
    """At each rate ρ every rung commits 1/ρ hm96's bytes' hashing, with the same draws, and proves cheaper by its rows'
    in-circuit hashing."""
    codes = [r for r in out["results"] if r["scheme"] != "hm96"]
    assert {r["rate"] for r in codes} == set(S.RATES)
    for r in codes:
        (hm,) = _results(out, scope=r["scope"], target=r["target"], h=r["h"], scheme="hm96", weight=r["weight"])
        for code, base in zip(r["rows"], hm["rows"]):
            assert code["rung"] == base["rung"] and code["fits"] == base["fits"]
            assert code["lean"]["commit"] == pytest.approx(base["lean"]["commit"] / r["rate"], rel=1e-9)
            assert code["lean"]["draws"] == pytest.approx(base["lean"]["draws"], rel=1e-9)
            assert code["lean"]["prove"] < base["lean"]["prove"], (r["scheme"], code["rung"])


def test_no_rungs_fit_turns_on_the_hm96_and_sha512_slots(out):
    """A rung that misses C-Flock's block misses it on a proof block's own slot (its ANDs and rows' bits) and the mask, or
    on its width, so the code-based commitment, which drops hm96's slots, fits the same rungs."""
    for name, rung in out["rungs"].items():
        if not rung["fits"]:
            assert any((1 << (round(p["gates"]) + 8 * sum(p["rows"])).bit_length()) + (1 << MASK_LOG) > 1 << K_MAX
                       or p["out_bits"] > S.UNIT_CLASS_MAX for p in rung["parts"] if not p["fits"]), name
