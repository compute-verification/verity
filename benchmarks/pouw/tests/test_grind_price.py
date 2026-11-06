"""grind_price: the template's θ meets `TileLuck`'s condition and is tight, κ must exceed the live rate, the premium is
ln(T/δ)/θ, H1's per-head floor is attn_noise's, and the shapes are eps_budget's."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_grind_price", Path(__file__).parents[1] / "grind_price.py")
G = importlib.util.module_from_spec(spec)
sys.modules["pouw_grind_price"] = G
spec.loader.exec_module(G)

C = G.TILE[1]
W = range(1, G.SEG + 1)


def test_a_one_product_skip_is_lucky_at_the_live_rate_so_kappa_must_exceed_it():
    assert G.Q_BIT + (1 - G.Q_BIT) * G.P_ALIVE == G.BU.P_PASS
    assert G.KAPPA_LIVE == 3
    assert G.y_star(C, G.KAPPA_LIVE) == 0.0
    assert G.y_star(C, G.KAPPA_LIVE, G.Q_KEYED) > 0


@pytest.mark.parametrize("kappa", [3.25, 4.0, 6.0, 12.0])
@pytest.mark.parametrize("q", [G.Q_BIT, G.Q_KEYED])
def test_y_star_meets_tile_lucks_condition_at_every_word_size_and_is_tight(kappa, q):
    y = G.y_star(C, kappa, q)
    assert y > 0
    assert all(G._log_f(w, y, C, kappa, q) <= 1e-12 for w in W)
    assert any(G._log_f(w, y * 1.001, C, kappa, q) > 0 for w in W)


def test_the_point_lean_proves_meets_the_condition_inside_the_optimum():
    kappa, y = G.PROVED
    assert all(G._log_f(w, y, C, kappa, G.Q_BIT) <= 0 for w in W)
    assert 2 < G.y_star(C, kappa) / y < 3


def test_y_star_grows_with_kappa_and_with_the_keyed_dither():
    ys = [G.y_star(C, k) for k in G.KAPPAS[1:]]
    assert ys == sorted(ys)
    assert all(G.y_star(C, k, G.Q_KEYED) > G.y_star(C, k) for k in (3.25, 4.0, 6.0))


def test_the_premium_is_ln_t_over_delta_over_theta_and_infinite_without_slack():
    assert G.ln_t_over_delta(128) == pytest.approx(168 * math.log(2))
    s = G.SeedGroup("linears", "x", 32 * C * 4096 * 10, 32, C, 4096)
    assert G.premium(s, 128, 4.0) == pytest.approx(G.ln_t_over_delta(128) * s.r / G.y_star(C, 4.0))
    assert G.premium(s, 128, G.KAPPA_LIVE) == math.inf
    assert G.levers()["cover_at_template"]["linear K=4096, κ=4, y*"]["pi_star_tiles"] == pytest.approx(
        G.premium(s, 128, 4.0) / s.tile_macs)


def test_h1s_floor_a_seed_a_head_is_attn_noises_and_a_call_divides_it_by_its_rows():
    shape = G.EB.MODELS["llama-3.1-8b"]
    head = G.h1_attention(shape, "decode", 1, 2048, 128, "head")
    assert head["qk"] == pytest.approx(168 / G.H1_NOISE["q"], rel=0.01)
    assert head["pv"] == pytest.approx(168 / G.H1_NOISE["p"], rel=0.01)
    call = G.h1_attention(shape, "decode", 8, 2048, 128, "call")
    assert head["pv"] / (shape.heads * 8) <= call["pv"] <= 1.05 * head["pv"] / (shape.heads * 8)


def test_h1s_h_form_is_attn_noises_closed_form_and_the_pick_after_the_seed_only_raises_it():
    shape = G.EB.MODELS["llama-3.1-8b"]
    for L in G.PREFILL:
        f = G.h1_h_form(shape, "prefill", L, L, 128)
        assert f["qk"] == pytest.approx(2.625 / (L + 1))
        assert f["pv"] == pytest.approx(1.3125 / L)
    f = G.h1_h_form(shape, "decode", 1, 32768, 128)
    assert (f["qk"], f["pv"]) == (pytest.approx(0.375), pytest.approx(0.75))
    for shape in G.EB.MODELS.values():
        for regime, n, ctx in G.regimes():
            for group in ("head", "call"):
                f, v = G.h1_h_form(shape, regime, n, ctx, 128, group), G.h1_attention(shape, regime, n, ctx, 128, group)
                assert all(f[k] <= v[k] <= 1.25 * f[k] for k in ("qk", "pv"))


def test_decode_cover_is_the_least_seed_group_under_a_tenth_of_a_percent():
    cov = G.decode_cover(128)
    assert cov["h_form"] == {"qk": 375, "pv": 750}
    shape = G.EB.MODELS["llama-3.1-70b"]
    for kind, g in cov["visible"].items():
        assert g >= cov["h_form"][kind]
        assert G.h1_attention(shape, "decode", 1, 32768, 128, g)[kind] <= G.FLAG
        assert G.h1_attention(shape, "decode", 1, 32768, 128, g - 1)[kind] > G.FLAG


def eps_shared(regime, ctx, pairs, gq, kind):
    rows, units = G._h1_rows(regime, ctx, 128, pairs, gq)[kind]
    return G.visible_premium(rows, 128) / units


def test_gqa_moves_decodes_cover_not_at_all_and_divides_a_prefill_calls_floor_by_the_kv_heads():
    cov = G.decode_cover(128)["visible"]
    for shape in G.EB.MODELS.values():
        gq = shape.heads // shape.kv
        for kind, g in cov.items():
            assert eps_shared("decode", 32768, g, gq, kind) <= G.FLAG < eps_shared("decode", 32768, g - 1, gq, kind)
        for L in G.PREFILL:
            f = G.h1_h_form(shape, "prefill", L, L, 128, "call")
            assert f["qk"] == pytest.approx(2.625 / ((L + 1) * shape.kv))
            assert f["pv"] == pytest.approx(1.3125 / (L * shape.kv))


def test_prefill_cover_counts_kv_head_pairs_the_same_for_every_model():
    assert G.prefill_cover(128, 2048) == {"qk": 2, "pv": 1}
    for ctx in G.PREFILL_COVER_CTX:
        c = G.prefill_cover(128, ctx)
        assert c == G.prefill_cover(128, ctx, "llama-3.1-70b")
        for model, shape in G.EB.MODELS.items():
            gq = shape.heads // shape.kv
            for kind, n in c.items():
                assert eps_shared("prefill", ctx, n * gq, gq, kind) <= G.FLAG
                assert n == 1 or eps_shared("prefill", ctx, (n - 1) * gq, gq, kind) > G.FLAG


def test_sites_are_eps_budgets_shapes():
    for shape in G.EB.MODELS.values():
        ss = G.sites(shape, "decode", 1, 2048)
        assert [(s.name, s.k) for s in ss if s.kind == "linears"] == [(n, k) for n, k, _ in shape.linears]
        assert sum(s.macs for s in ss if s.kind in ("qk", "pv")) == 2 * shape.heads * shape.d * 2048


def test_only_decode_attention_at_small_batch_is_flagged():
    t = G.table()
    assert {r["family"] for r in t["rows"]} == {f.name for f in G.BU.FAMILIES}
    for r in t["rows"]:
        assert "linears" not in r["flag"]
        assert not r["flag"] or r["regime"].startswith("decode b1 ")
    assert any(r["flag"] for r in t["rows"])
