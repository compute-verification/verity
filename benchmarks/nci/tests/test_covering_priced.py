import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import covering_priced as P  # noqa: E402
import kappa as K  # noqa: E402

M = K.MODELS["kimi-k2"]
GATES = K.schedule(M)[1]


def base():
    return P.counted(P.stages(M), head=False)


def test_the_walk_counts_kappas_products():
    """M* is kappa's N: every matrix, the router and the active experts, not the LM head or the norm gammas."""
    assert sum(s["products"] for s in base()) == pytest.approx(K.active_weights(M), rel=1e-12)
    sh = P.shapes(M)
    assert sum(pr for c, (_, pr, _) in sh.items() if P.CLASSES[c][2]) == pytest.approx(K.active_weights(M), rel=1e-12)


def test_d_lands_in_layer_0_at_1_mb():
    """The realism note's crossing: at k = 1 MB and c = 4, only layer 0's q_a and kv_a are unchained (0.048% of every
    forward product, the LM head's included)."""
    r = P.rho("two c's", base(), {}, 4, P.KS["1 MB"], 4.0, GATES)
    assert r["first"] == "layer 0 q_b"
    d, a = M["d"], M["attention"]
    unchained = (1 - r["chained"]) * K.active_weights(M)
    assert unchained == pytest.approx(a["q_rank"] * d + (a["kv_rank"] + a["rope"]) * d, rel=1e-9)
    assert unchained / (K.active_weights(M) + M["vocab"] * d) == pytest.approx(0.000478, rel=0.01)


@pytest.mark.parametrize("fmt,b", list(P.FMTS.items()))
def test_one_price_everywhere_is_the_two_cost_form(fmt, b):
    """Every weight at price c_w, chained by count at c_w: the priced bound is the two-cost one, up to its k p_max."""
    cw = 2.5
    p = {c: cw for c in P.CLASSES}
    for k in P.KS.values():
        two = P.rho("two c's", base(), p, b, k, cw, GATES)["rho"]
        priced = P.rho("priced", base(), p, b, k, cw, GATES, chain="B")["rho"]
        assert priced == pytest.approx(two, rel=1e-4)


def test_the_additive_form_is_four_times_the_priced_one():
    p = P.prices(P.PYTHIA["160m@100k"]["fp8"])
    one = P.rho("priced", base(), p, 8, P.KS["1 MB"], 1.0, GATES)["rho"]
    four = P.rho("priced additive", base(), p, 8, P.KS["1 MB"], 1.0, GATES)["rho"]
    assert four == pytest.approx(4 * one, rel=1e-4)


def test_daniels_table():
    """The covering bound write-up's marginal-slowdown table, Kimi K2 raw at k = 2e7, w = 2.5: 41, 166, 660."""
    d = P.check_daniel()
    assert d["base"] == pytest.approx(41, rel=0.03)
    assert d["additive"] == pytest.approx(166, rel=0.03)
    assert d["a=4"] == pytest.approx(660, rel=0.03)


def test_the_realism_notes_rows():
    """At 1 MB: one uniform c at the cheapest 1%
    tops out at about 1.2e3 (bf16); two c's at the mean price, which the priced form gives, at about 1.1e4."""
    g = P.grid(base(), P.PYTHIA, P.CHEAPEST, GATES)
    assert g["160m@100k"]["bf16"]["1 MB"]["one c"]["rho"] == pytest.approx(1.2e3, rel=0.05)
    assert g["160m@100k"]["bf16"]["1 MB"]["priced"]["rho"] == pytest.approx(1.1e4, rel=0.05)


def test_a_price_file_maps_hf_names_onto_roles(tmp_path):
    f = tmp_path / "olmo.json"
    tab = {"q_proj": 3.0, "k_proj": 4.0, "v_proj": 5.0, "o_proj": 6.0, "gate_proj": 2.0, "up_proj": 4.0,
           "down_proj": 3.5, "lm_head": 1.0}
    f.write_text(json.dumps({"points": {"x": {fmt: tab for fmt in P.FMTS}}, "cheapest": {"x": {"1%": {"bf16": 1.0}}}}))
    pts, cheapest = P.load_prices(str(f))
    roles = pts["x"]["bf16"]
    assert roles == {"qkv": 4.0, "attn_out": 6.0, "mlp_in": 3.0, "mlp_out": 3.5, "head": 1.0, "named": 2.0}
    assert P.named(roles) == 2.0
    p = P.prices(roles)
    assert p["expert_in"] == 3.0 and p["mla_out"] == 6.0 and p["router"] == 1.0 and p["norm"] == 0.0
    assert p["dense_in"] == p["shared_in"] == 3.0 and p["dense_out"] == p["shared_out"] == 3.5
    assert cheapest["x"]["1%"]["bf16"] == 1.0


def test_the_dense_mlp_and_shared_expert_can_be_priced_apart_from_the_routed_experts():
    """An MoE's routed experts (mlp_in / mlp_out) at a fraction of the dense price (dense_mlp_in / dense_mlp_out): only
    the routed experts' products get cheaper, and Pythia's defaults are unchanged."""
    dense = {"qkv": 6.0, "attn_out": 6.0, "mlp_in": 6.0, "mlp_out": 6.0, "head": 6.0, "router": 6.0}
    moe = dict(dense, mlp_in=3.0, mlp_out=3.0, dense_mlp_in=6.0, dense_mlp_out=6.0)
    p = P.prices(moe)
    assert p["expert_in"] == p["expert_out"] == 3.0
    assert p["dense_in"] == p["dense_out"] == p["shared_in"] == p["shared_out"] == 6.0
    sh = P.shapes(M)
    routed = (sh["expert_in"][1] + sh["expert_out"][1]) / K.active_weights(M)
    assert P.mean_price(base(), p) == pytest.approx(6.0 * (1 - routed / 2), rel=1e-12)
    pyt = P.prices(P.PYTHIA["160m@100k"]["fp8"])
    assert pyt["dense_in"] == pyt["expert_in"] and pyt["shared_out"] == pyt["expert_out"]


def test_the_olmoe_price_file_prices_every_class_at_every_point(capsys):
    """`prices_olmoe.json` (OLMoE-1B-7B's measured routed-expert prices) loads, prices every K2 class above 0 at every
    point and format, gives each OLMoE point the uniform forms' cheapest tier, and runs through the report."""
    pts, cheapest = P.load_prices(str(Path(P.__file__).parent / "prices_olmoe.json"))
    assert {pt for pt in pts if pt.startswith("olmoe@")} <= set(cheapest) <= set(pts)
    for pt, tab in pts.items():
        assert set(tab) == set(P.FMTS)
        for fmt in P.FMTS:
            p = P.prices(tab[fmt])
            assert all(p[c] > 0 for c in P.CLASSES), (pt, fmt, p)
            assert pt not in cheapest or cheapest[pt]["1%"][fmt] > 0
    P.report(pts, cheapest, "prices_olmoe")
    assert "Priced form at 1 MB" in capsys.readouterr().out


def test_theta_codes_alone_cost_at_least_every_codec():
    """``PYTHIA_FWD`` is the least over a subset of ``PYTHIA``'s codecs."""
    for pt, fmts in P.PYTHIA.items():
        for fmt, tab in fmts.items():
            assert all(P.PYTHIA_FWD[pt][fmt][r] >= v for r, v in tab.items()), (pt, fmt)


def test_the_tiers_nest():
    """A smaller cheapest set costs less, Adam-v's deciles refine theta's classes, and theta's classes refine the
    named ones."""
    olmo2 = P.load_prices(str(Path(P.__file__).parent / "prices_olmo2.json"))
    olmoe = P.load_prices(str(Path(P.__file__).parent / "prices_olmoe.json"))
    for pts, cheapest in ((P.PYTHIA_FWD, P.CHEAPEST), olmo2, olmoe):
        for pt, tiers in cheapest.items():
            if "theta 10%" not in tiers:
                continue
            for fmt in P.FMTS:
                t = {n: v[fmt] for n, v in tiers.items()}
                assert t["theta 0.1%"] <= t["theta 1%"] <= t["theta 10%"] <= P.named(pts[pt][fmt]), (pt, fmt)
                for fr in ("0.1%", "1%", "10%"):
                    assert t.get(f"adam-v {fr}", 0) <= t[f"theta {fr}"]


def test_the_gain_is_the_mean_price_over_c_w():
    """At 1 MB both forms chain the same products, so the priced form's gain over the uniform one is the
    product-weighted mean price over the uniform c_w."""
    rows = P.source_rows("Pythia", P.PYTHIA_FWD, P.PYTHIA, P.CHEAPEST, base(), GATES, P.KS["1 MB"])
    for r in rows:
        for reg in ("weights", "optimizer"):
            for u in r[reg]["uniform"].values():
                assert u["gain"] == pytest.approx(r[reg]["mean_price"] / u["c_w"], rel=0.01)


def test_a_price_file_prices_each_registration(tmp_path):
    """``points_optimizer`` is read only for a registration that covers the optimizer state."""
    f = tmp_path / "p.json"
    w = {"q_proj": 3.0, "o_proj": 4.0, "gate_proj": 5.0, "down_proj": 6.0}
    o = {k: v - 1 for k, v in w.items()}
    f.write_text(json.dumps({"points": {"x": {fmt: w for fmt in P.FMTS}},
                             "points_optimizer": {"x": {fmt: o for fmt in P.FMTS}}}))
    assert P.load_prices(str(f))[0]["x"]["fp8"]["qkv"] == 3.0
    assert P.load_prices(str(f), "optimizer")[0]["x"]["fp8"]["named"] == 2.0


def test_the_olmo2_price_file_and_the_baselines_report(capsys):
    """`prices_olmo2.json` holds six points with every tier, its optimizer prices are at most its weights' prices, and
    the baselines report runs over it and the OLMoE file beside Pythia."""
    f = str(Path(P.__file__).parent / "prices_olmo2.json")
    pw, cheapest = P.load_prices(f)
    po, _ = P.load_prices(f, "optimizer")
    assert len(pw) == 6 and set(cheapest) == set(pw)
    for pt in pw:
        for fmt in P.FMTS:
            assert all(po[pt][fmt][r] <= v for r, v in pw[pt][fmt].items())
    assert P.main(["--baselines", "--prices", f, "--prices", str(Path(P.__file__).parent / "prices_olmoe.json")]) == 0
    out = capsys.readouterr().out
    assert "olmo2-13b@late" in out and "olmoe@1150k" in out and "pythia-160m" not in out
