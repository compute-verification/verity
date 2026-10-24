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
    """The covering bound write-up's marginal-dampener table, Kimi K2 raw at k = 2e7, w = 2.5: 41, 166, 660."""
    d = P.check_daniel()
    assert d["base"] == pytest.approx(41, rel=0.03)
    assert d["additive"] == pytest.approx(166, rel=0.03)
    assert d["a=4"] == pytest.approx(660, rel=0.03)


def test_the_realism_notes_rows():
    """note:training-20261022T1045Z-cover-assumptions-realism, section 4, at 1 MB: one uniform c at the cheapest 1%
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
    assert roles == {"qkv": 4.0, "attn_out": 6.0, "mlp_in": 3.0, "mlp_out": 3.5, "head": 1.0}
    p = P.prices(roles)
    assert p["expert_in"] == 3.0 and p["mla_out"] == 6.0 and p["router"] == 1.0 and p["norm"] == 0.0
    assert cheapest["x"]["1%"]["bf16"] == 1.0
