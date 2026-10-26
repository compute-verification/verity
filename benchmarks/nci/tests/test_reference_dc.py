import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reference_dc as M  # noqa: E402
import tiers as T  # noqa: E402

P, FLEETS = M.load()
H200, NVL72 = FLEETS["h200-server"], FLEETS["nvl72"]


def test_every_parameter_has_a_value_a_unit_and_a_source_and_sits_in_its_range():
    raw = json.loads(M.PARAMS.read_text())
    specs = list(raw["parameters"].items())
    specs += [(f"{f}.{k}", v) for f, spec in raw["fleets"].items() for k, v in spec.items() if isinstance(v, dict)]
    for name, spec in specs:
        assert {"value", "unit", "source"} <= set(spec) and spec["source"], name
        if "range" in spec:
            low, high = spec["range"]
            assert spec["range_source"], name
            assert spec["value"] is None or low <= spec["value"] <= high, name


def test_the_spec_sanity_numbers():
    s = M.sanity(P, FLEETS)
    assert s["committed_flop_year"] == pytest.approx(8.8e30, rel=0.01)
    assert s["h100_hours"] == pytest.approx(1.2e12, rel=0.04)
    assert s["h100_equivalents"] == pytest.approx(1.4e8, rel=0.01)
    assert s["h200_servers"] == pytest.approx(1.8e7, rel=0.02)
    assert s["nvl72_racks"] == pytest.approx(7.8e5, rel=0.01)
    assert s["sync_years_at_1kbps"] == pytest.approx(4.3, rel=0.01)
    assert s["run_flop"] == pytest.approx(9.7e24, rel=0.01)
    assert s["step_h100_hours"] == pytest.approx(4, rel=0.03)
    assert s["run_years_h200_island"] == pytest.approx(19, rel=0.03)
    assert s["run_days_nvl72_island"] == pytest.approx(310, rel=0.01)
    assert M.h100_hour(P) == pytest.approx(7.1e18, rel=0.01)


def test_thread_counts_two_multiplications_a_product_as_tiers_lam_does():
    assert P["thread_c_reuse"] == 2
    assert T.lam({"e": 1.0, "d": 1.0, "b_w": 1.0}, 40.0) == 2 * 40.0 / 4


def test_certified_ingress_is_the_window_compute_over_phi_F_whatever_the_request_rate():
    for f in FLEETS.values():
        for rate in (5e15, 2e16, 8e16):
            b = M.block(P | {"requests_per_year": rate}, f)
            assert b["I_cert"] == pytest.approx(b["C"] / M.phi_F(P), rel=1e-12)
    assert M.phi_F(P) == pytest.approx(2 * 49e9 / 17)
    assert M.phi_F(P | {"phi_F_mode": "traffic"}) == 6.0e9


def test_fan_is_linear_in_K_on_the_products_not_token_reachable():
    for K in (1e6, 2e7, 1e9):
        a = M.allowance(P, "fan", K)
        assert a["q"] == pytest.approx(P["fan_e"] * K / P["fan_beta_x"] / (1 - P["fan_x_lead"]))
        assert a["regime"] == "linear" and a["premise"] is None
    assert M.allowance(P, "fan", 4e9)["q"] == pytest.approx(2 * M.allowance(P, "fan", 2e9)["q"])


def test_thread_is_linear_below_its_knee_and_quadratic_above():
    knee = 4 * P["thread_d"] * P["thread_b_w"] * P["thread_e"] / P["thread_c_reuse"]
    below, above = M.allowance(P, "thread", knee / 2), M.allowance(P, "thread", 2e7)
    assert below["knee_bits"] == pytest.approx(knee)
    assert below["regime"] == "linear" and below["Lambda"] == P["thread_e"]
    assert below["q"] == pytest.approx(knee / 2 * P["thread_e"] / P["thread_b_a"])
    lam = P["thread_c_reuse"] * 2e7 / (4 * P["thread_d"] * P["thread_b_w"])
    assert above["regime"] == "quadratic" and above["Lambda"] == pytest.approx(lam)
    assert above["products"] == pytest.approx(2e7 * lam / P["thread_b_a"] / P["thread_c_reuse"])


def test_cover_is_u_times_k_squared_over_b_plus_k():
    for K in (2e7, 5.57e11):
        q = M.allowance(P, "cover", K)["q"]
        assert q == pytest.approx(P["cover_unit_products"] * (K * K / P["cover_b"] + K))


def test_rho_at_the_inference_block():
    c = P["k_inference"] * M.phi_F(P)
    fan = P["fan_e"] * P["k_inference"] / P["fan_beta_x"] / (1 - P["fan_x_lead"])
    assert M.rho(P, c, M.allowance(P, "fan", P["k_inference"])) == pytest.approx(c / (P["kappa_F"] * fan))
    assert M.rho(P, c, M.allowance(P, "fan", P["k_inference"])) == pytest.approx(1.45e4, rel=0.01)


def test_side_channels_take_over_an_h200_server_day_near_6_Mbps():
    b = M.block(P, H200)
    assert b["beta_takeover"] == pytest.approx(6.45e6, rel=0.01)
    at = M.block(P | {"beta_side": b["beta_takeover"]}, H200)
    assert at["side"] == pytest.approx(at["R"] + at["I_cert"])


def test_the_one_block_regime_at_server_day_scale():
    for f, steps in ((H200, 49), (NVL72, 1120)):
        ob = M.one_block(P, f)
        assert ob["server_window"]["steps"] == steps and ob["server_window"]["binding"] == "replay"
        assert ob["inference"]["steps"] == 0


def test_remote_is_eps_w_bound_per_window():
    w = M.window(P, H200, "fan", 1e-6)
    u = P["eps_w"] + P["p_window_fail"] / 1e-6
    assert w["full"] / w["cut_b"] == pytest.approx(1 / ((1 - u) / w["rho_inference"] + u), rel=1e-9)


@pytest.mark.parametrize("fleet", sorted(FLEETS))
def test_onsite_never_beats_remote_and_U_grows_as_delta_shrinks(fleet):
    f = FLEETS[fleet]
    for H in P["horizons"].values():
        for cls in M.CLASSES:
            loose, tight = (M.period(P, f, cls, H, d, "remote") for d in (1e-2, 1e-6))
            assert tight["u"]["total"] > loose["u"]["total"]
            assert tight["u"]["slack"] == pytest.approx(P["p_window_fail"] / 1e-6)
            for d in P["deltas"]:
                on, rm = (M.period(P, f, cls, H, d, m) for m in ("onsite", "remote"))
                assert on["train_flop"] <= rm["train_flop"] * (1 + 1e-12)
                assert on["coordinated_t_run"] >= rm["coordinated_t_run"]


def test_an_nvl72_day_holds_the_whole_update_so_onsite_is_remote_there():
    b = M.block(P, NVL72)
    assert b["K"] > M.whole_update_bits(P)
    for cls in M.CLASSES:
        a = M.allowance(P, cls, b["K"])
        assert a["products"] is None and a["premise"].startswith("fails")
        on, rm = (M.period(P, NVL72, cls, M.YEAR, 1e-6, m) for m in ("onsite", "remote"))
        assert on["train_flop"] == pytest.approx(rm["train_flop"])
    hour = {(x["R_scope"], x["window_s"]): x for x in M.scenarios(P, NVL72)}[("location", 3600.0)]
    assert hour["K"] < M.whole_update_bits(P) and hour["fan"]["premise"] is None


def test_fan_onsite_is_the_server_window_rho_on_h200():
    m = M.period(P, H200, "fan", M.YEAR, 1e-6, "onsite")
    assert m["binding"] == "a"
    assert m["D"] == pytest.approx(M.window(P, H200, "fan", 1e-6)["rho_server_window"], rel=1e-6)


def test_islands_memory_and_cooperation():
    h, n = M.island(P, H200), M.island(P, NVL72)
    assert h["fits"] == "HBM + host + NVMe" and n["fits"] == "HBM + host"
    assert not h["offload_binds"] and not n["offload_binds"]
    d = M.dark(P, H200, 0.01, M.YEAR)
    assert d["locations"] == pytest.approx(0.01 / 0.99 * M.servers(P, H200))
    assert d["coop_t_run"] == pytest.approx(P["coop_syncs"] * P["exfil_sync_bits"] / P["beta_side"])
    at = M.dark(P | {"beta_side": d["coop_beats_island_above_bps"]}, H200, 0.01, M.YEAR)
    assert at["coop_t_run"] == pytest.approx(h["t_run"])


def test_exfiltration_closed_forms():
    e = M.exfil(P, H200, 0.01)
    assert e["weights"]["per_location"] / M.YEAR == pytest.approx(203, rel=0.01)
    assert e["rollouts"]["per_location"] / 86400 == pytest.approx(1.97, rel=0.01)
    n = P["exfil_weights_bits"] / P["sampled_step_bits"]
    assert e["weights"]["faulty_outputs_p_detect"] == pytest.approx(1 - math.exp(-M.sampled_q(P) * n))
    inj = M.injected(P, "fan", 1e-6)
    assert inj["bits"] == pytest.approx(math.log(1e6) * (1 / M.sampled_q(P) + 1) * P["sampled_step_bits"])


def test_the_tables_render():
    r = M.evaluate(P, FLEETS)
    md = M.render(P, FLEETS, r)
    assert "## Headline" in md and "## Caveats" in md and "not that dark training is efficient" in md
    assert "gates is a later pass" in md
    json.dumps(M._json(r))
