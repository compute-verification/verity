import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import frontier as F  # noqa: E402
import inputs as I  # noqa: E402
from model import (SHAPES, Params, Shape, Workload, _timing, at_width, decode, evaluate, gamma_cap,  # noqa: E402
                   gamma_pass, margins, pad, step)


NO_EPHEMERAL = {name: (0.0, 0.0) for name in SHAPES}
P512 = {"bits": 512, "masked": False}


def _eta0(inputs=I.INPUTS) -> dict:
    return I.values(F.with_eta(inputs, 0.0))


def _measured_step() -> dict:
    """ra-codec-tensor's step at p512: one GPU, bf16 KV, no KV receipt, no ephemeral traffic, no prompt to encode, at
    the weight path those runs measured (17:58Z and 19:05Z)."""
    v = I.values()
    v.update(P512, tau_kv={2: 0.0, 3: 0.0}, eph_read=NO_EPHEMERAL, eph_write=NO_EPHEMERAL, t_gap=0.0, p_over_o=0.0,
             o_w={2: ((8, 0.36), (32, 0.73)), 3: ((8, 0.97), (32, 1.00))},
             tau_w={2: ((8, 0.232), (32, 0.70)), 3: ((8, 0.22), (32, 0.71))})
    return v


@pytest.mark.parametrize("name, batch, ctx, k, measured", [
    ("8B", 32, 20608, 2, 1.330),  # r20261012-175841-5578
    ("8B", 32, 20608, 3, 1.834),
    ("8B", 80, 8256, 2, 1.275),  # r20261012-175906-b479
    ("70B", 8, 9280, 2, 1.484),  # r20261012-190554-dd7c, the receipt re-keyed every pass on sketch warps
    ("70B", 8, 9280, 3, 2.030),
])
def test_reproduces_the_measured_step(name, batch, ctx, k, measured):
    s = SHAPES[name]
    bf16_kv = Shape(s.name, s.layers, s.hidden, s.inter, s.linear_params, kv_heads=2 * s.kv_heads)
    st = step(Workload(bf16_kv, batch, ctx, tp=1), Params(k, 4096, 16, 0), _measured_step())
    assert st["cap"] / st["plain"] == pytest.approx(measured, abs=0.10)


def test_plain_step_is_the_measured_passes():
    v = I.values()
    v.update(eph_read=NO_EPHEMERAL, eph_write=NO_EPHEMERAL)
    st = step(Workload(SHAPES["70B"], 8, 9280, tp=1), Params(2, 4096, 16, 0), v)
    assert 1.03 <= st["plain"] * v["R"] * 1e9 / (st["W"] + st["KV"]) <= 1.10


@pytest.mark.parametrize("name, batch, read_mb, write_share", [
    ("8B", 1, 14.0, 0.016),  # ra-ephemeral's fused arm, r20261012-184939-520e, full depth
    ("8B", 64, 33.0, 0.010),
    ("70B", 1, 41.0, 0.007),
    ("70B", 64, 164.0, 0.008),
])
def test_ephemeral_traffic_is_the_measured_bytes(name, batch, read_mb, write_share):
    v = I.values()
    s = SHAPES[name]
    st = step(Workload(s, batch, 2048, tp=1), Params(3, 4096, 16, 0), v)
    kv_append = batch * s.kv_bytes_per_token
    reads = s.weight_bytes + batch * 2048 * s.kv_bytes_per_token + read_mb * 1e6
    assert st["eph_read"] / 1e6 == pytest.approx(read_mb, rel=0.15)
    assert (st["eph_write"] - kv_append) / reads == pytest.approx(write_share, rel=0.15)


def test_the_unregistered_reads_are_ra_ephemerals_mu_eph():
    v = I.values()
    for name, batch, mu in (("70B", 16, 0.00061), ("8B", 64, 0.00162)):  # its worst timed fused points, at 2k
        st = step(Workload(SHAPES[name], batch, 2048, tp=1), Params(3, 4096, 16, 0), v)
        assert st["eph_read"] / (v["R"] * 1e9 * st["plain"]) == pytest.approx(mu, rel=0.35)


def test_the_stated_codec_is_masked_p1024_at_three_rounds_with_two_the_fallback():
    v = I.values()
    assert v["bits"] == 1024 and v["masked"] and v["capacity"] == "hbm" and not v["bound"]
    assert F.STATED == "k3m" and F.ROUNDS == {"k3m": (3,), "k2m": (2,)} and F.BUDGET == 3.0
    assert v["eta_counted"] and v["n_session"] == 1000 and v["horizon"] == 600 and v["lambda_named"]


def test_conservation_over_hbm_has_no_host_link_or_identity_term():
    v = _eta0()
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(3, 4096, 16, 0)
    st = step(w, p, v)
    terms = {n for n in gamma_cap(w, p, v, st) if n.startswith("mu_")} | {"lambda"}
    assert terms == {"mu_chip", "mu_rebuild", "mu_partial", "mu_unreg", "mu_timing", "mu_eta", "mu_sample",
                     "mu_serve", "mu_sketch", "lambda"}
    assert "mu_link_n" not in evaluate(w, p, v)


@pytest.mark.parametrize("bits, stated, conservative, x1_adv", [
    (512, 1.567, 1.733, 111.7), (768, 1.562, 1.695, 216.8), (1024, 1.597, 1.885, 375.7)])
def test_the_rebuild_is_ra_squaring_ceilings_wide_adversaries(bits, stated, conservative, x1_adv):
    """`wide.adversaries(b)` on cursor/ra-squaring-ceiling-e3cd at 687c86282 (STEERING 21:22Z): the stated factor
    s·(1 + h), the conservative (Karatsuba) one, and x₁,adv at the stated."""
    v = {**I.values(), "bits": bits}
    factor = at_width(v, "s_adv") * (1 + at_width(v, "h_host_sq"))
    assert factor == pytest.approx(stated, abs=0.002)
    assert at_width({**v, "s_adv": I.RISKS["s_adv"][0]}, "s_adv") * (1 + at_width(v, "h_host_sq")) == pytest.approx(
        conservative, abs=0.002)
    assert at_width(v, "x1") / factor == pytest.approx(x1_adv, abs=0.2)
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(3, 4096, 16, 0)
    assert gamma_cap(w, p, v, step(w, p, v))["mu_rebuild"] == pytest.approx(factor / (3 * at_width(v, "x1")))


def test_gamma_terms_match_the_red_team():
    v = I.values()
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(2, 4096, 16, 0)
    st = step(w, p, v)
    g = gamma_pass(w, p, v, st)
    assert g["mu_chip"] == pytest.approx(263.2e6 / (st["W"] + st["KV"]))
    assert gamma_cap(w, p, v, st)["mu_chip"] == pytest.approx(263.2e6 / (1540e9 * st["cap"]))
    assert g["mu_rebuild"] == pytest.approx(g["d"] * 1.420 * 1.125 / (2 * 600))
    plain = {**v, **P512}
    assert gamma_pass(w, p, plain, step(w, p, plain))["mu_rebuild"] == pytest.approx(
        gamma_pass(w, p, plain, step(w, p, plain))["d"] * 1.388 * 1.129 / (2 * 175))
    assert g["mu_sketch"] == 2.0 ** -16
    n = 8 * v["H"] / 4096
    assert g["lambda"] == pytest.approx((n + 40) / (8 * 4096 * n))
    assert g["lambda"] == pytest.approx(0.00003, rel=0.02)  # ra-openings: named λ, 0.003% a pass at 4 KiB
    crude = gamma_pass(w, p, {**v, "lambda_named": False}, st)["lambda"]
    assert crude == pytest.approx((64 + math.log2(math.e) - math.log2(n)) / (8 * 4096))


def test_the_hybrid_draws_enough_openings_that_every_window_is_within_one_percent():
    v = I.values()
    w = Workload(SHAPES["8B"], 16, 8192)
    p = Params(2, 2048, 16, 0)
    g = evaluate(w, p, v)["cap"]
    assert g["openings"] == pytest.approx(40 * math.log(2) / 0.01) and g["window_sample"] == pytest.approx(0.01)
    assert g["mu_sample"] == pytest.approx(40 * math.log(2) / (g["openings"] * 600))
    # ra-openings §1: 2,761 a window at 4,522 B an opening is 12.5 MB/s
    assert g["openings"] * (2048 + v["opening_bytes"]) / 1e6 == pytest.approx(12.5, rel=0.02)
    # a mean over 10 minutes alone: 31 a window give 0.15%
    h = evaluate(w, p, {**v, "window_max": 40 * math.log(2) / 31})["cap"]
    assert h["mu_sample"] == pytest.approx(0.0015, rel=0.01)
    # a verifier link too slow for the hybrid's openings leaves the window's sampling above 1%: never a frontier point
    slow = {**_eta0(), "v_budget": 0.05}
    assert evaluate(w, p, slow)["cap"]["window_sample"] > 0.01
    assert not F.best(w, slow, "cap", (3,))["feasible"]


def test_binding_costs_only_when_bound():
    v = I.values()
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(3, 4096, 16, 0)
    unbound, bound = step(w, p, v), step(w, p, {**v, "bound": True})
    assert bound["cap"] > unbound["cap"] and bound["plain"] == unbound["plain"]
    # an unbound receipt's opening carries no z, so a tight link fits more of them
    tight = {**v, "v_budget": 0.1}
    assert evaluate(w, p, tight)["cap"]["openings"] > evaluate(w, p, {**tight, "bound": True})["cap"]["openings"]
    assert F.best(w, _eta0(), "cap", (3,))["params"]["binding"] == "unbound"


def test_cheaper_decode_never_slows_the_step():
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0)
    today = step(w, p, I.values())
    better = step(w, p, I.values(I.improved("o_w")))
    assert better["cap"] < today["cap"] and better["plain"] == today["plain"]


def test_at_512_bits_the_decode_is_as_measured_and_at_1024_it_is_ra_codec_mixs_attention():
    v = {**I.values(), **P512}
    for k, costs in ((2, (0.159, 0.286, 0.348)), (3, (0.65, 0.806, 0.884))):
        for o in costs:
            assert decode(o, k, v) == pytest.approx(o)
        # no form decodes below the standalone floor
        assert decode(0.0, k, v) == pytest.approx(v["decode_floor"][k])
    # ra-codec-mix's masked p1024 attention at 8B B80 (r20261012-190027-eaee)
    wide = I.values()
    for k, measured in ((2, 0.989), (3, 1.829)):
        o = F.interp(wide["o_kv"][k], 80)
        assert decode(o, k, wide) == pytest.approx(measured, abs=0.01)
    # a faster squaring cuts the wide decode, and the wide word decodes a byte dearer than p512 at equal k
    fast = {**wide, "sq_speed": I.INPUTS["sq_speed"].improved}
    assert decode(0.651, 3, fast) < decode(0.651, 3, wide)
    assert decode(0.651, 3, {**wide, "masked": False}) > 0.651


@pytest.mark.parametrize("bits, fplll, centre, low", [(512, 12.3, 1.2, 0.75), (640, 8.7, 0.8, 0.50),
                                                      (768, 6.7, 0.6, 0.37)])
def test_margins_are_the_red_teams_table(bits, fplll, centre, low):
    m = margins(bits)
    assert m["fplll"] == pytest.approx(fplll, rel=0.03)
    assert m["floor_centre"] == pytest.approx(centre, rel=0.05)
    assert m["floor_low"] == pytest.approx(low, rel=0.05)


def test_partial_completion_divides_the_rebuild_by_theta():
    v = {**I.values(), **P512}
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0)
    g = gamma_cap(w, p, v, step(w, p, v))
    assert g["mu_partial"] == pytest.approx(g["mu_rebuild"] / 3)
    v["theta"] = 1.0
    assert gamma_cap(w, p, v, step(w, p, v))["mu_partial"] == 0.0


def test_the_mask_costs_its_xor_and_removes_partial_completion():
    v = {**I.values(), **P512}
    masked = {**v, "masked": True}
    for k, o in ((2, 0.160), (3, 0.650)):
        assert decode(o, k, masked) == pytest.approx(o + v["mask_cost"][k])
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0)
    assert gamma_cap(w, p, v, step(w, p, v))["mu_partial"] > 0.0
    assert gamma_cap(w, p, masked, step(w, p, masked))["mu_partial"] == 0.0
    # at 640 bits plain k = 2 is past parity (θ < 1), and the mask holds
    wide = {**v, "bits": 640, "theta": 1.0}
    assert gamma_cap(w, p, wide, step(w, p, wide))["mu_partial"] > 0.0
    assert gamma_cap(w, p, {**wide, "masked": True}, step(w, p, wide))["mu_partial"] == 0.0


def test_eta_counts_its_phases_against_r_t():
    v = I.values()
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(3, 65536, 12, 0)
    st = step(w, p, v)
    declared = gamma_cap(w, p, {**v, "eta_counted": False}, st)
    against = gamma_cap(w, p, v, st)
    phases = v["eta_decode"] + v["eta_prefill"] + v["eta_gaps"] + v["eta_idle"]
    assert declared["mu_eta"] == 0.0 and against["gamma"] == pytest.approx(declared["gamma"] + phases)
    assert against["eta_allowance"] == pytest.approx(declared["eta_allowance"])
    assert phases > 0.25  # stock vLLM, ra-kv-filler trace 1
    assert not F.best(w, v, "cap", (3,))["feasible"]
    assert F.best(w, I.values(I.at_floor()), "cap", (3,))["feasible"]
    budget = I.values(F.with_eta(I.INPUTS, 0.0025))
    assert sum(budget[n] for n in F.ETA_PHASES) == pytest.approx(0.0025)
    assert all(F.with_eta(I.INPUTS, 0.0)[n].improved is None for n in F.ETA_PHASES)


def test_where_nothing_is_feasible_a_lever_is_scored_by_the_least_gamma():
    w = Workload(SHAPES["70B"], 16, 8192)
    today = F.best(w, I.values(), "cap", (3,))
    better = F.best(w, I.values(I.improved("eta_decode")), "cap", (3,))
    assert not today["feasible"] and today["gamma_least"] == today["gamma"]
    c = F.compare([today], [better])
    assert c["median_gain"] == 0.0 and c["gamma_drop"] == pytest.approx(0.190 - 0.05)


def test_timing_is_the_drivers():
    v = {**I.values(), "n_session": 100}
    # ra-remote-driver: 20 ms passes, a 100 µs gap, RTT 20 ms, 100 windows of 1 s: 0.52% (its formula, 0.54%)
    assert _timing(0.0201, 100e-6, v) == pytest.approx(0.0054, rel=0.02)
    assert _timing(0.0200, 0.0, v) == pytest.approx(0.0004)
    assert _timing(0.0200, 0.0, I.values()) == pytest.approx(0.00004)  # the default session, 1,000 windows


def test_chip_state_is_paid_per_pass_and_padding_spends_gamma_exactly():
    v = _eta0(I.all_improved())
    short, long = Workload(SHAPES["8B"], 1, 2048), Workload(SHAPES["70B"], 16, 8192)
    p = Params(3, 65536, 12, 0)
    a, b = step(short, p, v), step(long, p, v)
    ga, gb = gamma_cap(short, p, v, a), gamma_cap(long, p, v, b)
    assert ga["mu_chip"] * a["cap"] == pytest.approx(gb["mu_chip"] * b["cap"]) and ga["mu_chip"] > gb["mu_chip"]
    padded = pad(short, p, v, a, 0.01)
    assert padded["pad"] > 0 and gamma_cap(short, p, v, padded)["gamma"] == pytest.approx(0.01, abs=1e-9)


def test_copies_spread_the_carried_state_over_their_steps_where_they_fit():
    v = _eta0(I.all_improved())
    w = Workload(SHAPES["8B"], 1, 2048)
    one, four = Params(3, 65536, 12, 0), Params(3, 65536, 12, 0, 4)
    a, b = step(w, one, v), step(w, four, v)
    ga, gb = gamma_cap(w, one, v, a), gamma_cap(w, four, v, b)
    assert gb["mu_chip"] * b["cap"] == pytest.approx(ga["mu_chip"] * a["cap"] / 4)
    assert gb["mu_unreg"] * b["cap"] == pytest.approx(ga["mu_unreg"] * a["cap"])
    with_copies, without = F.best(w, v, "cap", (3,)), F.best(w, {**v, "copies": 1}, "cap", (3,))
    assert with_copies["params"]["copies"] > 1 and with_copies["overhead"] < without["overhead"]
    assert F.best(Workload(SHAPES["405B"], 1, 2048), v, "cap", (3,))["params"]["copies"] == 1


def test_frontier_points_respect_gamma_and_flag_the_budget():
    v = _eta0(I.all_improved())
    for w in (Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 4, 2048)):
        r = F.best(w, F.codec(v, F.STATED), "cap", F.ROUNDS[F.STATED])
        assert r["feasible"] and r["gamma"] <= 0.01 + 1e-12
        assert r["params"]["width_bits"] == 1024 and r["params"]["masked"] and r["params"]["k"] == 3
        assert r["within_budget"] == (r["overhead"] <= F.BUDGET)
        for rounds, ks in F.ROUNDS.items():
            assert F.best(w, F.codec(v, rounds), "cap", ks)["params"]["k"] in ks, rounds


def test_the_floor_is_never_slower_than_the_projection():
    projected, floor = _eta0(I.all_improved()), I.values(I.at_floor())
    for w in (Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 64, 2048), Workload(SHAPES["405B"], 4, 8192)):
        a, b = F.best(w, projected, "cap", (3,)), F.best(w, floor, "cap", (3,))
        assert b["feasible"] and b["overhead"] <= a["overhead"] + 1e-12


def test_every_lever_names_its_work_and_source():
    for name in I.levers():
        i = I.INPUTS[name]
        assert i.lever and i.source, name
    assert set(I.levers()) <= set(I.INPUTS)
    assert set(I.RISKS) <= set(I.INPUTS) and set(I.FLOOR) <= set(I.INPUTS)
