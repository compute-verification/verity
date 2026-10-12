import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import frontier as F  # noqa: E402
import inputs as I  # noqa: E402
from model import (SHAPES, Params, Shape, Workload, _timing, decode, evaluate, gamma_cap, gamma_pass,  # noqa: E402
                   link_n, margins, pad, sq_ratio, step)


NO_EPHEMERAL = {name: (0.0, 0.0) for name in SHAPES}


def _measured_step() -> dict:
    """ra-codec-tensor's step: one GPU, bf16 KV, no KV receipt, no binding, no host tier, no ephemeral traffic, no
    prompt to encode."""
    v = I.values()
    v.update(tau_kv={2: 0.0, 3: 0.0}, tau_bind=0.0, tau_bind_kv={2: 0.0, 3: 0.0}, eph_read=NO_EPHEMERAL,
             eph_write=NO_EPHEMERAL, t_gap=0.0, p_over_o=0.0)
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
    st = step(Workload(bf16_kv, batch, ctx, tp=1), Params(k, 4096, 16, 0, 0.0), _measured_step())
    assert st["cap"] / st["plain"] == pytest.approx(measured, abs=0.10)


def test_plain_step_is_the_measured_passes():
    v = I.values()
    v.update(eph_read=NO_EPHEMERAL, eph_write=NO_EPHEMERAL)
    st = step(Workload(SHAPES["70B"], 8, 9280, tp=1), Params(2, 4096, 16, 0, 0.0), v)
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
    st = step(Workload(s, batch, 2048, tp=1), Params(3, 4096, 16, 0, 0.0), v)
    kv_append = batch * s.kv_bytes_per_token
    reads = s.weight_bytes + batch * 2048 * s.kv_bytes_per_token + read_mb * 1e6
    assert st["eph_read"] / 1e6 == pytest.approx(read_mb, rel=0.15)
    assert (st["eph_write"] - kv_append) / reads == pytest.approx(write_share, rel=0.15)


def test_the_unregistered_reads_are_ra_ephemerals_mu_eph():
    v = I.values()
    for name, batch, mu in (("70B", 16, 0.00061), ("8B", 64, 0.00162)):  # its worst timed fused points, at 2k
        st = step(Workload(SHAPES[name], batch, 2048, tp=1), Params(3, 4096, 16, 0, 0.0), v)
        assert st["eph_read"] / (v["R"] * 1e9 * st["plain"]) == pytest.approx(mu, rel=0.35)


def test_gamma_terms_match_the_red_team():
    v = I.values()
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(2, 4096, 16, 0, 1.0)
    st = step(w, p, v)
    g = gamma_pass(w, p, v, st)
    sigma = 8 * 1540 + 819
    assert g["mu_chip"] == pytest.approx(263.2e6 / (st["W"] + st["KV"]))
    assert gamma_cap(w, p, v, st)["mu_chip"] == pytest.approx(263.2e6 / (1540e9 * st["cap"]) * 8 * 1540 / sigma)
    assert g["mu_rebuild"] == pytest.approx(g["d"] * 1.38 * 1.129 / (2 * 175))
    assert g["mu_sketch"] == 2.0 ** -16
    n = 8 * v["H"] / 4096
    assert g["lambda"] == pytest.approx((64 + math.log2(math.e) - math.log2(n)) / (8 * 4096))
    # ra-host-tier's loop covers 0.96 of the ceiling, and the GPUs' zero-copy stream adds only what exceeds it.
    none = Params(2, 4096, 16, 0, 0.0)
    for q in (none, p):
        assert gamma_cap(w, q, v, step(w, q, v))["mu_host"] == pytest.approx(0.04 * 819 / sigma)


def test_more_openings_with_a_bigger_budget_and_fewer_with_bigger_blocks():
    v = I.values()
    w = Workload(SHAPES["8B"], 16, 8192)
    small = evaluate(w, Params(2, 4096, 16, 0, 1.0), v)["cap"]
    big = evaluate(w, Params(2, 65536, 16, 0, 1.0), v)["cap"]
    assert big["openings"] < small["openings"] and big["lambda"] < small["lambda"]
    v["v_budget"] *= 10
    assert evaluate(w, Params(2, 4096, 16, 0, 1.0), v)["cap"]["mu_sample"] < small["mu_sample"]


def test_openings_cost_what_ra_openings_counts():
    v = I.values()
    v.update(x_from_transcript=True, v_budget=5959 * 18470 * 8 / 1e9)
    w = Workload(SHAPES["8B"], 16, 8192)
    p = Params(2, 4096, 16, 0, 1.0)
    g = evaluate(w, p, v)["cap"]
    assert g["openings"] == pytest.approx(18470, rel=0.01)
    assert g["mu_sample"] == pytest.approx(0.0015, rel=0.01)
    v.update(horizon=60, lambda_named=True)
    h = evaluate(w, p, v)["cap"]
    assert h["mu_sample"] == pytest.approx(g["mu_sample"] / 60) and h["lambda"] < g["lambda"] / 100


def test_cheaper_decode_never_slows_the_step():
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0, 1.0)
    today = step(w, p, I.values())
    better = step(w, p, I.values(I.improved("o_w")))
    assert better["cap"] < today["cap"] and better["plain"] == today["plain"]


def test_kv_relay_takes_the_whole_kv_share():
    v = I.values()
    w = Workload(SHAPES["8B"], 64, 32768)
    r = evaluate(w, Params(2, 8192, 16, 32, 1.0), v)
    assert r["mu_link_n"] > r["step"]["t_kv"] / r["step"]["cap"] * 0.9


def test_host_cpus_close_the_host_tier_and_f30_bounds_their_binding():
    v = I.values()
    v.update(host_cpu=1.0, b_host=819.0)
    w = Workload(SHAPES["8B"], 64, 8192)
    tile, sliced = Params(2, 8192, 16, 0, 0.0), Params(2, 8192, 16, 32, 0.0)
    st = step(w, sliced, v)
    assert st["host_cpu"] == 819.0 and gamma_cap(w, sliced, v, st)["mu_host"] == 0.0
    sigma = 8 * 1540 + 819
    host = 819 / sigma
    # Tile binding leaves no z beyond the weights' own, so every host block is relayable for free.
    assert link_n(w, tile, v, step(w, tile, v)) >= host
    # k-slice 32 at m = 64 has 8× the weights in z: the host blocks bind up to what PCIe carries down beside the
    # zero-copy stream (8 × 25.6 GB/s).
    wide = {**v, "b_pcie_down": 1e6}
    unbound = link_n(w, sliced, v, st) - link_n(w, sliced, wide, step(w, sliced, wide))
    assert unbound == pytest.approx((819 - 8 * 25.6) / sigma, rel=0.01)


def test_the_squaring_scales_by_ra_squaring_ceilings_cycle_model_and_512_is_as_measured():
    assert sq_ratio(512) == 1.0 and sq_ratio(512) * 712.6 == pytest.approx(713, abs=1)
    assert sq_ratio(384) < 0.6 < sq_ratio(448) < 1.0 < sq_ratio(640)
    v = I.values()
    for k, costs in ((2, (0.159, 0.36, 0.73, 0.97)), (3, (0.65, 0.97, 1.00))):
        for o in costs:
            assert decode(o, k, v) == pytest.approx(o)
        # no form decodes below the standalone floor
        assert decode(0.0, k, v) == pytest.approx(v["decode_floor"][k])
    # a round decodes a word with one squaring: narrower words decode a byte cheaper, by less than they rebuild
    narrow = {**v, "bits": 384}
    assert decode(0.159, 2, narrow) < 0.159
    assert sq_ratio(384) * 512 / 384 > sq_ratio(384)


@pytest.mark.parametrize("bits, fplll, centre, low", [(512, 12.3, 1.2, 0.75), (640, 8.7, 0.8, 0.50),
                                                      (768, 6.7, 0.6, 0.37)])
def test_margins_are_the_red_teams_table(bits, fplll, centre, low):
    m = margins(bits)
    assert m["fplll"] == pytest.approx(fplll, rel=0.03)
    assert m["floor_centre"] == pytest.approx(centre, rel=0.05)
    assert m["floor_low"] == pytest.approx(low, rel=0.05)


def test_partial_completion_divides_the_rebuild_by_theta():
    v = I.values()
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0, 0.0)
    g = gamma_cap(w, p, v, step(w, p, v))
    assert g["mu_partial"] == pytest.approx(g["mu_rebuild"] / 3)
    v["theta"] = 1.0
    assert gamma_cap(w, p, v, step(w, p, v))["mu_partial"] == 0.0


def test_the_mask_costs_its_xor_and_removes_partial_completion():
    v = I.values()
    masked = {**v, "masked": True}
    for k, o in ((2, 0.160), (3, 0.650)):
        assert decode(o, k, masked) == pytest.approx(o + v["mask_cost"][k])
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(2, 8192, 16, 0, 0.0)
    assert gamma_cap(w, p, v, step(w, p, v))["mu_partial"] > 0.0
    assert gamma_cap(w, p, masked, step(w, p, masked))["mu_partial"] == 0.0
    # at 640 bits plain k = 2 is past parity (θ < 1), and the mask holds
    wide = {**v, "bits": 640, "theta": 1.0}
    assert gamma_cap(w, p, wide, step(w, p, wide))["mu_partial"] > 0.0
    assert gamma_cap(w, p, {**wide, "masked": True}, step(w, p, wide))["mu_partial"] == 0.0


def test_a_relay_moves_half_the_distinct_z_bytes_as_ra_bound_receipt_bounds_it():
    # ra-bound-receipt §4: tile binding at m = 1 on 70B leaves 1/2,611 distinct z bytes a block byte
    assert 0.75 * 4 / 8192 == pytest.approx(1 / 2611, rel=0.05)
    v = I.values()
    w = Workload(SHAPES["70B"], 8, 8192)
    sliced = Params(2, 8192, 16, 32, 0.0)
    st = step(w, sliced, v)
    free = link_n(w, sliced, {**v, "b_nic": 1e-9}, st)
    kv = st["t_kv"] / st["cap"] * 8 * 1540 / (8 * 1540 + 819)
    # slice 32 at m = 8 has |z| = |E|: the NIC relays 1/(½ · 0.75) as many bound bytes as the trivial relay would
    bound = link_n(w, sliced, v, st) - free - kv
    trivial = link_n(w, sliced, {**v, "rho": 1.0, "z_entropy": 1.0}, st) - free - kv
    assert bound / trivial == pytest.approx(1 / 0.375, rel=0.01)


def test_the_cpus_tile_work_adds_to_gamma_when_counted():
    v = I.values()
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(3, 8192, 16, 0, 0.0)
    st = step(w, p, v)
    g = gamma_cap(w, p, v, st)["gamma"]
    v["mu_cpu"] = 0.005
    assert gamma_cap(w, p, v, st)["gamma"] == pytest.approx(g + 0.005)
    assert gamma_pass(w, p, v, st)["mu_cpu"] == 0.005
    v["mu_cpu"] = I.RISKS["mu_cpu"][0]
    assert gamma_cap(w, p, v, st)["mu_cpu"] == pytest.approx(0.006)  # ra-host-cpu's Pearl-C rate at batch 16


def test_eta_counts_its_phases_only_against_r_t():
    v = I.values(I.all_improved())
    w = Workload(SHAPES["70B"], 16, 8192)
    p = Params(3, 65536, 12, 0, 0.0)
    st = step(w, p, v)
    declared = gamma_cap(w, p, v, st)
    against = gamma_cap(w, p, {**v, "eta_counted": True}, st)
    phases = v["eta_decode"] + v["eta_prefill"] + v["eta_gaps"] + v["eta_idle"]
    assert declared["mu_eta"] == 0.0 and against["gamma"] == pytest.approx(declared["gamma"] + phases)
    assert against["eta_allowance"] == pytest.approx(declared["eta_allowance"])
    assert I.values()["eta_decode"] + I.values()["eta_idle"] > 0.25  # stock vLLM, ra-kv-filler trace 1
    assert not F.best(w, {**v, "eta_counted": True}, "cap", "C", (3,))["feasible"]
    assert F.best(w, {**I.values(I.at_floor()), "eta_counted": True}, "cap", "C", (3,))["feasible"]


def test_where_nothing_is_feasible_a_lever_is_scored_by_the_least_gamma():
    v = I.values()
    w = Workload(SHAPES["70B"], 16, 8192)
    today = F.best(w, v, "cap", "C", (3,))
    better = F.best(w, I.values(I.improved("host_cpu")), "cap", "C", (3,))
    assert not today["feasible"] and today["gamma_least"] == today["gamma"]
    c = F.compare([today], [better])
    # every parameter choice loses the 0.02 of host DRAM the better loop covers
    assert c["median_gain"] == 0.0 and c["gamma_drop"] >= 0.02 * 819 / (8 * 1540 + 819) - 1e-12


def test_timing_is_the_drivers():
    v = I.values()
    # ra-remote-driver: 20 ms passes, a 100 µs gap, RTT 20 ms, 100 windows of 1 s: 0.52% (its formula, 0.54%)
    assert _timing(0.0201, 100e-6, v) == pytest.approx(0.0054, rel=0.02)
    assert _timing(0.0200, 0.0, v) == pytest.approx(0.0004)


def test_chip_state_is_paid_per_pass_and_padding_spends_gamma_exactly():
    v = I.values(I.all_improved())
    short, long = Workload(SHAPES["8B"], 1, 2048), Workload(SHAPES["70B"], 16, 8192)
    p = Params(3, 65536, 12, 0, 0.0)
    a, b = step(short, p, v), step(long, p, v)
    ga, gb = gamma_cap(short, p, v, a), gamma_cap(long, p, v, b)
    assert ga["mu_chip"] * a["cap"] == pytest.approx(gb["mu_chip"] * b["cap"]) and ga["mu_chip"] > gb["mu_chip"]
    padded = pad(short, p, v, a, 0.01)
    assert padded["pad"] > 0 and gamma_cap(short, p, v, padded)["gamma"] == pytest.approx(0.01, abs=1e-9)


def test_copies_spread_the_carried_state_over_their_steps_where_they_fit():
    v = I.values(I.all_improved())
    w = Workload(SHAPES["8B"], 1, 2048)
    one, four = Params(3, 65536, 12, 0, 0.0), Params(3, 65536, 12, 0, 0.0, 4)
    a, b = step(w, one, v), step(w, four, v)
    ga, gb = gamma_cap(w, one, v, a), gamma_cap(w, four, v, b)
    assert gb["mu_chip"] * b["cap"] == pytest.approx(ga["mu_chip"] * a["cap"] / 4)
    assert gb["mu_unreg"] * b["cap"] == pytest.approx(ga["mu_unreg"] * a["cap"])
    with_copies, without = F.best(w, v, "cap", "C", (3,)), F.best(w, {**v, "copies": 1}, "cap", "C", (3,))
    assert with_copies["params"]["copies"] > 1 and with_copies["overhead"] < without["overhead"]
    assert F.best(Workload(SHAPES["405B"], 1, 2048), v, "cap", "C", (3,))["params"]["copies"] == 1


def test_frontier_points_respect_gamma():
    v = I.values(I.all_improved())
    ws = [Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 4, 2048)]
    for w in ws:
        r = F.best(w, v, "cap", "C")
        assert r["feasible"] and r["gamma"] <= 0.01 + 1e-12
        assert r["params"]["width_bits"] == 512 and r["params"]["k"] in (2, 3)
        for rounds, ks in F.ROUNDS.items():
            assert F.best(w, v, "cap", "C", ks)["params"]["k"] in ks, rounds


def test_the_floor_is_never_slower_than_the_projection():
    projected, floor = I.values(I.all_improved()), I.values(I.at_floor())
    for w in (Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 64, 2048), Workload(SHAPES["405B"], 4, 8192)):
        a, b = F.best(w, projected, "cap", "C"), F.best(w, floor, "cap", "C")
        assert b["feasible"] and b["overhead"] <= a["overhead"] + 1e-12


def test_every_lever_names_its_work_and_source():
    for name in I.levers():
        i = I.INPUTS[name]
        assert i.lever and i.source, name
    assert set(I.levers()) <= set(I.INPUTS)
