import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import frontier as F  # noqa: E402
import inputs as I  # noqa: E402
from model import SHAPES, Params, Shape, Workload, evaluate, gamma_cap, gamma_pass, link_n, step  # noqa: E402


NO_EPHEMERAL = {name: (0.0, 0.0) for name in SHAPES}


def _measured_step() -> dict:
    """ra-codec-tensor's step: one GPU, bf16 KV, no KV receipt, no binding, no host tier, no ephemeral traffic."""
    v = I.values()
    v.update(tau_kv={2: 0.0, 3: 0.0}, tau_bind=0.0, eph_read=NO_EPHEMERAL, eph_write=NO_EPHEMERAL)
    return v


@pytest.mark.parametrize("name, batch, ctx, k, measured", [
    ("8B", 32, 20608, 2, 1.330),  # r20261012-175841-5578
    ("8B", 32, 20608, 3, 1.834),
    ("8B", 80, 8256, 2, 1.275),  # r20261012-175906-b479
    ("70B", 8, 9280, 2, 1.679),  # r20261012-175923-0080
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


@pytest.mark.parametrize("name, batch, read_mb, write_mb", [
    ("8B", 1, 15.2, 108.9),  # r20261012-181016-6cbb, full depth
    ("8B", 64, 94.5, 469.3),
    ("70B", 1, 48.0, 492.2),  # r20261012-181715-5045, scaled to 80 layers
    ("70B", 64, 712.8, 2267.3),
])
def test_ephemeral_traffic_is_the_measured_bytes(name, batch, read_mb, write_mb):
    v = I.values()
    st = step(Workload(SHAPES[name], batch, 2048, tp=1), Params(3, 4096, 16, 0, 0.0), v)
    kv_append = batch * SHAPES[name].kv_bytes_per_token
    assert st["eph_read"] / 1e6 == pytest.approx(read_mb, rel=0.12)
    assert (st["eph_write"] - kv_append) / 1e6 == pytest.approx(write_mb, rel=0.12)


def test_gamma_terms_match_the_red_team():
    v = I.values()
    w = Workload(SHAPES["70B"], 8, 8192)
    p = Params(2, 4096, 16, 0, 1.0)
    st = step(w, p, v)
    g = gamma_pass(w, p, v, st)
    assert g["mu_chip"] == pytest.approx(263.2e6 / (st["W"] + st["KV"]))
    assert gamma_cap(w, p, v, st)["mu_chip"] == pytest.approx(0.0028 * 8 * 1540 / (8 * 1540 + 372), abs=1e-4)
    assert g["mu_rebuild"] == pytest.approx(g["d"] * 1.2 * 1.003 / (2 * 175))
    assert g["mu_sketch"] == 2.0 ** -16
    n = 8 * v["H"] / 4096
    assert g["lambda"] == pytest.approx((64 + math.log2(math.e) - math.log2(n)) / (8 * 4096))
    none = Params(2, 4096, 16, 0, 0.0)
    assert gamma_cap(w, none, v, step(w, none, v))["mu_host"] == pytest.approx(372 / (8 * 1540 + 372))
    assert gamma_cap(w, p, v, st)["mu_host"] == 0.0


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
    # k-slice 32 at m = 64 has 8× the weights in z: the host blocks bind up to what PCIe carries down (451 GB/s).
    wide = {**v, "b_pcie": 1e6}
    unbound = link_n(w, sliced, v, st) - link_n(w, sliced, wide, step(w, sliced, wide))
    assert unbound == pytest.approx((819 - 8 * 56.4) / sigma, rel=0.01)


def test_frontier_points_respect_gamma():
    v = I.values(I.all_improved())
    ws = [Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 4, 2048)]
    for w in ws:
        r = F.best(w, v, "cap", "C")
        assert r["feasible"] and r["gamma"] <= 0.01
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
