import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import frontier as F  # noqa: E402
import inputs as I  # noqa: E402
from model import SHAPES, Params, Shape, Workload, evaluate, gamma_cap, gamma_pass, step  # noqa: E402


def _measured_step() -> dict:
    """ra-codec-tensor's step: one GPU, bf16 KV, no KV receipt, no binding, no host tier, no activation traffic."""
    v = I.values()
    v.update(tau_kv={2: 0.0, 3: 0.0}, tau_bind=0.0, rho_l2=1.0)
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
    v["rho_l2"] = 1.0
    st = step(Workload(SHAPES["70B"], 8, 9280, tp=1), Params(2, 4096, 16, 0, 0.0), v)
    assert 1.03 <= st["plain"] * v["R"] * 1e9 / (st["W"] + st["KV"]) <= 1.10


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


def test_frontier_points_respect_gamma():
    v = I.values(I.all_improved())
    ws = [Workload(SHAPES["8B"], 16, 8192), Workload(SHAPES["70B"], 4, 2048)]
    for w in ws:
        r = F.best(w, v, "cap", "C")
        assert r["feasible"] and r["gamma"] <= 0.01
        assert r["params"]["width_bits"] == 512 and r["params"]["k"] in (2, 3)


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
