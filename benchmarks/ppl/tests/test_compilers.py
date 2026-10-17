"""PPL-COMPILE's steps: the formats' plain quantization (`fp`), Dither's coins on a format's integer grid (`native`) and
int8 run exactly by limb splitting (`emulate`)."""
import numpy as np
import pytest
import torch

import spec
import steps
from test_engine import evaluate, passed, ppl

C1 = ["pc-c1-e4m3-k3", "pc-c1-e4m3-k4", "pc-c1-fp4-k9"]
OTHERS = ["pc-int8", "pc-fp8-tensor", "pc-fp8-channel", "pc-mxfp8", "pc-nvfp4", "pc-mxfp4", "pc-c2-int5-2s",
          "pc-c2-int5-1s", "pc-c2-e4m3i-2s", "pc-c2-e4m3i-t-2s", "pc-c2-e4m3i-1s", "pc-c2-mx8-2s", "pc-c2-mx8-1s",
          "pc-c2-nvfp4-2s", "pc-c2-nvfp4-1s", "pc-c2-mxfp4-2s", "pc-c2-mxfp4-1s", "pc-c3-fp8", "pc-c3-fp4"]


def test_e4m3_integers_are_every_integer_e4m3_holds():
    v = torch.arange(-448, 449, dtype=torch.float32)
    held = v[v.to(torch.float8_e4m3fn).float() == v]
    assert steps.LEVELS["e4m3i"] == tuple(int(x) for x in held)


@pytest.mark.parametrize("route", steps.ROUTES)
def test_every_route_recombines_every_int8_pair_from_held_planes(route):
    x = torch.arange(-128, 128, dtype=torch.float32)
    a, b, pairs = steps.route_planes(route, x[:, None], x[None, :])
    fmt = "e2m1" if route.startswith("fp4") else "e4m3"
    assert all(steps.holds(fmt, p) for p in a + b)
    got = sum(c * a[i].double() * b[j].double() for i, j, c in pairs)
    assert torch.equal(got, x[:, None].double() * x[None, :].double())


def test_to_format_rounds_to_the_nearest_value():
    v = torch.tensor([0.2, 0.3, 1.2, 1.3, 2.4, 2.6, 3.4, 4.9, 5.1, 7.0, -2.6])
    assert steps.to_format(v, "e2m1").tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.0, 4.0, 6.0, 6.0, -3.0]
    assert steps.to_format(torch.tensor([17.0, 500.0]), "e4m3").tolist() == [16.0, 448.0]


@pytest.mark.parametrize("fmt,scale", [("e4m3", "tensor"), ("e4m3", "channel"), ("e4m3", "mx"), ("e2m1", "mx"),
                                       ("e2m1", "nv")])
def test_fake_quant_is_close_and_blockwise(fmt, scale):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 64, generator=g) * torch.logspace(-2, 2, 64)
    q = steps.fake_quant(x, fmt, scale)
    err = ((q - x).norm() / x.norm()).item()
    assert err < (0.05 if fmt == "e4m3" else 0.25)
    assert torch.equal(steps.fake_quant(x.T, fmt, scale, 0), q.T)


@pytest.mark.parametrize("grid,scale", [("e4m3i", "channel"), ("e4m3i", "mx"), ("e2m1i", "nv"), ("e2m1i", "mx")])
def test_native_parts_stay_on_the_grid(grid, scale):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 64, generator=g) * torch.logspace(-2, 2, 64)
    v, s, m = steps.native_parts(x, grid, scale, coined=True)
    top = steps.LEVELS[grid][-1]
    assert float(v.abs().max()) < top
    q = steps.to_grid(v, grid, "dither", torch.randint(0, 2, v.shape, generator=g).float())
    assert set(q.unique().tolist()) <= set(steps.LEVELS[grid])
    assert torch.allclose(steps.with_blocks(v, m) * s, x, rtol=1e-5)


def test_c1_emulation_is_dither_2s_bit_for_bit(llama, ids, calib):
    r = evaluate(llama, ids[:4], ["adopted-bd-lm", *C1], seeds=(0, 1), calib=calib)
    assert passed(r), r["gates"]
    assert r["gates"]["emulation"]["checked"] > 0
    for a in C1:
        for s in (0, 1):
            assert np.array_equal(r["nll"][f"{a}/s{s}"], r["nll"][f"adopted-bd-lm/s{s}"]), (a, s)


def test_baselines_and_native_grids_pass_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], OTHERS, calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in OTHERS:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.5, a
    assert r["gates"]["grid_range"]["checked"] > 0


def test_native_checks():
    with pytest.raises(ValueError, match="nv blocks"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "nv"}]}, "x")
    with pytest.raises(ValueError, match="int8 activations"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "mx", "act_grid": "int8"}]}, "x")
    with pytest.raises(ValueError, match="route"):
        spec.parse({"linear": [{"step": "int8"}, {"step": "emulate", "route": "e5m2"}]}, "x")
