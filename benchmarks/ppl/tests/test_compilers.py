"""The format steps: the formats' plain quantization (`fp`), Dither's coins on a format's integer grid (`native`) and
int8 run exactly by limb splitting (`emulate`)."""
import numpy as np
import pytest
import torch

import matmul
import spec
import steps
from test_engine import evaluate, passed, ppl

C1 = ["pc-c1-e4m3-k3", "pc-c1-e4m3-k4", "pc-c1-fp4-k9"]
OTHERS = ["pc-int8", "pc-fp8-tensor", "pc-fp8-channel", "pc-mxfp8", "pc-nvfp4", "pc-mxfp4", "pc-c2-int5-2s",
          "pc-c2-int5-1s", "pc-c2-e4m3i-2s", "pc-c2-e4m3i-t-2s", "pc-c2-e4m3i-1s", "pc-c2-mx8-2s", "pc-c2-mx8-1s",
          "pc-c2-nvfp4-2s", "pc-c2-nvfp4-1s", "pc-c2-mxfp4-2s", "pc-c2-mxfp4-1s", "pc-c3-fp8", "pc-c3-fp4",
          "pc-c2-nvfp4x2-2s", "pc-c2-nvfp4x2-1s", "pc-c2-mxfp4x2-2s", "pc-c3-fp4x2", "fpx-2sp-e4m3i",
          "fpx-2sp-nvfp4x2", "fpx-2sp-nvfp4x2i", "fpx-2sp-nvfp4x2-gptq", "fpx-g1-e4m3i120", "fpx-g1-nvfp4x2-nv9",
          "fpx-g1-nvfp4x2-nv9-gptq", "fpx-g1-nvfp4x2-nvi10", "fpx-g1-nvfp4x2-nvi10-gptq", "fpx-g1-nvfp4x2-nvi-gptq",
          "fpx-g1s-e4m3i120", "fpx-g1s-nvfp4x2-nvi10", "fpx-g1s-nvfp4x2-nvi10-gptq",
          "fpx-g1s-nvfp4x2-nvi10-lm8", "fpx-g1s-w4a8-nvi10", "fpx-g1s-w4a8-nvi10-gptq", "fpx-g1s-co32c-e4m3i120-w60",
          "fpx-g1s-w4a8-nvi10-lm8", "fpx-g1s-w4a8-nvi10-gptq-lm8", "fpx-g1s-co32c-w4a8-nvi5-lm8",
          "fpx-g1s-co32c-w4a8-nvi5-gptq-lm8", "fpx-g1s-e4m3i120-c6", "fpx-g1s-nvfp4x2-nvi10-lm8-c6",
          "fpx-g1s-w4a8-nvi10-gptq-lm8-c6", "fpx-g1s-co32c-w4a8-nvi5-gptq-lm8-c6", "fpx-a1-e4m3i",
          "fpx-g1h-e4m3i120-w104", "fpx-g1h-w4a8-nvi9-gptq-lm8-c6",
          *(f"{f}-lin" for f in ("pc-fp8-tensor", "pc-fp8-channel", "pc-mxfp8", "pc-nvfp4", "pc-mxfp4"))]


def test_e4m3_integers_are_every_integer_e4m3_holds():
    v = torch.arange(-448, 449, dtype=torch.float32)
    held = v[v.to(torch.float8_e4m3fn).float() == v]
    assert steps.LEVELS["e4m3i"] == tuple(int(x) for x in held)


def test_a_k32_word_of_e4m3i352_products_stays_in_one_binade():
    assert steps.LEVELS["e4m3i352"] == tuple(v for v in steps.LEVELS["e4m3i"] if abs(v) <= 352)
    assert steps.LEVELS["e4m3i352"][-1] == 352 and 32 * 352 ** 2 < 2 ** 22


def test_the_int8_subgrids_moved_by_one_stay_in_int8():
    assert steps.LEVELS["e4m3i120"] == tuple(v for v in steps.LEVELS["e4m3i"] if abs(v) <= 120)
    assert steps.LEVELS["e4m3i120"][-1] + 1 <= 127
    assert steps.LEVELS["e4m3i60"] == tuple(v for v in steps.LEVELS["e4m3i"] if abs(v) <= 60)
    assert steps.LEVELS["e4m3i60"][-1] + 1 <= 63
    assert steps.SCALES["int5"][-1] * steps.LEVELS["e2m1x2"][-1] + 1 <= 63
    assert steps.LEVELS["e4m3i104"] == tuple(v for v in steps.LEVELS["e4m3i"] if abs(v) <= 104)
    h1_top, h1_u = 110, 16
    for top in (steps.LEVELS["e4m3i104"][-1], steps.SCALES["int9"][-1] * steps.LEVELS["e2m1x2"][-1]):
        assert top <= h1_top and top + 1 + h1_u <= 127
    assert steps.LEVELS["e4m3i"][steps.LEVELS["e4m3i"].index(104) + 1] + 1 + h1_u > 127
    assert (steps.SCALES["int9"][-1] + 1) * steps.LEVELS["e2m1x2"][-1] > h1_top
    assert steps.SCALES["odd9"][-1] * (steps.LEVELS["e2m1x2"][-1] + 1) <= 127
    assert steps.SCALES["int10"][-1] * steps.LEVELS["e2m1x2"][-1] + 1 <= 127


def test_e2m1_grids_are_its_integers_and_twice_its_values():
    v = sorted({s * x for s in (-1, 1) for x in steps.E2M1})
    assert steps.LEVELS["e2m1i"] == tuple(int(x) for x in v if x == int(x))
    assert steps.LEVELS["e2m1x2"] == tuple(int(2 * x) for x in v)


@pytest.mark.parametrize("route", steps.ROUTES)
def test_every_route_recombines_every_int8_pair_from_held_planes(route):
    x = torch.arange(-128, 128, dtype=torch.float32)
    a, b, pairs = steps.route_planes(route, x[:, None], x[None, :])
    fmt = "e2m1" if route.startswith("fp4") else "e4m3"
    assert all(steps.holds(fmt, p) for p in a + b)
    got = sum(c * a[i].double() * b[j].double() for i, j, c in pairs)
    assert torch.equal(got, x[:, None].double() * x[None, :].double())


def test_exact_mm_splits_a_wide_operand_and_stays_exact():
    g = torch.Generator().manual_seed(3)
    levels = torch.tensor(steps.LEVELS["e4m3i"], dtype=torch.float32)
    a = levels[torch.randint(0, len(levels), (4, 3000), generator=g)]
    b = levels[torch.randint(0, len(levels), (3000, 5), generator=g)]
    a[:, :1500], b[:1500] = 448, 448
    assert torch.equal(matmul.exact_mm(a, b, 448 * 448), a.double() @ b.double())


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


@pytest.mark.parametrize("grid,scale", [("e4m3i", "channel"), ("e4m3i", "mx"), ("e2m1i", "nv"), ("e2m1i", "mx"),
                                        ("e2m1x2", "nv"), ("e2m1x2", "mx"), ("e4m3i352", "channel"),
                                        ("e2m1x2", "nvi"), ("e4m3i120", "channel"), ("e2m1x2", "nv9"),
                                        ("e2m1x2", "nvi10"), ("e2m1x2", "nvi5"), ("e4m3i104", "channel"),
                                        ("e2m1x2", "nvi9")])
def test_native_parts_stay_on_the_grid(grid, scale):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 64, generator=g) * torch.logspace(-2, 2, 64)
    v, s, m = steps.native_parts(x, grid, scale, coined=True)
    top = steps.LEVELS[grid][-1]
    assert float(v.abs().max()) < top
    q = steps.to_grid(v, grid, "dither", torch.randint(0, 2, v.shape, generator=g).float())
    assert set(q.unique().tolist()) <= set(steps.LEVELS[grid])
    assert torch.allclose(steps.with_blocks(v, m) * s, x, rtol=1e-5)


@pytest.mark.parametrize("grid,scale", [("e4m3i120", "channel"), ("e2m1x2", "nvi10")])
@pytest.mark.parametrize("coined", [False, True])
def test_a_clip_search_stays_on_the_grid_and_never_rounds_worse(grid, scale, coined):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(16, 64, generator=g) * torch.logspace(-1, 1, 64)
    x[:, 5] *= 30

    def err(v, s, m):
        return ((steps.with_blocks(steps.to_grid(v, grid, "round", None), m) * s - x) ** 2).sum(-1)

    base, clipped = steps.native_parts(x, grid, scale, coined), steps.native_parts(x, grid, scale, coined, clip=6)
    top = steps.LEVELS[grid][-1]
    assert float(clipped[0].abs().max()) < (top if coined else top * (1 + 1e-6))
    assert bool((err(*clipped) <= err(*base)).all()) and bool((err(*clipped) < err(*base)).any())
    assert all(torch.equal(a, b) for a, b in zip(steps.native_parts(x, grid, scale, coined, clip=0), base)
               if a is not None)


@pytest.mark.parametrize("grid,scale", [("e2m1x2", "nv"), ("e4m3i352", "channel"), ("e2m1x2", "nv9")])
def test_native_gptq_stays_on_the_grid_and_beats_rounding(grid, scale):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(512, 64, generator=g) @ torch.randn(64, 64, generator=g)
    w = torch.randn(8, 64, generator=g)
    v, _, _ = steps.native_parts(w, grid, scale, coined=False)
    rounded = steps.to_grid(v, grid, "round", None)
    q, s, m = steps.native_gptq(w, x.T @ x, grid, scale)
    assert set(q.unique().tolist()) <= set(steps.LEVELS[grid])

    def err(levels):
        return (x @ (w - steps.with_blocks(levels, m) * s).T).norm()

    assert err(q) < err(rounded)
    assert torch.equal(steps.native_gptq(w, torch.diag(torch.rand(64, generator=g) + 1), grid, scale)[0], rounded)


@pytest.mark.parametrize("grid,scale,act_grid", [("e4m3i120", "channel", ""), ("e2m1x2", "nvi10", ""),
                                                  ("e4m3i60", "channel", "e4m3i120"), ("e2m1x2", "nvi5", ""),
                                                  ("e4m3i104", "channel", "e4m3i120"), ("e2m1x2", "nvi9", "")])
@pytest.mark.parametrize("coin", [0.0, 1.0])
def test_g1_serves_the_centred_product_of_the_nearest_levels(grid, scale, act_grid, coin, monkeypatch):
    import types
    import gates as G
    g = torch.Generator().manual_seed(5)
    x, w = torch.randn(3, 64, generator=g), torch.randn(5, 64, generator=g)
    step = spec.parse({"linear": [{"step": "native", "grid": grid, "scale": scale, "act": "g1", "weight": "g1",
                                   "act_grid": act_grid}]}, "x").linear[0]
    gates = G.Gates()
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 0, types.SimpleNamespace(li=0), gates, {}, {}, None)
    mm = matmul.MM("linear", "mlp.up_proj", run, x, w).enter(step)
    monkeypatch.setattr(matmul.MM, "coins", lambda self, shape, levels=2: torch.full(tuple(shape), coin))
    y = step.defn.fn(mm, matmul.terminal, **step.kw)

    def served(a, on):
        v, s, m = steps.native_parts(a, on, scale, coined=False)
        return (steps.with_blocks(steps.to_grid(v, on, "round", None), m) + coin - 0.5) * s

    want = served(x, act_grid or grid).double() @ served(w, grid).double().T
    assert torch.allclose(y.double(), want, rtol=1e-6, atol=1e-5)
    assert gates.passed


@pytest.mark.parametrize("grid", sorted(steps.NATIVE))
def test_dither3_moves_the_nearest_level_by_its_gap_onto_three_levels(grid):
    levels = steps.LEVELS[grid]
    n = torch.tensor(levels, dtype=torch.float32)
    v = torch.cat([n, (n[:-1] + n[1:]) / 2 - 1e-3])
    near = steps.to_grid(v, grid, "round", None)
    outs = {(b, b2): steps.dither3(v, grid, torch.full_like(v, b), torch.full_like(v, b2))
            for b in (0.0, 1.0) for b2 in (0.0, 1.0)}
    assert all(set(o.unique().tolist()) <= set(levels) for o in outs.values())
    assert torch.equal(outs[0.0, 1.0], outs[1.0, 0.0])
    lo, mid, hi = outs[0.0, 0.0], outs[0.0, 1.0], outs[1.0, 1.0]
    assert bool((lo < mid).all()) and torch.equal(mid - lo, hi - mid)
    inner = (near != levels[0]) & (near != levels[-1])
    assert torch.equal(mid[inner], near[inner])
    assert float(hi[near == levels[-1]].max()) == levels[-1] and float(lo[near == levels[0]].min()) == levels[0]


def test_dither3_activations_beside_dithered_weights_serve_the_dithered_product(monkeypatch):
    import types
    import gates as G
    g = torch.Generator().manual_seed(8)
    x, w = torch.randn(3, 64, generator=g), torch.randn(5, 64, generator=g)
    step = spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "channel", "act": "dither3"}]},
                      "x").linear[0]
    gates = G.Gates()
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 0, types.SimpleNamespace(li=0), gates, {}, {}, None)
    mm = matmul.MM("linear", "mlp.up_proj", run, x, w).enter(step)
    drawn = [torch.randint(0, 2, x.shape, generator=g).float(), torch.randint(0, 2, x.shape, generator=g).float(),
             torch.randint(0, 2, w.shape, generator=g).float()]
    calls = iter(drawn)
    monkeypatch.setattr(matmul.MM, "coins", lambda self, shape, levels=2: next(calls))
    y = step.defn.fn(mm, matmul.terminal, **step.kw)
    v, sa, _ = steps.native_parts(x, "e4m3i", "channel", coined=False)
    u, sw, _ = steps.native_parts(w, "e4m3i", "channel", coined=True)
    xq = steps.dither3(v, "e4m3i", drawn[0], drawn[1]) * sa
    wq = steps.to_grid(u, "e4m3i", "dither", drawn[2]) * sw
    assert torch.allclose(y.double(), xq.double() @ wq.double().T, rtol=1e-6, atol=1e-5)
    assert gates.passed


@pytest.mark.parametrize("coin", [0.0, 1.0])
def test_g1_weights_beside_int8_activations_centre_only_the_weights(coin, monkeypatch):
    import types
    import gates as G
    g = torch.Generator().manual_seed(6)
    x, w = torch.randn(3, 64, generator=g), torch.randn(5, 64, generator=g)
    step = spec.parse({"linear": [{"step": "native", "grid": "e2m1x2", "scale": "nvi10", "act": "dither",
                                   "weight": "g1", "act_grid": "int8"}]}, "x").linear[0]
    gates = G.Gates()
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 0, types.SimpleNamespace(li=0), gates, {}, {}, None)
    mm = matmul.MM("linear", "mlp.up_proj", run, x, w).enter(step)
    monkeypatch.setattr(matmul.MM, "coins", lambda self, shape, levels=2: torch.full(tuple(shape), coin))
    y = step.defn.fn(mm, matmul.terminal, **step.kw)
    xq, sa = steps.act_quant(x, "dither", torch.full(x.shape, coin), 127)
    v, s, m = steps.native_parts(w, "e2m1x2", "nvi10", coined=False)
    wq = (steps.with_blocks(steps.to_grid(v, "e2m1x2", "round", None), m) + coin - 0.5) * s
    assert torch.allclose(y.double(), (xq * sa[:, None]).double() @ wq.double().T, rtol=1e-6, atol=1e-5)
    assert gates.passed and gates.log["int8_range"]["checked"] > 0


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


def test_row_chunks_change_no_bit(llama, ids, calib, monkeypatch):
    arms = ["pc-fp8-tensor", "pc-nvfp4", "pc-c2-e4m3i-t-2s", "pc-c2-nvfp4-2s", "pc-c1-e4m3-k4"]
    whole = evaluate(llama, ids[:2], arms, calib=calib)
    monkeypatch.setattr(steps, "ROW_CHUNK", 48)
    chunked = evaluate(llama, ids[:2], arms, calib=calib)
    assert passed(chunked), chunked["gates"]
    for k, v in whole["nll"].items():
        assert np.array_equal(v, chunked["nll"][k]), k


def test_a_native_weight_is_cached_once_across_seeds():
    import types
    import gates as G
    step = next(s for s in spec.load("pc-c2-nvfp4-2s").linear if s.defn.name == "native")
    store = {}
    for seed in (0, 1, 2):
        run = matmul.Run(types.SimpleNamespace(name="t"), seed, 0, types.SimpleNamespace(li=0), G.Gates(), store, {}, None)
        mm = matmul.MM("linear", "mlp.up_proj", run, torch.zeros(2, 32), torch.zeros(4, 32)).enter(step)
        mm.cached("native", object, coinless=True)
        mm.cached("per-seed", object)
    assert sorted(k[2] for k in store) == ["native"] + ["per-seed"] * 3


def test_native_checks():
    with pytest.raises(ValueError, match="nv blocks"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "nv"}]}, "x")
    with pytest.raises(ValueError, match="nv blocks"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i352", "scale": "nvi"}]}, "x")
    with pytest.raises(ValueError, match="gptq"):
        spec.parse({"linear": [{"step": "native", "grid": "e2m1x2", "scale": "nv", "gptq": True}]}, "x")
    with pytest.raises(ValueError, match="g1"):
        spec.parse({"linear": [{"step": "native", "grid": "e2m1x2", "scale": "mx", "act": "g1"}]}, "x")
    with pytest.raises(ValueError, match="int8 activations"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "mx", "act_grid": "int8"}]}, "x")
    with pytest.raises(ValueError, match="clip"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "scale": "tensor", "clip": 4}]}, "x")
    with pytest.raises(ValueError, match="dither3"):
        spec.parse({"linear": [{"step": "native", "grid": "e4m3i", "weight": "dither3"}]}, "x")
    with pytest.raises(ValueError, match="dither3"):
        spec.parse({"linear": [{"step": "native", "grid": "e2m1x2", "scale": "nvi10", "act": "dither3"}]}, "x")
    with pytest.raises(ValueError, match="own native grid"):
        spec.parse({"linear": [{"step": "native", "grid": "e2m1x2", "scale": "nvi10", "act_grid": "e2m1i"}]}, "x")
    with pytest.raises(ValueError, match="route"):
        spec.parse({"linear": [{"step": "int8"}, {"step": "emulate", "route": "e5m2"}]}, "x")
