import types

import numpy as np
import pytest
import torch

import gates as G
import matmul
import spec
import steps
from test_engine import evaluate, passed, ppl


@pytest.mark.parametrize("grid", sorted(steps.GRIDS))
def test_bracket_is_adjacent_levels(grid):
    levels = steps.GRIDS[grid]
    v = torch.linspace(levels[0], levels[-1], 4001)[:-1]
    lo, hi = steps.bracket(v, grid)
    table = torch.tensor(levels, dtype=torch.float32)
    assert bool(((lo <= v) & (v < hi)).all())
    assert bool(torch.isin(lo, table).all() and torch.isin(hi, table).all())
    idx = torch.searchsorted(table, lo)
    assert bool((table[idx + 1] == hi).all())


@pytest.mark.parametrize("grid", sorted(steps.GRIDS))
def test_on_grid_serves_levels_and_rounds_to_the_nearest(grid):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 96, generator=g) * torch.rand(64, 1, generator=g) * 10
    x[3, 5] = 300.0
    table = torch.tensor(steps.GRIDS[grid], dtype=torch.float32)
    q, s = steps.on_grid(x, grid, "round", None, 1)
    assert s.shape == (64, 1) and bool(torch.isin(q, table).all())
    dist = ((x / s)[..., None] - table).abs().amin(-1)
    assert torch.allclose((x / s - q).abs(), dist, atol=1e-5)
    assert bool((q.abs().amax(1) == table[-1]).all())


@pytest.mark.parametrize("grid", sorted(steps.GRIDS))
def test_dither_serves_the_level_below_or_above_by_its_bit(grid):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(32, 80, generator=g)
    table = torch.tensor(steps.GRIDS[grid], dtype=torch.float32)
    bits = torch.randint(0, 2, x.shape, generator=g).float()
    q, s = steps.on_grid(x, grid, "dither", bits, 1)
    q0, _ = steps.on_grid(x, grid, "dither", torch.zeros_like(x), 1)
    q1, _ = steps.on_grid(x, grid, "dither", torch.ones_like(x), 1)
    v = x / s
    assert bool(torch.isin(q, table).all()) and bool((v.abs() < table[-1]).all())
    assert bool(((q0 <= v) & (v < q1)).all()) and torch.equal(q, torch.where(bits == 1, q1, q0))
    if grid != "fp4":
        assert torch.equal(q, torch.floor(v) + bits)


@pytest.mark.parametrize("grid", sorted(steps.GRIDS))
def test_sr_is_unbiased_between_adjacent_levels(grid):
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 64, generator=g).expand(4096, -1)
    q, s = steps.on_grid(x, grid, "sr", torch.rand(x.shape, generator=g), 1)
    lo, hi = steps.bracket(x / s, grid)
    assert bool(((q == lo) | (q == hi)).all())
    assert torch.allclose(q.mean(0), (x / s)[0], atol=0.1 * (hi - lo).amax().item())


@pytest.mark.parametrize("step, kw", [
    (steps.grid_attention, {"grid": "fp4c", "q": "dither", "k": "dither", "p": "dither", "v": "dither"}),
    (steps.int8_attention, {"p": "dither", "p_top": 126, "v": "dither", "v_top": 63})])
def test_a_dithered_p_serves_no_future_key(step, kw):
    rep, s, d = 2, 12, 8
    g = torch.Generator().manual_seed(3)
    causal = torch.ones(s, s, dtype=torch.bool).tril()
    p = torch.softmax(torch.randn(rep, s, s, generator=g).masked_fill(~causal, float("-inf")), -1)
    v = torch.randn(s, d, generator=g)
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 1, types.SimpleNamespace(li=0), G.Gates(), {}, {}, None)
    mm = matmul.MM("pv", "pv", run, p, v, cache={}, extra={"group": 0, "rows": (0, s), "causal": causal, "rep": rep,
                                                           "s": s, "d": d})
    served = []

    def nxt(m):
        served.append(m.x)
        return m.int_product()

    out = step(mm, nxt, **kw)
    assert len(served) == 1 and bool((served[0][:, ~causal] == 0).all()) and bool((served[0][:, causal] >= 0).all())
    assert served[0].max() <= kw.get("p_top", 127)
    assert out.shape == (rep, s, d) and run.gates.passed and run.gates.log["int8_range"]["checked"] == 1


def test_a_dithered_int8_p_takes_its_bits_mean_off():
    """A bit's +1/2 on every causal key would add half of V's causal sum to the output (perplexity 1949 at 8B WT-2 2K)."""
    rep, s, d = 2, 256, 16
    g = torch.Generator().manual_seed(3)
    causal = torch.ones(s, s, dtype=torch.bool).tril()
    p = torch.softmax((torch.randn(rep, s, s, generator=g) * 2).masked_fill(~causal, float("-inf")), -1)
    v = torch.randn(s, d, generator=g)
    err = {}
    for mode in ("round", "dither"):
        run = matmul.Run(types.SimpleNamespace(name="t"), 0, 1, types.SimpleNamespace(li=0), G.Gates(), {}, {}, None)
        mm = matmul.MM("pv", "pv", run, p, v, cache={}, extra={"group": 0, "rows": (0, s), "causal": causal, "rep": rep,
                                                               "s": s, "d": d})
        out = steps.int8_attention(mm, lambda m: m.int_product(), p=mode)
        assert run.gates.passed
        err[mode] = float((out - p @ v).pow(2).mean().sqrt())
    assert err["dither"] < 3 * err["round"], err


@pytest.mark.parametrize("grid, mode", [("fp4", "round"), ("fp4", "dither"), ("fp4c", "dither"), ("fp4", "sr")])
def test_block_scales_are_the_smallest_odd_that_holds_the_block(grid, mode):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(24, 96, generator=g) * torch.rand(24, 96 // 16, generator=g).repeat_interleave(16, 1) * 5
    x[2, :16] = 0
    bits = torch.randint(0, 2, x.shape, generator=g).float() if mode == "dither" else torch.rand(x.shape, generator=g)
    q, s = steps.on_blocks(x, grid, mode, None if mode == "round" else bits, 16, "odd")
    top, odd = steps.GRIDS[grid][-1], torch.tensor(steps.SCALES["odd"], dtype=torch.float32)
    v = (x / s).view(24, 6, 16)
    sigma = odd[torch.searchsorted(odd, (v.abs().amax(-1) / top).contiguous())]
    assert bool((sigma * top >= v.abs().amax(-1)).all())
    assert bool(((sigma - 2) * top < v.abs().amax(-1)).logical_or(sigma == 1).all())
    codes = q.view(24, 6, 16) / sigma[..., None]
    assert torch.equal(codes, codes.round()) and bool(torch.isin(codes, torch.tensor(steps.GRIDS[grid], dtype=torch.float32)).all())
    assert float(q.abs().max()) <= top * 15
    if mode == "round":
        assert torch.equal(q[2, :16], torch.zeros(16))
    want = steps.to_grid(v / sigma[..., None], grid, mode, None if mode == "round" else bits.view(24, 6, 16))
    assert torch.equal(codes, want)


def test_coin_positions_are_a_fixed_count_per_word():
    mm = types.SimpleNamespace(seed=0, layer=3, name="mlp.down_proj", device="cpu")
    for k, density, short in ((4096, 0.25, None), (224, 0.5, 32), (128, 0.125, None)):
        m = steps.coin_positions(mm, k, density)
        assert m.shape == (k,) and m.dtype == torch.bool
        full = k // steps.WORD_K * steps.WORD_K
        assert bool((m[:full].view(-1, steps.WORD_K).sum(1) == round(density * steps.WORD_K)).all())
        if short:
            assert int(m[full:].sum()) == short * density
    assert torch.equal(steps.coin_positions(mm, 4096, 0.25), steps.coin_positions(mm, 4096, 0.25))
    other = types.SimpleNamespace(seed=1, layer=3, name="mlp.down_proj", device="cpu")
    assert not torch.equal(steps.coin_positions(mm, 4096, 0.25), steps.coin_positions(other, 4096, 0.25))


def test_density_coins_only_the_chosen_positions():
    g = torch.Generator().manual_seed(6)
    x = torch.randn(16, 128, generator=g)
    coined = torch.zeros(128, dtype=torch.bool)
    coined[::4] = True
    q, s = steps.on_grid(x, "int5", "dither", torch.ones_like(x), 1, coined)
    v = x / s
    assert torch.equal(q[:, coined], torch.floor(v[:, coined]) + 1)
    assert torch.equal(q[:, ~coined], steps.to_grid(v[:, ~coined], "int5", "round", None))


def test_the_int8_dither_at_a_density_rounds_the_uncoined_positions():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(16, 128, generator=g)
    coined = torch.zeros(128, dtype=torch.bool)
    coined[1::8] = True
    q, s = steps.act_quant(x, "dither", torch.ones_like(x), 16, coined)
    v = x / s[:, None]
    assert torch.equal(q[:, coined], torch.floor(v[:, coined]) + 1)
    assert torch.equal(q[:, ~coined], torch.round(v[:, ~coined]))
    assert float(q.abs().max()) <= 16
    with pytest.raises(ValueError, match="act = dither"):
        spec.parse_step("linear", {"step": "int8", "density": 0.25})


def test_a_wide_grid_takes_the_exact_fp32_route():
    """Block-scaled E2M1 operands reach 12 * 15 = 180, past int8: gated within the declared bound, and exact."""
    g = torch.Generator().manual_seed(7)
    x = torch.randint(-180, 181, (40, 256), generator=g).float()
    w = torch.randint(-180, 181, (24, 256), generator=g).float()
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 0, types.SimpleNamespace(li=0), G.Gates(), {}, {}, None)
    mm = matmul.MM("linear", "mlp.up_proj", run, x, w, cache={}).integer(x, w, top=180)
    acc = mm.int_product()
    assert torch.equal(acc.double(), x.double() @ w.double().T)
    log = run.gates.log
    assert run.gates.passed and log["grid_range"]["checked"] == 1 and log["int8_range"]["checked"] == 0
    assert log["product_exact"]["checked"] == 1
    mm.integer(x * 2, w, top=180).int_product()
    assert log["grid_range"]["failed"] == 1


def test_the_linear_peel_serves_w_plus_u_and_gates_it(monkeypatch):
    g = torch.Generator().manual_seed(8)
    x = torch.randint(-15, 17, (32, 128), generator=g).float()
    w = torch.randint(-8, 9, (16, 128), generator=g).to(torch.int8)
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 0, types.SimpleNamespace(li=0), G.Gates(), {}, {}, None)
    mm = matmul.MM("linear", "mlp.up_proj", run, x, w, cache={}).integer(x, w)
    served = []
    real = mm.int_product

    def spy(xx=None, ww=None):
        served.append(ww)
        return real(xx, ww)

    for alphabet, width in (("pm", 8), ("bit", 1)):
        served.clear()
        monkeypatch.setattr(mm, "int_product", spy)
        out = steps.peel_linear(mm, lambda m: m.int_product(), alphabet=alphabet, width=width, top=16)
        assert torch.equal(out, x.long() @ w.long().T)
        u = served[1].float() - w.float()
        assert torch.equal(u, u[:, :32].repeat(1, 4))
        assert set(u.unique().tolist()) <= ({-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8} if alphabet == "pm" else {0, 1})
        assert run.gates.passed and run.gates.log["peel"]["checked"] > 0
    steps.peel_linear(mm, lambda m: m.int_product(), alphabet="pm", width=8, top=12)
    assert run.gates.log["peel"]["failed"] == 1


ARMS = ["fp4c-dither", "fp4c-dither-all", "fp4-dither-all", "int5-round", "fp4c-dither-all-kv", "int5-sr"]
NEW_ARMS = ["nvfp4-round", "nvfp4-dither", "nvfp4-sr", "nvfp4-q4", "nvfp4c-dither", "int5-q8", "fp8-peel-w15",
            "fp8-peel-w8-gptq", "fp8-peel-w15-q4", "nvfp4-int-round"]


def test_the_lowbit_design_arms_pass_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], NEW_ARMS, seeds=(0, 1), calib=calib)
    assert passed(r), r["gates"]
    g = r["gates"]
    assert g["grid_range"]["checked"] > 0 and g["peel"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for a in NEW_ARMS:
        for s in spec.load(a).seeds((0, 1)):
            assert np.isfinite(r["nll"][f"{a}/s{s}"]).all(), a
    assert abs(ppl(r["nll"]["nvfp4-round/s0"]) / bf - 1) < 0.2


def test_the_grid_arms_pass_every_gate(llama, ids, calib):
    r = evaluate(llama, ids[:4], ARMS, seeds=(0, 1), calib=calib)
    assert passed(r), r["gates"]
    bf = ppl(r["nll"]["bf16/s0"])
    for a in ARMS:
        assert spec.load(a).seeded
        for s in (0, 1):
            assert np.isfinite(r["nll"][f"{a}/s{s}"]).all(), a
        assert not np.array_equal(r["nll"][f"{a}/s0"], r["nll"][f"{a}/s1"]), a
    assert abs(ppl(r["nll"]["int5-round/s0"]) / bf - 1) < 0.1
