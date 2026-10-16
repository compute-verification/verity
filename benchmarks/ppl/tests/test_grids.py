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


def test_a_dithered_p_serves_no_future_key():
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

    out = steps.grid_attention(mm, nxt, grid="fp4c", q="dither", k="dither", p="dither", v="dither")
    assert len(served) == 1 and bool((served[0][:, ~causal] == 0).all()) and bool((served[0][:, causal] >= 0).all())
    assert out.shape == (rep, s, d) and run.gates.passed and run.gates.log["int8_range"]["checked"] == 1


ARMS = ["fp4c-dither", "fp4c-dither-all", "fp4-dither-all", "int5-round", "fp4c-dither-all-kv"]


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
