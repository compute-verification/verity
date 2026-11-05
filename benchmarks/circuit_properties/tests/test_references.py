"""The properties of the declared reference circuits: the Boolean GEMM against the gate-by-gate definitions, and one served
layer (SmolLM2-135M's) against the shapes its GEMMs read and benchmarks/rho's for the model."""

from __future__ import annotations

import pytest
from circuit_properties import flat
from circuit_properties import references as R
from circuit_properties.measure import Blocks, measure, whole
from circuit_properties.ops import op_shape

K, N = 32, 4


@pytest.fixture(scope="module")
def gemm():
    p = R.boolean_gemm(K, N)
    return p, flat.explicit(p, R.GEMM_PRODUCTS)


def test_boolean_gemm_matches_flat(gemm):
    p, ex = gemm
    m = measure(p, R.GEMM_PRODUCTS)
    f = flat.properties(p, R.GEMM_PRODUCTS, ex)
    assert (m.products, m.a_out, m.a_in) == (K * N, N, K)
    assert (f.products, f.s_x, f.a_out, f.a_out_values, f.activation_values, f.constant_leaves, f.a_in) == \
        (m.products, m.s_x, m.a_out, m.a_out_values, m.activation_values, m.constant_leaves, m.a_in)
    assert m.s_x == 681 and m.s_x_max == 735


def test_boolean_gemm_partitions(gemm):
    p, ex = gemm
    last = len(p.root.body.nodes) - 1
    gemm_node = p.root.body.nodes[last]
    off, G = p.root.body.offsets[last], gemm_node.fn.gates
    by_coordinate = Blocks([(0, None)] + [(off + n * G, n) for n in range(N)])
    w = whole(p, R.GEMM_PRODUCTS, by_coordinate)
    assert w.per_block == {n: K for n in range(N)} and w.split == 0
    coord = gemm_node.fn
    k_step = next(k for k, nd in enumerate(coord.body.nodes) if nd.form == "call" and nd.fn.gates > 1000)
    step = coord.body.nodes[k_step].fn
    soff = off + coord.body.offsets[k_step]
    lane0 = op_shape(step, R.GEMM_PRODUCTS.sites[0]).members[0]
    for cut in (step.gates // 2, (lane0[0] + lane0[-1]) // 2):
        blocks = Blocks([(0, None), (soff, 0), (soff + cut, 1), (soff + step.gates, 2)])
        owner = [None] * soff + [0] * cut + [1] * (step.gates - cut) + [2] * (p.gates - soff - step.gates)
        w = whole(p, R.GEMM_PRODUCTS, blocks)
        per, split = flat.whole(p, R.GEMM_PRODUCTS, owner, ex)
        assert (w.per_block, w.split) == (dict(per), split)
    assert w.split >= 1


@pytest.mark.slow
def test_tiny_layer_matches_flat():
    """b1's layer at tiny shapes (3.9e6 gates), gate by gate: about 40 s of layout lookups. It is the one test holding the
    summaries and whole-in-block to the definitions through a served layer's wiring (σ's replayed references, the norm's
    and RoPE's outputs, LayerPre's tuple, the residual), which the synthetic circuits only imitate."""
    from verity.ml.definitions.registry import b1
    C = dict(b1.B0_CONFIG, H=16, NH=1, KVH=1, D=16, I=16, QKV=48, GU=32, V=64, NL=1)
    p = R.layer(C, 1)
    ex = flat.explicit(p, R.GEMM_PRODUCTS)
    m, f = measure(p, R.GEMM_PRODUCTS), flat.properties(p, R.GEMM_PRODUCTS, ex)
    assert (m.products, m.a_out, m.a_in) == (16 * (48 + 16 + 32) + 16 * 16, 48, 16)
    assert (f.products, f.s_x, f.a_out, f.a_out_values, f.activation_values, f.constant_leaves, f.a_in) == \
        (m.products, m.s_x, m.a_out, m.a_out_values, m.activation_values, m.constant_leaves, m.a_in)
    for blocks in (Blocks.by_root(p), Blocks([(0, 0), (p.gates // 2, 1)])):
        w = whole(p, R.GEMM_PRODUCTS, blocks)
        per, split = flat.whole(p, R.GEMM_PRODUCTS, [blocks.of(g) for g in range(p.gates)], ex)
        assert (w.per_block, w.split) == (dict(per), split)
        assert w.whole + w.split == m.products


@pytest.mark.slow
def test_smollm2_layer():
    """The served SmolLM2-135M layer on bits (5.6e9 gates; building its Definitions takes most of the time): what only a
    real layer shows, its four GEMMs' products and the MLP input's fan-out, and that they are the `fanout`, `legs` and
    layer products benchmarks/rho reads for the model (its `models.toml` row, and derived from the config's matrices)."""
    import run
    from verity.ml.definitions.registry import b1
    C = b1.B0_CONFIG
    m = measure(R.layer(C, 1), R.GEMM_PRODUCTS)
    H, I = C["H"], C["I"]
    assert m.products == H * (C["QKV"] + C["NH"] * C["D"] + C["GU"]) + I * H
    assert (m.a_out, m.a_in, m.s_x) == (C["GU"], H, 681)
    assert m.activation_values == 16 * (3 * H + I) and m.excluded == {}
    rho = run.rho_shapes("smollm2-135m", C)
    row = rho["models.toml"]
    assert (row["fanout"], row["legs"], row["layer_products"]) == (m.a_out, m.a_in, m.products)
    assert (row["derived_fanout"], row["derived_legs"]) == (m.a_out, m.a_in)
    assert (rho["config"]["fanout"], rho["config"]["legs"], rho["config"]["layer_products"]) == (m.a_out, m.a_in, m.products)
