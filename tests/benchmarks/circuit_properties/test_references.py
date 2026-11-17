"""The properties of the registered circuits: the Boolean GEMM against the gate-by-gate definitions, and one served
layer (SmolLM2-135M's) against the shapes its GEMMs read and benchmarks/slowdown's for the model."""

from __future__ import annotations

from pathlib import Path

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


def test_no_configuration_states_a_shape_of_its_own():
    """Each configuration is a served one (one of b1's, or Llama 3.2 1B's as the adapter reads its pinned config.json),
    or b1's Definitions at a shape benchmarks/slowdown states (`models.toml`,
    read through its loader), differing from the base only by that shape and what is no shape (the model's name, the
    norm's epsilon, the QKV bias); and a served configuration whose model the slowdown calculator has a row for has that row's shape."""
    from verity_vllm.definitions.registry import b1
    served = (b1.B0_CONFIG, b1.B1_CONFIG, b1.SMOL360_CONFIG)
    models = R.slowdown_models()
    constructed = []
    for name, (C, _) in R.configs().items():
        if any(C is s for s in served) or C == R.pinned(R.LLAMA32_1B):
            if name in models:
                assert R.shape(models[name]).items() <= C.items(), name
        else:
            constructed.append(name)
            assert C == b1.B1_CONFIG | R.shape(models[name]) | {k: C[k] for k in ("model", "EPS", "BIAS")}, name
    assert constructed == ["qwen25-7b", "llama3-405b"]
    with pytest.raises(ValueError, match="grouped-query"):
        R.shape(models["kimi-k2"])


def test_reading_the_slowdown_calculator_shadows_no_module_of_this_benchmark():
    """benchmarks/slowdown has a `run.py` of its own: reading its models leaves `run` this benchmark's."""
    R.slowdown_models()
    import run
    assert Path(run.__file__).parent == Path(R.__file__).parents[1]


@pytest.mark.slow
def test_tiny_layer_matches_flat():
    """b1's layer at tiny shapes (3.9e6 gates), gate by gate: about 40 s of layout lookups. It is the one test holding the
    summaries and whole-in-block to the definitions through a served layer's wiring (σ's replayed references, the norm's
    and RoPE's outputs, LayerPre's tuple, the residual), which the synthetic circuits only imitate."""
    from verity_vllm.definitions.registry import b1
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
    real layer shows, its four GEMMs' products and the MLP input's fan-out, and that they are the `fanout`, `fanin` and
    layer products benchmarks/slowdown reads for the model (its `models.toml` row, and derived from the config's matrices)."""
    import run
    from verity_vllm.definitions.registry import b1
    C = b1.B0_CONFIG
    m = measure(R.layer(C, 1), R.GEMM_PRODUCTS)
    H, I = C["H"], C["I"]
    assert m.products == H * (C["QKV"] + C["NH"] * C["D"] + C["GU"]) + I * H
    assert (m.a_out, m.a_in, m.s_x) == (C["GU"], H, 681)
    assert m.activation_values == 16 * (3 * H + I) and m.excluded == {}
    shapes = run.slowdown_shapes("smollm2-135m", C)
    row = shapes["models.toml"]
    assert (row["fanout"], row["fanin"], row["layer_products"]) == (m.a_out, m.a_in, m.products)
    assert (row["derived_fanout"], row["derived_fanin"]) == (m.a_out, m.a_in)
    assert (shapes["config"]["fanout"], shapes["config"]["fanin"], shapes["config"]["layer_products"]) == (m.a_out, m.a_in, m.products)


def test_a_run_of_every_configuration_writes_the_least_s_x_under_the_metrics_id(tmp_path, monkeypatch, capsys):
    """With every configuration measured (a tiny b1 layer here, beside the Boolean GEMM), the least s_× of the circuits
    goes into the run's result under the id `metrics.toml` declares, measured; a run of one configuration writes it
    partial, with that configuration's model, and each prints its point as one line; a run of some writes none."""
    import json
    import sys
    import tomllib

    import run
    from verity_vllm.definitions.registry import b1
    from verity_bench import kit

    for name in ("OUT", "RESEARCH_RUN_ID", "RESEARCH_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kit, "host_stamp", lambda: {"commit": None})
    tiny = dict(b1.B0_CONFIG, H=16, NH=1, KVH=1, D=16, I=16, QKV=48, GU=32, V=64, NL=1)
    monkeypatch.setattr(R, "configs", lambda: {"tiny": (tiny, "a test's"), "again": (tiny, "a test's"),
                                               "other": (b1.B0_CONFIG, "a test's")})
    (row,) = tomllib.loads((Path(run.__file__).parent / "metrics.toml").read_text())["metric"]
    monkeypatch.setattr(sys, "argv", ["run.py", "--models", "tiny,again", "--out", str(tmp_path / "some")])
    assert run.main() == 0 and (tmp_path / "some" / "properties.json").is_file()
    assert not (tmp_path / "some" / "result.json").exists()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["run.py", "--models", "tiny", "--out", str(tmp_path / "one")])
    assert run.main() == 0
    circuits = json.loads((tmp_path / "one" / "properties.json").read_text())
    (m,) = json.loads((tmp_path / "one" / "result.json").read_text())["measurements"]
    assert (m["name"], m["value"], m["mark"], m["model"]) == (row["id"], min(c["s_x"] for c in circuits), "partial",
                                                              tiny["model"])
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "name": row["id"], "value": m["value"], "unit": row["unit"], "mark": "partial", "model": tiny["model"],
        "card": None}
    monkeypatch.setattr(R, "configs", lambda: {"tiny": (tiny, "a test's")})
    monkeypatch.setattr(sys, "argv", ["run.py", "--out", str(tmp_path / "every")])
    assert run.main() == 0
    circuits = json.loads((tmp_path / "every" / "properties.json").read_text())
    (m,) = json.loads((tmp_path / "every" / "result.json").read_text())["measurements"]
    assert row["id"] == run.METRIC and len(circuits) == 2
    assert (m["name"], m["value"], m["unit"], m["mark"]) == (row["id"], min(c["s_x"] for c in circuits), row["unit"],
                                                             "measured")
    capsys.readouterr()
