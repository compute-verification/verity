"""te_core.arrays: M's array graph, which the products term's shared-arrays factor F reads (HEShared). On the
capture's recorded toy program with its causal attention (which makes the embedding's rows token-specific, so M holds
both layers), M is two layers' triangles of forward, input-gradient and weight-gradient products, ω = 3, the forward
products and the weight gradients each triangle-free; over two micro-batches each one's activations are its own arrays
and the weights they share one, so M's twelve products stay on distinct pairs. A clique of four charges F at its ω, a
repeated pair refuses the term in plain words (its products in the graph's `repeated`), and a graph whose M differs
from the step's rows is refused."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from te_core import arrays, formalism, io_bound
from te_core.capture import script

PROGRAM = Path(__file__).parents[1] / "capture" / "tests" / "recorded" / "islands" / "toy" / "train.py"
V, H, B, CTX = 16, 256, 4, 8
SHAPES = {"emb": [V, H], "fc": [H, H], "head": [V, H], "frozen": [V, H]}


def record(tmp_path: Path, monkeypatch, *args: str):
    monkeypatch.setattr(script, "STREAM_TOKENS", 10 ** 5)
    (tmp_path / "toy.start.json").write_text(json.dumps({"config": {"vocab": V, "d_model": H},
                                                         "tensors": {n: ["float32", s] for n, s in SHAPES.items()}}))
    return script.record(PROGRAM, ["--batch", str(B), "--attn", *args], step=1, starts=tmp_path, wall=[0.1] * 8,
                         step_tokens=B * CTX)


def layer(i: int) -> list[dict]:
    """A layer's forward (X, W), input gradient (dY, W) and weight gradient (dY, X)."""
    return [{"node": f"f{i}", "phase": "forward", "arrays": [f"X{i}", f"W{i}"], "dw": False},
            {"node": f"dx{i}", "phase": "backward", "arrays": [f"dY{i}", f"W{i}"], "dw": False},
            {"node": f"dw{i}", "phase": "backward", "arrays": [f"dY{i}", f"X{i}"], "dw": True}]


def test_a_recorded_steps_m_is_its_layers_triangles_on_distinct_pairs_micro_batches_included(tmp_path, monkeypatch):
    g = record(tmp_path, monkeypatch, "--update", "inplace")
    one = arrays.check(arrays.of_step(g))
    assert (one["omega"], one["omega_forward"], one["omega_dw"], one["arrays"], one["products"]) == (3, 2, 2, 6, 6)
    assert one["repeated"] == one["loops"] == [] and arrays.refusal(one) is None
    block = g.formalism["products"] | {"arrays": one}
    got = formalism.products_of_trace(block)
    assert got["arrays"] == one and all(v["arrays"] == one for v in got.get("at", {}).values())
    assert "arrays" not in formalism.products_of_trace(g.formalism["products"])
    two = arrays.check(arrays.of_step(record(tmp_path, monkeypatch, "--update", "adamw", "--micro", "2")))
    assert (two["omega"], two["omega_forward"], two["omega_dw"], two["arrays"], two["products"]) == (3, 2, 2, 10, 12)
    assert two["repeated"] == []
    g.formalism["products"]["rows"].append({"count": 1, "affected": [True, True], "self": False,
                                            "gathered": [False, False], "phase": "backward", "token_dims": ["m"]})
    with pytest.raises(ValueError, match="6 products in M, its rows 7"):
        arrays.of_step(g)


def test_f_is_at_the_graphs_clique_number_and_a_repeated_pair_refuses_with_its_products():
    base = arrays.check(layer(0) + layer(1))
    assert (base["omega"], base["omega_forward"], base["omega_dw"], base["arrays"], base["repeated"]) == (3, 2, 2, 6, [])
    k4 = arrays.check(layer(0) + [{"node": f"z{a}", "phase": "backward", "arrays": [a, "Z"]} for a in ("X0", "W0", "dY0")])
    assert (k4["omega"], k4["clique"]) == (4, ["W0", "X0", "Z", "dY0"])
    assert arrays.share(None) == arrays.share(2) == arrays.share(3) == io_bound.SHARE
    assert arrays.share(4) == pytest.approx(math.sqrt(1.5), rel=1e-15) and arrays.share(4) > io_bound.SHARE
    assert arrays.share(None, 1.0) == arrays.share(2, 1.0) == 1.0 and arrays.share(3, 1.0) == io_bound.SHARE
    again = arrays.check(layer(0) + [{"node": "f0r", "phase": "backward", "arrays": ["W0", "X0"]}])
    assert again["repeated"] == [{"pair": ["W0", "X0"], "products": ["f0", "f0r"], "count": 2}]
    why = arrays.refusal(again)
    assert why.startswith("unmeasured: the step's array graph puts 2 of M's products on 1 pair(s) of arrays")
    assert "W0" not in why and "f0" not in why and again["loops"] == []


def test_a_recomputation_a_copy_and_a_cast_are_their_originals_value_so_m_keeps_one_copy_of_each_product():
    """An activation checkpoint reruns its forward: the rerun's product reads the original's values in both operands,
    so its arrays are the original's and the pair repeats, as it does for a product on a copy cast down and back; a
    product of a value with its own rerun is a loop. Each refuses F. Products on other values stay apart: two weights
    made alike (zeros) and written in place by different updates, and two draws of randomness of one shape."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from te_core.capture.structure import _Flow

    def step(x, w, v):
        h, again = torch.relu(x @ w), torch.relu(x @ w)
        a, b = torch.zeros(8, 8), torch.zeros(8, 8)
        a.add_(v)
        b.add_(w)
        return (h @ v, again @ v, x.clone().to(torch.bfloat16).float() @ w, torch.relu(x @ v) @ v, h @ again.t(),
                x @ a, x @ b, x @ torch.randn(8, 8), x @ torch.randn(8, 8))

    gm = make_fx(step)(torch.randn(4, 8), torch.randn(8, 8), torch.randn(8, 8))
    flow = _Flow(gm, {}, lambda n, i: 0)
    of = arrays.values(flow)
    products = [{"node": n.name, "phase": "forward", "arrays": [arrays.array(flow, of, n, a) for a in n.args[:2]]}
                for n in gm.graph.nodes if n.target is torch.ops.aten.mm.default]
    assert len(products) == 12
    got = arrays.check(products)
    assert sorted(r["count"] for r in got["repeated"]) == [2, 3] and len(got["loops"]) == 1
    assert got["omega"] == 2 and got["arrays"] == 9
    why = arrays.refusal(got)
    assert "puts 5 of M's products on 2 pair(s)" in why and "1 of M's products of one value with a copy" in why
