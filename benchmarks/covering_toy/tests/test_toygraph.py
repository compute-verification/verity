import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bounds as BD  # noqa: E402
import toygraph as TG  # noqa: E402

SHAPES = [((2, 2, 2), 1, 2), ((3, 3, 3), 2, 2), ((2, 3, 3, 2), 2, 3), ((4, 4, 4), 3, 1)]


@pytest.mark.parametrize("dims,batch,steps", SHAPES)
def test_the_graph_computes_the_recipe(dims, batch, steps):
    """The graph's final weights are numpy SGD's on the same sources, so the covering runs on the real training graph."""
    G = TG.mlp(dims, batch, steps)
    for seed in range(3):
        inp = TG.random_inputs(G, seed)
        got = G.evaluate(inp)[G.outputs]
        ref = np.concatenate([w.ravel() for w in TG.reference_sgd(G, inp)])
        assert np.allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("dims,batch,steps", SHAPES)
def test_structure_matches_the_covering_vocabulary(dims, batch, steps):
    G = TG.mlp(dims, batch, steps)
    assert all(p < g for g in range(G.N) for p in G.preds[g])
    assert all(G.is_source(g) for g in range(G.N) if G.theta[g])
    S = sum(dims[l] * dims[l + 1] for l in range(len(dims) - 1))
    assert len(G.mul) == (steps - 1) * batch * S
    seen: set[int] = set()
    for gates in G.token_gates.values():
        assert not seen & set(gates)
        seen |= set(gates)
        assert not any(G.theta[g] for g in gates)
    for (slot, tok), g in G.mul.items():
        assert G.op[g] == "mul" and G.tag[g]["slot"] == slot and G.tag[g]["token"] == tok
        assert not G.theta[G.preds[g][0]]


def test_upstream_counts_one_row_per_hidden_layer():
    """A layer-l product reads an activation fed by one row of W_{l-1} (d slots) and all of the earlier layers."""
    G = TG.mlp((2, 3, 3, 2), 1, 2)
    up = BD.upstream(G)
    by_layer: dict[int, set[int]] = {}
    for (slot, _tok), u in up.items():
        by_layer.setdefault(slot[0], set()).add(u)
    assert by_layer == {0: {0}, 1: {2}, 2: {3 + 6}}


def test_lean_bound_and_budget_formulas():
    G = TG.mlp((2, 2, 2, 2), 1, 2)
    r = BD.lean_two_costs(G, 4, cw=1.0)
    assert r["D"] == 5 and r["per_block"] == 20 and r["M_star"] == 4 and r["n_lb"] == pytest.approx(0.2)
    assert BD.lean_two_costs(G, 4, cw=0.5)["D"] == 9
    assert BD.honest_block_size(G, 4) == 12 + 2 * (2 + 3 * 6 + 4)
