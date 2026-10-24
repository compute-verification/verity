import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cover as CV  # noqa: E402
import toygraph as TG  # noqa: E402


@pytest.fixture(scope="module")
def G():
    return TG.mlp((2, 2, 2), 1, 2)


@pytest.mark.parametrize("k,s", [(2, None), (3, 32), (4, 40), (6, None), (8, 80)])
def test_every_heuristic_returns_a_covering(G, k, s):
    for B in (CV.greedy(G, k, s), CV.partition_topo(G, k, s)):
        CV.validate(G, B, k, s)
    best = CV.lns(G, k, s, CV.greedy(G, k, s, rng=random.Random(1), noise=0.3), iters=20)
    CV.validate(G, best, k, s)


def test_partition_bisect_returns_a_covering(G):
    pytest.importorskip("networkx")
    CV.validate(G, CV.partition_bisect(G, 4, 40), 4, 40)


def test_covering_a_target_subset(G):
    B = CV.greedy(G, 3, None, U=set(G.values))
    CV.validate(G, B, 3, None, G.values)
    with pytest.raises(AssertionError):
        CV.validate(G, B, 3, None)


def test_validate_refuses_what_is_not_a_covering(G):
    k, s = 4, 40
    blocks = CV.greedy(G, k, s)
    with pytest.raises(AssertionError, match="uncovered"):
        CV.validate(G, blocks[1:], k, s)
    big = max(blocks, key=lambda b: b.unregistered(G))
    with pytest.raises(AssertionError, match="unregistered"):
        CV.validate(G, blocks, big.unregistered(G) - 1, s)
    g = next(iter(blocks[0].computed))
    broken = CV.Block(blocks[0].computed, blocks[0].inputs - set(G.preds[g]) - blocks[0].computed)
    if broken.inputs != blocks[0].inputs:
        with pytest.raises(AssertionError, match="outside the block"):
            CV.validate(G, [broken] + blocks[1:], k, s)


def test_one_block_recomputes_everything_from_the_data(G):
    """With no gate budget and k at least the data, one block computes the whole training run: the global paradox
    the gate budget resolves."""
    data = sum(1 for g in range(G.N) if G.op[g] == "data")
    B = CV.greedy(G, data, None)
    CV.validate(G, B, data, None)
    assert len(CV.drop_redundant(G, B)) == 1
