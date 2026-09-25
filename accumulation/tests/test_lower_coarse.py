"""Fast checks of :func:`accumulation.bounds.lower_coarse.lower_coarse` on the registry tiny circuits.

The exact values come from the red team's exact solver (``redteam/rollout_scale.py`` cells, ``results/
validation.md``): ``forward-nonfixed @ tiny2`` at the default point is solved exactly for ``Q_roll <= 4``."""
from __future__ import annotations

import json

import pytest

from accumulation.algorithms.registry import ALGORITHMS
from accumulation.bounds.lower import lower_coarse
from accumulation.graph import extract
from accumulation.sweeps.coarse_run import Cell, make_workload
from accumulation.sweeps.synthetic import model_cfg


def _cell(model: str, circuit: str, Q_inf: int, F_hat: float, G_hat: float, Q_roll: int | None):
    inf_name = "inference-moe" if "moe" in circuit else "inference-dense"
    cfg = model_cfg(model)
    seq = Q_inf
    c = Cell(model, circuit, Q_inf, F_hat, G_hat, Q_roll, 1 if Q_roll else None, seq)
    bp = ALGORITHMS[circuit].build(cfg, make_workload(c))
    g = extract(bp)
    inf = extract(ALGORITHMS[inf_name].build(cfg, make_workload(Cell(model, inf_name, Q_inf, F_hat, G_hat, None, None, seq))))
    fwd = inf.total_work()
    return g, bp, int(F_hat * fwd), int(G_hat * fwd)


def test_inference_is_tokens_only():
    g, bp, F, G = _cell("tiny2", "inference-dense", 4, 1.5, 4.0, None)
    lb = lower_coarse(g, F, G, program=bp.program)
    assert lb.total == lb.source == lb.token_seed
    assert lb.root_read == 0 and lb.recurring == 0 and lb.gap == 0.0 and lb.certified


def test_rollout_tiny2_matches_exact_optimum():
    # exact solver: I* = 748 / 752 / 760 for Q_roll = 1 / 2 / 4 (default F_hat = 1.5, G_hat = 4)
    for Q_roll, istar in ((1, 748), (2, 752), (4, 760)):
        g, bp, F, G = _cell("tiny2", "forward-nonfixed", 1, 1.5, 4.0, Q_roll)
        lb = lower_coarse(g, F, G, program=bp.program)
        assert lb.total == istar, (Q_roll, lb.total, lb.notes)
        assert lb.certified and lb.gap <= 1e-6
        assert lb.total == lb.token_seed + lb.root_read + lb.recurring


def test_rollout_tiny_moe_matches_exact_optimum():
    # exact solver: I* = 1048 / 1064 for Q_roll = 4 / 8 (default point)
    for Q_roll, istar in ((4, 1048), (8, 1064)):
        g, bp, F, G = _cell("tiny-moe", "forward-nonfixed-moe", 4, 1.5, 4.0, Q_roll)
        lb = lower_coarse(g, F, G, program=bp.program)
        assert lb.total == istar, (Q_roll, lb.total, lb.notes)
        assert lb.detail.get("moe"), "MoE block cells must be recognised"


def test_rollout_monotone_in_G_and_metadata_jsonable():
    g, bp, F, G = _cell("tiny2", "forward-nonfixed", 1, 1.5, 4.0, 16)
    lb4 = lower_coarse(g, F, G, program=bp.program)
    lb1 = lower_coarse(g, F, G // 4, program=bp.program)
    assert lb1.total >= lb4.total
    for lb in (lb1, lb4):
        assert lb.total == lb.token_seed + lb.root_read + lb.recurring
        assert 0.0 <= lb.gap
        json.dumps(lb.detail)


@pytest.mark.parametrize("Q_roll", [16])
def test_chain_partition_closes_the_double_count(Q_roll):
    """A token cannot be both ``I_{l-1}`` and a hopper: with the partition the G-only bound on tiny2 must exceed
    the plain token + root floor (the pre-partition certificate collapsed to it on these cells)."""
    g, bp, F, G = _cell("tiny2", "forward-nonfixed", 1, 1.5, 1.0, Q_roll)
    lb = lower_coarse(g, None, G, program=bp.program)
    assert lb.recurring > 0
