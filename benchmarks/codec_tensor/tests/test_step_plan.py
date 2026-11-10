"""The step's pass under the PoUB driver's registered chunk layout (`receipts.LAYOUTS`): the fused GEMM's and
attention's receipts, as `step.py` places them in the pass's one receipt buffer, are the store's registered chunks in
store order, so `receipts.plan_of` takes the step's byte map under store-order/v0 and returns the plan the step commits
(`Step.plan`, `Plan.whole` of the weights' chunks and the KV's token-layers)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sketch_receipt as SR  # noqa: E402
import step as ST  # noqa: E402

rb = SR.rb
SB = 8192


def nbs_of(shape: tuple[int, int]) -> int:
    return shape[0] // 16


@pytest.mark.parametrize("choose", ["one", "all", "mixed"])
def test_plan_of_takes_the_step_s_pass(choose):
    """The tiny model's step at batch 3 × ctx 5, each linear's launch at one group, at nbs groups (a task a CTA), or a
    mix: every receipt is one registered chunk, all of them once, the weights' then the KV's in store order."""
    layers, _, shapes, _, w_bytes = ST.store_layout("tiny")
    pick = {"one": lambda s: 1, "all": nbs_of, "mixed": lambda s: max(1, nbs_of(s) // 3)}[choose]
    groups = {name: pick(shape) for name, shape in shapes.items()}
    slots = 3 * 5
    blocks = w_bytes // SR.BLOCK + slots * layers
    m = ST.pass_vectors("tiny", groups, slots, SB)
    assert m.shape == (blocks, SR.BLOCK)
    assert rb.LAYOUTS == (rb.STORE_ORDER,)
    assert rb.plan_of(m, blocks, rb.STORE_ORDER) == rb.Plan.whole(blocks)


@pytest.mark.parametrize("model, nvec", [("8b", None), ("70b", 16_968_192)])
def test_each_linear_s_receipts_start_at_its_store_block(model, nvec):
    """At the measured models, without the maps: each launch has a receipt for each of its linear's chunks at any
    groups, so the receipts of linear (l, name) start at its first byte's chunk (the check `Step._launch_params` makes
    of `wt_sketch_vectors`); at 70B the weights' receipts number the step's log's (16,968,192 a pass)."""
    _, _, shapes, off, w_bytes = ST.store_layout(model)
    for name, (n, kd) in shapes.items():
        for g in (1, 7, nbs_of((n, kd))):
            assert len(SR.gemm_receipts(n // 16, kd // 512, SB, g)) * SR.BLOCK == n * kd
    assert all(o % SR.BLOCK == 0 for o in off.values())
    if nvec is not None:
        assert w_bytes // SR.BLOCK == nvec
