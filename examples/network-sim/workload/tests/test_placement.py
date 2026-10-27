"""The Program, the placement and its commitment, and the crossing edges a placement implies."""
from collections import Counter
from types import SimpleNamespace

import pytest

from verity.ml.commitments import advice
from workload import model as M
from workload import placement as P
from workload.hardware import OUTSIDE, mock

SPEC = M.Spec(d=2, layers=3, requests=2, groups=2)
WORDS = SPEC.requests * SPEC.d * SPEC.act_bits


@pytest.fixture(scope="module")
def model():
    return M.Model(SPEC)


def placed(model, plan, blocks, devices=1):
    hw = mock(blocks, devices)
    return hw, P.place(model, hw, P.PLANS[plan](model, hw))


def test_the_circuit_computes_the_integer_reference(model):
    x, ws = M.requests(SPEC, 1), M.weights(SPEC, 1)
    t = model.bind_inputs(x, ws)
    model.evaluate(t)
    assert model.output_words(t) == M.reference(SPEC, x, ws)


@pytest.mark.parametrize("plan", sorted(P.PLANS))
def test_a_plan_places_every_value_paced_and_causal(model, plan):
    hw, p = placed(model, plan, 3, 2)
    assert p.n == model.n
    assert {hw.unit(p.leaf[g]) for g in model.input_gates(0)} == {OUTSIDE}
    assert OUTSIDE not in {hw.unit(x) for g, x in enumerate(p.leaf) if model.kind[g] == "computed"}
    assert p.paced(range(len(hw.leaves)))
    assert P.causal(model, p, hw) == []
    assert all(p.tick(v) == p.tick_of[v] == p.ticks[p.leaf[v]][p.step[v]] for v in range(0, p.n, 97))


def test_a_weight_is_stored_with_its_first_reader(model):
    hw, p = placed(model, "pipeline", 3, 2)
    first = {}
    for g in range(model.n):
        for a in model.ops[g]:
            first.setdefault(a, g)
    registered = [g for g in range(model.n) if model.kind[g] == "registered"]
    assert registered and all(p.leaf[g] == p.leaf[first[g]] for g in registered)


def test_pipeline_runs_layer_l_on_block_l_at_step_1_plus_r_plus_l(model):
    hw, p = placed(model, "pipeline", 3)
    for c in model.calls:
        assert {hw.unit(p.leaf[g]) for g in range(c.lo, c.hi)} == {f"b{c.layer}"}
        assert {p.step[g] for g in range(c.lo, c.hi)} == {1 + c.request + c.layer}


def test_the_rows_round_trip_and_the_commitment_binds_them(model):
    hw, p = placed(model, "pipeline", 3, 2)
    rows = p.rows()
    assert P.Placement.from_rows(rows) == p
    reg = p.commit()
    assert p.commit(reg.salts).root == reg.root
    assert advice.verify(reg.root, len(rows), P.ROW_BITS, P.BINDING, reg.open(1))
    moved = P.Placement(p.step, (p.leaf[0] + 1,) + p.leaf[1:], p.ticks)
    assert moved.commit(reg.salts).root != reg.root


def test_a_read_in_the_tick_its_value_is_made_in_another_block_breaks_causality(model):
    hw, p = placed(model, "pipeline", 3)
    c = next(c for c in model.calls if c.layer == 1)
    step = list(p.step)
    for g in range(c.lo, c.hi):
        step[g] -= 1
    bad = P.causal(model, P.Placement(tuple(step), p.leaf, p.ticks), hw)
    assert bad and all(c.lo <= g < c.hi for g, _ in bad)


def test_pipeline_crossings_are_the_activations_between_consecutive_blocks(model):
    hw, p = placed(model, "pipeline", 3, 2)
    edges = P.crossings(model, p, hw)
    assert Counter((e.src, e.dst) for e in edges) == {(OUTSIDE, "b0"): WORDS, ("b0", "b1"): WORDS, ("b1", "b2"): WORDS,
                                                       ("b2", OUTSIDE): WORDS}
    assert all(e.read_tick > e.tick for e in edges)
    assert len({(e.value, e.dst) for e in edges}) == len(edges)
    assert edges == sorted(edges)


def test_one_block_crosses_only_with_the_client(model):
    hw, p = placed(model, "pipeline", 1, 2)
    assert Counter((e.src, e.dst) for e in P.crossings(model, p, hw)) == {(OUTSIDE, "b0"): WORDS, ("b0", OUTSIDE): WORDS}


def test_tensor_crossings_carry_each_group_to_the_other_block(model):
    hw, p = placed(model, "tensor", 2)
    half = WORDS // 2
    assert Counter((e.src, e.dst) for e in P.crossings(model, p, hw)) == {
        (OUTSIDE, "b0"): WORDS, (OUTSIDE, "b1"): WORDS,
        ("b0", "b1"): half * (SPEC.layers - 1), ("b1", "b0"): half * (SPEC.layers - 1),
        ("b0", OUTSIDE): half, ("b1", OUTSIDE): half}


def test_constants_never_cross_and_a_value_crosses_once_to_each_unit():
    table = SimpleNamespace(n=4, kind=["constant", "computed", "computed", "computed"], ops=[(), (0,), (0, 1), (1,)],
                            width=[1, 1, 1, 1], outputs=(2, 3))
    hw = mock(2, 2)
    b0, b1, b1_ = hw.leaf("b0.gpu0"), hw.leaf("b1.gpu0"), hw.leaf("b1.gpu1")
    p = P.Placement((1, 1, 3, 2), (b0, b0, b1, b1_), tuple(tuple(range(5)) for _ in hw.leaves))
    edges = P.crossings(table, p, hw)
    assert [(e.value, e.src, e.dst, e.read_tick, e.dst_leaf) for e in edges] == [
        (1, "b0", "b1", 2, b1_), (3, "b1", OUTSIDE, 3, hw.client), (2, "b1", OUTSIDE, 4, hw.client)]
