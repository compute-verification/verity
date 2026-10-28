import math

import pytest
from rho.theorem import (
    Blocks,
    Budget,
    Cut,
    Partition,
    ServingBlock,
    aggregate,
    least,
    local,
    overhead,
    rho,
)

#: weight a gate holds
W = 0.5


def block(*cuts, layout=None, gates=1000.0, name="s", mfu=1.0):
    return ServingBlock(name, gates, 1.0, cuts, layout, mfu)


def test_a_serving_block_contributes_its_least_cut_and_never_more_than_its_capacity():
    a = Cut("a", (Blocks("x", Budget(200, 10), 4),), {"undecl.gpu": 200})
    b = Cut("b", (Blocks("y", Budget(1000, 50), 1),))
    s = local(block(a, b), lambda bud: bud.inputs, W)
    # a: 4 blocks of min(10, w·200 = 100) plus w·200 uncertified = 140; b: min(50, 500) = 50; capacity w·1000 = 500
    assert s.cut == "b" and math.isclose(s.value, 50 / 500)
    assert local(block(a, b), lambda bud: math.inf, W).value == 1.0


def test_the_local_minimum_beats_the_minimum_of_global_bounds_when_blocks_bind_different_cuts():
    """Two serving blocks: the first's inference cut binds (few prompt bits), the second's serving cut (a small `k_S`).
    The local minimum takes each one's own least cut; the aggregate-only bound must pick one cut for both."""
    cap = lambda bud: bud.inputs
    one = block(Cut("inference", (Blocks("inference", Budget(100, 10), 9),), {"undecl.gpu": 100}),
                Cut("serving", (Blocks("serving", Budget(1000, 500), 1),)), name="one")
    two = block(Cut("inference", (Blocks("inference", Budget(100, 90), 9),), {"undecl.gpu": 100}),
                Cut("serving", (Blocks("serving", Budget(1000, 200), 1),)), name="two")
    part = Partition((one, two), 0.0)
    w = 1.0
    # one: inference 9·10/1000 + 0.1 = 0.19 against serving 0.5; two: inference 0.91 against serving 0.2
    assert (local(one, cap, w).cut, local(two, cap, w).cut) == ("inference", "serving")
    loc = overhead(part, cap, w)
    assert loc.g == pytest.approx((0.19 + 0.2) / 2)
    # the global sums: inference (0.19 + 0.91)/2 = 0.55, serving (0.5 + 0.2)/2 = 0.35, so serving for both
    agg = overhead(part, cap, w, aggregate_only=True)
    assert agg.g == pytest.approx(0.35) and {s.cut for s in aggregate(part, cap, w).values()} == {"serving"}
    assert loc.rho == pytest.approx(1 / 0.195) and agg.rho == pytest.approx(1 / 0.35) and loc.rho > agg.rho


def test_the_aggregate_bound_is_never_tighter_than_the_local_one():
    cap = lambda bud: bud.inputs ** 1.5
    blocks = tuple(block(Cut("inference", (Blocks("inference", Budget(100, k), 9),), {"undecl.gpu": 50}),
                         Cut("serving", (Blocks("serving", Budget(1000, 40 * k), 1),)), name=f"b{k}")
                   for k in (1, 3, 10, 30))
    part = Partition(blocks, 0.01)
    assert overhead(part, cap, 1.0).rho >= overhead(part, cap, 1.0, aggregate_only=True).rho


def test_an_attack_takes_the_layout_at_its_mfu_and_ignores_the_cuts():
    cut = Cut("tight", (Blocks("x", Budget(1000, 1), 1),))
    layout = Cut("layout", (Blocks("x", Budget(1000, 1), 1),), {"undecl.gpu": 200})
    s = local(block(cut, layout=layout), lambda bud: 0.0, W, side="attack")
    assert s.cut == "layout" and math.isclose(s.value, 0.2)
    assert local(block(cut, mfu=0.4), lambda bud: 0.0, W, side="attack").value == 0.4


def test_rho_is_the_dark_leaves_plus_the_monitored_weight():
    assert rho(0.0, 1.0) == 1.0
    assert math.isclose(rho(0.01, 0.0), 100.0)
    assert math.isclose(rho(0.01, 0.0, efficiency=3.0), 300.0)
    assert rho(0.0, 0.0) == math.inf


def test_the_bound_counts_the_dark_leaves_at_capacity_and_an_attack_at_its_mfu():
    part = Partition((block(mfu=0.4),), 0.01, dark_mfu=0.55)
    bound, attack = overhead(part, lambda b: 0.0, W), overhead(part, lambda b: 0.0, W, side="attack")
    assert bound.dark == 0.01 and bound.rho == pytest.approx(1.0)
    assert attack.dark == pytest.approx(0.0055) and attack.rho == pytest.approx(1 / (0.0055 + 0.99 * 0.4))


def test_a_bound_holds_uncertified_and_dark_gates_and_every_block_at_u_and_an_attack_at_w():
    """`u = ω_max / s_×` against the reference's `w`: the uncertified routes and the dark leaves scale by `u/w`, no
    block holds more than `u c`, and a serving block at most `u/w` of the reference's weight in it. An attack is a
    construction of the reference's operations, so `u` doesn't move it."""
    w, u = 0.5, 2.0
    part = Partition((block(Cut("c", (Blocks("x", Budget(200, 1e9), 1),), {"undecl.gpu": 100}), mfu=0.4),), 0.01)
    s = overhead(part, lambda b: b.inputs, w, u=u)
    # x: min(1e9, u·200) / (w·1000) = 0.8; undecl.gpu: u·100 / 500 = 0.4; dark: (u/w)·0.01
    assert dict(s.terms) == pytest.approx({"x": 0.8, "undecl.gpu": 0.4}) and s.dark == pytest.approx(0.04)
    assert s.rho == pytest.approx(1 / (0.04 + 0.99 * 1.2))
    assert local(block(), lambda b: math.inf, w, u=u).value == pytest.approx(u / w)
    assert overhead(part, lambda b: 0.0, w, side="attack", u=u) == overhead(part, lambda b: 0.0, w, side="attack")
    assert overhead(part, lambda b: b.inputs, w, u=w) == overhead(part, lambda b: b.inputs, w)


def test_overhead_weights_serving_blocks_by_their_gates():
    free = ServingBlock("free", 3000.0, 1.0, ())
    locked = ServingBlock("locked", 1000.0, 1.0, (Cut("c", (Blocks("x", Budget(1000, 0), 1),)),))
    s = overhead(Partition((free, locked), 0.0), lambda bud: bud.inputs, W)
    assert math.isclose(s.g, 0.75) and math.isclose(s.rho, 4 / 3)
    assert math.isclose(sum(s.terms.values()), s.g)


def test_with_no_dark_leaves_the_overhead_is_the_inverse_of_the_counted_share_times_the_efficiency():
    part = Partition((block(Cut("c", (Blocks("x", Budget(200, 30), 2),), {"undecl.gpu": 100})),), 0.0)
    for efficiency in (1.0, 2.5):
        s = overhead(part, lambda bud: bud.inputs, W, efficiency=efficiency)
        assert s.dark == 0.0 and math.isclose(s.rho, efficiency / s.g)


def test_least_takes_the_smallest_cap():
    a = least(lambda b: 5.0, lambda b: 3.0, lambda b: math.inf)
    assert a(Budget(1, 1)) == 3.0 and least()(Budget(1, 1)) == math.inf
