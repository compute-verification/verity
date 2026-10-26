"""The hierarchical measurement against the gate-by-gate definitions and against values computed by hand, on circuits
covering every node form (`synthetic`)."""

from __future__ import annotations

import random

import pytest
import synthetic as SY
from circuit_properties import flat
from circuit_properties.measure import Ambiguous, Blocks, measure, whole
from circuit_properties.ops import Designation, NotSeparable, Site, op_shape

from verity.ml.circuits.defs import bind


def test_op_shape_by_hand():
    shape = op_shape(bind(SY.MulLanes, N=2), SY.DESIGNATION.sites[0])
    assert shape.lanes == 2 and shape.sizes == [8, 7]
    assert not set(shape.members[0]) & set(shape.members[1])
    assert shape.reach == [0b11, 0b11]
    assert all(shape.rep[k] in shape.members[k] for k in range(2))


def test_net_by_hand_and_flat():
    p = SY.net()
    m = measure(p, SY.DESIGNATION)
    f = flat.properties(p, SY.DESIGNATION)
    got = {k: getattr(m, k) for k in SY.NET}
    assert got == SY.NET
    assert (f.products, f.s_x, f.a_out, f.a_out_values, f.activation_values, f.constant_leaves, f.a_in) == \
        (m.products, m.s_x, m.a_out, m.a_out_values, m.activation_values, m.constant_leaves, m.a_in)
    assert f.a_in == min(c for cs in f.entries.values() for c in cs if c)
    rows = {r.id: r for r in m.entries}
    assert rows["LinScan_v1{T=3,N=2}"].reached_min == 2 and rows["LinScan_v1{T=3,N=2}"].reached_max == 6
    assert rows["Coord_v1{K=4,N=2}"].activations == 4 + 3 + 2 + 2


def test_without_within_every_activation_counts():
    d = Designation(sites=(Site(op=r"MulLanes_v\d+", x="a", w="b"),), entries=SY.DESIGNATION.entries)
    p = SY.net()
    m, f = measure(p, d), flat.properties(p, d)
    assert m.products == f.products == SY.NET["products"] + 2 and m.excluded == {}
    assert (m.a_out, m.a_in) == (f.a_out, f.a_in)


def test_one_product_per_activation():
    d = Designation(sites=(Site(op=r"MulLanes_v\d+", x="a", w="b", lanes=False, within=r"Lin\w*_v\d+"),),
                    entries=SY.DESIGNATION.entries)
    p = SY.net()
    m, f = measure(p, d), flat.properties(p, d)
    assert m.products == f.products == SY.NET["products"] // 2
    assert (m.s_x, m.a_out, m.a_in, m.constant_leaves) == (f.s_x, f.a_out, f.a_in, f.constant_leaves)


def test_one_value_in_two_lanes_is_refused_not_miscounted():
    p = SY.dup()
    with pytest.raises(Ambiguous):
        measure(p, SY.DUP_DESIGNATION)
    assert flat.properties(p, SY.DUP_DESIGNATION).a_out == 2


def test_a_word_primitive_has_no_products_of_its_own():
    from verity.ml.circuits.ops import gemm as G
    from verity.ml.circuits.program import Program
    p = Program(bind(G.Gemm, K=16, N=1, **G.GEMM_AMPERE))
    with pytest.raises(NotSeparable):
        measure(p, Designation(sites=(Site(op=r"AmpereBF16TcDot16_v\d+", x="a", w="b"),)))


@pytest.mark.parametrize("seed", range(6))
def test_whole_matches_flat_on_random_partitions(seed):
    p = SY.net()
    ex = flat.explicit(p, SY.DESIGNATION)
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, p.gates), rng.randint(1, 40)))
    runs = [(0, 0)] + [(c, i + 1) for i, c in enumerate(cuts)]
    owner = [None] * p.gates
    for (s, b), (e, _b) in zip(runs, runs[1:] + [(p.gates, None)]):
        owner[s:e] = [b] * (e - s)
    w = whole(p, SY.DESIGNATION, Blocks(runs))
    per, split = flat.whole(p, SY.DESIGNATION, owner, ex)
    assert (w.per_block, w.split) == (dict(per), split)
    assert w.whole + w.split == SY.NET["products"]


def test_whole_by_root_calls():
    p = SY.net()
    runs = [(p.root.body.offsets[k], k) for k in range(len(p.root.body.nodes))]
    w = whole(p, SY.DESIGNATION, Blocks(runs))
    assert w.split == 0 and w.whole == SY.NET["products"]
