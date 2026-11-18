from __future__ import annotations

import pytest

from verity.circuits import boundary, census, cross_call, cut
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Coll, Explicit, Grid, Strided, recut
from verity.circuits.types import Array, Value

BIT = Value(1)
Not1 = PrimitiveDefinition("TRecutNot1", 1, (("a", BIT),), BIT, lambda a: a ^ 1, register=False)


def _toy(B, S, s):
    sp, ix, _ = s.refs[0]
    return B.batch(Not1, Coll(Array(32, BIT), Affine(sp, ix, 0, 32)), axes=(0,))


#: one recut read: a run input of two 16-bit leaves read as its 32 bits
TOY = Program(bind(CompositeDefinition("TRecutToy", 1, (), lambda S: ((("s", Array(2, Value(16))),), Array(32, BIT)), _toy,
                                       register=False)))


def test_a_reader_whose_width_divides_the_registered_width_reads_q_leaves_a_registered_leaf():
    assert recut(16, 8) == 2 and recut(16, 1) == 16 and recut(32, 32) == 1
    assert recut(16, 8, Affine("n", 0, 4, 6), 5) == 2
    assert recut(16, 8, Grid(2, (("n", 0, 0, 1, ((2, 4),)), ("n", 0, 2, 1, ((2, 4),)))), 4) == 2
    assert recut(16, 8, Strided("n", 0, 0, 4, 8, 1, 8), 8) == 2


@pytest.mark.parametrize("args, name", [
    ((16, 32), "recut-width"),
    ((16, 6), "recut-width"),
    ((16, 8, Affine("n", 0, 1, 2), 4), "recut-boundary"),
    ((16, 8, Affine("n", 0, 0, 3), 4), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 3, 4, 1, 6), 8), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 2, 3, 1, 4), 8), "recut-boundary"),
    ((16, 8, Strided("n", 0, 0, 2, 4, 2, 4), 8), "recut-boundary"),
    ((16, 8, Explicit((("n", 0, 0),)), 4), "recut-boundary"),
    ((16, 8, Affine("n", 0, 6, 4), 4), "recut-overrun"),
    ((16, 8, Strided("n", 0, 0, 4, 8, 1, 8), 5), "recut-overrun"),
])
def test_a_read_not_of_whole_registered_leaves_inside_its_node_is_refused_by_name(args, name):
    with pytest.raises(ValueError, match=name):
        recut(*args)


def test_the_toy_reads_its_input_node_recut_sixteen_ways():
    assert TOY.root.body.recut == {0: 16}


@pytest.mark.parametrize("where, read", [
    ("recut-gates", lambda: TOY.circuit),
    ("recut-unread: cross_call.analyze", lambda: cross_call.analyze(TOY.root)),
    ("recut-unread: cross_call.members_check", lambda: cross_call.members_check(TOY.root)),
    ("recut-unread: cut.call_scope", lambda: cut.evaluate_definition(TOY.root, 64)),
    ("recut-unread: cut.call_scope", lambda: cut.evaluate_call(TOY.root)),
    ("recut-unread: boundary.consumer_index", lambda: boundary.consumer_index(TOY.root)),
    ("recut-unread: boundary.used_param_intervals", lambda: boundary.used_param_intervals(TOY.root)),
    ("recut-unread: census.census", lambda: census.census(TOY)),
])
def test_a_pass_that_reads_whole_leaves_refuses_a_recut_read_by_name(where, read):
    with pytest.raises(ValueError, match=where):
        read()
