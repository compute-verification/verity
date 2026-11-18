from __future__ import annotations

import pytest

from verity.circuits.codec import definitions
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Coll
from verity.circuits.types import Array, Value
from verity.protocols.proofs.spot.verifier import second_stage as SS

BIT = Value(1)
Not1 = PrimitiveDefinition("TSecondRecutNot1", 1, (("a", BIT),), BIT, lambda a: a ^ 1, register=False)


def _toy(B, S, s):
    sp, ix, _ = s.refs[0]
    return B.batch(Not1, Coll(Array(32, BIT), Affine(sp, ix, 0, 32)), axes=(0,))


TOY = Program(bind(CompositeDefinition("TSecondRecutToy", 1, (), lambda S: ((("s", Array(2, Value(16))),), Array(32, BIT)),
                                       _toy, register=False)))


@pytest.mark.parametrize("read", [SS.link_runs, SS.wire_runs, SS.returned_bits])
def test_a_replay_block_reading_an_input_node_recut_is_refused_by_name(read):
    defs = definitions(TOY)
    with pytest.raises(ValueError, match="recut-unread: second_stage"):
        read(defs, TOY.root.id)
