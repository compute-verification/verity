"""`DsaMqaLogits_v4` is a composite over its keys, `DsaMqaLogitsQ_v1` once a row and `DsaMqaLogitsKey_v1` a key: it builds at a
long row's key count, and one key's logits lower to the ANDs of its one `DsaMqaLogit_v4` call, whose circuit the two blocks cut."""
from __future__ import annotations

import pytest


def _lowered_ands(t) -> int:
    """The ANDs of ``t``'s C-Flock lowering."""
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    C = gf2.Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for _, pt in t.signature()[0] for w in pt.leaf_widths())]
    for k, o in enumerate(IL.lower(C, t, ins)):
        C.output(f"out{k}", o)
    return C.counts()["ands"]


def test_a_long_rows_logits_are_its_q_once_and_a_key_block_a_key():
    """The composite's ANDs at GLM-5.3's 131,072 keys, read from its descriptor without tracing a row (a traced one would be
    some 10^11 gates at H 32, D 128); the cut keeps each of `DsaMqaLogit_v4`'s ANDs once."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa as BD
    from verity_flock.ir_lower import definition_ands

    q, key = definition_ands(bind(BD.DsaMqaLogitsQ, H=8, D=32)), definition_ands(bind(BD.DsaMqaLogitsKey, H=8, D=32))
    assert q + key == definition_ands(bind(BD.DsaMqaLogitV4, H=8, D=32))
    assert definition_ands(bind(BD.DsaMqaLogitsV4, N=131072, H=8, D=32)) == q + 131072 * key


@pytest.mark.slow
def test_one_keys_logits_lower_to_its_logits_ands():
    """Catches a lowering of the composite's Call and batch that repeats or drops a gate its descriptor counts once: no cheaper
    test lowers both.  Only the ANDs: the cut's circuit is not the logit's gate for gate."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa as BD
    from verity_flock.ir_lower import definition_ands

    one = bind(BD.DsaMqaLogitsV4, N=1, H=8, D=32)
    assert _lowered_ands(one) == _lowered_ands(bind(BD.DsaMqaLogitV4, H=8, D=32)) == definition_ands(one)
