"""`DsaMqaLogits_v4` is a composite over its keys: it builds at a long row's key count, and one key's logits lower to the circuit of
its one `DsaMqaLogit_v4` call, so the served N = 1 units keep their circuit."""
from __future__ import annotations

import pytest

from circuit_check import checks as CK


def _lowered(t) -> tuple[str, int]:
    """The C-Flock lowering of ``t``: its circuit type digest (a pin's ``type``) and its ANDs."""
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    C = gf2.Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for _, pt in t.signature()[0] for w in pt.leaf_widths())]
    for k, o in enumerate(IL.lower(C, t, ins)):
        C.output(f"out{k}", o)
    return CK.lowering_type(C), C.counts()["ands"]


def test_a_long_rows_logits_are_one_logit_a_key():
    """The composite's ANDs at GLM-5.3's 131,072 keys, read from its descriptor without tracing a row (a traced one would be
    some 10^11 gates at H 32, D 128)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsa as BD
    from verity_flock.ir_lower import definition_ands

    one = definition_ands(bind(BD.DsaMqaLogitV4, H=8, D=32))
    assert definition_ands(bind(BD.DsaMqaLogitsV4, N=131072, H=8, D=32)) == 131072 * one


@pytest.mark.slow
def test_one_keys_logits_lower_as_its_logit():
    """Catches a composite whose N = 1 lowering differs from the traced form it replaced (whose single key's circuit is
    `DsaMqaLogit_v4`'s), which the served units, cut to one key each, would carry: no cheaper test lowers both."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsa as BD

    assert _lowered(bind(BD.DsaMqaLogitsV4, N=1, H=8, D=32)) == _lowered(bind(BD.DsaMqaLogitV4, H=8, D=32))
