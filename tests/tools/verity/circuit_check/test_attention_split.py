"""The attention family's split into Definitions and vocabulary kept its Definitions, pins and bindings (`attention_split.py`).

Here: the snapshot's family Definitions and pins are kept, and each Definition it holds is still at a circuit-check binding, pin or
served binding, which binds the very object its id makes.
"""
from __future__ import annotations

import attention_split as AS
import pytest

SNAPSHOT = AS.SNAPSHOT


@pytest.mark.slow
def test_every_family_definition_and_pin_the_snapshot_names_is_kept():
    fam = AS.family()
    assert sorted(set(SNAPSHOT["family"]) - set(fam)) == []
    pins = AS.pinned(fam)
    assert {k: pins.get(k) for k in SNAPSHOT["pins"]} == SNAPSHOT["pins"]


@pytest.mark.slow
def test_the_served_bindings_are_unchanged():
    """`targets.attention_spec` binds the same Definition (`MASKED_FROM` included), or refuses the same way, at every served shape,
    and each is the very Definition its id makes."""
    served = AS.served_bindings()
    ids = {k: fn if isinstance(fn, str) else fn.id for k, fn in served.items()}
    assert {k: ids.get(k) for k in SNAPSHOT["served"]} == SNAPSHOT["served"]
    assert sorted({fn.id for fn in served.values() if not isinstance(fn, str) and AS.made(fn.id) is not fn}) == []


@pytest.mark.slow
def test_circuit_check_binds_every_definition_the_snapshot_holds(catalog):
    """Each Definition the snapshot holds is still at a circuit-check binding, a pin or a served binding, and circuit-check's is the
    very Definition its id makes."""
    fam = AS.family()
    at = {fid for fid in catalog.suite if catalog.reached[fid][0] in fam}
    served = {fn.id for fn in AS.served_bindings().values() if not isinstance(fn, str)}
    assert sorted(set(SNAPSHOT["definitions"]) - at - set(AS.pinned(fam)) - served) == []
    assert sorted(k for k in SNAPSHOT["definitions"] if k in at and AS.made(k) is not catalog.suite[k]) == []
