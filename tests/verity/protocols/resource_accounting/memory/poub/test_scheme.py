"""PoUB's declaration as a resource scheme (`verity.protocols.resource_accounting.memory.poub.scheme.SCHEME`)."""

from __future__ import annotations

import pytest

from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.memory.poub import scheme as poub
from verity.protocols.resource_accounting.memory.poub.verifier import p2, p2_parties


def test_registered_under_the_name_a_configuration_lists() -> None:
    assert S.get("bandwidth") is poub.SCHEME
    assert (p2_parties.COMPONENT, p2_parties.PROTOCOL) == (poub.SCHEME.name, poub.SCHEME.protocol)
    assert poub.SCHEME.unit == "bits"


def test_a_block_costs_the_bits_it_reads() -> None:
    [block] = poub.SCHEME.blocks
    assert block.cost({"BYTES": p2.UNIT_BYTES}) == 65_792


def test_the_roots_deadline_is_the_sessions() -> None:
    [timed] = [o for o in poub.SCHEME.log if o.within_ns is not None]
    assert timed.within_ns == p2_parties.SESSION["deadline_ns"] and timed.assumption in poub.SCHEME.assumptions


def test_refresh_is_its_one_setting_off_by_default() -> None:
    assert poub.settings({}) == {"refresh": False} == poub.settings({"refresh": False})
    with pytest.raises(NotImplementedError, match=poub.REFRESH_GOAL):
        poub.settings({"refresh": True})
    with pytest.raises(ValueError, match="has no setting"):
        poub.settings({"lifetime": 2})
    with pytest.raises(ValueError, match="is a bool"):
        poub.settings({"refresh": 1})
