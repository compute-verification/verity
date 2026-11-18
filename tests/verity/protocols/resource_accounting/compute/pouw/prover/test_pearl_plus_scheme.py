"""Pearl+ as a resource scheme (`verity.protocols.resource_accounting.compute.pouw.scheme`). Its block cost against the
run verifier's credit is `tests/benchmarks/pouw/test_pearl_plus_scheme_credit.py`, beside the verifier's other tests."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity.protocols.resource_accounting import scheme as S
from verity.protocols.resource_accounting.compute.pouw import scheme as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_c_tile import PearlCTileRead
from verity.protocols.resource_accounting.compute.pouw.prover.circuit.pearl_plus_tile import PearlPlusTile
from verity.protocols.resource_accounting.compute.pouw.prover.schemes.pearl_c import PearlC


def test_found_by_name() -> None:
    assert S.get("pearl-plus") is PS.PEARL_PLUS
    assert {b.definition for b in PS.PEARL_PLUS.blocks} == {PearlPlusTile.id, PearlCTileRead.id}
    assert PS.PEARL_PLUS.option == f"pouw:{PearlC().name}"


def test_bound_refuses_a_charge_it_does_not_state() -> None:
    names = {c.name: 0 for c in PS.PEARL_PLUS.charges}
    with pytest.raises(ValueError, match="pearl-plus: its delta_scheme .* is not said"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, names)
    with pytest.raises(ValueError, match="charges are"):
        S.bound(PS.PEARL_PLUS, Fraction(1, 100), 0, {**names, "rebuilt": 0})
