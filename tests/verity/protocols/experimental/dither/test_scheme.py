"""Dither as a resource scheme (`verity.protocols.experimental.dither.scheme`)."""
from __future__ import annotations

from fractions import Fraction

import pytest

from verity.protocols.experimental.dither import scheme as DS
from verity.protocols.experimental.dither.prover import dither as D
from verity.protocols.resource_accounting import scheme as S


def test_found_by_name() -> None:
    assert S.SCHEMES["dither"] is DS.DITHER
    assert {b.definition for b in DS.DITHER.blocks} == {D.PouwDitherBlock.id, D.PouwDitherLast.id}


def test_block_cost_is_an_output_cells_credit() -> None:
    sh = D.Shape(128, 128, 2048, lp=128, L=512)
    cost = {b.definition: b.cost for b in DS.DITHER.blocks}
    cells = [D.block_definition(sh, s) for s in range(1, sh.S + 1)]
    assert sum(cost[d.base_id](d.bindings) for d in cells) == sh.k - sh.L


def test_bound_refuses_a_charge_it_does_not_state() -> None:
    names = {c.name: 1 for c in DS.DITHER.charges}
    with pytest.raises(ValueError, match="dither: its delta_scheme .* is not said"):
        S.bound(DS.DITHER, Fraction(1, 10**6), 5, names)
    with pytest.raises(ValueError, match="charges are"):
        S.bound(DS.DITHER, Fraction(1, 10**6), 0, {"inert-work": 0})


def test_attach_waits_on_its_kernels() -> None:
    with pytest.raises(NotImplementedError, match="kernels builds them"):
        DS.DITHER.attach(object(), {})


def test_it_says_why_it_declares_no_tables_output_format_or_y_commitment() -> None:
    for field in ("tables", "number_format", "y_committed"):
        with pytest.raises(LookupError, match=f"dither declares no {field}: experimental: no application lists dither"):
            S.declared(DS.DITHER, field)
