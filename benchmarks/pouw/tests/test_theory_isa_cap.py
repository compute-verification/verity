"""One IMAD writes 4 single-entry fields and one IMAD.WIDE 6, above qstar's "IMAD ≤ 3" and the ISA cap's 32/7."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import isa_cap_check as ic


def test_outer_product_packings():
    """IMAD's 2 × 2 and signed IMAD.WIDE's 3 × 2 outer products decode exactly at every R bit and U entry; read as
    IMAD.WIDE.U32 the high word breaks."""
    assert ic.outer_imad(2, 2, 16, 32)
    assert ic.outer_imad(3, 2, 24, 64)
    assert not ic.outer_imad(3, 2, 24, 64, signed=False)


def test_idp2a_packs_five():
    """IDP.2A.LO writes 5 incidences into one word at radices 129 and 129·257, and one value short at either fails."""
    assert ic.idp2a_packing()
    assert not ic.idp2a_packing(m3=129 * 256)
    assert not ic.idp2a_packing(m2=128, m3=128 * 257)


def test_load_bound_against_the_cap():
    """QSTAR's 6.0 / 3.5 B at κ = 32/7, and 5.3 / 3.0 B at the 8 one IMAD.WIDE reaches."""
    assert round(ic.load_bound(32 / 7, 128), 2) == 6.04 and round(ic.load_bound(32 / 7, 216), 2) == 3.53
    assert round(ic.load_bound(8, 128), 2) == 5.27 and round(ic.load_bound(8, 216), 2) == 3.04


def test_table_route_prefers_loads_below_the_flip():
    """The table route's saving per load falls with ALU creation below κ = 28/(1.877 − 1) and rises above it."""
    for c_load in (128, 216, 108):
        assert ic.table_eps(0.1, 8, c_load) < ic.table_eps(0, 8, c_load)
        assert ic.table_eps(0.1, 33, c_load) > ic.table_eps(0, 33, c_load)
