"""One IMAD writes 4 single-entry fields and one IMAD.WIDE 6, above QSTAR's "IMAD ≤ 3" and ISA-CAP's 32/7."""
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


def test_load_bound_against_the_cap():
    """QSTAR's 6.0 / 3.5 B at κ = 32/7, and 5.3 / 3.0 B at the 8 one IMAD.WIDE reaches."""
    assert round(ic.load_bound(32 / 7, 128), 2) == 6.04 and round(ic.load_bound(32 / 7, 216), 2) == 3.53
    assert round(ic.load_bound(8, 128), 2) == 5.27 and round(ic.load_bound(8, 216), 2) == 3.04
