"""The whole lowering of a primitive whose output leaf is wider than a 64-bit lane, without circuit-check's catalog."""
from __future__ import annotations

import pytest

from circuit_check import checks as CK


@pytest.mark.slow
def test_an_output_wider_than_64_bits_is_compared_limb_by_limb(monkeypatch):
    """`PouwHopperE4m3Window_v1` returns 480 bits from bytes, and C-Flock lowers it through its Boolean version: the lowering
    is compared with the reference on every limb, and one that differs only in its top bit is caught."""
    import verity.ml.protocols.pouw.circuit.boolean  # noqa: F401  (registers the Boolean version that C-Flock lowers it through)
    from verity_flock import ir_lower as IL
    from verity.ml.protocols.pouw.circuit import words as W

    opts, pins = CK.Options(), CK.load_pins()
    e = CK.check_definition(W.PouwHopperE4m3Window, opts, pins=pins, as_call=False)
    (low,) = [r for r in e["correspondence"]["realizations"] if r["via"] == "c-flock:ir_lower"]
    assert low["lowered"] and low["compared"] and low["mismatches"] == 0 and not e["failures"]
    piece = IL.PIECES[W.PouwHopperE4m3Window.id]
    monkeypatch.setitem(IL.PIECES, W.PouwHopperE4m3Window.id, lambda C, *ins: (lambda o: o[:-1] + [o[-1].inv()])(piece(C, *ins)))
    e = CK.check_definition(W.PouwHopperE4m3Window, opts, pins=pins, as_call=False)
    assert any(f["name"].startswith("correspondence/lowering-mismatch PouwHopperE4m3Window_v1 [c-flock]") for f in e["failures"])
