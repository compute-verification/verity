"""fold_xor_search.py: its menu is the sheet's, its forms compute what they model, and its CEGIS proves and finds."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

POUW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POUW / "theory"))
sys.path.insert(0, str(POUW))
try:
    import fold_xor_search as S
    import z3
    from facet_break.loads import class_of
finally:
    sys.path.remove(str(POUW / "theory"))
    sys.path.remove(str(POUW))


def _val(e) -> int:
    return z3.simplify(e).as_long()


def test_cover_is_the_sheets_menu():
    """Every opcode the search covers is one DS-100 prices on the IMAD or FP pipe, every form it names is modelled, and
    what's off the menu isn't; fold_xor.py's menu and its 78's SASS forms are among the covered."""
    assert {op: class_of(op) for op in S.COVER} == {op: "FFMA" if S.COVER[op] in S.FP_OPS else "IMAD" for op in S.COVER}
    assert set(S.COVER.values()) == set(S.FORMS) == set(S.INT_OPS + S.FP_OPS)
    assert all(class_of(op) not in ("IMAD", "FFMA") for op in S.OFF_MENU)
    assert {"IMAD", "IMAD.HI.U32", "IDP.4A.U8.U8", "IDP.2A.LO.U16.U8", "FFMA2", "FADD2", "FMUL2", "FFMA"} <= set(S.COVER)


def test_forms_compute_what_they_model():
    """The integer forms against Python on concrete words, with the modifiers off and on, and an FFMA lane against
    numpy's binary32 add (b = 1.0) and its canonical NaN."""
    off = {n: z3.BoolVal(False) for n in ("sa", "sb", "na", "nb", "np", "nc", "inv", "cx", "hi")}
    rng = np.random.default_rng(7)
    for a, b, c, h in rng.integers(0, 2**32, size=(64, 4), dtype=np.uint64).tolist():
        o = [S.bv(a), S.bv(b), S.bv(c), S.bv(h)]
        assert _val(S.imad(o, off)[0]) == (a * b + c) % 2**32
        on = {**off, "np": z3.BoolVal(True), "nc": z3.BoolVal(True), "cx": z3.BoolVal(True)}
        assert _val(S.imad(o, on)[0]) == (-(a * b) - c + 1) % 2**32
        s = a * b + (h << 32 | c)
        assert [_val(v) for v in S.imad_hi(o, off)] == [s % 2**32, (s >> 32) % 2**32]
        sa = a - 2**32 if a >> 31 else a
        signed = {**off, "sa": z3.BoolVal(True)}
        assert _val(S.imad_hi(o, signed)[1]) == ((sa * b + (h << 32 | c)) >> 32) % 2**32
        dp = c + sum(((a >> 8 * i) & 255) * ((b >> 8 * i) & 255) for i in range(4))
        assert _val(S.idp4a(o, off)[0]) == dp % 2**32
    m = {n + "0": z3.BoolVal(False) for n in ("ftz", "sat", "satz", "fmz", "fmzx", "absa", "absb", "absc", "nega",
                                               "negb", "negc")}
    m["rm0"] = z3.RNE()
    one = S.bv(0x3F800000)
    for x, y in rng.integers(0, 2**32, size=(32, 2), dtype=np.uint64).tolist():
        fx, fy = (np.frombuffer(struct.pack("<I", v), dtype=np.float32)[0] for v in (x, y))
        with np.errstate(all="ignore"):
            r = np.float32(fx + fy)
        want = 0x7FFFFFFF if np.isnan(r) else struct.unpack("<I", r.tobytes())[0]
        assert _val(S.lane(S.bv(x), one, S.bv(y), m, "0")) == want


@pytest.mark.slow
def test_no_one_instruction_program():
    """k = 1: no single menu instruction computes x ^ y ^ K for any K (every sequence unsat)."""
    r = S.search(1)
    assert r["proved"] and r["sequences"] == 9, r["tally"]


@pytest.mark.slow
def test_it_finds_programs_that_exist():
    """The search isn't vacuous: it finds x + y + K (one IMAD) and hi32(x·y) + K (one IMAD.HI on x and y)."""
    assert S.cegis(("IMAD",), "add")["result"] == "FOUND"
    assert S.cegis((("IMAD.HI", (0, 1)),), "mulhi")["result"] == "FOUND"
