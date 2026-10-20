"""`circuit.h1window`: a window's Program over a tile unit derives one unit per tile under the tile's template-instance
query, numbered call by call and row block major, and `unit_reads` gives each unit's rows of the window's parameters from
the Program's references: its call's A block, its own B rows, the salt, its position and the roots. A stand-in tile of the
H1 tile's parameter shape keeps the Program small; `benchmarks/pouw/tests/test_h1_prove.py` runs the real one."""
from __future__ import annotations

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.types import Array, Value

from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P

V16 = Value(16)
XOR16 = PrimitiveDefinition("H1WindowTestXor16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: a ^ b, register=False)
R, C = 4, 8


def _tile_sig(S):
    return ((("a", Array(R, Array(2, V16))), ("b", Array(C, Array(2, V16))), ("salt", Array(4, V16)), ("position", V16),
             ("roots", Array(2, Array(4, V16)))), V16)


def _tile_body(B, S, a, b, salt, position, roots):
    return B.call(XOR16, B.call(XOR16, a[0][0], b[0][0]), B.call(XOR16, salt[0], B.call(XOR16, position, roots[0][0])))


TILE = bind(CompositeDefinition("H1WindowTestTile", 0, (), _tile_sig, _tile_body, register=False))
CALLS = (("q_proj", 16, 0), ("k_proj", 8, 0), ("o_proj", 16, 1))


def _tiles(m: int) -> list[tuple[int, int, int]]:
    """Each tile's (A block row, B row, call) in the window's numbering, written out independently of the Program."""
    out, b0 = [], 0
    for c, (_name, n, blk) in enumerate(CALLS):
        for ti in range(m // R):
            for tj in range(n // C):
                out.append((blk * m + R * ti, b0 + C * tj, c))
        b0 += n
    return out


def test_the_partition_derives_one_unit_per_tile_and_each_reads_its_rows():
    m = HW.slot(5)
    prog = HW.program(TILE, CALLS, m)
    tp = P.evaluate(prog, HW.query(TILE))
    tiles = _tiles(m)
    assert m == 8 and tp.population == len(tiles) == 2 * 2 + 2 * 1 + 2 * 2
    assert [(first, n) for _k, _t, first, n in tp.ranges] == [(u, 1) for u in range(len(tiles))]
    assert dict(prog.fn.signature()[0])["a"].n == 2 * m and dict(prog.fn.signature()[0])["b"].n == 40
    for u, (a0, b0, _c) in enumerate(tiles):
        assert HW.unit_reads(prog, tp, u) == {
            **{f"a[{i}]": ("a", a0 + i) for i in range(R)}, **{f"b[{j}]": ("b", b0 + j) for j in range(C)},
            "salt": ("salt", 0), "position": ("position", u), "roots": ("roots", 0), "z": ("z", u), "receipt": ("receipt", u)}


def test_the_parameters_take_the_names_a_served_path_gives_them():
    names = ("h1/w/a", "h1/w/b", "h1/w/salt", "h1/w/position", "h1/w/roots")
    prog = HW.program(TILE, CALLS, 8, names)
    tp = P.evaluate(prog, HW.query(TILE))
    assert [n for n, _t in prog.fn.signature()[0]] == list(names)
    got = HW.unit_reads(prog, tp, 5, outputs=("h1/w/z", "h1/w/receipt"))
    assert got["a[1]"] == ("h1/w/a", 4 + 1) and got["b[0]"] == ("h1/w/b", 16) and got["receipt"] == ("h1/w/receipt", 5)
    assert P.program_digest(prog) != P.program_digest(HW.program(TILE, CALLS, 8))


@pytest.mark.parametrize("calls,m,names,why", [
    (CALLS, 6, HW.ROLES, "power of two"),
    (CALLS, 2, HW.ROLES, "power of two"),
    ((("q_proj", 12, 0),), 8, HW.ROLES, "whole blocks"),
    ((("q_proj", 16, 0), ("o_proj", 16, 2)), 8, HW.ROLES, "every A block"),
    (CALLS, 8, ("a", "a", "salt", "position", "roots"), "once each"),
])
def test_a_layout_off_its_fixed_slots_is_refused(calls, m, names, why):
    with pytest.raises(ValueError, match=why):
        HW.program(TILE, calls, m, names)
