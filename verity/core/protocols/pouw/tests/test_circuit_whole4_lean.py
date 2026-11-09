"""The Lean whole-Program table is `pc4`'s `Pc4Check`, as `test_circuit_whole_lean.py` ties `whole8` to `Pc8Check`.
`lean_layout` restates `WholeCircuit.whole4` (`Security/Definitions/Pouw/PearlC/WholeCircuit4.lean`): `pc8`'s input regions at
`inW4`'s widths, the blocks (`row4B`, `row4A`, `tile4Spec`) and their ports (`portB8`, `portA8`, `portT4`, with `code4`,
`scale4`, `paOut4` and `pbOut4` naming the rows' outputs), and the outputs (`outs4`). `check_whole` checks every computed
gate's template gate, owner, width and relocated reads, and each parameter leaf's producer exactly, so it and the pinned
descriptor digests tie `tileCheck4`'s circuit to the Program. No primitive is evaluated."""
from __future__ import annotations

import pytest

from verity.ml.circuits import Program
from verity.ml.circuits.codec import program_digest

pc4 = pytest.importorskip("verity.ml.protocols.pouw.circuit.pc4", reason="the Program is `verity.ml.protocols.pouw.circuit.pc4`'s")

from test_circuit_whole_lean import check_whole, row_in  # noqa: E402

#: `program_digest` of the `Pc4Check` Programs the Lean layout was checked against; a change to any needs the Lean re-read
DIGESTS = {
    (2, 64, 3, (1, 1)): "0e4ecd8175f6ef252d105fcb5058ce4b47204ad424b7f8886644e6a7812db21e9fae17de76943643152742281b691d827091ed6d22beb28ec4558cab917ac594",
    (4, 128, 4, (2, 2)): "4a301d7956afb808c42d13bb22fdc5113ca10d9023e7af14f150c19eb2ff64be4f4686f02e41028b25f5bb2a5fc2cda41b89b82ac86212412abef4a72bdd9349",
    (2, 128, 2, (1, 2)): "c5d7e67421d147d96b3094a7ca3600bb5f78e23b0103a30987538aeb32339e221d62024778b5fa9acc664dc62947bba6988140090f2970d68bf395121d428c28",
}


# ---- `Row4Circuit`'s offsets -----------------------------------------------------------------------------------------------

def post4_at(k: int, b: int) -> int:
    """`post4At K b`, through `fmaAt` … `bk4At`."""
    n8 = k // 8
    rho = b + 2 * (k - 1) + n8 + min(16, n8) + min(8, n8) + min(4, n8) + min(2, n8) + min(1, n8)
    return rho + 2 + 13 + 2 * k + 52 * (k // 16)


def code4(k: int, b: int, t: int) -> int:
    """`code4 K b t`: `bk4At K b + 52·(t / 16) + 21 + 2·(t mod 16)`."""
    return post4_at(k, b) - 52 * (k // 16) + 52 * (t // 16) + 21 + 2 * (t % 16)


def scale4(k: int, b: int, i: int) -> int:
    """`scale4 K b i`: `bk4At K b + 52·i + 17`."""
    return post4_at(k, b) - 52 * (k // 16) + 52 * i + 17


def pa_out4(k: int, w: int) -> int:
    """`paOut4 K w`."""
    nw, b = k // 64, row_in(k, False)
    return post4_at(k, b) + (32 * (nw + 1) + w if w < 32 else (w - 32) * (nw + 1) + nw)


def pb_out4(k: int, w: int) -> int:
    """`pbOut4 K w`."""
    nw, b = k // 64, row_in(k, True)
    return post4_at(k, b) + 32 + (w * (nw + 35) + nw + 34 if w < 32 else 32 * (nw + 35) + (w - 32))


def row_n(k: int, b: bool) -> int:
    """`row4A_N`, `row4B_N`: the row tables' gate counts."""
    nw = k // 64
    return post4_at(k, row_in(k, b)) + (32 + 32 * (nw + 35) + 33 if b else 32 * (nw + 1) + 33)


def tile_in(k: int, tm: int, tn: int) -> int:
    """`tile4In K TM TN`."""
    return (tm + tn) * (k + k // 16) + 64 * (tm + tn)


def block_len(k: int) -> int:
    """`block4Len K`."""
    return k // 64 + 4


# ---- `WholeCircuit.whole4` -------------------------------------------------------------------------------------------------

def lean_layout(m: int, k: int, n: int, tm: int, tn: int):
    """`whole4`'s input count and widths, its blocks as (role, instances, template inputs, computed gates, port), and its
    outputs in `Pc4Check`'s return order."""
    ea, b = m * k, m * (k + 32)
    eb, fa = b + n * k, (m + n) * (k + 32)
    fb, g = fa + 32 * k, fa + 64 * k
    nin = g + 1024
    widths = [16] * ea + [4] * (b - ea) + [16] * (eb - b) + [4] * (g - eb) + [32] * 1024
    lb, la = row_n(k, True) - row_in(k, True), row_n(k, False) - row_in(k, False)
    lt = tm * tn * block_len(k)
    s, r4 = k // 16, k + k // 16

    def port_a(i, a):
        if a < k:
            return i * k + a
        if a < k + 32:
            return ea + 32 * i + (a - k)
        return fa + (a - (k + 32)) if a < 33 * k + 32 else fb + (a - (33 * k + 32))

    def port_b(j, a):
        if a < k:
            return b + j * k + a
        if a < k + 32:
            return eb + 32 * j + (a - k)
        if a < 33 * k + 32:
            return fb + (a - (k + 32))
        return fa + (a - (33 * k + 32)) if a < 65 * k + 32 else g + (a - (65 * k + 32))

    row_b = lambda j, a: nin + j * lb + (a - row_in(k, True))                  # noqa: E731
    row_a = lambda i, a: nin + n * lb + i * la + (a - row_in(k, False))        # noqa: E731
    tiles = nin + n * lb + m * la

    def port_t(e, a):
        rs, cs = e // (n // tn), e % (n // tn)
        if a < tm * k:
            return row_a(rs * tm + a // k, code4(k, row_in(k, False), a % k))
        if a < tm * r4:
            a -= tm * k
            return row_a(rs * tm + a // s, scale4(k, row_in(k, False), a % s))
        if a < tm * r4 + tn * k:
            a -= tm * r4
            return row_b(cs * tn + a // k, code4(k, row_in(k, True), a % k))
        if a < (tm + tn) * r4:
            a -= tm * r4 + tn * k
            return row_b(cs * tn + a // s, scale4(k, row_in(k, True), a % s))
        a -= (tm + tn) * r4
        if a < 64 * tm:
            return row_a(rs * tm + a // 64, pa_out4(k, a % 64))
        a -= 64 * tm
        return row_b(cs * tn + a // 64, pb_out4(k, a % 64))

    blocks = [("row_b", n, row_in(k, True), lb, port_b), ("row_a", m, row_in(k, False), la, port_a),
              ("tile", (m // tm) * (n // tn), tile_in(k, tm, tn), lt, port_t)]

    def entry(i, j, last):
        e, ti, tj = i // tm * (n // tn) + j // tn, i % tm, j % tn
        r = (ti * tn + tj) * block_len(k) + (block_len(k) - 1 if last else k // 64 - 1)
        return tiles + e * lt + r

    nw = k // 64
    outs = ([entry(i, j, False) for i in range(m) for j in range(n)] + [entry(i, j, True) for i in range(m) for j in range(n)]
            + [row_a(i, post4_at(k, row_in(k, False)) + 32 * (nw + 1) + 32) for i in range(m)]
            + [row_b(j, post4_at(k, row_in(k, True)) + 32 + 32 * (nw + 35) + 32) for j in range(n)])
    return nin, widths, blocks, outs


CASES = list(DIGESTS)


def check_layout(m: int, k: int, n: int, tile: tuple[int, int]) -> None:
    check_whole(pc4.check(pc4.Shape(m, k, n, tile)), pc4.unit_cut, pc4.units, pc4.templates(k, tile),
                lean_layout(m, k, n, *tile))


@pytest.mark.parametrize("m,k,n,tile", CASES)
def test_the_lean_whole_table_is_pc4_check(m, k, n, tile):
    check_layout(m, k, n, tile)


@pytest.mark.parametrize("m,k,n,tile", CASES)
def test_the_checked_programs_are_pinned(m, k, n, tile):
    assert program_digest(Program(pc4.check(pc4.Shape(m, k, n, tile)))) == DIGESTS[m, k, n, tile]


@pytest.mark.slow
def test_the_lean_whole_table_at_the_deployed_tile():
    """Catches a wiring the small tiles don't exercise, such as one gated on the tile being `pc4.TILE`: one 64 x 64 tile
    over 64 rows of A and B at K = 64."""
    assert pc4.TILE == (64, 64)
    check_layout(64, 64, 64, (64, 64))
