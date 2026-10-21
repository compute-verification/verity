"""`circuit.h1lane`: a leaf's lanes are its H1 tile cut by lane, so the XOR of their partials is the tile's receipt and their
oks together are the leaf's record check; the combine holds the partials to the receipt, refusing a swapped-in partial of
another leaf and a receipt off the lanes' XOR (a swap of two lanes' partials is the registered route's to refuse, by read
position: XOR doesn't see order); and the leaf's Program cuts into one unit a lane and one for the combine."""
from __future__ import annotations

import random
import struct

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG

from verity.core.protocols.pouw.circuit import h1lane as L
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import seeds as SD

M32 = (1 << 32) - 1
SALT, POSITION, ROOTS = bytes(range(64)), 5, (bytes(64), bytes([7]) * 64)


def _tile(defn=H.PouwH1Tile, r=32, c=32, k=64):
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return bind(defn, EXACT=exact, R=r, C=c, PAIRS=tuple((i, j) for i in range(r) for j in range(c)))


def _f32(rng: random.Random, hi: float) -> int:
    return struct.unpack("<I", struct.pack("<f", rng.uniform(1e-3, hi) / 127))[0]


def _leaf(seed: int, r: int = 32, c: int = 32, k: int = 64):
    rng = random.Random(seed)
    a = [[rng.randrange(-128, 128) & 0xFF for _ in range(k)] for _ in range(r)]
    b = [[rng.randrange(-128, 128) & 0xFF for _ in range(k)] for _ in range(c)]
    sa, sw = [_f32(rng, 0.1) for _ in range(r)], [_f32(rng, 0.5) for _ in range(c)]
    return a, b, sa, sw, Z.records(a, b, sa, sw)


def _lanes(a, b, sa, sw, zy, lay, salt=SALT):
    out = []
    for ln in range(lay.lanes):
        rows, cols = lay.rows(ln), lay.cols(ln)
        out.append(L.lane_reference([a[i] for i in rows], [b[j] for j in cols], [sa[i] for i in rows],
                                    [sw[j] for j in cols], zy, salt, POSITION, ROOTS, lay, ln))
    return out


def test_the_lanes_cut_the_leaf_into_its_cells_once_each():
    lay = H.Layout(32, 32, 64)
    cells = [t for ln in range(lay.lanes) for t in L.cells(lay, ln)]
    assert lay.lanes == 32 and sorted(cells) == list(range(32 * 32))
    assert L.cells(lay, 0)[:8] == [0, 1, 8, 9, 16, 17, 24, 25] and L.cells(lay, 0)[8] == 8 * 32
    assert L.cells(lay, 5)[0] == 1 * 32 + 2, "lane (g, q) = (1, 1): row g, column 2 q"


def test_the_lanes_partials_xor_to_the_tiles_receipt_and_their_oks_are_the_records_check():
    a, b, sa, sw, zy = _leaf(1)
    lay = H.Layout(32, 32, 64)
    z, receipt = H.reference(a, b, SALT, POSITION, ROOTS)
    lanes = _lanes(a, b, sa, sw, zy, lay)
    acc = [0] * len(receipt)
    for p, _ok in lanes:
        acc = [x ^ y for x, y in zip(acc, p)]
    assert len(receipt) == 2 * lay.sets and acc == receipt and zy[0] == z
    assert [ok for _p, ok in lanes] == [1] * 32
    assert L.combine_reference([p for p, _ok in lanes], receipt) == 1
    assert len({tuple(p) for p, _ok in lanes}) == 32, "each lane's partial is its own"


def test_a_lane_whose_cells_are_off_the_product_or_the_dequant_is_refused_and_only_that_lane():
    a, b, sa, sw, (zr, yr) = _leaf(2)
    lay = H.Layout(32, 32, 64)
    t = L.cells(lay, 7)[13]
    bad_y, bad_z = list(yr), list(zr)
    bad_y[t] ^= 1
    bad_z[t] = (bad_z[t] + 1) & M32
    for zy in ((zr, bad_y), (bad_z, yr)):
        assert [ok for _p, ok in _lanes(a, b, sa, sw, zy, lay)] == [int(ln != 7) for ln in range(32)]


def test_the_combine_refuses_another_leafs_partial_and_a_receipt_off_the_lanes_xor():
    a, b, sa, sw, zy = _leaf(3)
    lay = H.Layout(32, 32, 64)
    receipt = H.reference(a, b, SALT, POSITION, ROOTS)[1]
    mine = [p for p, _ok in _lanes(a, b, sa, sw, zy, lay)]
    other = [p for p, _ok in _lanes(a, b, sa, sw, zy, lay, salt=bytes([9]) * 64)]
    swapped = [mine[1], mine[0], *mine[2:]]
    assert L.combine_reference(swapped, receipt) == 1, "XOR is order-free: the route's read positions refuse a swap"
    assert L.combine_reference([*mine[:4], other[4], *mine[5:]], receipt) == 0
    off = list(receipt)
    off[3] ^= 1 << 17
    assert L.combine_reference(mine, off) == 0
    assert L.combine_reference(mine[:31], receipt) == 0, "every lane's partial"


def test_a_lane_of_a_one_lane_tile_is_the_tile_and_its_record_check():
    a, b, sa, sw, zy = _leaf(4, 4, 8)
    lay = H.Layout(4, 8, 64)
    assert lay.lanes == 1
    assert L.lane_reference(a, b, sa, sw, zy, SALT, POSITION, ROOTS, lay, 0) == Z.reference(a, b, sa, sw, zy, SALT,
                                                                                              POSITION, ROOTS)


def test_signatures_and_ports():
    for defn in (H.PouwH1Tile, H.PouwH1Tile3):
        tile = _tile(defn)
        u = L.lane(tile, 5)
        params, ret = u.signature()
        assert [(n, t.leaves) for n, t in params] == [("a", 4 * (8 * 64 + 32)), ("b", 8 * (8 * 64 + 32)),
                                                      ("zy", 48 * 32 * 32),
                                                      *((n, t.leaves) for n, t in tile.signature()[0][2:])]
        lay = H.Layout(32, 32, 64)
        assert [(n, t.leaves) for n, t in ret.fields] == [("partial", 32 * 2 * lay.sets), ("ok", 1)]
        assert {n: p.kind for n, p in u.defn.ports(u.S()).params.items()} == {"salt": "coin", "position": "position"}
        c = L.combine(lay)
        (pn, pt), (rn, rt) = c.signature()[0]
        assert (pn, pt.leaves, rn, rt.leaves, c.signature()[1].leaves) == ("p", 32 * 32 * 2 * lay.sets, "receipt",
                                                                             32 * 2 * lay.sets, 1)
        leaf = bind(L.PouwH1Leaf, TILE=tile)
        assert [n for n, _t in leaf.signature()[0]] == list(L.LEAF_PARAMS)
        assert leaf.signature()[1].leaves == 32 * (32 * 2 * lay.sets + 1) + 1


@pytest.mark.parametrize("bind_,why", [
    (lambda: L.lane(_tile(), 32), "lanes"),
    (lambda: L.lane(_tile(), -1), "lanes"),
    (lambda: L.lane(_tile(H.PouwH1Tile4), 0), "an H1 tile"),
    (lambda: bind(L.PouwH1Combine, N=0, W=4), "positive"),
])
def test_a_lane_out_of_range_or_over_another_tile_is_refused(bind_, why):
    with pytest.raises(ValueError, match=why):
        bind_().signature()


def _bits(v: int, n: int) -> list[int]:
    return [(v >> i) & 1 for i in range(n)]


def _int(bits) -> int:
    return sum(int(x) << i for i, x in enumerate(bits))


def test_the_combine_on_bits_is_its_word_view():
    rng = random.Random(5)
    n, w = 3, 4
    rows = [[rng.randrange(1 << 32) for _ in range(w)] for _ in range(n)]
    receipt = [rows[0][t] ^ rows[1][t] ^ rows[2][t] for t in range(w)]
    off = list(receipt)
    off[2] ^= 1 << 31
    c = bind(L.PouwH1Combine, N=n, W=w)
    flat = [x for r in rows for v in r for x in _bits(v, 32)]
    got = [_int(evaluate(c, flat, [x for v in rc for x in _bits(v, 32)])) for rc in (receipt, off)]
    assert got == [1, 0] == [L.combine_reference(rows, receipt), L.combine_reference(rows, off)]


def _row_bits(codes, scale):
    return [x for v in codes for x in _bits(v, 8)] + _bits(scale, 32)


def _lane_bits(tile, ln, a, b, sa, sw, zy, lay):
    rows, cols = lay.rows(ln), lay.cols(ln)
    out = evaluate(L.lane(tile, ln), [x for i in rows for x in _row_bits(a[i], sa[i])],
                   [x for j in cols for x in _row_bits(b[j], sw[j])],
                   [x for v in zy[0] for x in _bits(v, 32)] + [x for v in zy[1] for x in _bits(v, 16)],
                   SD.leaves(SALT, True), SD.position_leaves(POSITION, True),
                   [x for r in ROOTS for x in SD.leaves(r, True)])
    w = 2 * lay.sets
    return [_int(out[32 * t:32 * t + 32]) for t in range(w)], int(out[32 * w])


@pytest.mark.slow
@pytest.mark.parametrize("defn", [H.PouwH1Tile, H.PouwH1Tile3])
def test_a_lane_on_bits_is_its_word_view(defn):
    """One lane of a 32 × 32 leaf at K = 64 on bits (its seeds, its 12 rows' noise and its record check): what the word
    tests can't show is that the Definition's partial and ok are `lane_reference`'s, at the lane's offsets in the layout."""
    a, b, sa, sw, zy = _leaf(6)
    lay = H.Layout(32, 32, 64)
    ln = 13
    want = _lanes(a, b, sa, sw, zy, lay)[ln]
    assert _lane_bits(_tile(defn), ln, a, b, sa, sw, zy, lay) == want and want[1] == 1
    zr = list(zy[0])
    zr[L.cells(lay, ln)[3]] ^= 4
    assert _lane_bits(_tile(defn), ln, a, b, sa, sw, (zr, zy[1]), lay) == (want[0], 0)
