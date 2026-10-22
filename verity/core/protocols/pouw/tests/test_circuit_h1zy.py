"""`circuit.h1zy`: the dequant is the served epilogue bit for bit, RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) with the activation
scale first and no bias (`int8_exact.dequantize`, scaled_mm_zy's y), on words and on bits, where PoUW's `reference.dequant`
(scales first, a bias add) is not; `PouwZyCheck` holds a unit's record to its z and that dequant, cell by cell; and
`PouwH1TileZy` is the H1 tile and that check over the rows' codes and scales, returning the receipt and one bit; a leaf's
lane (`PouwH1LaneZy`) checks its committed partial of the receipt (zero for a dead leaf) and its own cells, and the combine
(`PouwH1Combine`) that the lanes' partials XOR to the receipt. `benchmarks/pouw/tests/test_h1_prove.py` and h1_prove's smoke run the unit end to end."""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from verity.ml.circuits import ports as PT
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate, evaluate_bits
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import Array

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import reference as REF
from verity.core.protocols.pouw.circuit import seeds as SD

M32 = (1 << 32) - 1


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def served(z: int, sa: int, sw: int) -> int:
    """The served epilogue in numpy: z as f32 (RN), times s_a, times s_w, each an f32 product (RN), then f32 to bf16 round to
    nearest even, every NaN 0x7FFF (cvt.rn.bf16.f32's; torch's CPU cast gives 0x7FC0)."""
    s32 = np.int32(np.uint32(z & M32).view(np.int32))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        v = np.float32(s32) * np.uint32(sa).view(np.float32) * np.uint32(sw).view(np.float32)
    bits = int(np.float32(v).view(np.uint32))
    if v != v:
        return 0x7FFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


INF, NAN, MAX, TINY = f32(float("inf")), 0x7FC00000, 0x7F7FFFFF, 0x00000001
#: (z, s_a, s_w) at the corners: zeros of each sign, the s32 extremes, z past 2^24 (int to f32 rounds), subnormal and huge
#: scales (an underflow to a subnormal or zero, an overflow to inf), inf and NaN scales, and served-like 1/127 scales
EDGES = [(0, f32(0.5), f32(0.25)), (0, f32(-1.0), f32(0.5)), (0, f32(-1.0), f32(-1.0)), (1, 0, f32(1.0)), (1, 0x80000000, f32(1.0)),
         ((1 << 31) - 1, f32(1.0), f32(1.0)), (-(1 << 31), f32(1.0), f32(1.0)), ((1 << 24) + 1, f32(1.0), f32(1.0)),
         ((1 << 24) + 3, f32(1.0), f32(3.0)), (12345, TINY, f32(1.0)), (3, TINY, f32(0.5)), (-7, f32(1e-20), f32(1e-20)),
         ((1 << 30), MAX, f32(2.0)), (-5, MAX, MAX), (2, INF, f32(1.0)), (0, INF, f32(1.0)), (9, NAN, f32(1.0)),
         (9, f32(1.0), NAN), (-1, f32(1 / 127), f32(1 / 127)), (1000, f32(3.1 / 127), f32(0.07 / 127)),
         (0x7F7F, f32(1.0), f32(1.0)), (0x8080, f32(1.0), f32(1.0)), (0x18181, f32(1.0), f32(1.0))]


def _rand(rng: random.Random) -> tuple[int, int, int]:
    """A served-like cell: an s32 dot of a K = 4096 tile and per-row scales of a layer's absmax range."""
    z = rng.randrange(-(1 << 26), 1 << 26) if rng.random() < 0.8 else rng.randrange(-(1 << 31), 1 << 31)
    return z & M32, f32(rng.uniform(1e-4, 0.1) / 127), f32(rng.uniform(1e-3, 0.5) / 127)


def _bits(v: int, n: int) -> list[int]:
    return [(v >> i) & 1 for i in range(n)]


def _int(bits) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))


def _dequant_bits(z: int, sa: int, sw: int) -> int:
    return _int(evaluate(Z.PouwZyDequant.bind(), _bits(z, 32), _bits(sa, 32), _bits(sw, 32)))


def test_the_dequant_on_words_is_the_served_epilogue_rounding_and_corners_included():
    rng = random.Random(33)
    cases = EDGES + [_rand(rng) for _ in range(2000)]
    assert [Z.dequant(*c) for c in cases] == [served(*c) for c in cases]
    assert Z.dequant(0, f32(-1.0), f32(0.5)) == 0x8000, "z = 0 at a negative scale is -0, not +0"
    assert Z.dequant(2, INF, f32(1.0)) == 0x7F80 and Z.dequant(0, INF, f32(1.0)) == Z.dequant(9, NAN, f32(1.0)) == 0x7FFF


def _nan(y: int) -> bool:
    return (y & 0x7F80) == 0x7F80 and bool(y & 0x7F)


def test_the_dequant_is_torchs_scaled_mm_epilogue_but_for_the_nan_encoding():
    """`int8_exact.dequantize`'s expression, (z.float() * s_a[:, None] * s_w).to(bfloat16), on the CPU: equal but where the
    result is NaN, which the GPU's cvt.rn encodes 0x7FFF and torch's CPU cast 0x7FC0 (scalar) or 0xFFFF (vectorized)."""
    torch = pytest.importorskip("torch")
    rng = random.Random(34)
    cases = EDGES + [_rand(rng) for _ in range(2000)]
    z = torch.from_numpy(np.array([c[0] & M32 for c in cases], np.uint32).view(np.int32))
    sa = torch.from_numpy(np.array([c[1] for c in cases], np.uint32).view(np.float32))
    sw = torch.from_numpy(np.array([c[2] for c in cases], np.uint32).view(np.float32))
    y = (z.float() * sa * sw).to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16).tolist()
    ours = [Z.dequant(*c) for c in cases]
    assert [o for o, t in zip(ours, y) if not _nan(t)] == [t for t in y if not _nan(t)]
    assert [o for o, t in zip(ours, y) if _nan(t)] == [0x7FFF] * sum(map(_nan, y)) and any(map(_nan, y))


#: served-like cells (z, s_a, s_w) whose y the scales-first order rounds to the neighbouring bf16, found among 4M random
#: ones (18 of them, about one 32 × 32 unit in 200 holding one), with the served y and `reference.dequant`'s
ORDER = [(0x1D413C7, 0x3991FC09, 0x3A3A8A99, 0x40C3, 0x40C2), (0xFCD70EEE, 0x3A210350, 0x3B2081AA, 0xC29F, 0xC2A0),
         (0x37EDBE9, 0x3995C507, 0x3B113B06, 0x4214, 0x4215), (0x363E7E8, 0x3A0E4D8A, 0x3B173AED, 0x428F, 0x428E),
         (0x2A9FE11, 0x39D26FE4, 0x3A9ACD6D, 0x41A9, 0x41AA), (0x13AE8B2, 0x39A3E035, 0x39F70066, 0x4043, 0x4042)]


def test_poUWs_reference_dequant_is_another_function():
    """`reference.dequant` multiplies the two scales first and adds a bf16 bias, so it is not the served epilogue: f32
    products don't associate, which moves a bf16 y wherever the f32 results straddle a rounding boundary, and its bias add
    maps -0 to +0."""
    assert [(Z.dequant(z, sa, sw), REF.dequant(z, sa, sw, 0)) for z, sa, sw, _y, _r in ORDER] == [(y, r) for *_c, y, r in ORDER]
    assert all(served(z, sa, sw) == y != r for z, sa, sw, y, r in ORDER)
    assert Z.dequant(0, f32(-1.0), f32(0.5)) == 0x8000 != REF.dequant(0, f32(-1.0), f32(0.5), 0)


def test_the_dequant_on_bits_is_its_word_view_at_the_corners():
    rng = random.Random(36)
    cases = EDGES[::3] + [_rand(rng) for _ in range(4)]
    assert [_dequant_bits(*c) for c in cases] == [Z.dequant(*c) for c in cases]


@pytest.mark.slow
def test_the_dequant_on_bits_is_its_word_view_at_every_corner():
    """Every corner and 40 served-like cells on bits, about 0.2 s each: the fast test's third of the corners leaves the
    subnormal products, the inf and NaN scales and the s32 extremes to this one."""
    rng = random.Random(37)
    cases = EDGES + [_rand(rng) for _ in range(40)]
    assert [_dequant_bits(*c) for c in cases] == [Z.dequant(*c) for c in cases]


def _unit_rows(rng: random.Random, r: int = 4, c: int = 8, k: int = 64):
    a = [[rng.randrange(-127, 128) & 0xFF for _ in range(k)] for _ in range(r)]
    b = [[rng.randrange(-127, 128) & 0xFF for _ in range(k)] for _ in range(c)]
    sa = [f32(rng.uniform(1e-3, 0.1) / 127) for _ in range(r)]
    sw = [f32(rng.uniform(1e-3, 0.5) / 127) for _ in range(c)]
    return a, b, sa, sw


def test_a_units_honest_record_is_its_product_and_dequant():
    rng = random.Random(38)
    a, b, sa, sw = _unit_rows(rng)
    z, y = Z.records(a, b, sa, sw)
    sx = lambda v: v - 256 if v & 0x80 else v  # noqa: E731
    want = [sum(sx(p) * sx(q) for p, q in zip(ai, bj)) & M32 for ai in a for bj in b]
    assert z == want and y == [served(z[i * 8 + j], sa[i], sw[j]) for i in range(4) for j in range(8)]
    assert Z.check(z, y, z, sa, sw) == 1
    bad_y, bad_z, swapped = list(y), list(z), [y[8], *y[1:8], y[0], *y[9:]]
    bad_y[5] ^= 1
    bad_z[5] = (bad_z[5] + 1) & M32
    assert Z.check(z, bad_y, z, sa, sw) == Z.check(bad_z, y, z, sa, sw) == Z.check(z, swapped, z, sa, sw) == 0
    assert Z.check(z[:16], y[:16], z, sa, sw) == 0, "a record of other than the unit's cells"


def _check_bits(zr, yr, z, sa, sw) -> int:
    rec = [x for v in zr for x in _bits(v, 32)] + [x for v in yr for x in _bits(v, 16)]
    out = evaluate(bind(Z.PouwZyCheck, R=4, C=8), rec, [x for v in z for x in _bits(v, 32)],
                   [x for v in sa for x in _bits(v, 32)], [x for v in sw for x in _bits(v, 32)])
    return _int(out)


@pytest.mark.slow
def test_the_check_on_bits_accepts_the_honest_record_and_no_cell_off_it():
    """`PouwZyCheck{4, 8}` on bits, about 6 s an evaluation: 1 on the honest record, 0 with one cell's y off its dequant, one
    cell's z off the product (its y that z's dequant, so only the product check refuses it), or two rows' cells swapped."""
    rng = random.Random(39)
    a, b, sa, sw = _unit_rows(rng)
    z, y = Z.records(a, b, sa, sw)
    zc, yc = list(z), list(y)
    zc[9] = (zc[9] + 1) & M32
    yc[9] = Z.dequant(zc[9], sa[1], sw[1])
    zs, ys = [*z[8:16], *z[:8], *z[16:]], [*y[8:16], *y[:8], *y[16:]]
    by = list(y)
    by[30] ^= 0x10
    got = [_check_bits(z, y, z, sa, sw), _check_bits(z, by, z, sa, sw), _check_bits(zc, yc, z, sa, sw),
           _check_bits(zs, ys, z, sa, sw)]
    assert got == [1, 0, 0, 0] == [Z.check(z, y, z, sa, sw), Z.check(z, by, z, sa, sw), Z.check(zc, yc, z, sa, sw),
                                   Z.check(zs, ys, z, sa, sw)]


def _tile(defn=H.PouwH1Tile, r=4, c=8, k=64, pairs=None):
    exact = bind(BG.GemmAccumulatorS8, K=k, **BG.GEMM_SM120_S8)
    return bind(defn, EXACT=exact, R=r, C=c, PAIRS=pairs or tuple((i, j) for i in range(r) for j in range(c)))


def test_the_unit_reads_scaled_rows_and_the_record_and_returns_the_receipt_and_ok():
    for defn in (H.PouwH1Tile, H.PouwH1Tile3):
        u = Z.unit(_tile(defn))
        params, ret = u.signature()
        assert [(n, t.leaves) for n, t in params] == [("a", 4 * (8 * 64 + 32)), ("b", 8 * (8 * 64 + 32)), ("zy", 1536),
                                                      *((n, t.leaves) for n, t in _tile(defn).signature()[0][2:])]
        assert params[0][1].elem == Z.ROW(64) and params[2][1] == Z.RECORD(4, 8)
        assert [n for n, _t in ret.fields] == ["receipt", "ok"] and ret.fields[1][1].leaves == 1
        assert ret.fields[0][1] == _tile(defn).signature()[1].fields[1][1], "the tile's receipt"
        assert {n: p.kind for n, p in u.defn.ports(u.S()).params.items()} == {"salt": "coin", "position": "position"}


@pytest.mark.parametrize("tile,why", [
    (lambda: _tile(H.PouwH1Tile4), "an H1 tile"),
    (lambda: BG.GemmAccumulatorS8.bind(K=64, **BG.GEMM_SM120_S8), "an H1 tile"),
    (lambda: _tile(pairs=tuple((i, j) for j in range(8) for i in range(4))), "row-major"),
    (lambda: _tile(pairs=((0, 0), (0, 1))), "row-major"),
])
def test_a_unit_over_another_tile_or_its_cells_out_of_the_records_order_is_refused(tile, why):
    with pytest.raises(ValueError, match=why):
        Z.unit(tile()).signature()


@pytest.mark.slow
def test_the_unit_is_the_tile_then_the_check_of_its_z():
    """The unit's body (tracing the tile's seed and noise, about 35 s): one call of the tile on the rows' codes, salt,
    position and roots, and one of `PouwZyCheck{4, 8}` on the record, the tile's z and the rows' scales, nothing else; what
    the signature test can't show is that the check reads the tile's z and not some other value of the unit."""
    tile = _tile()
    prog = Program(Z.unit(tile))
    nodes = [n for n in prog.root.body.nodes if not is_input(n.fn)]
    assert [n.fn.id for n in nodes] == [tile.id, "PouwZyCheck_v1{R=4,C=8}"]
    k_tile = prog.root.body.nodes.index(nodes[0])
    z_from = {(space, k) for space, k, _lo, _hi in runs(nodes[1].args[1].refs)}
    assert z_from == {("n", k_tile)}, "the check's z is the tile's output"


def test_reference_is_the_tile_and_the_check_on_words():
    rng = random.Random(40)
    a, b, sa, sw = _unit_rows(rng)
    salt, roots = bytes(range(64)), (bytes(64), bytes([7]) * 64)
    z, receipt = H.reference(a, b, salt, 5, roots)
    zy = Z.records(a, b, sa, sw)
    assert Z.reference(a, b, sa, sw, zy, salt, 5, roots) == (receipt, 1) and zy[0] == z
    assert Z.reference(a, b, sa, sw, (zy[0], [v ^ 1 for v in zy[1]]), salt, 5, roots) == (receipt, 0)


# -- a leaf's lanes and their combine ------------------------------------------------------------------------------------------

SALT, ROOTS = bytes(range(64)), (bytes(64), bytes([7]) * 64)


def _lane_rows(lay, ln, a, b, sa, sw):
    rows, cols = lay.rows(ln), lay.cols(ln)
    return [a[i] for i in rows], [b[j] for j in cols], [sa[i] for i in rows], [sw[j] for j in cols]


def test_a_lanes_cells_are_kernels_lane_and_the_lanes_cover_the_leaf_once():
    """At the served 32 × 32 leaf, lane l = 4 g + q holds rows g + 8 s and columns 2 q + 8 (s >> 1) + (s & 1) (kernels'
    ``lane_partials``, #3095), its cells by row slot then column slot (`lane_cells`), and the lanes' cells are the leaf's once
    each; its index is the low 5 bits of a 16-bit word."""
    lay = H.Layout(32, 32, 64)
    for ln in range(32):
        g, q = ln >> 2, ln & 3
        assert Z.lane_cells(lay, ln) == [32 * (g + 8 * s) + 2 * q + 8 * (t >> 1) + (t & 1) for s in range(4) for t in range(8)]
    assert sorted(u for ln in range(32) for u in Z.lane_cells(lay, ln)) == list(range(1024))
    assert (Z.lane_bits(32, 32), Z.lane_bits(8, 16), Z.lane_bits(8, 8)) == (5, 2, 1) and Z.LANE.n == 16


def test_a_lane_on_words_checks_its_committed_partial_and_its_own_cells():
    """`lane_reference`: lane l's honest partial (`lane_partial`) is `h1tile.lane_partials`' l, and ok is 1 on it and the
    honest record; one bit of the partial off, or a cell of its own off the record's dequant, makes ok 0, and one of
    another lane's cells leaves it 1 (that lane's check refuses it); an index past the leaf's lanes makes ok 0, and one past
    16 bits is refused. A dead leaf's partials are zero, its record z = 0 and y = dq(0, +0, s_w), and its live partials make
    ok 0. The combine is 1 exactly when the partials XOR to the receipt."""
    rng = random.Random(41)
    a, b, sa, sw = _unit_rows(rng, 8, 16)
    lay, live = H.Layout(8, 16, 64), [1] * 8
    zy = Z.records(a, b, sa, sw)
    parts = H.lane_partials(a, b, SALT, 5, ROOTS)
    for ln in range(lay.lanes):
        rows = _lane_rows(lay, ln, a, b, sa, sw)
        assert Z.lane_partial(*rows[:2], live, SALT, 5, ROOTS, ln, (8, 16)) == parts[ln]
        assert Z.lane_reference(*rows, live, zy, parts[ln], SALT, 5, ROOTS, ln, (8, 16)) == 1
        off = list(parts[ln])
        off[(3 * ln + 1) % len(off)] ^= 1 << 9
        assert Z.lane_reference(*rows, live, zy, off, SALT, 5, ROOTS, ln, (8, 16)) == 0
        for other, want in ((ln, 0), (ln ^ 3, 1)):
            bad = (zy[0], list(zy[1]))
            bad[1][Z.lane_cells(lay, other)[7]] ^= 1
            assert Z.lane_reference(*rows, live, bad, parts[ln], SALT, 5, ROOTS, ln, (8, 16)) == want
        for idx in (ln + lay.lanes, ln | 1 << 15):
            assert Z.lane_reference(*rows, live, zy, parts[ln], SALT, 5, ROOTS, idx, (8, 16)) == 0
    with pytest.raises(ValueError, match="16-bit word"):
        Z.lane_reference(*_lane_rows(lay, 0, a, b, sa, sw), live, zy, parts[0], SALT, 5, ROOTS, 1 << 16, (8, 16))
    receipt = H.reference(a, b, SALT, 5, ROOTS)[1]
    assert Z.combine_reference(parts, receipt) == 1
    assert Z.combine_reference(parts, [receipt[0] ^ 1, *receipt[1:]]) == 0
    assert Z.combine_reference([parts[0], parts[2], parts[2], parts[3]], receipt) == 0

    empty = [[0] * 64 for _ in range(8)]
    zy0, zeros = Z.records(empty, b, [0] * 8, sw), [0] * (2 * lay.sets)
    assert set(zy0[0]) == {0} and zy0[1] == [Z.dequant(0, 0, sw[j]) for _i in range(8) for j in range(16)]
    ungated = H.lane_partials(empty, b, SALT, 5, ROOTS)
    for ln in range(lay.lanes):
        rows = _lane_rows(lay, ln, empty, b, [0] * 8, sw)
        assert Z.lane_partial(*rows[:2], [0] * 8, SALT, 5, ROOTS, ln, (8, 16)) == zeros != ungated[ln]
        assert Z.lane_reference(*rows, [0] * 8, zy0, zeros, SALT, 5, ROOTS, ln, (8, 16)) == 1
        assert Z.lane_reference(*rows, [0] * 8, zy0, ungated[ln], SALT, 5, ROOTS, ln, (8, 16)) == 0
        assert Z.lane_reference(*rows, live, zy0, zeros, SALT, 5, ROOTS, ln, (8, 16)) == 0
    assert Z.combine_reference([zeros] * lay.lanes, H.reference(empty, b, SALT, 5, ROOTS, stream=True, gated=[0] * 8)[1]) == 1


@pytest.mark.parametrize("k,words", [(4096, 260), (14336, 902)])
def test_a_lane_reads_its_rows_admit_bits_record_partial_and_index_and_returns_ok(k, words):
    """At the served 32 × 32 leaf: the lane's 4 A and 8 B rows, the leaf's 32 admit bits and 6,144-byte record, its
    committed ``partials`` row of 2 (KS + m) words (#3095's 260 at K = 4096, 902 at K = 14336), the tile's salt, position
    and roots and its index, one 16-bit word (a public row of whole 16-bit words), and its one output ``ok``, a sink, so
    the lane is no seed (`ports._coin_role`); the combine reads the 32 ``partials`` rows and the receipt and returns one
    ``ok``."""
    t = _tile(H.PouwH1Tile, 32, 32, k)
    lane, comb = Z.lane(t), Z.combine(t)
    params, ret = lane.signature()
    receipt = t.signature()[1].fields[1][1]
    assert [(n, tp.leaves) for n, tp in params] == [
        ("a", 4 * (8 * k + 32)), ("b", 8 * (8 * k + 32)), ("admit", 32), ("zy", 8 * 6144), ("partials", 32 * words),
        *((n, tp.leaves) for n, tp in t.signature()[0][2:]), ("lane", 16)]
    assert sum(tp.leaves for n, tp in params if n in ("salt", "position", "roots", "lane")) % 16 == 0
    assert params[0][1].elem == Z.ROW(k) and params[3][1] == Z.RECORD(32, 32) and params[4][1] == receipt
    assert ret.fields == (("ok", Z.BIT),) and receipt.n == words
    decl = lane.defn.ports(lane.S())
    assert {n: p.kind for n, p in decl.params.items()} == {"salt": "coin", "position": "position"}
    assert set(decl.ret) == {"ok"} and decl.ret["ok"].sink and PT._coin_role(lane) is None
    cparams, cret = comb.signature()
    assert cparams == (("partials", Array(32, receipt)), ("receipt", receipt)) and cret.fields == (("ok", Z.BIT),)
    assert PT.port_types(comb) is None


@pytest.mark.parametrize("tile,why", [
    (lambda: _tile(H.PouwH1Tile4, 8, 16), "an H1 tile"),
    (lambda: _tile(r=4, c=8), "whole bits"),
    (lambda: _tile(r=12, c=8), "whole bits"),
    (lambda: _tile(r=8, c=24), "whole bits"),
])
def test_a_lane_or_combine_of_a_leaf_of_one_lane_or_no_whole_index_bits_is_refused(tile, why):
    for d in (Z.PouwH1LaneZy, Z.PouwH1Combine):
        with pytest.raises(ValueError, match=why):
            bind(d, TILE=tile()).signature()


def test_a_lane_is_staged_at_any_k_and_a_combine_from_the_k_whose_partials_compress_before_their_last_block():
    """A partial and the receipt are 8 (KS + m) bytes. The Lean verifier takes a v2 row of any length (#3148), so K = 384,
    416, 896 and 928, whose rows' padding spills into another block, are staged. The combine reads only those rows, and
    C-Flock needs a row of the statement that compresses before its last block, ceil((bytes + 17) / 128) - 1 >= 1, so
    112 bytes or more: K = 384 is the least, and K = 352's 104-byte rows are refused. A lane has such rows at every K."""
    assert [8 * H.Layout(4, 8, k).sets for k in (352, 384, 416, 448, 896, 928)] == [104, 112, 120, 128, 240, 248]
    assert Z.LEAST_K == 384
    for k in (64, 352, 384, 416, 896, 928, 4096):
        assert Z.lane(_tile(r=16, c=16, k=k)) == bind(Z.PouwH1LaneZy, TILE=_tile(r=16, c=16, k=k))
    for k in (384, 416, 448, 896, 928, 4096):
        assert Z.combine(_tile(r=16, c=16, k=k)) == bind(Z.PouwH1Combine, TILE=_tile(r=16, c=16, k=k))
    for k in (64, 352):
        with pytest.raises(ValueError, match=f"K = {k}: the combine reads only .* K = 384 is the least"):
            Z.combine(_tile(r=16, c=16, k=k))


def test_a_leaf_is_staged_only_at_a_positive_k_and_an_r_whose_admit_row_is_whole_16_bit_words():
    """A lane reads the leaf's R admit bits as one row, and C-Flock lowers only a row of whole 16-bit words
    (`class_statement.unit_lowering`), so `lane` and `combine` refuse a leaf of R = 4 or 8, whose lanes the Definitions
    bind, and take R = 16 and the served 32. Both refuse K = 0, which the Definitions bind too and `leaf_lanes` refuses."""
    for r, c in ((4, 16), (8, 8), (8, 16)):
        t = _tile(r=r, c=c, k=384)
        bind(Z.PouwH1LaneZy, TILE=t).signature()
        for stage in (Z.lane, Z.combine):
            with pytest.raises(ValueError, match=f"R = {r}: a lane reads the leaf's R admit bits as one row"):
                stage(t)
    for r, c in ((16, 8), (16, 16), (32, 32)):
        t = _tile(r=r, c=c, k=384)
        assert Z.lane(t) == bind(Z.PouwH1LaneZy, TILE=t) and Z.combine(t) == bind(Z.PouwH1Combine, TILE=t)
    t = _tile(r=32, c=32, k=0)
    bind(Z.PouwH1LaneZy, TILE=t).signature()
    for stage in (Z.lane, Z.combine):
        with pytest.raises(ValueError, match="K = 0: a leaf's lanes and combine are staged only at a positive K"):
            stage(t)


def _word_bits(words):
    return [x for v in words for x in _bits(v, 32)]


def test_the_combine_on_bits_is_1_exactly_when_the_partials_xor_to_the_receipt():
    rng = random.Random(42)
    lay = H.Layout(8, 16, 64)
    parts = [[rng.randrange(1 << 32) for _ in range(2 * lay.sets)] for _ in range(lay.lanes)]
    receipt = [0] * (2 * lay.sets)
    for p in parts:
        receipt = [x ^ v for x, v in zip(receipt, p)]
    off = [list(p) for p in parts]
    off[2][5] ^= 1 << 30
    cases = [(parts, receipt), (parts, [*receipt[:7], receipt[7] ^ 1]), (off, receipt)]
    args = [np.array([[x for p in ps for x in _word_bits(p)] for ps, _r in cases], np.uint8),
            np.array([_word_bits(r) for _ps, r in cases], np.uint8)]
    out = evaluate_bits(bind(Z.PouwH1Combine, TILE=_tile(r=8, c=16)), *args).reshape(len(cases))
    assert [int(o) for o in out] == [Z.combine_reference(*c) for c in cases] == [1, 0, 0]


def _row_bits(rows, scales):
    return [x for row, s in zip(rows, scales) for x in [y for v in row for y in _bits(v, 8)] + _bits(s, 32)]


@pytest.mark.slow
@pytest.mark.parametrize("defn", [H.PouwH1Tile, H.PouwH1Tile3], ids=["v1", "v3"])
def test_a_lane_on_bits_is_its_word_reference(defn):
    """`PouwH1LaneZy` over an 8 × 16 leaf at K = 64 (4 lanes, a 2-bit index) on bits, every case at once, about 50 s: ok is
    `lane_reference`'s for each lane on the honest record and partial (1), with one of its own cells off (0) and with one
    bit of its partial off (0); with another lane's cell off (1); with one row admitted, not the lane's (1); on a dead leaf
    with its zero partial (1) or its ungated one (0); on a live leaf with a zero partial (0); and on lane 1's honest rows
    at indices 5 and 1 + 2^15, past the leaf's lanes (0). It is the only test of the lane's gates: the mux trees that pick
    its noise blocks' counters, its keys' block and its cells by the public index's low bits, the bound on its high bits,
    its noise at the leaf's offsets, and the admit gate on the partial it checks."""
    rng = random.Random(43)
    a, b, sa, sw = _unit_rows(rng, 8, 16)
    lay, live = H.Layout(8, 16, 64), [1] * 8
    zeros = [0] * (2 * lay.sets)
    zy = Z.records(a, b, sa, sw)
    parts = H.lane_partials(a, b, SALT, 5, ROOTS)
    empty = [[0] * 64 for _ in range(8)]
    zy0 = Z.records(empty, b, [0] * 8, sw)
    dead = H.lane_partials(empty, b, SALT, 5, ROOTS)

    def cell_off(ln):
        rec = (list(zy[0]), list(zy[1]))
        rec[0][Z.lane_cells(lay, ln)[13]] ^= 4
        return rec

    def bit_off(p, ln):
        p = list(p)
        p[(2 * ln + 3) % len(p)] ^= 1 << (5 * ln)
        return p

    full = (a, sa)
    cases = [c for ln in range(lay.lanes) for c in ((ln, full, live, zy, parts[ln]), (ln, full, live, cell_off(ln), parts[ln]),
                                                    (ln, full, live, zy, bit_off(parts[ln], ln)))]
    cases += [(0, full, live, cell_off(3), parts[0]), (1, full, [0, 0, 0, 0, 0, 0, 0, 1], zy, parts[1]),
              (2, (empty, [0] * 8), [0] * 8, zy0, zeros), (3, (empty, [0] * 8), [0] * 8, zy0, dead[3]),
              (0, full, live, zy, zeros)]
    index = [c[0] for c in cases] + [1 + lay.lanes, 1 | 1 << 15]
    cases += [(1, full, live, zy, parts[1])] * 2
    want, rows = [], []
    for (ln, (aa, ss), admit, rec, partial), idx in zip(cases, index, strict=True):
        ra, rb, ras, rbs = _lane_rows(lay, ln, aa, b, ss, sw)
        want.append(Z.lane_reference(ra, rb, ras, rbs, admit, rec, partial, SALT, 5, ROOTS, idx, (8, 16)))
        rows.append([_row_bits(ra, ras), _row_bits(rb, rbs), list(admit),
                     _word_bits(rec[0]) + [x for v in rec[1] for x in _bits(v, 16)], _word_bits(partial),
                     SD.leaves(SALT, True), SD.position_leaves(5, True),
                     SD.leaves(ROOTS[0], True) + SD.leaves(ROOTS[1], True), _bits(idx, 16)])
    args = [np.array([r[p] for r in rows], np.uint8) for p in range(9)]
    got = evaluate_bits(bind(Z.PouwH1LaneZy, TILE=_tile(defn, 8, 16)), *args).reshape(len(cases))
    assert want == [1, 0, 0] * lay.lanes + [1, 1, 1, 0, 0, 0, 0]
    assert [int(o) for o in got] == want


@pytest.mark.slow
def test_the_port_rules_follow_the_coin_into_a_lane_its_statement_calls():
    """A statement's Definition calls its unit from a template (`h1window.template`'s form), so the lane is a callee: a
    seed there (a coin and a position port and no sink returned) would see its salt as a computed value, and its own
    seeds' coin port would refuse it. With ok a sink the port rules follow the coin into the lane, and a Program of one call
    passes them, about 35 s: no other test runs the rules over the lane's body (the bits test evaluates it, unchecked)."""
    lane = bind(Z.PouwH1LaneZy, TILE=_tile(H.PouwH1Tile3, 8, 16))
    ports = PT.Ports(params={"salt": PT.In(kind="coin"), "position": PT.In(kind="position")}, ret={"ok": PT.Out(sink=True)})
    template = CompositeDefinition("H1ZyTestLaneTemplate", 1, (), lambda S: lane.signature(),
                                   lambda B, S, *args: B.call(lane, *args), conformance="protocol-defined", register=False,
                                   ports=lambda S: ports)
    prog = Program(bind(template))
    assert [n.fn.id for n in prog.root.body.nodes if not is_input(n.fn)] == [lane.id]
