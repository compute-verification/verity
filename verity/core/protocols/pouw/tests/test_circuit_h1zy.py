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
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import reference as REF
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG

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
    """A partial and the receipt are 8 (KS + m) bytes. The Lean verifier takes a v2 row of any length, so K = 384,
    416, 896 and 928, whose rows' padding spills into another block, are staged. The combine reads only those rows, and
    Flock needs a row of the statement that compresses before its last block, ceil((bytes + 17) / 128) - 1 >= 1, so
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
    """A lane reads the leaf's R admit bits as one row, and Flock lowers only a row of whole 16-bit words
    (`class_statement.unit_lowering`), so `lane` and `combine` refuse a leaf of R = 4 or 8, whose lanes the Definitions
    bind, and take R = 16 and the served 32. Both refuse K = 0, which `leaf_lanes` refuses too; the Definitions refuse
    K = 0 and K = 32 themselves (KS >= 2, as `h1tile`'s scheme asks and the leaf verifier's `Tile.check`)."""
    for r, c in ((4, 16), (8, 8), (8, 16)):
        t = _tile(r=r, c=c, k=384)
        bind(Z.PouwH1LaneZy, TILE=t).signature()
        for stage in (Z.lane, Z.combine):
            with pytest.raises(ValueError, match=f"R = {r}: a lane reads the leaf's R admit bits as one row"):
                stage(t)
    for r, c in ((16, 8), (16, 16), (32, 32)):
        t = _tile(r=r, c=c, k=384)
        assert Z.lane(t) == bind(Z.PouwH1LaneZy, TILE=t) and Z.combine(t) == bind(Z.PouwH1Combine, TILE=t)
    for k in (0, 32):
        for d in (Z.PouwH1LaneZy, Z.PouwH1Combine):
            with pytest.raises(ValueError, match=f"K = {k} is under 64"):
                bind(d, TILE=_tile(r=32, c=32, k=k))
    t = _tile(r=32, c=32, k=0)
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


# -- a lane in chunks along K ---------------------------------------------------------------------------------------------------

def test_a_lane_past_4096_is_proved_in_the_fewest_chunks_of_whole_noise_blocks():
    """`chunks`: the whole lane up to K = 4096, past it the least n whose K / n is at most 4096 and a multiple of 512, at
    every served K (6144: 2 × 3072, 12288: 3 × 4096, 14336: 4 × 3584, 16384: 4 × 4096); a K with no such n is refused."""
    assert [Z.chunks(k) for k in (32, 448, 4096, 6144, 12288, 14336, 16384)] == [1, 1, 1, 2, 3, 4, 4]
    assert (Z.CHUNK_MAX, Z.CHUNK_ALIGN) == (4096, 512) and Z.CHUNK_ALIGN == SD.BLOCK_BITS
    for k in (6144, 12288, 14336, 16384):
        n = Z.chunks(k)
        assert k // n <= Z.CHUNK_MAX and (k // n) % 512 == 0
        assert not any(k % (m * 512) == 0 and k // m <= Z.CHUNK_MAX for m in range(2, n))
    for k, why in ((0, "positive K"), (4128, "no n splits it")):
        with pytest.raises(ValueError, match=why):
            Z.chunks(k)


def _chunk_rows(rows, n, c):
    kc = len(rows[0]) // n
    return [r[c * kc:c * kc + kc] for r in rows]


def test_a_lanes_chunks_on_words_chain_to_the_whole_lane():
    """`lane_chunk_carries`: chunk c's carry is the whole lane's chain after its (c + 1) KS_c segments (the fx1 state after
    those sets of `h1tile.lane_sets` from the lane's keys, the last of those sets as T, the rows' residue sums and each
    cell's z over those entries, the chain's words so far), and the last chunk's words and z are the lane's partial and
    its cells' exact products, at 2, 3 and 4 chunks; `lane_chunk_reference` is 1 on each chunk's honest rows and carries."""
    rng = random.Random(44)
    for r, c, k, n in ((8, 16, 64, 2), (8, 16, 96, 3), (32, 32, 1024, 4)):
        a, b, sa, sw = _unit_rows(rng, r, c, k)
        lay, live, kc = H.Layout(r, c, k), [1] * r, k // n
        zy = Z.records(a, b, sa, sw)
        parts = H.lane_partials(a, b, SALT, 5, ROOTS)
        noise = H.noise(SD.seed(SALT, 5, *ROOTS), lay)
        keys = H.fold_keys(SD.receipt_seed(SALT, 5, *ROOTS), lay)
        for ln in (0, lay.lanes - 1):
            ra, rb, ras, rbs = _lane_rows(lay, ln, a, b, sa, sw)
            carries = Z.lane_chunk_carries(ra, rb, ras, rbs, SALT, 5, ROOTS, ln, (r, c), n)
            sets = H.lane_sets(a, b, *noise, lay, ln)
            for ch, carry in enumerate(carries[:-1]):
                done = (ch + 1) * kc // 32
                st = keys[ln]
                for p in sets[:done]:
                    st = SG.absorb(H.FOLD, st, p)
                x = [[(v + noise[0][i][0][t]) & 0xFF for t, v in enumerate(a[i][:done * 32])] for i in lay.rows(ln)]
                assert carry["state"] == st and carry["t"] == sets[done - 1] and carry["words"] == parts[ln][:2 * done]
                assert carry["res"] == [[sum(v - 256 if v & 0x80 else v for v in xi[rho::32]) & M32 for rho in range(32)]
                                        for xi in x]
                assert carry["z"] == [SG.seg_dot(0, ra[s][:done * 32], rb[t][:done * 32]) for s in range(4) for t in range(8)]
                assert (carry["sa"], carry["sw"]) == (ras, rbs)
                assert len(Z.carry_words(carry)) == Z.CARRY(lay, n, ch).leaves // 32
            assert carries[-1]["words"] == parts[ln]
            assert carries[-1]["z"] == [SG.seg_dot(0, ra[s], rb[t]) for s in range(4) for t in range(8)]
            for ch in range(n):
                last = ch == n - 1
                assert Z.lane_chunk_reference(_chunk_rows(ra, n, ch), _chunk_rows(rb, n, ch), ras, rbs, live if last else None,
                                              zy if last else None, parts[ln] if last else None,
                                              carries[ch - 1] if ch else None, None if last else carries[ch], SALT, 5, ROOTS,
                                              ln, (r, c), n, ch) == 1


def _off(carry, field, at, bit):
    out = {f: [list(v) for v in w] if f == "res" else list(w) for f, w in carry.items()}
    if field == "res":
        out["res"][at // 32][at % 32] ^= 1 << bit
    else:
        out[field][at] ^= 1 << bit
    return out


def test_a_chunk_on_words_holds_its_carries_scales_partial_and_cells():
    """`lane_chunk_reference` at 3 chunks of an 8 × 16 leaf at K = 96: a bit off any field of a chunk's committed carry out
    makes it 0, and so does a carried scale that is not its rows' or a carry in off the honest one in a bit its chain
    reads (the chain it computes no longer gives the carry out, or the partial); at the last chunk a bit off the partial
    or one of its own cells off the record makes it 0, another lane's cell leaves it 1. A dead leaf's last chunk is 1 on
    the zero partial and 0 on its ungated chain's, and an index past the leaf's lanes makes every chunk 0. What holds a
    carry in to the honest one is the chunk before's check of the same row as its carry out: the last chunk's digits
    read only a residue sum's low bits and whether its quotient is 127, so a carried sum off in bit 31 leaves it 1."""
    rng = random.Random(45)
    n, lay = 3, H.Layout(8, 16, 96)
    a, b, sa, sw = _unit_rows(rng, 8, 16, 96)
    live, zy = [1] * 8, Z.records(a, b, sa, sw)
    ra, rb, ras, rbs = _lane_rows(lay, 1, a, b, sa, sw)
    cs = Z.lane_chunk_carries(ra, rb, ras, rbs, SALT, 5, ROOTS, 1, (8, 16), n)
    part = H.lane_partials(a, b, SALT, 5, ROOTS)[1]

    def ref(ch, cin=None, cout=None, partial=part, rec=zy, admit=live, rows=(ra, rb, ras, rbs), idx=1):
        last = ch == n - 1
        return Z.lane_chunk_reference(_chunk_rows(rows[0], n, ch), _chunk_rows(rows[1], n, ch), rows[2], rows[3],
                                      admit if last else None, rec if last else None, partial if last else None,
                                      (cin or cs[ch - 1]) if ch else None, None if last else (cout or cs[ch]), SALT, 5,
                                      ROOTS, idx, (8, 16), n, ch)

    assert [ref(ch) for ch in range(n)] == [1, 1, 1]
    for field, at in (("state", 3), ("t", 30), ("res", 77), ("z", 9), ("words", 1), ("sa", 2), ("sw", 7)):
        assert ref(0, cout=_off(cs[0], field, at, 4)) == 0 and ref(1, cout=_off(cs[1], field, at, 31)) == 0
        assert ref(1, cin=_off(cs[0], field, at, 4)) == 0 and ref(2, cin=_off(cs[1], field, at, 4)) == 0
    assert ref(2, cin=_off(cs[1], "res", 77, 31)) == 1 and ref(1, cout=_off(cs[1], "res", 77, 31)) == 0
    assert ref(2, partial=[*part[:5], part[5] ^ 1 << 12, *part[6:]]) == 0
    for other, want in ((1, 0), (2, 1)):
        bad = (list(zy[0]), zy[1])
        bad[0][Z.lane_cells(lay, other)[20]] ^= 1
        assert ref(2, rec=bad) == want
    empty = [[0] * 96 for _ in range(8)]
    rows0 = _lane_rows(lay, 1, empty, b, [0] * 8, sw)
    dead = Z.lane_chunk_carries(*rows0, SALT, 5, ROOTS, 1, (8, 16), n)
    zy0 = Z.records(empty, b, [0] * 8, sw)
    zeros = [0] * (2 * lay.sets)
    assert ref(2, cin=dead[1], partial=zeros, rec=zy0, admit=[0] * 8, rows=rows0) == 1
    assert ref(2, cin=dead[1], partial=dead[2]["words"], rec=zy0, admit=[0] * 8, rows=rows0) == 0
    assert [ref(ch, idx=idx) for ch in range(n) for idx in (1 + lay.lanes, 1 | 1 << 15)] == [0] * 6


@pytest.mark.parametrize("k,n", [(6144, 2), (14336, 4)])
def test_a_chunk_reads_its_rows_carries_and_index_and_returns_ok(k, n):
    """At the served 32 × 32 leaf: chunk c reads its 4 A and 8 B rows at K / n entries with their scales, the carry in past
    the first (880 + 8 c KS_c bytes) and the carry out before the last (880 + 8 (c + 1) KS_c), the last chunk the leaf's
    admit bits, record and the lane's whole partial (2 (KS + m) words), and every chunk the tile's salt, position and roots
    and the index; its one output ``ok`` is a sink, so no chunk is a seed. `statement_rows` reads them in that order, and
    `lane_statements` is the whole lane up to K = 4096 and the n chunks past it."""
    t = _tile(H.PouwH1Tile, 32, 32, k)
    kc, ks = k // n, k // n // 32
    lay = H.Layout(32, 32, k)
    tail = [(nm, tp.leaves) for nm, tp in t.signature()[0][2:]] + [("lane", 16)]
    assert Z.lane_statements(t) == [Z.lane_chunk(t, n, c) for c in range(n)]
    assert Z.lane_statements(_tile(H.PouwH1Tile, 32, 32, 4096)) == [Z.lane(_tile(H.PouwH1Tile, 32, 32, 4096))]
    for c in range(n):
        d = Z.lane_chunk(t, n, c)
        params, ret = d.signature()
        want = [("a", 4 * (8 * kc + 32)), ("b", 8 * (8 * kc + 32))]
        want += [("carry_in", 8 * (880 + 8 * c * ks))] if c else []
        want += ([("admit", 32), ("zy", 8 * 6144), ("partials", 64 * lay.sets)] if c == n - 1 else
                 [("carry_out", 8 * (880 + 8 * (c + 1) * ks))])
        assert [(nm, tp.leaves) for nm, tp in params] == want + tail and ret.fields == (("ok", Z.BIT),)
        assert params[0][1].elem == Z.ROW(kc)
        decl = d.defn.ports(d.S())
        assert {nm: p.kind for nm, p in decl.params.items()} == {"salt": "coin", "position": "position"}
        assert set(decl.ret) == {"ok"} and decl.ret["ok"].sink and PT._coin_role(d) is None
        rows = Z.statement_rows(d)
        assert [p for p, _ps in rows] == [*(f"a{i}" for i in range(4)), *(f"b{i}" for i in range(8)),
                                          *(nm for nm, _l in want[2:]), "pub"]
        assert Z.statement_ports(d)[-1] == ("o0", [("ok", 0, 1)])


@pytest.mark.parametrize("bindings,why", [
    (lambda: dict(TILE=_tile(H.PouwH1Tile3, 8, 16), CHUNKS=2, CHUNK=0), "over PouwH1Tile_v1"),
    (lambda: dict(TILE=_tile(H.PouwH1Tile4, 8, 16), CHUNKS=2, CHUNK=0), "an H1 tile"),
    (lambda: dict(TILE=_tile(r=4, c=8), CHUNKS=2, CHUNK=0), "whole bits"),
    (lambda: dict(TILE=_tile(r=8, c=16), CHUNKS=1, CHUNK=0), "at least 2"),
    (lambda: dict(TILE=_tile(r=8, c=16), CHUNKS=2, CHUNK=2), r"index in \[0, 2\)"),
    (lambda: dict(TILE=_tile(r=8, c=16, k=96), CHUNKS=2, CHUNK=0), "whole 32-entry segments"),
])
def test_a_chunk_of_another_tile_or_out_of_its_chunks_is_refused(bindings, why):
    with pytest.raises(ValueError, match=why):
        bind(Z.PouwH1LaneZyChunk, **bindings()).signature()


@pytest.mark.slow
def test_a_lane_chunk_on_bits_is_its_word_reference():
    """`PouwH1LaneZyChunk` over an 8 × 16 leaf on bits against `lane_chunk_reference`, about 2 minutes: the first and last
    of 2 chunks at K = 64 and the middle of 3 at K = 96, each on every lane's honest rows and carries (1) and an index past
    the leaf's lanes (0); the first and middle with a bit off their carry out (0); the middle and last with a carried
    scale off (0) and a carried state word off (0); the last with a bit off its partial (0), one of its own cells off (0),
    and a dead leaf with its zero partial (1) and its ungated one (0). It is the only test of a chunk's gates: its noise at
    the chunk's offsets and U's, the chain, residue sums and z continued from the carry, and the checks of the carry out,
    the scales and the last chunk's partial and cells."""
    rng = random.Random(46)
    got, want = [], []
    for k, n, ch in ((64, 2, 0), (64, 2, 1), (96, 3, 1)):
        lay, last = H.Layout(8, 16, k), ch == n - 1
        a, b, sa, sw = _unit_rows(rng, 8, 16, k)
        live, zy = [1] * 8, Z.records(a, b, sa, sw)
        parts = H.lane_partials(a, b, SALT, 5, ROOTS)
        empty = [[0] * k for _ in range(8)]
        zy0 = Z.records(empty, b, [0] * 8, sw)
        cases = []
        for ln in range(lay.lanes):
            rows = _lane_rows(lay, ln, a, b, sa, sw)
            cs = Z.lane_chunk_carries(*rows, SALT, 5, ROOTS, ln, (8, 16), n)
            cases.append((ln, rows, cs, live, zy, parts[ln]))
        ln, rows, cs, *_ = cases[1]
        cases.append((1 + lay.lanes, rows, cs, live, zy, parts[1]))
        if not last:
            cases.append((1, rows, [*cs[:ch], _off(cs[ch], "words", 2 * ch, 17), *cs[ch + 1:]], live, zy, parts[1]))
        if ch:
            cases.append((1, rows, [*cs[:ch - 1], _off(cs[ch - 1], "sw", 3, 0), *cs[ch:]], live, zy, parts[1]))
            cases.append((1, rows, [*cs[:ch - 1], _off(cs[ch - 1], "state", 11, 22), *cs[ch:]], live, zy, parts[1]))
        if last:
            cell = (list(zy[0]), zy[1])
            cell[0][Z.lane_cells(lay, 1)[5]] ^= 8
            cases += [(1, rows, cs, live, zy, [parts[1][0] ^ 1 << 3, *parts[1][1:]]), (1, rows, cs, live, cell, parts[1])]
            rows0 = _lane_rows(lay, 2, empty, b, [0] * 8, sw)
            dead = Z.lane_chunk_carries(*rows0, SALT, 5, ROOTS, 2, (8, 16), n)
            cases += [(2, rows0, dead, [0] * 8, zy0, [0] * (2 * lay.sets)), (2, rows0, dead, [0] * 8, zy0, dead[-1]["words"])]
        d = bind(Z.PouwH1LaneZyChunk, TILE=_tile(H.PouwH1Tile, 8, 16, k), CHUNKS=n, CHUNK=ch)
        names = [p for p, _t in d.signature()[0]]
        inputs = []
        for idx, (ra, rb, ras, rbs), cs, admit, rec, partial in cases:
            ca, cb = _chunk_rows(ra, n, ch), _chunk_rows(rb, n, ch)
            cin, cout = cs[ch - 1] if ch else None, None if last else cs[ch]
            want.append(Z.lane_chunk_reference(ca, cb, ras, rbs, admit if last else None, rec if last else None,
                                               partial if last else None, cin, cout, SALT, 5, ROOTS, idx, (8, 16), n, ch))
            v = {"a": _row_bits(ca, ras), "b": _row_bits(cb, rbs), "salt": SD.leaves(SALT, True),
                 "position": SD.position_leaves(5, True), "roots": SD.leaves(ROOTS[0], True) + SD.leaves(ROOTS[1], True),
                 "lane": _bits(idx, 16), "admit": list(admit), "partials": _word_bits(partial),
                 "zy": _word_bits(rec[0]) + [x for y in rec[1] for x in _bits(y, 16)]}
            v.update({nm: _word_bits(Z.carry_words(cr)) for nm, cr in (("carry_in", cin), ("carry_out", cout)) if cr})
            inputs.append([v[p] for p in names])
        out = evaluate_bits(d, *[np.array([row[p] for row in inputs], np.uint8) for p in range(len(names))])
        got += [int(o) for o in out.reshape(len(cases))]
    assert want == [1, 1, 1, 1, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0] + [1, 1, 1, 1, 0, 0, 0, 0]
    assert got == want


# -- the unit on chunk rows ------------------------------------------------------------------------------------------------------

def _split(rows, scales, n):
    """Each row's n chunk rows and their scales, row i's chunk c at n i + c, each chunk row its row's scale."""
    kc = len(rows[0]) // n
    return [r[c * kc:c * kc + kc] for r in rows for c in range(n)], [s for s in scales for _c in range(n)]


@pytest.mark.parametrize("r,c,k,n", [(4, 8, 64, 2), (32, 32, 6144, 2), (32, 32, 14336, 4)])
def test_the_unit_on_chunk_rows_reads_the_tables_chunk_rows_and_returns_the_units_outputs(r, c, k, n):
    """`PouwH1TileZyChunks` takes the window's tile parameters (`h1window.TILE_PARAMS`): n R A and n C B rows of `ROW`(K / n)
    in place of UNIT's R and C of `ROW`(K), then UNIT's record, salt, position and roots, and returns UNIT's receipt and ok
    with its ports (the salt a coin, the receipt a sink, so it is no seed); `window_unit` binds it at the served K past
    4096 with n = `chunks`(K), and UNIT itself up to 4096."""
    t = _tile(H.PouwH1Tile, r, c, k)
    u = Z.unit_chunks(t, n)
    params, ret = u.signature()
    inner = Z.unit(t).signature()
    assert [nm for nm, _t in params] == list(HW.TILE_PARAMS) and ret == inner[1]
    assert [(nm, tp.n, tp.elem) for nm, tp in params[:2]] == [("a", n * r, Z.ROW(k // n)), ("b", n * c, Z.ROW(k // n))]
    assert params[2:] == inner[0][2:]
    decl = u.defn.ports(u.S())
    assert {nm: p.kind for nm, p in decl.params.items()} == {"salt": "coin", "position": "position"}
    assert set(decl.ret) == {"receipt"} and decl.ret["receipt"].sink and PT._coin_role(u) is None
    if r == 32:
        assert Z.window_unit(t) == u and Z.window_unit(_tile(H.PouwH1Tile, r, c, 4096)) == Z.unit(_tile(H.PouwH1Tile, r, c, 4096))


@pytest.mark.parametrize("bindings,why", [
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile3, 4, 8)), CHUNKS=2), "over PouwH1Tile_v1, not PouwH1Tile_v3"),
    (lambda: dict(UNIT=_tile(H.PouwH1Tile, 4, 8), CHUNKS=2), "PouwH1TileZy_v1 over an H1 tile"),
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile, 4, 8)), CHUNKS=1), "at least 2"),
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile, 4, 8, 96)), CHUNKS=2), "whole 32-entry segments"),
])
def test_a_unit_on_chunk_rows_of_another_unit_or_chunking_is_refused(bindings, why):
    with pytest.raises(ValueError, match=why):
        bind(Z.PouwH1TileZyChunks, **bindings())


def test_a_window_unit_at_a_chunked_k_is_refused_over_v3_and_v4():
    """Past `CHUNK_MAX` a drawn leaf is proved as lanes in chunks, which only v1 has: `window_unit` refuses v3 there (the
    wrapper's refusal) and v4 anywhere (the zy unit's), and takes v3 at K <= 4096."""
    for defn, why in ((H.PouwH1Tile3, "over PouwH1Tile_v1"), (H.PouwH1Tile4, "an H1 tile")):
        with pytest.raises(ValueError, match=why):
            Z.window_unit(_tile(defn, 32, 32, 6144))
    assert Z.window_unit(_tile(H.PouwH1Tile3, 32, 32, 4096)) == Z.unit(_tile(H.PouwH1Tile3, 32, 32, 4096))


def test_the_unit_on_chunk_rows_reads_the_bits_of_the_leaves_its_table_commits():
    """A chunk row's bits are its `chunk_leaves` leaf's bytes, least significant bit first: its K / n codes, then its row's
    scale, so the unit's ``a`` and ``b`` are its table's leaves n r + c in order."""
    rng = random.Random(48)
    for k, n in ((4096, 1), (6144, 2), (14336, 4)):
        assert Z.chunks(k) == n
        a, _b, sa, _sw = _unit_rows(rng, 3, 1, k)
        raw = b"".join(leaf for row, s in zip(a, sa) for leaf in Z.chunk_leaves(bytes(row) + s.to_bytes(4, "little"), k))
        rows, scales = _split(a, sa, n)
        assert _row_bits(rows, scales) == [(byte >> i) & 1 for byte in raw for i in range(8)]


def test_the_unit_on_chunk_rows_on_words_is_the_unit_on_its_whole_rows():
    """`chunks_reference`: `reference` on each row's chunks joined, so the honest chunk rows give the whole unit's receipt
    and ok; a later chunk's scale off its row's leaves the receipt and makes ok 0, and a code off in a later chunk moves
    the receipt as it moves the whole row's."""
    rng = random.Random(49)
    a, b, sa, sw = _unit_rows(rng, 4, 8, 192)
    zy = Z.records(a, b, sa, sw)
    whole = Z.reference(a, b, sa, sw, zy, SALT, 5, ROOTS)
    for n in (2, 3):
        (ca, csa), (cb, csw) = _split(a, sa, n), _split(b, sw, n)
        assert whole[1] == 1 and Z.chunks_reference(ca, cb, csa, csw, zy, SALT, 5, ROOTS, n) == whole
        for at in (1, n * 2 + n - 1):
            off = list(csa)
            off[at] ^= 1 << 7
            assert Z.chunks_reference(ca, cb, off, csw, zy, SALT, 5, ROOTS, n) == (whole[0], 0)
        moved = [list(row) for row in cb]
        moved[n * 3 + n - 1][5] ^= 0x40
        b2 = [list(row) for row in b]
        b2[3][(n - 1) * (192 // n) + 5] ^= 0x40
        assert Z.chunks_reference(ca, moved, csa, csw, zy, SALT, 5, ROOTS, n)[0] == Z.reference(
            a, b2, sa, sw, zy, SALT, 5, ROOTS)[0] != whole[0]


@pytest.mark.slow
def test_the_unit_on_chunk_rows_on_bits_is_its_word_reference():
    """`PouwH1TileZyChunks` over the 4 × 8 unit at K = 64 in 2 chunk rows a row, on bits against `chunks_reference`, about
    50 s: the honest chunk rows (ok 1), a later A chunk's scale off (0), a later B chunk's (0) and a record cell off (0),
    each with the whole unit's receipt. It is the only test of the wrapper's gates: the whole rows it joins from the chunk
    rows (the receipt and the record check see the row's codes in order and chunk 0's scale) and the scales' equality."""
    rng = random.Random(47)
    n = 2
    a, b, sa, sw = _unit_rows(rng, 4, 8, 64)
    (ca, csa), (cb, csw) = _split(a, sa, n), _split(b, sw, n)
    zy = Z.records(a, b, sa, sw)
    off_a, off_b = list(csa), list(csw)
    off_a[3] ^= 1 << 5
    off_b[2 * 5 + 1] ^= 1 << 30
    cases = [(csa, csw, zy), (off_a, csw, zy), (csa, off_b, zy), (csa, csw, (zy[0], [zy[1][0] ^ 1, *zy[1][1:]]))]
    u = Z.unit_chunks(_tile(H.PouwH1Tile, 4, 8, 64), n)
    names = [p for p, _t in u.signature()[0]]
    inputs, want = [], []
    for xsa, xsw, rec in cases:
        want.append(Z.chunks_reference(ca, cb, xsa, xsw, rec, SALT, 5, ROOTS, n))
        v = {"a": _row_bits(ca, xsa), "b": _row_bits(cb, xsw), "zy": _word_bits(rec[0]) + [x for y in rec[1] for x in _bits(y, 16)],
             "salt": SD.leaves(SALT, True), "position": SD.position_leaves(5, True),
             "roots": SD.leaves(ROOTS[0], True) + SD.leaves(ROOTS[1], True)}
        inputs.append([v[p] for p in names])
    out = evaluate_bits(u, *[np.array([row[p] for row in inputs], np.uint8) for p in range(len(names))])
    got = [([_int(o[32 * w:32 * w + 32]) for w in range((len(o) - 1) // 32)], int(o[-1])) for o in out]
    receipt = Z.reference(a, b, sa, sw, zy, SALT, 5, ROOTS)[0]
    assert want == [(receipt, 1), (receipt, 0), (receipt, 0), (receipt, 0)]
    assert got == want


# -- the unit on chunk rows over J10-R ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("defn", [H.PouwH1Tile5, H.PouwH1Tile6], ids=["v5", "v6"])
@pytest.mark.parametrize("r,c,k,n", [(4, 8, 64, 2), (32, 32, 14336, 4)])
def test_the_j10r_unit_on_chunk_rows_takes_the_call_and_blocks_after_the_tiles_parameters(defn, r, c, k, n):
    """Over J10-R's tiles the unit on chunk rows is `PouwH1TileZyChunks_v2`, with `h1window.TILE_PARAMS_J10R`: n R A and n C
    B rows of `ROW`(K / n), then UNIT's record, salt, position, roots, call and blocks, its outputs and ports UNIT's;
    `window_unit` binds it past 4096 (Llama-3.1-8B's down projection at K = 14336 in 4 chunks), and UNIT up to 4096. v1's
    stays `PouwH1TileZyChunks_v1`."""
    t = _tile(defn, r, c, k)
    u = Z.unit_chunks(t, n)
    params, ret = u.signature()
    inner = Z.unit(t).signature()
    assert u.defn is Z.PouwH1TileZyChunks2 and u.id.startswith("PouwH1TileZyChunks_v2{")
    assert [nm for nm, _t in params] == list(HW.TILE_PARAMS_J10R) and ret == inner[1]
    assert [(nm, tp.n, tp.elem) for nm, tp in params[:2]] == [("a", n * r, Z.ROW(k // n)), ("b", n * c, Z.ROW(k // n))]
    assert params[2:] == inner[0][2:]
    decl = u.defn.ports(u.S())
    assert {nm: p.kind for nm, p in decl.params.items()} == {"salt": "coin", "position": "position"}
    assert set(decl.ret) == {"receipt"} and decl.ret["receipt"].sink and PT._coin_role(u) is None
    assert Z.unit_chunks(_tile(H.PouwH1Tile, r, c, k), n).defn is Z.PouwH1TileZyChunks
    if r == 32:
        assert Z.window_unit(t) == u and Z.window_unit(_tile(defn, r, c, 4096)) == Z.unit(_tile(defn, r, c, 4096))


@pytest.mark.parametrize("bindings,why", [
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile, 4, 8)), CHUNKS=2), "PouwH1Tile_v1's unit on chunk rows is .*_v1"),
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile3, 4, 8)), CHUNKS=2), "over J10-R's tiles"),
    (lambda: dict(UNIT=_tile(H.PouwH1Tile5, 4, 8), CHUNKS=2), "PouwH1TileZy_v1 over a J10-R tile"),
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile5, 4, 8)), CHUNKS=1), "at least 2"),
    (lambda: dict(UNIT=Z.unit(_tile(H.PouwH1Tile5, 4, 8, 96)), CHUNKS=2), "whole 32-entry segments"),
])
def test_a_j10r_unit_on_chunk_rows_of_another_unit_or_chunking_is_refused(bindings, why):
    """v2 takes only J10-R's zy unit, at whole 32-entry segments a chunk, and v1 still refuses a J10-R tile."""
    with pytest.raises(ValueError, match=why):
        bind(Z.PouwH1TileZyChunks2, **bindings())
    with pytest.raises(ValueError, match="over PouwH1Tile_v1, not PouwH1Tile_v5"):
        bind(Z.PouwH1TileZyChunks, UNIT=Z.unit(_tile(H.PouwH1Tile5, 4, 8)), CHUNKS=2)


#: a 4-tile-row slot's call at the J10-R unit (1, 0), its first unit at position 40 and the unit at 41
J10R_M, J10R_I, J10R_J, J10R_CALL, J10R_AT = 16, 1, 0, 40, 41


@pytest.mark.parametrize("stream", [False, True], ids=["v5", "v6"])
def test_the_j10r_unit_on_chunk_rows_on_words_is_j10r_reference_on_its_whole_rows(stream):
    """`chunks_reference` with ``j10r``: `h1tile.j10r_reference` on each row's chunks joined, at the blocks
    `h1tile.j10r_blocks` gives at the whole K (the call-scale stream's rows and columns), so a J10-R unit of
    Llama-3.1-8B's down projection (K = 14336, 4 chunks, on a 4 × 8 tile) has its whole-row receipt and ok 1. Blocks
    taken at a chunk's K / n move the receipt; a later chunk's scale off its row's leaves the receipt and makes ok 0; a
    code off in the last chunk moves the receipt as it moves the whole row's."""
    rng, k, n = random.Random(50), 14336, 4
    a, b, sa, sw = _unit_rows(rng, 4, 8, k)
    zy = Z.records(a, b, sa, sw)
    blocks = H.j10r_blocks(J10R_M, k, (4, 8), J10R_I, J10R_J)
    j10r = (J10R_CALL, blocks)
    z, receipt = H.j10r_reference(a, b, SALT, J10R_AT, ROOTS, J10R_CALL, blocks, stream=stream)
    assert Z.reference(a, b, sa, sw, zy, SALT, J10R_AT, ROOTS, stream, j10r) == (receipt, 1)
    (ca, csa), (cb, csw) = _split(a, sa, n), _split(b, sw, n)
    assert Z.chunks_reference(ca, cb, csa, csw, zy, SALT, J10R_AT, ROOTS, n, stream, j10r) == (receipt, 1)
    chunk_blocks = H.j10r_blocks(J10R_M, k // n, (4, 8), J10R_I, J10R_J)
    assert chunk_blocks != blocks and Z.chunks_reference(ca, cb, csa, csw, zy, SALT, J10R_AT, ROOTS, n, stream,
                                                         (J10R_CALL, chunk_blocks))[0] != receipt
    assert Z.chunks_reference(ca, cb, csa, csw, zy, SALT, J10R_AT, ROOTS, n)[0] != receipt, "v1's noise is no J10-R's"
    off = list(csw)
    off[n * 6 + n - 1] ^= 1 << 7
    assert Z.chunks_reference(ca, cb, csa, off, zy, SALT, J10R_AT, ROOTS, n, stream, j10r) == (receipt, 0)
    moved = [list(row) for row in ca]
    moved[n * 2 + n - 1][7] ^= 0x40
    a2 = [list(row) for row in a]
    a2[2][(n - 1) * (k // n) + 7] ^= 0x40
    got = Z.chunks_reference(moved, cb, csa, csw, zy, SALT, J10R_AT, ROOTS, n, stream, j10r)
    assert got[0] == H.j10r_reference(a2, b, SALT, J10R_AT, ROOTS, J10R_CALL, blocks, stream=stream)[1] != receipt
    assert got[1] == 0, "the record is the honest rows' product, not the moved row's"


def _j10r_tail(position, call, blocks):
    return {"salt": SD.leaves(SALT, True), "position": SD.position_leaves(position, True),
            "roots": SD.leaves(ROOTS[0], True) + SD.leaves(ROOTS[1], True), "call": SD.position_leaves(call, True),
            "blocks": SD.position_leaves(blocks[0], True) + SD.position_leaves(blocks[1], True)}


@pytest.mark.slow
def test_the_j10r_unit_on_chunk_rows_on_bits_is_its_word_reference():
    """`PouwH1TileZyChunks_v2` over the 4 × 8 v5 unit at K = 64 in 2 chunk rows a row, on bits against `chunks_reference`
    with ``j10r``, about a minute: the honest chunk rows (ok 1), a later A chunk's scale off (0), a code off in a later
    B chunk (the moved row's receipt, ok 0), and the blocks of another unit of the call (another receipt, ok 1). It is
    the only test of v2's gates: the whole rows it joins, the call and blocks it passes on to UNIT and the scales'
    equality."""
    rng, n, k = random.Random(51), 2, 64
    a, b, sa, sw = _unit_rows(rng, 4, 8, k)
    (ca, csa), (cb, csw) = _split(a, sa, n), _split(b, sw, n)
    zy = Z.records(a, b, sa, sw)
    blocks = H.j10r_blocks(J10R_M, k, (4, 8), J10R_I, J10R_J)
    other = H.j10r_blocks(J10R_M, k, (4, 8), J10R_I + 1, J10R_J)
    off_a = list(csa)
    off_a[3] ^= 1 << 5
    moved = [list(row) for row in cb]
    moved[n * 5 + 1][9] ^= 0x10
    cases = [(cb, csa, blocks), (cb, off_a, blocks), (moved, csa, blocks), (cb, csa, other)]
    u = Z.unit_chunks(_tile(H.PouwH1Tile5, 4, 8, k), n)
    names = [p for p, _t in u.signature()[0]]
    inputs, want = [], []
    for xb, xsa, blk in cases:
        want.append(Z.chunks_reference(ca, xb, xsa, csw, zy, SALT, J10R_AT, ROOTS, n, j10r=(J10R_CALL, blk)))
        v = {"a": _row_bits(ca, xsa), "b": _row_bits(xb, csw),
             "zy": _word_bits(zy[0]) + [x for y in zy[1] for x in _bits(y, 16)], **_j10r_tail(J10R_AT, J10R_CALL, blk)}
        inputs.append([v[p] for p in names])
    out = evaluate_bits(u, *[np.array([row[p] for row in inputs], np.uint8) for p in range(len(names))])
    got = [([_int(o[32 * w:32 * w + 32]) for w in range((len(o) - 1) // 32)], int(o[-1])) for o in out]
    receipt = H.j10r_reference(a, b, SALT, J10R_AT, ROOTS, J10R_CALL, blocks)[1]
    assert [(w[0] == receipt, w[1]) for w in want] == [(True, 1), (True, 0), (False, 0), (False, 1)]
    assert got == want


# -- a J10-R leaf's lanes ------------------------------------------------------------------------------------------------------

#: an 8 × 16 leaf of a 16-row slot's call at its unit (1, 0), and blocks whose sums carry across a counter's u32 halves
J10R_LEAF_BLOCKS = H.j10r_blocks(16, 64, (8, 16), 1, 0)
J10R_CARRY_BLOCKS = ((1 << 32) - 3, (5 << 32) - 1)


def test_a_j10r_leafs_lanes_on_words_are_v5s_receipt_split_by_lane():
    """Over J10-R's v5 (`lane_partial`, `lane_reference`, `lane_chunk_carries` and `lane_chunk_reference` with ``j10r``):
    the lanes' partials XOR to `h1tile.j10r_reference`'s receipt, at the call's blocks and at blocks whose counters carry
    into their high halves; each lane's reference is 1 on its partial and 0 on today's (v1's) partial or another unit's
    blocks'; its chain in 2 and 4 chunks ends at the whole lane's partial, each chunk 1 on its honest rows and carries."""
    rng = random.Random(52)
    for r, c, k, blocks in ((8, 16, 64, J10R_LEAF_BLOCKS), (8, 16, 64, J10R_CARRY_BLOCKS), (32, 32, 512, J10R_CARRY_BLOCKS)):
        a, b, sa, sw = _unit_rows(rng, r, c, k)
        lay, live, j10r = H.Layout(r, c, k), [1] * r, (J10R_CALL, blocks)
        zy = Z.records(a, b, sa, sw)
        receipt = H.j10r_reference(a, b, SALT, J10R_AT, ROOTS, J10R_CALL, blocks)[1]
        parts = [Z.lane_partial(*_lane_rows(lay, ln, a, b, sa, sw)[:2], live, SALT, J10R_AT, ROOTS, ln, (r, c), j10r)
                 for ln in range(lay.lanes)]
        assert Z.combine_reference(parts, receipt) == 1
        today = H.lane_partials(a, b, SALT, J10R_AT, ROOTS)
        other = (J10R_CALL, (blocks[0] + lay.rb, blocks[1]))
        for ln in (0, lay.lanes - 1):
            rows = _lane_rows(lay, ln, a, b, sa, sw)
            assert Z.lane_reference(*rows, live, zy, parts[ln], SALT, J10R_AT, ROOTS, ln, (r, c), j10r) == 1
            assert Z.lane_reference(*rows, live, zy, today[ln], SALT, J10R_AT, ROOTS, ln, (r, c), j10r) == 0
            assert Z.lane_reference(*rows, live, zy, parts[ln], SALT, J10R_AT, ROOTS, ln, (r, c), other) == 0
            for n in (2, 4) if k % 128 == 0 else (2,):
                cs = Z.lane_chunk_carries(*rows, SALT, J10R_AT, ROOTS, ln, (r, c), n, j10r)
                assert cs[-1]["words"] == parts[ln]
                for ch in range(n):
                    last = ch == n - 1
                    assert Z.lane_chunk_reference(_chunk_rows(rows[0], n, ch), _chunk_rows(rows[1], n, ch), *rows[2:],
                                                  live if last else None, zy if last else None, parts[ln] if last else None,
                                                  cs[ch - 1] if ch else None, None if last else cs[ch], SALT, J10R_AT,
                                                  ROOTS, ln, (r, c), n, ch, j10r) == 1


@pytest.mark.parametrize("k", [4096, 16384, 65536])
def test_a_j10r_leafs_lane_takes_the_call_and_blocks_before_the_index(k):
    """At the served 32 × 32 leaf over v5, `lane_statements` is `PouwH1LaneZy_v2` up to K = 4096 and `chunks`(K)
    `PouwH1LaneZyChunk_v2` past it (4 at 16,384, 16 at 65,536), each with v1's parameters and v5's ``call`` and
    ``blocks`` between ``roots`` and ``lane``, read as one public row (`PUBLIC_J10R`, whole 16-bit words), its one output
    ``ok`` a sink; the combine is `PouwH1Combine_v2` over the v5 tile, v1's rows and then the 32 n statements' ok rows
    (``ok0`` ..), one 16-bit word each, its one output the leaf's ok."""
    t, t1 = _tile(H.PouwH1Tile5, 32, 32, k), _tile(H.PouwH1Tile, 32, 32, k)
    stmts, n = Z.lane_statements(t), Z.chunks(k)
    assert [s.defn for s in stmts] == [Z.PouwH1LaneZy2] if n == 1 else [Z.PouwH1LaneZyChunk2] * n
    comb, comb1 = Z.combine(t), Z.combine(t1)
    assert comb == bind(Z.PouwH1Combine2, TILE=t) and comb1 == bind(Z.PouwH1Combine, TILE=t1)
    assert Z.leaf_statements(H.Layout(32, 32, k)) == 32 * n
    assert dict(comb.signature()[0])["oks"] == Array(32 * n, Z.OK_WORD)
    assert Z.statement_rows(comb) == Z.statement_rows(comb1) + [(f"ok{i}", [f"oks[{i}]"]) for i in range(32 * n)]
    assert Z.statement_ports(comb)[-2:] == [(f"ok{32 * n - 1}", [(f"oks[{32 * n - 1}]", 0, 16)]), ("o0", [("ok", 0, 1)])]
    for s, s1 in zip(stmts, Z.lane_statements(t1), strict=True):
        params, ret = s.signature()
        names = [nm for nm, _t in params]
        assert names == [nm for nm, _t in s1.signature()[0]][:-1] + ["call", "blocks", "lane"]
        assert dict(params)["call"] == SD.POS_BITS and dict(params)["blocks"] == Array(2, SD.POS_BITS)
        assert Z.statement_rows(s)[-1] == ("pub", list(Z.PUBLIC_J10R))
        assert Z.statement_rows(s)[:-1] == Z.statement_rows(s1)[:-1]
        assert sum(tp.leaves for nm, tp in params if nm in Z.PUBLIC_J10R) % 16 == 0
        assert ret.fields == (("ok", Z.BIT),) and Z.statement_ports(s)[-1] == ("o0", [("ok", 0, 1)])
        decl = s.defn.ports(s.S())
        assert {nm: p.kind for nm, p in decl.params.items()} == {"salt": "coin", "position": "position"}
        assert set(decl.ret) == {"ok"} and decl.ret["ok"].sink and PT._coin_role(s) is None


@pytest.mark.parametrize("d,bindings,why", [
    (lambda: Z.PouwH1LaneZy2, lambda: dict(TILE=_tile(H.PouwH1Tile, 8, 16)), "PouwH1Tile_v1's lane is PouwH1LaneZy_v1"),
    (lambda: Z.PouwH1LaneZy2, lambda: dict(TILE=_tile(H.PouwH1Tile6, 8, 16)), r"an H1 tile \(PouwH1Tile_v5\)"),
    (lambda: Z.PouwH1LaneZy2, lambda: dict(TILE=_tile(H.PouwH1Tile5, 4, 8)), "whole bits"),
    (lambda: Z.PouwH1LaneZy, lambda: dict(TILE=_tile(H.PouwH1Tile5, 8, 16)), "an H1 tile"),
    (lambda: Z.PouwH1Combine, lambda: dict(TILE=_tile(H.PouwH1Tile5, 8, 16)), "an H1 tile"),
    (lambda: Z.PouwH1Combine2, lambda: dict(TILE=_tile(H.PouwH1Tile6, 8, 16)), r"an H1 tile \(PouwH1Tile_v5\)"),
    (lambda: Z.PouwH1Combine2, lambda: dict(TILE=_tile(H.PouwH1Tile, 8, 16)), r"an H1 tile \(PouwH1Tile_v5\)"),
    (lambda: Z.PouwH1LaneZyChunk2, lambda: dict(TILE=_tile(H.PouwH1Tile, 8, 16), CHUNKS=2, CHUNK=0), "over PouwH1Tile_v5"),
    (lambda: Z.PouwH1LaneZyChunk2, lambda: dict(TILE=_tile(H.PouwH1Tile6, 8, 16), CHUNKS=2, CHUNK=0), "over PouwH1Tile_v5"),
    (lambda: Z.PouwH1LaneZyChunk2, lambda: dict(TILE=_tile(H.PouwH1Tile5, 8, 16), CHUNKS=2, CHUNK=2), r"in \[0, 2\)"),
    (lambda: Z.PouwH1LaneZyChunk, lambda: dict(TILE=_tile(H.PouwH1Tile5, 8, 16), CHUNKS=2, CHUNK=0), "over PouwH1Tile_v1"),
])
def test_a_j10r_lane_of_another_tile_is_refused(d, bindings, why):
    """v2's lane, lane chunk and combine take only J10-R's v5 (v6's one-stream z has no lane in chunks), and v1's
    refuse it."""
    with pytest.raises(ValueError, match=why):
        bind(d(), **bindings()).signature()


def test_a_j10r_combine_on_bits_folds_the_statements_ok_rows_into_its_own():
    """`PouwH1Combine_v2` over an 8 × 16 v5 leaf at K = 64 (4 lanes, one statement each) on bits against
    `combine_reference` with ``oks``: 1 on the honest partials, receipt and four ok rows of the word 1; 0 with one ok row
    0, with one the word 3 (a bit past the ok set), with every ok row 0, and with the receipt off by one bit."""
    rng = random.Random(55)
    lay = H.Layout(8, 16, 64)
    parts = [[rng.randrange(1 << 32) for _ in range(2 * lay.sets)] for _ in range(lay.lanes)]
    receipt = [0] * (2 * lay.sets)
    for p in parts:
        receipt = [x ^ v for x, v in zip(receipt, p)]
    ones = [1] * Z.leaf_statements(lay)
    cases = [(parts, receipt, ones), (parts, receipt, [1, 1, 0, 1]), (parts, receipt, [1, 3, 1, 1]),
             (parts, receipt, [0] * 4), (parts, [receipt[0] ^ 1, *receipt[1:]], ones)]
    args = [np.array([[x for p in ps for x in _word_bits(p)] for ps, _r, _o in cases], np.uint8),
            np.array([_word_bits(r) for _ps, r, _o in cases], np.uint8),
            np.array([[x for w in oks for x in _bits(w, 16)] for _ps, _r, oks in cases], np.uint8)]
    out = evaluate_bits(bind(Z.PouwH1Combine2, TILE=_tile(H.PouwH1Tile5, 8, 16)), *args).reshape(len(cases))
    assert [int(o) for o in out] == [Z.combine_reference(*c) for c in cases] == [1, 0, 0, 0, 0]


def _j10r_lane_inputs(names, ca, cb, ras, rbs, admit, rec, partial, cin, cout, idx, blocks):
    v = {"a": _row_bits(ca, ras), "b": _row_bits(cb, rbs), "admit": list(admit or []), "partials": _word_bits(partial or []),
         "zy": _word_bits(rec[0]) + [x for y in rec[1] for x in _bits(y, 16)] if rec else [], "lane": _bits(idx, 16),
         **_j10r_tail(J10R_AT, J10R_CALL, blocks)}
    v.update({nm: _word_bits(Z.carry_words(cr)) for nm, cr in (("carry_in", cin), ("carry_out", cout)) if cr})
    return [v[p] for p in names]


@pytest.mark.slow
def test_a_j10r_lane_on_bits_is_its_word_reference():
    """`PouwH1LaneZy_v2` over an 8 × 16 v5 leaf at K = 64 on bits against `lane_reference` with ``j10r``, every case at
    once: each lane on its honest rows (1), a lane with one of its cells off (0), with today's partial (0), at another
    unit's blocks (0), at blocks whose counters carry across their u32 halves (1), a dead leaf on its zero partial (1),
    and an index past the leaf's lanes (0). It is the only test of the v2 lane's gates: its seed_c and σ_R from one
    PouwCallSeeds, each block's counter the picked offset plus the blocks input, and v1's chain and checks on them."""
    rng = random.Random(53)
    a, b, sa, sw = _unit_rows(rng, 8, 16)
    lay, live = H.Layout(8, 16, 64), [1] * 8
    zy = Z.records(a, b, sa, sw)
    d = bind(Z.PouwH1LaneZy2, TILE=_tile(H.PouwH1Tile5, 8, 16))
    names = [p for p, _t in d.signature()[0]]

    def part(ln, blocks, admit=live, rows=(a, sa)):
        return Z.lane_partial(*_lane_rows(lay, ln, rows[0], b, rows[1], sw)[:2], admit, SALT, J10R_AT, ROOTS, ln, (8, 16),
                              (J10R_CALL, blocks))

    cell = (list(zy[0]), zy[1])
    cell[0][Z.lane_cells(lay, 2)[11]] ^= 2
    empty = [[0] * 64 for _ in range(8)]
    zy0 = Z.records(empty, b, [0] * 8, sw)
    today = H.lane_partials(a, b, SALT, J10R_AT, ROOTS)
    other = H.j10r_blocks(16, 64, (8, 16), 0, 0)
    blk = J10R_LEAF_BLOCKS
    cases = [(ln, (a, sa), live, zy, part(ln, blk), blk) for ln in range(lay.lanes)]
    cases += [(2, (a, sa), live, cell, part(2, blk), blk), (1, (a, sa), live, zy, today[1], blk),
              (3, (a, sa), live, zy, part(3, blk), other), (3, (a, sa), live, zy, part(3, J10R_CARRY_BLOCKS), J10R_CARRY_BLOCKS),
              (0, (empty, [0] * 8), [0] * 8, zy0, [0] * (2 * lay.sets), blk), (1 + lay.lanes, (a, sa), live, zy, part(1, blk), blk)]
    want, inputs = [], []
    for idx, (aa, ss), admit, rec, partial, blocks in cases:
        ra, rb, ras, rbs = _lane_rows(lay, idx % lay.lanes, aa, b, ss, sw)
        want.append(Z.lane_reference(ra, rb, ras, rbs, admit, rec, partial, SALT, J10R_AT, ROOTS, idx, (8, 16),
                                     (J10R_CALL, blocks)))
        inputs.append(_j10r_lane_inputs(names, ra, rb, ras, rbs, admit, rec, partial, None, None, idx, blocks))
    got = evaluate_bits(d, *[np.array([row[p] for row in inputs], np.uint8) for p in range(len(names))]).reshape(len(cases))
    assert want == [1] * lay.lanes + [0, 0, 0, 1, 1, 0]
    assert [int(o) for o in got] == want


@pytest.mark.slow
def test_a_j10r_lane_chunk_on_bits_is_its_word_reference():
    """`PouwH1LaneZyChunk_v2` over an 8 × 16 v5 leaf on bits against `lane_chunk_reference` with ``j10r``: the first and
    last of 2 chunks at K = 64, each on lanes 0 and 3's honest rows and carries (1), at another unit's blocks (0), at
    blocks whose counters carry across their u32 halves (1), and the last with a bit off its partial (0). It is the only
    test of the v2 chunk's gates: seed_c in every chunk, the counters from the blocks input at the chunk's entries and
    U's, and chunk 0's keys from σ_R."""
    rng, n = random.Random(54), 2
    a, b, sa, sw = _unit_rows(rng, 8, 16)
    lay, live = H.Layout(8, 16, 64), [1] * 8
    zy = Z.records(a, b, sa, sw)
    other = H.j10r_blocks(16, 64, (8, 16), 0, 0)
    got, want = [], []
    for ch in range(n):
        last = ch == n - 1
        d = bind(Z.PouwH1LaneZyChunk2, TILE=_tile(H.PouwH1Tile5, 8, 16), CHUNKS=n, CHUNK=ch)
        names = [p for p, _t in d.signature()[0]]
        cases = []
        for ln, honest, blocks in ((0, J10R_LEAF_BLOCKS, J10R_LEAF_BLOCKS), (3, J10R_LEAF_BLOCKS, J10R_LEAF_BLOCKS),
                                   (3, J10R_LEAF_BLOCKS, other), (3, J10R_CARRY_BLOCKS, J10R_CARRY_BLOCKS)):
            rows = _lane_rows(lay, ln, a, b, sa, sw)
            cs = Z.lane_chunk_carries(*rows, SALT, J10R_AT, ROOTS, ln, (8, 16), n, (J10R_CALL, honest))
            cases.append((ln, rows, cs, cs[-1]["words"], blocks))
        if last:
            ln, rows, cs, words, blocks = cases[1]
            cases.append((ln, rows, cs, [words[0], words[1] ^ 1 << 20, *words[2:]], blocks))
        inputs = []
        for ln, (ra, rb, ras, rbs), cs, partial, blocks in cases:
            ca, cb = _chunk_rows(ra, n, ch), _chunk_rows(rb, n, ch)
            cin, cout = cs[ch - 1] if ch else None, None if last else cs[ch]
            args = (live if last else None, zy if last else None, partial if last else None, cin, cout)
            want.append(Z.lane_chunk_reference(ca, cb, ras, rbs, *args, SALT, J10R_AT, ROOTS, ln, (8, 16), n, ch,
                                               (J10R_CALL, blocks)))
            inputs.append(_j10r_lane_inputs(names, ca, cb, ras, rbs, *args, ln, blocks))
        out = evaluate_bits(d, *[np.array([row[p] for row in inputs], np.uint8) for p in range(len(names))])
        got += [int(o) for o in out.reshape(len(cases))]
    assert want == [1, 1, 0, 1] + [1, 1, 0, 1, 0]
    assert got == want
