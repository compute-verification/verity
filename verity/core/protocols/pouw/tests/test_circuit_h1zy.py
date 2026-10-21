"""`circuit.h1zy`: the dequant is the served epilogue bit for bit, RN_bf16(RN32(RN32(RN32(z) s_a) s_w)) with the activation
scale first and no bias (`int8_exact.dequantize`, scaled_mm_zy's y), on words and on bits, where PoUW's `reference.dequant`
(scales first, a bias add) is not; `PouwZyCheck` holds a unit's record to its z and that dequant, cell by cell; and
`PouwH1TileZy` is the H1 tile and that check over the rows' codes and scales, returning the receipt and one bit.
`benchmarks/pouw/tests/test_h1_prove.py` and h1_prove's smoke run the unit end to end."""
from __future__ import annotations

import random
import struct

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import runs

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import h1zy as Z
from verity.core.protocols.pouw.circuit import reference as REF

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
