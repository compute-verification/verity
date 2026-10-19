"""`circuit.h1tile`: H1 + fx1 as a compile-pass tile unit under CP-W. Its lanes are the tile's cells once each, its noise is
seed_j's stream in counter mode and its fold keys σ_R's, its receipt is #2438's warp checkpoint (two words a set), and on a
compiled grid its gates are its word reference (`segment.call_words`, fx1's fold), with the served outputs unchanged."""
from __future__ import annotations

import hashlib
import random

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat, runs
from verity.ml.circuits.types import Array
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
COIN_RULES = C._ports is not None and hasattr(C._ports, "_coin_role") and "position" not in C._ports.KINDS
compiles = pytest.mark.skipif(COIN_RULES, reason="COIN-RULES refuses the salt's flow into PouwSeed: E_COIN_FLOW")
BIT = BL.BIT
I8 = Array(8, BIT)
SALT = hashlib.sha512(b"salt").digest()
ROOTS = (hashlib.sha512(b"A0").digest(), hashlib.sha512(b"B0").digest())


def _acc(K):
    return bind(BG.GemmAccumulatorS8, K=K, **BG.GEMM_SM120_S8)


def _grid_body(B, S, x, w):
    xs = Coll(Array(S.R * S.C, x.type.elem), concat([x[i].refs for i in range(S.R) for _j in range(S.C)]))
    ws = Coll(Array(S.R * S.C, w.type.elem), concat([w[j].refs for _i in range(S.R) for j in range(S.C)]))
    return B.batch(_acc(S.K), xs, ws, axes=(0, 0))


def _diag_body(B, S, x, w):
    return B.batch(_acc(S.K), x, w, axes=(0, 0))


def _def(name, ret, body):
    return CompositeDefinition(name, 1, ("R", "C", "K"),
                               lambda S: ((("x", Array(S.R, Array(S.K, I8))), ("w", Array(S.C, Array(S.K, I8)))), ret(S)),
                               body, conformance="protocol-defined", register=False)


GRID = _def("H1TestGrid", lambda S: Array(S.R * S.C, Array(32, BIT)), _grid_body)
DIAG = _def("H1TestDiag", lambda S: Array(S.R, Array(32, BIT)), _diag_body)


def _bits(codes):
    return [(v >> i) & 1 for v in codes for i in range(8)]


def _words(bits):
    return [sum(b << i for i, b in enumerate(bits[t:t + 32])) for t in range(0, len(bits), 32)]


def _rows(rng, n, k, lo, hi):
    return [[rng.randrange(lo, hi + 1) & 0xFF for _ in range(k)] for _ in range(n)]


def test_registered_at_a_warp():
    s = C._scheme(H.SCHEME)
    assert s is C.SCHEMES[H.SCHEME] and s.tile == (32, 32) and set(s.units) == set(H.KINDS)
    assert "h1tile" in C.BUILTIN


@pytest.mark.parametrize("r,c", [(4, 8), (8, 8), (4, 16), (32, 32)])
def test_lanes_are_the_tile_once(r, c):
    """Each lane's 32 entries, at their row and column slots, are the tile's cells once each; at 32 × 32 lane 4 g + q is
    cgemm.cu's thread (rows g + 8 s, columns 2 q + 8 (s >> 1) + (s & 1))."""
    lay = H.Layout(r, c, 64)
    cells = [(lay.rows(ln)[H.SG.row_slot(e)], lay.cols(ln)[H.SG.col_slot(e)]) for ln in range(lay.lanes) for e in range(32)]
    assert sorted(cells) == [(i, j) for i in range(r) for j in range(c)]
    if (r, c) == (32, 32):
        for g in range(8):
            for q in range(4):
                assert lay.rows(4 * g + q) == tuple(g + 8 * s for s in range(4))
                assert lay.cols(4 * g + q) == tuple(2 * q + 8 * (s >> 1) + (s & 1) for s in range(8))


def test_noise_is_counter_mode():
    """Every row's and column's noise is its own blocks of seed_j's stream, and every lane's fold keys a block of σ_R's
    (decision 121), so a kernel derives any of them alone; at a warp and K = 4096 the receipt is #2438's 8,320 bits a warp a
    call (2 (KS + m) words, KS = 128, m = 2)."""
    lay = H.Layout(32, 32, 4096)
    assert (lay.ks, lay.m, lay.rb, lay.cb, lay.lanes, lay.blocks) == (128, 2, 9, 9, 32, 576)
    assert 2 * lay.sets * 32 == 8320
    firsts = [lay.row(i) for i in range(32)] + [lay.col(j) for j in range(32)]
    spans = [lay.rb] * 32 + [lay.cb] * 32
    assert sorted(b for f, n in zip(firsts, spans) for b in range(f, f + n)) == list(range(lay.blocks))
    assert [lay.lane(ln) for ln in range(32)] == list(range(32))
    small = H.Layout(4, 8, 64)
    seed_j, sigma_r = SD.seed(SALT, 3, *ROOTS), SD.receipt_seed(SALT, 3, *ROOTS)
    rows, cols = H.noise(seed_j, small)
    block = SD.leaves(SD.noise_block(seed_j, small.row(2)), True)
    assert rows[2] == (block[:64], [block[64:96]])
    col = SD.noise_block(seed_j, small.col(5))
    assert cols[5][0] == SD.leaves(col, True)[:64]
    assert cols[5][1] == [H.u129(x) for h in SD.halves(col)[2:10] for x in (h & 0xFF, h >> 8 & 0xFF, h >> 16 & 0xFF, h >> 24)]
    assert H.fold_keys(sigma_r, small)[0] == SD.halves(SD.noise_block(sigma_r, small.lane(0)))


def test_fold_chains_start_from_sigma_r():
    """Decision 121: the unit's one seed call is `PouwUnitSeeds` (seed_j, σ_R); each lane's fold chain starts from a block
    of σ_R's stream, and every other noise block reads seed_j, so no fold key is a function of the noise seed."""
    grid = C.Tile(4, 16, tuple((i, j) for i in range(4) for j in range(16)))
    body = bind(H.PouwH1Tile, EXACT=_acc(64), R=4, C=16, PAIRS=grid.pairs).body
    defn = [getattr(nd.fn, "defn", nd.fn) for nd in body.nodes]
    (us,) = [k for k, d in enumerate(defn) if d in SD.SEEDS]
    assert defn[us] is SD.PouwUnitSeedsBits
    D = SD.DIGEST_BITS.leaves
    seed_j, sigma_r = [("n", us, 0, D)], [("n", us, D, 2 * D)]
    blocks = {k: runs(body.nodes[k].args[0].refs) for k, d in enumerate(defn) if d is SD.PouwNoiseBlockBits}
    lay = H.Layout(4, 16, 64)
    assert sorted(map(str, blocks.values())) == sorted(map(str, [seed_j] * lay.blocks + [sigma_r] * lay.lanes))
    absorbs = {k for k, d in enumerate(defn) if d is H.SG.PouwFx1Absorb}
    starts = [runs(body.nodes[k].args[0].refs) for k in absorbs]
    starts = [r for r in starts if not any(s == "n" and i in absorbs for s, i, _lo, _hi in r)]
    assert len(starts) == lay.lanes == len({r[0][1] for r in starts})
    assert all(len(r) == 1 and r[0][0] == "n" and r[0][2:] == (0, H.KEY_BITS) and blocks.get(r[0][1]) == sigma_r
               for r in starts), starts


def test_u129_on_bits_is_the_word():
    """`PouwU129_v2` is `PouwU129_v1` on every byte, and U is in ±64."""
    for x in range(256):
        want = H.u129(x)
        assert -64 <= (want - 256 if want & 0x80 else want) <= 64
        assert list(evaluate(H.PouwU129, [(x >> i) & 1 for i in range(8)])) == [(want >> i) & 1 for i in range(8)]
        assert evaluate(H.PouwU129Word, x) == (want,)


def test_unit_signature():
    grid = C.Tile(4, 8, tuple((i, j) for i in range(4) for j in range(8)))
    params, ret = bind(H.PouwH1Tile, EXACT=_acc(64), R=4, C=8, PAIRS=grid.pairs).signature()
    assert [n for n, _t in params] == ["a", "b", "salt", "position", "roots"]
    assert ret.fields[0][1] == Array(32, Array(32, BIT)) and ret.fields[1][1] == Array(2 * 4, Array(32, BIT))


@pytest.mark.parametrize("program,why", [
    (lambda: Program(bind(G.GemmAccumulatorS8, K=64, **G.GEMM_SM120_S8)), "GemmAccumulatorS8_v1"),
    (lambda: Program(bind(GRID, R=2, C=8, K=64)), "4 | r"),
    (lambda: Program(bind(GRID, R=4, C=8, K=32)), "KS = K / 32 >= 2"),
    (lambda: Program(bind(DIAG, R=8, C=8, K=64)), "full r × c grid"),
], ids=["words", "rows", "k", "diagonal"])
def test_refused(program, why):
    """A word-level kind, a tile of part of a lane, one segment, or a site that isn't a grid (a population of coordinates)."""
    with pytest.raises(C.CompileError) as e:
        C.compile(program(), H.SCHEME)
    assert e.value.code == "scheme" and why in str(e.value), str(e.value)


def test_reference_follows_the_seed():
    """On words: z is the exact product; the receipt moves with the salt, the position and the roots (each seeds its own
    noise), and is two words a set."""
    rng = random.Random(4)
    a, b = _rows(rng, 4, 64, -127, 127), _rows(rng, 16, 64, -60, 60)
    z, rc = H.reference(a, b, SALT, 0, ROOTS)
    assert len(rc) == 2 * H.Layout(4, 16, 64).sets == 8
    assert z == [H.SG.seg_dot(0, a[i], b[j]) for i in range(4) for j in range(16)]
    for other in (H.reference(a, b, SALT, 1, ROOTS), H.reference(a, b, hashlib.sha512(b"x").digest(), 0, ROOTS),
                  H.reference(a, b, SALT, 0, (ROOTS[1], ROOTS[0]))):
        assert other[0] == z and all(x != y for x, y in zip(other[1], rc))


@compiles
@pytest.mark.slow
def test_h1_tile_is_its_reference():
    """A 4 × 16 grid at K = 64 compiles to one H1 tile unit of two lanes; on random rows its served outputs are the served
    Program's, and its receipt is `reference`'s (E2E-120's `call_words` and fx1 fold from σ_R's keys, then the lanes' XOR a
    set). Slow: the two seeds' and 22 noise blocks' SHA-512 in gates and 64 exact s8 dots on bits; it is the only test of
    the unit's gates."""
    served = Program(bind(GRID, R=4, C=16, K=64))
    c = C.compile(served, H.SCHEME)
    (u,) = c.units
    assert u.tile.grid and (u.tile.r, u.tile.c) == (4, 16) and u.defn.base_id == "PouwH1Tile_v1"
    rng = random.Random(9)
    a, b = _rows(rng, 4, 64, -127, 127), _rows(rng, 16, 64, -60, 60)
    args = [[v for row in a for v in _bits(row)], [v for row in b for v in _bits(row)]]
    want = list(evaluate(served.fn, *args))
    got = list(evaluate(c.program.fn, *args, *c.inputs(SALT, [ROOTS])))
    S = c.served_leaves
    assert got[:S] == want
    z, rc = H.reference(a, b, SALT, 0, ROOTS)
    assert _words(want) == z
    assert _words(got[S:]) == rc


# -- v3, the one stream ------------------------------------------------------------------------------------------------------

STREAM = H.STREAM
GRID48 = tuple((i, j) for i in range(4) for j in range(8))


def _s32(v):
    return v - (1 << 32) if v >> 31 & 1 else v


def _sx8(v):
    return v - 256 if v & 0x80 else v


def _served(a, b):
    """A Bᵀ, the served linear's product (vLLM's `_int_mm`), as integers."""
    return [sum(_sx8(x) * _sx8(w) for x, w in zip(ai, bj)) for ai in a for bj in b]


def _noised(a, b, position=0):
    """X Ŵᵀ, the MMAs' product of the noised operands: X = A + b mod 2^8 (the int8 the MMA reads) and Ŵ = B + c."""
    lay = H.Layout(len(a), len(b), len(a[0]))
    rows, cols = H.noise(SD.seed(SALT, position, *ROOTS), lay)
    X = [[_sx8((v + rows[i][0][t]) & 0xFF) for t, v in enumerate(a[i])] for i in range(lay.r)]
    W = [[_sx8(v) + cols[j][0][t] for t, v in enumerate(b[j])] for j in range(lay.c)]
    return [sum(x * w for x, w in zip(X[i], W[j])) for i in range(lay.r) for j in range(lay.c)]


def _full(rng, n, k):
    """Full-range int8 rows with A = 127 every 7th entry, so X = A + b wraps (xa) wherever the dither bit is set."""
    rows = _rows(rng, n, k, -128, 127)
    for row in rows:
        row[3::7] = [127] * len(row[3::7])
    return rows


def _cells(a, b, position=0):
    """Each cell's folded O' and D words, and the peel's and the wrap's inputs: X, wb, (b, xa) by row, (B, c) by column."""
    lay = H.Layout(len(a), len(b), len(a[0]))
    rows, cols = H.noise(SD.seed(SALT, position, *ROOTS), lay)
    words = {}
    for ln in range(lay.lanes):
        sets = H.lane_sets(a, b, rows, cols, lay, ln)
        for e in range(32):
            words[(lay.rows(ln)[H.SG.row_slot(e)], lay.cols(ln)[H.SG.col_slot(e)])] = (
                sets[lay.ks][e], [d[e] for d in sets[lay.ks + 1:]])
    X = [[(a[i][t] + rows[i][0][t]) & 0xFF for t in range(lay.k)] for i in range(lay.r)]
    wbs = [H.col_wrap(b[j], cols[j]) for j in range(lay.c)]
    pis = [(rows[i][0][:lay.k], H.row_wrap(a[i], rows[i])) for i in range(lay.r)]
    pjs = [(b[j], cols[j][0][:lay.k]) for j in range(lay.c)]
    return words, X, wbs, pis, pjs


def _z(words, X, wbs, pis, pjs):
    return {(i, j): H.stream_cell(*words[(i, j)], X[i], wbs[j], pis[i], pjs[j]) for i, j in words}


def test_v3_is_registered_beside_v1():
    """v3 is a scheme and a Definition of its own; v1's scheme and Definition stay as they were."""
    v1, v3 = C._scheme(H.SCHEME), C._scheme(H.SCHEME_V3)
    assert v1.stream == () and v3.stream == STREAM and v3.tile == v1.tile == (32, 32) and set(v3.units) == set(H.KINDS)
    assert (H.PouwH1Tile.id, H.PouwH1Tile3.id) == ("PouwH1Tile_v1", "PouwH1Tile_v3")
    assert H.PouwH1Tile3.signature is H.PouwH1Tile.signature


def _wraps(r, b, position=0):
    """How many entries of B wrap in W' = B + c + U, each way, in an r-row tile."""
    lay = H.Layout(r, len(b), len(b[0]))
    _rows_, cols = H.noise(SD.seed(SALT, position, *ROOTS), lay)
    wb = [w for j in range(lay.c) for w in H.col_wrap(b[j], cols[j])]
    return wb.count(1), wb.count(2)


@pytest.mark.parametrize("r,c,k", [(32, 32, 64), (4, 8, 4096), (4, 8, 14336)])
def test_one_stream_is_the_served_product(r, c, k):
    """v3's z, O' - sum_d 128^d D_d of each cell's folded words plus the wrap correction less the dither peel, is A Bᵀ
    exactly (as s32), v1's z, on random full-range int8 A and B (m = 2 at K <= 4096, m = 3 at 14336), where W' wraps both
    ways and X = A + b wraps at A = 127; and its receipt is v1's: the same words, folded the same way."""
    rng = random.Random(k)
    a, b = _full(rng, r, k), _rows(rng, c, k, -128, 127)
    z3, rc3 = H.reference(a, b, SALT, 0, ROOTS, stream=True)
    z1, rc1 = H.reference(a, b, SALT, 0, ROOTS)
    assert H.Layout(r, c, k).m == (3 if k > 4096 else 2)
    assert min(_wraps(r, b)) > 0
    lay = H.Layout(r, c, k)
    rows = H.noise(SD.seed(SALT, 0, *ROOTS), lay)[0]
    assert all(sum(H.row_wrap(a[i], rows[i])) > 0 for i in range(r))
    assert [_s32(v) for v in z3] == _served(a, b) != _noised(a, b)
    assert rc3 == rc1 and z3 == z1


def test_the_wrap_correction_is_needed():
    """At B's edges (127, -128, ±110) z is A Bᵀ; without the wrap correction (wb taken as 0) some cell's is not, and where
    B + c + U never wraps ([-64, 62] under U in ±64) the correction is zero."""
    k = 64
    a = [[v & 0xFF for v in row] for row in ([[127] * k, [-128] * k, [126] * k, [-127] * k])]
    for edge, wraps in (([127, -128], True), ([110, -110], True), ([62, -64], False)):
        b = [[v & 0xFF for v in row] for row in ([[edge[0]] * k, [edge[1]] * k] * 4)]
        assert [_s32(v) for v in H.reference(a, b, SALT, 0, ROOTS, stream=True)[0]] == _served(a, b)
        words, X, wbs, pis, pjs = _cells(a, b)
        bare = _z(words, X, [[0] * k for _ in wbs], pis, pjs)
        plain = [_s32(bare[(i, j)]) for i in range(4) for j in range(8)]
        assert (plain != _served(a, b)) == wraps and (sum(_wraps(4, b)) > 0) == wraps


def test_the_dither_peel_is_needed():
    """Without the peel z is X Ŵᵀ, the noised operands' product, not A Bᵀ; with the peel but no X wrap term (xa taken as
    0) exactly the rows where A = 127 meets a set dither bit are off, cell (i, j) by -256 sum_t xa(i, t) B(j, t)."""
    rng = random.Random(14)
    k = 4096
    a, b = _full(rng, 4, k), _rows(rng, 8, k, -128, 127)
    a[0] = [rng.randrange(-126, 127) & 0xFF for _ in range(k)]
    words, X, wbs, pis, pjs = _cells(a, b)
    lay = H.Layout(4, 8, k)
    unpeeled = {}
    for (i, j), (op, d) in words.items():
        v = H.one_stream(op, d)
        for s in range(0, k, 32):
            v = H.wrap_seg(v, X[i][s:s + 32], wbs[j][s:s + 32])
        unpeeled[(i, j)] = _s32(v)
    assert [unpeeled[(i, j)] for i in range(4) for j in range(8)] == _noised(a, b) != _served(a, b)
    no_xa = _z(words, X, wbs, [(bd, [0] * k) for bd, _xa in pis], pjs)
    ab = dict(zip([(i, j) for i in range(4) for j in range(8)], _served(a, b)))
    assert sum(pis[0][1]) == 0 and all(sum(pis[i][1]) > 0 for i in range(1, lay.r))
    for (i, j), v in no_xa.items():
        off = -256 * sum(_sx8(w) for w, x in zip(b[j], pis[i][1]) if x)
        assert _s32(v) == ab[(i, j)] + off and (off != 0) == (i != 0)


def test_wrap_bits_on_bits_is_the_word():
    """`PouwWrapBits_v2` is its word, [B + c + U > 127] and [B + c + U < -128], on every (B, U) at the edges and at random."""
    rng = random.Random(11)
    edge = [0, 1, 62, 63, 64, 126, 127, 128, 129, 191, 192, 193, 254, 255]
    for b, c, u in [(b, c, u) for b in edge for c in (0, 1) for u in edge] + [
            (rng.randrange(256), rng.randrange(2), rng.randrange(256)) for _ in range(512)]:
        s = _sx8(b) + c + _sx8(u)
        want = 1 if s > 127 else 2 if s < -128 else 0
        got = evaluate(H.PouwWrapBits, [(b >> i) & 1 for i in range(8)], [c], [(u >> i) & 1 for i in range(8)])
        assert list(got) == [want & 1, want >> 1] and list(evaluate(H.PouwWrapBits.word, b, c, u)) == [want]


def test_x_wrap_on_bits_is_the_word():
    """`PouwXWrap_v2` is its word, [A = 127] AND b, on every A and b."""
    for a in range(256):
        for b in (0, 1):
            want = int(a == 0x7F and b == 1)
            assert list(evaluate(H.PouwXWrap, [(a >> i) & 1 for i in range(8)], [b])) == [want]
            assert list(evaluate(H.PouwXWrap.word, a, b)) == [want]


def _mask(bits):
    return sum(v << e for e, v in enumerate(bits))


def test_peel_seg_on_bits_is_the_word():
    """`PouwPeelSeg_v2` is its word on random and edge operands: acc - sum_e (c_e x_e + bd_e w_e - 256 xa_e w_e)."""
    rng = random.Random(15)
    edge = [0, 0x7F, 0x80, 0xFF]
    for _ in range(96):
        acc = rng.choice([0, H.M32, 1 << 31, rng.getrandbits(32)])
        x, w = ([rng.choice(edge + [rng.randrange(256)]) for _ in range(32)] for _ in range(2))
        bd, xa, c = ([rng.randrange(2) for _ in range(32)] for _ in range(3))
        got = evaluate(H.PouwPeelSeg, [(acc >> i) & 1 for i in range(32)], [(v >> i) & 1 for v in x for i in range(8)],
                       bd, xa, [(v >> i) & 1 for v in w for i in range(8)], c)
        want = H.peel_seg(acc, x, bd, xa, w, c)
        assert _words(list(got)) == [want] == list(evaluate(H.PouwPeelSeg.word, acc, *x, *map(_mask, (bd, xa)), *w, _mask(c)))
        assert want == (acc - sum(_sx8(xe) * ce + _sx8(we) * (be - 256 * ae)
                                  for xe, be, ae, we, ce in zip(x, bd, xa, w, c))) & H.M32


def test_wrap_seg_on_bits_is_the_word():
    """`PouwWrapSeg_v2` is its word on random and edge operands, both wrap bits set included (0: wb⁺ - wb⁻)."""
    rng = random.Random(12)
    for _ in range(96):
        acc = rng.choice([0, H.M32, 1 << 31, rng.getrandbits(32)])
        x = [rng.choice([0, 0x7F, 0x80, 0xFF, rng.randrange(256)]) for _ in range(32)]
        wb = [rng.randrange(4) for _ in range(32)]
        got = evaluate(H.PouwWrapSeg, [(acc >> i) & 1 for i in range(32)], [(v >> i) & 1 for v in x for i in range(8)],
                       [(w >> i) & 1 for w in wb for i in range(2)])
        want = H.wrap_seg(acc, x, wb)
        assert _words(list(got)) == [want] == list(evaluate(H.PouwWrapSeg.word, acc, *x, *wb))
        assert want == (acc + 256 * sum(_sx8(v) * ((w & 1) - (w >> 1)) for v, w in zip(x, wb))) & H.M32


def test_a_flipped_wrap_bit_moves_only_its_z():
    """Flipping one wrap bit wb(j, t), either one, moves exactly the cells it enters: each z(i, j) of column j by
    ±256 X(i, t) (+ for wb⁺ set or wb⁻ cleared), and no other column's z (the receipt never reads wb:
    `test_the_receipt_never_reads_wb`). Kernels' gate mirrors this fault."""
    rng = random.Random(13)
    k = 4096
    a, b = _full(rng, 4, k), _rows(rng, 8, k, -128, 127)
    words, X, wbs, pis, pjs = _cells(a, b)

    def z(w):
        return _z(words, X, w, pis, pjs)

    base = z(wbs)
    assert [base[(i, j)] for i in range(4) for j in range(8)] == H.reference(a, b, SALT, 0, ROOTS, stream=True)[0]
    flips = 0
    for _ in range(24):
        j, t, bit = rng.randrange(8), rng.randrange(k), rng.choice([1, 2])
        flipped = [list(w) for w in wbs]
        flipped[j][t] ^= bit
        sign = (1 if bit == 1 else -1) * (1 if flipped[j][t] & bit else -1)
        moved = z(flipped)
        assert moved == {(i, jj): (v + 256 * sign * _sx8(X[i][t])) & H.M32 if jj == j else v
                         for (i, jj), v in base.items()}
        flips += sum(moved[(i, j)] != base[(i, j)] for i in range(4))
    assert flips >= 24 * 4 - 4


def test_a_flipped_dither_bit_moves_only_its_z():
    """Flipping one dither bit as the peel reads it moves exactly the cells it enters: a column's coin bit c(j, t) each
    z(i, j) of column j by -/+ X(i, t) (set/cleared), a row's dither bit b(i, t) each z(i, j) of row i by -/+ B(j, t), or by
    ±255 B(j, t) where A(i, t) = 127 (its X wrap bit flips with it), and no other cell. Kernels' gate mirrors this fault."""
    rng = random.Random(16)
    k = 4096
    a, b = _full(rng, 4, k), _rows(rng, 8, k, -128, 127)
    words, X, wbs, pis, pjs = _cells(a, b)
    base = _z(words, X, wbs, pis, pjs)
    assert [base[(i, j)] for i in range(4) for j in range(8)] == H.reference(a, b, SALT, 0, ROOTS, stream=True)[0]
    moves = 0
    for n in range(32):
        if n % 2:
            j, t = rng.randrange(8), rng.randrange(k)
            c = list(pjs[j][1])
            c[t] ^= 1
            flipped = list(pjs)
            flipped[j] = (pjs[j][0], c)
            moved = _z(words, X, wbs, pis, flipped)
            step = -1 if c[t] else 1
            want = {(i, jj): (v + step * _sx8(X[i][t])) & H.M32 if jj == j else v for (i, jj), v in base.items()}
        else:
            i, t = rng.randrange(4), 3 + 7 * rng.randrange(k // 7) if n % 4 == 0 else rng.randrange(k)
            bd = list(pis[i][0])
            bd[t] ^= 1
            xa = H.row_wrap(a[i], (bd,))
            flipped = list(pis)
            flipped[i] = (bd, xa)
            moved = _z(words, X, wbs, flipped, pjs)
            step = (-1 if bd[t] else 1) * (-255 if a[i][t] == 127 else 1)
            want = {(ii, j): (v + step * _sx8(b[j][t])) & H.M32 if ii == i else v for (ii, j), v in base.items()}
        assert moved == want
        moves += sum(moved[p] != base[p] for p in base)
    assert moves >= 16 * 4 + 16 * 8 - 8


def test_the_receipt_never_reads_wb():
    """In v3's body no wrap bit, wrap correction, X wrap bit or dither peel is in the receipt's cone: the receipt is v1's
    sets alone, and they reach only z."""
    v3 = bind(H.PouwH1Tile3, EXACT=_acc(64), R=4, C=8, PAIRS=GRID48)
    body = v3.body
    lz = v3.signature()[1].fields[0][1].leaves
    cone, todo = set(), [i for s, i, _lo, _hi in runs(body.ret.slice(lz, body.ret.count)) if s == "n"]
    while todo:
        k = todo.pop()
        if k not in cone:
            cone.add(k)
            todo += [i for a in body.nodes[k].args for s, i, _lo, _hi in runs(a.refs) if s == "n"]
    fns = {getattr(body.nodes[k].fn, "defn", body.nodes[k].fn) for k in cone}
    made = {getattr(nd.fn, "defn", nd.fn) for nd in body.nodes}
    pieces = {H.PouwWrapBits, H.PouwWrapSeg, H.PouwXWrap, H.PouwPeelSeg}
    assert pieces <= made and not fns & pieces


def test_a_tampered_word_moves_z():
    """z is a function of the words the receipt folds: changing a cell's O' by any δ, or its D_d by a δ inside D's range
    (|D_d| <= 32 · 128 · 64 = 2^18), changes the cell's z; only a D_d change by a multiple of 2^(32 - 7 d), far outside that
    range, leaves it."""
    rng = random.Random(3)
    k = 14336
    a, b = _rows(rng, 4, k, -128, 127), _rows(rng, 8, k, -128, 127)
    lay = H.Layout(4, 8, k)
    rows, cols = H.noise(SD.seed(SALT, 0, *ROOTS), lay)
    sets = H.lane_sets(a, b, rows, cols, lay, 0)
    ks = lay.ks
    for e in range(32):
        op, ds = sets[ks][e], [d[e] for d in sets[ks + 1:]]
        assert all(abs(_s32(d)) <= 1 << 18 for d in ds)
        z = H.one_stream(op, ds)
        for _ in range(4):
            delta = rng.choice([-1, 1]) * rng.randrange(1, 1 << 19)
            assert H.one_stream(op + delta, ds) != z
            for t in range(len(ds)):
                assert H.one_stream(op, [d + delta if u == t else d for u, d in enumerate(ds)]) != z
        assert H.one_stream(op, [ds[0] + (1 << 25), *ds[1:]]) == z


def test_one_stream_on_bits_is_the_word():
    """`PouwOneStream2_v2` and `PouwOneStream3_v2` are their words on random and edge operands."""
    rng = random.Random(5)
    edge = [0, 1, H.M32, 1 << 31, (1 << 25) - 1]
    for d, n in ((H.PouwOneStream2, 2), (H.PouwOneStream3, 3)):
        for _ in range(64):
            ws = [rng.choice(edge) if rng.random() < 0.3 else rng.getrandbits(32) for _ in range(n)]
            got = evaluate(d, *[[(w >> i) & 1 for i in range(32)] for w in ws])
            assert _words(list(got)) == [H.one_stream(ws[0], ws[1:])] == list(evaluate(d.word, *ws))


def _bad_body(B, S, a, b, salt, position, roots):
    receipt, words = H._sets(B, S, a, b, salt, position, roots)[:2]
    z = [B.call(H.PouwOneStream2, B.call(S.EXACT, a[i], b[j]), words[(i, j)][1]).refs for i, j in S.PAIRS]
    return Coll(H._signature(S)[1], concat(z + receipt))


BAD = CompositeDefinition("H1TestStreamBeside", 1, ("EXACT", "R", "C", "PAIRS"), H._signature, _bad_body,
                          conformance="protocol-defined", register=False)


def test_the_stream_check():
    """The unit check takes v3 under a one-stream scheme and refuses it under a plain one (its z reads the seed), refuses v1
    under a one-stream scheme (its z is a product of its own), refuses a unit recombining an exact product it doesn't fold,
    and refuses v3 under a stream without the dither peel (z is then `PouwPeelSeg`'s output, not a recombination) or
    without the wrap correction too."""
    v3 = bind(H.PouwH1Tile3, EXACT=_acc(64), R=4, C=8, PAIRS=GRID48)
    C._check_unit(v3, True, STREAM)
    for unit, stream, code, why in ((v3, (), "noise-in-z", "z is the exact product"),
                                    (bind(H.PouwH1Tile, EXACT=_acc(64), R=4, C=8, PAIRS=GRID48), STREAM, "stream",
                                     "not a recombination"),
                                    (bind(BAD, EXACT=_acc(64), R=4, C=8, PAIRS=GRID48), STREAM, "stream",
                                     "which the receipt doesn't fold"),
                                    (v3, (*H.ONE_STREAM.values(), H.PouwWrapSeg, H.PouwWrapBits), "stream",
                                     "not a recombination"),
                                    (v3, tuple(H.ONE_STREAM.values()), "stream", "not a recombination")):
        with pytest.raises(C.CompileError) as e:
            C._check_unit(unit, True, stream)
        assert e.value.code == code and why in str(e.value), str(e.value)


@compiles
@pytest.mark.slow
def test_h1_tile_v3_is_its_reference():
    """A 4 × 16 grid at K = 64 compiles under `SCHEME_V3` to one `PouwH1Tile_v3`; on random full-range rows (W' and X
    wrap) its served outputs are the served Program's, A Bᵀ, and `reference(stream=True)`'s z, and its receipt v1's. Slow
    (about 200 s): the only test of v3's gates, wrap correction and dither peel included, as `test_h1_tile_is_its_reference`
    is v1's."""
    served = Program(bind(GRID, R=4, C=16, K=64))
    c = C.compile(served, H.SCHEME_V3)
    (u,) = c.units
    assert u.defn.base_id == "PouwH1Tile_v3"
    rng = random.Random(9)
    a, b = _full(rng, 4, 64), _rows(rng, 16, 64, -128, 127)
    args = [[v for row in a for v in _bits(row)], [v for row in b for v in _bits(row)]]
    got = list(evaluate(c.program.fn, *args, *c.inputs(SALT, [ROOTS])))
    S = c.served_leaves
    z, rc = H.reference(a, b, SALT, 0, ROOTS, stream=True)
    assert _words(got[:S]) == z == _words(list(evaluate(served.fn, *args))) and [_s32(v) for v in z] == _served(a, b)
    assert _words(got[S:]) == rc == H.reference(a, b, SALT, 0, ROOTS)[1]
    assert sum(_wraps(4, b)) > 0
