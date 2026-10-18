"""`circuit.h1tile`: H1 + fx1 as a compile-pass tile unit under CP-W. Its lanes are the tile's cells once each, its noise is
seed_j's stream in counter mode and its fold keys σ_R's, its receipt is #2438's warp checkpoint (two words a set), a dead
tile's (all-zero A) is zero words, and on a compiled grid its gates are its word reference (`segment.call_words`, fx1's
fold), with the served outputs unchanged."""
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


def test_dead_tile_bits_are_the_words():
    """`PouwAnyByte_v2`, `PouwOrBit_v2` and `PouwMask32_v2` are their `_v1` words: any nonzero byte of 32, a bit's OR, and a
    word masked by a bit."""
    rng = random.Random(6)
    for x in ([0] * 32, [0] * 31 + [0x80], [rng.randrange(256) for _ in range(32)], [1] + [0] * 31):
        want = int(any(x))
        assert list(evaluate(H.PouwAnyByte, _bits(x))) == [want] and evaluate(H.PouwAnyByteWord, *x) == (want,)
    for a in (0, 1):
        for b in (0, 1):
            assert list(evaluate(H.PouwOrBit, [a], [b])) == [a | b] == list(evaluate(H.PouwOrBitWord, a, b))
    w = rng.getrandbits(32)
    for live in (0, 1):
        got = list(evaluate(H.PouwMask32, [(w >> i) & 1 for i in range(32)], [live]))
        assert _words(got) == [w * live] == list(evaluate(H.PouwMask32Word, w, live))


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
    dead = [[0] * 64 for _ in range(4)]
    zd, rd = H.reference(dead, b, SALT, 0, ROOTS)
    assert zd == [0] * 64 and rd == [0] * 8 and any(H.reference(dead, b, SALT, 0, ROOTS, gated=False)[1])
    one = [row[:] for row in dead]
    one[3][63] = 1
    assert H.reference(one, b, SALT, 0, ROOTS)[1] == H.reference(one, b, SALT, 0, ROOTS, gated=False)[1]


def test_live_salt_is_the_round_key_stream():
    """A live window's unit salt is round r's `window.round_key` stream's first 64 bytes, from the round's 64-byte coin and
    root; another round, root or coin gives another salt, and the unit's receipt follows it."""
    from verity.core.protocols.pouw import window as Wd
    from verity.core.protocols.sampled_proofs import challenger as Ch

    count = iter(range(1 << 20))
    ch = Wd.live_challenger(7, 2, entropy=lambda n: hashlib.shake_256(next(count).to_bytes(8, "big")).digest(n))
    rec = Ch.LiveRecord(ch, Wd.ROUND)
    roots = [hashlib.sha512(b"round 0").digest(), hashlib.sha512(b"round 1").digest()]
    o0, o1 = rec.record(roots[0]), rec.record(roots[1])
    assert rec.reason() is None and len(o0.coin) == 64
    s0 = H.live_salt(ch.commitment, o0.coin, 0, roots[0])
    assert len(s0) == H.SALT_BYTES == 64 and s0 == Wd.round_key(ch.commitment, o0.coin, 0, roots[0]).stream(0, 64)
    others = {H.live_salt(ch.commitment, o1.coin, 1, roots[1]), H.live_salt(ch.commitment, o0.coin, 1, roots[0]),
              H.live_salt(ch.commitment, o0.coin, 0, roots[1]), H.live_salt(ch.commitment, o1.coin, 0, roots[0])}
    assert s0 not in others and len(others) == 4
    rng = random.Random(5)
    a, b = _rows(rng, 4, 64, -127, 127), _rows(rng, 8, 64, -60, 60)
    assert H.reference(a, b, s0, 3, ROOTS)[1] != H.reference(a, b, H.live_salt(ch.commitment, o1.coin, 1, roots[1]), 3,
                                                             ROOTS)[1]


@compiles
@pytest.mark.slow
def test_h1_tile_is_its_reference():
    """A 4 × 16 grid at K = 64 compiles to one H1 tile unit of two lanes; on random rows its served outputs are the served
    Program's, and its receipt is `reference`'s (E2E-120's `call_words` and fx1 fold from σ_R's keys, then the lanes' XOR a
    set); on all-zero rows of A (a dead tile) its receipt is zero words. Slow: the two seeds' and 22 noise blocks' SHA-512
    in gates and 64 exact s8 dots on bits, twice; it is the only test of the unit's gates."""
    served = Program(bind(GRID, R=4, C=16, K=64))
    c = C.compile(served, H.SCHEME)
    (u,) = c.units
    assert u.tile.grid and (u.tile.r, u.tile.c) == (4, 16) and u.defn.base_id == "PouwH1Tile_v2"
    rng = random.Random(9)
    b = _rows(rng, 16, 64, -60, 60)
    for a in (_rows(rng, 4, 64, -127, 127), [[0] * 64 for _ in range(4)]):
        args = [[v for row in a for v in _bits(row)], [v for row in b for v in _bits(row)]]
        want = list(evaluate(served.fn, *args))
        got = list(evaluate(c.program.fn, *args, *c.inputs(SALT, [ROOTS])))
        S = c.served_leaves
        assert got[:S] == want
        z, rc = H.reference(a, b, SALT, 0, ROOTS)
        assert _words(want) == z
        assert _words(got[S:]) == rc and any(rc) == any(v for row in a for v in row)
