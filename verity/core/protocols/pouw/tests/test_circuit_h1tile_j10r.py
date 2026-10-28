"""`circuit.h1tile`'s J10-R units, `PouwH1Tile_v5` and `PouwH1Tile_v6`
(J10-R): one seed a call from its first position and its registered roots, a unit's rows' and columns'
noise the call-scale stream's at public block counters, its fold keys σ_R's at its own position; and the SHA-512
compressions that saves a unit against the note's formula."""
from __future__ import annotations

import hashlib
import random
from fractions import Fraction

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.refs import runs
from verity.ml.circuits.types import Array
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment) are not on this tree")
SALT = hashlib.sha512(b"salt").digest()
ROOTS = (hashlib.sha512(b"a/64@3").digest(), hashlib.sha512(b"b/64").digest())
GRID48 = tuple((i, j) for i in range(4) for j in range(8))


def _v(d):
    return bind(d, EXACT=bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8), R=4, C=8, PAIRS=GRID48)


def _defns(body):
    return [getattr(nd.fn, "defn", nd.fn) for nd in body.nodes]


def _bits(codes):
    return [(v >> i) & 1 for v in codes for i in range(8)]


def _words(bits):
    return [sum(b << i for i, b in enumerate(bits[t:t + 32])) for t in range(0, len(bits), 32)]


def _sx8(v):
    return v - 256 if v & 0x80 else v


def _s32(v):
    return v - (1 << 32) if v >> 31 & 1 else v


def _rows(rng, n, k):
    rows = [[rng.randrange(256) for _ in range(k)] for _ in range(n)]
    for row in rows:
        row[3::7] = [127] * len(row[3::7])
    return rows


@pytest.mark.parametrize("m,n,k,tile", [(8, 32, 64, (4, 8)), (12, 16, 128, (4, 8)), (64, 96, 4096, (32, 32))])
def test_unit_slices_are_the_call_scale_noise(m, n, k, tile):
    """Every unit (I, J) of an M × N call reads, at `j10r_blocks`' counters, exactly rows R I .. R I + R - 1 and columns
    C J .. C J + C - 1 of the call-scale `noise(seed_c, Layout(M, N, K))`: dither bits, digit coins, coin bits and U."""
    r, c = tile
    seed_c = SD.call_seed(SALT, 40, *ROOTS)
    rows, cols = H.noise(seed_c, H.Layout(m, n, k))
    lay = H.Layout(r, c, k)
    for I in range(m // r):
        for J in range(n // c):
            ur, uc = H.j10r_noise(seed_c, lay, H.j10r_blocks(m, k, tile, I, J))
            assert ur == rows[r * I:r * I + r] and uc == cols[c * J:c * J + c], (I, J)


def test_the_call_streams_are_shared_once_each():
    """The call's units read each of its row streams once a column tile and each column stream once a row tile, and every
    block they read is the call-scale Layout's: rows' blocks [0, M RB), columns' [M RB, M RB + N CB)."""
    m, n, k, tile = 8, 32, 64, (4, 8)
    lay, call = H.Layout(*tile, k), H.Layout(m, n, k)
    reads = [b for I in range(2) for J in range(4) for b in H.j10r_unit_blocks(lay, H.j10r_blocks(m, k, tile, I, J))]
    assert sorted(set(reads)) == list(range(call.blocks))
    assert sorted(reads) == sorted([*range(m * call.rb)] * (n // tile[1]) + [*range(m * call.rb, call.blocks)] * (m // tile[0]))


@pytest.mark.parametrize("args", [(30, 64, (4, 8), 0, 0), (8, 64, (4, 8), 2, 0), (8, 64, (4, 8), 0, -1),
                                  (2, 64, (4, 8), 0, 0)])
def test_blocks_of_no_unit_are_refused(args):
    with pytest.raises(ValueError, match="no R × C"):
        H.j10r_blocks(*args)


def _note(m, n, k, tile=(32, 32)):
    """The note's per-unit compressions, 9216 (1/M + 1/N) + 32 + 4/units at a warp and RB = CB = 9, in general
    R C (RB/N + CB/M) + lanes + 4/units: each call row's and column's blocks once, the lanes' key blocks, the seeds once."""
    lay = H.Layout(*tile, k)
    units = Fraction(m * n, tile[0] * tile[1])
    return tile[0] * tile[1] * (Fraction(lay.rb, n) + Fraction(lay.cb, m)) + lay.lanes + 4 / units, units


@pytest.mark.parametrize("m,n,k", [(32, 4096, 4096), (256, 4096, 4096), (256, 4096, 14336), (128, 14336, 4096)],
                         ids=["decode", "prefill", "prefill-k14336", "prefill-n14336"])
def test_compressions_a_unit_against_the_note(m, n, k):
    """`j10r_compressions` counts the distinct seed_c blocks the call's units read: the note's formula plus σ_R at each
    unit's own position (2 - 2/units), since the fold keys stay the per-unit form's. Today a unit takes 612 at K = 4096 (576
    blocks, 32 key blocks, 4 for its two seeds); at decode J10-R takes 324.27 against the note's 322.28."""
    note, units = _note(m, n, k)
    got = H.j10r_compressions(m, n, k)
    assert got == note + 2 - 2 / units
    lay = H.Layout(32, 32, k)
    assert H.unit_compressions(k) == lay.blocks + lay.lanes + 4
    if k == 4096:
        assert H.unit_compressions(k) == 612 and (lay.rb, lay.cb) == (9, 9)
        assert note == 9216 * (Fraction(1, m) + Fraction(1, n)) + 32 + 4 / units
    if (m, n, k) == (32, 4096, 4096):
        assert round(float(got), 2) == 324.27 and round(float(note), 2) == 322.28


@pytest.mark.slow
@pytest.mark.parametrize("d", ["PouwH1Tile", "PouwH1Tile5"])
def test_in_proof_compressions_are_today_s(d):
    """Inside a drawn unit's proof J10-R hashes what today's form does, its blocks, its lanes' key blocks and four for its
    seeds (`unit_compressions`): the saving is the serving side's, which derives each block once a call."""
    body = _v(getattr(H, d)).body
    defn = _defns(body)
    per = {SD.PouwNoiseBlockBits: 1, SD.PouwUnitSeedsBits: 4, SD.PouwCallSeedsBits: 4}
    assert sum(per.get(f, 0) for f in defn) == H.unit_compressions(64, (4, 8)) == 17


def test_signature_takes_no_seed():
    """v5's and v6's inputs are v1's and the public ``call`` and ``blocks``: no seed, noise block or counter is an input, so
    no looser seeding is admitted; the salt is the one coin port."""
    for d in (H.PouwH1Tile5, H.PouwH1Tile6):
        params, ret = _v(d).signature()
        assert [nm for nm, _t in params] == ["a", "b", "salt", "position", "roots", "call", "blocks"]
        assert params[5][1] == SD.POS_BITS and params[6][1] == Array(2, SD.POS_BITS)
        assert ret == _v(H.PouwH1Tile).signature()[1]
        if H._ports is not None:
            u = _v(d)
            assert {nm: p.kind for nm, p in u.defn.ports(u.S()).params.items()} == {"salt": "coin", "position": "position"}


def test_every_column_stream_reads_the_a_root():
    """In v5's gates seed_c is one `PouwCallSeeds` of the whole roots input (A's, then B's), every row and column block is
    seed_c's at a `PouwAdd64` of its ``blocks`` base, rows' at blocks[0] and columns' at blocks[1], and every key block σ_R's."""
    body = _v(H.PouwH1Tile5).body
    defn = _defns(body)
    (cs,) = [k for k, f in enumerate(defn) if f in SD.SEEDS]
    assert defn[cs] is SD.PouwCallSeedsBits
    D, P = SD.DIGEST_BITS.leaves, SD.POS_BITS.leaves
    assert [runs(a.refs) for a in body.nodes[cs].args] == [[("p", 2, 0, D)], [("p", 3, 0, P)], [("p", 5, 0, P)],
                                                             [("p", 4, 0, 2 * D)]]
    seed_c, sigma_r = [("n", cs, 0, D)], [("n", cs, D, 2 * D)]
    adds = {k: runs(body.nodes[k].args[0].refs) for k, f in enumerate(defn) if f is SD.PouwAdd64Bits}
    bases = {}
    for k, f in enumerate(defn):
        if f is SD.PouwNoiseBlockBits:
            s, t = (runs(a.refs) for a in body.nodes[k].args)
            if s == sigma_r:
                assert t[0][0] != "n" or t[0][1] not in adds
                continue
            assert s == seed_c and len(t) == 1 and t[0][0] == "n" and t[0][1] in adds
            bases[tuple(adds[t[0][1]])] = bases.get(tuple(adds[t[0][1]]), 0) + 1
    assert bases == {(("p", 6, 0, P),): 4, (("p", 6, P, 2 * P),): 8}


def test_the_seed_moves_every_slice():
    """seed_c reads the salt, the call and both roots: another A root (or B root, salt or call) moves every row and column of
    a unit's noise, so no column stream is seeded without the A root; two calls of one step and K, which share their roots,
    share no stream; and seed_c is neither a unit's seed_j nor its σ_R at the call's position."""
    lay, blocks = H.Layout(4, 8, 64), H.j10r_blocks(8, 64, (4, 8), 1, 2)
    base = H.j10r_noise(SD.call_seed(SALT, 40, *ROOTS), lay, blocks)
    other = hashlib.sha512(b"other").digest()
    for seed_c in (SD.call_seed(SALT, 40, other, ROOTS[1]), SD.call_seed(SALT, 40, ROOTS[0], other),
                   SD.call_seed(other, 40, *ROOTS), SD.call_seed(SALT, 41, *ROOTS)):
        rows, cols = H.j10r_noise(seed_c, lay, blocks)
        assert all(x != y for x, y in zip(rows, base[0])) and all(x != y for x, y in zip(cols, base[1]))
    assert SD.call_seed(SALT, 40, *ROOTS) not in (SD.seed(SALT, 40, *ROOTS), SD.receipt_seed(SALT, 40, *ROOTS))
    for bad in ((SALT, 40, ROOTS[0][:32], ROOTS[1]), (SALT, 40, ROOTS[0], b""), (SALT, -1, *ROOTS), (SALT, 1 << 64, *ROOTS)):
        with pytest.raises(ValueError):
            SD.call_seed(*bad)


def test_the_fold_keys_are_today_s():
    """`j10r_reference`'s receipt folds from today's keys, σ_R at the unit's own position; its z is the exact product (v5)
    or the one stream's (v6), both A Bᵀ."""
    rng = random.Random(3)
    a, b = _rows(rng, 4, 64), _rows(rng, 8, 64)
    blocks = H.j10r_blocks(8, 64, (4, 8), 1, 2)
    z5, rc5 = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks)
    z6, rc6 = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks, stream=True)
    assert [_s32(v) for v in z5] == [sum(_sx8(x) * _sx8(w) for x, w in zip(ai, bj)) for ai in a for bj in b] == \
        [_s32(v) for v in z6]
    assert rc5 == rc6 and rc5 != H.reference(a, b, SALT, 46, ROOTS)[1]
    lay = H.Layout(4, 8, 64)
    rows, cols = H.j10r_noise(SD.call_seed(SALT, 40, *ROOTS), lay, blocks)
    keys = H.fold_keys(SD.receipt_seed(SALT, 46, *ROOTS), lay)
    assert rc5 == H.fold(keys[0], H.lane_sets(a, b, rows, cols, lay, 0))


def _seed_args(boolean, salt, position, call, roots):
    return [SD.leaves(salt, boolean), SD.position_leaves(position, boolean), SD.position_leaves(call, boolean),
            SD.leaves(b"".join(roots), boolean)]


def test_call_seeds_on_words_and_add64():
    """`PouwCallSeeds_v1` is [`call_seed`, `receipt_seed`]; `PouwAdd64` v1 and v2 are a + b mod 2^64, carry and wrap
    included."""
    got = list(evaluate(bind(SD.PouwCallSeeds), *_seed_args(False, SALT, 46, 1 << 40, ROOTS)))
    assert got == SD.halves(SD.call_seed(SALT, 1 << 40, *ROOTS)) + SD.halves(SD.receipt_seed(SALT, 46, *ROOTS))
    rng = random.Random(5)
    M = (1 << 64) - 1
    for x, y in [(0, 0), ((1 << 32) - 1, 1), (M, 1), (M, M), *((rng.getrandbits(64), rng.getrandbits(40)) for _ in range(16))]:
        want = (x + y) & M
        assert list(evaluate(bind(SD.PouwAdd64), SD.position_leaves(x, False), SD.position_leaves(y, False))) == \
            SD.position_leaves(want, False)
        assert list(evaluate(SD.PouwAdd64Bits, SD.position_leaves(x, True), SD.position_leaves(y, True))) == \
            SD.position_leaves(want, True)


@pytest.mark.slow
def test_call_seeds_on_bits_is_the_reference():
    """`PouwCallSeeds_v2` on bits (four SHA-512 compressions) is [seed_c, σ_R]: the only test of its gates."""
    got = list(evaluate(SD.PouwCallSeedsBits, *_seed_args(True, SALT, 46, 1 << 40, ROOTS)))
    assert got == SD.leaves(SD.call_seed(SALT, 1 << 40, *ROOTS), True) + SD.leaves(SD.receipt_seed(SALT, 46, *ROOTS), True)


def _tail(position, call, blocks):
    return [SD.leaves(SALT, True), SD.position_leaves(position, True), SD.leaves(ROOTS[0] + ROOTS[1], True),
            SD.position_leaves(call, True), SD.position_leaves(blocks[0], True) + SD.position_leaves(blocks[1], True)]


@pytest.mark.slow
@pytest.mark.parametrize("d,stream", [("PouwH1Tile5", False), ("PouwH1Tile6", True)])
def test_j10r_tile_is_its_reference(d, stream):
    """On bits, unit (1, 2) of an 8 × 32 call at K = 64 (blocks (4, 24), the call at position 40, the unit at 46) with
    full-range rows: z and receipt are `j10r_reference`'s, z the served A Bᵀ. Slow (about 90 s for v5, 200 s for v6): the
    only test that the units' gates, `PouwCallSeeds`' and `PouwAdd64`'s counters among them, compute J10-R's noise."""
    rng = random.Random(9)
    a, b = _rows(rng, 4, 64), _rows(rng, 8, 64)
    blocks = H.j10r_blocks(8, 64, (4, 8), 1, 2)
    assert blocks == (4, 24)
    args = [[v for row in a for v in _bits(row)], [v for row in b for v in _bits(row)]]
    got = list(evaluate(_v(getattr(H, d)), *args, *_tail(46, 40, blocks)))
    z, rc = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks, stream=stream)
    zl = 32 * len(GRID48)
    assert _words(got[:zl]) == z and _words(got[zl:]) == rc
    assert [_s32(v) for v in z] == [sum(_sx8(x) * _sx8(w) for x, w in zip(ai, bj)) for ai in a for bj in b]


def test_no_scheme_takes_j10r():
    """J10-R is a Definition only: every registered scheme's units still bind v1 or v3."""
    from verity.core.protocols.pouw.circuit import compile as C

    for name in (H.SCHEME, H.SCHEME_V3):
        assert {u.defn for u in C._scheme(name).units.values()} <= {H.PouwH1Tile, H.PouwH1Tile3}
