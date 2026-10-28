"""`circuit.h1tile`'s J10-R v2 units, `PouwH1Tile_v7` and `PouwH1Tile_v8`: U fresh at every entry of K, so the
period-32 fold that gives v1's b Uᵀ from 32 words a row no longer does; their sets, their reference and the SHA-512
blocks the longer U stream costs a call."""
from __future__ import annotations

import hashlib
import random
from fractions import Fraction

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD

pytestmark = pytest.mark.skipif(H.SG is None, reason="H1's receipts (circuit.segment, E2E-120) are not on this tree")
SALT = hashlib.sha512(b"salt").digest()
ROOTS = (hashlib.sha512(b"a/64@3").digest(), hashlib.sha512(b"b/64").digest())
GRID48 = tuple((i, j) for i in range(4) for j in range(8))
P = (1 << 61) - 1


def _v(d):
    return bind(d, EXACT=bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8), R=4, C=8, PAIRS=GRID48)


def _sx8(v):
    return v - 256 if v & 0x80 else v


def _s32(v):
    return v - (1 << 32) if v >> 31 & 1 else v


def _bits(codes):
    return [(v >> i) & 1 for v in codes for i in range(8)]


def _words(bits):
    return [sum(b << i for i, b in enumerate(bits[t:t + 32])) for t in range(0, len(bits), 32)]


def _fold(bits):
    """A row's residue sums: entry rho the sum of its entries at rho, rho + 32, ...; what v1's fold reads of b."""
    return [sum(bits[rho::32]) for rho in range(32)]


def _rank(rows):
    """Rank mod P of an integer matrix, a lower bound on its rank over Q."""
    m, rank = [[x % P for x in r] for r in rows], 0
    for col in range(len(m[0]) if m else 0):
        piv = next((i for i in range(rank, len(m)) if m[i][col]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][col], P - 2, P)
        for i in range(len(m)):
            if i != rank and m[i][col]:
                f = m[i][col] * inv % P
                m[i] = [(x - f * y) % P for x, y in zip(m[i], m[rank])]
        rank += 1
    return rank


def _call_noise(v, m=8, n=64, k=128):
    return H.noise(SD.call_seed(SALT, 40, *ROOTS), H.Layout(m, n, k, v))


@pytest.mark.parametrize("v", [1, 2])
def test_the_period_32_fold_gives_b_u_under_v1_only(v):
    """With A = B = 0, X = b and the lane's T_last is b (c + U)ᵀ. Under v1 it is b cᵀ plus the fold, sum_rho Y_rho U_rho over
    b's 32 residue sums Y, which a prover gets in 32 / K of a product; under v2 that fold is not b Uᵀ, and no function of the
    residue sums is: two dither rows with the same sums (a 1 at t or at t + 32) give different b Uᵀ in every column."""
    k = 128
    lay = H.Layout(4, 8, k, v)
    rows, cols = H.j10r_noise(SD.call_seed(SALT, 40, *ROOTS), lay, H.j10r_blocks(8, k, (4, 8), 1, 2, v))
    zero_a, zero_b = [[0] * k for _ in range(4)], [[0] * k for _ in range(8)]
    t_last = H.lane_sets(zero_a, zero_b, rows, cols, lay, 0)[lay.ks - 1]
    R, Cs = lay.rows(0), lay.cols(0)
    folded = []
    for e in range(32):
        bi, (cj, uj) = rows[R[H.SG.row_slot(e)]][0], cols[Cs[H.SG.col_slot(e)]]
        bc = sum(x & y for x, y in zip(bi, cj))
        folded.append((bc + sum(y * _sx8(u) for y, u in zip(_fold(bi), uj[:32]))) & H.M32)
    assert len(cols[0][1]) == (32 if v == 1 else k)
    if v == 1:
        assert folded == t_last
        return
    assert all(f != t for f, t in zip(folded, t_last))
    for _cj, uj in cols:
        assert any(_sx8(uj[t]) != _sx8(uj[t + 32]) for t in range(k - 32))


def test_u_is_a_fresh_matrix_under_v2():
    """At call scale U is a K × N matrix: under v1 its rank is at most 32 (its rows repeat every 32 entries of K), so b Uᵀ
    is a product through a 32-wide fold; under v2 it has full rank K, so b (c + U)ᵀ is a product of fresh random
    matrices."""
    k, n = 128, 128
    for v, want in ((1, 32), (2, k)):
        _rows, cols = _call_noise(v, n=n, k=k)
        u = [[_sx8(c[1][t % len(c[1])]) for c in cols] for t in range(k)]
        assert _rank(u) == want, v


def test_v2_sets_are_t_and_o_prime():
    """Under v2 a lane absorbs KS + 1 sets, T_0 .. T_(KS-1) of X (B + c + U)ᵀ and O' = T_last - X Uᵀ (no digits); a row's
    noise is its K dither bits alone and a column's its K coin bits then K bytes of U."""
    rng = random.Random(7)
    k = 96
    lay = H.Layout(4, 8, k, 2)
    assert (lay.m, lay.sets, lay.row_bits, lay.col_bits) == (1, k // 32 + 1, k, 9 * k)
    a, b = [[rng.randrange(256) for _ in range(k)] for _ in range(4)], [[rng.randrange(256) for _ in range(k)] for _ in range(8)]
    rows, cols = H.j10r_noise(SD.call_seed(SALT, 40, *ROOTS), lay, H.j10r_blocks(8, k, (4, 8), 1, 2, 2))
    assert all(len(r[0]) == k and r[1] == [] for r in rows) and all(len(c[1]) == k for c in cols)
    sets = H.lane_sets(a, b, rows, cols, lay, 0)
    assert len(sets) == lay.sets
    R, Cs = lay.rows(0), lay.cols(0)
    for e in range(32):
        i, j = R[H.SG.row_slot(e)], Cs[H.SG.col_slot(e)]
        x = [_sx8((a[i][t] + rows[i][0][t]) & 0xFF) for t in range(k)]
        w = [_sx8((b[j][t] + cols[j][0][t] + cols[j][1][t]) & 0xFF) for t in range(k)]
        for s in range(lay.ks):
            assert sets[s][e] == sum(x[t] * w[t] for t in range(32 * s + 32)) & H.M32
        assert sets[-1][e] == (sets[lay.ks - 1][e] - sum(x[t] * _sx8(cols[j][1][t]) for t in range(k))) & H.M32


def test_v2_reference_serves_a_b():
    """`j10r_reference` at v2: z is A Bᵀ in both forms, the receipt 2 (KS + 1) words, the lanes' folds over v2's sets, and
    it is not v1's receipt."""
    rng = random.Random(3)
    a, b = [[rng.randrange(256) for _ in range(64)] for _ in range(4)], [[rng.randrange(256) for _ in range(64)] for _ in range(8)]
    for row in a:
        row[3::7] = [127] * len(row[3::7])
    blocks = H.j10r_blocks(8, 64, (4, 8), 1, 2, 2)
    z7, rc7 = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks, v=2)
    z8, rc8 = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks, stream=True, v=2)
    exact = [sum(_sx8(x) * _sx8(w) for x, w in zip(ai, bj)) for ai in a for bj in b]
    assert [_s32(v) for v in z7] == exact == [_s32(v) for v in z8]
    assert rc7 == rc8 and len(rc7) == 2 * (64 // 32 + 1)
    lay = H.Layout(4, 8, 64, 2)
    rows, cols = H.j10r_noise(SD.call_seed(SALT, 40, *ROOTS), lay, blocks)
    keys = H.fold_keys(SD.receipt_seed(SALT, 46, *ROOTS), lay)
    assert rc7 == H.fold(keys[0], H.lane_sets(a, b, rows, cols, lay, 0))
    assert rc7 != H.j10r_reference(a, b, SALT, 46, ROOTS, 40, H.j10r_blocks(8, 64, (4, 8), 1, 2))[1][:len(rc7)]


@pytest.mark.parametrize("m,n,k,tile", [(8, 32, 64, (4, 8)), (64, 96, 2048, (32, 32))])
def test_v2_unit_slices_are_the_call_scale_noise(m, n, k, tile):
    """Under v2 every unit (I, J) still reads, at `j10r_blocks`' counters, exactly its rows and columns of the call-scale
    noise."""
    r, c = tile
    seed_c = SD.call_seed(SALT, 40, *ROOTS)
    rows, cols = H.noise(seed_c, H.Layout(m, n, k, 2))
    lay = H.Layout(r, c, k, 2)
    for I in range(m // r):
        for J in range(n // c):
            ur, uc = H.j10r_noise(seed_c, lay, H.j10r_blocks(m, k, tile, I, J, 2))
            assert ur == rows[r * I:r * I + r] and uc == cols[c * J:c * J + c], (I, J)


#: Llama-3.2-1B INT8's linears at K = 2048 (name, n) and the 1B window's slot (T = 512 rows), layers and steps
LLAMA_1B_K2048 = (("qkv", 3072), ("o", 2048), ("gate_up", 16384))
SLOT_1B, LAYERS_1B, STEPS_1B = 512, 16, 16


def test_the_longer_u_stream_s_blocks_at_1b_k2048():
    """At K = 2048 a column's stream grows from CB = 5 blocks (2048 coin bits and 32 bytes of U) to 36 (2048 coin bits and
    2048 bytes) and a row's shrinks from RB = 5 to 4 (no digit coins), so a call's stream grows by 31 N - M blocks. At the
    1B window's K = 2048 calls (M = 512): 94,720 (qkv), 62,976 (o) and 507,392 (gate_up) more SHA-512 blocks a call, 10.6M
    more a step of 16 layers (12.5M against 1.84M, 6.8x)."""
    v1, v2 = H.Layout(32, 32, 2048), H.Layout(32, 32, 2048, 2)
    assert (v1.rb, v1.cb, v2.rb, v2.cb) == (5, 5, 4, 36)
    extra = {}
    for name, n in LLAMA_1B_K2048:
        a, b = H.Layout(SLOT_1B, n, 2048).blocks, H.Layout(SLOT_1B, n, 2048, 2).blocks
        extra[name] = b - a
        assert b - a == 31 * n - SLOT_1B
        units = (SLOT_1B // 32) * (n // 32)
        assert H.j10r_compressions(SLOT_1B, n, 2048, v=2) - H.j10r_compressions(SLOT_1B, n, 2048) == \
            Fraction(b - a, units)
    assert extra == {"qkv": 94720, "o": 62976, "gate_up": 507392}
    step = {v: LAYERS_1B * sum(H.Layout(SLOT_1B, n, 2048, v).blocks + 2 for _nm, n in LLAMA_1B_K2048) for v in (1, 2)}
    assert step == {1: 1843296, 2: 12484704} and step[2] - step[1] == LAYERS_1B * sum(extra.values()) == 10641408


def test_v7_v8_signatures_are_v5_s_with_v2_s_receipt():
    """v7's and v8's inputs are v5's (no seed, block or counter an input); their receipt is 2 (KS + 1) words."""
    want = _v(H.PouwH1Tile5).signature()[0]
    for d in (H.PouwH1Tile7, H.PouwH1Tile8):
        params, ret = _v(d).signature()
        assert params == want
        assert dict(ret.fields)["receipt"].n == 2 * (64 // 32 + 1)


def test_no_scheme_takes_v2():
    from verity.core.protocols.pouw.circuit import compile as C

    for name in (H.SCHEME, H.SCHEME_V3):
        assert {u.defn for u in C._scheme(name).units.values()} <= {H.PouwH1Tile, H.PouwH1Tile3}


def test_noise_version_is_checked():
    with pytest.raises(ValueError, match="noise version"):
        H.Layout(4, 8, 64, 3)


def _tail(position, call, blocks):
    return [SD.leaves(SALT, True), SD.position_leaves(position, True), SD.leaves(ROOTS[0] + ROOTS[1], True),
            SD.position_leaves(call, True), SD.position_leaves(blocks[0], True) + SD.position_leaves(blocks[1], True)]


@pytest.mark.slow
@pytest.mark.parametrize("d,stream", [("PouwH1Tile7", False), ("PouwH1Tile8", True)])
def test_v2_tile_is_its_reference(d, stream):
    """On bits, unit (1, 2) of an 8 × 32 call at K = 64 under v2 (blocks (4, 40)) with full-range rows: z and receipt are
    `j10r_reference`'s at v2, z the served A Bᵀ. Slow (minutes, as v5's and v6's): the only test that v7's and v8's gates,
    U's K bytes a column and O' over every segment among them, compute v2."""
    rng = random.Random(9)
    a, b = [[rng.randrange(256) for _ in range(64)] for _ in range(4)], [[rng.randrange(256) for _ in range(64)] for _ in range(8)]
    for row in a + b:
        row[3::7] = [127] * len(row[3::7])
    blocks = H.j10r_blocks(8, 64, (4, 8), 1, 2, 2)
    assert blocks == (4, 40)
    args = [[v for row in a for v in _bits(row)], [v for row in b for v in _bits(row)]]
    got = list(evaluate(_v(getattr(H, d)), *args, *_tail(46, 40, blocks)))
    z, rc = H.j10r_reference(a, b, SALT, 46, ROOTS, 40, blocks, stream=stream, v=2)
    zl = 32 * len(GRID48)
    assert _words(got[:zl]) == z and _words(got[zl:]) == rc
    assert [_s32(v) for v in z] == [sum(_sx8(x) * _sx8(w) for x, w in zip(ai, bj)) for ai in a for bj in b]
