"""`circuit.lead`: each piece is its word and has its stated ANDs, the reference's words are Layout.lean's `word`, the unit
accepts a call's committed words and rejects a changed word, a tag weight outside [-3, 3], an even lane coin and a served
word past 2^22, the fold reads back A_i · B_j and serves verity_int8's bf16 bit for bit, a ragged call's padding rows and
columns carry their coins, a unit's ANDs are `unit_ands`, its output is one bit, and a binding past the calls the verifier
admits is refused; the window's record, positions, domains and stream order are the served committer's, a unit's opening
reads back its words, and the verifier refuses each window the Definition refuses."""
from __future__ import annotations

import copy
import hashlib
import random
import struct

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.evaluation.bits import compile as compile_bits
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.core.protocols.pouw.circuit import lead as L

WORDS = {L.PouwLeadStep0: L.PouwLeadStep0Word, L.PouwLeadStep: L.PouwLeadStepWord, L.PouwLeadPad0: L.PouwLeadPad0Word,
         L.PouwLeadPad: L.PouwLeadPadWord, L.PouwLeadWeigh: L.PouwLeadWeighWord,
         L.PouwLeadTagWeight: L.PouwLeadTagWeightWord, L.PouwLeadPairS8: L.PouwLeadPairS8Word,
         L.PouwLeadPair32: L.PouwLeadPair32Word, L.PouwLeadPop: L.PouwLeadPopWord, L.PouwLeadSub32: L.PouwLeadSub32Word,
         L.PouwLeadFold: L.PouwLeadFoldWord}
#: p at and just past the fold's window's ends, and far past them
EDGES = [0, L.LEAD_FOLD_BOUND, -L.LEAD_FOLD_BOUND, L.LEAD_FOLD_BOUND + 1, -L.LEAD_FOLD_BOUND - 1, 1 << 31, (1 << 31) - 1]


def _fold_inputs(rng: random.Random, p: int) -> list[int]:
    """The fold's inputs, random sources and bytes, with the final word chosen so the fold word is FOLD_MAGIC + p."""
    r, c = rng.randrange(1 << 32), rng.randrange(1 << 32)
    nt, g, na, cj = ([rng.randrange(256) for _ in range(4)] for _ in range(4))
    w = (L.FOLD_MAGIC + p - L.fold_word(0, r, c, nt, g, na, cj)) & L.M32
    return [w, r, c, *nt, *g, *na, *cj]
#: each step piece's product lanes and whether it takes a tag lane
STEPS = {L.PouwLeadStep0: (L.LANES_0, False), L.PouwLeadStep: (L.LANES_T, True), L.PouwLeadPad0: (L.COINS_0, False),
         L.PouwLeadPad: (0, True)}


def _vectors(piece, rng: random.Random, n: int) -> list[list[int]]:
    """n inputs of ``piece``, one int per word element; a step's odd ones honest, the fold's p near its window's ends."""
    out = []
    for k in range(n):
        if piece in STEPS:
            lanes, tagged = STEPS[piece]
            a, b = [rng.randrange(256) for _ in range(lanes)], [rng.randrange(256) for _ in range(lanes)]
            prev, lane, tag = rng.randrange(1 << 32), rng.randrange(1 << 32), rng.randrange(2)
            w = prev + sum(L._sx8(x) * L._sx8(y) for x, y in zip(a, b)) + lane + (1 + tag if tagged else 0)
            tail = [lane, tag] if tagged else [lane]
            out.append([prev, (w + (0 if k % 2 else rng.choice([1, -1, 1 << 31, 1 << 16]))) & L.M32, *a, *b, *tail])
        elif piece is L.PouwLeadFold:
            out.append(_fold_inputs(rng, rng.choice(EDGES + [rng.randrange(-(1 << 31), 1 << 31)])))
        elif piece is L.PouwLeadTagWeight:
            out.append([k % 256])
        else:
            out.append([rng.randrange(1 << w) for w in _widths(piece)])
    return out


def _widths(piece) -> list[int]:
    from verity.ml.circuits import boolean as BL

    params, _ = bind(piece).signature()
    return [w for _n, t in params for w in BL.elements(t)]


def _split(piece, rows: list[list[int]]) -> list[np.ndarray]:
    from verity.ml.circuits import boolean as BL

    params, _ = bind(piece).signature()
    out, k = [], 0
    for _n, t in params:
        e = len(BL.elements(t))
        out.append(np.array([r[k:k + e] for r in rows], dtype=np.uint64))
        k += e
    return out


@pytest.mark.parametrize("piece", L.PIECES, ids=lambda d: d.id)
def test_each_piece_is_its_word(piece):
    rows = _vectors(piece, random.Random(piece.id), 256)
    got = np.asarray(evaluate_words(piece, *_split(piece, rows))).reshape(len(rows), -1)
    for r, g in zip(rows, got):
        assert tuple(int(x) for x in g) == evaluate(WORDS[piece], *r)


def test_each_piece_has_its_stated_ands():
    assert {d.id: d.traced().counts["and"] for d in L.PIECES} == L.PIECE_ANDS


@pytest.mark.parametrize("piece", list(STEPS), ids=lambda d: d.id)
def test_the_steps_the_fold_and_the_tag_weight_decide_what_they_say(piece):
    rows = _vectors(piece, random.Random(1), 64)
    assert [evaluate(WORDS[piece], *r)[0] for r in rows] == [k % 2 for k in range(64)]
    rng = random.Random(2)
    for p in EDGES:
        assert evaluate(L.PouwLeadFoldWord, *_fold_inputs(rng, p)) == (int(L.fold_admits(p)),)
    assert [G for G in range(256) if evaluate(L.PouwLeadTagWeightWord, G) == (1,)] == [0, 1, 2, 3, 253, 254, 255]
    assert [evaluate(L.PouwLeadPopWord, x)[0] for x in (0, 1, L.M32, 0x80000001)] == [0, 1, 32, 2]


def _ediv(a: int, b: int) -> int:
    """Lean's `Int` division by a positive b: Euclidean, which is floor division."""
    return a // b


def lean_word(c: L.Call, i: int, t: int, j: int) -> int:
    """Layout.lean's `word` of row i, step t, column j of the lead at the call's shape, transliterated: κ the step's dot
    product of A and B (zero on a padding row or column), `coinTerm` and `laneTerm` as Lean states them."""
    def bal13(z, q):
        return _ediv(z + 14280, 13 ** q) % 13 - 6

    def row_tag(i, q):
        return bal13(i // 2 + 1 if i % 2 == 0 else -(i // 2 + 1), q)

    def col_code(j, q):
        return bal13(j + 1, q)

    def rot_weight(i, ell):
        return 127 if ell == i % 2 else 1

    S = 1 + (max(c.K - 21, 0) + 27) // 28
    bias = _ediv(sum(col_code(j, q) for q in range(4)) + 3 * (S - 1) + 1, 2)

    def kappa(tau):
        lo, hi = (0, 21) if tau == 0 else (21 + 28 * (tau - 1), 21 + 28 * tau)
        if i >= c.rows or j >= c.cols:
            return 0
        return sum(c.A[i][k] * c.B[j][k] for k in range(lo, min(hi, c.K)))

    def lane_term(tau):
        return sum(rot_weight(i, ell) * (2 * c.u[tau][i // 64][ell][j] - (2 ** 7 - 1)) for ell in range(3))

    def coin_term(tau):
        if tau == 0:
            head = sum(row_tag(i, q) * (c.g[q][j] - 3) for q in range(4)) + sum(c.a1[i][q] * col_code(j, q) for q in range(4))
        else:
            head = 1 + c.b[i][tau]
        return head + lane_term(tau)

    return -bias + sum(kappa(tau) + coin_term(tau) for tau in range(S) if tau <= t)


@pytest.mark.parametrize("rows, cols, K", [(17, 13, 77), (70, 9, 21), (16, 8, 50)])
def test_the_reference_words_are_leans(rows, cols, K):
    c = L.Call.draw(random.Random(rows * K), rows, cols, K)
    W = c.words()
    assert (len(W), len(W[0]), len(W[0][0])) == (c.S, c.m, c.n)
    assert all(W[t][i][j] == lean_word(c, i, t, j) for t in range(c.S) for i in range(c.m) for j in range(c.n))


def test_steps_and_words_fit_s32_at_every_admitted_k():
    assert all(L.steps_for(K) == 1 + (max(K - 21, 0) + 27) // 28 for K in range(0, 2000))
    assert L.steps_for(L.MAX_K) == 2341
    assert L.word_bound(L.steps_for(L.MAX_K)) < 1 << 31


def test_the_unit_accepts_the_committed_words_and_rejects_a_changed_one():
    c = L.Call.draw(random.Random(2), 16, 8, 49)
    W = c.words()
    assert c.unit_ok(0, 0, 0, W) == 1
    for t, i, j, d in [(0, 0, 0, 1), (1, 15, 7, -1), (1, 3, 2, 1 << 31), (0, 9, 4, 1 << 32)]:
        bad = copy.deepcopy(W)
        bad[t][i][j] += d
        assert c.unit_ok(0, 0, 0, bad) == (1 if d == 1 << 32 else 0)


@pytest.mark.parametrize("g", [7, -1])
def test_a_tag_weight_outside_its_coins_is_rejected(g):
    """G = g - 3 is 4 or -4: in no `Coins`, which draw g from Fin 7."""
    c = L.Call.draw(random.Random(3), 16, 8, 21)
    c.g[2][5] = g
    assert c.unit_ok(0, 0, 0, c.words()) == 0


@pytest.mark.parametrize("K, BLK, step", [(21, 0, 0), (917, 1, 20)])
def test_an_even_lane_coin_is_rejected(K, BLK, step):
    """s = 2u - 127 is odd; an even byte is no coin, whatever the words. A last block reads every step's coins."""
    c = L.Call.draw(random.Random(K), 16, 8, K)
    u = c.unit(0, 0, BLK)
    args = c.unit_args(0, 0, BLK)
    k = [name for name, _t in u.params()].index("s")
    assert c.unit_ok(0, 0, BLK) == 1
    args[k][0][(step * 8 + 3) * 3 + 1] ^= 1
    fn = bind(L.PouwLeadTile, K=K, R0=0, C0=0, ROWS=16, COLS=8, BLK=BLK)
    assert int(np.asarray(evaluate_words(fn, *args)).reshape(-1)[0]) == 0


@pytest.mark.parametrize("rows, i", [(16, 4), (7, 12)])
def test_blocks_chain_through_the_words_before_them(rows, i):
    """K = 917 is 33 steps: block 0 is 32 steps without the fold, block 1 one step from the words of step 31 with the fold;
    a 7-row call's tile is ragged, and its row 12 a padding row."""
    c = L.Call.draw(random.Random(4), rows, 8, 917)
    W = c.words()
    assert (c.unit_ok(0, 0, 0, W), c.unit_ok(0, 0, 1, W)) == (1, 1)
    for t, want in [(5, (0, 1)), (31, (0, 0)), (32, (1, 0))]:
        bad = copy.deepcopy(W)
        bad[t][i][6] += 1
        assert (c.unit_ok(0, 0, 0, bad), c.unit_ok(0, 0, 1, bad)) == want


@pytest.mark.parametrize("K, a, ok", [(260, 127, 1), (261, 127, 0), (256, -128, 1), (301, -128, 0)])
def test_a_served_word_past_the_folds_range_is_a_visible_failure(K, a, ok):
    """All-127 B against all-a A: A_i · B_j is 127 · a · K, 4,193,540 at K = 260 and 4,209,669 at K = 261; -128 · 127 ·
    256 is -2^22 + 2^15."""
    c = L.Call.draw(random.Random(K), 16, 8, K)
    c.A, c.B = [[a] * K for _ in range(16)], [[127] * K for _ in range(8)]
    assert (c.refuse() is None) == bool(ok)
    assert c.unit_ok(0, 0, 0) == ok


@pytest.mark.parametrize("rows, cols, K", [(17, 13, 77), (1, 64, 2048), (16, 8, 917)])
def test_the_fold_reads_back_a_times_b_and_serves_verity_int8s_bf16(rows, cols, K):
    """Every coin term leaves the fold word, which is FOLD_MAGIC + A_i · B_j, and the served bf16 under any fp32 scales is
    verity_int8's epilogue on A_i · B_j; a decode row at Llama's K among them."""
    rng = random.Random(rows * K)
    c = L.Call.draw(rng, rows, cols, K)
    W = c.words()
    for i in range(rows):
        for j in range(cols):
            assert (c.fold(i, j, W) - L.FOLD_MAGIC) & L.M32 == c.product(i, j) & L.M32
            si, sj = (L.f32(rng.uniform(2.0 ** -12, 2.0 ** -2)) for _ in range(2))
            assert c.served(i, j, si, sj, W) == L.int8_epilogue(c.product(i, j), si, sj)


def test_the_fold_word_misreads_past_its_range():
    """FADD reads FOLD_MAGIC + p back as p exactly at |p| <= 2^22; 2^22 + 1 reads as 2^22 + 2 and -2^22 - 1 as
    -2^22 - 0.5."""
    def fadd(p):
        return struct.unpack("<f", struct.pack("<I", (L.FOLD_MAGIC + p) & L.M32))[0] - L.FOLD_OFFSET

    assert all(fadd(p) == p for p in (0, 1, -1, L.LEAD_FOLD_BOUND, -L.LEAD_FOLD_BOUND, 12345, -999999))
    assert (fadd(L.LEAD_FOLD_BOUND + 1), fadd(-L.LEAD_FOLD_BOUND - 1)) == (L.LEAD_FOLD_BOUND + 2, -L.LEAD_FOLD_BOUND - 0.5)


@pytest.mark.parametrize("change", ["tag", "a1", "g"])
def test_the_last_block_folds_with_every_coin_it_reads(change):
    """At a cell with A_i · B_j = 2^22 exactly, a tag bit of an earlier block's step (the walk), an a₁ bit or a G read
    differently moves p to 2^22 + 1, and the last block refuses; the honest coins pass. Row 0's tag and column 0's code
    are both (1, 0, 0, 0), so a₁[0, 0] from 1 to 0 and G[0, 0] from 0 to -1 each add 1 to p."""
    K = 917
    c = L.Call.draw(random.Random(9), 16, 8, K)
    c.A[0], c.B[0] = [-128] * 256 + [0] * (K - 256), [-128] * 256 + [0] * (K - 256)
    c.b[0][5], c.a1[0][0], c.g[0][0] = 1, 1, 3
    assert c.product(0, 0) == L.LEAD_FOLD_BOUND and L.row_tag(0) == L.col_code(0) == [1, 0, 0, 0]
    W = c.words()
    assert c.unit_ok(0, 0, 1, W) == 1
    if change == "tag":
        c.b[0][5] = 0
    elif change == "a1":
        c.a1[0][0] = 0
    else:
        c.g[0][0] = 2
    assert c.unit_ok(0, 0, 1, W) == 0


def test_a_ragged_calls_padding_rows_and_columns_carry_their_coins():
    """A 17 × 13 call commits 32 × 16 words: rows 17 .. 31 and columns 13 .. 15 are padding, zero data and their own coins.
    The unit of the tile at row 16 holds one row of the call; its padding rows' words are the coins' (Lean's `word` at zero
    data), so words computed without their coins, or a padding row's coin changed under the words, are rejected."""
    c = L.Call.draw(random.Random(5), 17, 13, 49)
    W = c.words()
    assert (c.m, c.n) == (32, 16)
    for R0 in (0, 16):
        for C0 in (0, 8):
            assert c.unit_ok(R0, C0, 0, W) == 1
    assert c.unit(16, 8, 0) == L.Unit(49, 16, 8, 1, 5, 0)

    coinless = copy.deepcopy(W)
    for t in range(c.S):
        for i in range(17, 32):
            for j in range(16):
                coinless[t][i][j] = -L.bias(c.S, j)
    assert coinless != W
    assert (c.unit_ok(16, 0, 0, coinless), c.unit_ok(16, 8, 0, coinless)) == (0, 0)
    assert c.unit_ok(0, 0, 0, coinless) == 1

    zeroed = copy.deepcopy(W)
    for t in range(c.S):
        for i in range(32):
            for j in range(13, 16):
                zeroed[t][i][j] = 0
    assert (c.unit_ok(0, 8, 0, zeroed), c.unit_ok(16, 8, 0, zeroed), c.unit_ok(0, 0, 0, zeroed)) == (0, 0, 1)

    for change in ("a1", "b", "u0", "u1", "u2"):
        d = copy.deepcopy(c)
        if change == "a1":
            d.a1[20][1] ^= 1
        elif change == "b":
            d.b[31][1] ^= 1
        else:
            d.u[1][0][int(change[1])][14] ^= 1
        assert d.unit_ok(16, 8, 0, W) == 0, change


SHAPES = [(21, 0, 0, 16, 8, 0), (49, 16, 8, 1, 5, 0), (917, 0, 0, 16, 8, 0), (917, 0, 0, 16, 8, 1), (1000, 32, 8, 16, 8, 1),
          (1, 0, 0, 3, 2, 0), (917, 0, 0, 7, 8, 1), (77, 0, 8, 16, 3, 0)]


@pytest.mark.parametrize("K, R0, C0, ROWS, COLS, BLK", SHAPES)
def test_a_units_ands_are_unit_ands_and_its_output_one_bit(K, R0, C0, ROWS, COLS, BLK):
    fn = bind(L.PouwLeadTile, K=K, R0=R0, C0=C0, ROWS=ROWS, COLS=COLS, BLK=BLK)
    assert compile_bits(fn).counts["and"] == L.unit_ands(K, BLK, ROWS, COLS)
    _params, ret = fn.signature()
    assert ret.leaves == 1 <= 16


@pytest.mark.parametrize("change, why", [
    ({"K": 0}, "outside"), ({"K": L.MAX_K + 1}, "outside"), ({"R0": 8}, "grid"), ({"C0": 4}, "grid"),
    ({"R0": -16}, "grid"), ({"R0": L.MAX_ROWS}, "passes"), ({"C0": L.MAX_COLS}, "passes"), ({"ROWS": 0}, "holds"),
    ({"ROWS": 17}, "holds"), ({"COLS": 0}, "holds"), ({"COLS": 9}, "holds"), ({"BLK": 1}, "past"),
    ({"K": 917, "BLK": 2}, "past"),
])
def test_a_binding_past_the_admitted_calls_is_refused(change, why):
    shape = {"K": 21, "R0": 0, "C0": 0, "ROWS": 16, "COLS": 8, "BLK": 0, **change}
    assert why in L.Unit(**shape).refuse()
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(L.PouwLeadTile, **shape)


@pytest.mark.parametrize("shape", [
    {"K": L.MAX_K}, {"K": 1}, {"R0": L.MAX_ROWS - 16}, {"C0": L.MAX_COLS - 8}, {"ROWS": 1, "COLS": 1},
    {"K": L.MAX_K, "BLK": (L.steps_for(L.MAX_K) - 1) // 32},
])
def test_the_largest_admitted_bindings_bind(shape):
    shape = {"K": 21, "R0": 0, "C0": 0, "ROWS": 16, "COLS": 8, "BLK": 0, **shape}
    assert L.Unit(**shape).refuse() is None
    assert bind(L.PouwLeadTile, **shape).signature()[1].leaves == 1


# -- the window ---------------------------------------------------------------------------------------------------------------

SALT = hashlib.sha512(b"lead test salt").digest()
#: the served committer's (`verity_vllm.commit.lead_rows`, at c0cf0bdec76) instance domains' ids and its leaf at rank 1 mod
#: count of b"\x01\x02\x03" under the words schema, at salt SALT
COMMITTER_DOMAINS = {
    (5, 3): ("5e67997b97e08262931bfaf3a58681ba00be1f7f58fecfee134fef4bd054a2fa432802e7ebfebf35c4099f6344c842a8c3ada4b55d8d77ba08256589075a9778",
             "23c7c503d1967b46612de07e48495b64ef14d35aa7527c6ee2ec93265d9b52c26a154d50c3738b98fa7b9007c50631a432ca9f56a77206ddea1aa1859805bc32"),
    ((1 << 63) + 2, 7): ("ab7974576b72383f85b53b481ff43ea2be712c2179bd23c062266419e6ebbf388af6050fd8c098598ad538a7d4813c35280b142a166d7db1d5085d6e14071cd8",
                         "f5b05afba1ad03dc9473943b1ef8491d085beb8fb8f005a4e607c3be8779be9335c971abe5108e131c27a4656c76e3abaf3bcf62ad0b561de6d91e97a16ebd9d"),
    (0, 1): ("a271f286fc6b5fcfc9164dab183c373428358d37e0ff3167545d9e6c11b5b670c1b9265a565d8dc2d639a2e98b759e255a1ed6b51d6d162cf6021bc64f909f84",
             "2d3e8274c04dc6962216f533bf00d13304d9da80145526e1cd6b81357faf9bd48b375ef5d31848fc43c33fe02864babc195de6211315b3dab171eaac2112d37e"),
}
#: SHA-256 (first 16 bytes) of the committer's `committed_tiles(rows, cols, S, bn)`, each committed tile's export word
#: offset in stream order as little-endian int64s
COMMITTER_TILES = {(17, 13, 3, 128): "15e3e310e4fb5c57a4556fef03784728", (70, 300, 2, 128): "c9328da82b6c6adf8b23e11836e143b6",
                   (130, 520, 2, 256): "c56fa2b02c8206640070ac654821715d", (1, 8, 4, 256): "07530363844e4b4415f2c1ad6be283ad",
                   (64, 256, 1, 256): "73c8193684a8bac478729d82c3349374"}


def _shape_at(rows: int, cols: int, S: int, bn: int) -> L.Shape:
    K = 1 if S == 1 else L.LANES0 + L.LANES * (S - 1)
    sh = L.Shape(rows, cols, K, bn)
    assert sh.S == S
    return sh


def test_the_record_is_the_tagged_preimage():
    root, leaf = hashlib.sha512(b"weights").digest(), hashlib.sha512(b"key leaf").digest()
    assert L.lead_record(root, leaf, 7) == hashlib.sha512(b"pouw/lead/register/v1" + root + leaf + bytes(7) + b"\x07").digest()
    assert len({L.lead_record(root, leaf, c) for c in (1, 2, 1 << 62)} | {L.lead_record(leaf, root, 1)}) == 4
    for bad in ((root[:63], leaf, 1), (root, leaf + b"\0", 1), (root, leaf, 0), (root, leaf, 1 << 63)):
        with pytest.raises(ValueError):
            L.lead_record(*bad)


def test_positions_are_injective_and_the_domains_are_the_committers():
    calls = 5
    seen = {L.a_position(f, calls, j) for f in range(20) for j in range(calls)} | {L.words_position(f) for f in range(20)}
    assert len(seen) == 120 and max(L.a_position(f, calls, j) for f in range(20) for j in range(calls)) < L.WORDS_POSITION
    assert L.position_refusal(0, calls, calls) and L.position_refusal(-1, calls, 0)
    assert L.position_refusal((L.WORDS_POSITION - 1) // calls, calls, calls - 1)
    assert L.position_refusal((L.WORDS_POSITION - 1) // calls - 1, calls, calls - 1) is None
    for (pos, count), (dom, leaf) in COMMITTER_DOMAINS.items():
        d = L.instance_domain(SALT, pos, count)
        assert d.domain_id.hex() == dom
        assert d.leaf(1 % count, 1 % count, L.WORDS_SCHEMA, b"\x01\x02\x03").hex() == leaf


@pytest.mark.parametrize("key", list(COMMITTER_TILES))
def test_the_stream_order_is_the_committers(key):
    """Each committed tile's stream index (`Shape.tile`) placed at its export word offset gives the committer's
    `committed_tiles`, every index once."""
    rows, cols, S, bn = key
    sh = _shape_at(*key)
    nct = -(-cols // bn)
    offsets = {}
    for R0 in range(0, sh.m, 16):
        for C0 in range(0, sh.n, 8):
            rt, w, ct, ni = R0 // 64, R0 % 64 // 16, C0 // bn, C0 % bn // 8
            for t in range(S):
                offsets[sh.tile(R0, C0, t)] = ((rt * nct + ct) * S + t) * 64 * bn + (ni * 4 + w) * 128
    assert sorted(offsets) == list(range(sh.m // 16 * sh.n // 8 * S))
    got = np.array([offsets[k] for k in range(len(offsets))], dtype="<i8").tobytes()
    assert hashlib.sha256(got).hexdigest()[:32] == COMMITTER_TILES[key]


def test_fragment_order_is_wgmmas():
    assert sorted(L.fragment(r, c) for r in range(16) for c in range(8)) == list(range(128))
    assert [L.fragment(r, c) for r, c in [(0, 0), (0, 1), (8, 0), (0, 2), (1, 0), (15, 7)]] == [0, 1, 2, 4, 16, 127]


def _stream(W: list, sh: L.Shape) -> list[int]:
    """A call's committed words W in stream order, as the committer lays them out."""
    out = [0] * (sh.m * sh.n * sh.S)
    for R0 in range(0, sh.m, 16):
        for C0 in range(0, sh.n, 8):
            for t in range(sh.S):
                base = sh.tile(R0, C0, t) * 128
                for r in range(16):
                    for col in range(8):
                        out[base + L.fragment(r, col)] = W[t][R0 + r][C0 + col] & L.M32
    return out


def test_a_units_opening_reads_back_its_words_and_rows():
    """A forward of three calls, the second's chunks after the first's: every unit's words, read from the forward's words
    leaves where `unit_opening` says, are the unit's committed words, and its A runs are its rows' bytes at the block."""
    rng = random.Random(11)
    calls = [L.Call.draw(rng, 17, 13, 917), L.Call.draw(rng, 70, 300, 120), L.Call.draw(rng, 3, 8, 21)]
    shapes = [L.Shape(c.rows, c.cols, c.K, bn) for c, bn in zip(calls, (128, 128, 256))]
    words = [c.words() for c in calls]
    leaves = []
    for c, sh, W in zip(calls, shapes, words):
        s = _stream(W, sh)
        leaves += [s[k:k + L.CHUNK_WORDS] for k in range(0, len(s), L.CHUNK_WORDS)]
        assert -(-len(s) // L.CHUNK_WORDS) == sh.chunks
    assert len(leaves) == sum(sh.chunks for sh in shapes) == 4
    for j, (c, sh, W) in enumerate(zip(calls, shapes, words)):
        a = bytes(x & 0xFF for row in c.A for x in row)
        for R0, C0, BLK in [(0, 0, 0), (16, 8, 0), (0, 0, 1), (64, 296, 0), (48, 128, 1)]:
            if R0 >= sh.m or C0 >= sh.n or 32 * BLK >= sh.S:
                continue
            u, op = c.unit(R0, C0, BLK), L.unit_opening(shapes, j, R0, C0, BLK)
            assert [t for t, _r, _w in op["words"]] == list(range(u.t0 - (1 if BLK else 0), u.t1))
            for t, rank, word in op["words"]:
                assert all(leaves[rank][word + L.fragment(r, col)] == W[t][R0 + r][C0 + col] & L.M32
                           for r in range(16) for col in range(8))
            assert [a[lo:hi] for lo, hi in op["a"]] == [bytes(x & 0xFF for x in c.A[i][u.k0:u.k0 + u.kb])
                                                        for i in range(R0, R0 + u.ROWS)]
            assert op["segments"] == sorted({k // L.A_SEGMENT for lo, hi in op["a"] for k in range(lo, hi)})
    with pytest.raises(ValueError, match="column tile"):
        L.unit_opening([L.Shape(16, 8, 21, 64)], 0, 0, 0, 0)
    with pytest.raises(ValueError, match="call 1: K = 0"):
        L.unit_opening([shapes[0], L.Shape(16, 8, 0, 128)], 0, 0, 0, 0)
    for call in (-1, 3):
        with pytest.raises(ValueError, match="not one of the forward's 3"):
            L.unit_opening(shapes, call, 0, 0, 0)
    with pytest.raises(ValueError, match="holds"):
        L.unit_opening(shapes, 0, 32, 0, 0)


def test_egress_is_bounded_per_forward_under_the_hold():
    """A forward's bits are cap · Σ rows · n · K / fwdMacs: 32 · rows at cap 32 for a whole forward. Under the hold its
    span's buckets are at r = 0 and count nothing, and the boundary's release counts ⌈log2(r + 1)⌉ once."""
    shapes = [L.Shape(5, 4096, 2048, 128), L.Shape(5, 2048, 4096, 128)]
    fwd = sum(s.macs for s in shapes) // 5
    assert not L.egress_refused(32 * 5, shapes, 32, fwd) and L.egress_refused(32 * 5 + 1, shapes, 32, fwd)
    assert L.forward_count([(0, [])] * 1000, 0) == 0
    assert L.forward_count([(0, [])] * 1000, 255) == 8
    assert L.forward_count([(3, [1, 2]), (1, [7])], 3) == 2 + 3 + 1 + 7 + 2


def _window(calls: int, events: list[dict], record: bytes | None = None) -> str | None:
    root, leaf = hashlib.sha512(b"weights").digest(), hashlib.sha512(b"key leaf").digest()
    return L.window_refusal(L.lead_record(root, leaf, calls) if record is None else record, root, leaf, calls, events)


def _a(f: int, j: int, rows: int = 5, cols: int = 64, K: int = 128, bn: int = 128) -> dict:
    return {"kind": "a_root", "forward": f, "call": j, "rows": rows, "cols": cols, "K": K, "bn": bn}


def _w(f: int, verdict: str = L.ACCEPTED) -> dict:
    return {"kind": "words_root", "forward": f, "verdict": verdict}


SALTED = {"kind": "salt"}


@pytest.mark.parametrize("calls, events, record, why", [
    (2, [SALTED, _a(0, 0), _a(0, 1), _w(0), _a(1, 1), _a(1, 0), _w(1)], None, None),
    (2, [SALTED], None, None),
    (2, [SALTED, _a(0, 0), _w(0)], b"\0" * 64, "does not open"),
    (3, [SALTED, _a(0, 0), _w(0)], L.lead_record(hashlib.sha512(b"weights").digest(), hashlib.sha512(b"key leaf").digest(),
                                                 2), "does not open"),
    (2, [_a(0, 0), SALTED, _w(0)], None, "before salt_w"),
    (2, [SALTED, SALTED], None, "second salt_w"),
    (2, [SALTED, _a(0, 0), _a(1, 0), _w(1), _w(0)], None, "overlap"),
    (2, [SALTED, _a(0, 0), _w(0), _a(0, 1), _w(0)], None, "closed"),
    (2, [SALTED, _a(0, 0), _a(0, 0), _w(0)], None, "already"),
    (2, [SALTED, _a(0, 2), _w(0)], None, "no A position"),
    (2, [SALTED, _a(0, 0, K=0), _w(0)], None, "inner dimensions"),
    (2, [SALTED, _a(0, 0, bn=64), _w(0)], None, "column tile"),
    (2, [SALTED, _w(0)], None, "not open"),
    (2, [SALTED, _a(0, 0), _w(0, "past its bits")], None, "past its bits"),
    (2, [SALTED, _a(0, 0)], None, "open when the record ends"),
    (2, [SALTED, {"kind": "tick"}], None, "not a salt"),
    (2, [SALTED, "salt"], None, "not a salt"),
    (2, [SALTED, {**_a(0, 0), "K": "128"}, _w(0)], None, "a_root event's fields are malformed"),
    (2, [SALTED, _a(0, 0), {"kind": "words_root", "forward": 0}], None, "words_root event's fields are malformed"),
    (0, [], b"\0" * 64, "calls a forward"),
])
def test_the_window_refusals(calls, events, record, why):
    got = _window(calls, events, record)
    assert got is None if why is None else why in got
