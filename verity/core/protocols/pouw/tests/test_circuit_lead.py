"""`circuit.lead`: each piece is its word and has its stated ANDs, the reference's words are Layout.lean's `word`, the unit
accepts a call's committed words and rejects a changed word, a tag weight outside [-3, 3], an even lane coin and a fold past
2^22, a ragged call's padding rows and columns carry their coins, a unit's ANDs are `unit_ands`, its output is one bit, and
a binding past the calls the verifier admits is refused."""
from __future__ import annotations

import copy
import random

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
         L.PouwLeadPair32: L.PouwLeadPair32Word, L.PouwLeadFold: L.PouwLeadFoldWord}
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
            sigma = rng.randrange(1 << 32)
            p = rng.choice([0, L.FOLD_RANGE, -L.FOLD_RANGE, L.FOLD_RANGE + 1, -L.FOLD_RANGE - 1, 1 << 31,
                            rng.randrange(-(1 << 31), 1 << 31)])
            out.append([(p + sigma) & L.M32, sigma])
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
    sigma = (127 * -381 + 7 - 9) & L.M32
    for p, ok in [(0, 1), (L.FOLD_RANGE, 1), (-L.FOLD_RANGE, 1), (L.FOLD_RANGE + 1, 0), (-L.FOLD_RANGE - 1, 0)]:
        assert evaluate(L.PouwLeadFoldWord, (p + sigma) & L.M32, sigma) == (ok,)
    assert [G for G in range(256) if evaluate(L.PouwLeadTagWeightWord, G) == (1,)] == [0, 1, 2, 3, 253, 254, 255]


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


@pytest.mark.parametrize("K, a, ok", [(245, 127, 1), (301, 127, 0), (301, -128, 0)])
def test_the_fold_past_its_window_is_a_visible_failure(K, a, ok):
    """All-127 B against all-a A: |p| is about 127 · |a| · K, within 2^22 at K = 245 and past it at K = 301."""
    c = L.Call.draw(random.Random(K), 16, 8, K)
    c.A, c.B = [[a] * K for _ in range(16)], [[127] * K for _ in range(8)]
    W = c.words()
    sigma = lambda i, j: sum(L.rot_weight(i, ell) * c.s(t, i, ell, j) for t in range(c.S) for ell in range(3))  # noqa: E731
    within = all(abs(W[-1][i][j] - sigma(i, j)) <= L.FOLD_RANGE for i in range(16) for j in range(8))
    assert within == bool(ok)
    assert c.unit_ok(0, 0, 0, W) == ok


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
