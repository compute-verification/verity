"""The checkpointed spot check's pieces (`verity.core.protocols.pouw.circuit.segment`) are their word primitives bit for bit on
each operand's edges and seeded random words, and `PouwSegCheck_v2{S=3}` (duo and XOR-in) and `PouwSegCheckDither_v2{S=3}`
return what `segment.win_check` does: 1 on honest instances at every position of a first and a later window, 0 with a
selected operand bit flipped, the window's receipt word changed, T_j off by one, or a nonzero incoming set in the first
window; a flipped bit in an operand row the thread doesn't read changes nothing."""
from __future__ import annotations

import numpy as np
import pytest

from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_words

S, N = 3, 64


def _rnd(rng, shape, bits):
    return rng.integers(0, 1 << bits, size=shape, dtype=np.uint64)


def _wide(rng, words):
    return [sum(int(w) << (32 * i) for i, w in enumerate(rng.integers(0, 1 << 32, words))) for _ in range(N)]


def _pieces(rng):
    edges8 = np.array([0, 1, 0x7F, 0x80, 0x81, 0xFE, 0xFF] * 2, dtype=np.uint64)
    bit = np.array([0] * 7 + [1] * 7, dtype=np.uint64)

    def cat(e, r):
        return np.concatenate([e, r])

    ok = [_rnd(rng, (N,), 32) for _ in range(2)] + [_rnd(rng, (N, 32), 32) for _ in range(3)] + [_rnd(rng, (N,), 1)]
    for i in range(0, N, 3):
        ok[1][i], ok[3][i] = ok[0][i], ok[2][i]
        if i % 2:
            ok[4][i] = 0
    return {SG.PouwI8Add: [cat(edges8, _rnd(rng, (N,), 8)), cat(edges8[::-1], _rnd(rng, (N,), 8))],
            SG.PouwI8Pm1: [cat(edges8, _rnd(rng, (N,), 8)), cat(bit, _rnd(rng, (N,), 1))],
            SG.PouwI8AddBit: [cat(edges8, _rnd(rng, (N,), 8)), cat(bit, _rnd(rng, (N,), 1))],
            SG.PouwSel8: [_rnd(rng, (N,), 1), _rnd(rng, (N,), 8)],
            SG.PouwXor8: [_rnd(rng, (N,), 8), _rnd(rng, (N,), 8)],
            SG.PouwSel32: [_rnd(rng, (N,), 1), _rnd(rng, (N,), 32)],
            SG.PouwSegDot: [_rnd(rng, (N,), 32), _rnd(rng, (N, 32), 8), _rnd(rng, (N, 32), 8)],
            SG.PouwWinOk: ok}


def test_each_piece_is_its_word_primitive():
    for d, args in _pieces(np.random.default_rng(7)).items():
        got = np.asarray(evaluate_words(bind(d), *args)).reshape(-1)
        want = np.asarray(evaluate_batch(d.word, *args)).reshape(-1)
        assert (got == want).all(), (d.id, int((got != want).sum()))


@pytest.mark.parametrize("d,words", [(SG.PouwDuoAbsorb, 16), (SG.PouwDuoFinal, 16), (SG.PouwXinAbsorb, 8),
                                     (SG.PouwXinFinal, 8)])
def test_each_fold_piece_is_its_word_primitive(d, words):
    """The fold's state is one wide word, past `evaluate_words`' 64 bits, so these compare on bits."""
    from verity.ml.circuits.evaluation.bits import evaluate_bits

    rng = np.random.default_rng(words)
    st = _wide(rng, words)
    p = rng.integers(0, 1 << 32, (N, 32))
    absorb = d in (SG.PouwDuoAbsorb, SG.PouwXinAbsorb)
    args = [np.array([[(v >> i) & 1 for i in range(32 * words)] for v in st], dtype=np.uint8)]
    if absorb:
        args.append(((p[:, :, None] >> np.arange(32)) & 1).astype(np.uint8))
    got = evaluate_bits(bind(d), *args).reshape(N, -1)
    for n in range(N):
        want = d.word.evaluate(st[n], *[int(w) for w in p[n]]) if absorb else d.word.evaluate(st[n])
        assert sum(int(b) << i for i, b in enumerate(got[n])) == want, (d.id, n)


def _instance(rng, fold, dither, w, i, fault):
    """Thread 99 of tile 7 at segment w S + i, its window w of NW = 4."""
    a = rng.integers(-63, 64, (SG.RL, 32)) & 0xFF
    b = rng.integers(-119, 120, (SG.RL, 32)) & 0xFF
    ia, ib = int(rng.integers(16)), int(rng.integers(8))
    u = rng.integers(1, 65, (4, 32)) * rng.choice([-1, 1], (4, 32)) & 0xFF
    f = rng.integers(0, 2, (8, 32))
    ar = np.array([a[SG.a_offset(ia) + 8 * r] for r in range(4)])
    br = np.array([b[SG.b_offset(ib) + 8 * (c >> 1) + (c & 1)] for c in range(8)])
    ap, bp = (ar + u) & 0xFF, (br + (f if dither else 1 - 2 * f)) & 0xFF
    p = [[int(v) for v in rng.integers(0, 1 << 32, 32)] for _ in range(S + 1)]
    if w == 0 and fault != "prefix":
        p[0] = [0] * 32
    p[i + 1] = [SG.seg_dot(p[i][q], ap[SG.row_slot(q)], bp[SG.col_slot(q)]) for q in range(32)]
    if fault == "product":
        p[i + 1][int(rng.integers(32))] += 1
    st = SG.fold_keys(fold, 0x1234567, 7, 99, w, 4)
    state = list(st)
    for ps in p:
        state = SG.fold_absorb(fold, state, ps)
    pos = int(rng.integers(SG.RW))
    rc = [int(v) for v in rng.integers(0, 1 << 32, SG.RW)]
    rc[pos] = SG.fold_final(fold, state)
    if fault == "operand":
        b[SG.b_offset(ib) + 8 * int(rng.integers(4)) + int(rng.integers(2)), rng.integers(32)] ^= 1
    elif fault == "unread":
        a[SG.a_offset((ia + 1) % 16), rng.integers(32)] ^= 1
    elif fault == "receipt":
        rc[pos] ^= 1 << int(rng.integers(32))
    asel, bsel, rsel, jsel = ([int(k == x) for k in range(n)] for x, n in ((ia, 16), (ib, 8), (pos, SG.RW), (i, S)))
    first = int(w == 0)
    x = [a, asel, u, b, bsel, f, p, st, rc, rsel, jsel, first]
    want = SG.win_check(*[v.tolist() if isinstance(v, np.ndarray) else v for v in x], fold=fold, dither=dither)
    def pack(bits):
        return sum(int(v) << k for k, v in enumerate(bits))

    x[1], x[4], x[10] = pack(asel), pack(bsel), pack(jsel)
    x[5] = [pack(f[c]) for c in range(8)]
    x[9] = [pack(rsel[32 * k:32 * k + 32]) for k in range(SG.RW // 32)]
    return [np.asarray(v) for v in x], want


@pytest.mark.parametrize("definition,fold,dither", [(SG.PouwSegCheck, "duo", False), (SG.PouwSegCheck, "xin", False),
                                                    (SG.PouwSegCheckDither, "duo", True)])
def test_the_check_accepts_honest_windows_and_refuses_faults(definition, fold, dither):
    rng = np.random.default_rng(11 + 2 * dither + (fold == "xin"))
    honest = [(w, i, None) for w in (0, 2) for i in range(S)] + [(1, 1, "unread")]
    faults = [(w, i, k) for w, i in ((0, 0), (2, 2)) for k in ("operand", "receipt", "product")] + [(0, 1, "prefix")]
    insts = [_instance(rng, fold, dither, w, i, k) for w, i, k in honest + faults]
    want = [v for _x, v in insts]
    assert want == [1] * len(honest) + [0] * len(faults)
    args = [np.stack([x[p] for x, _v in insts]).astype(np.uint64) for p in range(12)]
    got = np.asarray(evaluate_words(bind(definition, S=S, FOLD=fold), *args)).reshape(-1).tolist()
    assert got == want
