"""The per-segment spot check's pieces (`verity.core.protocols.pouw.circuit.segment`) are their word primitives bit for bit on
each operand's edges and seeded random words, and `PouwSegCheck_v1{KS=4}` and `PouwSegCheckDither_v1{KS=4}` return what
`segment.seg_check` does: 1 on honest instances at every segment, 0 with one served-operand bit flipped, a receipt word
changed, or a nonzero prefix at the first segment."""
from __future__ import annotations

import numpy as np
import pytest

from verity.core.protocols.pouw.circuit import segment as SG
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_words

KS, N = 4, 64


def _rnd(rng, shape, bits):
    return rng.integers(0, 1 << bits, size=shape, dtype=np.uint64)


def _pieces(rng):
    edges8 = np.array([0, 1, 0x7F, 0x80, 0x81, 0xFE, 0xFF] * 2, dtype=np.uint64)
    bit = np.array([0] * 7 + [1] * 7, dtype=np.uint64)
    def cat(e, r):
        return np.concatenate([e, r])

    ok = [_rnd(rng, (N,), 32) for _ in range(4)] + [_rnd(rng, (N, 32), 32), _rnd(rng, (N,), 1)]
    for i in range(0, N, 3):
        ok[3][i], ok[2][i] = ok[1][i], ok[0][i]
        if i % 2:
            ok[4][i] = 0
    return {SG.PouwI8Add: [cat(edges8, _rnd(rng, (N,), 8)), cat(edges8[::-1], _rnd(rng, (N,), 8))],
            SG.PouwI8Pm1: [cat(edges8, _rnd(rng, (N,), 8)), cat(bit, _rnd(rng, (N,), 1))],
            SG.PouwI8AddBit: [cat(edges8, _rnd(rng, (N,), 8)), cat(bit, _rnd(rng, (N,), 1))],
            SG.PouwSel32: [_rnd(rng, (N,), 1), _rnd(rng, (N,), 32)],
            SG.PouwSegDot: [_rnd(rng, (N,), 32), _rnd(rng, (N, 32), 8), _rnd(rng, (N, 32), 8)],
            SG.PouwDuoSeg: [_rnd(rng, (N, 8), 32), _rnd(rng, (N, 8), 32), _rnd(rng, (N, 32), 32)],
            SG.PouwSegOk: ok}


def test_each_piece_is_its_word_primitive():
    for d, args in _pieces(np.random.default_rng(7)).items():
        got = np.asarray(evaluate_words(bind(d), *args)).reshape(-1)
        want = np.asarray(evaluate_batch(d.word, *args)).reshape(-1)
        assert (got == want).all(), (d.id, int((got != want).sum()))


def _instance(rng, j, dither, fault):
    a = rng.integers(-63, 64, (4, 32)) & 0xFF
    u = rng.integers(1, 65, (4, 32)) * rng.choice([-1, 1], (4, 32)) & 0xFF
    b = rng.integers(-119, 120, (8, 32)) & 0xFF
    f = rng.integers(0, 2, (8, 32))
    prev = [0] * 32 if j == 0 and fault != "prefix" else [int(v) for v in rng.integers(0, 1 << 32, 32)]
    st = [[w for half in SG.init(0x1234567, 7, 99, s, KS) for w in half] for s in (max(j - 1, 0), j)]
    bp = (b + (f if dither else 1 - 2 * f)) & 0xFF
    tj = [SG.seg_dot(prev[q], (a + u)[SG.row_slot(q)] & 0xFF, bp[SG.col_slot(q)]) for q in range(32)]
    rc = [int(v) for v in rng.integers(0, 1 << 32, KS)]
    rc[j] = SG.duo_seg(st[1][:8], st[1][8:], tj)
    if j:
        rc[j - 1] = SG.duo_seg(st[0][:8], st[0][8:], prev)
    if fault == "operand":
        b[rng.integers(8), rng.integers(32)] ^= 1
    elif fault == "receipt":
        rc[j - 1 if j else j] ^= 1 << int(rng.integers(32))
    sel = [int(s == j) for s in range(KS)]
    want = SG.seg_check(a.tolist(), u.tolist(), b.tolist(), f.tolist(), prev, st, rc, sel, dither=dither)
    fw = np.array([sum(int(f[c][k]) << k for k in range(32)) for c in range(8)])
    return [a, u, b, fw, np.array(prev), np.array(st), np.array(rc), np.array(sum(s << i for i, s in enumerate(sel)))], want


@pytest.mark.parametrize("definition,dither", [(SG.PouwSegCheck, False), (SG.PouwSegCheckDither, True)])
def test_the_check_accepts_honest_segments_and_refuses_faults(definition, dither):
    rng = np.random.default_rng(11 + dither)
    cases = [(j, None) for j in range(KS)] + [(j, "operand") for j in range(KS)] + [(1, "receipt"), (0, "receipt"),
                                                                                      (0, "prefix")]
    insts = [_instance(rng, j, dither, fault) for j, fault in cases]
    want = [w for _x, w in insts]
    assert want == [1] * KS + [0] * (len(cases) - KS)
    args = [np.stack([x[p] for x, _w in insts]).astype(np.uint64) for p in range(8)]
    got = np.asarray(evaluate_words(bind(definition, KS=KS), *args)).reshape(-1).tolist()
    assert got == want
