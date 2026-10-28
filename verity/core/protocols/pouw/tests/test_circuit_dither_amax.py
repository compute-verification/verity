"""Down's A′ check beside C2-O (`circuit.dither_amax`): the served SiLU·mul restated, the max leaves, a gate block's max and
the down block's amax on bits against the reference, with a block max or amax shrunk or grown refused."""
from __future__ import annotations

import functools
import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops import fp32 as F
from verity.ml.circuits.ops.prims import BlackwellS8ImmaDot32  # noqa: F401  (registers the s8 steps' families)
from verity.ml.definitions.registry import act
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import dither_amax as AM
from verity.core.protocols.pouw.circuit import dither_c2o as O
from verity.core.protocols.pouw.circuit import h1zy as H

SEEDS = D.seeds(hashlib.sha512(b"call coin").digest(), hashlib.sha512(b"operand-root receipt").digest(), 0)
EDGES = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3F80, 0xBF80, 0x0001, 0x8001, 0xC2B2, 0x42B2, 0xC2B4]


def _evaluate(d, *ws: int) -> int:
    params, _ = d.signature({})
    bits = [[(w >> e) & 1 for e in range(t.n if hasattr(t, 'n') else 1)] for w, (_, t) in zip(ws, params)]
    out = list(evaluate(d, *bits))
    return sum(b << e for e, b in enumerate(out))


def test_the_silu_mul_word_is_the_served_one_on_every_gate():
    """`PouwSiluMulBf16_v1` is `SiluMulBf16_v2` (`registry.act`) on all 65,536 gate words at u = 1, and on the edge words
    against each other."""
    assert all(AM.silu_mul(g, AM.ONE_BF16) == act.SiluMulBf16.evaluate(g, AM.ONE_BF16) for g in range(1 << 16))
    rng = random.Random(2)
    pairs = [(g, u) for g in EDGES for u in EDGES] + [(rng.randrange(1 << 16), rng.randrange(1 << 16)) for _ in range(2000)]
    assert all(AM.silu_mul(g, u) == act.SiluMulBf16.evaluate(g, u) for g, u in pairs)


@pytest.mark.slow
def test_silu_mul_on_bits_is_its_word():
    """On the band, either side of it, at every edge word, and at random words."""
    rng = random.Random(4)
    gs = EDGES + [s << 15 | e << 7 | rng.randrange(128) for s in (0, 1) for e in range(110, 140)]
    gs += [rng.randrange(1 << 16) for _ in range(64)]
    for g in gs:
        u = rng.choice(EDGES + [rng.randrange(1 << 16)])
        assert _evaluate(AM.PouwSiluMulBf16, g, u) == AM.silu_mul(g, u), (hex(g), hex(u))


def test_the_max_leaves_on_bits_are_their_words():
    rng = random.Random(6)
    mags = [0, 1, 0x3F80, AM.MAG_MAX - 1, AM.MAG_MAX, AM.MAG_MAX + 1, 0x7FC0, 0x7FFF]
    words = mags + [m | 0x8000 for m in mags] + [rng.randrange(1 << 16) for _ in range(40)]
    for h in words:
        for acc in mags + [h & 0x7FFF, (h & 0x7FFF) + 1, (h & 0x7FFF) - 1 & 0xFFFF, h | 0x8000, 0xFFFF]:
            acc &= 0xFFFF
            assert _evaluate(AM.PouwAmaxFold, acc, h) == AM.PouwAmaxFoldWord.evaluate(acc, h)
            assert _evaluate(AM.PouwAmaxMax, acc, h) == max(acc, h)
            assert _evaluate(AM.PouwAmaxEq, acc, h) == int(acc == h)


def test_a_blocks_max_is_the_served_fold():
    hs = [0x3F80, 0xC000, 0x7FC0, 0x4000, 0xBF80, 0xFF80, 0x0001, 0x8000]
    assert AM.abs_max(hs) == 0x7F80 and AM.block_maxes(hs, 4) == [0x4000, 0x7F80] and AM.block_maxes(hs, 8) == [0x7F80]
    assert AM.abs_max([0x7FC0, 0xFFFF]) == 0 and AM.check_gate([0x7FC0, 0xFFFF], 0) == 1
    with pytest.raises(ValueError, match="tile"):
        AM.block_maxes(hs, 3)


K, I, W = 64, 64, 8


@functools.cache
def _row():
    """Gate-up's row 3 of a served call at S = 1 (n = 2I), its h and its scales."""
    sh = O.C2OShape(128, 128, K)
    rng = np.random.default_rng(11)
    A, B = rng.integers(-128, 128, size=(sh.m, K)).tolist(), rng.integers(-128, 128, size=(2 * I, K)).tolist()
    Z = O.outputs(sh, SEEDS, A, B)
    sa = F.bits_of(np.float32(0.0123))
    sw = [F.bits_of(np.float32(x)) for x in rng.uniform(0.001, 0.01, size=2 * I)]
    hs = [AM.act_h(Z[3][j], Z[3][j + I], sa, sw[j], sw[j + I]) for j in range(I)]
    return Z, hs, sa, sw


def test_h_is_the_served_epilogue_then_silu_mul():
    Z, hs, sa, sw = _row()
    assert hs[0] == AM.silu_mul(H.dequant(Z[3][0], sa, sw[0]), H.dequant(Z[3][I], sa, sw[I]))
    assert len(set(hs)) > 1


def test_a_wrong_amax_passes_down_only_beside_a_wrong_block_max():
    _, hs, *_ = _row()
    ms, amax = AM.block_maxes(hs, W), AM.abs_max(hs)
    assert len(ms) == I // W and max(ms) == amax and AM.check_row(amax, ms) == 1
    assert all(AM.check_gate(hs[b * W:(b + 1) * W], m) == 1 for b, m in enumerate(ms))
    assert AM.check_row(amax - 1, ms) == 0 and AM.check_row(amax + 1, ms) == 0
    top = ms.index(amax)
    for wrong in (amax - 1, amax + 1):
        forged = ms[:top] + [wrong] + ms[top + 1:]
        assert AM.check_row(max(forged), forged) == 1 and AM.check_gate(hs[top * W:(top + 1) * W], wrong) == 0


def test_the_gate_and_down_checks_on_bits_are_the_reference():
    """Honest, each block max and the amax shrunk and grown, one h bit flipped in each cell, and an all-NaN block."""
    _, hs, *_ = _row()
    ms, amax = AM.block_maxes(hs, W), AM.abs_max(hs)
    gate, row = AM.gate_definition(W), AM.row_definition(I, W)
    rng = random.Random(9)
    cases = [(hs[b * W:(b + 1) * W], m + e & 0xFFFF) for b, m in enumerate(ms) for e in (0, -1, 1)]
    for b in range(I // W):
        cell = list(hs[b * W:(b + 1) * W])
        cell[rng.randrange(W)] ^= 1 << rng.randrange(16)
        cases.append((cell, ms[b]))
    cases.append(([0x7FC0, 0xFFFF, 0x7F81, 0xFFC1, 0x7FFF, 0xFF81, 0x7FC0, 0xFFFF], 0))
    want = [AM.check_gate(h, m) for h, m in cases]
    assert sum(want) >= I // W + 1 and 0 in want
    assert [list(evaluate(gate, AM.bits(h), AM.bits([m])))[0] for h, m in cases] == want
    rows = [(amax + e & 0xFFFF, ms) for e in (0, -1, 1)] + [(amax, ms[:b] + [ms[b] ^ 1 << rng.randrange(16)] + ms[b + 1:])
                                                             for b in range(0, I // W, 3)]
    want = [AM.check_row(a, m) for a, m in rows]
    assert want[:3] == [1, 0, 0]
    assert [list(evaluate(row, AM.bits([a]), AM.bits(m)))[0] for a, m in rows] == want


@pytest.mark.parametrize("make", [lambda: AM.gate_definition(0).signature(), lambda: AM.gate_definition(1 << 16).signature(),
                                  lambda: bind(AM.PouwAmaxRow, N=True).signature(), lambda: AM.row_definition(64, 0),
                                  lambda: AM.row_definition(64, 7), lambda: AM.row_definition(1 << 16, 1).signature()])
def test_the_checks_refuse_bad_statics(make):
    with pytest.raises(ValueError):
        make()
