"""Down's A′ check beside C2-O (`circuit.dither_amax`): the served SiLU·mul restated, a row's record, the two check leaves,
and the down block on bits against the reference, with an amax shrunk or grown refused."""
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


def test_the_check_leaves_on_bits_are_their_words():
    rng = random.Random(6)
    mags = [0, 1, 0x3F80, AM.MAG_MAX - 1, AM.MAG_MAX, AM.MAG_MAX + 1, 0x7FC0, 0x7FFF]
    for h in mags + [m | 0x8000 for m in mags] + [rng.randrange(1 << 16) for _ in range(40)]:
        for amax in mags + [h & 0x7FFF, (h & 0x7FFF) + 1, (h & 0x7FFF) - 1 & 0xFFFF, h | 0x8000]:
            amax &= 0xFFFF
            assert _evaluate(AM.PouwAmaxBound, h, amax) == AM.PouwAmaxBoundWord.evaluate(h, amax)
            for k, n in ((0, 1), (13, 13), (12, 13), (0xFFFF, 14336), (14335, 14336)):
                assert _evaluate(AM.PouwAmaxAt, k, n, h, amax) == AM.PouwAmaxAtWord.evaluate(k, n, h, amax)


def test_a_rows_record_is_its_amax_and_first_argmax():
    hs = [0x3F80, 0xC000, 0x7FC0, 0x4000, 0xBF80]
    assert AM.record_of(hs) == (0x4000, 1) and AM.abs_max(hs) == 0x4000
    assert AM.record_bytes(0x4000, 1) == bytes([0x00, 0x40, 0x01, 0x00]) and len(AM.record_bytes(0, 0)) == AM.RECORD
    assert AM.record_of([0x8000, 0x0000]) == (0, 0)
    with pytest.raises(ValueError, match="every h is NaN"):
        AM.record_of([0x7FC0, 0xFFFF])


K, I = 64, 64


@functools.cache
def _row():
    """Gate-up's row 3 of a served call at S = 1 (n = 2I), its h, record, scales and the down block's honest inputs."""
    sh = O.C2OShape(128, 128, K)
    rng = np.random.default_rng(11)
    A, B = rng.integers(-128, 128, size=(sh.m, K)).tolist(), rng.integers(-128, 128, size=(2 * I, K)).tolist()
    Z = O.outputs(sh, SEEDS, A, B)
    sa = F.bits_of(np.float32(0.0123))
    sw = [F.bits_of(np.float32(x)) for x in rng.uniform(0.001, 0.01, size=2 * I)]
    hs = [AM.act_h(Z[3][j], Z[3][j + I], sa, sw[j], sw[j + I]) for j in range(I)]
    amax, k = AM.record_of(hs)
    da, db = D.dithers(sh, SEEDS, "A", [3])[0], D.dithers(sh, SEEDS, "B", range(2 * I))

    def inputs(k_: int, amax_: int):
        return AM.block_inputs(A[3], da, B[k_ % (2 * I)], B[(k_ + I) % (2 * I)], db[k_ % (2 * I)], db[(k_ + I) % (2 * I)],
                               sa, sw[k_ % (2 * I)], sw[(k_ + I) % (2 * I)], k_, amax_)

    return Z, hs, amax, k, sa, sw, inputs


def test_h_is_the_served_epilogue_then_silu_mul():
    Z, hs, *_, sa, sw, _ = _row()
    assert hs[0] == AM.silu_mul(H.dequant(Z[3][0], sa, sw[0]), H.dequant(Z[3][I], sa, sw[I]))
    assert len(set(hs)) > 1


def test_the_reference_refuses_a_shrunk_or_grown_amax_and_a_k_past_i():
    _, hs, amax, k, *_, inputs = _row()
    assert AM.check_block(inputs(k, amax), I) == 1
    assert AM.check_block(inputs(k, amax - 1), I) == 0 and AM.check_block(inputs(k, amax + 1), I) == 0
    others = [j for j in range(I) if AM.magnitude(hs[j]) != amax]
    assert others and all(AM.check_block(inputs(j, amax), I) == 0 for j in others)
    assert AM.check_block(inputs(k + I, AM.magnitude(hs[k])), I) == 0
    shrunk = AM.magnitude(sorted(hs, key=AM.magnitude)[-2])
    assert AM.PouwAmaxBoundWord.evaluate(hs[k], shrunk) == 0


@pytest.mark.slow
def test_the_down_block_on_bits_is_the_reference():
    """Honest, a shrunk amax, a k past I, and one bit of each port flipped: the Definition agrees with the reference."""
    _, hs, amax, k, *_, inputs = _row()
    d = AM.block_definition(K, I)
    rng = random.Random(9)
    cases = [inputs(k, amax), inputs(k, amax - 1), inputs(k + I, amax)]
    for port, width in enumerate((8, 1, 8, 8, 1, 1, 32, 32, 32, 16, 16)):
        bad = [list(v) for v in inputs(k, amax)]
        bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(width)
        cases.append(bad)
    want = [AM.check_block(c, I) for c in cases]
    assert want[:3] == [1, 0, 0]
    assert [list(evaluate(d, *AM.block_bits(c)))[0] for c in cases] == want


@pytest.mark.parametrize("params", [dict(K=48, I=8), dict(K=D.K_LIMIT, I=8), dict(K=32, I=0), dict(K=32, I=1 << 16),
                                    dict(K=32, I=True)])
def test_the_down_block_refuses_bad_statics(params):
    with pytest.raises(ValueError):
        bind(AM.PouwDitherAmax, **params).signature()
