"""`registry.dead_free`'s versions compute their v1's function word for word, with no dead gate where the v1 has the counts named.

Each new version and its v1 run through the reference evaluator on the same inputs: bf16 and f32 edge words (±0, ±inf, NaNs,
subnormals, the largest finite), random words and normals, rows whose maximum is first, last, tied or behind a NaN, all -inf keys,
and subnormal rows whose sum underflows.  Dead gates are `liveness.dead_gates`, transitive: circuit-check's count.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.program import Program
from verity_flock import ir_lower as IL
from verity_flock.gf2 import Circuit
from verity_vllm.program.registry import b1, dead_free as DF, sampling as SA, topp_words as TW

RNG = np.random.default_rng(20261024)
#: bf16 words at the edges: ±0, ±inf, NaNs, the largest finite, the smallest subnormal, ±1, one ulp above 1
BF16_EDGES = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F7F, 0xFF7F, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F81]
#: the v1s scan no lane at V = 1, and a zero-length scan's carry reads other words, so V = 1 is held to index 0 instead
VOCABS = [2, 3, 16, 257]
TOP_P = [int(np.float32(p).view(np.uint32)) for p in (1.0, 0.95, 0.5, 0.0)]
TEMPS = [int(np.float32(t).view(np.uint32)) for t in (0.0, -0.0, 1.0, 0.8)]
SPLITS = [1, 2, 32, 3]


def _dead(fn) -> int:
    return dead_gates(Program(fn), transitive=True).computed


def _bf16(x) -> list[int]:
    return [int(w) for w in (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16)]


def _rows(n: int, count: int) -> list[list[int]]:
    """Rows of n bf16 words: constant rows of each edge word, edge words mixed, and random words and normals."""
    rows = [[w] * n for w in BF16_EDGES]
    rows += [[int(w) for w in RNG.choice(BF16_EDGES, n)] for _ in range(count)]
    rows += [[int(w) for w in RNG.integers(0, 1 << 16, n)] for _ in range(count)]
    rows += [_bf16(RNG.normal(0, 10.0 ** RNG.integers(-3, 4), n)) for _ in range(count)]
    return rows


def _logit_rows(V: int) -> list[list[int]]:
    """`_rows` and rows whose maximum is first, last, tied at both ends, or behind a NaN."""
    rows = _rows(V, 4)
    for at in ({0}, {V - 1}, {0, V - 1}):
        rows.append([0x3F80 if j in at else 0xBF80 for j in range(V)])
    rows.append([0x7FC0] + [0x3F80] * (V - 1))
    return rows


def _f32_rows(V: int) -> list[list[int]]:
    """f32 rows: each bf16 logit row widened, and random normals."""
    rows = [[int(w) << 16 for w in r] for r in _logit_rows(V)]
    return rows + [[int(np.float32(x).view(np.uint32)) for x in RNG.normal(0, 3, V)] for _ in range(4)]


def _same(new, old, *args) -> None:
    assert evaluate(new, *args) == evaluate(old, *args), args


@pytest.mark.parametrize("N,warps", [(16, 4), (64, 4), (1024, 4), (2048, 8)])
def test_rmsnorm_triton_v5_equals_v1(N, warps):
    new, old = bind(DF.RMSNormTritonV5, N=N, EPS=1e-5), bind(b1.RMSNormTriton, N=N, EPS=1e-5)
    assert (_dead(new), _dead(old)) == (0, 129 * warps)
    rows = _rows(N, 3)
    tiny = [0x0001 + (j % 7) for j in range(N)]          # subnormal squares: the sum underflows to 0 and eps alone is left
    for x, w in zip(rows + [tiny], rows[3:] + rows[:4]):
        _same(new, old, x, w)


@pytest.mark.parametrize("EPS", [0.0, 1e-6, float("inf")])
def test_rmsnorm_triton_v5_equals_v1_at_other_eps(EPS):
    new, old = bind(DF.RMSNormTritonV5, N=64, EPS=EPS), bind(b1.RMSNormTriton, N=64, EPS=EPS)
    rows = _rows(64, 2)
    for x, w in zip(rows, rows[1:] + rows[:1]):
        _same(new, old, x, w)


@pytest.mark.parametrize("V", VOCABS)
def test_token_select_v3_equals_v1(V):
    new, old = bind(DF.TokenSelectV3, V=V), bind(b1.TokenSelect, V=V)
    assert (_dead(new), _dead(old)) == (0, 3)
    for row in _logit_rows(V):
        _same(new, old, row)


def _chunk_rows(V: int) -> list[list[int]]:
    """Rows that turn on `TokenSelect_v4`'s chunks: a maximum tied across each chunk boundary, a NaN at each later chunk's first
    logit ahead of the maximum, and the only number above -inf in the last chunk."""
    rows = []
    for lo, _hi in b1.chunks(V, min(DF.CHUNK, V))[1:]:
        rows.append([0x3F80 if j in (lo - 1, lo) else 0xBF80 for j in range(V)])
        rows.append([0x7FC0 if j == lo else 0x4000 if j == V - 1 else 0x3F80 for j in range(V)])
    return rows + [[0xFF80] * (V - 1) + [0xBF80]]


@pytest.mark.parametrize("V", [*VOCABS, 63, 64, 65, 130, 200])
def test_token_select_v4_equals_v1(V):
    new, old = bind(DF.TokenSelectV4, V=V), bind(b1.TokenSelect, V=V)
    assert (_dead(new), _dead(old)) == (0, 3)
    for row in _logit_rows(V) + _chunk_rows(V):
        _same(new, old, row)


@pytest.mark.parametrize("C", [1, 2, 8, 64])
def test_token_select_chunk_v2_equals_v1(C):
    new, old = bind(DF.TokenSelectChunkV2, C=C), bind(b1.TokenSelectChunk, C=C)
    assert _dead(new) == 0
    for row in _logit_rows(C):
        _same(new, old, row)


@pytest.mark.parametrize("V,C", [(2, 1), (130, 64), (200, 64), (64 * 9, 64)])
def test_token_select_combine_v2_equals_v1(V, C):
    """On partials of random and edge words, their indices anywhere in a chunk."""
    new, old = bind(DF.TokenSelectCombineV2, V=V, C=C), bind(b1.TokenSelectCombine, V=V, C=C)
    assert _dead(new) == 0
    n = len(b1.chunks(V, C))
    for row in _rows(n + 1, 3):
        _same(new, old, row[0], row[1:], [int(i) for i in RNG.integers(0, C, n)])


def _flat_ands(V: int) -> tuple[int, int]:
    """``TokenSelect_v4{V}``'s flat lowering (`ir_lower`): its ANDs and its widest operand."""
    Cc = Circuit()
    IL.lower(Cc, bind(DF.TokenSelectV4, V=V), [Cc.input(f"x{j}", 16) for j in range(V)])
    ops = [len(f.s) for k, d in zip(Cc.kind, Cc.data) if k == "and" for f in d]
    return len(ops) // 2, max(ops)


def test_token_select_v4_lowers_linearly_in_v():
    """Every chunk's ANDs, the combine's included, are the same within 1% at 4, 8 and 16 chunks, and no operand widens: the
    combine's selects keep the running best out of their XOR branch, as the chunk step's do (v1 and v3 lower quadratically)."""
    (a1, w1), (a2, w2), (a3, w3) = (_flat_ands(DF.CHUNK * n) for n in (4, 8, 16))
    assert w1 == w2 == w3
    per = [a1 / 4, a2 / 8, a3 / 16]
    assert max(per) - min(per) < 0.01 * min(per), per


@pytest.mark.parametrize("V", [1, *VOCABS])
def test_topp_mask_v3_equals_v1(V):
    new, old = bind(DF.TopPMaskV3, V=V), bind(SA.TopPMask, V=V)
    assert (_dead(new), _dead(old)) == (0, 2)
    for i, row in enumerate(_f32_rows(V)):
        _same(new, old, row, TOP_P[i % 4], SPLITS[(i // 4) % 4])


@pytest.mark.parametrize("V", VOCABS)
def test_gumbel_select_given_greedy_v2_equals_v1(V):
    new, old = bind(DF.GumbelSelectF32GivenGreedyV2, V=V), bind(SA.GumbelSelectF32GivenGreedy, V=V)
    assert (_dead(new), _dead(old)) == (0, 3)
    for i, row in enumerate(_f32_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        _same(new, old, row, seed, pos, i % 2)


@pytest.mark.parametrize("V", VOCABS)
def test_gumbel_top_p_shared_greedy_v2_equals_v1(V):
    new, old = bind(DF.GumbelTopPTokenSelectSharedGreedyV2, V=V), bind(SA.GumbelTopPTokenSelectSharedGreedy, V=V)
    assert (_dead(new), _dead(old)) == (0, 5)
    for i, row in enumerate(_logit_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        _same(new, old, row, TOP_P[i % 4], seed, pos, TEMPS[(i // 4) % 4], SPLITS[(i // 16) % 4])


@pytest.mark.parametrize("V,S,dead", [(2, 1, 561), (3, 2, 1013), (16, 32, 3787)])
def test_topp_keep_word_v2_equals_v1(V, S, dead):
    """v1's dead gates are lanes of its vector operations that nothing reads after them."""
    new, old = bind(DF.TopPKeepWordV2, V=V, S=S), bind(TW.TopPKeepWord, V=V, S=S)
    assert (_dead(new), _dead(old)) == (0, dead)
    for i, row in enumerate(_f32_rows(V)):
        _same(new, old, row, TOP_P[i % 4])


@pytest.mark.parametrize("V,S", [(3, 2)])
def test_topp_mask_v4_equals_v2(V, S):
    """At the split count S and at two counts it is not, where neither keeps a lane (the sampler's test runs it at S = 32)."""
    new, old = bind(DF.TopPMaskV4, V=V, S=S), bind(TW.TopPMaskV2, V=V, S=S)
    assert _dead(new) == 0 < _dead(old)
    for i, row in enumerate(_f32_rows(V)):
        _same(new, old, row, TOP_P[i % 4], (S, 3, 0)[i % 3])


@pytest.mark.parametrize("V", VOCABS)
def test_gumbel_select_f32_v3_equals_v1(V):
    new, old = bind(DF.GumbelSelectF32V3, V=V), bind(SA.GumbelSelectF32, V=V)
    assert (_dead(new), _dead(old)) == (0, 3)
    for i, row in enumerate(_f32_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        _same(new, old, row, seed, pos, TEMPS[i % 4])


@pytest.mark.parametrize("V,S,dead", [(3, 2, 1016), (16, 32, 3790)])
def test_gumbel_top_p_v3_equals_v2(V, S, dead):
    """The registered compiler's sampler (`prune.dead_free_versions`): v2's dead gates are its keep word's and the Gumbel scan's
    last 3."""
    new, old = bind(DF.GumbelTopPTokenSelectV3, V=V, S=S), bind(TW.GumbelTopPTokenSelectV2, V=V, S=S)
    assert (_dead(new), _dead(old)) == (0, dead)
    for i, row in enumerate(_logit_rows(V)):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        _same(new, old, row, TOP_P[i % 4], seed, pos, TEMPS[(i // 4) % 4], (S, S, 3)[(i // 16) % 3])


def test_one_logit_selects_index_0():
    for fn in (DF.TokenSelectV3, DF.TokenSelectV4, DF.GumbelSelectF32GivenGreedyV2, DF.GumbelTopPTokenSelectSharedGreedyV2,
               DF.GumbelSelectF32V3):
        assert _dead(bind(fn, V=1)) == 0
    assert _dead(bind(DF.GumbelTopPTokenSelectV3, V=1, S=32)) == 0
    for (w,) in _logit_rows(1):
        seed, pos = int(RNG.integers(0, 1 << 63)), int(RNG.integers(0, 1 << 20))
        assert evaluate(bind(DF.TokenSelectV3, V=1), [w]) == (0,) == evaluate(bind(DF.TokenSelectV4, V=1), [w])
        for greedy in (0, 1):
            assert evaluate(bind(DF.GumbelSelectF32GivenGreedyV2, V=1), [w << 16], seed, pos, greedy) == (0,)
        for temp in TEMPS:
            assert evaluate(bind(DF.GumbelTopPTokenSelectSharedGreedyV2, V=1), [w], TOP_P[1], seed, pos, temp, 32) == (0,)
            assert evaluate(bind(DF.GumbelSelectF32V3, V=1), [w << 16], seed, pos, temp) == (0,)
            assert evaluate(bind(DF.GumbelTopPTokenSelectV3, V=1, S=32), [w], TOP_P[1], seed, pos, temp, 32) == (0,)
