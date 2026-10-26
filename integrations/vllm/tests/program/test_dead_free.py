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
from verity_vllm.program.registry import b1, dead_free as DF, fa2_check_inf as F2, hopper, prims as P, sampling as SA, topp_words as TW

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


FA2 = dict(DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum)


def _attention_cases(D: int, T: int) -> list[tuple[list[int], list[int], list[int]]]:
    """(q, K rows, V rows): `_rows` cut to a query and T keys and values, random normals, and all -inf keys."""
    cases = [(row[:D], row[:T * D], row[-T * D:]) for row in _rows(2 * T * D, 1)]
    cases += [(_bf16(RNG.normal(0, 1, D)), _bf16(RNG.normal(0, 1, T * D)), _bf16(RNG.normal(0, 1, T * D))) for _ in range(3)]
    return cases + [(_bf16(RNG.normal(0, 1, D)), [0xFF80] * (T * D), _bf16(RNG.normal(0, 1, T * D)))]


@pytest.mark.parametrize("T,BN,MF,dead", [(16, 32, 0, 2), (48, 32, 1, 2), (48, 32, 0, 2), (40, 32, 1, 0), (1, 32, 0, 0),
                                          (32, 32, 0, 0), (17, 16, 1, 0)])
def test_attention_head_v18_equals_v5(T, BN, MF, dead):
    """v5's 2 dead gates a head (its first block's zero row and P's zero word) where that block's key count is a multiple of 16 below
    BN; v18 equal to v5 there and wherever v5 has none."""
    new, old = bind(DF.AttentionHeadV18, T=T, D=16, BN=BN, MASKED_FROM=MF, **FA2), bind(F2.AttentionHeadV5, T=T, D=16, BN=BN, MASKED_FROM=MF, **FA2)
    assert (_dead(new), _dead(old)) == (0, dead)
    for q, k, v in _attention_cases(16, T):
        _same(new, old, q, k, v)


def test_attention_v18_equals_v5():
    new, old = (bind(fn, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, **FA2) for fn in (DF.AttentionV18, F2.AttentionV5))
    assert (_dead(new), _dead(old)) == (0, 2 * 4)
    for _ in range(2):
        q, kc, vc = (_bf16(RNG.normal(0, 1, n)) for n in (4 * 16, 48 * 2 * 16, 48 * 2 * 16))
        _same(new, old, q, kc, vc)


@pytest.mark.parametrize("T", [16, 64, 112, 144, 65])
def test_attention_v18_has_no_dead_gate_at_llama_geometry(T):
    """Llama-3.2-1B's FA2 rows on sm_120 (32 query heads over 8 KV heads of 64, kBlockN 128): v5 has 64 dead gates a row whose first
    block's key count is a multiple of 16 below 128."""
    new, old = (bind(fn, T=T, NH=32, KVH=8, D=64, BN=128, MASKED_FROM=0, **FA2) for fn in (DF.AttentionV18, F2.AttentionV5))
    assert (_dead(new), _dead(old)) == (0, 0 if T % 16 else 64)


def _inlined(fn, inline: frozenset[str] = frozenset()) -> tuple[list[tuple], list[tuple]]:
    """``fn``'s nodes with the Calls of the families ``inline`` names written in place, each (form, Definition id, axes, n, n_xs,
    operands), every operand leaf a parameter leaf of ``fn`` or an output leaf of an earlier node here; and the return's leaves.
    Equal for two Definitions, they are one circuit, so one function on every input."""
    nodes: list[tuple] = []

    def run(body, params: list) -> list:
        outs: list = []

        def src(r):
            s, i, j = r
            return params[i][j] if s == "p" else outs[i][j]

        for node in body.nodes:
            args = [tuple(map(src, a.refs)) for a in node.args]
            if node.form == "call" and node.fn.base_id in inline:
                outs.append(run(node.fn.body, args))
            else:
                nodes.append((node.form, node.fn.id, node.axes, node.n, node.n_xs, args))
                outs.append([("n", len(nodes) - 1, j) for j in range(node.out.leaves)])
        return [src(r) for r in body.ret]

    ret = run(fn.body, [[("p", i, j) for j in range(t.leaves)] for i, (_, t) in enumerate(fn.body.params)])
    return nodes, ret


V19_STEPS = frozenset({"AttnBlockFirst_v1", "AttnHeadEnd_v1"})


@pytest.mark.parametrize("T,BN,MF", [(1, 32, 0), (15, 32, 0), (16, 32, 0), (17, 32, 0), (32, 32, 0), (40, 32, 1), (48, 32, 0),
                                     (48, 32, 1), (96, 32, 0), (96, 32, 2), (17, 16, 1)])
def test_attention_head_v19_equals_v18(T, BN, MF):
    """A first block of 1 ... BN keys, its V rows padded (40) or not (48), and blocks below it guarded or not."""
    new, old = (bind(fn, T=T, D=16, BN=BN, MASKED_FROM=MF, **FA2) for fn in (DF.AttentionHeadV19, DF.AttentionHeadV18))
    assert _dead(new) == 0
    for q, k, v in _attention_cases(16, T):
        _same(new, old, q, k, v)


def test_attention_v19_equals_v18():
    new, old = (bind(fn, T=48, NH=4, KVH=2, D=16, BN=32, MASKED_FROM=1, **FA2) for fn in (DF.AttentionV19, DF.AttentionV18))
    assert _dead(new) == 0
    for _ in range(2):
        q, kc, vc = (_bf16(RNG.normal(0, 1, n)) for n in (4 * 16, 48 * 2 * 16, 48 * 2 * 16))
        _same(new, old, q, kc, vc)


@pytest.mark.parametrize("D,BN", [(64, 128), (128, 64)])
@pytest.mark.parametrize("T", [1, 15, 16, 17, 64, 127, 128, 129, 200, 256, 385])
def test_attention_head_v19_is_v18_with_its_calls_inlined_at_llama_geometry(D, BN, T):
    """Llama-3.2-1B's heads (D 64, kBlockN 128) and Llama-3.1-8B's (D 128, kBlockN 64): v19's body is Calls only, and with
    `AttnBlockFirst_v1` and `AttnHeadEnd_v1` written in place it is v18's circuit node for node, at every MASKED_FROM's extremes."""
    nb = -(-T // BN)
    for MF in {0, nb - 1}:
        new, old = (bind(fn, T=T, D=D, BN=BN, MASKED_FROM=MF, **FA2) for fn in (DF.AttentionHeadV19, DF.AttentionHeadV18))
        assert {(n.form, n.fn.base_id) for n in new.body.nodes} <= {("call", f) for f in (*V19_STEPS, "AttnBlock_v5")}
        assert len(new.body.nodes) == nb + 1
        assert _inlined(new, V19_STEPS) == _inlined(old)
