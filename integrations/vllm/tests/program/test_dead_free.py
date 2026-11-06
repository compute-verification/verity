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
from verity.ml.windows import interface as IF
from verity_flock import circuit as FC, class_statement as CS, ir_lower as IL
from verity_flock.gf2 import Circuit
from verity.ml.definitions.registry import b1, dead_free as DF, fa2_check_inf as F2, hopper, prims as P, sampling as SA, topp_words as TW

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


ONE, NEG_ONE, NAN, NEG_INF = 0x3F80, 0xBF80, 0x7FC0, b1.NEG_INF_BF16
#: Llama-3's vocabulary as 48 units of `TokenSelectGroups_v1{C=2672, G=16}`: 167 groups a unit, a group one LM-head slice of
#: `Gemm_v2{K=2048, N=16}`, 8,016 in all for `TokenSelectCombine_v2{V=128256, C=16}`
LLAMA_V, LLAMA_C, LLAMA_G = 128256, 2672, 16


def _boundary_rows(V: int, bounds: list[int]) -> list[list[int]]:
    """Rows that turn on boundaries: at each, a maximum tied across it, +0 and -0 tied across it both ways round, and a NaN at
    its first logit ahead of the maximum; a maximum at both ends of the row; a NaN logits[0] (positive and negative) and a -inf
    one; and rows of only -inf, of -inf and NaNs, and of -inf with one number last."""
    rows = []
    for b in bounds:
        rows.append([ONE if j in (b - 1, b) else NEG_ONE for j in range(V)])
        for z0, z1 in ((0x0000, 0x8000), (0x8000, 0x0000)):
            rows.append([z0 if j == b - 1 else z1 if j == b else NEG_ONE for j in range(V)])
        rows.append([NAN if j == b else 0x4000 if j == V - 1 else ONE for j in range(V)])
    rows.append([ONE if j in (0, V - 1) else NEG_ONE for j in range(V)])
    rows += [[x0] + [ONE] * (V - 1) for x0 in (NAN, 0xFFC1, NEG_INF)]
    return rows + [[NEG_INF] * V, [NAN if j % 2 else NEG_INF for j in range(V)], [NEG_INF] * (V - 1) + [NEG_ONE]]


@pytest.mark.parametrize("C,G", [(1, 1), (16, 16), (64, 16), (40, 16), (7, 3), (28, 1)])
def test_token_select_groups_v1_is_each_groups_chunk(C, G):
    """A batch of `TokenSelectChunk_v2{G}` over the whole groups, and a narrower chunk over the rest: each group's (best, idx)."""
    fn = bind(DF.TokenSelectGroupsV1, C=C, G=G)
    assert _dead(fn) == 0
    for row in _logit_rows(C) + _boundary_rows(C, list(range(G, C, G))):
        parts = [evaluate(bind(DF.TokenSelectChunkV2, C=hi - lo), row[lo:hi]) for lo, hi in b1.chunks(C, G)]
        assert evaluate(fn, row) == tuple(p[0] for p in parts) + tuple(p[1] for p in parts), row


def test_token_select_groups_v1_refuses_groups_wider_than_its_logits():
    with pytest.raises(ValueError, match="groups of G = 17 of C = 16 logits"):
        bind(DF.TokenSelectGroupsV1, C=16, G=17).signature()


def _grouped(V: int, C: int, G: int, row: list[int], ev=evaluate) -> tuple[int]:
    """``row``'s token by groups: `TokenSelectGroups_v1{C, G}` over each C logits (the last unit the rest, one group if that is
    narrower than G), then `TokenSelectCombine_v2{V, G}` over logits[0] and every group's partial."""
    assert C % G == 0, "a unit's groups are the combine's chunks of G only when every unit but the last is whole groups"
    best, idx = [], []
    for lo, hi in b1.chunks(V, C):
        out = ev(bind(DF.TokenSelectGroupsV1, C=hi - lo, G=min(G, hi - lo)), row[lo:hi])
        best += out[:len(out) // 2]
        idx += out[len(out) // 2:]
    return ev(bind(DF.TokenSelectCombineV2, V=V, C=G), row[0], best, idx)


def _chunked(V: int, C: int, row: list[int], ev=evaluate) -> tuple[int]:
    """``row``'s token as the window selects it now: `TokenSelectChunk_v2` over each C logits, then `TokenSelectCombine_v2{V, C}`."""
    parts = [ev(bind(DF.TokenSelectChunkV2, C=hi - lo), row[lo:hi]) for lo, hi in b1.chunks(V, C)]
    return ev(bind(DF.TokenSelectCombineV2, V=V, C=C), row[0], [p[0] for p in parts], [p[1] for p in parts])


def _memo():
    """`evaluate` that remembers each specialization's outputs by its inputs (`bind` shares one specialization per binding)."""
    seen: dict = {}

    def ev(fn, *args):
        key = (fn, *(tuple(a) if isinstance(a, list) else a for a in args))
        if key not in seen:
            seen[key] = evaluate(fn, *args)
        return seen[key]
    return ev


@pytest.mark.parametrize("V,C,G,CC", [(2, 1, 1, 2), (48, 48, 16, 48), (200, 48, 16, 64), (200, 64, 16, 64), (257, 32, 16, 64),
                                      (130, 40, 8, 64), (100, 21, 7, 16), (192, 96, 16, 64)])
def test_token_select_by_groups_is_the_chunked_select(V, C, G, CC):
    """Units of C logits in groups of G, then the combine over every group, against chunks of CC and their combine, and
    `TokenSelect_v1`: ties across every group, unit and chunk boundary, a last unit narrower than a group (200 at 48), a last
    group narrower than G (130 at 40 in 8s, 100 at 21 in 7s)."""
    bounds = sorted({b for step in (G, C, CC) for b in range(step, V, step)})
    for row in _logit_rows(V) + _boundary_rows(V, bounds):
        assert _grouped(V, C, G, row) == _chunked(V, CC, row) == evaluate(bind(b1.TokenSelect, V=V), row), row


def _first_max(row: list[int]) -> int:
    """numpy's argmax of a bf16 row with no NaN logits[0], a NaN never the maximum: torch.argmax's lowest index on ties."""
    f = (np.array(row, np.uint32) << 16).view(np.float32)
    return int(np.argmax(np.where(np.isnan(f), -np.inf, f)))


@pytest.mark.slow
def test_token_select_by_groups_is_the_chunked_select_at_llama_vocabulary():
    """Llama-3's V in 48 units of 167 groups of 16, against the window's chunks of 768 (about 12 s, the rows sharing all but a few
    units): normals, a maximum tied across a unit boundary that is no chunk's and one across a group boundary inside a unit, +0
    and -0 tied across a unit boundary both ways round, a NaN unit head ahead of the maximum, a NaN logits[0], and all -inf but
    the last logit."""
    V, C, BIG = LLAMA_V, LLAMA_C, 0x4400
    noise = _bf16(RNG.normal(0, 1, V))

    def put(at: dict[int, int], base: list[int]) -> list[int]:
        return [at.get(j, w) for j, w in enumerate(base)]

    cases = [(_first_max(noise), noise),
             (C - 1, put({C - 1: BIG, C: BIG, 5 * C + 31: BIG, 5 * C + 32: BIG}, noise)),
             (5 * C + 31, put({5 * C + 31: BIG, 5 * C + 32: BIG}, noise)),
             (9 * C + 3, put({7 * C: NAN, 9 * C + 3: BIG}, noise)),
             (2 * C - 1, put({2 * C - 1: 0x0000, 2 * C: 0x8000}, [NEG_ONE] * V)),
             (2 * C - 1, put({2 * C - 1: 0x8000, 2 * C: 0x0000}, [NEG_ONE] * V)),
             (0, [NAN] + noise[1:]),
             (V - 1, [NEG_INF] * (V - 1) + [NEG_ONE])]
    ev = _memo()
    for want, row in cases:
        assert _grouped(V, C, LLAMA_G, row, ev) == _chunked(V, 768, row, ev) == (want,), want


def _unit(fn, cuts: dict[str, int]) -> tuple[int, int]:
    """A unit of ``fn`` under ``cuts`` as Flock's `UnitStager` composes it (`interface.unit_rows`: a bit row a parameter or a
    cut's row, then one an output, each of exactly its words): its rows in and out, and the SHA-512 compression slots
    `circuit.compose` gives its rows' blocks before their last (`circuit._row_slots`).  A row of at most 111 bytes is its last
    block, SHA-512's padding with it, and compose refuses a unit of no slot."""
    ins, _ = IF.unit_rows(fn, cuts)
    ports = [FC._port(r, -(-bits // 16), bits) for r, _schema, bits, _ps in ins]
    ports += [FC.Port(o, w, 16 * w) for o, w in IF.layout(fn).outs]
    return len(ports), FC._row_slots(ports)[0]


def test_every_token_select_unit_at_llama_vocabulary_has_a_row_past_one_sha512_block():
    """With the LM head's gathers folded into their readers, a unit reads its slices of 16 logits as rows of 32 bytes.  Each of
    the 48 `TokenSelectGroups_v1{2672, 16}` units reads 167 and writes its bests and indices as rows of 334 and 668 bytes (169
    rows), and `TokenSelectCombine_v2{V, 16}` reads the 48 units' rows and logits[0] (98 rows): each has a row past 111 bytes
    and at most 256 rows.  The window's chunks of 768 have none, reading 48 slices and writing 2 and 4 bytes, and neither does
    their combine, reading 2 and 4 bytes a chunk."""
    n = LLAMA_V // LLAMA_C
    assert n * LLAMA_C == LLAMA_V
    groups = _unit(bind(DF.TokenSelectGroupsV1, C=LLAMA_C, G=LLAMA_G), {"x": LLAMA_C // LLAMA_G})
    combine = _unit(bind(DF.TokenSelectCombineV2, V=LLAMA_V, C=LLAMA_G), {"best": n, "idx": n})
    assert (groups, combine) == ((169, 3), (98, 112))
    assert _unit(bind(DF.TokenSelectChunkV2, C=768), {"x": 48}) == (50, 0)
    assert _unit(bind(DF.TokenSelectCombineV2, V=LLAMA_V, C=768), {"best": 167, "idx": 167}) == (336, 0)


@pytest.mark.slow
def test_the_row_rule_is_the_one_compose_refuses_by():
    """Flock's `UnitStager` (about 10 s, its SHA-512 slot circuits) refuses a unit of 27 groups of one logit, whose rows are 2,
    54 and 108 bytes, and composes one of 28, whose index row is 112: where `_unit` puts the line."""
    for C, slots in ((27, 0), (28, 1)):
        fn = bind(DF.TokenSelectGroupsV1, C=C, G=1)
        assert _unit(fn, {"x": C}) == (C + 2, slots)
        ins, outs = IF.unit_rows(fn, {"x": C})
        if slots:
            assert FC._row_slots(CS.UnitStager(fn, f"unit/groups-{C}", ins, outs).ports)[0] == slots
        else:
            with pytest.raises(ValueError, match="no row has a compression before its last block"):
                CS.UnitStager(fn, f"unit/groups-{C}", ins, outs)


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


#: the k16 steps the first block runs: Hopper's (FA3, and FA2 on sm_90 and sm_120) and Ampere's (FA2 on sm_80 and sm_89)
FIRST_DOTS = pytest.mark.parametrize("DOT", [hopper.HopperBF16WgmmaDot16, P.AmpereBF16TcDot16], ids=["hopper", "ampere"])


def _first_blocks(D: int, NVIS: int, BN: int, VS: int, DOT=FA2["DOT"]):
    return (bind(DF.AttnBlockFirstV2, D=D, NVIS=NVIS, BN=BN, DOT=DOT, VS=VS),
            bind(DF.AttnBlockFirstV1, D=D, NVIS=NVIS, BN=BN, DOT=DOT))


@FIRST_DOTS
@pytest.mark.parametrize("VS", [16, 8, 4, 1])
@pytest.mark.parametrize("NVIS,BN", [(1, 32), (8, 32), (15, 32), (16, 32), (17, 32), (31, 32), (32, 32), (16, 16), (5, 128)])
def test_attn_block_first_v2_equals_v1(NVIS, BN, VS, DOT):
    """A first block of 1 ... BN keys, its V rows padded to a k-step or not, each key's value in 1 to D slices: v2's circuit is
    v1's node for node over the same leaves of v in the same order, and the two agree word for word on edge words, random words
    and normals, and all -inf keys."""
    new, old = _first_blocks(16, NVIS, BN, VS, DOT)
    assert _dead(new) == 0
    assert _inlined(new) == _inlined(old)
    for q, k, v in _attention_cases(16, NVIS):
        _same(new, old, q, k, v)


@FIRST_DOTS
@pytest.mark.parametrize("NVIS", [1, 15, 16, 17, 35, 128])
def test_attn_block_first_v2_equals_v1_at_llama_geometry(NVIS, DOT):
    """Llama-3.2-1B's heads (D 64, kBlockN 128) with a key's value in 4 slices of 16 columns."""
    new, old = _first_blocks(64, NVIS, 128, 16, DOT)
    assert _inlined(new) == _inlined(old)
    for q, k, v in _attention_cases(64, NVIS):
        _same(new, old, q, k, v)


@pytest.mark.parametrize("VS", [0, -16, 3, 128, 16.0, True])
def test_attn_block_first_v2_refuses_a_slice_width_not_dividing_d(VS):
    with pytest.raises(ValueError, match="VS must be a positive divisor of D"):
        bind(DF.AttnBlockFirstV2, D=64, NVIS=4, BN=128, DOT=FA2["DOT"], VS=VS)


def test_attn_block_first_v2_reads_a_value_slice_a_row_at_4_32_4():
    """At 4,32,4 (35 positions: a head's one key block of kBlockN 128 holds 1 ... 35 keys) a Llama-3.2-1B head's first block
    reads q as RoPE's one row, k as RoPE's row a key, and v as the value projection's `Gemm_v2{2048, n}` slices at the window's
    grain, 64 / n a key.  v2's v cut once an element is those rows, n words each, where v1's v has only NVIS elements to cut; with
    its 3 outputs a unit is 5·NVIS + 4 rows in and out, within the grain's cap.  q's row and each k row (128 B) hash past one
    SHA-512 block, which Flock's compose needs of some row of a unit; the value slices (32 B) don't."""
    from verity.ml.windows import interface as IF
    from verity_flock.circuit import Port
    from verity_vllm.program.predict.window_builder import Grain
    from verity_vllm.program.predict.window_circuit import WindowClass

    grain = Grain()
    n = grain.gemm_words // 2048
    S = 64 // n
    assert (n, S) == (16, 4)
    for NVIS in range(1, WindowClass(4, 32, 4).positions + 1):
        new, old = _first_blocks(64, NVIS, 128, n)
        lay = IF.layout(new)
        assert (lay.outer[1:], IF.layout(old).outer[1:]) == ((NVIS, S * NVIS), (NVIS, NVIS))
        rows = lay.cut({"k": NVIS, "v": S * NVIS})
        assert [(p.partition("#")[0], w) for p, w, _ in rows] == [("q", 64)] + [("k", 64)] * NVIS + [("v", n)] * (S * NVIS)
        assert len(rows) + len(lay.outs) == 5 * NVIS + 4 <= grain.rows
        assert [Port(p, w, bits=16 * w).comps for p, w, _ in rows] == [1] * (1 + NVIS) + [0] * (S * NVIS)
