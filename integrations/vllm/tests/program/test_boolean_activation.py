"""`verity_vllm.program.registry.boolean_activation`: Gemma-2's Boolean activations agree with their word Definitions
(`trace.word_of`): the tanh and scalar elements and rows on all 65,536 words, the GeGLU on every gate word, on every pair of
special words and on random and edge-product pairs.  Each carries exactly its builder's live ANDs, is pinned by gate count and
program digest, and partitions under `Q_word_v1{X=16, W=32}` into its word Definition's units, one per element."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir import partition_object as PO
from verity.ir.defs import REGISTRY, bind
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit
from verity_vllm.program.registry import boolean_activation as BA
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

K30, R30 = P.bits_of(np.float32(30.0)), D.reciprocal_bits(30.0)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "GeluTanhMulBf16_v2": (1353, 6097, 156, "defa19a677ee07be805d074c54570be9e8d1edbe67cf8ca4e42201f6ebbe15c3"),
    "GeluTanhMul_v2{I=8}": (10824, 48776, 1248, "b2c436912baa61205cf63dcc582b9e59420564d598c8721b5825012de8016892"),
    "TanhBf16_v2": (391, 1628, 31, "e6d9d24d463514967c9630141c5794340d8fdbcb0d3f3cf14024c19f1b7da912"),
    "Bf16Tanh_v2{N=8}": (3128, 13024, 248, "1fa5eaeb7f5accef60cf881e9ae2a9818ad3d30a041e8d384641e304ec9928ba"),
    f"Bf16MulConst_v2{{K={K30}}}": (629, 834, 94, "2641db1933ea5fa546ac0b0d3b97fe6d3cd5b2469093e255d8192bc40bc93afc"),
    f"Bf16MulConst_v2{{K={R30}}}": (865, 1283, 117, "f5ae8076857b61844c620b6d309b2366f0b2c68924211352b46ea01af48e09ba"),
    "Bf16DivScalar_v2{N=8,C=30.0}": (6920, 10264, 936, "8ca94870ba9531f7c343cfc18d8395c7cd904e7ef82f29ebc343f306694802e2"),
    "Bf16MulScalar_v2{N=8,C=30.0}": (5032, 6672, 752, "6e12cc5445d75a5cba7dbdd1b8891c4b746ece09380e0b45d42e1407c6eb1801"),
}
PINNED = [bind(BA.GeluTanhMulBf16), bind(BA.GeluTanhMul, I=8), bind(BA.TanhBf16), bind(BA.Bf16Tanh, N=8),
          bind(BA.Bf16MulConst, K=K30), bind(BA.Bf16MulConst, K=R30), bind(BA.Bf16DivScalar, N=8, C=30.0),
          bind(BA.Bf16MulScalar, N=8, C=30.0)]

#: Gemma-2's final-logit (30) and attention-logit (50) caps, and the test models' scalars
CAPS = [30.0, 50.0, 0.5, 1.0]

_EXP = [0, 1, 2, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 133, 134, 135, 254, 255]
#: per sign: zero and the subnormal ends, each band edge's exponents at the mantissa ends, ones, the finite maximum, the
#: infinities and NaNs (quiet, signalling, all ones)
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)
WORDS = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _word(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got, want = _bits(target, *inputs), _word(target, *inputs)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


def _rows(seed: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.integers(0, 1 << 16, (4000, width), dtype=np.int64).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (1000, width))]])


def test_pinned_counts_and_digests() -> None:
    for target in PINNED:
        ands, xors, nots, digest = PINS[target.id]
        assert BL.is_boolean(target)
        assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
        prog, _ = standalone_call(target)
        desc = codec.encode_program(prog)
        assert codec.program_digest(desc) == digest
        decoded = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
        assert decoded.gates == prog.gates and codec.program_digest(codec.encode_program(decoded)) == digest


@pytest.mark.parametrize("d", [BA.GeluTanhMulBf16, BA.TanhBf16], ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS[bind(d).id][0]


def test_the_word_views() -> None:
    assert TR.word_of(bind(BA.GeluTanhMulBf16)) is D.GeluTanhMulBf16
    assert TR.word_of(bind(BA.TanhBf16)).id == "TanhBf16_v1"
    assert TR.word_of(bind(BA.Bf16MulConst, K=K30)).id == f"Bf16MulConst_v1{{K={K30}}}"
    assert TR.word_of(bind(BA.GeluTanhMul, I=8)).id == bind(D.GeluTanhMul, I=8).id == "GeluTanhMul_v1{I=8}"
    assert TR.word_of(bind(BA.Bf16Tanh, N=8)).id == "Bf16Tanh_v1{N=8}"
    assert TR.word_of(bind(BA.Bf16DivScalar, N=8, C=30.0)).id == "Bf16DivScalar_v1{N=8,C=30.0}"
    assert TR.word_of(bind(BA.Bf16MulScalar, N=8, C=30.0)).id == "Bf16MulScalar_v1{N=8,C=30.0}"


def test_the_element_words_are_the_served_rows_elements() -> None:
    """`TanhBf16_v1` and `Bf16MulConst_v1{K}` are `Bf16Tanh_v1`'s and the scalar composites' bodies at N = 1 (the element
    Definitions are checked against them exhaustively, and their rows against the served rows)."""
    x = np.concatenate([SPECIALS, np.random.default_rng(0x6E0).integers(0, 1 << 16, 4096, dtype=np.int64).astype(np.uint16)])
    pairs = [(bind(BA.TanhBf16Word), bind(D.Bf16Tanh, N=1))]
    for c in CAPS:
        pairs += [(bind(BA.Bf16MulConstWord, K=D.reciprocal_bits(c)), bind(D.Bf16DivScalar, N=1, C=c)),
                  (bind(BA.Bf16MulConstWord, K=P.bits_of(np.float32(c))), bind(D.Bf16MulScalar, N=1, C=c))]
    for element, row in pairs:
        got = np.asarray(E.evaluate_batch(element, x), np.uint64).reshape(-1)
        assert np.array_equal(got, np.asarray(E.evaluate_batch(row, x[:, None]), np.uint64).reshape(-1)), element.id


def _half(g: int) -> int:
    """``bf16_rn(f32(g) / 2)``, the sign kept."""
    f = np.array([g << 16], np.uint32).view(np.float32) / np.float32(2)
    return P.f32_to_bf16_rn_bits(int(f.view(np.uint32)[0]), BA.NAN_F2FP)


def _pair(sign: int, e0: int) -> range:
    return range(sign << 15 | e0 << 7, sign << 15 | (e0 + 2) << 7)


def test_gelu_band_is_where_the_closed_forms_fail() -> None:
    """Every gate word off GeLU's band is one of the module docstring's closed forms; on the band the sign is g's and the
    magnitude the table's; and the band's first exponent pair and its last are each needed (a closed form fails there)."""
    lo, hi = BA.GELU_BAND
    gelu = [D.gelu_tanh_bf16_bits(g) for g in range(1 << 16)]
    table = BA.gelu_table()
    low = lambda g: _half(g)                                              # noqa: E731
    high = lambda g: 0x8000 if g >> 15 else g                             # noqa: E731
    for g, a in enumerate(gelu):
        sign, e, m = g >> 15, (g >> 7) & 0xFF, g & 0x7F
        if e == 255 and (m or sign):
            assert a == BA.NAN_F2FP
        elif e < lo:
            assert a == low(g)
        elif e > hi:
            assert a == high(g)
        else:
            assert a >> 15 == sign and a & 0x7FFF == table[m | (e & 0xF) << 7 | sign << 11]
    for sign in (0, 1):
        assert any(gelu[g] != low(g) for g in _pair(sign, lo))
        assert any(gelu[g] != high(g) for g in _pair(sign, hi - 1))


def test_tanh_band_is_where_the_closed_forms_fail() -> None:
    """Every word off tanh's band is x, +-1 or the NaN; on the band the sign is x's and the magnitude the (odd) table's; and
    the band's first exponent pair and its last are each needed."""
    lo, hi = BA.TANH_BAND
    tanh = [int(v) for v in _word(bind(BA.TanhBf16), WORDS)]
    table = BA.tanh_table()
    one = lambda x: x & 0x8000 | 0x3F80                                   # noqa: E731
    for x, t in enumerate(tanh):
        sign, e, m = x >> 15, (x >> 7) & 0xFF, x & 0x7F
        if e == 255 and m:
            assert t == BA.NAN_F2FP
        elif e < lo:
            assert t == x
        elif e > hi:
            assert t == one(x)
        else:
            assert t >> 15 == sign and t & 0x7FFF == table[m | (e & 0xF) << 7]
    for sign in (0, 1):
        assert any(tanh[x] != x for x in _pair(sign, lo))
        assert any(tanh[x] != one(x) for x in _pair(sign, hi - 1))


@pytest.mark.parametrize("u", [0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8080, 0x7F7F, 0xFF80, 0x7FC0, 0x4000, 0x3B80, 0x0080])
def test_every_gate_word(u: int) -> None:
    """All 65,536 gate words at an up word: at 1.0 the GeLU half alone, then signs, zeros (-0 is +0 after ``+ 0.0f``),
    subnormals, the finite maximum, an infinity, a NaN and scales that move the product's exponent."""
    _agree(bind(BA.GeluTanhMulBf16), WORDS, np.full(WORDS.shape, u, np.uint16))


def _edge_products(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(g, u) whose product f32(a) * f32(u), a the GeLU half, lands in or near f32's subnormal range or at its overflow: g
    random, u's exponent field chosen against a's so that the product's is in -26..3 or 250..257."""
    g = rng.integers(0, 1 << 16, n, dtype=np.int64)
    a = np.array([D.gelu_tanh_bf16_bits(int(x)) for x in g], np.int64)
    ea = (a >> 7) & 0xFF
    target = np.where(rng.integers(0, 2, n) == 1, rng.integers(-26, 4, n), rng.integers(250, 258, n))
    eu = np.clip(target - ea + 127, 0, 254)
    u = rng.integers(0, 2, n) << 15 | eu << 7 | rng.integers(0, 128, n)
    return g.astype(np.uint16), u.astype(np.uint16)


def test_special_pairs_random_and_edge_products() -> None:
    """Every pair of the special words, 80,000 random pairs and 20,000 products at f32's subnormal and overflow ends."""
    rng = np.random.default_rng(0x6E1)
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=2)), np.int64).T
    rand = rng.integers(0, 1 << 16, (2, 80_000), dtype=np.int64).astype(np.uint16)
    eg, eu = _edge_products(rng, 20_000)
    _agree(bind(BA.GeluTanhMulBf16), np.concatenate([SPECIALS[grid[0]], rand[0], eg]),
           np.concatenate([SPECIALS[grid[1]], rand[1], eu]))


def test_every_word_of_tanh() -> None:
    """The tanh row of one on all 65,536 words, against the served `Bf16Tanh_v1{N=1}`."""
    _agree(bind(BA.Bf16Tanh, N=1), WORDS[:, None])


@pytest.mark.parametrize("c", CAPS)
def test_every_word_of_the_scalar_rows(c: float) -> None:
    """The scalar rows of one at each cap on all 65,536 words, against the served `Bf16DivScalar_v1` and `Bf16MulScalar_v1`."""
    _agree(bind(BA.Bf16DivScalar, N=1, C=c), WORDS[:, None])
    _agree(bind(BA.Bf16MulScalar, N=1, C=c), WORDS[:, None])


def test_the_rows() -> None:
    _agree(bind(BA.GeluTanhMul, I=8), _rows(0x6E2, 16))
    _agree(bind(BA.Bf16Tanh, N=8), _rows(0x6E3, 8))
    _agree(bind(BA.Bf16DivScalar, N=8, C=30.0), _rows(0x6E4, 8))
    _agree(bind(BA.Bf16MulScalar, N=8, C=30.0), _rows(0x6E5, 8))


@pytest.mark.parametrize("target", [bind(BA.GeluTanhMul, I=2), bind(BA.Bf16Tanh, N=3), bind(BA.Bf16DivScalar, N=3, C=30.0),
                                    bind(BA.Bf16MulScalar, N=3, C=50.0)], ids=lambda t: t.id)
def test_the_word_definitions_units_under_q_word(target) -> None:
    """One unit per element, as many as the word Definition's: the switch swaps one for the other without moving a cut."""
    q = PO.query(16, 32)
    prog, _ = standalone_call(target)
    word, _ = standalone_call(TR.word_of(target))
    n = target.signature()[1].n
    assert PO.evaluate(prog, q).population == PO.evaluate(word, q).population == n
    assert PO.verify(PO.build(prog, q), prog).ok
