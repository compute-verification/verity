"""`verity_vllm.program.registry.boolean_dsv4_moe`: DeepSeek-V4's routers and clamped SwiGLU on bits agree with their word
Definitions (`trace.word_of`): MUFU.LG2 on all 2^23 mantissas (the SHA-256 of the words measured on the RTX PRO 6000) and on
random words of every exponent and sign; ex2.approx, the hash weight and the sqrtsoftplus score on special, edge and random
words; the clamped SwiGLU on every gate word and every up word against the clamp's edges, and on random pairs; the hash router
on random and special logits under random, repeated and out-of-range table ids, and the Triton router on random rows, ties, a
deciding bias, non-finite logits and biases and rows with fewer than six currents above -inf.  They are pinned by gate count and
program digest, and the routers pass `Q_word_v1{X=16, W=32}`'s cut check."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import evaluation as E
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity_vllm.program import boolean as SW
from verity_vllm.program.registry import boolean_dsv4_moe as BD
from verity_vllm.program.registry import dsv4_moe as D

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MufuLg2Mantissa_v2": (986, 4749, 92, "420c0779217331399abead35f33ffff9201fce329ef972d4e50b3bc0aaa19e6dee5d40a7c01fa46e72ac0df73c8b1f56be7c4f98c692c721814861ff2f2f221d"),
    "SiluMulClampBf16_v2{LIMIT=10.0}": (1489, 7153, 207, "d8cb5e0c6612969c40d2c8dae79e629625ce1fb675970bf3a5c1220662421f708fea9f4efabee3972e7096bec46d2f0b5e9709ebde952f27de3f25a60fb7766b"),
    "SiluMulClamp_v2{I=4,LIMIT=10.0}": (5956, 28612, 828, "49ee7a9fa8385d3022c224922f231f569d9a908778809416e0797b52269d989041a7c43011fca22c475d42a2211326c5ff69bd2ec68d8aca400d20539d13caf6"),
    "Ex2Approx_v2": (3857, 9772, 300, "e0464671c57dca8de9abe8b6692b8471b782d76869557426f67824c77e1ba74c9e19c7fc1d1977af5f8b67a649ac8d5ada92a427e45aee510937364fa0a35fd2"),
    "Dsv4HashWeight_v2": (11576, 26283, 1059, "b90892d782f915cf1bc388ce51e0cda6d5453e003eec58997da9bb1615b56c391b1078d0d82f73c0965200724dad0e63da31ecba7e9408b88cdd4b4c95eb20f6"),
    "MoeRouterHash_v2{E=8,SCALE=2.5}": (93595, 205463, 9019, "478cd3da7feb6d8c222e241662d072c6d3ca48b4c4a154cc05431e4b9e808bfce7446ce4a131bfe879089d6030f4ecad63a5614f18134f83b5aea27209717416"),
    "Dsv4SqrtSoftplus_v2": (50376, 93380, 3493, "a93fb47b11aac786622772916da070b6f1e23ea667d9a5b269114b9c68d5e36ba4caaefb915870996a5d3238b223d2831c95d7cf4c282b681ae52be9326ecf59"),
    "MoeRouterSqrtSoftplus_v2{E=8,SCALE=2.5}": (435870, 808659, 32308, "0b3e11c44e7883154b4b03c058455257c801435ad6c08407357061e3fe7a13238e9400038f38c6324ba345892217526f7e93141fa3257c2f3e0efc07f342b9e1"),
}
TARGETS = [bind(BD.MufuLg2Mantissa), bind(BD.SiluMulClampBf16, LIMIT=10.0), bind(BD.SiluMulClamp, I=4, LIMIT=10.0),
           *[pytest.param(t, marks=pytest.mark.slow) for t in (bind(BD.Ex2Approx), bind(BD.Dsv4HashWeight),
                                                               bind(BD.MoeRouterHash, E=8, SCALE=2.5), bind(BD.Dsv4SqrtSoftplus),
                                                               bind(BD.MoeRouterSqrtSoftplus, E=8, SCALE=2.5))]]

NEG_INF = 0xFF800000


def _f32(*xs) -> np.ndarray:
    return np.array(xs, np.float32).view(np.uint32)


#: signed zeros and smallest subnormals, +-1, infinities, NaNs of both kinds and signs, the finite maxima
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x3F800000, 0xBF800000, 0x7F800000, NEG_INF,
                         0x7FC00000, 0x7F800001, 0xFFC00001, 0x7FFFFFFF, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
#: where the functions change form: ex2.approx's t < -126 and its range's ends (-149, -150, 128), the hash weight's ``1 +
#: e^-|x|`` rounding to 1 or 2 (|x| near 2^-24 and 16.6) and its ex2 operand below -126 (|x| > 87.3), the score's x > 20 and
#: its ``1 + e^x`` rounding to 1 (x below -16.6, e^x subnormal below -87.3)
F32_EDGES = np.concatenate([
    np.array([0xC2FC0000, 0xC2FC0001, 0xC2FBFFFF, 0x41A00000, 0x41A00001, 0x419FFFFF, 0xC1A00000, 0x33800000, 0x33800001,
              0x337FFFFF, 0xB3800000], np.uint32),
    _f32(-127, -149, -149.5, -150, -151, 127.99, 128, 129, 16.5, 16.6, 16.7, -16.6, 87.3, 87.4, -87.3, -87.4, -103, -104,
         -105, 88.7, 89, 19.99, 20.01, 0.5, -0.5, 1e-30, -1e-30)])
#: bf16 words at the clamp: +-0, +-1, LIMIT 10 and its neighbours on both signs, the finite maxima, infinities and NaNs
CLAMP_EDGES = np.array([0, 0x8000, 0x3F80, 0xBF80, 0x4120, 0x411F, 0x4121, 0xC120, 0xC11F, 0xC121, 0x7F7F, 0xFF7F, 0x7F80,
                        0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x0001, 0x8001], np.uint16)


def _rows(out, n: int) -> np.ndarray:
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in parts], axis=1)


def _agree(target, *args) -> None:
    word = TR.word_of(target)
    n = len(args[0])
    got, want = _rows(BS.evaluate_words(target, *args), n), _rows(E.evaluate_batch(word, *args), n)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {n} rows differ from {word.id}, first {bad[:5]}"


def _words(rng, n: int, lo: float, hi: float) -> np.ndarray:
    """F32_SPECIALS, F32_EDGES, n random words and n values in [lo, hi] and in [-1, 1]."""
    return np.concatenate([F32_SPECIALS, F32_EDGES, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32),
                           rng.uniform(lo, hi, n).astype(np.float32).view(np.uint32),
                           rng.uniform(-1, 1, n).astype(np.float32).view(np.uint32)])


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


def test_the_word_views() -> None:
    assert TR.word_of(bind(BD.MufuLg2Mantissa)).id == D.MufuLg2Mantissa.id == "MufuLg2Mantissa_v1"
    assert SW.boolean_version(D.MufuLg2Mantissa).id == "MufuLg2Mantissa_v2"
    for b, w in ((BD.Ex2Approx, D.Ex2Approx), (BD.Dsv4HashWeight, D.Dsv4HashWeight), (BD.Dsv4SqrtSoftplus, D.Dsv4SqrtSoftplus)):
        assert TR.word_of(bind(b)).id == bind(w).id
        assert SW.boolean_version(bind(w)).id == bind(b).id
    for b, w, st in ((BD.MoeRouterHash, D.MoeRouterHash, {"E": 384, "SCALE": 2.5}),
                     (BD.MoeRouterSqrtSoftplus, D.MoeRouterSqrtSoftplus, {"E": 384, "SCALE": 2.5}),
                     (BD.SiluMulClampBf16, D.SiluMulClampBf16, {"LIMIT": 10.0}),
                     (BD.SiluMulClamp, D.SiluMulClamp, {"I": 384, "LIMIT": 10.0})):
        assert TR.word_of(bind(b, **st)).id == bind(w, **st).id
        assert SW.boolean_version(bind(w, **st)).id == bind(b, **st).id


def test_lg2_where_measured_and_on_every_exponent_and_sign() -> None:
    rng = np.random.default_rng(0xD54)
    m = np.concatenate([np.array([0, 1, 2, 3, 0x1FFFF, 0x20000, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF], np.uint32),
                        rng.integers(0, 1 << 23, 2000, dtype=np.uint32)])
    x = m | (rng.integers(0, 1 << 9, len(m), dtype=np.uint32) << 23)
    _agree(bind(BD.MufuLg2Mantissa), x)


@pytest.mark.slow
def test_lg2_on_every_mantissa_is_the_measured_unit() -> None:
    """All 2^23 words of [1, 2) against `LG2_MEASURED`'s SHA-256 (about 10 s): the Boolean interpolator is the measured unit, not
    only its word view.  Its exponent field is 0 exactly at m = 0, which `Dsv4HashWeightOut_v1` reads as m = 0."""
    h = hashlib.sha256()
    for lo in range(0, 1 << 23, 1 << 19):
        m = np.arange(lo, lo + (1 << 19), dtype=np.uint32)
        y = np.asarray(BS.evaluate_words(bind(BD.MufuLg2Mantissa), m | np.uint32(0x3F800000))).astype("<u4").reshape(-1)
        assert np.array_equal((y >> 23 & 0xFF) == 0, m == 0)
        h.update(y.tobytes())
    assert h.hexdigest() == D.LG2_MEASURED["sha256"]


@pytest.mark.slow
def test_ex2_approx_on_special_edge_and_random_words() -> None:
    _agree(bind(BD.Ex2Approx), _words(np.random.default_rng(0xE2), 3000, -160, 130))


@pytest.mark.slow
def test_the_hash_weight_on_special_edge_and_random_words() -> None:
    _agree(bind(BD.Dsv4HashWeight), _words(np.random.default_rng(0x4A5), 3000, -100, 100))


@pytest.mark.slow
def test_the_sqrtsoftplus_score_on_special_edge_and_random_words() -> None:
    _agree(bind(BD.Dsv4SqrtSoftplus), _words(np.random.default_rng(0x5F), 3000, -110, 40))


def test_the_clamped_swiglu_on_its_edges_and_random_pairs() -> None:
    rng = np.random.default_rng(0xC19)
    g = np.concatenate([np.repeat(CLAMP_EDGES, len(CLAMP_EDGES)), rng.integers(0, 1 << 16, 8000).astype(np.uint16)])
    u = np.concatenate([np.tile(CLAMP_EDGES, len(CLAMP_EDGES)), rng.integers(0, 1 << 16, 8000).astype(np.uint16)])
    _agree(bind(BD.SiluMulClampBf16, LIMIT=10.0), g, u)


@pytest.mark.slow
def test_the_clamped_swiglu_on_every_gate_and_every_up_word() -> None:
    """Every gate word against four of the clamp's edges and every up word against one (about 17 s)."""
    every = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    rng = np.random.default_rng(0xC1A)
    g = np.concatenate([np.repeat(every, 4), np.tile(CLAMP_EDGES, 1 << 14)[:1 << 16], rng.integers(0, 1 << 16, 4000).astype(np.uint16)])
    u = np.concatenate([rng.choice(CLAMP_EDGES, 1 << 18), every, rng.integers(0, 1 << 16, 4000).astype(np.uint16)])
    _agree(bind(BD.SiluMulClampBf16, LIMIT=10.0), g, u)


def test_the_clamped_swiglu_row() -> None:
    rng = np.random.default_rng(0xC1B)
    gu = rng.integers(0, 1 << 16, (500, 8)).astype(np.uint16)
    gu[::3] = rng.choice(CLAMP_EDGES, (len(gu[::3]), 8))
    _agree(bind(BD.SiluMulClamp, I=4, LIMIT=10.0), gu)


def _logits(rng, n: int, E_: int) -> np.ndarray:
    """n rows of each kind: values in [-8, 8], random words, and rows a quarter of whose logits are special words."""
    x = np.concatenate([rng.uniform(-8, 8, (n, E_)).astype(np.float32).view(np.uint32),
                        rng.integers(0, 1 << 32, (n, E_), dtype=np.uint64).astype(np.uint32),
                        rng.uniform(-30, 30, (n, E_)).astype(np.float32).view(np.uint32)])
    special = rng.integers(0, 4, x[2 * n:].shape) == 0
    x[2 * n:][special] = rng.choice(np.concatenate([F32_SPECIALS, F32_EDGES]), int(special.sum()))
    return x


def _hash_rows(rng, n: int, E_: int) -> tuple[np.ndarray, np.ndarray]:
    """`_logits`' rows under random table ids in [0, E) (repeats included), every eighth row's ids also above E (logits[0])."""
    x = _logits(rng, n, E_)
    ids = rng.integers(0, E_, (len(x), D.TOPK)).astype(np.uint32)
    ids[::8, ::2] = rng.integers(E_, 1 << 32, (len(ids[::8]), 3), dtype=np.uint64).astype(np.uint32)
    x[1::9] = NEG_INF
    x[2::9, 0] = 0x7F800000
    return x, ids


@pytest.mark.slow
@pytest.mark.parametrize("E_", [8, 6])
def test_the_hash_router(E_) -> None:
    """At 8 experts and at 6 (the pick tree padded), 200 rows of each kind, rows of -inf (no weight) and of a +inf logit."""
    _agree(bind(BD.MoeRouterHash, E=E_, SCALE=2.5), *_hash_rows(np.random.default_rng(0x4A6 + E_), 200, E_))


def _router_rows(rng, n: int, E_: int) -> tuple[np.ndarray, np.ndarray]:
    """`_logits`' rows with biases in [-0.5, 0.5]; ties (multiples of 0.5, bias 0), a bias deciding the slots (one large),
    non-finite biases, and rows with fewer than six currents above -inf (-inf biases) or none."""
    x = _logits(rng, n, E_)
    b = rng.uniform(-0.5, 0.5, x.shape).astype(np.float32).view(np.uint32)
    k = len(x)
    ties = slice(0, k, 7)
    x[ties] = (rng.integers(-4, 5, x[ties].shape) * 0.5).astype(np.float32).view(np.uint32)
    b[ties] = 0
    b[3::7, 1] = _f32(50.0)[0]
    few = rng.integers(0, 3, b[4::7].shape) > 0
    b[4::7][few] = NEG_INF
    b[5::7] = NEG_INF
    b[6::7][rng.integers(0, 4, b[6::7].shape) == 0] = rng.choice(F32_SPECIALS, 1)[0]
    return x, b


@pytest.mark.slow
@pytest.mark.parametrize("E_", [8, 6])
def test_the_triton_router(E_) -> None:
    """At 8 experts and at 6 (every expert selected), 200 rows of each kind; about 30 s to trace (its MUFU.SQRT and logf)."""
    _agree(bind(BD.MoeRouterSqrtSoftplus, E=E_, SCALE=2.5), *_router_rows(np.random.default_rng(0x7B + E_), 200, E_))


@pytest.mark.slow
@pytest.mark.parametrize("target", [bind(BD.MoeRouterHash, E=8, SCALE=2.5), bind(BD.MoeRouterSqrtSoftplus, E=8, SCALE=2.5)],
                         ids=lambda t: t.id)
def test_the_routers_pass_the_word_querys_cut_check(target) -> None:
    """Each value a router reads is decoded in one of its Calls (each expert's MUFUs and pieces, the pick or the slots, the hash
    router's renormalisation): the cut check finds no value computed twice."""
    prog, _ = standalone_call(target)
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
