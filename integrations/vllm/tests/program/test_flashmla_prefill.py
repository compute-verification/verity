"""DeepSeek-V4-Pro's FlashMLA sparse prefill and its K-cache gather on the H200 (`flashmla_prefill`, `boolean_flashmla_prefill`,
`kernels.flashmla_prefill_twin`).

* the twin (the arithmetic written again over numpy words) word for word against the reference on every binding it registers;
* the gather's scale word and row validity on their edge words, the gather against exact dequantisation and against the decode's
  staging where their scale words agree;
* the prefill against float64 softmax with the sink at DeepSeek-V4-Pro's statics, over an even and an odd number of blocks;
* the small Boolean Definitions against their word views; the large ones (compiled flat) are circuit-check's
  (`boolean_flashmla_prefill.circuit-check.py`).
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.kernels import tc as CK
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import flashmla_prefill_twin as T
from verity.ml.definitions.numerics import flashmla_twin as FT
from verity.ml.definitions.registry import boolean_flashmla_prefill as BP
from verity.ml.definitions.registry import flashmla_prefill as P

U = np.uint32
DEFS = {d.id: d for d in P.FLASHMLA_PREFILL_COMPOSITES}
TWIN_TARGETS = [t for family in T.KERNELS for t in T._targets(DEFS, family)]
DSV4, GATHER = P.FLASHMLA_PREFILL_DSV4, P.FLASHMLA_GATHER_DSV4
E4M3 = CK.e4m3_to_f32_batch(np.arange(127, dtype=np.uint8))[0].view(np.float32).astype(np.float64)


def _bf16_f64(w) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(U) << U(16)).view(np.float32).astype(np.float64)


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=24, seed=0)
    assert sc.ok, str(sc)
    assert sc.declined == 0, str(sc)


# --- the gather and the rows ------------------------------------------------------------------------------------------------------

def test_the_scale_word_is_the_byte_in_the_bf16_exponent():
    assert [P.E8m0ShlBf16.evaluate(x) for x in (0, 1, 127, 254, 255)] == [0, 0x0080, 0x3F80, 0x7F00, 0x7F80]
    assert [int(v) for v in T.e8m0_shl_bf16(np.arange(256))] == [P.E8m0ShlBf16.evaluate(x) for x in range(256)]


def test_the_gather_dequantises_exactly():
    """Every finite E4M3 byte under every scale 2^-126 .. 2^127 is its value times the scale rounded once to bf16 (the product is
    exact in float64)."""
    nope = np.tile(np.arange(256, dtype=np.uint8), (254, 1))
    nope = np.where((nope & 0x7F) == 0x7F, 0, nope).astype(np.uint8)
    scale = np.repeat(np.arange(1, 255, dtype=np.uint8)[:, None], 4, axis=1)
    got = T.gather_rows(nope, scale, np.zeros((254, 0), dtype=np.uint16), QT=64)
    x = CK.e4m3_to_f32_batch(nope)[0].view(np.float32).astype(np.float64) * 2.0 ** (np.repeat(scale, 64, axis=1) - 127.0)
    want = DR.f2fp_bf16(x.astype(np.float32).view(U))
    assert np.array_equal(got, want)


def test_the_gather_is_the_decode_staging_where_the_scale_words_agree():
    """With every scale byte in 1..254 the gather writes the row `FlashMlaKvRow_v1` stages for a valid row; the bytes 0 and 0xFF
    (the gather's +0 and +inf, the decode's 0x0040 and 0x7FFF) give other words."""
    rng = np.random.default_rng(5)
    nope, rope = FT._e4m3(rng, (64, 448)), FT._bf16_words(rng, (64, 64))
    scale = rng.integers(1, 255, (64, 7)).astype(np.uint8)
    assert np.array_equal(T.gather_rows(nope, scale, rope, 64), FT.kv_rows(nope, scale, rope, np.ones(64), 64))
    for s in (0, 0xFF):
        edge = scale.copy()
        edge[:, 3] = s
        assert not np.array_equal(T.gather_rows(nope, edge, rope, 64), FT.kv_rows(nope, edge, rope, np.ones(64), 64))


def test_a_row_is_valid_when_its_index_lies_in_the_workspace_and_its_position_below_the_length():
    m = lambda v: v & 0xFFFFFFFF  # noqa: E731
    cases = {(5, 10, 9, 6): 1, (5, 10, 10, 6): 0, (5, 10, 9, 5): 0, (0, 1, 0, 1): 1, (m(-1), 10, 0, 6): 0, (m(-2), 10, 0, 6): 0,
             (5, 0, 0, 6): 0, (5, m(-1), 0, 6): 0, (0x7FFFFFFF, 1, 0, 0x7FFFFFFF): 0, (0x7FFFFFFE, 1, 0, 0x7FFFFFFF): 1,
             (0, 1, 0, 0): 0, (0, 1, 0, m(-1)): 0, (0, 0x7FFFFFFF, 1 << 30, 1): 1}
    assert {c: evaluate(bind(P.FlashMlaPrefillValid), *c)[0] for c in cases} == cases
    idx = np.array([m(c[0]) for c in cases], dtype=U)
    assert [bool(T.row_valid(idx[i:i + 1], c[1], c[3])[0]) for i, c in enumerate(cases) if c[2] == 0] == \
        [bool(v) for c, v in cases.items() if c[2] == 0]


# --- the prefill against float64 -------------------------------------------------------------------------------------------------

def _to_e4m3(x: np.ndarray) -> np.ndarray:
    idx = np.abs(np.clip(np.abs(x), 0, 448)[..., None] - E4M3).argmin(axis=-1).astype(np.uint8)
    return np.where(x < 0, idx | 0x80, idx).astype(np.uint8)


def _operands(rng, N: int, n: int, skv: int, NH: int = 16):
    """A token's q and its N workspace rows as the gather writes them from cache rows: each 64-element NoPE block e4m3 under the
    UE8M0 scale 2^ceil(log2(amax / 448)), the RoPE bf16; 1 in 10 indices outside [0, skv)."""
    DN, DR, QT = GATHER["DN"], GATHER["DR"], GATHER["QT"]
    k = rng.standard_normal((N, DN + DR))
    e = np.ceil(np.log2(np.abs(k[:, :DN]).reshape(N, DN // QT, QT).max(axis=2) / 448)).astype(int)
    nope = _to_e4m3(k[:, :DN] / np.repeat(2.0 ** e, QT, axis=1))
    rope = (k[:, DN:].astype(np.float32).view(U) >> U(16)).astype(np.uint16)
    kv = T.gather_rows(nope, (e + 127).astype(np.uint8), rope, QT)
    idx = np.where(rng.random(N) < 0.9, rng.integers(0, skv, N), rng.choice([-1, skv, skv + 7], N)).astype(np.int64).astype(U)
    q = ((rng.standard_normal((NH, DN + DR)) * 2).astype(np.float32).view(U) >> U(16)).astype(np.uint16)
    return q, kv, idx, U(n), U(skv), rng.standard_normal(NH).astype(np.float32).view(U)


def _softmax_with_sink(q, kv, idx, n, skv, sink) -> np.ndarray:
    valid = T.row_valid(idx, n, skv)
    k = np.where(valid[:, None], _bf16_f64(kv), 0.0)
    s = np.where(valid[None, :], _bf16_f64(q) @ k.T * DSV4["SCALE"], -np.inf)
    sk = sink.view(np.float32).astype(np.float64)
    m = np.maximum(s.max(axis=1), sk)
    w = np.exp(s - m[:, None])
    return (w @ k) / (w.sum(axis=1) + np.exp(sk - m))[:, None]


@pytest.mark.parametrize("seed,n", [(0, 600), (1, 520), (2, 577)], ids=["even", "odd", "odd-cut"])
def test_the_prefill_is_softmax_with_the_sink(seed, n):
    """A token's 16 heads over 640 workspace rows (C4's 512 indexed and the 128-row window: five pairs), topk_length 600 (ten
    blocks), 520 (nine: the fifth pair's second block lies past it) and 577: each head within 0.01 of its largest output from
    float64 softmax with the sink (a sweep over seeds 0-19 at each length: worst 0.0049; P is rounded to bf16 before P V)."""
    rng = np.random.default_rng(seed)
    ops = _operands(rng, 640, n, skv=int(rng.integers(700, 4000)))
    want = _softmax_with_sink(*ops)
    got = _bf16_f64(T.prefill(*ops, BN=DSV4["BN"], SCALE=DSV4["SCALE"]))
    top = np.abs(want).max(axis=1, keepdims=True)
    assert (np.abs(got - want) / top).max() <= 0.01


def test_a_token_with_no_valid_row_writes_zeros():
    rng = np.random.default_rng(7)
    q, kv, idx, _, skv, sink = _operands(rng, 128, 1, skv=64, NH=4)
    for n, s in ((U(0), skv), (U(128), U(0))):
        assert not T.prefill(q, kv, idx, n, s, sink, BN=64, SCALE=DSV4["SCALE"]).any()


# --- the statics --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,needle", [
    (P.FlashMlaGatherRow, dict(DN=448, DR=64, QT=48), "FlashMlaGatherRow: QT=48"),
    (P.FlashMlaPrefillPair, dict(D=48, BN=16, FIRST=True), "D=48"),
    (P.FlashMlaPrefillPair, dict(D=32, BN=8, FIRST=False), "BN=8"),
    (P.FlashMlaPrefillHead, dict(NP=0, D=32, BN=16), "NP=0"),
], ids=["scale-block", "halves", "block", "pairs"])
def test_bad_statics_are_refused_by_name(fn, kw, needle):
    extra = {k: v for k, v in (("DOT", P.HopperBF16WgmmaDot16), ("SCALE", 0.25)) if k in fn.statics}
    with pytest.raises(ValueError, match=needle):
        bind(fn, **kw, **extra).body


# --- the Boolean Definitions --------------------------------------------------------------------------------------------------------

def _valid_args(rng, n: int) -> list[np.ndarray]:
    near = lambda: rng.integers(-3, 40, n)  # noqa: E731
    far = lambda: rng.integers(-(1 << 31), 1 << 31, n)  # noqa: E731
    return [np.where(rng.random(n) < 0.2, far(), near()).astype(np.int64).astype(U).astype(np.uint64) for _ in range(4)]


SMALL = [(BP.E8m0ShlBf16V2, lambda rng: [np.arange(256, dtype=np.uint64)]),
         (BP.FlashMlaPrefillValidV2, lambda rng: _valid_args(rng, 1024)),
         (bind(BP.FlashMlaGatherRowV2, DN=16, DR=16, QT=8), lambda rng: T._sample_gather(rng, 128, None, DN=16, DR=16, QT=8))]


@pytest.mark.parametrize("target,args", SMALL, ids=[t.id for t, _ in SMALL])
def test_the_small_boolean_definitions_are_their_words(target, args):
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    a = args(np.random.default_rng(1))
    got = np.asarray(evaluate_words(target, *a), dtype=np.uint64).reshape(len(a[0]), -1)
    want = np.asarray(evaluate_batch(word, *a, kernel="reference"), dtype=np.uint64).reshape(len(a[0]), -1)
    assert np.array_equal(got, want)
