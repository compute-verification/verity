"""`verity.circuits.ops.boolean.elementwise`: the Boolean element-wise operations (the f32 ones with the word primitives' NaN words and,
as the `_v4`s, with the GPU's) agree with their word primitives (`trace.word_of`) on
every pair (triple) of special words, on random words and on words chosen to land in the subnormal range (a bf16 operation also
on every word against a few specials, and a product at f32's subnormal and overflow ends), carry exactly their
builders' live ANDs, and are pinned by gate count and program digest. Bf16AddF2fp_v2 is checked on every pair of words when
``VERITY_ML_BOOLEAN_EXHAUSTIVE`` is on."""
from __future__ import annotations

import itertools
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from verity_config import verity_settings as settings
from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits import boolean as BL
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.types import Array
from verity.circuits.ops import fp32 as W
from verity.circuits.kernels import tc as K
from verity.circuits.ops.boolean import elementwise as EW
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import const_bits

from _boolean_checks import assert_live_ands, assert_pinned

DEFS = [EW.F32Add, EW.F32Mul, EW.F32Fma, EW.F32Div, EW.F32AddV4, EW.F32MulV4, EW.F32FmaV4, EW.F32DivV4, EW.Bf16AddF2fp, EW.Bf16Add,
        EW.Bf16MulBf16, EW.Bf16MulF32, EW.I32Add]
F32_DEFS = [EW.F32Add, EW.F32Mul, EW.F32Fma, EW.F32Div, EW.F32AddV4, EW.F32MulV4, EW.F32FmaV4, EW.F32DivV4]
#: the Boolean versions of the `_v2` words (every NaN the GPU's)
GPU_NAN_DEFS = (EW.F32AddV4, EW.F32MulV4, EW.F32FmaV4, EW.F32DivV4)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32Add_v3": (812, 1395, 115, "5deb2e4ea8202a513d24a778bfb36fb5e196c439bb9aed741474bcf601356ee59dbe617cbdb5a4ecc02c962ba216a29d591e229688d4eeea7d243ab81f33efbc"),
    "F32Mul_v3": (2430, 4650, 110, "e685450808ff9329950c551509ca16b35f3a67bad9a5c61b90b8df4c5d5dba65196cf1a12d0865b33d8bfcceb843983832572df11943a5b74a9af0211c395d08"),
    "F32Fma_v3": (4485, 7863, 267, "dc6682804700e950c3d17ae847497dc1f4dc3c5bdacbed763f4cadaa4e2b1c05e12a1842e649f1ee1f5d762629e95f883fd93f39b6f97d61116bd20caf5c45ea"),
    "F32Div_v3": (2517, 5592, 252, "0dec0a0642913945a8b7b730797079ac18b4a41052123fbf2933097f20e26ac988f4510753d840391c9adf0ae246bec123e7a38036a3b27feab60aef2cb326ec"),
    "F32Add_v4": (781, 1331, 145, "1f0dc0d1dfe4f0be91ab937213dd367ab8d2b10be661d52f981f066853577046d36fb9c81fcd3d6a2bb612a0ad1c0933e5dda635e22a24c2eb4e9de0d656c8e6"),
    "F32Mul_v4": (2398, 4562, 140, "67800b6ac99bcd8abc3b48c189a59f7ec1189ec6c40a505f14d07939a96bae309333260a691862c9202c8d493ccf369ef0e1d6a3cf490d4bdc5541ef4f7cdd35"),
    "F32Fma_v4": (4485, 7863, 289, "1a4c901385389d07734d2f85da3d681a44055e7eb69e46dadef00dc49e00569b0002c66f87b5be9b488a007cd55c24f7d7960956aa39e2a8f018a31f449369a5"),
    "F32Div_v4": (2485, 5497, 282, "ddb7d2a83bb387cf98211c08366a18ee95456cdfb9221798028df7159cb803e4f5a729c08e89d8cecd534075fb2435b0d172765bd9540f66b06a5b1a4d76c983"),
    "Bf16AddF2fp_v2": (717, 1061, 122, "5ac1d8ff17a79b9603e960f67fea8107157f51f3aeb075611661a378370a1cdebd47f26fdc4ed54a74926113b4eeea3a409fecf8a598d53ac226939dbbfde4a9"),
    "Bf16Add_v2": (717, 1061, 116, "5d1412221c5b2e0c1ca563049097ea12a3af1dfc68a06d0e334a4dcece8306ce8374f33a735be6f29bb81de0fbe24e9667c564d74df300bcc6e9aafecccb721f"),
    "Bf16MulBf16_v2": (799, 1155, 99, "9d8c73c03d6430b014ad4925018ecf576ba4b6c9bbe0a1021984df3ce132cf846763b1ecbebe1249f5a7d1baa3fc909cc6554fa1c63c3ad4aa703c0c4d8db027"),
    "Bf16MulF32_v2": (1220, 2042, 112, "813e36a7f3961a8b227a7be5013f52759f773031384a32eee4d2d1af8f5dd1261a3c571feb669f7ed0945b025ea65cc133adc4780390750b205469ed6c71e4ae"),
    "I32Add_v2": (31, 153, 0, "71fd0abd870a47fb9f9679bf126866aee21ff6b3c8a0f8dcaf27035072da9b884ecbf20628f8a97a6e51e85e79d7c97ddb8ca88610b9beba71efe1b57b75a8e7"),
}

#: signed zeros, subnormals (the smallest, the largest, and around the gradual-underflow boundary), the normal range's ends,
#: ones and their neighbours, powers of two that scale into and out of the subnormal range, infinities, and NaNs of both
#: kinds (signalling and quiet, the smallest and largest payloads)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x00FFFFFF,
               0x01000000, 0x0C000000, 0x33800000, 0x34000000, 0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x3FC00000,
               0x40000000, 0x4B000000, 0x4B800000, 0x7E800000, 0x7F000000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FBFFFFF,
               0x7FC00000, 0x7FFFFFFF]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _words(rng, n: int, lo: int, hi: int) -> np.ndarray:
    """Random f32 words of either sign whose exponent field is in [lo, hi)."""
    e = rng.integers(lo, hi, n, dtype=np.uint64) << np.uint64(23)
    m = rng.integers(0, 1 << 23, n, dtype=np.uint64)
    s = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(31)
    return (s | e | m).astype(np.uint32)


def _f32_inputs(n_params: int, rng) -> list[np.ndarray]:
    """Every tuple of special words; 60k uniform words; 20k words of exponent fields in [0, 24) (subnormal and tiny) or [100, 154)
    (around 1, so a product, quotient or fused sum crosses into the subnormal range); and 20k tuples whose second word is the
    first's negation moved by a few ulps (cancellation down to subnormal and zero sums)."""
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    pick = rng.integers(0, 2, (n_params, 20_000)).astype(bool)
    near = np.where(pick, _words(rng, (n_params, 20_000), 0, 24), _words(rng, (n_params, 20_000), 100, 154))
    a = np.concatenate([_words(rng, 10_000, 0, 30), _words(rng, 10_000, 0, 255)])
    b = (a.astype(np.int64) + rng.integers(-4, 5, a.size)).clip(0, 0xFFFFFFFF).astype(np.uint32) ^ np.uint32(0x80000000)
    canc = [a, b] + [_words(rng, a.size, 0, 255) for _ in range(n_params - 2)]
    return [np.concatenate([SPECIALS[g], r, x, c]) for g, r, x, c in zip(grid, rand, near, canc)]


def _agree(target, inputs) -> None:
    word = TR.word_of(target)
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("d", DEFS, ids=lambda d: f"{d.name}_v{d.version}")
def test_pinned_counts_and_digests(d) -> None:
    assert_pinned(bind(d), PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: f"{d.name}_v{d.version}")
def test_the_ands_are_the_builders_live_ands(d) -> None:
    assert_live_ands(d, PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: f"{d.name}_v{d.version}")
def test_its_word_view_is_its_word_primitive(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 2 if d in GPU_NAN_DEFS else 1)
    (params, ret), (word_params, word_ret) = E.batch.signature_of(target), E.batch.signature_of(word)
    assert [BL.word_type(t) for t in params] == list(word_params) and BL.word_type(ret) == word_ret


@pytest.mark.parametrize("d", F32_DEFS, ids=lambda d: f"{d.name}_v{d.version}")
def test_the_f32_operations_agree_with_their_word_primitives(d) -> None:
    _agree(bind(d), _f32_inputs(len(d.signature(None)[0]), np.random.default_rng(zlib.crc32(d.name.encode()))))


_BF16_MAGNITUDES = [0x0000, 0x0001, 0x0002, 0x003F, 0x0040, 0x007F, 0x0080, 0x0081, 0x00FF, 0x0100, 0x3F00, 0x3F7F, 0x3F80,
                    0x3F81, 0x3FC0, 0x4000, 0x4B00, 0x7F00, 0x7F7E, 0x7F7F, 0x7F80, 0x7F81, 0x7FBF, 0x7FC0, 0x7FFF]
BF16_SPECIALS = np.array([m | s for m in _BF16_MAGNITUDES for s in (0, 0x8000)], np.uint16)


def _edge_exponents(rng, ea: np.ndarray) -> np.ndarray:
    """For operands of exponent fields ``ea`` (bf16's and f32's are the same 8 bits), the other operand's exponent fields chosen so
    that the product's is in -26..3 (f32's subnormal range and below, rounded to f32 and then to bf16) or in 250..257 (around
    f32's overflow)."""
    target = np.where(rng.integers(0, 2, ea.size) == 1, rng.integers(-26, 4, ea.size), rng.integers(250, 258, ea.size))
    return np.clip(target - ea + 127, 0, 254)


@pytest.mark.parametrize("d", [EW.Bf16AddF2fp, pytest.param(EW.Bf16Add, marks=pytest.mark.slow),
                               pytest.param(EW.Bf16MulBf16, marks=pytest.mark.slow)], ids=lambda d: d.name)
def test_the_bf16_operations_agree_with_their_word_primitives(d) -> None:
    """Every pair of special words, every word against each of 8 specials (both orders), random pairs, pairs that cancel to a
    few ulps (subnormal and zero sums) and pairs whose product lands at f32's subnormal or overflow end."""
    rng = np.random.default_rng(21)
    every = np.arange(1 << 16, dtype=np.uint16)
    some = BF16_SPECIALS[[0, 1, 5, 7, 16, 19, 20, 23]]
    grid = np.array(list(itertools.product(range(len(BF16_SPECIALS)), repeat=2)), np.int64).T
    a = np.concatenate([BF16_SPECIALS[grid[0]], np.repeat(every, len(some)), np.tile(some, len(every)),
                        rng.integers(0, 1 << 16, 60_000).astype(np.uint16)])
    b = np.concatenate([BF16_SPECIALS[grid[1]], np.tile(some, len(every)), np.repeat(every, len(some)),
                        rng.integers(0, 1 << 16, 60_000).astype(np.uint16)])
    c = rng.integers(0, 1 << 15, 20_000).astype(np.int64)
    d_ = ((c + rng.integers(-3, 4, c.size)).clip(0, 0x7FFF) | (1 << 15)).astype(np.uint16)
    p = rng.integers(0, 1 << 16, 20_000).astype(np.int64)
    q = rng.integers(0, 2, p.size) << 15 | _edge_exponents(rng, p >> 7 & 0xFF) << 7 | rng.integers(0, 128, p.size)
    _agree(bind(d), [np.concatenate([a, c.astype(np.uint16), p.astype(np.uint16)]),
                     np.concatenate([b, d_, q.astype(np.uint16)])])


@pytest.mark.slow
def test_the_bf16_by_f32_multiply_agrees_with_bf16_mul_f32_v1() -> None:
    """Every bf16 word against each of 12 special f32 words, every pair of special words (bf16 by f32), random pairs, and pairs
    whose product lands at f32's subnormal or overflow end."""
    rng = np.random.default_rng(24)
    every = np.arange(1 << 16, dtype=np.uint16)
    some = SPECIALS[[0, 1, 5, 8, 12, 15, 17, 23, 26, 27, 32, 49]]
    grid = np.array(list(itertools.product(range(len(BF16_SPECIALS)), range(len(SPECIALS)))), np.int64).T
    p = rng.integers(0, 1 << 16, 20_000).astype(np.int64)
    q = (rng.integers(0, 2, p.size).astype(np.uint64) << np.uint64(31)
         | _edge_exponents(rng, p >> 7 & 0xFF).astype(np.uint64) << np.uint64(23)
         | rng.integers(0, 1 << 23, p.size).astype(np.uint64))
    a = np.concatenate([np.repeat(every, len(some)), BF16_SPECIALS[grid[0]], rng.integers(0, 1 << 16, 60_000).astype(np.uint16),
                        p.astype(np.uint16)])
    s = np.concatenate([np.tile(some, len(every)), SPECIALS[grid[1]],
                        rng.integers(0, 1 << 32, 60_000, dtype=np.uint64).astype(np.uint32), q.astype(np.uint32)])
    _agree(bind(EW.Bf16MulF32), [a, s])


def test_a_shared_operand_split_is_the_operation() -> None:
    """`shared_operand` cuts `Bf16MulBf16_v2`'s builder at s: the two parts have exactly the builder's gates, the first reads s
    alone, and the second over (x, s, the first's frontier) is `Bf16MulBf16_v1` on every x against 4 special words and on random
    pairs."""
    own, rest = EW.shared_operand(("SharedOwnCheck", "SharedRestCheck"), 1, EW.Bf16MulBf16.build, (("x", EW.BF16), ("s", EW.BF16)),
                                  EW.BF16, 1, ("", ""), register=False)
    whole = EW.Bf16MulBf16.traced().counts
    assert {k: own.traced().counts[k] + rest.traced().counts[k] for k in whole} == whole
    assert [n for n, _ in own.signature(None)[0]] == ["s"] and [n for n, _ in rest.signature(None)[0]] == ["x", "s", "sd"]
    rng = np.random.default_rng(25)
    every = np.arange(1 << 16, dtype=np.uint16)
    some = BF16_SPECIALS[[1, 7, 16, 23]]
    x = np.concatenate([np.repeat(every, len(some)), rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    s = np.concatenate([np.tile(some, len(every)), rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    got = np.asarray(BS.evaluate_words(rest, x, s, BS.evaluate_words(own, s))).reshape(-1)
    assert np.array_equal(got, np.asarray(E.evaluate_batch(W.Bf16MulBf16, x, s)).reshape(-1))


@pytest.mark.parametrize("element", [BL.BIT, Array(1, BL.BIT)], ids=["word", "bits"])
def test_a_shared_specialized_split_is_the_specialized_operation(element) -> None:
    """`shared_specialized` cuts ``x * bf16(s + K)`` (K a bf16 static, folded) at s for each K: the two parts have exactly the
    `specialized` circuit's gates, the first reads s alone, and the second over (x, s, the first's frontier, one word or 1-bit
    words) is the `specialized` Definition on every x against 4 special words and on random pairs."""
    def build(S):
        return lambda C, x, s: EW.Bf16MulBf16.build(C, x, EW.bf16_add(C, s, const_bits(S.K, 16)))

    def sig(S):
        return (("x", EW.BF16), ("s", EW.BF16)), EW.BF16

    whole = EW.specialized("SharedSpecializedWholeCheck", 1, ["K"], build, sig, register=False)
    own, rest = EW.shared_specialized(("SharedSpecializedOwnCheck", "SharedSpecializedRestCheck"), 1, ["K"], build, sig, 1, ("", ""),
                                      register=False, element=element)
    rng = np.random.default_rng(26)
    every = np.arange(1 << 16, dtype=np.uint16)
    some = BF16_SPECIALS[[1, 7, 16, 23]]
    x = np.concatenate([np.repeat(every, len(some)), rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    s = np.concatenate([np.tile(some, len(every)), rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    for k in (0x3F80, 0xC1F0):
        S = SimpleNamespace(K=k)
        counts = whole.traced(S).counts
        assert {g: own.traced(S).counts[g] + rest.traced(S).counts[g] for g in counts} == counts
        assert [n for n, _ in own.signature(S)[0]] == ["s"] and [n for n, _ in rest.signature(S)[0]] == ["x", "s", "sd"]
        assert own.signature(S)[1] == Array(len(own.traced(S).outputs), element)
        got = np.asarray(BS.evaluate_words(bind(rest, K=k), x, s, BS.evaluate_words(bind(own, K=k), s))).reshape(-1)
        assert np.array_equal(got, np.asarray(BS.evaluate_words(bind(whole, K=k), x, s)).reshape(-1))
    assert own.traced(SimpleNamespace(K=0x3F80)).gates != own.traced(SimpleNamespace(K=0xC1F0)).gates


@pytest.mark.parametrize("element", [BL.BIT, Array(1, BL.BIT)], ids=["word", "bits"])
def test_a_shared_specialized_split_by_element_is_the_specialized_operation(element) -> None:
    """`shared_specialized` with ``by_element`` cuts ``bf16(x_j + K) * s`` over x of 3 bf16 words at each x_j: the first part is
    one word's circuit, its 3 Calls and the rest have exactly the `specialized` circuit's gates, and the rest over (x, s, the 3
    frontiers) is the `specialized` Definition on random words; a builder whose x gates read two words refuses."""
    X3 = Array(3, EW.BF16)

    def build(S):
        return lambda C, *e: [EW.Bf16MulBf16.build(C, EW.bf16_add(C, xj, const_bits(S.K, 16)), e[3]) for xj in e[:3]]

    def sig(S):
        return (("x", X3), ("s", EW.BF16)), X3

    whole = EW.specialized("ByElementWholeCheck", 1, ["K"], build, sig, register=False)
    own, rest = EW.shared_specialized(("ByElementOwnCheck", "ByElementRestCheck"), 1, ["K"], build, sig, 0, ("", ""),
                                      register=False, element=element, by_element=True)
    rng = np.random.default_rng(27)
    n = 30_000
    x = rng.integers(0, 1 << 16, (n, 3)).astype(np.uint16)
    x[:len(BF16_SPECIALS), 1] = BF16_SPECIALS
    s = rng.integers(0, 1 << 16, n).astype(np.uint16)
    for k in (0x3F80, 0xC1F0):
        S = SimpleNamespace(K=k)
        counts = whole.traced(S).counts
        assert {g: 3 * own.traced(S).counts[g] + rest.traced(S).counts[g] for g in counts} == counts
        dt = Array(len(own.traced(S).outputs), element)
        assert own.signature(S) == ((("x", EW.BF16),), dt) and rest.signature(S)[0][-1] == ("xd", Array(3, dt))
        xd = np.asarray(BS.evaluate_words(bind(own, K=k), x.reshape(-1))).reshape(n, -1)
        got = np.asarray(BS.evaluate_words(bind(rest, K=k), x, s, xd)).reshape(n, 3)
        assert np.array_equal(got, np.asarray(BS.evaluate_words(bind(whole, K=k), x, s)).reshape(n, 3))
    mixed = EW.shared_specialized(("MixedOwnCheck", "MixedRestCheck"), 1, ["K"],
                                  lambda S: lambda C, *e: [EW.Bf16MulBf16.build(C, EW.bf16_add(C, e[0], e[1]), e[3])] * 3, sig, 0,
                                  ("", ""), register=False, by_element=True)[0]
    with pytest.raises(ValueError, match="reads elements"):
        mixed.traced(SimpleNamespace(K=0))


def test_a_specialized_constant_is_folded() -> None:
    """`specialized` folds the f32 word K into `bf16_mul_f32`: no gate reads a constant wire, it has fewer ANDs than
    `Bf16MulF32_v2`, and on every bf16 word it is `Bf16MulF32_v1(x, K)` at K = 30.0 and K = RN32(1 / 30)."""
    d = EW.specialized("SpecializedCheck", 1, ["K"], lambda S: lambda C, x: EW.bf16_mul_f32(C, x, const_bits(S.K, 32)),
                       lambda S: ((("x", EW.BF16),), EW.BF16), register=False)
    every = np.arange(1 << 16, dtype=np.uint16)
    for word in (0x41F00000, 0x3D088889):
        target = bind(d, K=word)
        T = d.traced(SimpleNamespace(K=word))
        assert all(o >= 2 for _, ops in T.gates for o in ops)
        assert T.counts["and"] < PINS["Bf16MulF32_v2"][0]
        got = np.asarray(BS.evaluate_words(target, every)).reshape(-1)
        assert np.array_equal(got, np.asarray(E.evaluate_batch(W.Bf16MulF32, every, np.full(every.shape, word, np.uint32))).reshape(-1))


def test_a_composite_on_bits_has_the_word_composite_as_its_word_view() -> None:
    """`on_bits` over a row of `Bf16Add_v2` takes its doc from the body and its word view from the word composite bound at the same
    N, whose signature it has on words and whose values it has on random rows."""
    def row(B, S, a, b):
        """a row of sums"""
        return B.batch(EW.Bf16Add, a, b, axes=(0, 0))

    def word_row(B, S, a, b):
        return B.batch(W.Bf16Add, a, b, axes=(0, 0))

    sig = lambda T: lambda S: ((("a", Array(S.N, T)), ("b", Array(S.N, T))), Array(S.N, T))  # noqa: E731
    word = CompositeDefinition("OnBitsWordCheck", 1, ("N",), sig(W.BF16), word_row, register=False)
    d = EW.on_bits("OnBitsCheck", 2, ("N",), sig(EW.BF16), row, word, register=False)
    target = bind(d, N=5)
    assert d.doc == "a row of sums" and TR.word_of(target).id == bind(word, N=5).id
    (params, ret), (word_params, word_ret) = E.batch.signature_of(target), E.batch.signature_of(TR.word_of(target))
    assert [BL.word_type(t) for t in params] == list(word_params) and BL.word_type(ret) == word_ret
    rng = np.random.default_rng(26)
    _agree(target, [rng.integers(0, 1 << 16, (2_000, 5)).astype(np.uint16) for _ in range(2)])


def test_on_bits_names_a_static_free_word_itself_and_reads_a_lifted_static_through_its_word_view() -> None:
    """A static-free composite on bits names the word composite itself (a Definition, which the switch keys by id); a function
    static in ``lifted``, bound to a Boolean Definition, appears in the word view as that Definition's word."""
    def pair(B, S, a, b):
        return B.call(EW.Bf16Add, a, b)

    def word_pair(B, S, a, b):
        return B.call(W.Bf16Add, a, b)

    sig = lambda T: lambda S: ((("a", T), ("b", T)), T)  # noqa: E731
    word = CompositeDefinition("OnBitsStaticFreeWord", 1, (), sig(W.BF16), word_pair, register=False)
    d = EW.on_bits("OnBitsStaticFree", 2, (), sig(EW.BF16), pair, word, register=False)
    assert isinstance(d.word, SpecializedDefinition) and TR.word_of(bind(d)).id == bind(word).id

    def apply(B, S, a, b):
        return B.call(S.ADD, a, b)

    word = CompositeDefinition("OnBitsLiftedWord", 1, ("ADD",), sig(W.BF16), apply, register=False)
    d = EW.on_bits("OnBitsLifted", 2, ("ADD",), sig(EW.BF16), apply, word, register=False, lifted=("ADD",))
    target = bind(d, ADD=EW.Bf16Add)
    assert TR.word_of(target).id == bind(word, ADD=W.Bf16Add).id
    rng = np.random.default_rng(27)
    _agree(target, [rng.integers(0, 1 << 16, 2_000).astype(np.uint16) for _ in range(2)])


_I32_SPECIALS = np.array([0, 1, 2, 0x7FFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF, 0x0000FFFF, 0x00010000,
                          0x55555555, 0xAAAAAAAA] + [(1 << k) - 1 for k in range(3, 32, 4)] + [1 << k for k in range(3, 32, 4)],
                         np.uint32)


def test_the_i32_add_agrees_with_i32_add_v1() -> None:
    """Every pair of special words (full carry chains among them) and 100k random pairs."""
    rng = np.random.default_rng(22)
    grid = np.array(list(itertools.product(range(len(_I32_SPECIALS)), repeat=2)), np.int64).T
    a, b = (np.concatenate([_I32_SPECIALS[g], rng.integers(0, 1 << 32, 100_000, dtype=np.uint64).astype(np.uint32)]) for g in grid)
    _agree(bind(EW.I32Add), [a, b])


def _bf16_add_f2fp_words(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`Bf16AddF2fp_v1` on arrays: the f32 sum of the widened words, rounded by F2FP's batch kernel. Every NaN sum gives 0x7FFF,
    so which NaN numpy's vector add returns does not matter."""
    s = ((a.astype(np.uint32) << np.uint32(16)).view(np.float32) + (b.astype(np.uint32) << np.uint32(16)).view(np.float32))
    return K.f2fp_bf16_batch(s.view(np.uint32))[0].astype(np.uint16)


def test_the_array_form_of_bf16_add_f2fp_is_its_evaluator() -> None:
    """The exhaustive check's reference is `Bf16AddF2fp_v1`'s evaluator on the special pairs and on random pairs."""
    rng = np.random.default_rng(23)
    grid = np.array(list(itertools.product(range(len(BF16_SPECIALS)), repeat=2)), np.int64).T
    a = np.concatenate([BF16_SPECIALS[grid[0]], rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    b = np.concatenate([BF16_SPECIALS[grid[1]], rng.integers(0, 1 << 16, 20_000).astype(np.uint16)])
    assert np.array_equal(_bf16_add_f2fp_words(a, b), [W.Bf16AddF2fp.evaluate(int(x), int(y)) for x, y in zip(a, b)])


def _counter_rows(lo_bits: int, hi: np.ndarray, words: int) -> np.ndarray:
    """Bit-sliced rows of the instance counter i in [0, 64 * words): ``lo_bits`` rows of i's low bits, then rows of the bits of
    ``hi`` (a constant per row) above them."""
    w = np.arange(words, dtype=np.uint64)
    rows = []
    for j in range(lo_bits):
        if j < 6:
            pat = sum(1 << i for i in range(64) if (i >> j) & 1)
            rows.append(np.full(words, pat, np.uint64))
        else:
            rows.append(np.uint64(0) - ((w >> np.uint64(j - 6)) & np.uint64(1)))
    rows += [np.full(words, ~np.uint64(0) if b else np.uint64(0), np.uint64) for b in hi]
    return np.stack(rows)


def test_the_bf16_add_agrees_with_bf16_add_f2fp_v1_on_every_b_for_some_a() -> None:
    """The exhaustive check's chunks where a is +0 or a small subnormal, near 1, -0 or a negative subnormal, and +inf or a NaN:
    every b against each of those 64 words of a."""
    _every_pair_chunks([0x000, 0x3F8, 0x800, 0x7F8])


@pytest.mark.skipif(not settings.VERITY_ML_BOOLEAN_EXHAUSTIVE,
                    reason="every pair of bf16 words (2^32 instances, about 7 minutes); set VERITY_ML_BOOLEAN_EXHAUSTIVE=1")
def test_the_bf16_add_agrees_with_bf16_add_f2fp_v1_on_every_pair() -> None:
    """All 2^32 operand pairs, 2^20 per chunk: b and a's low 4 bits are the instance counter, a's high 12 bits the chunk."""
    _every_pair_chunks(range(1 << 12))


def _every_pair_chunks(chunks) -> None:
    c = BS.compile(bind(EW.Bf16AddF2fp))
    words = (1 << 20) // 64
    b = np.tile(np.arange(1 << 16, dtype=np.uint16), 16)
    a_lo = np.repeat(np.arange(16, dtype=np.uint32), 1 << 16)
    for hi in chunks:
        rows = _counter_rows(20, [(hi >> k) & 1 for k in range(12)], words)
        got = BS._run(c, np.concatenate([rows[16:], rows[:16]]))          # leaves: a's 16 bits, then b's
        want = _bf16_add_f2fp_words(((hi << 4) | a_lo).astype(np.uint16), b)
        want_rows = np.stack([np.packbits(((want >> np.uint16(k)) & 1).astype(np.uint8), bitorder="little").view("<u8")
                              for k in range(16)])
        bad = np.flatnonzero((got != want_rows).any(axis=0))
        assert not bad.size, f"a >> 4 = {hi:#05x}: {bad.size} of {words} words differ, first {bad[:5]}"
