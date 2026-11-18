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
    "F32Add_v3": (812, 1395, 115, "d8e105be18d77cc9a4a8f695908a5444c3730fa436d6e80039d91afa0f588ce383bdc8a922f5f45d2c9ddddbd25057c70a99dec583b051a03b52c953463e7ba3"),
    "F32Mul_v3": (2430, 4650, 110, "77d1d3d129c059b2e3238552d579b48c6c29df58d68748cd45b213e28f6445d869da148e09882523590a22d2380f57621fb7460d1cb93f6d1ee7e0856a4904cb"),
    "F32Fma_v3": (4485, 7863, 267, "6875f0f1ceaaf132f9fbd5f9f2a21a9b8ee2ef8b89618cff55af6bfc795caba8587bdb65627bd86b1b82d1a85f57b6d392488ac79dce57abff545c94365b1e64"),
    "F32Div_v3": (2517, 5592, 252, "00637a68217f42faf939501b2c4d41899050ec1678fb896345481c5fb37be983690a41e1cae3f0d4edce8f7c3317842bb41129c3571d1e8f624fed0d82ff5c2e"),
    "F32Add_v4": (781, 1331, 145, "a9e9aa84b6a08c1c2ca4d42788bd30011b29756e4a61633a682eae9e8dadd11049e4752075b06463951621fc39fb5b67247d3433a4b0cabb0acfeac222d1e7f3"),
    "F32Mul_v4": (2398, 4562, 140, "9a291f9c69cdd96152da624128f6a382c455d5631ab6c1909d31e8f01f660d052e4b743c0ad30bdae2740e1f4da83b8f0b248b106976c80b7917d6ff1d4258ef"),
    "F32Fma_v4": (4485, 7863, 289, "eef8af99665eb057720b049dd28441b5e06b4c6a63e98bde85953a2f61224812a783da370e1393889ae220084b116e9e309f345654dcc12d6cc283831f83d8f8"),
    "F32Div_v4": (2485, 5497, 282, "3dfbfb84759ce251f055ae1786ce283d4c7a369777db9b6af9edcce381e5f1580994b12d231bb88d9b3ec9e67eb47c7042de159442f26ef3acc6d1442068cde0"),
    "Bf16AddF2fp_v2": (717, 1061, 122, "ee71f2d1598afb06dafd5cef930e3ede2f41fc5c494026c9547ce9f4d02901f091cf9398ecebe1665fd8ad929b1576d7d5a76aca726f2ff1feb566a566f1172a"),
    "Bf16Add_v2": (717, 1061, 116, "a025d6a895c33027d1d475bc7dc8284a4f85e5ea5b604cbf46c13999529b584239528697df78bb16709a76c5dc48d0d93935f1bfd61c117bbe070d1e01cbc3bd"),
    "Bf16MulBf16_v2": (799, 1155, 99, "bacdc1448624db3b5df53f7f78386fd2fbccb57bfddf8cd06dfbf2b26db99e21a7170167635a41519056d181a07def1171f7d6735de1a8d6b39276532249305c"),
    "Bf16MulF32_v2": (1220, 2042, 112, "3b094c25eb19d197f18f938ac26bc09b8b612941da17a1d499c8963ea10d1bd69c00f6e5a6e194aca8f50349819108b641d2231788728494019303936a888428"),
    "I32Add_v2": (31, 153, 0, "7bf24c92b25a494103f96004e68e80a090aa0789d1b73d034b776833864d305a642df933fecf61fec0011dccf9eac467da477d132e8aa4bbaada5f514dd601a8"),
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
