"""`verity.circuits.ops.boolean.scalar`: the bf16 comparator, the selects and the bit AND agree with their word primitives on every input
(the comparator on all 2^32 pairs of words, each select over its whole truth table); the f32 equality, arithmetic, max / min, casts
and 16-bit halves agree with theirs (`trace.word_of`) on every tuple of special words and on random words (the bf16 and E4M3
widenings on every word, the E4M3 cast on every exponent around its range, the E2M1 cast on every boundary of its grid and on
all 2^32 words, the s8 cast on every rounding position of its range);
each carries exactly its builder's live ANDs and is pinned by gate count and program digest."""
from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.evaluation import bits as BS
from verity.circuits import boolean as BL
from verity.circuits.defs import bind
from verity.circuits.types import Array
from verity.circuits.ops import prims as P
from verity.circuits.ops import scalar as S
from verity.circuits.ops.boolean import scalar as BSC
from verity.circuits.boolean import trace as TR
from verity.circuits.silicon.cast import E2M1_UP

from _boolean_checks import assert_live_ands, assert_pinned

DEFS = [BSC.Bf16GtStrict, BSC.SelectBf16, BSC.SelectI32, BSC.SelectF32, BSC.F32Eq, BSC.F32Add, BSC.F32Mul, BSC.F32Fma, BSC.F32Div, BSC.F32Fabs, BSC.F32Fmaxf,
        BSC.F32Fminf, BSC.Bf16ToF32, BSC.F32ToE4m3Sat, BSC.E4m3ToF32, BSC.F32ToE2m1Sat, BSC.BitAnd, BSC.F32ToS8RniSat, BSC.U32Hi16]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Bf16GtStrict_v2": (56, 133, 19, "a69f964fab96ea805dee9e0d7a7ab9077415cc9a2836b493c2404224eff40ac9ca530f7b09b8a5072b5de7155b18bc797c819f08eb6b551a9655ca6c5018fe3e"),
    "SelectBf16_v2": (16, 32, 0, "5c4eae8d2c5e12205130cbbc5dc20e52bfc187a935359605a994d0d547ae3259d6d1583c0ac02491f1ba39df69d4bfee65048dc43f9a7351574270d575a23f45"),
    "SelectI32_v2": (32, 64, 0, "aaff91b40ba4bcfd91647a5c2b2750ea31ef502358e1661b1c09a5eaecd2d310bdeeabb2315e24440bdf92a66c9baca18bcab87baaa93b59ecd0f064372c634a"),
    "SelectF32_v2": (32, 64, 0, "1de1b7b9c99b84fdb6e5ac569dfbb43b34b1c653ea0a0969f8491d7ae2b08c20aeddde2ebd004396d46409c378bfa1670959a7d19e900b5d2040dae02b8958db"),
    "F32Eq_v2": (117, 168, 35, "7d60aa2e9375f123c850fda67817017aae299393fd4eda09f353cd125ba567440836eedf8dfd703a9df68397c63781df411a14fe142b6b71d72d409776706dc2"),
    "F32Add_v3": (812, 1395, 115, "d8e105be18d77cc9a4a8f695908a5444c3730fa436d6e80039d91afa0f588ce383bdc8a922f5f45d2c9ddddbd25057c70a99dec583b051a03b52c953463e7ba3"),
    "F32Mul_v3": (2430, 4650, 110, "77d1d3d129c059b2e3238552d579b48c6c29df58d68748cd45b213e28f6445d869da148e09882523590a22d2380f57621fb7460d1cb93f6d1ee7e0856a4904cb"),
    "F32Fma_v3": (4485, 7863, 267, "6875f0f1ceaaf132f9fbd5f9f2a21a9b8ee2ef8b89618cff55af6bfc795caba8587bdb65627bd86b1b82d1a85f57b6d392488ac79dce57abff545c94365b1e64"),
    "F32Div_v3": (2517, 5592, 252, "00637a68217f42faf939501b2c4d41899050ec1678fb896345481c5fb37be983690a41e1cae3f0d4edce8f7c3317842bb41129c3571d1e8f624fed0d82ff5c2e"),
    "F32Fabs_v2": (0, 0, 0, "4626452be3f5a591d672b96f5b726ac15e942b121855eaf3c39906ceac7cf211b2984dcab094fa1e3379a8d4ee2688088c45275ce3ccc4bb7c9985e8e1b42799"),
    "F32Fmaxf_v2": (314, 573, 102, "e3c28876143eddb64f5735b78924f2ab518aceea64aed35738a5eb7e9ff23ba1767a239cf0633d9b5313675f3aa79e6229dea164a767f282058dfc4d106f856a"),
    "F32Fminf_v2": (314, 574, 102, "5e7a7892dc52801c2f6c42e24886579288ac576950f47acc4af548b8b6cbcaa2020c6f19ca61c38ead5ac06de189f009c2a1e7dc7b2ee47af098945b098d8157"),
    "Bf16ToF32_v2": (0, 0, 0, "6711a531584c05bf66f8797cf9cc165b5796a63b1669340396fa47f34e154a1b45a91670a6ecb4cd098ef8f858383542422bf65c09ab4dd8874907082d8ada07"),
    "F32ToE4m3Sat_v2": (231, 376, 34, "7fb65d048cebeaca478653d18885228a97416be4e20014909b4816c64eb7f20eff2d5dd9a78736917acf53bf75253613749a8b57cb959f0860af282bde72bf19"),
    "E4m3ToF32_v2": (15, 22, 4, "62257e1fae9270dd478066ccc058b413d2d12b4e7fab8487d8a8a152ddac85f88f771005d1794965c9b65f628f625d7daf7943c76e3d73aa675f77ac00952d15"),
    "F32ToE2m1Sat_v2": (60, 86, 2, "49d9801a71bf1c6d002f186dec599efbac0160ce1fe6e0cacf6ec6ec3e24e4b538469c4f55abf6ef236b95dedd92739ef0d234a0ffbbe7123703e8e26bf92c6a"),
    "BitAnd_v2": (1, 0, 0, "7719a106965b991e6e396584ccb4f745abadef38619c572ee92049bf2f0438d9f9b32a7a25e36d65a5990f67bebb6e66d0f35c733d1a5d690f46d0a74c7cb82e"),
    "F32ToS8RniSat_v2": (313, 482, 45, "6f4a8031b45a315f8ad3017418652dc880de8dd83df2f38a230bd055606ee96c687e8ddbf3769dfcdb0e5114a9bf1c4ff4ed75e99b6899a922ffd6906d4d5643"),
    "U32Hi16_v2": (0, 0, 0, "3f88751a5f3978e7462d4219df1dbb83027b25a2e2a5d016ea4605ca3ed98cb6e82389ff179bc406a4d90e16db53854098fa6be87865327ce07b4756b17856f5"),
}

#: signed zeros, the smallest subnormals, the largest finite values, the smallest normals, ones, infinities and NaNs (quiet,
#: signalling, all-ones)
BF16_SPECIALS = np.array([m | s for m in (0x0000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF)
                          for s in (0, 0x8000)], np.uint16)

#: signed zeros, subnormals, the normal range's ends, ones, infinities, NaNs of both signs and kinds, and the FP8 linear's
#: constants and boundaries (448, 1 / (448 * 512), the E4M3 grid's ends and rounding midpoints)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x4B000000, 0x7E800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x43E00000, 0x43E80000,
               0x43F00000, 0x3A124925, 0x3C800000, 0x3B000000, 0x3A800000, 0x3A000000]
F32_SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _words(w: int, n: int, rng) -> np.ndarray:
    return rng.integers(0, 1 << w, n, dtype=np.uint64).astype(np.uint16 if w == 16 else np.uint32)


def _f32_inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(F32_SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x3C000000, 0x44000000, (n_params, 20_000), dtype=np.uint64)
            | (rng.integers(0, 2, (n_params, 20_000), dtype=np.uint64) << np.uint64(31))).astype(np.uint32)
    return [np.concatenate([F32_SPECIALS[g], r, x]) for g, r, x in zip(grid, rand, near)]


def _e4m3_range(rng) -> np.ndarray:
    """Every exponent from 2^-20 to past 448 (fields 107..140), each with its mantissa's low 8 bits exhausted under 64 random high
    parts (the rounding position is at or above bit 20), both signs."""
    e = np.arange(107, 141, dtype=np.uint32)
    hi = rng.integers(0, 1 << 15, 64, dtype=np.uint64).astype(np.uint32) << np.uint32(8)
    lo = np.arange(256, dtype=np.uint32)
    m = (hi[:, None] | lo[None, :]).reshape(-1)
    w = (e[:, None] << np.uint32(23) | m[None, :]).reshape(-1)
    return np.concatenate([w, w | np.uint32(0x80000000)])


def _e2m1_edges(rng) -> np.ndarray:
    """Each boundary of the E2M1 grid (`E2M1_UP`: the six rounding ties and 5, where saturation begins) within 4 ulps and with
    each one of its low 21 bits set; ±0, the subnormals' ends, the largest finite word, ±inf and a NaN per payload bit; and
    every exponent field from 118 (below the grid) to 133 (past 6) under each of the fraction's top three bits, its low 20 bits
    0, 1, all ones or one of 32 random patterns; each word with both signs."""
    near = [b + d for b, _ in E2M1_UP for d in range(-4, 5)] + [b | 1 << i for b, _ in E2M1_UP for i in range(21)]
    special = [0x00000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0x7FFFFFFF]
    special += [0x7F800000 | 1 << i for i in range(23)]
    low = np.concatenate([np.array([0, 1, (1 << 20) - 1], np.uint32),
                          rng.integers(0, 1 << 20, 32, dtype=np.uint64).astype(np.uint32)])
    e = np.arange(118, 134, dtype=np.uint32)[:, None, None] << np.uint32(23)
    grid = (e | np.arange(8, dtype=np.uint32)[None, :, None] << np.uint32(20) | low[None, None, :]).reshape(-1)
    w = np.concatenate([np.array(near + special, np.uint32), grid])
    return np.concatenate([w, w | np.uint32(0x80000000)])


def _s8_range(rng) -> np.ndarray:
    """Every exponent from 2^-9 to 2^9 (fields 118..136, where the s8 cast rounds or saturates), each under all 512 patterns of
    the fraction's top nine bits (the rounding position is at or above bit 15, so every tie and its neighbours) with its low 14
    bits 0, 1, all ones or one of four random patterns; both signs."""
    e = np.arange(118, 137, dtype=np.uint32)[:, None, None] << np.uint32(23)
    hi = np.arange(512, dtype=np.uint32)[None, :, None] << np.uint32(14)
    lo = np.concatenate([np.array([0, 1, (1 << 14) - 1], np.uint32), rng.integers(0, 1 << 14, 4, dtype=np.uint64).astype(np.uint32)])
    w = (e | hi | lo[None, None, :]).reshape(-1)
    return np.concatenate([w, w | np.uint32(0x80000000)])


def _inputs(d, rng) -> list[np.ndarray]:
    if d is BSC.Bf16GtStrict:
        grid = np.array(list(itertools.product(range(len(BF16_SPECIALS)), repeat=2)), np.int64).T
        return [np.concatenate([BF16_SPECIALS[g], _words(16, 60_000, rng)]) for g in grid]
    if d in (BSC.SelectBf16, BSC.SelectI32, BSC.SelectF32):
        w = 16 if d is BSC.SelectBf16 else 32
        return [rng.integers(0, 2, 60_000, dtype=np.uint8), _words(w, 60_000, rng), _words(w, 60_000, rng)]
    if d is BSC.Bf16ToF32:
        return [np.arange(1 << 16, dtype=np.uint16)]
    if d is BSC.E4m3ToF32:
        return [np.arange(1 << 8, dtype=np.uint8)]
    if d is BSC.BitAnd:
        return [np.array([0, 0, 1, 1], np.uint8), np.array([0, 1, 0, 1], np.uint8)]
    inputs = _f32_inputs(len(d.signature(None)[0]), rng)
    if d is BSC.F32ToE2m1Sat:
        return [np.concatenate([inputs[0], _e2m1_edges(rng)])]
    if d is BSC.F32ToS8RniSat:
        return [np.concatenate([inputs[0], _s8_range(rng)])]
    return [np.concatenate([inputs[0], _e4m3_range(rng)])] if d is BSC.F32ToE4m3Sat else inputs


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_pinned_counts_and_digests(d) -> None:
    assert_pinned(bind(d), PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    assert_live_ands(d, PINS)


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_agrees_with_its_word_primitive(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 1)
    inputs = _inputs(d, np.random.default_rng(zlib.crc32(d.name.encode())))
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"


# --- the comparator on every pair ------------------------------------------------------------------------------------------

def _tables() -> tuple[np.ndarray, np.ndarray]:
    """`Bf16GtStrict_v1` as two tables over the 2^16 words: a > b iff neither is a NaN and ``order[a] > order[b]``."""
    w = np.arange(1 << 16, dtype=np.int64)
    mag = w & 0x7FFF
    return mag > 0x7F80, np.where(w & 0x8000, -mag, mag)


def test_the_tables_are_the_word_primitive() -> None:
    """The tables against `Bf16GtStrict_v1` itself: every word against each special word both ways (the NaN, sign, zero and
    infinity boundaries), and every number against its successor in the tables' order both ways. The primitive compares
    integers (a strict weak order), so agreeing on successive numbers it agrees on every pair of numbers."""
    nan, order = _tables()
    words = np.arange(1 << 16, dtype=np.uint16)
    x, y = np.repeat(words, len(BF16_SPECIALS)), np.tile(BF16_SPECIALS, 1 << 16)
    chain = words[~nan][np.argsort(order[~nan], kind="stable")]
    a = np.concatenate([x, y, chain[1:], chain[:-1]])
    b = np.concatenate([y, x, chain[:-1], chain[1:]])
    want = ~nan[a] & ~nan[b] & (order[a] > order[b])
    got = E.evaluate_batch(S.Bf16GtStrict, a, b).astype(bool)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{bad.size} of {len(a)} pairs differ, first {[(hex(a[i]), hex(b[i])) for i in bad[:5]]}"


#: lane l of a packed word carries a = 64 * hi + l: row j < 6 of operand a is bit j of the lane number
_LANES = np.arange(64, dtype=np.uint64)
_LANE_BITS = np.array([np.bitwise_or.reduce(((_LANES >> np.uint64(j)) & np.uint64(1)) << _LANES) for j in range(6)], np.uint64)
_CHUNK = 4096


def _all_ones_where(bits: np.ndarray) -> np.ndarray:
    return np.uint64(0) - (bits & np.uint64(1))


@pytest.mark.slow
def test_bf16_gt_strict_on_every_pair_of_words() -> None:
    """All 2^32 pairs (a, b), 64 values of a across each packed word's lanes and b across the words, against `_tables` (about
    30 s): a wrong output on a pair that no special or sampled pair reaches, such as one carry of the key comparison."""
    c = BS.compile(bind(BSC.Bf16GtStrict))
    nan, order = _tables()
    valid = ~nan
    words = np.arange(1 << 16, dtype=np.uint64)
    args = np.empty((32, 1 << 16), np.uint64)
    args[:6] = _LANE_BITS[:, None]
    args[16:] = _all_ones_where(words[None, :] >> np.arange(16, dtype=np.uint64)[:, None])
    bad = []
    for hi in range(1 << 10):
        args[6:16] = _all_ones_where(np.uint64(hi) >> np.arange(10, dtype=np.uint64))[:, None]
        got = np.concatenate([BS._run(c, np.ascontiguousarray(args[:, s:s + _CHUNK]))[0] for s in range(0, 1 << 16, _CHUNK)])
        a = 64 * hi + np.arange(64)
        want = valid[a][:, None] & valid[None, :] & (order[a][:, None] > order[None, :])
        want = np.ascontiguousarray(np.packbits(want, axis=0, bitorder="little").T).view("<u8").reshape(-1)
        diff = np.flatnonzero(got != want)
        if diff.size:
            bad += [(hex(64 * hi + int(l)), hex(int(b))) for b in diff[:3] for l in range(64) if (int(got[b] ^ want[b]) >> l) & 1][:3]
    assert not bad, f"pairs (a, b) that differ from Bf16GtStrict_v1, first {bad[:5]}"


# --- the E2M1 cast on every word -------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_f32_to_e2m1_sat_on_every_f32_word() -> None:
    """All 2^32 words, 64 across each packed word's lanes (bits 0..5), bits 6..20 across the words and bits 21..31 the chunk
    (about 25 s): a wrong code on a word no boundary, edge or sampled word reaches, such as one input of the sticky OR below
    the grid's resolution. Every boundary `F32ToE2m1Sat_v1` compares |x| with (`E2M1_UP` and the NaN test's 0x7F800000) has
    its low 21 bits clear, so on each chunk it is one code at low bits 0 and one code at every other word, which
    `evaluate_batch` gives."""
    assert all(b & 0x1FFFFF == 0 for b, _ in E2M1_UP)
    c = BS.compile(bind(BSC.F32ToE2m1Sat))
    base = np.arange(1 << 11, dtype=np.uint32) << np.uint32(21)
    at0 = E.evaluate_batch(P.F32ToE2m1Sat, base).astype(np.uint64).reshape(-1)
    above = E.evaluate_batch(P.F32ToE2m1Sat, base | np.uint32(1)).astype(np.uint64).reshape(-1)
    words, j = np.arange(1 << 15, dtype=np.uint64), np.arange(4, dtype=np.uint64)
    args = np.empty((32, 1 << 15), np.uint64)
    args[:6] = _LANE_BITS[:, None]
    args[6:21] = _all_ones_where(words[None, :] >> np.arange(15, dtype=np.uint64)[:, None])
    bad = []
    for hi in range(1 << 11):
        args[21:] = _all_ones_where(np.uint64(hi) >> np.arange(11, dtype=np.uint64))[:, None]
        got = BS._run(c, args)
        want = np.repeat(_all_ones_where(above[hi] >> j)[:, None], 1 << 15, axis=1)
        want[:, 0] = want[:, 0] & ~np.uint64(1) | at0[hi] >> j & np.uint64(1)
        diff = np.bitwise_or.reduce(got ^ want, axis=0)
        w = np.flatnonzero(diff)
        if w.size:
            d = int(diff[w[0]])
            bad.append(hex(hi << 21 | int(w[0]) << 6 | (d & -d).bit_length() - 1))
    assert not bad, f"words that differ from F32ToE2m1Sat_v1, first per chunk {bad[:5]}"


# --- f32_key ---------------------------------------------------------------------------------------------------------------

F32_SPECIALS = np.array([m | s for m in (0x00000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0x7F800000,
                                         0x7F800001, 0x7FC00000, 0x7FFFFFFF) for s in (0, 0x80000000)], np.uint32)


def _f32_key_pair(C, a, b):
    na, ka = BSC.f32_key(C, a)
    nb, kb = BSC.f32_key(C, b)
    return [na, nb, BSC.gt_u(C, ka, kb), BSC.gt_u(C, kb, ka)]


F32KeyPair = TR.definition("F32KeyPair", 1, _f32_key_pair, (("a", BSC.I32), ("b", BSC.I32)), Array(4, BL.BIT),
                           register=False)


def test_f32_key_orders_f32_values() -> None:
    """`f32_key`'s NaN flags and the unsigned order of its keys against the f32 values' order (subnormals as they are, +0 and
    -0 equal): every special pair both ways, random pairs, and each word against its neighbours (every carry boundary of the
    sign fold)."""
    rng = np.random.default_rng(3)
    grid = np.array(list(itertools.product(range(len(F32_SPECIALS)), repeat=2)), np.int64).T
    w = _words(32, 100_000, rng).astype(np.uint64)
    a = np.concatenate([F32_SPECIALS[grid[0]], _words(32, 100_000, rng), w.astype(np.uint32), w.astype(np.uint32)])
    b = np.concatenate([F32_SPECIALS[grid[1]], _words(32, 100_000, rng), ((w + 1) & 0xFFFFFFFF).astype(np.uint32),
                        (w ^ 0x80000000).astype(np.uint32)])
    got = BS.evaluate_words(bind(F32KeyPair), a, b).astype(np.int64).reshape(-1)
    mag_a, mag_b = a.astype(np.int64) & 0x7FFFFFFF, b.astype(np.int64) & 0x7FFFFFFF
    va, vb = np.where(a >> 31, -mag_a, mag_a), np.where(b >> 31, -mag_b, mag_b)
    nan_a, nan_b = mag_a > 0x7F800000, mag_b > 0x7F800000
    num = ~nan_a & ~nan_b
    assert np.array_equal(got & 1, nan_a) and np.array_equal(got >> 1 & 1, nan_b)
    assert np.array_equal((got >> 2 & 1)[num], (va > vb)[num]) and np.array_equal((got >> 3 & 1)[num], (vb > va)[num])


# --- the selects on their whole truth table --------------------------------------------------------------------------------

def _supports(c: BS.Compiled) -> list[frozenset[int]]:
    """Each wire's parameter leaves, through the compiled body's gates (wires 0 and 1 the constants, then the leaves)."""
    sup = [frozenset()] * 2 + [frozenset((i,)) for i in range(c.n_params)] + [frozenset()] * (c.size - 2 - c.n_params)
    for step in c.steps:
        assert step[0] == "gates", f"{c.target}: a {step[0]} step in a select"
        for _kind, out, a, b in step[1]:
            for o, x, y in zip(out.tolist(), a.tolist(), b.tolist()):
                sup[o] = sup[x] | sup[y]
    return sup


@pytest.mark.parametrize("d", [BSC.SelectBf16, BSC.SelectI32, BSC.SelectF32], ids=lambda d: d.name)
def test_select_on_its_whole_truth_table(d) -> None:
    """Output bit j of the compiled body reads only c, a_j and b_j, so the 8 assignments of (c, a_j, b_j), made on every bit at
    once, are each output bit's whole truth table; the word primitive (``a if c else b``) is bitwise, and both agree on all 8."""
    target = bind(d)
    c = BS.compile(target)
    w = c.params[1].leaves
    sup = _supports(c)
    for j, o in enumerate(c.ret_idx.tolist()):
        assert sup[o] <= {0, 1 + j, 1 + w + j}, f"{target.id}: output bit {j} reads leaves {sorted(sup[o])}"
    dt = np.uint16 if w == 16 else np.uint32
    ones = (1 << w) - 1
    cs, a_s, b_s = (np.array(v, dt) for v in zip(*((x, y * ones, z * ones) for x, y, z in itertools.product((0, 1), repeat=3))))
    got = BS.evaluate_words(target, cs.astype(np.uint8), a_s, b_s).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(TR.word_of(target), cs.astype(np.uint8), a_s, b_s).astype(np.uint64).reshape(-1)
    assert got.tolist() == ref.tolist() == [z * ones if not x else y * ones for x, y, z in itertools.product((0, 1), repeat=3)]
