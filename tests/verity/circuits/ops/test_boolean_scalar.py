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
    "Bf16GtStrict_v2": (56, 133, 19, "c38a3bfafc0cc34992e47beb54dc094da1368a565821f34e9b2910032fe20a666071aa9864dca6ccf036752dc746652f18eee668dffdea80442ab52d209601fe"),
    "SelectBf16_v2": (16, 32, 0, "53a9c498867c827a7cec660ed2f034033497a90c1e32e57d790e9807ee4aacb00afa679009a48eff7c49eebacd9770cc4f726914efed27a72f0879fa46bbdc93"),
    "SelectI32_v2": (32, 64, 0, "fdfd1cdc137afa6d9d40a1ced87eceb14a0c1ea3d0877c27ee144371acb1d95f54486f5982d943bc3ea2e2a597c38aa16883838dccda599c8bf85470e7755862"),
    "SelectF32_v2": (32, 64, 0, "39d6b6e3664b02f23626733a0b58a83118c3072a849bf868f5e0b7b7060c74c009d4b4aef0b61a864d4324ef8c880d86018d0dd3300eb0c638b77179596072a6"),
    "F32Eq_v2": (117, 168, 35, "71c399a28e8dc11e63d1feede33597f94e69bc4cb51efce2bb636fdcf2bb5bc75ea14f1dec3420069b692d62e99ee4d8c9c4b9fdb85883bb26b849385a7b18fc"),
    "F32Add_v3": (812, 1395, 115, "5deb2e4ea8202a513d24a778bfb36fb5e196c439bb9aed741474bcf601356ee59dbe617cbdb5a4ecc02c962ba216a29d591e229688d4eeea7d243ab81f33efbc"),
    "F32Mul_v3": (2430, 4650, 110, "e685450808ff9329950c551509ca16b35f3a67bad9a5c61b90b8df4c5d5dba65196cf1a12d0865b33d8bfcceb843983832572df11943a5b74a9af0211c395d08"),
    "F32Fma_v3": (4485, 7863, 267, "dc6682804700e950c3d17ae847497dc1f4dc3c5bdacbed763f4cadaa4e2b1c05e12a1842e649f1ee1f5d762629e95f883fd93f39b6f97d61116bd20caf5c45ea"),
    "F32Div_v3": (2517, 5592, 252, "0dec0a0642913945a8b7b730797079ac18b4a41052123fbf2933097f20e26ac988f4510753d840391c9adf0ae246bec123e7a38036a3b27feab60aef2cb326ec"),
    "F32Fabs_v2": (0, 0, 0, "475b93370ad25e9aa5efdb40025ad9f2c14c32477bdf2a412bc717c576f6c42820aac88b5a6783fd1a0eaeb71d7a8ecb214591028963f8ec6305ecdbe5b64d70"),
    "F32Fmaxf_v2": (314, 573, 102, "9d46646b35c3b6ecf362ef2cab8eca1b64230a16d8be4d487f4385d096a1d18ac78e5a8c7a4dfbf4cb2d916aa3026dd69dcb0e49b6eb73a4b8b88de5e196d83a"),
    "F32Fminf_v2": (314, 574, 102, "5b4c836e5c25febcd508aa388632c896303fab8cf6eb29605273ad93d0bf89bca5bd2a849bc00c90d2ad303b5cb4e3e5dad6b1f62fb5f94dd61847dc457e6846"),
    "Bf16ToF32_v2": (0, 0, 0, "10a89d045c1b43927914a3563f4f64775d302072d76f36511e4744a6cd022c22e42e3505ac87c697cf6deaef1775f7dcb2a8ae1444340175bba73538392e6d93"),
    "F32ToE4m3Sat_v2": (231, 376, 34, "b3480c16f430d8cb4e95dcca65d28d98ed810d0190da3bf864fcb3678ddb589df3e8f86001ffd9682d93e76934d60017611148a97f55f05a374f90fbc9ee3151"),
    "E4m3ToF32_v2": (15, 22, 4, "0651ba00bbda2c30c96ac442060828d74577f7e5cc1050c2d365759929346fb2cf0a5483972a8e64e5eb73a9350041c483a77b5f444a507fe1f9fb4197bf2a21"),
    "F32ToE2m1Sat_v2": (60, 86, 2, "6e774ac9b34159840599635bc4e4ac417f1e59355f43c61c71e28d821e79a35953a616032d5e3e5571aaf08e457293aba322503253355b46b83c1126a3d1da89"),
    "BitAnd_v2": (1, 0, 0, "1f45f502f3f0759a656e37bb235eba7df6ae1883c1ec03df089895338cd086bb4eeee5482df2e2a717cc26a64ef9434f550ea5c3172d158e3a64927cff3ba838"),
    "F32ToS8RniSat_v2": (313, 482, 45, "f13749672cf84de6a69bd7bec5050deb0591c1e367d78a739226d1f50546a1755eac3818598303114ba28e49ac4f424f6e751dc71f259097978db629c9207414"),
    "U32Hi16_v2": (0, 0, 0, "387d4fb81dbe06622f831b059ab8adeac8d68d485e215907c337c4af2265eaca4a0df21140687fc41a08f1ce8acab984250d0ea9e83503e46e9df0bd0c5a3c91"),
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
