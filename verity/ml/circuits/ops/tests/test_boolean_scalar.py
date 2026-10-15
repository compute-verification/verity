"""`verity.ml.circuits.ops.boolean.scalar`: the bf16 comparator, the selects and the bit AND agree with their word primitives on every input
(the comparator on all 2^32 pairs of words, each select over its whole truth table); the f32 equality, arithmetic, max / min and casts
agree with theirs (`trace.word_of`) on every tuple of special words and on random words (the bf16 and E4M3 widenings on every
word, the E4M3 cast on every exponent around its range, the E2M1 cast on every boundary of its grid and on all 2^32 words);
each carries exactly its builder's live ANDs and is pinned by gate count and program digest."""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.ops.boolean import scalar as BSC
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.silicon.cast import E2M1_UP

DEFS = [BSC.Bf16GtStrict, BSC.SelectBf16, BSC.SelectI32, BSC.SelectF32, BSC.F32Eq, BSC.F32Add, BSC.F32Mul, BSC.F32Fma, BSC.F32Div, BSC.F32Fabs, BSC.F32Fmaxf,
        BSC.F32Fminf, BSC.Bf16ToF32, BSC.F32ToE4m3Sat, BSC.E4m3ToF32, BSC.F32ToE2m1Sat, BSC.BitAnd]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Bf16GtStrict_v2": (56, 133, 19, "7ca8aa49ba9b7baa4b9ab176868a7bc0058d4542ae0f5fbb1b37f63ca777f7fc5b557901e21cd9162de35ca12f357e7ed09a88b2003ded3937c1bce272695d11"),
    "SelectBf16_v2": (16, 32, 0, "efa9f34e83b477709aa8b5b53fec1075a9ccc7998289a509a2f87920952af8e79ba3600c02c27f2626e96f6c06d703e9c4d27fe7700a0a65837ec3057df3a229"),
    "SelectI32_v2": (32, 64, 0, "e53a00a25c382116e2bfb11d8a7ab94f5971edf1c1097fa23d3caf0645fdb94e32829ec00a373c914990cb11dfd8cf95dff3203be82bcd414d4aab063733cc9f"),
    "SelectF32_v2": (32, 64, 0, "b0bf84c463743b8d27735050199b1506b2581be38eaf57feb6d106bee380a01128bad22589e018f3a69447ce03028f14744149f66b7d058dbe87a319d093f150"),
    "F32Eq_v2": (117, 168, 35, "fdd60d0a71241f98066d0cd765d466cc77f4bb31437dd499f4aeb4a100410a94585c1a2a59b9d5a1cd60107794ae9434478c1398b9e2b14f7725f4aebacede64"),
    "F32Add_v3": (812, 1395, 115, "f977f860eaea446464733b3698a68a80a1cfa359ff3c49e2db1ea4fc4256cc80e02821083af0702261b52ecd0905809a8945acabd5ee299c9949005d6890a17b"),
    "F32Mul_v3": (2430, 4650, 110, "460ec851f41d5b72ef4d1b3db8fa735af47da1e623aac240bb9b88000baf47ffa4c2640ad4ce655e7d10a7ee93a77c6eca86a9156f5a509170da1af6f44bc8fa"),
    "F32Fma_v3": (4485, 7863, 267, "ca2f047972f3e47ee47974f7630c5c06bc0ed20cd310752b26b45f37914f812242a37dc85949a0a7c21a9c452240f0d0b69e7e40100728183cb4cd4a65476d39"),
    "F32Div_v3": (2517, 5592, 252, "3ad9a8fd80800cb597f9c2917ae4330295ec896c2c68a1329c68e6527d7db44750d24d5afd88db4c450e44ea4d0676c7739735ad881d96fb24d70e545266587d"),
    "F32Fabs_v2": (0, 0, 0, "70ddf8bdea84a31cdec7b5c104a28f2a647fc634c387a0c3f5431d49b201be2a7c5713a3e5fdfe677788416e216a63177bea7e04d05998d3b4addd26d7e4d4ba"),
    "F32Fmaxf_v2": (314, 573, 102, "685fd1251e7012a75093b9e8710cd8b32772520dee2a8f9cc0ebe576986c2d630fab8d862f935346f1817db8650ec2c4581a308d225c31df239cd7b580818f83"),
    "F32Fminf_v2": (314, 574, 102, "7284d8b1ad48ee3735e6d89f06b7e4bd9f583a9203b028c80be52af9081bc6ea95a959e814434c28b9cd2346b5e8b120218ce0736f3a356d4c1d0de1f90b85bf"),
    "Bf16ToF32_v2": (0, 0, 0, "4455f054cb3eecec9e611f77120ac07d4cca117bdf9ad07af18df7f0ea6ac01b0dbd68b9e3fa5d97e44b4c190470c5e0c61d9c455990627f5362e3704c21672a"),
    "F32ToE4m3Sat_v2": (231, 376, 34, "0d31e021655e1813d3b6aac7076f2ca808a05f7eae0bffaefe30381f02f0c14a3016480bf66ec4a52a4065c1150cedfe1ce86efd7f93e917825673de0b3e1596"),
    "E4m3ToF32_v2": (15, 22, 4, "f400f0c9cc9c035ebc678b71131bc62ae2c3030414ca993045a8dc89d454bca14dd6a50992ed275edad0cf039c0cabe3a38cf4205974097e53e79d30ae3b913e"),
    "F32ToE2m1Sat_v2": (60, 86, 2, "262b97bd6873c99e7ae520c63d16238c32e397f6717d21dab5ebe665223172c7337563f8d904c21dd927e8870fac35dc7882acf3a799b612e911373b46d14334"),
    "BitAnd_v2": (1, 0, 0, "aa243ece6f21049abb7a8c4ecb98adbc92d9c237b824273463b4ae7e06e691f8d02cf3fb592d4465e11914e6a39f609f8d18a3d41cdb6752e8880ef302c30ae7"),
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
    return [np.concatenate([inputs[0], _e4m3_range(rng)])] if d is BSC.F32ToE4m3Sat else inputs


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_pinned_counts_and_digests(d) -> None:
    target = bind(d)
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("d", DEFS, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert T.counts["and"] == C.counts()["ands"] == PINS[f"{d.name}_v{d.version}"][0]


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
