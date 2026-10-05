"""`verity.ml.boolean.scalar`: the bf16 comparator, the selects and the bit AND agree with their word primitives on every input
(the comparator on all 2^32 pairs of words, each select over its whole truth table); the f32 arithmetic, max / min and casts
agree with theirs (`trace.word_of`) on every tuple of special words and on random words (the bf16 and E4M3 widenings on every
word, the E4M3 cast on every exponent around its range, the E2M1 cast on every boundary of its grid and on all 2^32 words);
each carries exactly its builder's live ANDs and is pinned by gate count and program digest."""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import REGISTRY, bind
from verity.ir.types import Array
from verity.ml import prims as P
from verity.ml import scalar as S
from verity.ml.boolean import scalar as BSC
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit
from verity.ml.tc.cast import E2M1_UP

DEFS = [BSC.Bf16GtStrict, BSC.SelectBf16, BSC.SelectI32, BSC.F32Add, BSC.F32Mul, BSC.F32Fma, BSC.F32Div, BSC.F32Fabs, BSC.F32Fmaxf,
        BSC.F32Fminf, BSC.Bf16ToF32, BSC.F32ToE4m3Sat, BSC.E4m3ToF32, BSC.F32ToE2m1Sat, BSC.BitAnd]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "Bf16GtStrict_v2": (56, 133, 19, "66dfd1ed5bdc1d5f62e045bd5dceff8fe7a3ade2c0dae39ee9983db1c6416ab2"),
    "SelectBf16_v2": (16, 32, 0, "5690d8726cc1a582f29c4a6083a8f74ec97a5e13efa5b908390f465f3cd41d37"),
    "SelectI32_v2": (32, 64, 0, "693785819c9067e60513aa3f943f86f1b1799853944003afad7ace49d5155530"),
    "F32Add_v3": (812, 1395, 115, "9fdc2e8807fce92fa895519eb1ee942b7b2b46f4971122495fe2a6282c9aaec0"),
    "F32Mul_v3": (2430, 4650, 110, "61f93e5519c55cf5ded0dcb0d867979431bcd5508839842ec8832bcd86755311"),
    "F32Fma_v3": (4485, 7863, 267, "fd402c079584408027c10870dbba98867e7ba3088dd798c48b0b5289c7474667"),
    "F32Div_v3": (2517, 5592, 252, "13f6fbbdffbcb6506ae3fa9555bffb62376ec4827a74ef6303f528abe6365d93"),
    "F32Fabs_v2": (0, 0, 0, "e6928242db378dd9949140e955abe855dcd29469cb514a0d203e98b7fca39460"),
    "F32Fmaxf_v2": (314, 573, 102, "98062192009fa6ad5c43fc1a55a0d6b4e170a5597e1a8bf28a7cf4ffc647631f"),
    "F32Fminf_v2": (314, 574, 102, "eba9c1ccaeefcadc6d6567e7d8903e70b4831d6dcdc66d4302cb3378a148bde8"),
    "Bf16ToF32_v2": (0, 0, 0, "39168eff727f5ef51d263a418e3473c7037d1f03e3c93afc8d9a219958f21eaf"),
    "F32ToE4m3Sat_v2": (231, 376, 34, "e397be3ee5361aec4f561816c731220bfa101b1d777d0797731abcba3afbc361"),
    "E4m3ToF32_v2": (15, 22, 4, "3bf9194e080d15053593179eb14117aa2a58ded2911bb01df2d9600e335abc36"),
    "F32ToE2m1Sat_v2": (60, 86, 2, "6ac4002981fe5cfe28d638ab0642dadfd0bdcab373fd3e86dd8aa3d5514f220d"),
    "BitAnd_v2": (1, 0, 0, "aee8be5dcd3a028f822fd2b61cb651e5daeb97b0493c24ebf046083b2ca18bf4"),
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
    if d in (BSC.SelectBf16, BSC.SelectI32):
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


@pytest.mark.parametrize("d", [BSC.SelectBf16, BSC.SelectI32], ids=lambda d: d.name)
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
