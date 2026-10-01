"""`verity.ml.boolean.elementwise`: the Boolean element-wise operations agree with their word primitives (`trace.word_of`) on
every pair (triple) of special words, on random words and on words chosen to land in the subnormal range, carry exactly their
builders' live ANDs, and are pinned by gate count and program digest. Bf16AddF2fp_v2 is checked on every pair of words when
``VERITY_ML_BOOLEAN_EXHAUSTIVE`` is set."""
from __future__ import annotations

import itertools
import json
import os
import zlib

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import REGISTRY, bind
from verity.ml import fp32 as W
from verity.ml import kernels as K
from verity.ml.boolean import elementwise as EW
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit

DEFS = [EW.F32Add, EW.F32Mul, EW.F32Fma, EW.F32Div, EW.Bf16AddF2fp, EW.I32Add]
F32_DEFS = [EW.F32Add, EW.F32Mul, EW.F32Fma, EW.F32Div]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32Add_v3": (812, 1395, 115, "9fdc2e8807fce92fa895519eb1ee942b7b2b46f4971122495fe2a6282c9aaec0"),
    "F32Mul_v3": (2430, 4650, 110, "61f93e5519c55cf5ded0dcb0d867979431bcd5508839842ec8832bcd86755311"),
    "F32Fma_v3": (4485, 7863, 267, "fd402c079584408027c10870dbba98867e7ba3088dd798c48b0b5289c7474667"),
    "F32Div_v3": (2517, 5592, 252, "13f6fbbdffbcb6506ae3fa9555bffb62376ec4827a74ef6303f528abe6365d93"),
    "Bf16AddF2fp_v2": (717, 1061, 122, "f8ebc9ea23ac0786e86867d665f03b9a7ac05e5e295e91293fe0dc2257f907d1"),
    "I32Add_v2": (31, 153, 0, "d5522187d01b2e76d64689c006ef304fc0b87d8b83337a7534ef2d3f1306cdca"),
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
def test_its_word_view_is_its_word_primitive(d) -> None:
    target = bind(d)
    word = TR.word_of(target)
    assert (word.name, word.version) == (d.name, 1)
    (params, ret), (word_params, word_ret) = E.batch.signature_of(target), E.batch.signature_of(word)
    assert [BL.word_type(t) for t in params] == list(word_params) and BL.word_type(ret) == word_ret


@pytest.mark.parametrize("d", F32_DEFS, ids=lambda d: d.name)
def test_the_f32_operations_agree_with_their_word_primitives(d) -> None:
    _agree(bind(d), _f32_inputs(len(d.signature(None)[0]), np.random.default_rng(zlib.crc32(d.name.encode()))))


_BF16_MAGNITUDES = [0x0000, 0x0001, 0x0002, 0x003F, 0x0040, 0x007F, 0x0080, 0x0081, 0x00FF, 0x0100, 0x3F00, 0x3F7F, 0x3F80,
                    0x3F81, 0x3FC0, 0x4000, 0x4B00, 0x7F00, 0x7F7E, 0x7F7F, 0x7F80, 0x7F81, 0x7FBF, 0x7FC0, 0x7FFF]
BF16_SPECIALS = np.array([m | s for m in _BF16_MAGNITUDES for s in (0, 0x8000)], np.uint16)


def test_the_bf16_add_agrees_with_bf16_add_f2fp_v1() -> None:
    """Every pair of special words, every word against each of 8 specials (both orders), random pairs, and pairs that cancel to
    a few ulps (subnormal and zero sums)."""
    rng = np.random.default_rng(21)
    every = np.arange(1 << 16, dtype=np.uint16)
    some = BF16_SPECIALS[[0, 1, 5, 7, 16, 19, 20, 23]]
    grid = np.array(list(itertools.product(range(len(BF16_SPECIALS)), repeat=2)), np.int64).T
    a = np.concatenate([BF16_SPECIALS[grid[0]], np.repeat(every, len(some)), np.tile(some, len(every)),
                        rng.integers(0, 1 << 16, 60_000).astype(np.uint16)])
    b = np.concatenate([BF16_SPECIALS[grid[1]], np.tile(some, len(every)), np.repeat(every, len(some)),
                        rng.integers(0, 1 << 16, 60_000).astype(np.uint16)])
    c = rng.integers(0, 1 << 15, 20_000).astype(np.int64)
    d = ((c + rng.integers(-3, 4, c.size)).clip(0, 0x7FFF) | (1 << 15)).astype(np.uint16)
    _agree(bind(EW.Bf16AddF2fp), [np.concatenate([a, c.astype(np.uint16)]), np.concatenate([b, d])])


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


@pytest.mark.skipif(not os.environ.get("VERITY_ML_BOOLEAN_EXHAUSTIVE"),
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
