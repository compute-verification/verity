"""`verity.ml.boolean.scalar`: the Boolean f32 arithmetic, max / min and casts agree with their word primitives (`trace.word_of`) on
every tuple of special words and on random words (the bf16 cast on every word, the E4M3 cast on every exponent around its range),
carry exactly their builders' live ANDs, and are pinned by gate count and program digest."""
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
from verity.ml.boolean import scalar as SC
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit

DEFS = [SC.F32Add, SC.F32Mul, SC.F32Fma, SC.F32Div, SC.F32Fabs, SC.F32Fmaxf, SC.F32Fminf, SC.Bf16ToF32, SC.F32ToE4m3Sat]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32Add_v3": (812, 1395, 115, "9fdc2e8807fce92fa895519eb1ee942b7b2b46f4971122495fe2a6282c9aaec0"),
    "F32Mul_v3": (2430, 4650, 110, "61f93e5519c55cf5ded0dcb0d867979431bcd5508839842ec8832bcd86755311"),
    "F32Fma_v3": (4485, 7863, 267, "fd402c079584408027c10870dbba98867e7ba3088dd798c48b0b5289c7474667"),
    "F32Div_v3": (2517, 5592, 252, "13f6fbbdffbcb6506ae3fa9555bffb62376ec4827a74ef6303f528abe6365d93"),
    "F32Fabs_v2": (0, 0, 0, "e6928242db378dd9949140e955abe855dcd29469cb514a0d203e98b7fca39460"),
    "F32Fmaxf_v2": (314, 573, 102, "98062192009fa6ad5c43fc1a55a0d6b4e170a5597e1a8bf28a7cf4ffc647631f"),
    "F32Fminf_v2": (314, 574, 102, "eba9c1ccaeefcadc6d6567e7d8903e70b4831d6dcdc66d4302cb3378a148bde8"),
    "Bf16ToF32_v2": (0, 0, 0, "39168eff727f5ef51d263a418e3473c7037d1f03e3c93afc8d9a219958f21eaf"),
    "F32ToE4m3Sat_v2": (231, 376, 34, "e397be3ee5361aec4f561816c731220bfa101b1d777d0797731abcba3afbc361"),
}

#: signed zeros, subnormals, the normal range's ends, ones, infinities, NaNs of both signs and kinds, and the FP8 linear's
#: constants and boundaries (448, 1 / (448 * 512), the E4M3 grid's ends and rounding midpoints)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x4B000000, 0x7E800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x43E00000, 0x43E80000,
               0x43F00000, 0x3A124925, 0x3C800000, 0x3B000000, 0x3A800000, 0x3A000000]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x3C000000, 0x44000000, (n_params, 20_000), dtype=np.uint64)
            | (rng.integers(0, 2, (n_params, 20_000), dtype=np.uint64) << np.uint64(31))).astype(np.uint32)
    return [np.concatenate([SPECIALS[g], r, x]) for g, r, x in zip(grid, rand, near)]


def _e4m3_range(rng) -> np.ndarray:
    """Every exponent from 2^-20 to past 448 (fields 107..140), each with its mantissa's low 8 bits exhausted under 64 random high
    parts (the rounding position is at or above bit 20), both signs."""
    e = np.arange(107, 141, dtype=np.uint32)
    hi = rng.integers(0, 1 << 15, 64, dtype=np.uint64).astype(np.uint32) << np.uint32(8)
    lo = np.arange(256, dtype=np.uint32)
    m = (hi[:, None] | lo[None, :]).reshape(-1)
    w = (e[:, None] << np.uint32(23) | m[None, :]).reshape(-1)
    return np.concatenate([w, w | np.uint32(0x80000000)])


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
    rng = np.random.default_rng(zlib.crc32(d.name.encode()))
    if d is SC.Bf16ToF32:
        inputs = [np.arange(1 << 16, dtype=np.uint16)]
    else:
        inputs = _inputs(len(d.signature(None)[0]), rng)
        if d is SC.F32ToE4m3Sat:
            inputs = [np.concatenate([inputs[0], _e4m3_range(rng)])]
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"
