"""`catalog.definitions.boolean.attention`: the Boolean softmax operations agree with their word primitives (`trace.word_of`) on every
pair of special words and on random words (the inverse sum also on every mantissa), carry exactly their builders' live ANDs, and are pinned by gate count and program
digest."""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pytest

from verity.primitives.circuits import evaluation as E
from verity.primitives.circuits.evaluation import bits as BS
from verity.primitives.circuits.evaluation.reference import standalone_call
from verity.primitives.circuits import boolean as BL
from verity.primitives.circuits import codec
from verity.primitives.circuits.defs import REGISTRY, bind
from catalog.definitions.boolean import attention as BA
from verity.primitives.circuits.boolean import trace as TR
from verity.primitives.circuits.boolean.forms import Circuit

DEFS = [BA.F32AddFtz, BA.F32SubFtz, BA.F32MulFtz, BA.F32FmaFtz, BA.F32FmaSubFtz, BA.F32Max, BA.GuardNegInfZero, BA.Fa2InvSum]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32AddFtz_v3": (888, 1409, 115, "54dd4edd86f9c57184ad1bf30c4c5102d5c565fef3416e6c569dff27d53b157d"),
    "F32SubFtz_v3": (952, 1473, 135, "f569d60f3b9dc52d97e492765bcae656f553a5fad172db19a40bb9725d824604"),
    "F32MulFtz_v3": (2506, 4664, 110, "f95431e264e4bf7f6078ddcc4e20bba63f15bfd776d50e71bfa502269abb78a0"),
    "F32FmaFtz_v3": (4584, 7877, 267, "beeb387b6c27c6406e9da758098e2fddc5857bfcaa01410f5939e49d71d0a4ef"),
    "F32FmaSubFtz_v3": (4584, 7877, 268, "c40a13e35744d2326b4c46690191d4aefbd92a6ac20e835fa376457c22b4ac46"),
    "F32Max_v2": (237, 408, 70, "51cbd51389f3a835b06a9d18aede9466d22a6a4b8c3072559c6d15e8331b0958"),
    "GuardNegInfZero_v2": (63, 0, 24, "cecb89db1b7296e52d903a278f0a5a4321ff61bca24bda4c28c7c921c96db130"),
    "Fa2InvSum_v2": (947, 4866, 109, "8a134c216e0888fee6993aaac41a9ef07185b70712ab9581a5066dc95307408e"),
}

#: signed zeros, subnormals (the FTZ boundary), the normal range's ends, ones, infinities, NaNs of both signs and kinds, and
#: words near FA2's softmax values (log2(e) / sqrt(d) scales, exp2 arguments)
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x3FB8AA3B, 0x3E3504F3, 0x4B000000, 0x4B800000, 0x7E800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000,
               0x7FFFFFFF, 0x42FE0000, 0x33800000, 0x0C000000]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x3C000000, 0x44000000, (n_params, 20_000), dtype=np.uint64)
            | (rng.integers(0, 2, (n_params, 20_000), dtype=np.uint64) << np.uint64(31))).astype(np.uint32)
    return [np.concatenate([SPECIALS[g], r, x]) for g, r, x in zip(grid, rand, near)]


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
    inputs = _inputs(len(d.signature(None)[0]), np.random.default_rng(zlib.crc32(d.name.encode())))
    if d is BA.Fa2InvSum:
        inputs = [np.concatenate([inputs[0], np.arange(1 << 23, dtype=np.uint32) | np.uint32(0x3F800000)])]
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"

