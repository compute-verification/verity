"""`verity.ml.circuits.ops.boolean.attention`: the Boolean softmax operations agree with their word primitives (`trace.word_of`) on every
pair of special words and on random words (the inverse sum also on every mantissa), carry exactly their builders' live ANDs, and are pinned by gate count and program
digest."""
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
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit

DEFS = [BA.F32AddFtz, BA.F32SubFtz, BA.F32MulFtz, BA.F32FmaFtz, BA.F32FmaSubFtz, BA.F32Max, BA.GuardNegInfZero, BA.Fa2InvSum]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "F32AddFtz_v3": (888, 1409, 115, "31859a9b9fa9cf8a1443c2394f5f0c0440190d389e57d917588397c66356544d2076dacbd0fd9121930e3a4db01b0cb9d499991caa21e4f9c686cd3de0168ded"),
    "F32SubFtz_v3": (952, 1473, 135, "20911d6fb9751c3060aa86aa0a93afbddb1179d770a3c61d7a9fd7e603f29d7718b39e0aa8311e91d66ae2d3175189b85516713e446a6327b465aa406a42e9b2"),
    "F32MulFtz_v3": (2506, 4664, 110, "77a4da7a23a12dc3a93a3399a5f21b6daa7cb9b218fab0aa78fd8835ce1fc02333c2ada8e803794ff15c2e4b0b1929d6a359a05c28acd0a101a2639a4ed33a28"),
    "F32FmaFtz_v3": (4584, 7877, 267, "25ddfff45c50aac2650696dcf79686a29a0d579c2ae8a13786d51aa61776d0d97a05c65d5485b99378fc15f5df60d7e3a58ec8329cea805db8657672b86243a0"),
    "F32FmaSubFtz_v3": (4584, 7877, 268, "8eccc11b98e3c726bacc13bef779b8d257bdddf8f288b593f968c356615d956f668bb52deecbc6b93ecbd577376c54f65fa8432803bbb80d9d5690e04f13620d"),
    "F32Max_v2": (237, 408, 70, "78a50c6f01cbda859383570a1e9ab5ce865d4ab2d1e2522dc43008a13ed3b939de92857526ed0eeb034a929159f7425bee00dc25bf9e495af1cc23363cb67a5d"),
    "GuardNegInfZero_v2": (63, 0, 24, "95b20e44d40be20324a12a3b47dc1c5163d6c2e730f6a5dcde2b95702eeeb3f730765bc222bd6794dd2780f6dc62d1306eb4528bf0bd8588b5e6e8c5ddae2a7d"),
    "Fa2InvSum_v2": (947, 4866, 109, "9e1360306e0f58643c72103e964c571e867989ad7da7cd4a2de7ea2cca9349ebd53049822c4ead8f22de302fb4fe8e9b6949e8f8978a81f7531360e4e2444709"),
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

