"""`verity.ml.boolean.mufu`: the Boolean MUFU operations agree with their word Definitions (`trace.word_of`) on special and random
words and, through every table index, on whole binades; each table's interpolator reproduces the measured words; and every
Definition is pinned by gate count and program digest."""
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
from verity.ml import mufu as M
from verity.ml.boolean import mufu as BM
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit

DEFS = [BM.MufuEx2Ftz, BM.MufuRcpFtz, BM.MufuSqrtFtz, BM.RsqrtApprox, BM.DivFullRcp, BM.DivFullScaleA]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MufuEx2Ftz_v2": (1073, 4917, 76, "c2e2924f375157547e7c1bfe0c283037f0f324b841420177fafff23c01af7105"),
    "MufuRcpFtz_v2": (902, 5639, 70, "f49db0c381ce0b47b5548563e0c7a714b62284cf93b86b77261dcb783f171f00"),
    "MufuSqrtFtz_v2": (879, 5665, 89, "6431fb0de8d2779784fc068c1ac406596e147c63c15776479ee47eb889af5346"),
    "RsqrtApprox_v2": (1134, 6115, 134, "482071de38099933443314f94012941c01f7ea84ba1fd2dda78ab17f7f195197"),
    "DivFullRcp_v2": (1152, 6001, 113, "d1c351899328070278c4a4229cb22f8b4a7aff2393e64a9f92d33126b2740a5e"),
    "DivFullScaleA_v3": (531, 860, 94, "f8d0f52de422db61c053accdf13643773dc9b93ff29f099428f3299b70572011"),
}

#: each table's words that are not its interpolant: (index, word - interpolant)
OVERRIDES = {"ex2": (), "rcp": (), "rsq": ((0, 1),), "sqrt": ()}

#: signed zeros, subnormals (the FTZ boundary), the normal range's ends, ones, the range-scaling thresholds 2^-126 and 2^126,
#: infinities, NaNs of both signs and kinds, ex2's saturation points, and words just past a binade's ends
_MAGNITUDES = [0x00000000, 0x00000001, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x3F7FFFFF, 0x3F800000, 0x3F800001,
               0x3FFFFFFF, 0x40000000, 0x407FFFFF, 0x3E800000, 0x4B800000, 0x7E7FFFFF, 0x7E800000, 0x7E800001, 0x7F000000,
               0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x43000000, 0x42FE0000, 0x43160000, 0x33800000,
               0x34000000, 0x0C000000]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x80000000)], np.uint32)


def _inputs(n_params: int, rng) -> list[np.ndarray]:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=n_params)), np.int64).T
    rand = rng.integers(0, 1 << 32, (n_params, 60_000), dtype=np.uint64).astype(np.uint32)
    near = (rng.integers(0x30000000, 0x50000000, (n_params, 20_000), dtype=np.uint64)
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
    got = BS.evaluate_words(target, *inputs).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_the_interpolator_is_every_table_word(name) -> None:
    f = BM.fit(name)
    words = np.asarray(M.table(name), dtype=np.int64)
    q = f.interpolant(np.arange(len(words), dtype=np.int64))
    off = np.flatnonzero(q != words)
    assert tuple((int(i), int(words[i] - q[i])) for i in off) == BM.overrides(name) == OVERRIDES[name]


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_the_circuit_square_is_the_fit_square(name) -> None:
    f = BM.fit(name)
    C = Circuit()
    C.output("q", BM.truncated_square(C, C.input("u", f.nb), f.p))
    x = np.arange(1 << f.nb, dtype=np.int64)

    def sliced(bits):
        return int.from_bytes(np.packbits(bits.astype(np.uint8), bitorder="little").tobytes(), "little")

    def unsliced(v):
        return np.unpackbits(np.frombuffer(v.to_bytes(len(x) // 8, "little"), np.uint8), bitorder="little").astype(np.int64)

    outs, _ = C.evaluate({"u": [sliced((x >> k) & 1) for k in range(f.nb)]}, len(x))
    q = sum(unsliced(v) << j for j, v in enumerate(outs["q"]))
    assert (q == f.square(x)).all()


def _agree(target, *words) -> None:
    got = BS.evaluate_words(target, *words).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(TR.word_of(target), *words).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} differ, first {[[hex(int(w[i])) for w in words] for i in bad[:5]]}"


def _span(*ranges) -> np.ndarray:
    return np.concatenate([np.arange(lo, hi, dtype=np.uint64) for lo, hi in ranges]).astype(np.uint32)


#: (id, Definition, word ranges): a binade (two for rsq / sqrt, by exponent parity) reads every index of its table; ex2's
#: negative binade reads them through ~K, and its exponents 103 and 133 are the ends of its 5-bit shift; DivFullRcp's and
#: RsqrtApprox's subnormals, and DivFullRcp's (2^126, inf], are their range scalings
_RANGES = [("MufuEx2Ftz", BM.MufuEx2Ftz, [(0x3F800000, 0x40000000)]),
           ("MufuEx2Ftz-negative-and-shift-ends", BM.MufuEx2Ftz, [(0xBF800000, 0xC0000000), (0x33800000, 0x34000000),
                                                                  (0xC2800000, 0xC3000000)]),
           ("MufuRcpFtz", BM.MufuRcpFtz, [(0x3F800000, 0x40000000)]),
           ("MufuSqrtFtz", BM.MufuSqrtFtz, [(0x3F800000, 0x40800000)]),
           ("RsqrtApprox", BM.RsqrtApprox, [(0x3F800000, 0x40800000)]),
           ("RsqrtApprox-subnormal", BM.RsqrtApprox, [(0x00000000, 0x00800000)]),
           ("DivFullRcp-scaled", BM.DivFullRcp, [(0x00000000, 0x00800000), (0xFE800000, 0xFF800001)])]


@pytest.mark.slow
@pytest.mark.parametrize("d, ranges", [r[1:] for r in _RANGES], ids=[r[0] for r in _RANGES])
def test_every_word_of_a_range(d, ranges) -> None:
    """Every word of the ranges (2^23 to 2^24 + 2^23 words, 10-35 s each): the random words above reach about 1 % of a table's
    indices, and only a few hundred words of each scaled range."""
    _agree(bind(d), _span(*ranges))


@pytest.mark.slow
@pytest.mark.parametrize("b", [0x7E800001, 0x80000001], ids=["a*0.25", "a*2^24"])
def test_div_full_scale_a_where_it_rounds_or_normalizes(b) -> None:
    """Every a of exponent 0 to 2 (3 * 2^23 words, about 30 s) against a b of each scaling: where a * 0.25 is subnormal and
    rounds, and where a * 2^24 normalizes a subnormal; random words reach a few thousand of them."""
    a = _span((0x00000000, 0x01800000))
    _agree(bind(BM.DivFullScaleA), a, np.full(len(a), b, np.uint32))
