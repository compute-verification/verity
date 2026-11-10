"""`verity.ml.definitions.registry.boolean_tanh`: RN32 tanh and Gemma-2's GeGLU by arithmetic are their word Definitions
(`trace.word_of`) and compute the registered tables they replace. The activation inside `GeluTanhMulBf16_v4` is
`gelu_tanh_bf16`'s word at every one of the 65,536 g (the words hash to the table's pin), and `GeluTanhMulBf16_v4` is
`GeluTanhMulBf16_v1` on every gate word at twelve up words, every pair of special words and random and edge-product pairs;
`TanhF32Rn_v3` is `TanhF32Rn_v1` at every rule's and the region's edges, every exponent and random words of the region and of all
f32. They carry exactly their builders' live ANDs, are pinned by gate count and program digest, hold no data, and are what a
Program on bits calls for their word Definitions.

An opt-in check runs every f32 word (`VERITY_VLLM_TANH_EXHAUSTIVE=1`: all 2^32 against `TanhF32Rn_v1`, the region's positive
results assembled into `tanh_rn` and hashed against its pin), over every core the process may use, against a vectorized
reference a test here ties to the word Definition. No check runs the 2^32 (g, u) pairs."""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import operations as O
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import boolean_tanh as BT
from verity.ml.definitions.registry import dense as D
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry import tables as T

from tests.program.test_boolean_dense import SPECIALS, _agree, _bits

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "TanhF32Rn_v3": (15719, 43891, 954, "3af26da1261fc9f6b7be841c942d5550cf43d27aecad393a6d6d59e72f49bdde805fa291e0d0fef32163ba5373c86af47ef12c61306d2a51eb45ffc5ed090b11"),
    "GeluTanhMulBf16_v4": (15616, 33430, 1334, "0f52a5368895b447405f800d6c999b7ee7ca57621a452f698d11a57913c1e0a1413881f47a7311ea77a5e65ca49d7f6c737538e96b0c905de55e7e993f0c143c"),
    "GeluTanhMul_v4{I=8}": (124928, 267440, 10672, "bb24ab0dc6586ab71124384d0223835b28d8e5c07989514ef4956f076078abb7b658b5e1b491fb627887f5ead394326ecb4bf08eed9e128f8972ede5f5b944ef"),
}
#: the pinned tables' sha256 over their little-endian u32 entries (`verity_flock.tail_pieces.TABLE_SHA256`)
TABLE_SHA256 = {"tanh_rn": "32fb35b4a1d0402a8f4b96a601c8ffc5e986266d20666a224eaf3bffeace308a",
                "gelu_tanh_bf16": "892c2fbca46fed05daf5b4fca3b49a392e93fdc6b826f8fd7bfa3c9bae0d8865"}
#: the reads the arithmetic replaces: the multiplexer tree over 2^IB entries of EW words, (2^IB - 1) 16 EW ANDs
#: (`verity.ml.circuits.boolean.table`)
MUX_ANDS = {"gelu_tanh_bf16": ((1 << 16) - 1) * 16, "tanh_rn": ((1 << 27) - 1) * 32}
BF16_ONE = 0x3F80

TARGETS = [bind(BT.TanhF32Rn), bind(BT.GeluTanhMulBf16), bind(BT.GeluTanhMul, I=8)]
SLOW_PINS = ("TanhF32Rn_v3", "GeluTanhMulBf16_v4")


@pytest.mark.parametrize("target", [pytest.param(t, marks=pytest.mark.slow) if t.id in SLOW_PINS else t for t in TARGETS], ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


def test_a_read_is_a_small_part_of_its_table_s_multiplexer() -> None:
    assert PINS["TanhF32Rn_v3"][0] * 10 ** 5 < MUX_ANDS["tanh_rn"]
    assert PINS["GeluTanhMulBf16_v4"][0] * 60 < MUX_ANDS["gelu_tanh_bf16"]


@pytest.mark.slow
@pytest.mark.parametrize("d", [BT.TanhF32Rn, BT.GeluTanhMulBf16], ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS[bind(d).id][0]


@pytest.mark.slow
def test_the_word_views_and_the_boolean_versions() -> None:
    """Each names its word Definition, is the newest Boolean version of it (`program.boolean.boolean_version`), and holds no
    data (`operations.program_data`)."""
    from verity.ml.definitions.boolean import boolean_version

    assert TR.word_of(bind(BT.TanhF32Rn)) is D.TanhF32Rn
    assert TR.word_of(bind(BT.GeluTanhMulBf16)) is D.GeluTanhMulBf16
    assert TR.word_of(bind(BT.GeluTanhMul, I=8)).id == bind(D.GeluTanhMul, I=8).id == "GeluTanhMul_v1{I=8}"
    assert boolean_version(D.TanhF32Rn).id == "TanhF32Rn_v3"
    assert boolean_version(D.GeluTanhMulBf16).id == "GeluTanhMulBf16_v4"
    assert boolean_version(bind(D.GeluTanhMul, I=8)).id == "GeluTanhMul_v4{I=8}"
    for target in TARGETS:
        assert O.program_data(Program(target).root) == {}, target.id


def _tanh_words(rng) -> np.ndarray:
    """Both signs of every rule's and region's edge, of each exponent's mantissa ends, and of random words of the region and
    of all f32."""
    lo, hi = T.TANH_RN_LO, T.TANH_RN_HI
    edges = [0, 1, 0x007FFFFF, 0x00800000, lo - 1, lo, lo + 1, 0x37FFFFFF, 0x38000000, 0x3F800000, 0x3FFFFFFF, 0x40000000,
             hi - 1, hi, 0x411FFFFF, 0x41200000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
    exponents = [e << 23 | m for e in range(256) for m in (0, 1, 0x400000, 0x7FFFFF)]
    words = edges + exponents + rng.integers(lo, hi, 4000).tolist() + rng.integers(0, 1 << 31, 2000).tolist()
    return np.array([(w & 0x7FFFFFFF) | s for w in words for s in (0, 1 << 31)], np.uint64)


def _tanh_ref(words: np.ndarray) -> np.ndarray:
    """`TanhF32Rn_v1` (`dense.tanh_f32_bits`) on f32 words, vectorized: NaN -> 0x7FC00000, |x| >= 10 (inf included) -> ±1,
    ±0 -> ±0, else RN32 of float64 `math.tanh`."""
    w = words.astype(np.uint32)
    a = w & 0x7FFFFFFF
    out = np.where(w >> 31, np.uint32(0xBF800000), np.uint32(0x3F800000))
    mid = (a != 0) & (a < 0x41200000)
    f = w[mid].view(np.float32).astype(np.float64)
    out[mid] = np.array(list(map(math.tanh, f.tolist())), np.float64).astype(np.float32).view(np.uint32)
    out[a == 0] = w[a == 0]
    out[a > 0x7F800000] = P.F32_NAN
    return out.astype(np.uint64)


def _act() -> np.ndarray:
    return np.array(T.gelu_tanh_bf16_words(), np.uint32)


@pytest.mark.slow
def test_tanh_is_the_word_definition_at_the_edges_every_exponent_and_random_words() -> None:
    words = _tanh_words(np.random.default_rng(21))
    got, want = _bits(bind(BT.TanhF32Rn), words), np.array([D.tanh_f32_bits(int(w)) for w in words], np.uint64)
    bad = np.flatnonzero(got != want)
    assert not bad.size, [(hex(int(words[i])), hex(int(got[i])), hex(int(want[i]))) for i in bad[:5]]
    region = (words < T.TANH_RN_HI) & (words >= T.TANH_RN_LO)
    assert all(int(t) == T.tanh_rn_entry(int(w) & 0x7FFFFFF) for w, t in zip(words[region], got[region]))


def test_the_vectorized_reference_is_the_word_definition() -> None:
    """What the exhaustive check compares against: `_tanh_ref` is `TanhF32Rn_v1`, subnormals and both signs included."""
    rng = np.random.default_rng(22)
    words = np.concatenate([_tanh_words(rng), rng.integers(0, 1 << 23, 512).astype(np.uint64) | np.uint64(1 << 31)])
    assert _tanh_ref(words).tolist() == [D.tanh_f32_bits(int(w)) for w in words]


def _edge_products(rng, n: int, act) -> tuple[np.ndarray, np.ndarray]:
    """(g, u) whose product f32(act(g)) * f32(u) lands in or near f32's subnormal range (rounded twice, to f32 then to bf16) or
    at its overflow: g random, u's exponent field chosen against act(g)'s so that the product's is in -26..3 or 250..257."""
    g = rng.integers(0, 1 << 16, n, dtype=np.int64)
    ea = (np.array([act(int(x)) for x in g], np.int64) >> 7) & 0xFF
    target = np.where(rng.integers(0, 2, n) == 1, rng.integers(-26, 4, n), rng.integers(250, 258, n))
    eu = np.clip(target - ea + 127, 0, 254)
    u = rng.integers(0, 2, n) << 15 | eu << 7 | rng.integers(0, 128, n)
    return g.astype(np.uint16), u.astype(np.uint16)


def _pairs(rng) -> tuple[np.ndarray, np.ndarray]:
    sg, su = np.meshgrid(SPECIALS, SPECIALS, indexing="ij")
    eg, eu = _edge_products(rng, 4096, D.gelu_tanh_bf16_bits)
    g = np.concatenate([sg.reshape(-1), eg, rng.integers(0, 1 << 16, 8192, dtype=np.uint16)])
    u = np.concatenate([su.reshape(-1), eu, rng.integers(0, 1 << 16, 8192, dtype=np.uint16)])
    return g.astype(np.uint64), u.astype(np.uint64)


@pytest.mark.slow
def test_the_activation_is_the_table_on_every_word() -> None:
    """`gelu_tanh_bf16(g)` at all 65,536 g: `gelu_tanh_bf16`'s words, which hash to its pin."""
    act = TR.definition("GeluTanhBf16ArithCheck", 1, lambda C, g: BT.gelu_tanh_bf16(C, g), (("g", BT.BF16),), BT.BF16,
                        register=False)
    got = _bits(act, np.arange(1 << 16, dtype=np.uint64))
    bad = np.flatnonzero(got != _act())
    assert not bad.size, f"{bad.size} activation words differ, first g {[hex(int(x)) for x in bad[:5]]}"
    assert hashlib.sha256(got.astype("<u4").tobytes()).hexdigest() == TABLE_SHA256["gelu_tanh_bf16"]


@pytest.mark.slow
@pytest.mark.parametrize("u", [BF16_ONE, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8080, 0x7F7F, 0xFF80, 0x7FC0, 0x4000, 0x3B80, 0x0080])
def test_every_gate_word(u: int) -> None:
    """All 65,536 gate words at an up word: at 1.0 the activation alone, then signs, zeros (-0 becomes +0), subnormals, the
    finite maximum, an infinity, a NaN and scales that move the product's exponent."""
    g = np.arange(1 << 16, dtype=np.uint64)
    _agree(bind(BT.GeluTanhMulBf16), g, np.full(g.shape, u, np.uint64))


@pytest.mark.slow
def test_gelu_is_the_word_definition_on_special_random_and_edge_pairs() -> None:
    _agree(bind(BT.GeluTanhMulBf16), *_pairs(np.random.default_rng(23)))


def test_gelu_tanh_mul_rows() -> None:
    rng = np.random.default_rng(24)
    rows = np.concatenate([rng.integers(0, 1 << 16, (400, 16), dtype=np.int64).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (100, 16))]])
    _agree(bind(BT.GeluTanhMul, I=8), rows)


# ---- the exhaustive check: every core this process may use, one chunk of words per task -------------------------------------

CHUNK = 1 << 22


def _procs() -> int:
    return len(os.sched_getaffinity(0))


def _tanh_chunk(c0: int):
    words = np.arange(c0, c0 + CHUNK, dtype=np.uint64)
    got, want = _bits(bind(BT.TanhF32Rn), words), _tanh_ref(words)
    bad = np.flatnonzero(got != want)
    region = (words >= T.TANH_RN_LO) & (words < T.TANH_RN_HI)
    return bad.size, [(int(words[i]), int(got[i]), int(want[i])) for i in bad[:5]], (words[region] & 0x7FFFFFF), got[region]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VERITY_VLLM_TANH_EXHAUSTIVE") != "1",
                    reason="VERITY_VLLM_TANH_EXHAUSTIVE=1: all 2^32 f32 words, about 3 core-hours (minutes on a pod's cores)")
def test_tanh_on_every_f32_word() -> None:
    """`TanhF32Rn_v3` is `TanhF32Rn_v1` on all 2^32 words, and its results at the region's 123,179,281 positive words, by their
    low 27 bits, are the pinned `tanh_rn`."""
    BS.compile(bind(BT.TanhF32Rn))
    table, bad, first = np.zeros(1 << 27, np.uint32), 0, []
    with mp.get_context("fork").Pool(_procs()) as pool:
        for n, few, idx, vals in pool.imap_unordered(_tanh_chunk, range(0, 1 << 32, CHUNK)):
            bad, first = bad + n, first + few
            table[idx.astype(np.int64)] = vals.astype(np.uint32)
    assert bad == 0, f"{bad} words differ, first (x, got, want) {[tuple(map(hex, f)) for f in first[:5]]}"
    assert hashlib.sha256(table.astype("<u4").tobytes()).hexdigest() == TABLE_SHA256["tanh_rn"]
