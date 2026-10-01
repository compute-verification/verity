"""`verity_vllm.program.registry.boolean_dense`: the Boolean versions of `registry.dense`'s Definitions agree with their word
Definitions (`trace.word_of`) on every activation word, on every pair of special words and on random and edge-product pairs,
carry exactly their builder's live ANDs, are pinned by gate count and program digest, and partition under
`Q_word_v1{X=16, W=32}` into one unit per element."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir import partition_object as PO
from verity.ir.defs import REGISTRY, bind
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit
from verity_vllm.program.registry import boolean_dense as BD
from verity_vllm.program.registry import dense as D
from verity_vllm.program.registry import prims as P

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "GeluTanhMulBf16_v2": (1347, 5960, 148, "f0e19a849febe1c41aa13ccb29726c60fb02eb1ef6a98b18a5285cbd170e3b30"),
    "GeluTanhMul_v2{I=8}": (10776, 47680, 1184, "e40e773a4348acaf7293a66e35cd3203918a64bb88c4f048e38557406c7b333b"),
}

_EXP = [0, 1, 2, 116, 117, 118, 119, 120, 126, 127, 128, 129, 130, 131, 133, 134, 135, 254, 255]
#: per sign: zero and the subnormal ends, each band edge's exponents at the mantissa ends, ones, the finite maximum, the
#: infinities and NaNs (quiet, signalling, all ones)
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)


def _word(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got, want = _bits(target, *inputs), _word(target, *inputs)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


def _edge_products(rng, n: int, act) -> tuple[np.ndarray, np.ndarray]:
    """(g, u) whose product f32(act(g)) * f32(u) lands in or near f32's subnormal range (rounded twice, to f32 then to bf16) or
    at its overflow: g random, u's exponent field chosen against act(g)'s so that the product's is in -26..3 or 250..257."""
    g = rng.integers(0, 1 << 16, n, dtype=np.int64)
    ea = (np.array([act(int(x)) for x in g], np.int64) >> 7) & 0xFF
    target = np.where(rng.integers(0, 2, n) == 1, rng.integers(-26, 4, n), rng.integers(250, 258, n))
    eu = np.clip(target - ea + 127, 0, 254)
    u = rng.integers(0, 2, n) << 15 | eu << 7 | rng.integers(0, 128, n)
    return g.astype(np.uint16), u.astype(np.uint16)


TARGETS = [bind(BD.GeluTanhMulBf16), bind(BD.GeluTanhMul, I=8)]


@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


def test_the_ands_are_the_builders_live_ands() -> None:
    d = BD.GeluTanhMulBf16
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS["GeluTanhMulBf16_v2"][0]


def test_the_word_views() -> None:
    assert TR.word_of(bind(BD.GeluTanhMulBf16)) is D.GeluTanhMulBf16
    assert TR.word_of(bind(BD.GeluTanhMul, I=8)).id == bind(D.GeluTanhMul, I=8).id == "GeluTanhMul_v1{I=8}"


def _half(g: int) -> int:
    f = np.array([np.array([g << 16], np.uint32).view(np.float32)[0] / np.float32(2)], np.float32)
    return P.f32_to_bf16_rn_bits(int(f.view(np.uint32)[0]), P.BF16_NAN_F2FP)


def test_the_gelu_band_is_where_the_closed_forms_fail() -> None:
    """Every activation word off the band is one of the module docstring's closed forms, and on the band its sign is g's; the
    band's lowest exponent and its highest, per sign, are each needed (a closed form fails on some word there)."""
    act = [D.gelu_tanh_bf16_bits(g) for g in range(1 << 16)]
    for g, a in enumerate(act):
        sign, e, m = g >> 15, (g >> 7) & 0xFF, g & 0x7F
        if e == 255 and (m or sign):
            assert a == P.BF16_NAN_F2FP
        elif BD.GELU_BAND_LO <= e <= BD.GELU_BAND_HI[sign]:
            assert a >> 15 == sign
        elif e < BD.GELU_BAND_LO:
            assert a == _half(g)
        else:
            assert a == (0x8000 if sign else g)
    for sign in (0, 1):
        lo = range(sign << 15 | BD.GELU_BAND_LO << 7, sign << 15 | (BD.GELU_BAND_LO + 1) << 7)
        assert any(act[g] != _half(g) for g in lo)
        top = range(sign << 15 | BD.GELU_BAND_HI[sign] << 7, sign << 15 | (BD.GELU_BAND_HI[sign] + 1) << 7)
        assert any(act[g] != (0x8000 if sign else g) for g in top)


def test_the_activation_on_every_word() -> None:
    act = TR.definition("GeluTanhBf16Check", 1, lambda C, g: BD.gelu_tanh_bf16(C, g), (("g", BD.BF16),), BD.BF16, register=False)
    g = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    want = np.array([D.gelu_tanh_bf16_bits(int(x)) for x in g], np.uint64)
    assert (_bits(act, g) == want).all()


@pytest.mark.parametrize("u", [0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8080, 0x7F7F, 0xFF80, 0x7FC0, 0x4000, 0x3B80, 0x0080])
def test_every_gate_word(u: int) -> None:
    """All 65,536 gate words at an up word: at 1.0 the activation alone, then signs, zeros (-0 becomes +0), subnormals, the
    finite maximum, an infinity, a NaN and scales that move the product's exponent."""
    g = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    _agree(bind(BD.GeluTanhMulBf16), g, np.full(g.shape, u, np.uint16))


@pytest.mark.slow
def test_special_pairs_random_and_edge_products() -> None:
    """Every pair of the special words, 100,000 random pairs and 20,000 products at f32's subnormal and overflow ends."""
    rng = np.random.default_rng(0x6E1)
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=2)), np.int64).T
    rand = rng.integers(0, 1 << 16, (2, 100_000), dtype=np.int64).astype(np.uint16)
    eg, eu = _edge_products(rng, 20_000, D.gelu_tanh_bf16_bits)
    g = np.concatenate([SPECIALS[grid[0]], rand[0], eg])
    u = np.concatenate([SPECIALS[grid[1]], rand[1], eu])
    _agree(bind(BD.GeluTanhMulBf16), g, u)


def test_gelu_tanh_mul_rows() -> None:
    rng = np.random.default_rng(0x6E2)
    rows = np.concatenate([rng.integers(0, 1 << 16, (4000, 16), dtype=np.int64).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (1000, 16))]])
    _agree(bind(BD.GeluTanhMul, I=8), rows)


def test_one_unit_per_element_under_q_word() -> None:
    prog, _ = standalone_call(bind(BD.GeluTanhMul, I=2))
    part = PO.evaluate(prog, PO.query(16, 32))
    assert part.population == 2
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
