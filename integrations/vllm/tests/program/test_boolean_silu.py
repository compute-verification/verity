"""`verity_vllm.program.registry.boolean_silu`: the Boolean SiLU Definitions agree with their word Definitions (`trace.word_of`)
on every gate word, on every pair of special words and on random and edge-product pairs, carry exactly their builder's live
ANDs, are pinned by gate count and program digest, and partition under `Q_word_v1{X=16, W=32}` into one unit per element."""
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
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_silu as BSL
from verity_vllm.program.registry import prims as P

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "SiluMulBf16_v3": (1372, 7034, 162, "d228d6ecc77f994dd1e2fd23208223ee85fdc11d58b8effa8c4fd759a0643613"),
    "SiluMul_v3{I=8}": (10976, 56272, 1296, "f34ab37e75145eb2a12c911e8e7344fcdd5a6650347e83e016d4dd16525e1d32"),
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


def _edge_products(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(g, u) whose product f32(s) * f32(u) lands in or near f32's subnormal range (rounded twice, to f32 then to bf16) or at
    its overflow: g random, u's exponent field chosen against s's so that the product's is in -26..3 or 250..257."""
    g = rng.integers(0, 1 << 16, n, dtype=np.int64)
    s = np.array([P.SiluMulBf16.evaluate(int(x), 0x3F80) for x in g], np.int64)
    es = (s >> 7) & 0xFF
    target = np.where(rng.integers(0, 2, n) == 1, rng.integers(-26, 4, n), rng.integers(250, 258, n))
    eu = np.clip(target - es + 127, 0, 254)
    u = rng.integers(0, 2, n) << 15 | eu << 7 | rng.integers(0, 128, n)
    return g.astype(np.uint16), u.astype(np.uint16)


def test_pinned_counts_and_digests() -> None:
    for target in (bind(BSL.SiluMulBf16), bind(BSL.SiluMul, I=8)):
        ands, xors, nots, digest = PINS[target.id]
        assert BL.is_boolean(target)
        assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
        prog, _ = standalone_call(target)
        desc = codec.encode_program(prog)
        assert codec.program_digest(desc) == digest
        D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
        assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


def test_the_ands_are_the_builders_live_ands() -> None:
    d = BSL.SiluMulBf16
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS["SiluMulBf16_v3"][0]


def test_the_word_views() -> None:
    assert TR.word_of(bind(BSL.SiluMulBf16)) is P.SiluMulBf16
    assert TR.word_of(bind(BSL.SiluMul, I=8)).id == bind(b1.SiluMul, I=8).id == "SiluMul_v1{I=8}"


def _half(g: int) -> int:
    """``bf16_rn(g / 2)``, and +0 for -0."""
    f = np.array([np.array([g << 16], np.uint32).view(np.float32)[0] / np.float32(2)], np.float32)
    return 0 if g == 0x8000 else P.f32_to_bf16_rn_bits(int(f.view(np.uint32)[0]), 0x7FC0)


def test_the_band_is_where_the_closed_forms_fail() -> None:
    """Every word off the band is one of the module docstring's closed forms; on the band the sign is g's and the magnitude
    the table's; and the band's first odd exponent and its last are each needed (a closed form fails on some word there)."""
    silu = [P.SiluMulBf16.evaluate(g, 0x3F80) for g in range(1 << 16)]
    table = BSL.band_table()
    for g, s in enumerate(silu):
        sign, e, m = g >> 15, (g >> 7) & 0xFF, g & 0x7F
        if BSL.BAND_LO <= e <= BSL.BAND_HI[sign]:
            assert s >> 15 == sign and s & 0x7FFF == table[m | (e & 0xF) << 7 | sign << 11]
        elif e == 255 and m:
            assert s == BSL.NAN_TORCH
        elif e < BSL.BAND_LO:
            assert s == _half(g)
        else:
            assert s == (0x8000 if sign else g)
    for sign in (0, 1):
        assert any(silu[g] != _half(g) for g in range(sign << 15 | (BSL.BAND_LO + 1) << 7, sign << 15 | (BSL.BAND_LO + 2) << 7))
        top = range(sign << 15 | BSL.BAND_HI[sign] << 7, sign << 15 | (BSL.BAND_HI[sign] + 1) << 7)
        assert any(silu[g] != (0x8000 if sign else g) for g in top)


@pytest.mark.parametrize("u", [0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8080, 0x7F7F, 0xFF80, 0x7FC0, 0x4000, 0x3B80, 0x0080])
def test_every_gate_word(u: int) -> None:
    """All 65,536 gate words at an up word: at 1.0 the silu half alone, then signs, zeros, subnormals, the finite maximum, an
    infinity, a NaN and scales that move the product's exponent."""
    g = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    _agree(bind(BSL.SiluMulBf16), g, np.full(g.shape, u, np.uint16))


def test_special_pairs_random_and_edge_products() -> None:
    """Every pair of the special words, 100,000 random pairs and 20,000 products at f32's subnormal and overflow ends."""
    rng = np.random.default_rng(0x5111)
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=2)), np.int64).T
    rand = rng.integers(0, 1 << 16, (2, 100_000), dtype=np.int64).astype(np.uint16)
    eg, eu = _edge_products(rng, 20_000)
    g = np.concatenate([SPECIALS[grid[0]], rand[0], eg])
    u = np.concatenate([SPECIALS[grid[1]], rand[1], eu])
    _agree(bind(BSL.SiluMulBf16), g, u)


def test_silu_mul_rows() -> None:
    rng = np.random.default_rng(0x5112)
    rows = np.concatenate([rng.integers(0, 1 << 16, (4000, 16), dtype=np.int64).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (1000, 16))]])
    _agree(bind(BSL.SiluMul, I=8), rows)


def test_one_unit_per_element_under_q_word() -> None:
    prog, _ = standalone_call(bind(BSL.SiluMul, I=2))
    part = PO.evaluate(prog, PO.query(16, 32))
    assert part.population == 2
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
