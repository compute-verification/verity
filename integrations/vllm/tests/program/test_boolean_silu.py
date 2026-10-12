"""`verity_vllm.program.registry.boolean_silu`: the Boolean SiLU Definitions agree with their word Definitions (`trace.word_of`)
on every gate word, on every pair of special words and on random and edge-product pairs, carry exactly their builder's live
ANDs, are pinned by gate count and program digest, and partition under `Q_word_v1{X=16, W=32}` into one unit per element, as
their word rows do. The v4 pair (word view the served `SiluMulBf16_v2`) is also exact on the overflow window, the gates
whose ``expf(-g)`` overflows at every up word."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity_vllm.program.registry import act
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_silu as BSL
from verity_vllm.program.registry import prims as P

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "SiluMulBf16_v3": (1372, 7034, 162, "a7ddde8f9a7785aba7f60449a99e5de9c00248d00f7d7711ccca96e1d5186aa4862dbdf971c0bf876e99a613a0b48ea290c9b2e1b3013b7c24a2f8c5427360f8"),
    "SiluMul_v3{I=8}": (10976, 56272, 1296, "43ea066e735705e41661357e1b89c8c1cbcd94398c81c8cd9fee0bdbb21c2af7b676581f27e9a29bfbc28b4a291a47712d55c7316830f1f689b387985c9b0c81"),
    "SiluMulBf16_v4": (1380, 7023, 166, "5b166b381858235d19f218e96e43d0e2102bbe73f6cdf9315fa850f80a4afed4e032ac57b5e2efd271c071893212d93dd6b542ac86ab33fa119558b976f8aa7b"),
    "SiluMul_v4{I=8}": (11040, 56184, 1328, "28e2e644cc7ab2a39d60e14429558cc92c3019807248898144be4f77010d17f646c05383e5e5049b756449aa12af9abbfe4d17f29bf566f87f1d35e51c5d3505"),
}
#: version -> (element, row)
VERSIONS = {"v3": (BSL.SiluMulBf16, BSL.SiluMul), "v4": (BSL.SiluMulBf16Expf, BSL.SiluMulExpf)}

_EXP = [0, 1, 2, 116, 117, 118, 119, 120, 126, 127, 128, 129, 130, 131, 133, 134, 135, 254, 255]
#: per sign: zero and the subnormal ends, each band edge's exponents at the mantissa ends, ones, the finite maximum, the
#: infinities and NaNs (quiet, signalling, all ones)
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)
#: the gates -88.5 .. -103.5: bf16 -89 (0xC2B2) is the first whose expf(-g) overflows f32, and from about -103 on `_v1`'s
#: g * e^g rounds to -0 as well
OVERFLOW_WINDOW = np.arange(0xC2B1, 0xC2CF + 1, dtype=np.uint16)


def _word(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got, want = _bits(target, *inputs), _word(target, *inputs)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


def _f(w: int) -> float:
    return float(np.array([w << 16], np.uint32).view(np.float32)[0])


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
    for element, row in VERSIONS.values():
        for target in (bind(element), bind(row, I=8)):
            ands, xors, nots, digest = PINS[target.id]
            assert BL.is_boolean(target)
            assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
            prog, _ = standalone_call(target)
            desc = codec.encode_program(prog)
            assert codec.program_digest(desc) == digest
            D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
            assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_the_ands_are_the_builders_live_ands(version: str) -> None:
    d = VERSIONS[version][0]
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS[bind(d).id][0]


def test_the_word_views() -> None:
    assert TR.word_of(bind(BSL.SiluMulBf16)) is P.SiluMulBf16
    assert TR.word_of(bind(BSL.SiluMul, I=8)).id == bind(b1.SiluMul, I=8).id == "SiluMul_v1{I=8}"
    assert TR.word_of(bind(BSL.SiluMulBf16Expf)) is act.SiluMulBf16 and act.SiluMulBf16.id == "SiluMulBf16_v2"
    assert TR.word_of(bind(BSL.SiluMulExpf, I=8)).id == bind(act.SiluMul, I=8).id == "SiluMul_v2{I=8}"


def _half(g: int) -> int:
    """``bf16_rn(g / 2)``, and +0 for -0."""
    f = np.array([np.array([g << 16], np.uint32).view(np.float32)[0] / np.float32(2)], np.float32)
    return 0 if g == 0x8000 else P.f32_to_bf16_rn_bits(int(f.view(np.uint32)[0]), 0x7FC0)


def _index(g: int) -> int:
    return g & 0x7F | ((g >> 7) & 0xF) << 7 | (g >> 15) << 11


def test_the_band_is_where_the_closed_forms_fail() -> None:
    """Every word off the band is one of the module docstring's closed forms; on the band the sign is g's and the magnitude
    the table's; and the band's first odd exponent and its last are each needed (a closed form fails on some word there)."""
    silu = [P.SiluMulBf16.evaluate(g, 0x3F80) for g in range(1 << 16)]
    table = BSL.band_table()
    for g, s in enumerate(silu):
        sign, e, m = g >> 15, (g >> 7) & 0xFF, g & 0x7F
        if BSL.BAND_LO <= e <= BSL.BAND_HI[sign]:
            assert s >> 15 == sign and s & 0x7FFF == table[_index(g)]
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


def test_the_expf_band_table_differs_only_where_expf_overflows() -> None:
    """`SiluMulBf16_v2`'s silu half is `_v1`'s closed forms off the band but at NaN and -inf (0x7FFF) and -0 (kept), and on the
    band `expf_band_table` is `band_table` but at the gates whose ``expf(-g)`` overflows, each 0 (``g / inf`` = -0); some of those
    are nonzero in `band_table` (the finding's window)."""
    silu = [act.SiluMulBf16.evaluate(g, 0x3F80) for g in range(1 << 16)]
    old, new = BSL.band_table(), BSL.expf_band_table()
    over = {_index(g) for g in range(0x8000 | BSL.BAND_LO << 7, 0x8000 | (BSL.BAND_HI[1] + 1) << 7)
            if -_f(g) >= act.LN_F32_OVERFLOW}
    assert min(g for g in range(0x8000, 1 << 16) if -_f(g) >= act.LN_F32_OVERFLOW and (g >> 7) & 0xFF < 255) == 0xC2B2
    assert all(new[i] == 0 for i in over)
    assert {i for i in range(len(old)) if old[i] != new[i]} == {i for i in over if old[i]} != set()
    for g, s in enumerate(silu):
        sign, e, m = g >> 15, (g >> 7) & 0xFF, g & 0x7F
        if e == 255 and (m or sign):
            assert s == BSL.NAN_F2FP
        elif BSL.BAND_LO <= e <= BSL.BAND_HI[sign]:
            assert s >> 15 == sign and s & 0x7FFF == new[_index(g)]
        elif e < BSL.BAND_LO:
            assert s == (0x8000 if g == 0x8000 else _half(g))
        else:
            assert s == (0x8000 if sign else g)


def test_the_finding_pair() -> None:
    """cov-gm001's element: gate -97.0, up 16.375. The GPU commits -0 (``expf(97)`` overflows); `_v1`, and so `_v3`, does not."""
    g, u = np.array([0xC2C2], np.uint16), np.array([0x4183], np.uint16)
    assert _bits(bind(BSL.SiluMulBf16Expf), g, u).tolist() == _word(bind(BSL.SiluMulBf16Expf), g, u).tolist() == [0x8000]
    assert _bits(bind(BSL.SiluMulBf16), g, u).tolist() == _word(bind(BSL.SiluMulBf16), g, u).tolist() == [0x8010]


@pytest.mark.parametrize("version", sorted(VERSIONS))
@pytest.mark.parametrize("u", [0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x8080, 0x7F7F, 0xFF80, 0x7FC0, 0x4000, 0x3B80, 0x0080,
                               0x4183])
def test_every_gate_word(version: str, u: int) -> None:
    """All 65,536 gate words at an up word: at 1.0 the silu half alone, then signs, zeros, subnormals, the finite maximum, an
    infinity, a NaN, scales that move the product's exponent, and the finding's 16.375."""
    g = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    _agree(bind(VERSIONS[version][0]), g, np.full(g.shape, u, np.uint16))


@pytest.mark.slow
def test_the_overflow_window_at_every_up_word() -> None:
    """`SiluMulBf16_v4` on the 31 gates of `OVERFLOW_WINDOW` at all 65,536 up words (2,031,616 pairs, the slow word's ~20 s):
    the window is where `_v2` and `_v1` part, and every up word is every scale and special the -0 meets there."""
    u = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    _agree(bind(BSL.SiluMulBf16Expf), np.repeat(OVERFLOW_WINDOW, len(u)), np.tile(u, len(OVERFLOW_WINDOW)))


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_special_pairs_random_and_edge_products(version: str) -> None:
    """Every pair of the special words, 100,000 random pairs and 20,000 products at f32's subnormal and overflow ends."""
    rng = np.random.default_rng(0x5111)
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=2)), np.int64).T
    rand = rng.integers(0, 1 << 16, (2, 100_000), dtype=np.int64).astype(np.uint16)
    eg, eu = _edge_products(rng, 20_000)
    g = np.concatenate([SPECIALS[grid[0]], rand[0], eg])
    u = np.concatenate([SPECIALS[grid[1]], rand[1], eu])
    _agree(bind(VERSIONS[version][0]), g, u)


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_silu_mul_rows(version: str) -> None:
    rng = np.random.default_rng(0x5112)
    rows = np.concatenate([rng.integers(0, 1 << 16, (4000, 16), dtype=np.int64).astype(np.uint16),
                           SPECIALS[rng.integers(0, len(SPECIALS), (1000, 16))]])
    _agree(bind(VERSIONS[version][1], I=8), rows)


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_one_unit_per_element_under_q_word(version: str) -> None:
    row = bind(VERSIONS[version][1], I=2)
    for target in (row, TR.word_of(row)):
        prog, _ = standalone_call(target)
        part = PO.evaluate(prog, PO.query(16, 32))
        assert part.population == 2
        assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
