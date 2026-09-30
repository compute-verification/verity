"""`Fp8GroupQuantPacked_v1{K,G}`: vLLM's `_C.per_token_group_fp8_quant_packed` (DeepGEMM's activation quantizer: UE8M0 scales, four
to an int32) on one bf16 row.

What is under test: the two primitives' bit rules; the Definition equals the vectorised row (`kernels.fp8_packed_rows`) and a scalar
model written from the kernel source (`per_token_group_quant_8bit_packed_register_kernel`), on edge groups (all zero, NaN, infinities,
subnormals, bf16's largest word, the scale at a power of two and just above it, e4m3 rounding ties) and random rows, with and without
padding bytes in the last word; `Q_word_v1` cuts it."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from verity.ir import Program, bind
from verity_vllm.program.kernels.fp8_packed_rows import fp8_group_quant_packed_row
from verity_vllm.program.registry import fp8_packed as FP
from verity_vllm.query import word as W

from tests.program.test_fp8 import _evaluate_returns

G = 128


def _f32(v: float) -> float:
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _bf16(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (w & 0xFFFF) << 16))[0]


def _e4m3_values() -> list[tuple[float, int]]:
    out = []
    for c in range(0x7F):                                             # the positive codes, 0x7F (NaN) excluded
        e, m = c >> 3, c & 7
        out.append(((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7), c))
    return out


E4M3 = _e4m3_values()


def _e4m3_rne(v: float) -> int:
    """cvt.rn.satfinite.e4m3 of an f32 in [-448, 448]: nearest code, ties to the even code."""
    sign = 0x80 if math.copysign(1.0, v) < 0 else 0
    a = abs(v)
    best = min(E4M3, key=lambda t: (abs(t[0] - a), t[1] & 1))
    return sign | best[1]


def _model_row(x: list[int]) -> tuple[list[int], list[int]]:
    """The kernel, from its source: fmaxf fold from eps, div.rn by 448, fmaxf with 1e-10f, the exponent field plus one on a nonzero
    mantissa, x * (1 / 2^(e - 127)), fminf(fmaxf(., -448), 448) (a NaN operand yields the other), e4m3; bytes packed little-endian."""
    eps = _f32(1e-10)
    qs, es = [], []
    for g in range(len(x) // G):
        vals = [_bf16(w) for w in x[g * G:(g + 1) * G]]
        m = eps
        for v in vals:
            if not math.isnan(v):
                m = max(m, abs(v))
        y = max(_f32(m / 448.0) if math.isfinite(m) else math.inf, eps)
        u = _bits(y)
        e = (((u >> 23) & 0xFF) + (1 if u & 0x7FFFFF else 0)) & 0xFF
        es.append(e)
        inv = 0.0 if e == 255 else 2.0 ** (127 - e)
        for v in vals:
            p = math.nan if (math.isnan(v) or (math.isinf(v) and inv == 0.0)) else _f32(v * inv)
            p = -448.0 if math.isnan(p) else max(p, -448.0)
            qs.append(_e4m3_rne(min(p, 448.0)))
    es += [0] * (-len(es) % 4)
    return qs, [es[j] | es[j + 1] << 8 | es[j + 2] << 16 | es[j + 3] << 24 for j in range(0, len(es), 4)]


def _edge_row() -> np.ndarray:
    """Five groups: all zero; the scale exactly 1 (448 present) with e4m3 ties; the scale one bf16 step above 1 with NaNs, subnormals
    and -0; an infinity with finite words and a NaN; bf16's largest word with small ones."""
    rng = np.random.default_rng(3)
    g0 = np.zeros(G, np.uint16)
    g1 = np.array([0x43E0] + [0x3F88, 0xBF88, 0x3F98, 0x3E08, 0x3C88, 0x3B08] * 21 + [0x3F80], np.uint16)[:G]   # 448; 1.0625 and other ties
    g2 = rng.integers(0, 0x43E0, G).astype(np.uint16) | (rng.integers(0, 2, G) << 15).astype(np.uint16)
    g2[:6] = [0x43E1, 0x7FC0, 0xFFC1, 0x0001, 0x807F, 0x8000]                                                   # 448 * (1 + 2^-7); NaNs; subnormals; -0
    g3 = rng.integers(0, 0x4400, G).astype(np.uint16)
    g3[:3] = [0x7F80, 0xFF80, 0x7FC0]
    g4 = rng.integers(0, 0x3F80, G).astype(np.uint16)
    g4[7] = 0x7F7F
    return np.concatenate([g0, g1, g2, g3, g4])


def _random_row(K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = (rng.standard_normal(K) * 2.0 ** rng.integers(-30, 30, K // G).repeat(G)).astype(np.float32)
    v[rng.integers(0, K, K // 64)] *= 1000.0
    return (v.view(np.uint32) >> 16).astype(np.uint16)


def test_the_ue8m0_ceiling_and_the_pack():
    assert FP.F32Ue8m0Ceil.evaluate(0x3F800000) == 127 and FP.F32Ue8m0Ceil.evaluate(0x3F800001) == 128 and FP.F32Ue8m0Ceil.evaluate(0x3F7FFFFF) == 127
    assert FP.F32Ue8m0Ceil.evaluate(0x7F800000) == 255 and FP.F32Ue8m0Ceil.evaluate(0x00000000) == 0 and FP.F32Ue8m0Ceil.evaluate(0xBF800000) == 127
    assert FP.F32Ue8m0Ceil.evaluate(_bits(_f32(1e-10))) == ((_bits(_f32(1e-10)) >> 23) & 0xFF) + 1
    assert FP.U8x4Pack.evaluate(0x01, 0x02, 0x03, 0x1FF) == 0xFF030201


@pytest.mark.parametrize("row", ["edge", "random-256", "random-2048"])
def test_the_definition_is_the_kernel_model_and_the_vectorised_row(row):
    x = {"edge": _edge_row, "random-256": lambda: _random_row(256, 5), "random-2048": lambda: _random_row(2048, 6)}[row]()
    K = x.size
    want_q, want_w = _model_row(x.tolist())
    got = _evaluate_returns(Program(bind(FP.Fp8GroupQuantPacked, K=K, G=G)), [x])
    assert got[:K] == want_q, [i for i in range(K) if got[i] != want_q[i]][:8]
    assert got[K:] == want_w, (got[K:], want_w)
    q, w = fp8_group_quant_packed_row(x, G=G)
    assert q.tolist() == want_q and w.tolist() == want_w


def test_the_edge_groups_scales_and_padding():
    x = _edge_row()
    q, w = fp8_group_quant_packed_row(x, G=G)
    e = [int(w[0]) >> (8 * i) & 0xFF for i in range(4)] + [int(w[1]) & 0xFF]
    assert e[1] == 127 and e[2] == 128 and e[3] == 255                   # 448 -> scale 1; one step above -> 2; an infinity -> byte 255
    assert int(w[1]) >> 8 == 0                                           # groups 5..7 do not exist: zero bytes
    assert q[G + 1] == 0x38 and q[2 * G + 1] == 0xFE and q[2 * G + 5] == 0x80   # a tie to the even code; NaN -> -448; -0 stays -0
    assert q[3 * G] == 0xFE and q[3 * G + 3] in (0x00, 0x80)            # inf * 0 -> NaN -> -448; a finite word * 0 -> +-0


def test_q_word_cuts_it():
    rule = W.unit_rule(bind(FP.Fp8GroupQuantPacked, K=256, G=128))
    assert rule["units_per_call"] >= 1 and rule["max_out_bits"] <= 32, rule
