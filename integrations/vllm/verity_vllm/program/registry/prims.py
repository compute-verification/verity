"""One-output primitives for the B1 eager batch-invariant profile (registered `Name_vN`, total on encodings).

Encodings: bf16 = `Value<16>` (IEEE binary16-exponent bfloat16 bit pattern), f32 = `Value<32>`, i32 =
`Value<32>` (two's complement), bit = `Value<1>`. Every evaluator is total: NaN/inf/signed-zero/subnormal
encodings map to ordinary outputs following the registered semantics stated in each docstring.

The FP32 and BF16 arithmetic primitives (`F32Add` … `GuardNegInfZero`, `Bf16Add` … `Bf16MulBf16`) and their encoding
helpers are core's (`verity.ml.fp32`) and are re-exported here under the same names, as are the MUFU primitives
(`MufuEx2Ftz`, `MufuRcpFtz`, `Fa2InvSum`, `MufuSqrtFtz`, `DivFullRcp`, `DivFullScaleA`, `RsqrtApprox`, `MufuTanh`) and
their measured sm_89 tables (`verity.ml.mufu`).

Hardware-shaped primitives wrap the existing exact models, they do not re-derive them:
  * `TcDot16` — `veritor.core.silicon_total` (`mma.sync.m16n8k16` k-step, total on every encoding), evaluated through
    `registry.tc_total` (the pure-Python 3.11 port the pods run; `veritor` / `verity_vllm.program.kernels.mma` are the test-only references).
Conformance statuses are per definition and are reported by `conformance.py`; the MUFU tables are conformance
evidence measured on sm_89 — this pod is sm_80 (A100), so every table-backed primitive is at most
`model-untested` on this architecture until the correspondence run says otherwise.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from verity.ir.defs import PrimitiveDefinition, primitive
from verity.ir.types import Array, Value
# Re-exported: this package reaches the FP32/BF16 primitives and helpers through this module.
from verity.ml.fp32 import (BF16_NAN_F2FP, BF16_NAN_TORCH, F32_NAN, Bf16Add, Bf16AddF2fp, Bf16MulBf16, Bf16MulF32, F32Add,
                            F32AddFtz, F32Div, F32Fma, F32FmaFtz, F32FmaSubFtz, F32Max, F32Mul, F32MulFtz, F32SubFtz,
                            GuardNegInfZero, _decode, _round_f32, bf16_to_f32_bits, bits_of, f32_add_bits, f32_div_bits,
                            f32_fma_bits, f32_is_nan, f32_max_bits, f32_mul_bits, f32_of, f32_sub_bits, f32_to_bf16_rn_bits,
                            ftz_bits)
# Re-exported: the MUFU primitives, their tables and MufuTanh's measured shard are core's.
from verity.ml.mufu import (MUFU_TANH_RULES, MUFU_TANH_SHARD_FORMAT, MUFU_TANH_TABLE_DIR, MUFU_TANH_TABLE_FORMAT,
                            MUFU_TANH_TABLE_MANIFEST, MUFU_TANH_TABLE_MANIFEST_SHA256, DivFullRcp, DivFullScaleA, Fa2InvSum,
                            MufuEx2Ftz, MufuRcpFtz, MufuSqrtFtz, MufuTanh, MufuTanhUnmeasured, RsqrtApprox, _tanh_shards,
                            mufu_tanh_coverage, mufu_tanh_table_manifest)
from verity.ml.prims import Bf16ToF32, F2fpBf16, F32ToBf16Rn
from verity.ml.scalar import Bf16GtStrict, I32Add, I32Eq, I32Le, SelectBf16, SelectF32, SelectI32
from verity.ml.tc.silicon import AMPERE_BF16_M16N8K16 as _AMPERE_BF16_M16N8K16
from verity.ml.tc.total import tc_dot_total as _tc_dot_total

BF16, F32, I32, BIT = Value(16), Value(32), Value(32), Value(1)

np.seterr(all="ignore")


def _silu_f32_bits(g_f32_bits: int) -> int:
    """W4b's registered stand-in for the CUDA `silu` (expf + division): RN32 of the exact real value
    g / (1 + e^-g). Total: NaN -> NaN, +inf -> +inf, -inf -> -0 (exact limit)."""
    c, v = _decode(g_f32_bits)
    if c == 2:
        return F32_NAN
    if c == 1:
        return 0x7F800000
    if c == -1:
        return 0x80000000
    g = float(v)
    if g < -80.0:
        return bits_of(np.float32(g * math.exp(g)))  # underflows to -0 well before this
    return bits_of(np.float32(g / (1.0 + math.exp(-g))))


@primitive("SiluMulBf16", 1, [("g", BF16), ("u", BF16)], BF16, conformance="exact-model-tested")
def SiluMulBf16(g: int, u: int) -> int:
    """vLLM CUDA `act_and_mul_kernel<silu>` (eager): bf16(f32(bf16(silu_f32(g))) * u); W4b's model, exact on
    every captured element in numerics.md; `silu_f32` is the float64-exact stand-in for expf+div."""
    s = f32_to_bf16_rn_bits(_silu_f32_bits(bf16_to_f32_bits(g)), BF16_NAN_TORCH)
    return f32_to_bf16_rn_bits(f32_mul_bits(bf16_to_f32_bits(s), bf16_to_f32_bits(u)), BF16_NAN_TORCH)


@primitive("RopeOut", 1, [("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16)], BF16, conformance="exact-model-tested")
def RopeOut(x: int, y: int, c: int, s: int) -> int:
    """vLLM CUDA `rotary_embedding_kernel` (NeoX): bf16_rn(RN32(x*c - y*s)) with the bf16 cos/sin cache
    words; products of two bf16 are exact in f32, so this equals fma(x, c, -(y*s)) (W11/numerics.md model)."""
    t = f32_mul_bits(bf16_to_f32_bits(y), bf16_to_f32_bits(s))  # exact
    o = f32_fma_bits(bf16_to_f32_bits(x), bf16_to_f32_bits(c), t ^ 0x80000000)
    return f32_to_bf16_rn_bits(o, BF16_NAN_TORCH)


@primitive("RopeOutAdd", 1, [("x", BF16), ("y", BF16), ("c", BF16), ("s", BF16)], BF16, conformance="exact-model-tested")
def RopeOutAdd(x: int, y: int, c: int, s: int) -> int:
    """Second half of the NeoX rotation: bf16_rn(RN32(y*c + x*s))."""
    t = f32_mul_bits(bf16_to_f32_bits(x), bf16_to_f32_bits(s))
    o = f32_fma_bits(bf16_to_f32_bits(y), bf16_to_f32_bits(c), t)
    return f32_to_bf16_rn_bits(o, BF16_NAN_TORCH)


# ---------------------------------------------------------------------------------------------------------
# tensor core k-step
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _mma():
    """The finite reference (`verity_vllm.program.kernels.mma` -> `verity.ml.tc.silicon`): used by the tests as one of the oracles
    `_tc_dot16_total` is checked against; NOT used at evaluation time."""
    from verity_vllm.program.kernels import mma  # noqa: WPS433
    return mma


def _tc_dot16_total(acc: int, a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """`AmpereBF16TcDot16`: acc + sum_{k<16} a[k]*b[k] as `mma.sync.m16n8k16` computes one accumulator coordinate --
    `tc_total.tc_dot_total` on `AMPERE_BF16_M16N8K16` (groups (8, 8), 25-bit truncating adder, exponent floor -132), the
    pure-Python (3.11) gate-for-gate port of `veritor.core.silicon_total.tc_dot_total(AMPERE_BF16_DOT16_TOTAL)` (TOTAL family
    version 3, measured on the A100 and the RTX 4090): on finite operands = `silicon.tc_dot(AMPERE_BF16_M16N8K16)` (the R15/R16
    relation, unchanged); on every other encoding the MEASURED total rules -- any NaN operand / acc, inf * 0, or infinities of
    both signs (an overflowed group counts as its infinity) -> the canonical 0x7FFFFFFF; a one-signed infinity -> that infinity.
    [R17 dense, int-3] Before this the prim evaluated through `verity_vllm.program.kernels.mma` (unimportable on the pods' 3.11) with a
    documented-not-measured extension path that gave 0x7FC00000 for NaN and ignored a group's finite overflow beside an infinite
    operand; both differences are on non-finite outcomes only, which the vectorised twin flags and the sampled replay never
    compares (tests/program/test_derived_rows.py states them)."""
    return _tc_dot_total(_AMPERE_BF16_M16N8K16, acc, a, b)


A16 = Array(16, BF16)


@primitive("AmpereBF16TcDot16", 1, [("acc", F32), ("a", A16), ("b", A16)], F32, conformance="exact-model-tested")
def AmpereBF16TcDot16(acc: int, *ab: int) -> int:
    """One accumulator-coordinate transition of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    (16 products into `acc`): `verity.ml.tc.total.tc_dot_total` on `AMPERE_BF16_M16N8K16` (see `_tc_dot16_total`)."""
    return _tc_dot16_total(acc, tuple(ab[:16]), tuple(ab[16:32]))


# ---------------------------------------------------------------------------------------------------------
# the C++ models the MUFU primitives were transliterated from, with their tables (twins, the replay driver)
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _fa2():
    from verity_vllm.program.kernels import fa2_model, fa2_relation
    return fa2_model, fa2_relation.tables()


@lru_cache(maxsize=None)
def _rms():
    from verity_vllm.program.kernels import rms_relation, rms_triton_model
    return rms_triton_model, rms_relation.tables("cuda")


# ---------------------------------------------------------------------------------------------------------
# gather (wide fan-in: the whole vocabulary column is an operand collection)
# ---------------------------------------------------------------------------------------------------------

_GATHERS: dict[int, PrimitiveDefinition] = {}


def gather_bf16(vocab: int) -> PrimitiveDefinition:
    """`GatherBf16x{V}`: (token: i32, column: Array<V, bf16>) -> bf16 = column[token]; out-of-range or negative
    token -> 0x7FFF (a total, documented extension; production would fault)."""
    p = _GATHERS.get(vocab)
    if p is None:
        def ev(tok: int, *col: int, _V=vocab) -> int:
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return col[t] if 0 <= t < _V else 0x7FFF

        def ev_lazy(get, _V=vocab) -> int:
            tok = get(0)
            t = tok - (1 << 32) if tok & 0x80000000 else tok
            return get(1 + t) if 0 <= t < _V else 0x7FFF

        p = PrimitiveDefinition(f"GatherBf16x{vocab}", 1, (("token", I32), ("column", Array(vocab, BF16))), BF16, ev,
                                doc=gather_bf16.__doc__ or "", conformance="exact-model-tested")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _GATHERS[vocab] = p
    return p


ALL_SCALAR = [Bf16ToF32, F32ToBf16Rn, F2fpBf16, F32Add, F32Mul, F32Fma, F32Div, F32AddFtz, F32SubFtz, F32MulFtz,
              F32FmaFtz, F32FmaSubFtz, F32Max, GuardNegInfZero, Bf16Add, Bf16AddF2fp, Bf16MulF32, Bf16MulBf16, SiluMulBf16,
              RopeOut, RopeOutAdd, AmpereBF16TcDot16, MufuEx2Ftz, MufuRcpFtz, Fa2InvSum, MufuSqrtFtz, DivFullRcp,
              DivFullScaleA, RsqrtApprox, I32Le, I32Add, I32Eq, SelectF32, SelectBf16, SelectI32, Bf16GtStrict]
