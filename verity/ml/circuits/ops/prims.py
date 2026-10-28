"""Registered primitives over the tensor-core semantics: the BF16 k16 and E4M3 k32 accumulator steps and the
BF16/E4M3/FP32 casts.

Encodings: bf16 = `Value<16>`, e4m3 = `Value<8>`, f32 = `Value<32>`.  Every evaluator is total on encodings.  The
evaluators are the functions of `verity.ml.circuits.silicon`; nothing is re-derived here.

Provenance.  `Bf16ToF32_v1`, `F32ToBf16Rn_v1`, `F2fpBf16_v1` are the definitions registered in veritor
`verity.ir/registry/prims.py` (same ids, same functions).  `HopperBF16WgmmaDot16_v1` is veritor's
`verity.ir/registry/hopper.py` definition (it already evaluated through `silicon_total.tc_dot_total`).
`AmpereBF16TcDot16` is registered here as **version 2**: veritor's `_v1` evaluated the finite domain through the same
model but extended it to NaN/Inf encodings by a hand-written rule ("documented, not measured") that disagrees with the
A100/RTX 4090 campaign on the NaN word (`0x7FC00000` vs the measured `0x7FFFFFFF`) and on a saturated group followed by
an opposite-sign infinity (rule R6).  `_v2` is `tc_dot_total(AMPERE_BF16_M16N8K16, ...)`, the measured total function;
equal to `_v1` on every finite input.  A registered id denotes one function, so the id changed.
`F32ToE4m3Sat_v1` and `HopperE4m3QgmmaDot32_v1` are the definitions the vLLM integration registered for its FP8 linear
(same ids, same functions).  `BlackwellE4m3QmmaDot32_v1` is the consumer-Blackwell (sm_120) FP8 `mma.sync` step,
`tc_dot_total_e4m3` on `models.BLACKWELL_SM120_E4M3_M16N8K32`, and `BlackwellE5m2QmmaDot32_v1` the same with E5M2 operands
(`tc_dot_total_e5m2`); `BlackwellNvf4OmmaDot64_v1` its NVFP4 step
(`models.BLACKWELL_SM120_NVF4`), with the two casts NVFP4 quantization uses, `F32ToE2m1Sat_v1` and `E4m3ToF32_v1`, and
`BlackwellMxf4OmmaDot64_v1` its MXFP4 step (`models.BLACKWELL_SM120_MXF4`).  `HopperE4m3MmaV2Dot32_v1` is the k32 step of an e4m3
`tl.dot` that Triton 3.7.1 lowers to MMAv2 on sm_90 (`total_fp8.HOPPER_E4M3_MMAV2_K32` in `total_fp8.MMAV2_E4M3_K_ORDER`).
`BlackwellS8ImmaDot32_v1` is sm_120's s8 `mma.sync` step with `.satfinite` (`integer.BLACKWELL_SM120_S8_M16N8K32`, an s32
accumulator word), and `S32ToF32Rn_v1` the `cvt.rn.f32.s32` that hands its sum to an f32 epilogue.
`BlackwellU8S8ImmaDot32_v1` is the step's one sibling, u8 by s8 without `.satfinite` (`integer.BLACKWELL_SM120_U8S8_M16N8K32`,
wrapping modulo 2^32), from the same probe run.  Each tensor-core step is bound to the model it evaluates through
(`census.silicon_step`: `.silicon`), whose operand format is its census facet `tc:<precision>`.
The two s8 steps are also the H200's (sm_90a: `wgmma` m64nNk32 and `mma.sync` m16n8k32) and the B200's (sm_100a: `tcgen05.mma`
kind::i8 and `mma.sync` m16n8k32) at the same formats: one function on every card (`integer.HOPPER_SM90_S8_K32`,
`integer.BLACKWELL_SM100_S8_K32`), so a circuit over them names no architecture (`gemm.s8_gemm`: `gemm.GEMM_SM90_S8` is the
H200's binding, this step).

The store ids a Definition's conformance cites are `EVIDENCE_IDS` (`prims_evidence.json`), keyed by Definition id and
then by role: the conformance names each role it cites in backticks (`run`, or a dotted path into a nested role such as
`specials.capture`).
"""
from __future__ import annotations

import json
from pathlib import Path

from verity.ml.circuits.census import silicon_step
from verity.ml.circuits.defs import PrimitiveDefinition, primitive
from verity.ml.circuits.types import Array, Value

from verity.ml.circuits.silicon import cast
from verity.ml.circuits.silicon import models as _tc_models
from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, BLACKWELL_SM120_U8S8_M16N8K32
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_E4M3_M16N8K32, BLACKWELL_SM120_E5M2_M16N8K32, BLACKWELL_SM120_MXF4, BLACKWELL_SM120_NVF4
from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from verity.ml.circuits.silicon.total_fp8 import (HOPPER_E4M3_MMAV2_K32, HOPPER_E4M3_WGMMA_K32, MMAV2_E4M3_K_ORDER, in_mmav2_e4m3_k_order,
                                                  tc_dot_total_e4m3, tc_dot_total_e5m2)

BF16, F32, E4M3, E5M2, E2M1 = Value(16), Value(32), Value(8), Value(8), Value(4)
A16 = Array(16, BF16)
A32_8 = Array(32, E4M3)
A32_E5M2 = Array(32, E5M2)       # the same type as A32_8: an encoding type carries only its width
A64_4 = Array(64, E2M1)
A4_8 = Array(4, E4M3)            # four UE4M3 block-scale bytes (the E4M3 encoding with the sign bit clear)
A2_8 = Array(2, Value(8))        # two UE8M0 block-scale bytes
S8, S32 = Value(8), Value(32)    # two's-complement integer words
A32_S8 = Array(32, S8)
A32_U8 = Array(32, Value(8))     # 32 u8 codes: the same type as A32_S8, read unsigned
#: `cvt.rn.satfinite.e2m1x2.f32`'s code for a NaN input (E2M1 has none): +6, measured on the RTX PRO 6000 for every NaN element
#: of `cvt_fp16_to_fp4` (whose FMUL hands it the canonical +NaN; the sign of a NaN input is not observable there, and dropped)
E2M1_NAN_CODE = 0x7

__all__ = ["A16", "A2_8", "A32_8", "A32_E5M2", "A32_S8", "A32_U8", "A4_8", "A64_4", "BF16", "E2M1", "E4M3", "E5M2", "EVIDENCE_IDS", "F32", "S32", "S8", "AmpereBF16TcDot16", "Bf16ToF32",
           "BlackwellE4m3QmmaDot32", "BlackwellE5m2QmmaDot32", "BlackwellMxf4OmmaDot64", "BlackwellNvf4OmmaDot64", "BlackwellS8ImmaDot32", "BlackwellU8S8ImmaDot32", "E4m3ToF32", "F2fpBf16", "F32ToBf16Rn", "F32ToE2m1Sat", "F32ToE4m3Sat", "HopperBF16WgmmaDot16",
           "HopperE4m3MmaV2Dot32", "HopperE4m3QgmmaDot32", "S32ToF32Rn", "const"]

#: Definition id -> role -> the store id (run, ``art:``) its conformance names by that role.
EVIDENCE_IDS: dict[str, dict] = json.loads(Path(__file__).with_name("prims_evidence.json").read_text())


@primitive("Bf16ToF32", 1, [("a", BF16)], F32, conformance="exact-model-tested")
def Bf16ToF32(a: int) -> int:
    """Exact widening: bf16 encoding << 16."""
    return cast.bf16_to_f32_word(a)


@primitive("F32ToBf16Rn", 1, [("a", F32)], BF16,
           conformance="exact-model-tested; the software canonicalisation (c10::BFloat16::round_to_nearest_even); equal to "
                       "F2fpBf16_v1 on every non-NaN word")
def F32ToBf16Rn(a: int) -> int:
    """torch round-to-nearest-even f32 -> bf16; NaN -> 0x7FC0 (the software canonical word)."""
    return cast.f32_to_bf16_word(a)


@primitive("F2fpBf16", 1, [("a", F32)], BF16,
           conformance="exact-model-tested; cvt.rn.bf16.f32 / F2FP.BF16.F32: NaN -> 0x7FFF on the A100 and the RTX 4090 for all "
                       "84 measured NaN inputs (tc-total-2026-09-07 cvt_special, 4403 words per device, 0 mismatches)")
def F2fpBf16(a: int) -> int:
    """SASS `F2FP.BF16.F32.PACK_AB` (`cvt.rn.bf16.f32`): RNE f32 -> bf16; every NaN -> 0x7FFF (measured)."""
    return cast.f32_to_bf16_hw_word(a)


@primitive("AmpereBF16TcDot16", 2, [("acc", F32), ("a", A16), ("b", A16)], F32,
           conformance="exact-model-tested; finite domain: Hawkeye A100 parameters, RTX 4090 24,898,304/24,898,304 words bit-exact; "
                       "non-finite encodings: A100 + RTX 4090 special-value campaign (tc-total-2026-09-07), 0 mismatches")
def AmpereBF16TcDot16(acc: int, *ab: int) -> int:
    """One accumulator-coordinate transition of `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`: two groups of
    eight exact products (the accumulator joins the first), 25-bit truncating adder, exponent floor -132; total on every
    encoding (NaN -> 0x7FFFFFFF, sticky/saturating infinities, zero always +0)."""
    return tc_dot_total(AMPERE_BF16_M16N8K16, acc, ab[:16], ab[16:32])


@primitive("HopperBF16WgmmaDot16", 1, [("acc", F32), ("a", A16), ("b", A16)], F32,
           conformance="exact-model-tested; H100 HGMMA.64x128x16.F32.BF16 campaign (tc-hopper-2026-09-07): 307,281,920 output words, "
                       "0 mismatches, 0 nondeterministic tiles")
def HopperBF16WgmmaDot16(acc: int, *ab: int) -> int:
    """One accumulator-coordinate transition of `wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16`: one group of
    sixteen products plus the accumulator, 26-bit truncating adder, exponent floor -133; total on every encoding."""
    return tc_dot_total(HOPPER_BF16_WGMMA_K16, acc, ab[:16], ab[16:32])


@primitive("F32ToE4m3Sat", 1, [("x", F32)], E4M3,
           conformance="exact-model-tested; every f32 word (2^32) through cvt.rn.satfinite.e4m3x2.f32 on the H100 (tc-fp8-2026-09-08 "
                       "cvt_e4m3): 0 mismatches; NaN of either sign -> 0x7F")
def F32ToE4m3Sat(x: int) -> int:
    """`cvt.rn.satfinite.e4m3x2.f32` (SASS `F2FP.E4M3.F32.PACK_AB` with SATFINITE), one lane: `cast.f32_to_e4m3_sat_word`."""
    return cast.f32_to_e4m3_sat_word(x)


@primitive("HopperE4m3QgmmaDot32", 1, [("acc", F32), ("a", A32_8), ("b", A32_8)], F32,
           conformance="exact-model-tested; H100 80GB QGMMA.64x64x32.F32.E4M3.E4M3 campaign (tc-fp8-2026-09-08): 16 families, "
                       "35,289 tiles x3, 144,543,744 words, 0 mismatches")
def HopperE4m3QgmmaDot32(acc: int, *ab: int) -> int:
    """One accumulator-coordinate transition of `wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3` (SASS
    `QGMMA.64x64x32.F32.E4M3.E4M3`): one group of 32 exact E4M3 products plus `acc` through a 14-bit truncating adder,
    exponent floor -139 (only bounded to [-152, -126] by the measurement; every floor in that range computes the same word
    on E4M3 operands); total on every encoding (NaN -> 0x7FFFFFFF, sticky/saturating infinities, zero always +0).  The
    accumulator itself passes through the 14-bit adder, so `acc + 0 != acc` for most f32 words."""
    return tc_dot_total_e4m3(HOPPER_E4M3_WGMMA_K32, acc, list(ab[:32]), list(ab[32:64]))


@primitive("HopperE4m3MmaV2Dot32", 1, [("acc", F32), ("a", A32_8), ("b", A32_8)], F32,
           conformance="exact-model-tested; vLLM d9105ea80's fused_moe_kernel on the H200 (cc 9.0, 132 SMs) at vLLM's defaults, at "
                       "GLM-5.3's tuned decode tiles 8x32x128 and 8x64x128/SWAP_AB (Triton 3.7.1: 8 chained "
                       "mma.sync.m16n8k16.f32.f16.f16.f32 per 128-wide K tile, cvt.rn.f16x2.e4m3x2; SASS F2FP.F16.E4M3.UNPACK_B, "
                       "HMMA.16816.F32): 638,976 w13 and w2 words, 0 mismatches, and an operand probe (+448 * +448 and +448 * -448 "
                       "in the two k16 halves over small codes) on which the natural k order, the halves swapped, adder widths 24, "
                       "25, 27 and 28, Ampere's 8 + 8 groups, one k32 group, QGMMA's step and a chain of FMAs each differ "
                       "(integrations/vllm/tests/properties/glm53_block_fp8_capture_gpu.py v4; run `run`, `art`); HMMA.16816.F32 "
                       "from +0 is HOPPER_BF16_M16N8K16's group on the H100 (`h100`)")
def HopperE4m3MmaV2Dot32(acc: int, *ab: int) -> int:
    """One 32-wide k block of an e4m3 `tl.dot` that Triton 3.7.1 lowers to MMAv2 on sm_90 (a dot whose M is not a multiple of 64 or
    whose N is not a multiple of 16, or under fewer than 4 warps): the operands promoted to f16 exactly, then two chained
    `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (SASS `HMMA.16816.F32`), the first over the k with k mod 4 in {0, 1}
    from `acc`, the second over k mod 4 in {2, 3} from the first's result; each one group of 16 exact products plus its
    accumulator through the 26-bit truncating adder (`total_fp8.HOPPER_E4M3_MMAV2_K32` in `MMAV2_E4M3_K_ORDER`), total on every
    encoding (NaN -> 0x7FFFFFFF, sticky/saturating infinities, zero always +0).  Not ptxas's own lowering of PTX
    `mma.sync.m16n8k32` e4m3 on sm_90 (two HMMA from +0 and IEEE FADDs), and not the wgmma step `HopperE4m3QgmmaDot32_v1`, whose
    14-bit adder truncates the accumulator."""
    return tc_dot_total_e4m3(HOPPER_E4M3_MMAV2_K32, acc, in_mmav2_e4m3_k_order(ab[:32]), in_mmav2_e4m3_k_order(ab[32:64]))


@primitive("BlackwellE4m3QmmaDot32", 1, [("acc", F32), ("a", A32_8), ("b", A32_8)], F32,
           conformance="exact-model-tested; RTX PRO 6000 Blackwell Server Edition (cc 12.0) QMMA.16832.F32.E4M3.E4M3: fresh-seed sweep "
                       "`sweep`, 25,041,024 words in 14 families, 0 mismatches (trust dossier `dossier`, PINNED); specials under "
                       "this total function: `specials.capture` evaluated by `specials.eval`, 0 mismatches")
def BlackwellE4m3QmmaDot32(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` on sm_120a (SASS
    `QMMA.16832.F32.E4M3.E4M3`): one group of 32 exact E4M3 products plus `acc` through the 26-bit adder of sm_120's BF16
    pipeline, exponent floor -133 (unobservable on E4M3 operands); total on every encoding.  Unlike the Hopper and Ada FP8
    adders (14 bits), the accumulator is not truncated."""
    return tc_dot_total_e4m3(BLACKWELL_SM120_E4M3_M16N8K32, acc, list(ab[:32]), list(ab[32:64]))


@primitive("BlackwellE5m2QmmaDot32", 1, [("acc", F32), ("a", A32_E5M2), ("b", A32_E5M2)], F32,
           conformance="exact-model-tested; RTX PRO 6000 Blackwell Server Edition (cc 12.0) QMMA.16832.F32.E5M2.E5M2: fresh-seed sweep "
                       "`sweep`, 25,041,024 words in 14 families, 0 mismatches, and `second_die` on a second die, 6,440,960 words "
                       "(trust dossier `dossier`, PINNED); every word of `sweep`'s specials, signed-zero and near-overflow tiles "
                       "under this total function "
                       "(838,912 words, 4,824 with a NaN or infinite operand or accumulator): 0 mismatches")
def BlackwellE5m2QmmaDot32(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` on sm_120a (SASS
    `QMMA.16832.F32.E5M2.E5M2`): `BlackwellE4m3QmmaDot32_v1`'s pipeline (one group of 32 exact products plus `acc`, the 26-bit
    adder, floor -133) with E5M2 products; total on every encoding: a NaN operand, or an infinity times a zero, is a NaN; an
    infinite product joins the group like an infinite accumulator (NaN when both signs meet); NaN -> 0x7FFFFFFF, zero +0."""
    return tc_dot_total_e5m2(BLACKWELL_SM120_E5M2_M16N8K32, acc, list(ab[:32]), list(ab[32:64]))


@primitive("BlackwellNvf4OmmaDot64", 1, [("acc", F32), ("a", A64_4), ("b", A64_4), ("sa", A4_8), ("sb", A4_8)], F32,
           conformance="exact-model-tested; pinned (dossier `dossier`: two RTX 5090 dies, 17.3M words); on the RTX PRO 6000 "
                       "Blackwell Server Edition (driver 580.173.02, CUDA 13.0, SASS-checked) sweeps `sweeps`, 16,932,864 gated "
                       "words each on two GPUs, 0 mismatches, their specials tiles for the accumulator rule; the scale bytes outside "
                       "UE4M3 by `scale_bytes` (1,024 single-instruction tiles, 131,072 words, every rule below exact)")
def BlackwellNvf4OmmaDot64(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... e2m1.e2m1.f32.ue4m3`
    on sm_120a (SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`): 64 E2M1 codes per operand in four blocks of 16, one UE4M3 scale
    byte per block.  Total on every encoding, as measured:

    1. a scale byte whose low seven bits are 0x7F (the UE4M3 NaN, 0xFF included) -> 0x7FFFFFFF, whatever the accumulator;
    2. else a NaN accumulator -> 0x7FFFFFFF, and +/-inf -> itself;
    3. else bit 7 of every scale byte is ignored, and the word is `models.BLACKWELL_SM120_NVF4.step_scaled` (`BlockScaledAlignAdd`:
       exact group dots, lossless block scales, the five-way align-add, truncation, every zero +0).  From a finite accumulator
       the sum cannot leave binary32 (the accumulator's window puts its grid far above every group dot).

    Rule 1 was measured on groups with nonzero products."""
    sa, sb = list(ab[128:132]), list(ab[132:136])
    if any(s & 0x7F == 0x7F for s in sa + sb):
        return 0x7FFFFFFF
    acc &= 0xFFFFFFFF
    if (acc & 0x7F800000) == 0x7F800000:
        return 0x7FFFFFFF if acc & 0x7FFFFF else acc
    return BLACKWELL_SM120_NVF4.step_scaled(acc, list(ab[:64]), list(ab[64:128]), [s & 0x7F for s in sa], [s & 0x7F for s in sb])


def _block_scaled_sum_negative(model, acc: int, a: list[int], b: list[int], sa: list[int], sb: list[int]) -> bool:
    """Whether ``model``'s align-add sum (`BlockScaledAlignAdd._transducer`: the truncated addends on the common grid) is
    negative; read only where that sum leaves binary32."""
    terms = model.group_terms(a, b, sa, sb)
    acc_m, acc_e = _tc_models._fp32_dyadic(acc)
    anchors = [unit - model.frac_bits for _, unit, participates in terms if participates]
    if acc_m:
        anchors.append(max((acc >> 23) & 0xFF, 1) - 127 - (model.acc_window - 1))
    lsb = max(*anchors, model.grid_floor)
    total = sum(_tc_models._trunc_rz_to_grid(m, unit, lsb) for m, unit, _ in terms if m)
    return total + (_tc_models._trunc_rz_to_grid(acc_m, acc_e, lsb) if acc_m else 0) < 0


@primitive("BlackwellMxf4OmmaDot64", 1, [("acc", F32), ("a", A64_4), ("b", A64_4), ("sa", A2_8), ("sb", A2_8)], F32,
           conformance="exact-model-tested; pinned on sm_120 (dossier `dossier`: RTX 5090, 7,389,184 words and 9,658,368, "
                       "0 mismatches); on the RTX PRO 6000 Blackwell Server Edition (driver 580.173.02, CUDA 13.0, SASS-checked) "
                       "run `run`, 1,089,536 gated words against the model, 0 mismatches, and its record-only edge "
                       "families (fixtures tc-sm120-mxf4-2026-10-01: 73,728 words, 9,216 under a 0xFF scale byte, overflowing and "
                       "cancelling blocks, +/-inf and NaN accumulators) every rule below exact")
def BlackwellMxf4OmmaDot64(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64 ... e2m1.e2m1.f32.ue8m0`
    on sm_120a (SASS `OMMA.SF.16864.F32.E2M1.E2M1.E8`): 64 E2M1 codes per operand in two blocks of 32, one UE8M0 scale byte
    (2^(code-127)) per block.  Total on every encoding:

    1. a scale byte 0xFF (the UE8M0 NaN) -> 0x7FFFFFFF, whatever the accumulator;
    2. else a NaN accumulator -> 0x7FFFFFFF, and +/-inf -> itself;
    3. else `models.BLACKWELL_SM120_MXF4.step_scaled` (`BlockScaledAlignAdd`, as for `BlackwellNvf4OmmaDot64_v1`), and a sum
       that leaves binary32 (UE8M0 scales reach it; UE4M3 scales cannot) -> +/-inf of the sum's sign."""
    a, b, sa, sb = list(ab[:64]), list(ab[64:128]), list(ab[128:130]), list(ab[130:132])
    if 0xFF in sa or 0xFF in sb:
        return 0x7FFFFFFF
    acc &= 0xFFFFFFFF
    if (acc & 0x7F800000) == 0x7F800000:
        return 0x7FFFFFFF if acc & 0x7FFFFF else acc
    try:
        return BLACKWELL_SM120_MXF4.step_scaled(acc, a, b, sa, sb)
    except InvalidArtifact:
        return 0xFF800000 if _block_scaled_sum_negative(BLACKWELL_SM120_MXF4, acc, a, b, sa, sb) else 0x7F800000


@primitive("F32ToE2m1Sat", 1, [("x", F32)], E2M1,
           conformance="exact-model-tested; every element of vllm::cvt_fp16_to_fp4 on the RTX PRO 6000 (run `run`: "
                       "NaN, +-inf, signed zeros and the E2M1 rounding midpoints included)")
def F32ToE2m1Sat(x: int) -> int:
    """`cvt.rn.satfinite.e2m1x2.f32` (SASS `F2FP.SATFINITE.E2M1.F32.PACK_AB`), one lane: `cast.f32_to_e2m1_sat_word`."""
    return cast.f32_to_e2m1_sat_word(x, E2M1_NAN_CODE)


@primitive("E4m3ToF32", 1, [("x", E4M3)], F32, conformance="exact-model-tested")
def E4m3ToF32(x: int) -> int:
    """`float(__nv_fp8_e4m3)`: the exact widening `cast.e4m3_to_f32_word` (NaN -> 0x7FFFE000, documented, not measured)."""
    return cast.e4m3_to_f32_word(x)


@primitive("BlackwellS8ImmaDot32", 1, [("acc", S32), ("a", A32_S8), ("b", A32_S8)], S32,
           conformance="exact-model-tested; RTX PRO 6000 Blackwell Server Edition (cc 12.0) IMMA.16832.S8.S8.SAT: run "
                       "`run`, 10,243,968 words, 0 mismatches, the clamp tiles admitting only a clamp at D (trust dossier "
                       "`dossier`, PINNED); the same words on an H200 (IGMMA.64x8x32.S8.S8.SAT and IMMA.16832.S8.S8.SAT, run "
                       "`h200`) and a B200 (UTCIMMA kind::i8 and IMMA.16832.S8.S8.SAT, run `b200`), each PINNED (silicon.instructions sm90.*.s8, "
                       "sm100.*.s8)")
def BlackwellS8ImmaDot32(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` on sm_120a (SASS
    `IMMA.16832.S8.S8.SAT`): `acc` (an s32 word) plus the 32 exact s8 x s8 products, summed exactly and clamped once to
    [-2^31, 2^31 - 1]. Total on every encoding: `integer.BLACKWELL_SM120_S8_M16N8K32.step`. The same function on sm_90a
    (`wgmma` m64nNk32 `.satfinite`, `mma.sync` m16n8k32) and sm_100a (`tcgen05.mma` kind::i8 saturating, `mma.sync`)."""
    return BLACKWELL_SM120_S8_M16N8K32.step(acc, list(ab[:32]), list(ab[32:64]))


@primitive("BlackwellU8S8ImmaDot32", 1, [("acc", S32), ("a", A32_U8), ("b", A32_S8)], S32,
           conformance="exact-model-tested; RTX PRO 6000 Blackwell Server Edition (cc 12.0) IMMA.16832.U8.S8: run "
                       "`run`, 10,243,968 words, 0 mismatches, out-of-range sums wrapping modulo 2^32; the same "
                       "words from the u8.s8 forms without .satfinite on an H200 (IGMMA.64x8x32.U8.S8, IMMA.16832.U8.S8, run "
                       "`h200`) and a B200 (UTCIMMA kind::i8, IMMA.16832.U8.S8, run `b200`)")
def BlackwellU8S8ImmaDot32(acc: int, *ab: int) -> int:
    """One accumulator coordinate of `mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32` on sm_120a (SASS `IMMA.16832.U8.S8`):
    `acc` (an s32 word) plus the 32 exact u8 x s8 products, summed exactly and reduced modulo 2^32 (no `.satfinite`). Total on
    every encoding: `integer.BLACKWELL_SM120_U8S8_M16N8K32.step`. The same function on sm_90a and sm_100a."""
    return BLACKWELL_SM120_U8S8_M16N8K32.step(acc, list(ab[:32]), list(ab[32:64]))


for _step, _model in ((AmpereBF16TcDot16, AMPERE_BF16_M16N8K16), (HopperBF16WgmmaDot16, HOPPER_BF16_WGMMA_K16),
                      (HopperE4m3QgmmaDot32, HOPPER_E4M3_WGMMA_K32), (HopperE4m3MmaV2Dot32, HOPPER_E4M3_MMAV2_K32),
                      (BlackwellE4m3QmmaDot32, BLACKWELL_SM120_E4M3_M16N8K32), (BlackwellE5m2QmmaDot32, BLACKWELL_SM120_E5M2_M16N8K32),
                      (BlackwellNvf4OmmaDot64, BLACKWELL_SM120_NVF4), (BlackwellMxf4OmmaDot64, BLACKWELL_SM120_MXF4),
                      (BlackwellS8ImmaDot32, BLACKWELL_SM120_S8_M16N8K32), (BlackwellU8S8ImmaDot32, BLACKWELL_SM120_U8S8_M16N8K32)):
    silicon_step(_step, _model)


@primitive("S32ToF32Rn", 1, [("a", S32)], F32,
           conformance="exact-model-tested; cvt.rn.f32.s32 (SASS I2FP.F32.S32) is IEEE 754's convertFromInt, round to nearest even; "
                       "the evaluator is that function (`cast.s32_to_f32_rn_word`), checked against numpy's int32 -> float32 on "
                       "every exponent; not separately probed on a device")
def S32ToF32Rn(a: int) -> int:
    """`cvt.rn.f32.s32`: the s32 word as the nearest f32 word, ties to even, zero +0 (`cast.s32_to_f32_rn_word`)."""
    return cast.s32_to_f32_rn_word(a)


# ---- constants: one primitive per (width, bits), materialised on demand and decodable by id -------------------------------------------

_CONSTS: dict[tuple[int, int], PrimitiveDefinition] = {}
#: the constant family's ids, of which the decoder takes only the spelling `const` writes
CONST_ID = r"^Const([1-9][0-9]*)\[0x([0-9a-f]+)\]_v1$"


def const(w: int, bits: int) -> PrimitiveDefinition:
    """`Const<w>[0x..] : () -> Value<w>`: a gate with a fixed correct output (a constant of the program), for w >= 1 and
    0 <= bits < 2^w. Its id spells w in decimal and bits in lowercase hex zero-padded to w // 4 digits (`PROTOCOL.md` §9)."""
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError(f"a constant has a width w >= 1, not {w!r}")
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits < 1 << w:
        raise ValueError(f"a {w}-bit constant is in [0, 2^{w}), not {bits!r}")
    if w == 1:
        from verity.ml.circuits.boolean import constant

        return constant(bits)
    key = (w, bits)
    p = _CONSTS.get(key)
    if p is None:
        p = PrimitiveDefinition(f"Const{w}[0x{bits:0{w // 4}x}]", 1, (), Value(w), (lambda _b=bits: _b),
                                doc=f"constant {w}-bit encoding 0x{bits:0{w // 4}x}", conformance="exact-model-tested")
        p.family = "Const"  # type: ignore[attr-defined]
        _CONSTS[key] = p
    return p


ZERO32 = const(32, 0)


def _register_lazy_families() -> None:
    from verity.ml.circuits.codec import register_lazy_family

    register_lazy_family(CONST_ID, lambda m: const(int(m[1]), int(m[2], 16)))


_register_lazy_families()


def _register_kernels() -> None:
    from verity.ml.circuits.evaluation import register_kernel

    from verity.ml.kernels import tc as K

    every = "every encoding"
    register_kernel(Bf16ToF32, K.bf16_to_f32_batch, domain=every)
    register_kernel(F32ToBf16Rn, K.f32_to_bf16_rn_batch, domain=every)
    register_kernel(F2fpBf16, K.f2fp_bf16_batch, domain=every)
    register_kernel(AmpereBF16TcDot16, lambda acc, a, b: K.group_sum_total_batch(AMPERE_BF16_M16N8K16, acc, a, b),
                    domain=every, sample=K.sample_tc)
    register_kernel(HopperBF16WgmmaDot16, lambda acc, a, b: K.group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc, a, b),
                    domain=every, sample=K.sample_tc)
    register_kernel(HopperE4m3QgmmaDot32, lambda acc, a, b: K.group_sum_total_batch(HOPPER_E4M3_WGMMA_K32, acc, a, b),
                    domain=every, sample=K.sample_tc)
    order = list(MMAV2_E4M3_K_ORDER)
    register_kernel(HopperE4m3MmaV2Dot32,
                    lambda acc, a, b: K.group_sum_total_batch(HOPPER_E4M3_MMAV2_K32, acc, a[..., order], b[..., order]),
                    domain=every, sample=K.sample_tc)
    register_kernel(BlackwellE4m3QmmaDot32, lambda acc, a, b: K.group_sum_total_batch(BLACKWELL_SM120_E4M3_M16N8K32, acc, a, b),
                    domain=every, sample=K.sample_tc)
    register_kernel(BlackwellE5m2QmmaDot32, lambda acc, a, b: K.group_sum_total_batch(BLACKWELL_SM120_E5M2_M16N8K32, acc, a, b),
                    domain=every, sample=K.sample_tc_e5m2)
    register_kernel(BlackwellNvf4OmmaDot64, lambda acc, a, b, sa, sb: K.nvf4_step_total_batch(BLACKWELL_SM120_NVF4, acc, a, b, sa, sb),
                    domain=every, sample=K._sample_block_scaled(BLACKWELL_SM120_NVF4))
    register_kernel(BlackwellMxf4OmmaDot64, lambda acc, a, b, sa, sb: K.mxf4_step_total_batch(BLACKWELL_SM120_MXF4, acc, a, b, sa, sb),
                    domain=every, sample=K._sample_block_scaled(BLACKWELL_SM120_MXF4))
    register_kernel(F32ToE2m1Sat, lambda x: K.f32_to_e2m1_sat_batch(x, E2M1_NAN_CODE), domain=every)
    register_kernel(E4m3ToF32, K.e4m3_to_f32_batch, domain=every)


_register_kernels()
