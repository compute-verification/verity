"""IR-registered primitives over the tensor-core semantics: the BF16 k16 and E4M3 k32 accumulator steps and the
BF16/E4M3/FP32 casts.

Encodings: bf16 = `Value<16>`, e4m3 = `Value<8>`, f32 = `Value<32>`.  Every evaluator is total on encodings.  The
evaluators are the functions of `verity.ml.tc`; nothing is re-derived here.

Provenance.  `Bf16ToF32_v1`, `F32ToBf16Rn_v1`, `F2fpBf16_v1` are the definitions registered in veritor
`verity.ir/registry/prims.py` (same ids, same functions).  `HopperBF16WgmmaDot16_v1` is veritor's
`verity.ir/registry/hopper.py` definition (it already evaluated through `silicon_total.tc_dot_total`).
`AmpereBF16TcDot16` is registered here as **version 2**: veritor's `_v1` evaluated the finite domain through the same
model but extended it to NaN/Inf encodings by a hand-written rule ("documented, not measured") that disagrees with the
A100/RTX 4090 campaign on the NaN word (`0x7FC00000` vs the measured `0x7FFFFFFF`) and on a saturated group followed by
an opposite-sign infinity (rule R6).  `_v2` is `tc_dot_total(AMPERE_BF16_M16N8K16, ...)`, the measured total function;
equal to `_v1` on every finite input.  A registered id denotes one function, so the id changed.
`F32ToE4m3Sat_v1` and `HopperE4m3QgmmaDot32_v1` are the definitions the vLLM integration registered for its FP8 linear
(same ids, same functions).
"""
from __future__ import annotations

from verity.ir.defs import PrimitiveDefinition, primitive
from verity.ir.types import Array, Value

from .tc import cast
from .tc.silicon import AMPERE_BF16_M16N8K16
from .tc.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from .tc.total_fp8 import HOPPER_E4M3_WGMMA_K32, tc_dot_total_e4m3

BF16, F32, E4M3 = Value(16), Value(32), Value(8)
A16 = Array(16, BF16)
A32_8 = Array(32, E4M3)

__all__ = ["A16", "A32_8", "BF16", "E4M3", "F32", "AmpereBF16TcDot16", "Bf16ToF32", "F2fpBf16", "F32ToBf16Rn", "F32ToE4m3Sat",
           "HopperBF16WgmmaDot16", "HopperE4m3QgmmaDot32", "const"]


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


# ---- constants: one primitive per (width, bits), materialised on demand and decodable by id -------------------------------------------

_CONSTS: dict[tuple[int, int], PrimitiveDefinition] = {}


def const(w: int, bits: int) -> PrimitiveDefinition:
    """`Const<w>[0x..] : () -> Value<w>`: a gate with a fixed correct output (a constant of the program)."""
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
    from verity.ir.codec import register_lazy_family

    register_lazy_family(r"^Const(\d+)\[0x([0-9a-fA-F]+)\]_v1$", lambda m: const(int(m[1]), int(m[2], 16)))


_register_lazy_families()


def _register_kernels() -> None:
    from verity.evaluation import register_kernel

    from . import kernels as K

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


_register_kernels()
