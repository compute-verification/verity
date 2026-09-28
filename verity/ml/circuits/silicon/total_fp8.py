"""Total semantics of one Hopper FP8 tensor-core step: ``veritor.tensor-core.hopper_e4m3_wgmma_k32@1`` (lane QZ).

The instruction is one accumulator coordinate of ``wgmma.mma_async.sync.aligned.m64n{N}k32.f32.e4m3.e4m3``
(SASS ``QGMMA.64x{N}x32.F32.E4M3.E4M3``), the step vLLM's CUTLASS ``cutlass_scaled_mm`` (sm90 FP8, the
``--quantization fp8`` linear of port ``b1-fp8``) executes on an H100.  :mod:`verity.ml.circuits.silicon.silicon` carries
Hawkeye's *finite* H100 FP8 pipeline (:data:`~verity.ml.circuits.silicon.silicon.HOPPER_E4M3_K32`: one group of 32 exact
E4M3 products plus the accumulator, a 14-bit truncating adder, exponent floor ``-139``), validated by Hawkeye
with a zero accumulator only.  This module gives the **total function on every encoding** of the gate's
signature, by the same rules as :mod:`verity.ml.circuits.silicon.total` (L6's R1-R8 for BF16) restricted to what
E4M3 can encode:

* an E4M3 byte is a NaN (``0x7F``, ``0xFF``: exponent and mantissa all ones -- OCP E4M3 has *no* infinities),
  a zero (``0x00``, ``0x80``), or a finite value (subnormals ``0x01..0x07`` included);
* the **product** of two bytes is NaN if either is a NaN, else the exact finite product of
  :func:`~verity.ml.circuits.silicon.silicon.e4m3_product` (a product never overflows: ``448 * 448 < 2^18``);
* the **accumulator** word decodes as in the BF16 module (NaN, ``±inf``, or a finite term);
* one **group step** is :func:`~verity.ml.circuits.silicon.total.group_step_total`: NaN if any term is NaN; the
  accumulator's infinity if it is one (products are finite, so the two-sign NaN rule can only fire through the
  accumulator); otherwise the finite :func:`~verity.ml.circuits.silicon.silicon.group_sum`, which saturates to an infinity of
  its sign above the FP32 range;
* the **result word** is ``0x7FFFFFFF`` for NaN, ``±0x7F800000`` for the infinities, :func:`~verity.ml.circuits.silicon.silicon.pack_fp32`
  otherwise -- zero is always ``+0``.

On finite operands this is :func:`verity.ml.circuits.silicon.silicon.tc_dot` on the same pipeline.  The special-value rules and
the *pipeline parameters* (groups, adder width, exponent floor -- in particular the behaviour with a NON-ZERO
accumulator, which Hawkeye did not validate and which every k32 step after the first of a K=1536..8960 chain
exercises) are what the campaign under ``gpu/tc-fp8/`` measured on the device (record:
``docs/data/tc-fp8-2026-09-08/``).  The registry primitive that wraps it
(``verity.ml.circuits.ops.prims.HopperE4m3QgmmaDot32``) carries the campaign's outcome in its conformance status.

Promoted verbatim from ``veritor`` ``src/veritor/core/silicon_total_fp8.py`` (commit 0903f60d6, release
``r18-int-3``); only the import paths changed.

E5M2 (:func:`tc_dot_total_e5m2`) differs only in its operand classes: it is IEEE-like, so exponent field 31 is an
infinity (fraction 0) or a NaN, and the product rules are BF16's (:func:`~verity.ml.circuits.silicon.total.bf16_product_total`):
NaN if either operand is a NaN or an infinity meets a zero, else an infinity of the sign XOR if either is one, else
the exact finite product.  An infinite product enters :func:`~verity.ml.circuits.silicon.total.group_step_total` like an infinite
accumulator, so infinities of both signs in one group are a NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import InvalidArtifact
from .silicon import HOPPER_E4M3_K32, Pipeline, e4m3_product
from .term import e5m2_product
from .total import NAN, Inf, Value, fp32_value, group_step_total, pack_value

__all__ = [
    "HOPPER_E4M3_WGMMA_K32",
    "HOPPER_E4M3_FAMILY_VERSION",
    "e4m3_class",
    "e4m3_product_total",
    "e5m2_class",
    "e5m2_product_total",
    "tc_dot_total_e4m3",
    "tc_dot_total_e5m2",
]


def e4m3_class(byte: int) -> str:
    """``"nan"``, ``"zero"`` or ``"finite"`` (subnormals included); E4M3 has no infinities."""

    if not 0 <= byte < 1 << 8:
        raise InvalidArtifact("E4M3 operand outside 8 bits")
    if (byte & 0x7F) == 0x7F:
        return "nan"
    return "zero" if (byte & 0x7F) == 0 else "finite"


def e4m3_product_total(a: int, b: int) -> Value:
    """The product of two E4M3 bytes as a :data:`~verity.ml.circuits.silicon.total.Value`."""

    if e4m3_class(a) == "nan" or e4m3_class(b) == "nan":
        return NAN
    return e4m3_product(a, b)


def tc_dot_total_e4m3(pipeline: Pipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """``D = acc + sum_k a[k] * b[k]`` on every encoding, as ``pipeline``'s FP8 silicon computes it."""

    if pipeline.dtype != "e4m3":
        raise InvalidArtifact("the FP8 total semantics are defined for E4M3 pipelines")
    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise InvalidArtifact(f"{pipeline.name} takes {pipeline.k} operand pairs")
    acc: Value = fp32_value(acc_bits)
    start = 0
    for size in pipeline.groups:
        products = [e4m3_product_total(a[i], b[i]) for i in range(start, start + size)]
        acc = group_step_total(acc, products, width=pipeline.width, zero_exponent=pipeline.zero_exponent)
        start += size
    return pack_value(acc)


def e5m2_class(byte: int) -> str:
    """``"nan"``, ``"inf"``, ``"zero"`` or ``"finite"`` (subnormals included)."""

    if not 0 <= byte < 1 << 8:
        raise InvalidArtifact("E5M2 operand outside 8 bits")
    exponent, mantissa = (byte >> 2) & 0x1F, byte & 0x3
    if exponent == 0x1F:
        return "nan" if mantissa else "inf"
    return "zero" if exponent == 0 and mantissa == 0 else "finite"


def e5m2_product_total(a: int, b: int) -> Value:
    """The product of two E5M2 bytes as a :data:`~verity.ml.circuits.silicon.total.Value`."""

    ca, cb = e5m2_class(a), e5m2_class(b)
    if ca == "nan" or cb == "nan":
        return NAN
    if ca == "inf" or cb == "inf":
        if ca == "zero" or cb == "zero":
            return NAN
        return Inf(((a ^ b) >> 7) & 1 == 1)
    return e5m2_product(a, b)


def tc_dot_total_e5m2(pipeline: Pipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """``D = acc + sum_k a[k] * b[k]`` on every encoding, as ``pipeline``'s E5M2 silicon computes it."""

    if pipeline.dtype != "e5m2":
        raise InvalidArtifact("tc_dot_total_e5m2 takes an E5M2 pipeline")
    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise InvalidArtifact(f"{pipeline.name} takes {pipeline.k} operand pairs")
    acc: Value = fp32_value(acc_bits)
    start = 0
    for size in pipeline.groups:
        products = [e5m2_product_total(a[i], b[i]) for i in range(start, start + size)]
        acc = group_step_total(acc, products, width=pipeline.width, zero_exponent=pipeline.zero_exponent)
        start += size
    return pack_value(acc)


#: One accumulator coordinate of ``wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3`` (SASS
#: ``QGMMA.64x64x32.F32.E4M3.E4M3``), PINNED on an H100 80GB HBM3 (driver 580.126.09) by lane QZ's campaign
#: ``gpu/tc-fp8/campaign.py`` + ``conformance.py`` on 2026-09-08 (record ``docs/data/tc-fp8-2026-09-08/``):
#: 16 case families (L6's, with E4M3 operand classes and FP32 accumulator classes incl. subnormal / huge / ±inf /
#: NaN accumulators), 35,289 tiles x 3 repetitions (every tile bit-identical across repetitions), 144,543,744 output
#: words compared against this total function -- **0 mismatches**.  The fit grid over groups x width x floor
#: (3,060 candidates, 6,000 records + the ``accid`` family) leaves ``groups = (32,)`` and ``width = 14`` unique;
#: the floor is determined only to ``-152 <= floor <= -126`` -- E4M3 products are never below ``2^-18`` and the FP32
#: accumulator's decoded exponent is never below ``-126``, so every floor at or below ``-126`` computes the same
#: word on every reachable input.  Hawkeye's ``-139`` is kept.  The 32 products of one step and the accumulator are
#: summed in ONE group (no intra-step partial rounding); the accumulator is truncated to the 14-bit adder like a
#: product (``accid``: 261,900 of 262,144 random finite accumulators do NOT pass through a zero-product step
#: unchanged -- the "FP8 accumulation precision loss" DeepSeek-V3 reports, here word-exact).
HOPPER_E4M3_WGMMA_K32 = Pipeline(
    name="hopper_e4m3_wgmma_k32",
    arch="sm_90",
    dtype="e4m3",
    operand_bits=8,
    groups=HOPPER_E4M3_K32.groups,
    width=HOPPER_E4M3_K32.width,
    zero_exponent=HOPPER_E4M3_K32.zero_exponent,
    validated="pinned 2026-09-08 (lane QZ, H100 80GB HBM3, driver 580.126.09, triton 3.7.1 probe "
              "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 / QGMMA.64x64x32.F32.E4M3.E4M3): 16 families, "
              "35,289 tiles x3, 144,543,744 words, 0 mismatches; groups=(32,) and width=14 unique in the fit grid, "
              "floor bounded to [-152, -126] (indistinguishable below -126 for E4M3 operands; -139 kept) -- "
              "docs/data/tc-fp8-2026-09-08/",
)

HOPPER_E4M3_FAMILY_VERSION = "1"
