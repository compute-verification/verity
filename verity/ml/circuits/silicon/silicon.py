"""Silicon-level semantics of tensor-core multiply-accumulate steps -- the historical entry point.

A tensor-core ``mma`` instruction computes, for each output element,

    D = tc_dot(C, A[0..K-1], B[0..K-1])

where the K products are formed exactly, then summed together with the incoming FP32 accumulator in fixed-size
*groups*: all terms of a group are aligned to the group's maximum exponent by truncating right shifts, added
exactly in a wide integer adder, and the sum is normalised to a fixed internal significand width by truncation
(round toward zero).  The result of one group is the accumulator of the next.  No IEEE rounding happens anywhere;
this is what NVIDIA's silicon does and it is *not* what a sequence of FP32 FMAs would do.

This module is a thin shim: the terms live in :mod:`verity.ml.circuits.silicon.term`
(``Term``, ``bf16_product``, ``e4m3_product``, ``fp32_term``, ``pack_fp32`` and the new E5M2 / E2M1 products and
decoders), the models in :mod:`verity.ml.circuits.silicon.models` (``GroupSum`` -- of which :class:`Pipeline` is the alias --
``group_sum``, ``tc_dot``, ``tc_dot_chain``, the validated instances, and autoproof's ``BlockScaledAlignAdd``),
and the instruction registry in :mod:`verity.ml.circuits.silicon.instructions`.  Every name this module exported before is
re-exported here with the same behaviour; ``tc_dot`` on ``AMPERE_BF16_M16N8K16`` is still the executable
definition ``verity.ml.circuits.targets.target.FIRST`` names.
"""
from __future__ import annotations

from .models import (  # noqa: F401
    ADA_BF16_M16N8K16,
    ADA_E4M3_M16N8K32,
    AMPERE_BF16_M16N8K16,
    HAWKEYE_AMPERE_GROUPSUM_E4M3_V0,
    HOPPER_BF16_M16N8K16,
    HOPPER_E4M3_K32,
    PIPELINES,
    GroupSum,
    Pipeline,
    group_sum,
    pipeline_for,
    tc_dot,
    tc_dot_chain,
)
from .term import (  # noqa: F401
    FP32_MAX_EXPONENT,
    FP32_MIN_NORMAL_EXPONENT,
    FP32_SIGNIFICAND_WIDTH,
    PRODUCT_WIDTH,
    ZERO,
    Term,
    bf16_product,
    e4m3_product,
    fp32_term,
    pack_fp32,
)

__all__ = [
    "ADA_BF16_M16N8K16",
    "ADA_E4M3_M16N8K32",
    "AMPERE_BF16_M16N8K16",
    "HAWKEYE_AMPERE_GROUPSUM_E4M3_V0",
    "HOPPER_BF16_M16N8K16",
    "HOPPER_E4M3_K32",
    "PIPELINES",
    "Pipeline",
    "Term",
    "ZERO",
    "bf16_product",
    "e4m3_product",
    "fp32_term",
    "group_sum",
    "pack_fp32",
    "pipeline_for",
    "tc_dot",
    "tc_dot_chain",
]
