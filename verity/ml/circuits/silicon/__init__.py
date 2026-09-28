"""verity.ml.circuits.silicon: the exact, hardware-faithful semantics of one tensor-core accumulator step, as integer arithmetic on encodings.

    term          operand formats (bf16, e4m3, e5m2, e2m1): exact decoders and exact products as `Term`s; the FP32 word codecs
    models        the step semantics: `GroupSum` (Hawkeye's grouped truncating accumulation; alias `Pipeline`) and
                  `BlockScaledAlignAdd` (autoproof's model of Blackwell's block-scaled FP4 path); the validated instances
    instructions  the registry `INSTRUCTIONS`: PTX instruction class -> model, status (pinned / hypothesis / unprobed), evidence
    silicon       the historical entry point; a shim re-exporting `term` and `models` (`silicon.tc_dot` is the frozen target's step)
    total         the total function on every encoding (NaN, infinities, saturation; measured on A100, RTX 4090, H100)
    total_fp8     the same total rules for the Hopper E4M3 k32 step (wgmma m64nNk32 f32.e4m3.e4m3; measured on H100)
    cast          FP32 <-> BF16 word casts
    fp32          IEEE binary32 add, mul, fma, div and sqrt on words, ties to even, no FTZ (the scalar FP32 pipe)
    relation      the step as a minimal algebraic relation (witness columns, mul/lin/lookup predicates) and its census

The step is `model.step(acc_word, a_words, b_words) -> acc_word'` (`total.tc_dot_total` on every encoding), with `model`
one of `models.MODELS`, e.g. `silicon.AMPERE_BF16_M16N8K16` or `total.HOPPER_BF16_WGMMA_K16`; the FP8 step is
`total_fp8.tc_dot_total_e4m3(total_fp8.HOPPER_E4M3_WGMMA_K32, ...)`.  This package imports nothing but the standard library.
"""
from .cast import bf16_to_f32_word, f32_is_nan, f32_to_bf16_hw_word, f32_to_bf16_word  # noqa: F401
from .errors import InvalidArtifact  # noqa: F401
from .instructions import INSTRUCTIONS, Instruction  # noqa: F401
from .models import MODELS, BlockScaledAlignAdd, GroupSum, Model  # noqa: F401
from .silicon import (  # noqa: F401
    ADA_BF16_M16N8K16,
    AMPERE_BF16_M16N8K16,
    HOPPER_BF16_M16N8K16,
    PIPELINES,
    Pipeline,
    Term,
    ZERO,
    bf16_product,
    fp32_term,
    group_sum,
    pack_fp32,
    pipeline_for,
    tc_dot,
    tc_dot_chain,
)
from .total import (  # noqa: F401
    HOPPER_BF16_WGMMA_K16,
    Inf,
    NaN,
    Value,
    bf16_product_total,
    cvt_rn_bf16_f32,
    fp32_value,
    group_step_total,
    pack_value,
    tc_dot_total,
)
from .total_fp8 import (  # noqa: F401
    HOPPER_E4M3_FAMILY_VERSION,
    HOPPER_E4M3_WGMMA_K32,
    e4m3_class,
    e4m3_product_total,
    tc_dot_total_e4m3,
)
