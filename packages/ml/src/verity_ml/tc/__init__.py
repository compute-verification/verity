"""verity_ml.tc: the exact, hardware-faithful semantics of one tensor-core accumulator step, as integer arithmetic on encodings.

    silicon   the finite-domain model (Hawkeye's grouped truncating accumulation, generalised over groups/width/floor)
    total     the total function on every encoding (NaN, infinities, saturation; measured on A100, RTX 4090, H100)
    cast      FP32 <-> BF16 word casts
    relation  the step as a minimal algebraic relation (witness columns, mul/lin/lookup predicates) and its census

The step is `total.tc_dot_total(pipeline, acc_word, a_words, b_words) -> acc_word'`, with `pipeline` one of
`silicon.AMPERE_BF16_M16N8K16` or `total.HOPPER_BF16_WGMMA_K16`.  This package imports nothing but the standard library.
"""
from .cast import bf16_to_f32_word, f32_is_nan, f32_to_bf16_hw_word, f32_to_bf16_word  # noqa: F401
from .errors import InvalidArtifact  # noqa: F401
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
