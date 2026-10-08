"""verity.ml.circuits.ops.boolean: word arithmetic as Boolean circuits.

    gemm         the BF16 steps, F2FP cast, DotBf16_v3, GemmCoordinate_v3 and Gemm_v3 on bits, and the E4M3, E5M2, NVFP4 and
                 MXFP4 steps and their coordinates
    attention    FA2 attention's FTZ f32 operations, max, -inf guard and inverse sum as Boolean Definitions
    softmax      attention's FTZ f32 operations, strict compare, max and -inf guard as circuits over `forms`
    mufu         the MUFU operations as Boolean Definitions, every table read a plain And/Xor/Not circuit
    elementwise  element-wise FP32, BF16 and I32 arithmetic as Boolean Definitions
    cast         the bf16 -> f32 widening as a Boolean Definition
    scalar       the bf16 comparator, the selects, the bit AND, PTX max / min and the E4M3 and E2M1 casts as Boolean Definitions
                 (and names the IEEE f32 arithmetic of `elementwise` and the bf16 widening of `cast`)

The builders live elsewhere: `forms`, `gather` and `trace` in `verity.ml.circuits.boolean`; `fp`, `fp4` and `tc_step` in
`verity.ml.circuits.silicon`; the universal unit in `verity.ml.private_circuits.universal`; `sha512` and `merkle` in
`verity.ml.commitments.gates`. C-Flock's `verity_flock.gf2`, `.fp`, `.unit` and `.unit_fp4` are `forms`, `fp`, `tc_step`
and `fp4` under their C-Flock names, `verity_flock.tail_pieces` imports `softmax`'s and `scalar`'s, and
`verity_flock.sha512_circuit` imports the gates' `sha512`.
"""
