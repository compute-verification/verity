"""verity.ml.circuits.ops.boolean: word arithmetic as Boolean circuits.

    gemm         the BF16 steps, F2FP cast, DotBf16_v3, GemmCoordinate_v3 and Gemm_v3 on bits, and the E4M3, E5M2, NVFP4 and
                 MXFP4 steps and their coordinates
    attention    FA2 attention's FTZ f32 operations, max, -inf guard and inverse sum as Boolean Definitions
    softmax      attention's FTZ f32 operations, strict compare, max and -inf guard as circuits over `forms`
    mufu         the MUFU operations as Boolean Definitions, every table read a plain And/Xor/Not circuit
    elementwise  element-wise FP32, BF16 and I32 arithmetic as Boolean Definitions
    cast         the bf16 <-> f32 casts as Boolean Definitions
    s8           the exact s8 matmul family on bits: the u8 x s8 step, S8Dot_v2, S8Matmul_v2, LimbSplit_v2 and its four limb steps
    int8         the exact-int8 profile's f32 / f64 steps (rint, the s8 casts, the exact and RN64 products, the u32 cast, the LUT
                 index) as circuits over `forms`, for `verity.ml.definitions.registry.boolean_int8`'s Definitions
    moe          a padded static MoE layer's pair GEMMs on bits: MoeSelectBf16x{E}_v2 (the multiplexer tree over the expert id's
                 low bits and its range bit), MoePairCoordinate(W)_v2 and MoePairGemm(W)_v2
    sqrt         sqrt.rn.f32 as a circuit over `forms` (PoUW's `PouwF32SqrtRn_v2` is traced from it) and its digit-by-digit root
    scalar       the bf16 comparator, the selects, the bit AND, PTX max / min and the E4M3 and E2M1 casts as Boolean Definitions
                 (and names the IEEE f32 arithmetic of `elementwise` and the bf16 widening of `cast`)
    rowwise      CMP-92's row ops split so that each Definition outputs at most 16 bits: a bf16 row's softmax (its max, a
                 16-bit half of its reciprocal sum, one output) and the embedding's coordinate

The builders live elsewhere: `forms`, `gather` and `trace` in `verity.ml.circuits.boolean`; `fp`, `fp4` and `tc_step` in
`verity.ml.circuits.silicon`; the universal unit in `verity.ml.private_circuits.universal`; `sha512` and `merkle` in
`verity.ml.commitments.gates`. C-Flock's `verity_flock.gf2`, `.fp`, `.unit` and `.unit_fp4` are `forms`, `fp`, `tc_step`
and `fp4` under their C-Flock names, `verity_flock.tail_pieces` imports `softmax`'s and `scalar`'s, and
`verity_flock.sha512_circuit` imports the gates' `sha512`.
"""
