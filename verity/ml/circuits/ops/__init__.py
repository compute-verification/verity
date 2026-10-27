"""verity.ml.circuits.ops: exact, hardware-faithful semantics of ML primitives, and their registrations in the Verity IR.

    verity.ml.circuits.ops.prims     IR primitives: the tensor-core steps of the registered instructions (BF16, E4M3, E5M2, NVFP4,
                                     MXFP4) and the casts between their formats; `const`, the constant family
    verity.ml.circuits.ops.scalar    IR primitives: integer, bit, select and comparison operations; FP32 abs / neg / max / min /
                                     cvt.sat; the f32 -> s8 rounding cast and a 32-bit word's high 16 bits
    verity.ml.circuits.ops.fp32      IR primitives: FP32 add / mul / fma / div (RN-even), the FTZ family, FA2 max and -inf guard,
                                     BF16 element-wise add / mul
    verity.ml.circuits.ops.sqrt      IR primitive: PouwF32SqrtRn, sqrt.rn.f32 with the GPU's NaN word (PoUW's and a served
                                     kernel's root)
    verity.ml.circuits.ops.ncp       IR primitive and composite: PouwI32ToF32Rn (cvt.rn.f32.s32) and NcpDequant, ncp-v2's
                                     dequantization (PoUW's linear and a served PoUW linear's row)
    verity.ml.circuits.ops.mufu      IR primitives: the MUFU ex2 / rcp / sqrt / rsqrt / tanh approximations over their measured
                                     sm_89 tables (package data), FA2's inverse sum, div.full's range scaling; numpy kernels for
                                     all but tanh
    verity.ml.circuits.ops.kv_cache  IR primitive: KvCacheAppendE4m3, the BF16 -> E4M3 conversion vLLM applies when it appends a
                                     key or value to an E4M3 KV cache
    verity.ml.circuits.ops.gemm      IR composites: DotBf16 / GemmCoordinate / Gemm parameterised by the step; GEMM_AMPERE,
                                     GEMM_HOPPER; the sm_120 GemmCoordinateE4m3 / GemmCoordinateNvf4 / GemmCoordinateMxf4 and
                                     their GEMM_SM120_* bindings
    verity.ml.circuits.ops.s8        IR composites: the exact s8 matmul family S8Dot / S8Matmul over the s8 and u8 x s8 IMMA steps
                                     (prefix words, tail atoms, operand bounds, noise, limbs and committed layout as statics), and
                                     LimbSplit, the registered split of a runtime s32 word into s8 limbs
    verity.ml.circuits.ops.moe       IR composites: a padded static MoE layer's expert GEMMs, one unit per (token slot, top-k
                                     slot) pair with the expert a value selected inside it (MoeSelectBf16x{E},
                                     MoePairCoordinate(W), MoePairGemm(W)), and their numpy kernels
    verity.ml.circuits.ops.quant     IR composites: torch's ReLU and per-row s8 quantize of a bf16 row, QuantRowS8 (the
                                     row's codes and scale), and for a row wider than one unit RowAbsMax (its amax) and
                                     QuantChunkS8 (a chunk's codes under it)
    verity.ml.circuits.ops.rows      IR composites: row units, one proof block a row (the residual add Bf16AddRow, the greedy
                                     Argmax)
    verity.ml.circuits.ops.argmax    IR composites: the greedy argmax of a bf16 logits row as chunk and combine Definitions of at
                                     most 16 output bits each
    verity.ml.circuits.ops.boolean   word arithmetic as Boolean circuits, among them the next version of each Definition above on
                                     bits; its docstring lists its modules

The step semantics these evaluate through are `verity.ml.circuits.silicon`'s, and their numpy batch forms are
`verity.ml.kernels.tc`'s, registered with `verity.ml.circuits.evaluation` and self-checked against `verity.ml.circuits.silicon`.
"""
from __future__ import annotations

__version__ = "0.1.0"
