# verity.circuits.ops

These are the primitive circuits: the operations ML kernels are built from, such as a floating-point add, a
tensor-core accumulator step, a cast or a GEMM coordinate. Each is a Definition that says bit for bit what a GPU
computes, its rounding and its NaN words included. The served kernels' circuits, in the vLLM adapter's
`verity_vllm.definitions`, are composed from them.

## Try one

It needs nothing but the workspace.

~~~python
from verity.circuits.evaluation import evaluate
from verity.circuits.ops.fp32 import F32Add, F32AddV2

evaluate(F32AddV2, 0x3F800000, 0x40000000)    # 0x40400000: 1.0 + 2.0 = 3.0 in binary32
evaluate(F32Add, 0x7F800000, 0xFF800000)      # 0xFFC00000: inf + -inf, the host's NaN word
evaluate(F32AddV2, 0x7F800000, 0xFF800000)    # 0x7FFFFFFF: the same sum, the GPU's NaN word
~~~

Each call returns a tuple of the Definition's output words. `F32Add_v1` and `F32Add_v2` agree on every word but a
NaN result: a registered id denotes one function, so the GPU's NaN came as a new version.

## What is here

The families: integer, bit and comparison operations; FP32 and BF16 arithmetic, with the FTZ family of
FlashAttention-2's fast-math build; the special-function unit's approximations, from measured tables; the tensor-core
accumulator steps and the BF16, E4M3 and FP32 casts; the GEMM coordinate; exact s8 matmul and the s8 quantizer of a
W8A8 linear's activations; `ncp-v2` dequantization; a padded MoE layer's routed expert GEMMs; greedy argmax; row
composites; and a KV-cache append's BF16 to E4M3 conversion. Each module registers one family, and its docstring lists
what it registers.

`ops.boolean` holds their Boolean versions, the same functions on bits. The terms are
[`verity.circuits`](../README.md)'s, and the tests, with the hardware captures the semantics are replayed against
([`fixtures/`](../../../tests/verity/circuits/ops/fixtures/README.md)), are in `tests/verity/circuits/ops/`.
