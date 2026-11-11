# verity.definitions.ops

The primitive circuits' Definitions: the exact, hardware-faithful semantics of the operations ML kernels are built from,
registered as Definitions that `verity.definitions` composes into the served kernels. Each module registers one family
and its docstring lists what it registers. The terms and the front doors are [`verity.circuits`](../../circuits/README.md)'s.

| Module | What it registers |
| --- | --- |
| `scalar` | integer, bit, select and comparison operations, the FP32 instructions that are a bit operation or an exact comparison on the encoding, and the f32 -> s8 cast |
| `fp32` | FP32 and BF16 arithmetic rounded to nearest even, the FTZ family of FlashAttention-2's fast-math build, `max` and the `-inf` guard |
| `sqrt` | `sqrt.rn.f32` as the GPU computes it (`PouwF32SqrtRn_v1`) |
| `mufu` | the special-function unit's `ex2`, `rcp`, `sqrt`, `rsqrt` and `tanh` approximations, from measured tables (`tables/`) |
| `prims` | the tensor-core accumulator steps and the BF16, E4M3 and FP32 casts, over `verity.definitions.silicon` |
| `gemm` | the GEMM coordinate, parameterised by its tensor-core step |
| `s8` | the exact s8 matmul family, one chain of exact s8 steps in every arm |
| `quant` | the per-row s8 quantizer of a W8A8 linear's activations |
| `ncp` | `ncp-v2` dequantization (`NcpDequant_v1`) and the int32 -> f32 conversion it starts from |
| `moe` | a padded static MoE layer's routed expert GEMMs, one proof block per (token slot, top-k slot) pair |
| `argmax` | the greedy argmax of a bf16 logits row, in Definitions of at most 16 output bits |
| `rows` | row composites, one proof block a row |
| `kv_cache` | the BF16 -> E4M3 conversion of a KV-cache append |

`ops.boolean` holds their Boolean versions: each module the next version of an `ops` module's word Definitions, the same
function on bits. They are built with `verity.circuits.boolean`'s `forms`, `gather` and `trace` and
`verity.definitions.silicon`'s `fp`, `fp4` and `tc_step`, and `ops.boolean.trace_cache` lets a prover's or a developer's
build reuse traced circuits across processes (a verifier never does).

Beside the modules are each Definition module's binding file and pins (`<module>.circuit-check.py`,
`<module>.circuit-check.json`), the store ids a module's conformance texts cite (`<module>_evidence.json`), and
`metadata.json`, the names and formats of these Definitions' parts, outside every digest (`verity.circuits.metadata`).

## Tests

`tests/` is this package's suite (`uv run tools/verity/check/suites.py verity/definitions/ops`), which tests
`verity.definitions.silicon` too. [`tests/fixtures/`](tests/fixtures/README.md) holds the hardware captures the semantics
are replayed against.
