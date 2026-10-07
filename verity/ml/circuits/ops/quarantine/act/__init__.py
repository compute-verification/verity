"""verity.ml.circuits.ops.quarantine.act — re-derived activation Definitions for vLLM d9105ea80
`csrc/libtorch_stable/activation_kernels.cu`, one module per definition (one `SEMANTIC_ID` each), a `*_difftest.py` adapter next to each.

  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2

Both are PROMOTED to `verity.ml.circuits.ops.act` (the `b1-eager-v3` vocabulary's `silu_mul` kind); the modules here
re-export them for the difftest adapter. `SiluMul_v1` (b1) stays the kind of the older vocabularies.
"""
from __future__ import annotations

from verity.ml.circuits.ops.quarantine.act import silu_mul, silu_mul_bf16  # noqa: F401

ALL_DEFINITIONS = [silu_mul.SiluMul]
ALL_PRIMITIVES = [silu_mul_bf16.SiluMulBf16]
