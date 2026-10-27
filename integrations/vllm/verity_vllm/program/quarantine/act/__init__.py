"""verity_vllm.program.quarantine.act — re-derived activation Definitions for vLLM d9105ea80
`csrc/libtorch_stable/activation_kernels.cu`, one module per definition (one `SEMANTIC_ID` each), a `*_difftest.py` adapter next to each.

  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2

Both are PROMOTED to `verity.ml.definitions.registry.act` (the `b1-eager-v3` vocabulary's `silu_mul` kind); `silu_mul` here
re-exports `SiluMul_v2` for the difftest adapter. `SiluMul_v1` (b1) stays the kind of the older vocabularies.
"""
from __future__ import annotations

from verity_vllm.program.quarantine.act import silu_mul

ALL_DEFINITIONS = [silu_mul.SiluMul]
