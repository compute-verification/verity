"""`SiluMul_v2{I}` — vLLM `_C.silu_and_mul` on one row: `SiluMul_v1`'s shape (gate = gu[:I], up = gu[I:]) over the element
`SiluMulBf16_v2` (see `silu_mul_bf16.py`)."""
from __future__ import annotations

from verity.ir.defs import composite
from verity.ir.types import Array

from verity_vllm.program.registry.quarantine.act.silu_mul_bf16 import BF16, SiluMulBf16

SEMANTIC_ID = "act/SiluMul@v2"


@composite("SiluMul", 2, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="quarantine (lane vllm-coverage-defs): SiluMulBf16_v2 per element")
def SiluMul(B, S, gu):
    """`out[i] = SiluMulBf16_v2(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))
