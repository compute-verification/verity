"""`aten.gelu(x [M,N] bf16, approximate='none')` — the GPT-NeoX MLP activation (`nn.GELU()`, vLLM `get_act_fn("gelu")`):
`out[j] = GeluErfBf16(x[j])`, one element per word (see `gelu_erf_bf16.py`)."""
from __future__ import annotations

from verity.primitives.circuits.defs import composite

from verity_vllm.program.registry.quarantine.ln._common import BF16, Array
from verity_vllm.program.registry.quarantine.ln.gelu_erf_bf16 import GeluErfBf16

SEMANTIC_ID = "ln/GeluErf@v1"


@composite("GeluErf", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="quarantine (lane cov-ln): difftest vs torch.nn.functional.gelu (bf16, sm_89) on the pod")
def GeluErf(B, S, x):
    """`out[j] = GeluErfBf16(x[j])` over one row of N bf16 words."""
    return B.batch(GeluErfBf16, x, axes=(0,))
