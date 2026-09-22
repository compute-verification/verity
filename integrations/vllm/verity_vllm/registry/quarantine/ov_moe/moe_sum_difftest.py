"""difftest adapter — `ov-moe/MoeSum@v1` (the routed sum over a token's TOPK weighted expert rows) against `_moe_C::moe_sum`.

Operand: `y` [TOPK][H] bf16 (one token's `cache3[t]`).  Production: `torch.ops._moe_C.moe_sum(y.view(1, TOPK, H), out[1, H])`
exactly as vLLM's `fused_experts_impl` calls it; only the specialization the OLMoE run uses (TOPK=8, H=2048) is admitted —
the production kernel's reduction order is a property of the (TOPK, H) launch, not of the mathematics.
"""
from __future__ import annotations


from verity_vllm.registry.quarantine.ov_moe.common_difftest import as_words16, kind_for

SEMANTIC_ID = "ov-moe/MoeSum@v1"
TOPK, H = 8, 2048


def cases(rng, i: int) -> dict:
    k = kind_for(i)
    return {"statics": {"TOPK": TOPK, "H": H},
            "inputs": [{"shape": [TOPK, H], "dtype": "bfloat16", "kind": k}],
            "edge": k != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._moe_C)

    (y,) = inputs
    out = torch.empty((1, statics["H"]), dtype=torch.bfloat16, device=y.device)
    torch.ops._moe_C.moe_sum(y.reshape(1, statics["TOPK"], statics["H"]).contiguous(), out, None, None)
    torch.cuda.synchronize()
    return [as_words16(out.reshape(statics["H"]))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ov_moe.moe import MoeSum
    return bind(MoeSum, TOPK=statics["TOPK"], H=statics["H"])
