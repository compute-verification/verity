"""difftest adapter — `ov-moe/MoeRouterTopK@v1` (softmax + top-k of one token's router logits) against `_moe_C::topk_softmax`.

Operand: `logits` [E] bf16 (one row of the router GEMM output).  Production: `topk_softmax(topk_weights, topk_ids,
token_expert_indices, gating_output[1, E], renormalize=False, bias=None, is_padding=[False])`, the exact call vLLM's
`fused_topk` makes for OLMoE (norm_topk_prob=False).  Returns (topk_weights f32 [TOPK], topk_ids i32 [TOPK]).

Lane ov-moe recorded 10/24 bit-exact with the correctly rounded stand-in exp (every miss in the weight words).  Since cov-moe
(2026-09-17) the body's exponential is `NvExpf_v1`, the CUDA math-library `expf` sequence; the recorded 24 cases are
bit-exact (weights and ids) — `verity_capture/tests/test_gen_ov_moe.py::test_router_program_is_bit_exact_on_the_recorded_kernel_cases`.
"""
from __future__ import annotations


from verity_vllm.registry.quarantine.ov_moe.common_difftest import kind_for

SEMANTIC_ID = "ov-moe/MoeRouterTopK@v1"
E, TOPK, VPT = 64, 8, 8


def cases(rng, i: int) -> dict:
    k = kind_for(i)
    return {"statics": {"E": E, "TOPK": TOPK, "VPT": VPT},
            "inputs": [{"shape": [E], "dtype": "bfloat16", "kind": k}],
            "edge": k != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    import vllm._custom_ops  # noqa: F401  (registers torch.ops._moe_C)

    (logits,) = inputs
    dev = logits.device
    n = statics["E"]
    w = torch.empty((1, statics["TOPK"]), dtype=torch.float32, device=dev)
    ids = torch.empty((1, statics["TOPK"]), dtype=torch.int32, device=dev)
    tei = torch.empty((1, statics["TOPK"]), dtype=torch.int32, device=dev)
    pad = torch.zeros((1,), dtype=torch.bool, device=dev)
    torch.ops._moe_C.topk_softmax(w, ids, tei, logits.reshape(1, n).contiguous(), False, None, pad)
    torch.cuda.synchronize()
    return [w.reshape(statics["TOPK"]), ids.reshape(statics["TOPK"])]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ov_moe.moe import MoeRouterTopK
    return bind(MoeRouterTopK, E=statics["E"], TOPK=statics["TOPK"], VPT=statics["VPT"])
