"""Shared pieces of the ov-moe difftest adapters (torch-free at module level; torch/vLLM imported inside functions only).

The production operation of the pair GEMMs is the pinned `fused_moe_kernel` launch of vLLM d9105ea80 exactly as the
OLMoE capture records it (BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, SPLIT_K=1, num_warps=4,
num_stages=3, bf16 compute, no quantisation, no bias, SWAP_AB=False, USE_TD=False), invoked through vLLM's own
`invoke_fused_moe_triton_kernel` on a ONE-pair problem: M=1 token, top_k=1, one expert block whose first slot is the pair
and whose 63 other slots are padding (`sorted_token_ids == num_valid_tokens`), `expert_ids = [e]`.  Every output word of a
pair is computed by the kernel independently of the other pairs of its block, so the one-pair launch IS the production
arithmetic of a pair of the real M*TOPK problem.
"""
from __future__ import annotations

SEMANTIC_ID = "ov-moe/CommonDifftest@v1"   # helper module (no definition); the lint wants an id on every adapter
KINDS = ("random", "zeros", "max", "subnormal", "alternating", "random", "random", "random")

#: the recorded launch pins (`gen_ov_moe_patterns.FUSED_MOE_PINS` / `FUSED_MOE_NUM_WARPS` / `FUSED_MOE_NUM_STAGES`)
FUSED_MOE_CONFIG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}


def kind_for(i: int) -> str:
    return KINDS[i % len(KINDS)]


def pick(seq, i: int):
    return seq[i % len(seq)]


def as_words16(t):
    """A bf16 tensor reinterpreted as float16 — the SAME 16-bit words.  The PROTECTED `difftest.produce` stores outputs via
    `.view(dtype).numpy()`, which numpy cannot do for bfloat16; `check` compares words, so the label is immaterial.
    PROPOSED (difftest.py): store 16-bit outputs through `.view(torch.int16)` and drop this workaround."""
    import torch
    return t.contiguous().view(torch.float16)


def fused_moe_one_pair(x, w, e: int, g=None):
    """`fused_moe_kernel` on one pair: x [K] bf16, w [E][N][K] bf16, expert e, routed weight g (f32 [1] tensor, or None
    for the unweighted w13 launch).  Returns the pair's output row [N] bf16 as the production kernel wrote it."""
    import torch
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel

    E, N, K = w.shape
    dev = x.device
    A = x.reshape(1, K).contiguous()
    C = torch.zeros((1, 1, N), dtype=torch.bfloat16, device=dev)
    bm = FUSED_MOE_CONFIG["BLOCK_SIZE_M"]
    sorted_ids = torch.full((bm,), 1, dtype=torch.int32, device=dev)   # num_valid_tokens = M*top_k = 1 is the padding sentinel
    sorted_ids[0] = 0
    expert_ids = torch.tensor([int(e)], dtype=torch.int32, device=dev)
    post_pad = torch.tensor([bm], dtype=torch.int32, device=dev)
    tw = None if g is None else g.reshape(1, 1).to(torch.float32).contiguous()
    invoke_fused_moe_triton_kernel(A, w.contiguous(), C, None, None, tw, sorted_ids, expert_ids, post_pad,
                                   g is not None, 1, dict(FUSED_MOE_CONFIG), tl.bfloat16, False, False, False, False, False, None, None)
    torch.cuda.synchronize()
    return as_words16(C.reshape(N))
