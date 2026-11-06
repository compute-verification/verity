"""circuit-check's bindings of `verity_numerical.bench.templates`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations


#: the parameters each template is checked at: the workload ones where Flock pins its unit circuit, else small (a template's
#: example is its served size)
TEMPLATE_PARAMS = {
    "gemm-coordinate": [{"K": 64, "semantics": "sm80.mma.m16n8k16.bf16"}, {"K": 64, "semantics": "sm90.wgmma.m64n8k16.bf16"},
                        {"K": 64, "semantics": "sm89.mma.m16n8k32.e4m3"}, {"K": 64, "semantics": "sm90.wgmma.m64n8k32.e4m3"},
                        {"K": 128, "semantics": "sm120.mma.m16n8k64.e2m1.nvf4"}],
    "attention-head": [{"D": 64, "BN": 128, "T": "1:17"}],
    # one block of keys (MASKED_FROM 0), then two (MASKED_FROM 1: dead_free's own binding of the head)
    "attention-head-v18": [{"D": 16, "BN": 32, "T": "16:48"}],
    "rope-head": [{"D": 64}],
    "rmsnorm-fused-cuda": [{"N": 2048, "EPS": 1e-05}],
    "rmsnorm-triton": [{"N": 128, "EPS": 1e-06}],
    "rmsnorm-triton-v5": [{"N": 128, "EPS": 1e-06}],
    "silu-mul": [{"I": 8192}],
    "silu-mul-v2": [{"I": 256}, {"I": 1536}],
    # GLM-5.3's hidden 6144 at both launches of each kernel (fewer than 256 tokens, then 256 or more)
    "rmsnorm-fused-cuda-vec8": [{"N": 6144, "EPS": 1e-05, "BLOCK": 1024}, {"N": 6144, "EPS": 1e-05, "BLOCK": 256}],
    "rmsnorm-cuda-vec8": [{"N": 6144, "EPS": 1e-05, "BLOCK": 768}, {"N": 6144, "EPS": 1e-05, "BLOCK": 256}],
    "gumbel-top-p-token-select": [{"V": 64}],
    "token-select-greedy": [{"V": 64}],
    "embedding-row": [{"V": 64, "H": 16}],
    "moe-expert-coordinate": [{"E": 4, "K": 64}],
    "moe-expert-coordinate-weighted": [{"E": 4, "K": 64}],
    "scaled-mm-fp8-block-coordinate": [{"K": 256, "G": 128}],
    "ingress-prompt": [{"MODEL": "m", "LP": 3, "V": 1000}, {"MODEL": "served-model", "LP": 16, "V": 128256}],
    "token-select-chunk": [{"C": 64}],
    "token-select-combine": [{"V": 200, "C": 64}, {"V": 128256, "C": 64}],
}
