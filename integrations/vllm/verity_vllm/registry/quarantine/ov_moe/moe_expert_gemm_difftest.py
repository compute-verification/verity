"""difftest adapter — `ov-moe/MoeExpertGemm@v1` (the unweighted w13 pair GEMM) against vLLM's `fused_moe_kernel`.

Operands: `x` [K] bf16, `e` I32 (the pair's expert index), `w` [E][N][K] bf16.  The PROTECTED generator draws an int32
operand uniformly over [0, 2^31) for every kind but `zeros`, and an index outside [0, E) is not in the operand domain of
either side (the Verity gather returns the documented 0x7FFF extension, the production kernel would read out of bounds),
so `e` is drawn with `kind="zeros"`: every case selects expert 0 of E >= 4 (the gather must pick slab 0 and no other).
Non-zero expert indices are exercised by Tier A on the captured run (24 pairs per layer over 64 experts).
PROPOSED (difftest.py, PROTECTED): an `index` kind with an explicit `[lo, hi)` range for index-valued operands.
"""
from __future__ import annotations


from verity_vllm.registry.quarantine.ov_moe.common_difftest import kind_for, pick, fused_moe_one_pair

SEMANTIC_ID = "ov-moe/MoeExpertGemm@v1"
SIZES = ((4, 64, 64), (8, 128, 96), (64, 64, 64), (4, 64, 32))   # (E, N, K); K multiple of 16 and of BLOCK_SIZE_K=32


def cases(rng, i: int) -> dict:
    E, N, K = pick(SIZES, i)
    kx, kw = kind_for(i), kind_for(i // 2 + 1)
    return {"statics": {"E": E, "N": N, "K": K},
            "inputs": [{"shape": [K], "dtype": "bfloat16", "kind": kx},
                       {"shape": [], "dtype": "int32", "kind": "zeros"},
                       {"shape": [E, N, K], "dtype": "bfloat16", "kind": kw}],
            "edge": kx != "random" or kw != "random"}


def production(statics: dict, inputs: list) -> list:
    x, e, w = inputs
    return [fused_moe_one_pair(x, w, int(e.reshape(-1)[0].item()))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ov_moe.moe import MoeExpertGemm
    return bind(MoeExpertGemm, E=statics["E"], N=statics["N"], K=statics["K"])
