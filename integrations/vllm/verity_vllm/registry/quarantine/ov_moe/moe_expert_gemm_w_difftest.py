"""difftest adapter — `ov-moe/MoeExpertGemmW@v1` (the w2 pair GEMM with `MUL_ROUTED_WEIGHT=True`) against `fused_moe_kernel`.

Operands: `x` [K] bf16, `e` I32 (expert index, drawn `zeros` — see `moe_expert_gemm_difftest`), `g` F32 (the pair's
routed weight `topk_weights[t, s]`), `w` [E][N][K] bf16.  The f32 edge kinds of `g` (zeros, ±max, subnormal, alternating)
exercise the `acc * g` epilogue (overflow to Inf, flush behaviour of the final bf16 rounding).
"""
from __future__ import annotations


from verity_vllm.registry.quarantine.ov_moe.common_difftest import kind_for, pick, fused_moe_one_pair

SEMANTIC_ID = "ov-moe/MoeExpertGemmW@v1"
SIZES = ((4, 64, 64), (8, 128, 96), (64, 64, 64), (4, 64, 32))   # (E, N, K)


def cases(rng, i: int) -> dict:
    E, N, K = pick(SIZES, i)
    kx, kg, kw = kind_for(i), kind_for(i // 3 + 2), kind_for(i // 2 + 1)
    return {"statics": {"E": E, "N": N, "K": K},
            "inputs": [{"shape": [K], "dtype": "bfloat16", "kind": kx},
                       {"shape": [], "dtype": "int32", "kind": "zeros"},
                       {"shape": [], "dtype": "float32", "kind": kg},
                       {"shape": [E, N, K], "dtype": "bfloat16", "kind": kw}],
            "edge": kx != "random" or kg != "random" or kw != "random"}


def production(statics: dict, inputs: list) -> list:
    x, e, g, w = inputs
    return [fused_moe_one_pair(x, w, int(e.reshape(-1)[0].item()), g.reshape(1))]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.quarantine.ov_moe.moe import MoeExpertGemmW
    return bind(MoeExpertGemmW, E=statics["E"], N=statics["N"], K=statics["K"])
