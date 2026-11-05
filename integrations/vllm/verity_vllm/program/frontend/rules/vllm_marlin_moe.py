"""[DSV4] vLLM's MXFP4 Marlin MoE as the H200 serves DeepSeek-V4-Pro (`fused_marlin_moe`, vLLM d9105ea80
`model_executor/layers/fused_moe/experts/marlin_moe.py`), over the leaves `marlin_moe_leaf.mxfp4_marlin_moe_leaf` exports in place
of its `moe_align_block_size` and its two `ops.moe_wna16_marlin_gemm` launches (`vllm_iface`):

    marlin_moe_align(topk_ids [M, TOPK] i32, key [M, TOPK] i32, E, BM) -> (blk [M * TOPK] i32, par [1] i32)
        `MoeAlignRanks{E,P=M*TOPK,BM}` over the router ids, each pair's key (the TP rank Program's `schedule` input,
        `CollectiveBus.schedule_key`, so the application is tagged `schedule` for `tp_links.json`) and the pair indices as constants,
        then `MoeAlignBlocks{P,BM}` over the ids and its ranks
    marlin_moe_mxfp4_gemm(x, w [E, N, K/2] u8, w_scale [E, N, K/32] u8, topk_ids, blk, par, topk_weights, BM, top_k, mul_topk_weights)
        -> [M * TOPK, N] bf16: per pair p, `MarlinMoeExpertGemm{E,K,N,TK,TN,GRID,DOT}` of x[p // top_k], ids[p], blk[p] and par (the
        w13 launch), or `MarlinMoeExpertGemmW` of x[p], ids[p], topk_weights[p], blk[p] and par (the w2 launch, top_k 1)

The launch's thread config (TK, TN) and its GRID, 132 SMs times the blocks per SM, are ops.cu's choice from the registers the driver
gave the JIT'd kernel, which no host code sees: `MARLIN_SERVED` is the config the H200 capture saw each launch run at each decode
batch it swept, and a batch it did not sweep takes the config ops.cu's rule gives under every per-config limit consistent with the
sweep at its moe block (`marlin_moe_rows.served_config`), and refuses where those differ.  The moe block and the config are read at
the launch's token count (`launch_tokens_of`).  A launch that also routes other requests' tokens is refused: `moe_align_block_size`
sorts every pair of the launch, so this request's blocks and the launch's block count, which place its pairs in Marlin's split-K
partition, read their routing, and no input of this request's Program holds it.  A role the vocabulary lacks refuses by name.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import call_args, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile, launch_tokens_of
from verity_vllm.program.frontend.rules.vllm_dsa import _missing, _value
from verity_vllm.program.frontend.rules.vllm_iface import iface
from verity.ml.definitions.numerics import marlin_moe_rows as MR
from verity.ml.definitions.registry import marlin_moe as MM
from verity.ml.definitions.registry.prims import F32 as F32_WORD, I32 as I32_WORD

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

BF16, F32, I32, U8 = torch.bfloat16, torch.float32, torch.int32, torch.uint8

#: the roles the vocabulary may lack, and which kernel each states
ROLE_KERNELS = {"marlin_moe_rank": "moe_align_block_size's ranks over a key per pair (MoeAlignRanks_v1)",
                "marlin_moe_align": "moe_align_block_size's schedule over those ranks (MoeAlignBlocks_v1)",
                "const": "a constant word",
                "marlin_moe_gemm": "Marlin's w13 launch on one pair (MarlinMoeExpertGemm_v1)",
                "marlin_moe_gemm_w": "Marlin's w2 launch on one pair, the routed weight multiplied in (MarlinMoeExpertGemmW_v1)"}

#: the (TK, TN, blocks per SM) each launch of DeepSeek-V4-Pro's TP8 rank ran at each decode batch of M tokens the H200 capture swept
#: (132 SMs, driver 580.173.02), by the launch's (E, routed TOPK, K, N, launch top_k): w13 (384, 6, 7168, 768, 6), w2 (384, 6, 384,
#: 7168, 1); a GO run of tests/properties/dsv4_marlin_capture_gpu.py (dsv4-marlin-capture/v3, its `sweep`)
MARLIN_SERVED = {(384, 6, 7168, 768, 6): {1: (128, 64, 2), 2: (128, 64, 4), 3: (128, 64, 4), **dict.fromkeys((*range(4, 18), 460), (64, 128, 4)),
                                          461: (64, 128, 3), 921: (64, 128, 3), 922: (64, 128, 2), 1024: (64, 128, 2)},
                 (384, 6, 384, 7168, 1): {**dict.fromkeys((*range(1, 18), 460), (64, 128, 4)),
                                          461: (64, 128, 3), 921: (64, 128, 3), 922: (64, 128, 2), 1024: (64, 128, 2)}}
#: the target the capture ran on: (compute capability, SMs)
MARLIN_TARGET = ((9, 0), 132)


def served_config(E: int, TOPK: int, K: int, N: int, top_k: int, M: int, target=None) -> tuple[int, int, int] | str:
    """The (TK, TN, GRID) a launch of N columns over K, top_k, runs at for M tokens routed top TOPK of E on `target` (default the
    active target profile), or a refusal."""
    target = target or active_target_profile()
    if (tuple(target.compute_capability), target.num_sms) != MARLIN_TARGET:
        return (f"numerics_unregistered: the Marlin configs are the capture's target's (cc {MARLIN_TARGET[0]}, {MARLIN_TARGET[1]} SMs); "
                f"the target is cc {tuple(target.compute_capability)}, {target.num_sms} SMs")
    served = MARLIN_SERVED.get((E, TOPK, K, N, top_k))
    if served is None:
        return f"numerics_unregistered: no Marlin capture of a launch at E={E}, TOPK={TOPK}, K={K}, N={N}, top_k={top_k} (MARLIN_SERVED)"
    cfg = MR.served_config(served, M, N, TOPK, top_k, E, MARLIN_TARGET[1])
    if cfg is None:
        return (f"numerics_unregistered: the capture's sweep does not determine the Marlin config at {M} tokens (moe block "
                f"{MR.moe_block(M, TOPK, E)})")
    TK, TN, blocks = cfg
    return TK, TN, MARLIN_TARGET[1] * blocks


class MarlinMoeAlignRule(OpRule):
    """`vllm_iface.marlin_moe_align`: `MoeAlignRanks{E,P,BM}` over the P = M * TOPK router ids, keys and pair indices, then
    `MoeAlignBlocks{P,BM}`; the node is (blk, par)."""
    modes = frozenset({"profile"})
    info = RuleInfo("MarlinMoeAlign[vllm-marlin-moe]", 2, "op-binding", applies_to="vllm_iface.marlin_moe_align",
                    preconditions=("topk_ids and key are i32 values of one shape [M, TOPK]",
                                   "num_experts and moe_block_size are static positive ints",
                                   "the launch routes this request's M tokens only"),
                    evidence_domain="structural; numerics: MoeAlignRanks_v1, MoeAlignBlocks_v1 (registry.marlin_moe)")

    def targets(self):
        return {iface.marlin_moe_align.default}

    def applies(self, node, ctx):
        if (r := _missing(ctx, ["marlin_moe_rank", "marlin_moe_align", "const"], ROLE_KERNELS)):
            return r
        a = call_args(node)
        if (r := _value(ctx, a["topk_ids"], I32, (None, None), "topk_ids")):
            return r
        if (r := _value(ctx, a["key"], I32, shape(ctx, a["topk_ids"]), "key")):
            return r
        E, BM = a["num_experts"], a["moe_block_size"]
        if not (isinstance(E, int) and isinstance(BM, int) and E > 0 and BM > 0):
            return f"num_experts={E!r} moe_block_size={BM!r} are not static positive ints"
        M = shape(ctx, a["topk_ids"])[0]
        if (L := launch_tokens_of(node, M)) != M:
            return (f"numerics_unregistered: the launch routes {L} tokens, {L - M} of them other requests': moe_align_block_size sorts "
                    f"every pair of the launch, so this request's blocks and the launch's block count read their routing")
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        M, TOPK = shape(ctx, a["topk_ids"])
        S = {"E": a["num_experts"], "P": M * TOPK, "BM": a["moe_block_size"]}
        ranks, blocks = ctx.kind("marlin_moe_rank", node, **S), ctx.kind("marlin_moe_align", node, P=S["P"], BM=S["BM"])
        ids, key = (ctx.coll(a[k]).reshape(Array(S["P"], I32_WORD)) for k in ("topk_ids", "key"))
        h = ctx.hint_name(node)
        ix = array_of([ctx.call(ctx.kind("const", node, W=32, BITS=p), node=node, rule=self, name=f"{h}/i32[{p}]") for p in range(S["P"])])
        rank, start = ctx.call(ranks, ids, key, ix, node=node, rule=self, name=f"{h}/rank")
        out = ctx.call(blocks, ids, rank, start, node=node, rule=self, name=h)
        ctx.define(node, [out[0], array_of([out[1]])])
        ctx.record(self, node, S, emitted=f"{ranks.id} + {blocks.id}", schedule="marlin_moe_align")


class MarlinMoeGemmRule(OpRule):
    """`vllm_iface.marlin_moe_mxfp4_gemm`: `MarlinMoeExpertGemm{E,K,N,TK,TN,GRID,DOT}` (w13) or `MarlinMoeExpertGemmW` (w2) per pair,
    at the config `served_config` gives; the node is the [M * TOPK, N] rows."""
    modes = frozenset({"profile"})
    info = RuleInfo("MarlinMoeGemm[vllm-marlin-moe]", 1, "op-binding", applies_to="vllm_iface.marlin_moe_mxfp4_gemm",
                    preconditions=("topk_ids an i32 value [M, TOPK]; blk an i32 value [M * TOPK] and par [1] (marlin_moe_align's)",
                                   "w13: top_k == TOPK, mul_topk_weights False, x a bf16 value [M, K]; w2: top_k 1, mul_topk_weights True, "
                                   "x a bf16 value [M * TOPK, K], topk_weights an f32 value [M, TOPK]",
                                   "w a u8 value [E, N, K / 2] and w_scale a u8 value [E, N, K / 32] (the e8m0 bytes Mxfp4MoEMethod holds)",
                                   "the capture gives the launch's config at its tokens (MARLIN_SERVED) and moe_block_size is "
                                   "fused_marlin_moe's there"),
                    evidence_domain="structural; numerics: MarlinMoeExpertGemm_v1 / MarlinMoeExpertGemmW_v1 (registry.marlin_moe; the "
                                    "capture tests/properties/dsv4_marlin_capture_gpu.py)")

    def targets(self):
        return {iface.marlin_moe_mxfp4_gemm.default}

    @staticmethod
    def _role(a) -> str:
        return "marlin_moe_gemm_w" if a["mul_topk_weights"] else "marlin_moe_gemm"

    def applies(self, node, ctx):
        a = call_args(node)
        if (r := _missing(ctx, [self._role(a)], ROLE_KERNELS)):
            return r
        if (r := _value(ctx, a["topk_ids"], I32, (None, None), "topk_ids")):
            return r
        (M, TOPK), top_k, mul, BM = shape(ctx, a["topk_ids"]), a["top_k"], a["mul_topk_weights"], a["moe_block_size"]
        if (top_k, mul) not in ((TOPK, False), (1, True)):
            return f"numerics_unregistered: top_k={top_k!r}, mul_topk_weights={mul!r} is neither the w13 launch (top_k {TOPK}) nor w2 (top_k 1)"
        P = M * TOPK
        if (r := _value(ctx, a["x"], BF16, (P // top_k, None), "x")):
            return r
        K = shape(ctx, a["x"])[1]
        if (r := _value(ctx, a["w"], U8, (None, None, K // 2), "w")):
            return r
        E, N, _ = shape(ctx, a["w"])
        if (r := _value(ctx, a["w_scale"], U8, (E, N, K // MM.GROUP), "w_scale") or _value(ctx, a["blk"], I32, (P,), "blk")
                or _value(ctx, a["par"], I32, (1,), "par") or (mul and _value(ctx, a["topk_weights"], F32, (M, TOPK), "topk_weights"))):
            return r
        L = launch_tokens_of(node, M)
        if BM != MR.moe_block(L, TOPK, E):
            return f"moe_block_size {BM!r} is not fused_marlin_moe's at {L} tokens over {E} experts ({MR.moe_block(L, TOPK, E)})"
        cfg = served_config(E, TOPK, K, N, top_k, L)
        if isinstance(cfg, str):
            return cfg
        try:
            MM.check_gemm(SimpleNamespace(K=K, N=N, TK=cfg[0], TN=cfg[1]))
        except ValueError as e:
            return f"numerics_unregistered: {e}"
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        M, TOPK = shape(ctx, a["topk_ids"])
        P, top_k, mul = M * TOPK, a["top_k"], a["mul_topk_weights"]
        K = shape(ctx, a["x"])[1]
        E, N, _ = shape(ctx, a["w"])
        TK, TN, GRID = served_config(E, TOPK, K, N, top_k, launch_tokens_of(node, M))
        S = {"E": E, "K": K, "N": N, "TK": TK, "TN": TN, "GRID": GRID}
        fn = ctx.kind(self._role(a), node, **S, DOT=MM.DOT)
        x, _ = rows_of(ctx.coll(a["x"]))
        ids = ctx.coll(a["topk_ids"]).reshape(Array(P, I32_WORD))
        g = ctx.coll(a["topk_weights"]).reshape(Array(P, F32_WORD)) if mul else None
        blk, par, wq, ws = ctx.coll(a["blk"]), ctx.coll(a["par"])[0], ctx.coll(a["w"]), ctx.coll(a["w_scale"])
        h = ctx.hint_name(node)
        outs = [ctx.call(fn, x[p // top_k], ids[p], *((g[p],) if mul else ()), blk[p], par, wq, ws, node=node, rule=self, name=f"{h}[{p}]")
                for p in range(P)]
        ctx.define(node, from_rows(outs, (P,)))
        ctx.bind_weight_role(a["w"], self, "moe.expert_weight_e2m1x2", E=E, K=K, N=N)
        ctx.bind_weight_role(a["w_scale"], self, "moe.expert_scale_e8m0", E=E, K=K, N=N)
        ctx.record(self, node, {**S, "DOT": MM.DOT.name, "M": M, "TOPK": TOPK, "top_k": top_k, "pairs": P, "BM": a["moe_block_size"]},
                   emitted=f"{fn.id} x{P}")


VLLM_MARLIN_MOE_RULES = [MarlinMoeAlignRule(), MarlinMoeGemmRule()]

__all__ = ["MARLIN_SERVED", "MarlinMoeAlignRule", "MarlinMoeGemmRule", "VLLM_MARLIN_MOE_RULES", "served_config"]
