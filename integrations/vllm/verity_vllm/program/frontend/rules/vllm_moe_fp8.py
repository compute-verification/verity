"""The block-scaled FP8 launches of vLLM's fused MoE: `fused_moe_kernel` with `use_fp8_w8a8` and `block_shape` [G, G], as
`TritonExperts` (`TargetProfile.moe_backend = "triton"`) runs a pre-quantised checkpoint's experts.  Generic in the block's shapes
(E, K, N, G, tokens); nothing about a model.

    verity_cba::triton_launch(fused_moe_kernel; a e4m3 [rows, K], b e4m3 [E, N, K], c, a_scale f32 [rows, K/G],
                              b_scale f32 [E, cdiv(N, G), K/G], w, sorted_ids, expert_ids, num_post_padded, ...)  [x2]
        -> `MoeExpertFp8Gemm{E,K,N,G,KB,SWAP,DOT}` (MUL_ROUTED_WEIGHT=False) resp. `MoeExpertFp8GemmW` (True) once per (token, slot)
           pair p, with the pair's quantised row `a[p // top_k]`, its group scales `a_scale[p // top_k]`, its expert id and (W) its
           routed weight as value operands, and the whole `b` / `b_scale` stacks as weight operands; `c` is rebound.

`invoke_fused_moe_triton_kernel` (vLLM d9105ea80, fused_moe.py:763-910) passes `group_n, group_k = block_shape`, A_scale's strides
(K/G, 1), B_scale's (cdiv(N, G) * K/G, 1, K/G) as (stride_bse, stride_bsk, stride_bsn), and `BLOCK_SIZE_K = min(BLOCK_SIZE_K, G)`.
The pins are `vllm_moe.MOE_GEMM_PINS`' (the same kernel body, the batch-invariant default tile BLOCK 64/64/32, GROUP_SIZE_M 8, 4 warps,
3 stages) with the FP8 numeric constexprs; SWAP_AB is False (`enable_swap_ab` holds on cc 9.0 only) and DOT is the target's e4m3 step
(`frontend.targets.moe_expert_fp8_dot`, the GEMM target's `fp8_moe_dot`; a target that records none refuses by name).  The schedule
(sorted or naive) and the routed weights are `FusedMoeKernelRule`'s.
"""

from __future__ import annotations

import math

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.rules.base import RuleInfo
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.triton_iface import launch_meta, launch_tensor
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, MOE_GEMM_PINS, FusedMoeKernelRule, _flat_pairs, _not_owned
from verity_vllm.program.registry.fp8 import weight_blocks

FP8_E4M3 = torch.float8_e4m3fn

#: the numeric constexprs of the block-FP8 path; `group_n == group_k == G > 0` is checked beside them
MOE_FP8_CONSTEXPRS = {"SPLIT_K": 1, "HAS_BIAS": False, "SWAP_AB": False, "USE_TD": False, "use_fp8_w8a8": True, "use_int8_w8a8": False,
                      "use_int8_w8a16": False, "per_channel_quant": False, "compute_type": "<dtype:bf16>"}

_OPERANDS = ("a_ptr", "b_ptr", "c_ptr", "a_scale_ptr", "b_scale_ptr")


def _fp8_launch_pins(node, meta, sc) -> str | None:
    if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
        return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
    if meta.get("src_sha256") != MOE_GEMM_PINS["src_sha256"]:
        return f"kernel source sha256 {meta.get('src_sha256')} differs from the pin {MOE_GEMM_PINS['src_sha256']} (same name, another kernel body)"
    for k, v in MOE_FP8_CONSTEXPRS.items():
        if sc.get(k) != v:
            return f"numerics_unregistered: constexpr {k}={sc.get(k)!r} differs from the modelled block-FP8 path ({v!r})"
    G = sc.get("group_k")
    if not isinstance(G, int) or G <= 0 or sc.get("group_n") != G:
        return f"numerics_unregistered: block_shape (group_n, group_k) = ({sc.get('group_n')!r}, {G!r}); the Definitions take one square block G"
    if sc.get("b_bias_ptr") is not None or launch_tensor(node, meta, "b_bias_ptr") is not None:
        return "numerics_unregistered: b_bias_ptr is set (a bias epilogue is a different numeric)"
    for k, v in MOE_GEMM_PINS["tiles"].items():
        if sc.get(k) != v:
            return f"numerics_unregistered: tiling {k}={sc.get(k)!r} is not the modelled launch geometry ({MOE_GEMM_PINS['tiles']})"
    geom = meta.get("geometry") or {}
    if (geom.get("num_warps", sc.get("num_warps")), geom.get("num_stages", sc.get("num_stages"))) != (MOE_GEMM_PINS["num_warps"], MOE_GEMM_PINS["num_stages"]):
        return f"numerics_unregistered: num_warps/num_stages {geom.get('num_warps')}/{geom.get('num_stages')} are not {MOE_GEMM_PINS['num_warps']}/{MOE_GEMM_PINS['num_stages']}"
    if G % sc["BLOCK_SIZE_K"]:
        return f"BLOCK_SIZE_K={sc['BLOCK_SIZE_K']} does not divide the block G={G} (a K tile would straddle two scale blocks)"
    return None


def moe_expert_fp8_role(routed: bool) -> tuple[str, dict]:
    """(vocabulary role, its DOT) of a block-FP8 launch's per-pair Definition on the active target: the `_dot` roles' block-FP8 forms
    (`vocab._moe_expert_dot` with `G`); raises `frontend.targets.UnregisteredTarget` for a target without a registered e4m3 step."""
    from verity_vllm.program.frontend import targets as T
    return ("moe_expert_gemm_w_dot" if routed else "moe_expert_gemm_dot"), {"DOT": T.moe_expert_fp8_dot(active_target_profile())}


class FusedMoeKernelFp8Rule(FusedMoeKernelRule):
    """Block-FP8 `fused_moe_kernel` launch -> `MoeExpertFp8Gemm{E,K,N,G,KB,SWAP,DOT}` / `MoeExpertFp8GemmW` once per (token, slot) pair;
    `c` is rebound."""
    info = RuleInfo("FusedMoeKernelFp8[vllm-moe]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of fused_moe.fused_moe_kernel with use_fp8_w8a8 and block_shape [G, G] (kernel name + source + constexprs pinned)",
                    preconditions=("block-FP8 path: SPLIT_K 1, HAS_BIAS False, SWAP_AB False, USE_TD False, use_fp8_w8a8 True, no int8, per_channel_quant False, "
                                   "group_n == group_k == G, compute_type bf16; the target registers an e4m3 step (cc 12.0)",
                                   "a is e4m3 [rows, K] row-major with rows*top_k == num_valid_tokens; b is e4m3 [E, N, K] contiguous; c a bf16 "
                                   "allocation/value of num_valid_tokens rows of N",
                                   "a_scale is an f32 value [rows, K/G] (strides K/G, 1); b_scale an f32 weight [E, cdiv(N, G), K/G] (strides cdiv(N, G)*K/G, "
                                   "K/G, 1); K % G == 0 and G % BLOCK_SIZE_K == 0",
                                   "launch geometry BLOCK 64/64/32, GROUP_SIZE_M 8, 4 warps, 3 stages (vLLM's batch-invariant default)",
                                   "schedule and routed weights as FusedMoeKernel[vllm-moe]"),
                    evidence_domain="structural; numerics: MoeExpertFp8Gemm_v1 / MoeExpertFp8GemmW_v1 over MoeExpertFp8BlockGemm(W)_v3{DOT} "
                                    "(model-untested on cc 12.0: registry.fp8_moe DOT_CONF)")

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
        if (r := _not_owned(INDEXED_READ)):
            return r
        sc = meta.get("scalars", {})
        if (r := _fp8_launch_pins(node, meta, sc)):
            return r
        from verity_vllm.program.frontend import targets as T
        try:
            T.moe_expert_fp8_dot(active_target_profile())
        except T.UnregisteredTarget as e:
            return f"numerics_unregistered: {e}"
        mrw, top_k = sc.get("MUL_ROUTED_WEIGHT"), sc.get("top_k")
        if mrw not in (True, False):
            return f"MUL_ROUTED_WEIGHT={mrw!r}"
        if not isinstance(top_k, int) or top_k < 1:
            return f"top_k={top_k!r}"
        a, b, c, a_s, b_s = (launch_tensor(node, meta, n) for n in _OPERANDS)
        if any(t is None for t in (a, b, c, a_s, b_s)):
            return "a_ptr/b_ptr/c_ptr/a_scale_ptr/b_scale_ptr not all tensors"
        for nm, t, dt in (("a", a, FP8_E4M3), ("b", b, FP8_E4M3), ("a_scale", a_s, torch.float32), ("b_scale", b_s, torch.float32)):
            if not isinstance(ctx.val(t), Coll):
                return f"{nm} is not a value collection ({type(ctx.val(t)).__name__})"
            if not dtype_is(ctx, t, dt):
                return f"numerics_unregistered: {nm} dtype {ctx.dtype_of(t)} (the Definitions take {dt})"
        if ctx.weight_of(b) is None or ctx.weight_of(b_s) is None:
            return "b / b_scale are not (views of) parameters; the Definitions take the expert stack and its block scales as weight Inputs"
        if not isinstance(ctx.val(c), (Alloc, Coll)) or ctx.dtype_of(c) != torch.bfloat16:
            return f"c is not a bf16 allocation/value ({type(ctx.val(c)).__name__}, {ctx.dtype_of(c)})"
        return self._shapes(sc, *(shape(ctx, t) for t in (a, b, c, a_s, b_s))) or self._schedule(node, ctx, meta, sc, sc.get("num_valid_tokens"), mrw) or True

    @staticmethod
    def _shapes(sc, as_, bs, cs, ass, bss) -> str | None:
        if len(as_) != 2 or len(bs) != 3 or len(cs) < 2:
            return f"ranks a{as_} b{bs} c{cs}"
        (rows, K), (E, N, K2), G, nv = as_, bs, sc["group_k"], sc.get("num_valid_tokens")
        if K != K2 or (sc.get("N"), sc.get("K")) != (N, K):
            return f"shapes a{as_} b{bs} vs scalars N={sc.get('N')} K={sc.get('K')}"
        if K % G:
            return f"numerics_unregistered: K={K} is not a multiple of the block G={G}"
        NB, KG = weight_blocks(N, G), K // G
        if ass != (rows, KG) or bss != (E, NB, KG):
            return f"scale shapes a_scale {ass} (want ({rows}, {KG})), b_scale {bss} (want ({E}, {NB}, {KG})) are not the [{G}, {G}] block layout"
        if rows * sc["top_k"] != nv or math.prod(cs[:-1]) != nv or cs[-1] != N:
            return f"a rows {rows} x top_k {sc['top_k']} / c {cs} are not num_valid_tokens={nv} rows of N={N}"
        strides = {"a": (sc.get("stride_am"), sc.get("stride_ak")), "b": (sc.get("stride_be"), sc.get("stride_bn"), sc.get("stride_bk")),
                   "c": (sc.get("stride_cm"), sc.get("stride_cn")), "a_scale": (sc.get("stride_asm"), sc.get("stride_ask")),
                   "b_scale": (sc.get("stride_bse"), sc.get("stride_bsn"), sc.get("stride_bsk"))}
        want = {"a": (K, 1), "b": (N * K, K, 1), "c": (N, 1), "a_scale": (KG, 1), "b_scale": (NB * KG, KG, 1)}
        bad = [f"{k} {strides[k]} (want {want[k]})" for k in want if strides[k] != want[k]]
        return f"strides are not row-major: {', '.join(bad)}" if bad else None

    def emit(self, node, ctx):
        meta = launch_meta(node)
        sc = meta["scalars"]
        a, b, c, a_s, b_s = (launch_tensor(node, meta, n) for n in _OPERANDS)
        mrw, top_k, nv, G, KB, SWAP = sc["MUL_ROUTED_WEIGHT"], sc["top_k"], sc["num_valid_tokens"], sc["group_k"], sc["BLOCK_SIZE_K"], sc["SWAP_AB"]
        E, N, K = shape(ctx, b)
        a_rows, _ = rows_of(ctx.coll(a))
        s_rows, _ = rows_of(ctx.coll(a_s))
        bc, bsc = ctx.coll(b), ctx.coll(b_s)
        naive = sc.get("naive_block_assignment") is True
        ids_node = launch_tensor(node, meta, "expert_ids_ptr") if naive else ctx.val(launch_tensor(node, meta, "sorted_token_ids_ptr")).ids_node
        from verity_vllm.program.registry.prims import F32, I32
        ids = _flat_pairs(ctx.coll(ids_node), nv, I32)
        ws = _flat_pairs(ctx.coll(launch_tensor(node, meta, "topk_weights_ptr")), nv, F32) if mrw else None
        role, dot = moe_expert_fp8_role(mrw)
        fn = ctx.kind(role, node, E=E, K=K, N=N, G=G, KB=KB, SWAP=SWAP, **dot)
        hint = ctx.hint_name(node)
        outs = []
        for p in range(nv):
            t = p // top_k
            args = (a_rows[t], s_rows[t], ids[p]) + ((ws[p],) if mrw else ()) + (bc, bsc)
            outs.append(ctx.call(fn, *args, node=node, rule=self, name=f"{hint}[{p}]"))
        ctx.rebind(c, from_rows(outs, tuple(shape(ctx, c)[:-1])), self, what="c")
        ctx.bind_weight_role(b, self, "moe.expert_weight", E=E, K=K, N=N)
        ctx.bind_weight_role(b_s, self, "moe.expert_weight_scale", E=E, K=K, N=N, G=G)
        wl, sl = ctx.weight_of(b), ctx.weight_of(b_s)
        ctx.record(self, node, {"E": E, "K": K, "N": N, "G": G, "KB": KB, "SWAP": SWAP, "DOT": dot["DOT"].id, "pairs": nv, "top_k": top_k,
                                "MUL_ROUTED_WEIGHT": mrw, "schedule": "naive" if naive else "sorted",
                                "tiles": {k: sc.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")},
                                "num_warps": (meta.get("geometry") or {}).get("num_warps"), "src_sha256": meta.get("src_sha256"),
                                "W_is_parameter": wl.name if wl else None, "Ws_is_parameter": sl.name if sl else None}, emitted=f"{fn.id} x{nv}")


VLLM_MOE_FP8_RULES = [FusedMoeKernelFp8Rule()]

__all__ = ["FP8_E4M3", "FusedMoeKernelFp8Rule", "MOE_FP8_CONSTEXPRS", "VLLM_MOE_FP8_RULES", "moe_expert_fp8_role"]
