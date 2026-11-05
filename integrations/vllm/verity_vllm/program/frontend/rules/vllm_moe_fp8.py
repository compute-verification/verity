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
The pins are `vllm_moe.MOE_GEMM_PINS`' kernel body with the FP8 numeric constexprs, and DOT is the launch's e4m3 step
(`frontend.targets.moe_expert_fp8_launch_dot`; a target or tile without a registered step refuses by name).  The tile is
the one vLLM picks: on a batch-invariant target `get_default_config`'s fixed BLOCK 64/64/32, GROUP_SIZE_M 8, 4 warps, 3 stages, SWAP_AB
False; on cc 9.0 without batch invariance the tuned config's (fused_moe.py:1422-1451), with SWAP_AB =
`enable_swap_ab` (BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64, fused_moe/utils.py:628-633) and DOT selected by the
`fp8_moe.sm90_dot` rule (MMAv2 at the small tiles, QGMMA elsewhere): the acceptance rule is
`frontend.targets.moe_fp8_tuned_tile_refusal`, which the fold applies too.  The Definitions read only KB = BLOCK_SIZE_K and SWAP.
The schedule (sorted or naive) and the routed weights are `FusedMoeKernelRule`'s.

    _C.silu_and_mul_per_block_quant(q! e4m3 [pairs, I], gu bf16 [pairs, 2I], scale! f32 [pairs, I/G], G, None, False)
        -> `SiluMulFp8BlockQuant{I,G}` per (token, slot) row; `q` and `scale` are rebound (the w2 launch's `a` and `a_scale`).
"""

from __future__ import annotations

import math

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.rules.common import dtype_is, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.triton_iface import launch_meta, launch_tensor
from verity_vllm.program.frontend.rules.vllm_bindings.observations import active_target_profile
from verity_vllm.program.frontend.rules.base import InPlaceRule, RuleInfo
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, MOE_GEMM_PINS, FusedMoeKernelRule, _flat_pairs, _not_owned
from verity.ml.definitions.registry.fp8 import weight_blocks

FP8_E4M3 = torch.float8_e4m3fn

#: the numeric constexprs of the block-FP8 path; `group_n == group_k == G > 0` is checked beside them
MOE_FP8_CONSTEXPRS = {"SPLIT_K": 1, "HAS_BIAS": False, "SWAP_AB": False, "USE_TD": False, "use_fp8_w8a8": True, "use_int8_w8a8": False,
                      "use_int8_w8a16": False, "per_channel_quant": False, "compute_type": "<dtype:bf16>"}

_OPERANDS = ("a_ptr", "b_ptr", "c_ptr", "a_scale_ptr", "b_scale_ptr")


def _tile_pins(meta, sc, profile) -> str | None:
    """None when the launch's tile is the one vLLM picks on the declared target and its dot is modelled (the module doc)."""
    geom = meta.get("geometry") or {}
    warps, stages = geom.get("num_warps", sc.get("num_warps")), geom.get("num_stages", sc.get("num_stages"))
    if profile.batch_invariant:
        if sc.get("SWAP_AB") is not False:
            return f"numerics_unregistered: constexpr SWAP_AB={sc.get('SWAP_AB')!r} differs from the modelled block-FP8 path (False)"
        for k, v in MOE_GEMM_PINS["tiles"].items():
            if sc.get(k) != v:
                return f"numerics_unregistered: tiling {k}={sc.get(k)!r} is not the modelled launch geometry ({MOE_GEMM_PINS['tiles']})"
        if (warps, stages) != (MOE_GEMM_PINS["num_warps"], MOE_GEMM_PINS["num_stages"]):
            return f"numerics_unregistered: num_warps/num_stages {warps}/{stages} are not {MOE_GEMM_PINS['num_warps']}/{MOE_GEMM_PINS['num_stages']}"
        return None
    from verity_vllm.program.frontend import targets as T
    return T.moe_fp8_tuned_tile_refusal(profile, sc.get("BLOCK_SIZE_M"), sc.get("BLOCK_SIZE_N"), sc.get("BLOCK_SIZE_K"), warps, sc.get("SWAP_AB"))


def _fp8_launch_pins(node, meta, sc, profile) -> str | None:
    if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
        return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
    if meta.get("src_sha256") != MOE_GEMM_PINS["src_sha256"]:
        return f"kernel source sha256 {meta.get('src_sha256')} differs from the pin {MOE_GEMM_PINS['src_sha256']} (same name, another kernel body)"
    for k, v in MOE_FP8_CONSTEXPRS.items():
        if k != "SWAP_AB" and sc.get(k) != v:
            return f"numerics_unregistered: constexpr {k}={sc.get(k)!r} differs from the modelled block-FP8 path ({v!r})"
    G = sc.get("group_k")
    if not isinstance(G, int) or G <= 0 or sc.get("group_n") != G:
        return f"numerics_unregistered: block_shape (group_n, group_k) = ({sc.get('group_n')!r}, {G!r}); the Definitions take one square block G"
    if sc.get("b_bias_ptr") is not None or launch_tensor(node, meta, "b_bias_ptr") is not None:
        return "numerics_unregistered: b_bias_ptr is set (a bias epilogue is a different numeric)"
    if (r := _tile_pins(meta, sc, profile)):
        return r
    if G % sc["BLOCK_SIZE_K"]:
        return f"BLOCK_SIZE_K={sc['BLOCK_SIZE_K']} does not divide the block G={G} (a K tile would straddle two scale blocks)"
    return None


def moe_expert_fp8_role(routed: bool, bm: int, bn: int, warps: int, swap: bool) -> tuple[str, dict]:
    """(vocabulary role, its DOT) of a block-FP8 launch's per-pair Definition on the active target: the `_dot` roles' block-FP8 forms
    (`vocab._moe_expert_dot` with `G`); raises `frontend.targets.UnregisteredTarget` for a target without a registered e4m3 step."""
    from verity_vllm.program.frontend import targets as T
    dot = T.moe_expert_fp8_launch_dot(active_target_profile(), bm, bn, warps, swap)
    return ("moe_expert_gemm_w_dot" if routed else "moe_expert_gemm_dot"), {"DOT": dot}


class FusedMoeKernelFp8Rule(FusedMoeKernelRule):
    """Block-FP8 `fused_moe_kernel` launch -> `MoeExpertFp8Gemm{E,K,N,G,KB,SWAP,DOT}` / `MoeExpertFp8GemmW` once per (token, slot) pair;
    `c` is rebound."""
    info = RuleInfo("FusedMoeKernelFp8[vllm-moe]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of fused_moe.fused_moe_kernel with use_fp8_w8a8 and block_shape [G, G] (kernel name + source + constexprs pinned)",
                    preconditions=("block-FP8 path: SPLIT_K 1, HAS_BIAS False, USE_TD False, use_fp8_w8a8 True, no int8, per_channel_quant False, "
                                   "group_n == group_k == G, compute_type bf16; the target registers an e4m3 step (cc 9.0, cc 12.0)",
                                   "a is e4m3 [rows, K] row-major with rows*top_k == num_valid_tokens; b is e4m3 [E, N, K] contiguous; c a bf16 "
                                   "allocation/value of num_valid_tokens rows of N",
                                   "a_scale is an f32 value [rows, K/G] (strides K/G, 1); b_scale an f32 weight [E, cdiv(N, G), K/G] (strides cdiv(N, G)*K/G, "
                                   "K/G, 1); K % G == 0 and G % BLOCK_SIZE_K == 0",
                                   "batch invariant: BLOCK 64/64/32, GROUP_SIZE_M 8, 4 warps, 3 stages, SWAP_AB False (vLLM's default); otherwise "
                                   "cc 9.0 with SWAP_AB = enable_swap_ab(BLOCK_SIZE_M, BLOCK_SIZE_N), and the tile's MMAv2/QGMMA step registered "
                                   "(vLLM's tuned config; the shared tile-lowering rule)",
                                   "schedule and routed weights as FusedMoeKernel[vllm-moe]"),
                    evidence_domain="structural; numerics: MoeExpertFp8Gemm_v1 / MoeExpertFp8GemmW_v1 over MoeExpertFp8BlockGemm(W)_v3{DOT} "
                                    "(registry.fp8_moe DOT_CONF: exact-model-tested at cc 12.0's step; at cc 9.0's, the _v1 forms, model-untested)")

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != MOE_GEMM_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {MOE_GEMM_PINS['kernel']}"
        if (r := _not_owned(INDEXED_READ)):
            return r
        sc = meta.get("scalars", {})
        profile = active_target_profile()
        if (r := _fp8_launch_pins(node, meta, sc, profile)):
            return r
        from verity_vllm.program.frontend import targets as T
        geom = meta.get("geometry") or {}
        try:
            T.moe_expert_fp8_launch_dot(profile, sc.get("BLOCK_SIZE_M"), sc.get("BLOCK_SIZE_N"),
                                        geom.get("num_warps", sc.get("num_warps")), sc.get("SWAP_AB"))
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
        BM, BN = sc["BLOCK_SIZE_M"], sc["BLOCK_SIZE_N"]
        WARPS = (meta.get("geometry") or {}).get("num_warps", sc.get("num_warps"))
        E, N, K = shape(ctx, b)
        a_rows, _ = rows_of(ctx.coll(a))
        s_rows, _ = rows_of(ctx.coll(a_s))
        bc, bsc = ctx.coll(b), ctx.coll(b_s)
        naive = sc.get("naive_block_assignment") is True
        ids_node = launch_tensor(node, meta, "expert_ids_ptr") if naive else ctx.val(launch_tensor(node, meta, "sorted_token_ids_ptr")).ids_node
        from verity.ml.definitions.registry.prims import F32, I32
        ids = _flat_pairs(ctx.coll(ids_node), nv, I32)
        ws = _flat_pairs(ctx.coll(launch_tensor(node, meta, "topk_weights_ptr")), nv, F32) if mrw else None
        role, dot = moe_expert_fp8_role(mrw, BM, BN, WARPS, SWAP)
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


class SiluAndMulPerBlockQuantRule(InPlaceRule):
    """`_C.silu_and_mul_per_block_quant(q! e4m3 [rows, I], gu bf16 [rows, 2I], scale! f32 [rows, I/G], G, None, False)` ->
    `SiluMulFp8BlockQuant{I,G}` per row; `q` and `scale` are rebound (ports 0 and 1)."""
    modes = frozenset({"profile"})
    info = RuleInfo("SiluAndMulPerBlockQuant[vllm-moe]", 1, "in-place-binding",
                    applies_to="_C.silu_and_mul_per_block_quant(Tensor! out, input, Tensor! scales, group_size, scale_ub, is_scale_transposed)",
                    preconditions=("input is a bf16 value [rows, 2I] = [gate || up]", "out is an unwritten float8_e4m3fn allocation [rows, I]",
                                   "scales is an unwritten f32 allocation [rows, I/G], row-major (is_scale_transposed False)",
                                   "group_size G a power of two dividing I; scale_ub None"),
                    evidence_domain="structural; numerics: SiluMulFp8BlockQuant_v1 (registry.fp8_moe, model-untested: not on "
                                    "vLLM's served cc 9.0 path; glm53_block_fp8_capture/v3 compared it on cc 9.0 at "
                                    "VLLM_USE_DEEP_GEMM=0)")

    def targets(self):
        return targets_for("silu_and_mul_per_block_quant")

    @staticmethod
    def _args(node) -> dict:
        d = dict(zip(("out", "input", "scales", "group_size", "scale_ub", "is_scale_transposed"), node.args))
        d.update(node.kwargs)
        return d

    def applies(self, node, ctx):
        a = self._args(node)
        if not all(k in a for k in ("out", "input", "scales", "group_size")):
            return "unexpected operator arity"
        x, G = a["input"], a["group_size"]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return "input is not a value collection"
        if not dtype_is(ctx, x, torch.bfloat16):
            return f"numerics_unregistered: input dtype {ctx.dtype_of(x)} (the Definition reads bf16 [gate || up])"
        xs = shape(ctx, x)
        if len(xs) != 2 or xs[1] % 2:
            return f"input {xs} is not [rows, 2I]"
        rows, I = xs[0], xs[1] // 2
        if not isinstance(G, int) or isinstance(G, bool) or G <= 0 or G & (G - 1) or I % G:
            return f"numerics_unregistered: group_size {G!r} is not a power of two dividing I={I}"
        if a.get("scale_ub") is not None:
            return "numerics_unregistered: scale_ub is set (the scale clamp is a different numeric)"
        if a.get("is_scale_transposed", False) is not False:
            return f"numerics_unregistered: is_scale_transposed={a.get('is_scale_transposed')!r} (the Definition writes [rows, I/G] row-major)"
        for nm, dt, want in (("out", FP8_E4M3, (rows, I)), ("scales", torch.float32, (rows, I // G))):
            t = a[nm]
            if not isinstance(ctx.val(t), Alloc):
                return f"{nm} is not an unwritten allocation ({type(ctx.val(t)).__name__})"
            if not dtype_is(ctx, t, dt):
                return f"numerics_unregistered: {nm} dtype {ctx.dtype_of(t)} (the Definition writes {dt})"
            if shape(ctx, t) != want:
                return f"{nm} {shape(ctx, t)} is not {want}"
        return True

    def emit(self, node, ctx):
        a = self._args(node)
        rows, lead = rows_of(ctx.coll(a["input"]))
        I, G = rows[0].type.n // 2, int(a["group_size"])
        fn = ctx.kind("silu_mul_fp8_block_quant", node, I=I, G=G)
        outs = [ctx.call(fn, r, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.rebind(a["out"], from_rows([o[0] for o in outs], lead), self, what="out")
        ctx.rebind(a["scales"], from_rows([o[1] for o in outs], lead), self, what="scales")
        ctx.record(self, node, {"I": I, "G": G, "rows": len(rows)}, emitted=f"{fn.id} x{len(rows)}")


VLLM_MOE_FP8_RULES = [FusedMoeKernelFp8Rule(), SiluAndMulPerBlockQuantRule()]

__all__ = ["FP8_E4M3", "FusedMoeKernelFp8Rule", "MOE_FP8_CONSTEXPRS", "SiluAndMulPerBlockQuantRule", "VLLM_MOE_FP8_RULES",
           "moe_expert_fp8_role"]
