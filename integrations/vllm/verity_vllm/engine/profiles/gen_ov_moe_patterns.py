"""Lane `ov-moe` patterns: the OLMoE fused-MoE block as PAIR-MAJOR Computes over quarantine definitions.

Wave 1 (`gen_moe_patterns.MoEBlockScope`, retired in P6) resolved `vllm.moe_forward*` as one recognised `Unsupported` and wrote the
mechanism note.  This module is the constructive half of the answer to "is MoE just routing + grouped-GEMM Definitions
under the existing Program machinery?": the block IS representable with the existing operand kinds (`PerRow`, `Gather`
with *public* indices, `Shared`) once the unit of computation is the (token, slot) PAIR p = t*TOPK + k and the expert
index is an ordinary I32 *value operand* of the pair body (the gather over the expert axis happens inside the body,
cf. `Embedding_v1`).  Nothing in the fold reads device data to resolve dataflow: the A-row of pair p is hidden row
`p // TOPK` (structural), the expert id of pair p is word p of `topk_ids` (a value the router Compute produced), the
weight operand is the whole expert table (`Shared`).

Resolution of the scope (`MoEFusedExperts`), in fold order:
  1. `_moe_C.topk_softmax`             -> Compute(MoeRouterTopK_v1{E,TOPK,VPT}, rows=M): logits[M,E] -> (weights[M,TOPK] f32, ids[M,TOPK] i32)
  2. `_moe_C.moe_align_block_size`     -> Transparent(noop) claiming the kernel: its outputs (sorted_token_ids, expert_ids,
                                          num_tokens_post_padded) are a SCHEDULE — they decide which pairs a CTA processes
                                          together and never enter the arithmetic of an output word (§4 of the wave-1
                                          mechanism note; the Tier A comparison of 3./5. is the check of that claim).
                                          Absent on the naive path (M <= 2: `naive_block_assignment=True`).
  3. `fused_moe_kernel` (w13)          -> Compute(MoeExpertGemm_v1{E,K=H,N=2I}, rows=M*TOPK):
                                          (Gather(hidden, p//TOPK), PerRow(ids.flat), Shared(w13)) -> PerRow(cache1 as [M*TOPK, 2I])
  4. `_C.silu_and_mul`                 -> the base `SiluAndMul` Compute (SiluMul_v1{I}, rows=M*TOPK)
  5. `fused_moe_kernel` (w2, weighted) -> Compute(MoeExpertGemmW_v1{E,K=I,N=H}, rows=M*TOPK):
                                          (PerRow(cache2), PerRow(ids.flat), PerRow(weights.flat), Shared(w2)) -> PerRow(cache3 as [M*TOPK, H])
  6. `_moe_C.moe_sum`                  -> Compute(MoeSum_v1{TOPK,H}, rows=M): PerRow(cache3[M,TOPK,H]) -> PerRow(out[M,H])
Everything else under the scope is aten glue (workspace slices / dtype views / empties) handled by `scope_transparents`.

Pins (κ): the router kernel's reduction geometry `VPT` (from the `topkGating<VPT, E, ...>` template instantiation),
`renormalize=False`, no bias, no padding rows; the fused kernel's non-schedule constexprs (`compute_type=bf16`, no
quantisation, `SPLIT_K=1`, `HAS_BIAS=False`, `SWAP_AB=False`) plus — for parity with the base GEMM's κ pins — its tiling
(`FUSED_MOE_PINS`).  Anything else is `Unsupported` with the observed values.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from verity.ir.defs import bind
from verity_vllm.program.registry import moe as _MOE
from verity_vllm.program.registry.quarantine.ov_moe import moe as Q

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (SiluAndMul, bound_args, find_descendants, is_int, is_row_major, op_name, scope_transparents, triton_name,
                        unsupported)
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.engine.profiles.gen_llama_patterns import GemmLaunchTuned

#: The op name prefix of the MoE custom op(s) (`vllm.moe_forward.default`, `vllm.moe_forward_shared.default`).
MOE_OP_PREFIX = "moe_forward"

#: Device kernels of the fused-MoE block on the pinned vLLM (Triton function names / CUDA kernel tokens), for the census reader.
MOE_KERNELS_EXPECTED = ("topk_softmax", "moe_align_block_size", "fused_moe_kernel", "silu_and_mul", "moe_sum")


def _view(v: Any) -> dict[str, Any]:
    return {"shape": list(v.shape), "dtype": v.dtype, "strides": list(v.strides), "offset": v.offset, "alloc": v.alloc}


def _scalar(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    if hasattr(v, "shape") and hasattr(v, "strides"):
        return _view(v)
    return repr(v)


def _depth(root: TreeNode, target: TreeNode) -> int:
    def go(n: TreeNode, d: int) -> int | None:
        if n is target:
            return d
        for c in n.children:
            r = go(c, d + 1)
            if r is not None:
                return r
        return None
    return go(root, 0) or 0


def moe_subtree_census(node: TreeNode) -> dict[str, Any]:
    """Every event of the scope with its tensor metadata: dispatcher ops with input/output views, Triton launches with
    grid / constexprs / num_warps / num_stages and their tensor arguments.  Serialisable (plain dicts) so it survives
    the resolution report's JSON.  (Wave 1's `gen_moe_patterns.moe_subtree_census`, moved here when that module retired in P6.)"""
    ops: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    for n in node.walk():
        if n is node:
            continue
        if n.is_triton:
            e = n.enter
            launches.append({"id": n.id, "fn": triton_name(n), "grid": list(e.grid), "constexprs": dict(e.constexprs),
                             "num_warps": e.num_warps, "num_stages": e.num_stages,
                             "args": {k: _scalar(v) for k, v in e.args.items() if k not in e.constexprs}})
        else:
            ops.append({"id": n.id, "op": n.op, "depth": _depth(node, n),
                        "in": [_view(v) for v in n.input_views()], "out": [_view(v) for v in n.output_views()]})
    return {"dispatcher_ops": ops, "triton_launches": launches, "n_events": len(node.ids()) - 1}

#: Kineto / Triton kernel tokens the parts claim (census).
K_ROUTER = "topkGating"
#: `_moe_C.moe_align_block_size` dispatches on the pair count (vllm csrc/moe/moe_align_sum_kernels.cu): one
#: `moe_align_block_size_small_batch_expert_kernel` launch below ~1024 pairs (`topk_ids.numel() < 1024 && E <= 64`:
#: every decode step), else the two-kernel sorted path `moe_align_block_size_kernel` + `count_and_sort_expert_tokens_kernel`
#: (a 155-token prefill = 1240 pairs; census grid [2,1,1]x1024 and [5,1,1]x256).  All three compute the same schedule
#: (lane cov-moe, 2026-09-17: the prefill pair was census residue — G1 FAIL, 32 kernels — under the single-name claim).
K_ALIGN = "moe_align_block_size_small_batch_expert_kernel"
K_ALIGN_KERNELS = (K_ALIGN, "moe_align_block_size_kernel", "count_and_sort_expert_tokens_kernel")
K_FUSED = "fused_moe_kernel"
K_SUM = "moe_sum_vec_kernel"

#: The fused kernel's constexprs that must hold for the bodies to apply (numerical, not schedule).
FUSED_MOE_NUMERIC_PINS: dict[str, Any] = {"compute_type": "bf16", "use_fp8_w8a8": False, "use_int8_w8a8": False, "use_int8_w8a16": False,
                                          "per_channel_quant": False, "HAS_BIAS": False, "SWAP_AB": False, "USE_TD": False,
                                          "SPLIT_K": 1, "group_n": 0, "group_k": 0}
#: Its tiling on the pinned κ (vLLM d9105ea80 default MoE config at E=64, sm89, bf16) — schedule, pinned for parity
#: with the base GEMM's κ pins (a different tiling is a new κ that must be looked at, not silently accepted).
FUSED_MOE_PINS: dict[str, Any] = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
FUSED_MOE_NUM_WARPS = 4
FUSED_MOE_NUM_STAGES = 3

#: `topkGating<VPT, NUM_EXPERTS, WARPS_PER_CTA, BYTES_PER_LDG, WARP_SIZE, ...>`: values per thread for bf16 gating logits -- 16-byte loads
#: = 8 bf16 words per thread-vector, ONE vector per thread while E <= 8 * 32 (`VECs_PER_THREAD = max(1, E / (ELTS_PER_LDG * WARP_SIZE))`), so
#: VPT = 8 at E = 64 (OLMoE; bit-exact on the recorded topk_softmax cases) and at E = 128 (Qwen3-30B-A3B; the frontend's
#: `vllm_moe.TOPK_SOFTMAX_VPT[bf16]`, the same 8 the Build of record bound -- `MoeRouterTopKNorm_v1{E=128,TOPK=8,VPT=8}`).
ROUTER_VPT_BY_E: dict[int, int] = {64: 8, 128: 8}



def _root_name(ctx, view, fallback: str) -> str:
    """The expert stack's weight ROOT name (the parameter fqn, `model.layers.<k>.mlp.experts.routed_experts.w13_weight`), the hint tail
    every other pattern uses (`hint_of(node, roots[0].name)`): `cb_a.compare` reads the record's weight field names from it.  [R13 MOE-01:
    the literal `w13`/`w2` tails made every expert-GEMM weight operand differ from the derived Program's by NAME only.]"""
    roots = ctx.roots_of(view)
    return roots[0].name if roots else fallback

def _flat(v: ViewDesc, n: int) -> ViewDesc:
    """A contiguous `[M, TOPK]` view as `[M*TOPK]` (same words, same allocation)."""
    return ViewDesc(v.alloc, v.offset, (n,), (1,), v.dtype, v.device)


def _as_rows(v: ViewDesc, rows: int, cols: int) -> ViewDesc:
    """A contiguous `[M, TOPK, N]` view as `[M*TOPK, N]` (same words)."""
    return ViewDesc(v.alloc, v.offset, (rows, cols), (cols, 1), v.dtype, v.device)


def _contig(v: ViewDesc) -> bool:
    exp = []
    s = 1
    for d in reversed(v.shape):
        exp.append(s)
        s *= d
    return tuple(v.strides) == tuple(reversed(exp))


@dataclass
class MoEFusedExperts:
    """`vllm.moe_forward*(hidden_states [M,H] bf16, router_logits [M,E] bf16, ...) -> [M,H] bf16` as the six-part Group above."""

    name = "moe.fused_experts.pairwise"
    config: dict[str, Any]
    router_vpt: dict[int, int] = field(default_factory=lambda: dict(ROUTER_VPT_BY_E))
    #: [R19 qwen3] the router the scope must carry: `renormalize=False` -> `MoeRouterTopK_v1` (OLMoE, `norm_topk_prob=false`);
    #: `renormalize=True` -> `MoeRouterTopKNorm_v1` (Qwen3-MoE, `norm_topk_prob=true`: the TOPK probabilities rescaled to sum 1, same ids).
    #: A profile states ONE of them (the checkpoint's `norm_topk_prob`, `C["NORM_TOPK_PROB"]`); the other value observed is Unsupported.
    renormalize: bool = False
    #: [R19 qwen3, TP] patterns for descendants of the scope that are neither the six MoE kernels nor scope glue -- vLLM's `FusedMoE`
    #: issues the tensor-parallel all-reduce of the routed sum INSIDE `vllm.moe_forward` when `reduce_results=True`; a rank profile puts
    #: its `TpCollective` here so the collective is resolved (and its Compute appended after `MoeSum`) instead of making the whole scope
    #: `unexplained child vllm.all_reduce`.  Empty for TP1 (nothing else may appear under the scope).
    scope_patterns: tuple = ()
    #: [R19 qwen3] the fused_moe_kernel tiling this profile pins (κ; default = the module constants).  A profile whose capture shows another
    #: tiling pins it HERE (from the rank log), never by loosening the check: an unpinned tiling is Unsupported with the observed values.
    fused_pins: dict[str, Any] = field(default_factory=lambda: dict(FUSED_MOE_PINS))
    fused_num_warps: int = FUSED_MOE_NUM_WARPS
    fused_num_stages: int = FUSED_MOE_NUM_STAGES

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or not op_name(node.op).startswith(MOE_OP_PREFIX):
            return None
        C = self.config
        E, TOPK, H, I = C.get("NUM_EXPERTS"), C.get("TOPK"), C["H"], C["I"]
        if not all(isinstance(v, int) for v in (E, TOPK, H, I)):
            return unsupported(node, "profile config lacks NUM_EXPERTS / TOPK / H / I", config={k: C.get(k) for k in ("NUM_EXPERTS", "TOPK", "H", "I")})
        ins = node.input_views()
        sx = bound_args(node).get("shared_experts_input")
        if sx is not None:
            # [R19 qwen3] a shared-experts branch (Qwen2-MoE / DeepSeek-style) is not the block these Definitions state: refuse by name
            return unsupported(node, "MoE scope has a shared_experts_input: shared experts are not modelled", shared_experts_input=repr(sx))
        hidden = next((v for v in ins if v.rank == 2 and v.shape[1] == H), None)
        if hidden is None:
            return unsupported(node, "MoE scope: no [M,H] hidden_states input", inputs=[repr(v) for v in ins])
        M = hidden.shape[0]
        if M != len(ctx.start.rows) or ctx.start.num_tokens != ctx.start.num_tokens_padded:
            return unsupported(node, "MoE rows are not exactly the step's token rows (padding rows would reach the router)",
                               M=M, rows=len(ctx.start.rows), num_tokens=ctx.start.num_tokens, num_tokens_padded=ctx.start.num_tokens_padded)

        routers = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "topk_softmax")
        aligns = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "moe_align_block_size")
        launches = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_FUSED)
        silus = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "silu_and_mul")
        sums = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "moe_sum")
        if len(routers) != 1 or len(launches) != 2 or len(silus) != 1 or len(sums) != 1 or len(aligns) > 1:
            return unsupported(node, "MoE scope does not have the fused_experts shape (1 topk_softmax, <=1 align, 2 fused_moe_kernel, 1 silu_and_mul, 1 moe_sum)",
                               counts={"topk_softmax": len(routers), "align": len(aligns), "fused_moe_kernel": len(launches),
                                       "silu_and_mul": len(silus), "moe_sum": len(sums)}, census=moe_subtree_census(node))
        router, up, down, silu, msum = routers[0], launches[0], launches[1], silus[0], sums[0]
        keep = {router.id, up.id, down.id, silu.id, msum.id} | {a.id for a in aligns}
        keep |= set().union(*(set(n.ids()) for n in (router, up, down, silu, msum, *aligns)))
        # [R19 qwen3, TP] descendants the scope patterns own (the in-scope all-reduce of a rank): resolved here, kept out of the glue walk
        extra: list[Compute] = []
        if self.scope_patterns:
            for d in find_descendants(node, lambda n: n.id not in keep and not n.is_triton):
                if d.id in keep:
                    continue
                for pt in self.scope_patterns:
                    r = pt.match(d, ctx)
                    if r is None:
                        continue
                    if not isinstance(r, Compute):
                        return unsupported(node, f"scope pattern {getattr(pt, 'name', type(pt).__name__)} refused {op_name(d.op)} inside {op_name(node.op)}",
                                           child_id=d.id, child=d.op, reason=getattr(r, "reason", None), child_detail=getattr(r, "detail", None))
                    extra.append(r)
                    keep |= {d.id} | set(d.ids())
                    break
        before, bad = scope_transparents(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        hint = node.enter.module or "mlp.experts"

        # 1. router --------------------------------------------------------------------------------------------
        a = bound_args(router)
        weights, ids, logits = a.get("topk_weights"), a.get("topk_indices"), a.get("gating_output")
        if not all(isinstance(v, ViewDesc) for v in (weights, ids, logits)):
            return unsupported(node, "topk_softmax arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        router_def = _MOE.MoeRouterTopKNorm if self.renormalize else Q.MoeRouterTopK
        if a.get("renormalize") is not bool(self.renormalize) or a.get("bias") is not None:
            return unsupported(node, f"{router_def.id} is the renormalize={bool(self.renormalize)}, bias-free softmax router",
                               renormalize=a.get("renormalize"), bias=repr(a.get("bias")), profile_renormalize=bool(self.renormalize))
        if logits.shape != (M, E) or logits.dtype != "bfloat16" or not is_row_major(logits):
            return unsupported(node, "router logits are not a row-major bf16 [M,E] view", logits=repr(logits))
        if weights.shape != (M, TOPK) or weights.dtype != "float32" or not _contig(weights):
            return unsupported(node, "topk_weights is not a contiguous f32 [M,TOPK] view", weights=repr(weights))
        if ids.shape != (M, TOPK) or ids.dtype != "int32" or not _contig(ids):
            return unsupported(node, "topk_indices is not a contiguous i32 [M,TOPK] view", ids=repr(ids))
        vpt = self.router_vpt.get(E)
        if vpt is None:
            return unsupported(node, "no pinned router reduction geometry (VPT) for this expert count", E=E, pinned=self.router_vpt)
        router_c = Compute(bind(router_def, E=E, TOPK=TOPK, VPT=vpt), M, (PerRow(logits),), (PerRow(weights), PerRow(ids)), (router.id,),
                           hint=f"{hint}/router", kernels=(K_ROUTER,), attribution=tuple(ctx.start.rows), pattern=self.name)
        parts: list[Compute | Transparent] = list(before) + [router_c]

        # 2. alignment: a schedule, no numerical effect ----------------------------------------------------------
        for al in aligns:
            parts.append(Transparent("noop", (al.id,), hint=f"{hint}/moe_align_block_size: pair->expert-block schedule (sorted_token_ids, "
                                     "expert_ids, num_tokens_post_padded); no output word depends on it", kernels=K_ALIGN_KERNELS, pattern=self.name))

        # 3. up/gate GEMM over pairs --------------------------------------------------------------------------------
        P = M * TOPK
        ids_flat = _flat(ids, P)
        r = self._fused(node, up, expect_weighted=False, expect_top_k=TOPK, E=E, K=H, N=2 * I, P=P)
        if isinstance(r, Unsupported):
            return r
        x_up, w13, c1 = r
        if x_up.alloc != hidden.alloc or x_up.shape != (M, H) or not is_row_major(x_up):
            return unsupported(node, "up-GEMM A operand is not the [M,H] hidden_states view", a=repr(x_up), hidden=repr(hidden))
        up_c = Compute(bind(Q.MoeExpertGemm, E=E, K=H, N=2 * I), P,
                       (Gather(x_up, tuple(p // TOPK for p in range(P))), PerRow(ids_flat), Shared(w13)),
                       (PerRow(_as_rows(c1, P, 2 * I)),), (up.id,), hint=f"{hint}/{_root_name(ctx, w13, 'w13')}", kernels=(K_FUSED,), pattern=self.name)
        parts.append(up_c)

        # 4. silu_and_mul on the pair rows ------------------------------------------------------------------------
        silu_r = SiluAndMul().match(silu, ctx)
        if not isinstance(silu_r, Compute):
            return unsupported(node, "silu_and_mul inside the MoE scope did not resolve as SiluMul_v1",
                               detail=getattr(silu_r, "detail", None), reason=getattr(silu_r, "reason", None))
        if silu_r.rows != P:
            return unsupported(node, "silu_and_mul rows are not the M*TOPK pair rows", rows=silu_r.rows, P=P)
        parts.append(dataclasses.replace(silu_r, hint=f"{hint}/silu_and_mul", pattern=self.name))

        # 5. down GEMM over pairs, routed weight in the epilogue ---------------------------------------------------
        r = self._fused(node, down, expect_weighted=True, expect_top_k=1, E=E, K=I, N=H, P=P)
        if isinstance(r, Unsupported):
            return r
        x_dn, w2, c3 = r
        wv = bound_args(down).get("topk_weights_ptr")
        if not isinstance(wv, ViewDesc) or wv.alloc != weights.alloc or wv.offset != weights.offset or wv.shape != (M, TOPK):
            return unsupported(node, "down-GEMM routed weights are not the router's topk_weights view", topk_weights_ptr=repr(wv), weights=repr(weights))
        if x_dn.shape != (P, I) or not is_row_major(x_dn):
            return unsupported(node, "down-GEMM A operand is not the [M*TOPK, I] silu output", a=repr(x_dn))
        down_c = Compute(bind(Q.MoeExpertGemmW, E=E, K=I, N=H), P,
                         (PerRow(x_dn), PerRow(ids_flat), PerRow(_flat(weights, P)), Shared(w2)),
                         (PerRow(_as_rows(c3, P, H)),), (down.id,), hint=f"{hint}/{_root_name(ctx, w2, 'w2')}", kernels=(K_FUSED,), pattern=self.name)
        parts.append(down_c)

        # 6. routed sum -------------------------------------------------------------------------------------------
        a = bound_args(msum)
        y, out = a.get("input"), a.get("output")
        if not isinstance(y, ViewDesc) or not isinstance(out, ViewDesc):
            return unsupported(node, "moe_sum arguments not bound as expected", args={k: repr(v) for k, v in a.items()})
        if a.get("topk_ids") is not None or a.get("expert_map") is not None:
            return unsupported(node, "moe_sum with topk_ids / expert_map (padding-aware variant) is not the modelled kernel")
        if y.shape != (M, TOPK, H) or not _contig(y) or y.dtype != "bfloat16" or y.alloc != c3.alloc or y.offset != c3.offset:
            return unsupported(node, "moe_sum input is not the contiguous bf16 [M,TOPK,H] down-GEMM output", input=repr(y), cache3=repr(c3))
        if out.shape != (M, H) or not is_row_major(out) or out.dtype != "bfloat16":
            return unsupported(node, "moe_sum output is not a row-major bf16 [M,H] view", output=repr(out))
        parts.append(Compute(bind(Q.MoeSum, TOPK=TOPK, H=H), M, (PerRow(y),), (PerRow(out),), (msum.id,), hint=f"{hint}/moe_sum",
                             kernels=(K_SUM,), attribution=tuple(ctx.start.rows), pattern=self.name))
        # 7. [TP] the in-scope collective(s) the scope patterns resolved (log order): the all-reduce of the routed sum, after MoeSum
        parts.extend(sorted(extra, key=lambda c: min(c.consumed)))
        return Group(tuple(parts), tuple(sorted(node.ids())), hint=hint, pattern=self.name)

    def _fused(self, node: TreeNode, launch: TreeNode, *, expect_weighted: bool, expect_top_k: int, E: int, K: int, N: int, P: int):
        """Check one `fused_moe_kernel` launch against the pins and return `(a_view, b_view, c_view)` or an Unsupported."""
        e = launch.enter
        consts = dict(e.constexprs)
        bad = {k: (consts.get(k), v) for k, v in FUSED_MOE_NUMERIC_PINS.items() if consts.get(k) != v}
        if bad:
            return unsupported(node, "fused_moe_kernel numeric constexprs differ from the modelled bf16 unquantised kernel",
                               mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
        bad = {k: (consts.get(k), v) for k, v in self.fused_pins.items() if consts.get(k) != v}
        if bad or e.num_warps != self.fused_num_warps or e.num_stages != self.fused_num_stages:
            return unsupported(node, "fused_moe_kernel tiling differs from the pinned profile",
                               mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()},
                               num_warps=e.num_warps, num_stages=e.num_stages)
        if consts.get("MUL_ROUTED_WEIGHT") is not expect_weighted or consts.get("top_k") != expect_top_k:
            return unsupported(node, "fused_moe_kernel launch order / role does not match (MUL_ROUTED_WEIGHT, top_k)",
                               observed={"MUL_ROUTED_WEIGHT": consts.get("MUL_ROUTED_WEIGHT"), "top_k": consts.get("top_k")},
                               expected={"MUL_ROUTED_WEIGHT": expect_weighted, "top_k": expect_top_k})
        a = bound_args(launch)
        for k in ("b_bias_ptr", "a_scale_ptr", "b_scale_ptr"):
            if a.get(k) is not None:
                return unsupported(node, f"fused_moe_kernel has a {k}", **{k: repr(a.get(k))})
        av, bv, cv = a.get("a_ptr"), a.get("b_ptr"), a.get("c_ptr")
        if not all(isinstance(v, ViewDesc) for v in (av, bv, cv)):
            return unsupported(node, "fused_moe_kernel pointer arguments not bound as expected", args={k: repr(v) for k, v in a.items() if k.endswith("_ptr")})
        if a.get("N") != N or a.get("K") != K or a.get("num_valid_tokens") != P:
            return unsupported(node, "fused_moe_kernel N / K / num_valid_tokens differ from the scope's geometry",
                               observed={"N": a.get("N"), "K": a.get("K"), "num_valid_tokens": a.get("num_valid_tokens")}, expected={"N": N, "K": K, "P": P})
        if bv.shape != (E, N, K) or not _contig(bv) or bv.dtype != "bfloat16":
            return unsupported(node, "fused_moe_kernel B operand is not a contiguous bf16 [E,N,K] expert table", b=repr(bv))
        if (a.get("stride_be"), a.get("stride_bn"), a.get("stride_bk")) != (N * K, K, 1):
            return unsupported(node, "fused_moe_kernel B strides are not [E][N][K] row-major", strides=[a.get("stride_be"), a.get("stride_bn"), a.get("stride_bk")])
        if av.dtype != "bfloat16" or av.rank != 2 or not is_row_major(av) or av.shape[1] != K:
            return unsupported(node, "fused_moe_kernel A operand is not a row-major bf16 [rows,K] view", a=repr(av))
        if cv.dtype != "bfloat16" or cv.numel != P * N or not _contig(cv) or (a.get("stride_cm"), a.get("stride_cn")) != (N, 1):
            return unsupported(node, "fused_moe_kernel C operand is not a contiguous bf16 [M*TOPK, N] pair-major output", c=repr(cv),
                               strides=[a.get("stride_cm"), a.get("stride_cn")])
        ids = a.get("expert_ids_ptr")
        if consts.get("naive_block_assignment") is True:
            if not isinstance(ids, ViewDesc) or not is_int(ids) or ids.numel != P or a.get("sorted_token_ids_ptr") is not None:
                return unsupported(node, "naive path: expert_ids_ptr is not topk_ids.flat", expert_ids_ptr=repr(ids))
        else:
            st = a.get("sorted_token_ids_ptr")
            if not isinstance(st, ViewDesc) or not isinstance(ids, ViewDesc) or not is_int(st) or not is_int(ids):
                return unsupported(node, "sorted path: sorted_token_ids_ptr / expert_ids_ptr not bound", sorted=repr(st), expert_ids=repr(ids))
        return av, bv, cv


@dataclass
class MoEFusedExpertsPadded(MoEFusedExperts):
    """[R17-42 route A (4), `moe_construction = padded`] The SAME six kernels, the SAME pins and shape checks as `MoEFusedExperts` --
    folded to ONE `Compute(MoeFfnFixedL_v1{E,TOPK,VPT,H,I}, rows=M)` per token: `(PerRow(hidden), PerRow(logits), Shared(w13), Shared(w2))
    -> PerRow(out)`.  This is the fold-side twin of the frontend's `PaddedMoeSumRule` (verity_vllm/program/frontend/rules/vllm_moe.py), so GM-01
    sees the record and the Build in the same vocabulary; which twin folds is the workload's DECLARED construction
    (`target.moe_construction`, a Program static of record, read from the manifest by run_config and passed down) -- never read off the
    trace, which is construction-agnostic (the engine ran the same kernels either way).

    What the record no longer names: the router's (weights, ids), the pair caches (cache1 / cache2 / cache3) and the alignment schedule
    are INTERIOR to the one Compute -- no instance is emitted for them and no other kernel reads them (moe_sum's output is this Compute's).
    Their (token, slot) reading for Commit is Stage 3a's (`verity_vllm/observe/moe_pad_transcript.py`: pair row p of token t with `ids[t][k] = e`
    is slot e; the E-TOPK other slots are DISABLED, `enc(bot)`), not the fold's.  Kernel census: every launch of the scope is claimed
    (`kernels`), exactly as before; alignment stays a Transparent noop (a schedule, no numerical effect)."""

    name = "moe.fused_experts.padded"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        r = super().match(node, ctx)
        if not isinstance(r, Group):
            return r      # None (not the scope) or the Unsupported the pairwise checks produced: identical refusals under both constructions
        from verity_vllm.program.registry import moe_pad as MP  # noqa: PLC0415  (the padded Definitions register on first use, as vocab.py's kind does)
        if self.renormalize:
            return unsupported(node, "padded construction with renormalize=True: MoeFfnFixedL_v1 states the MoeRouterTopK_v1 (unnormalised) router only",
                               renormalize=True)
        C = self.config
        E, TOPK, H, I = C["NUM_EXPERTS"], C["TOPK"], C["H"], C["I"]
        parts = list(r.parts)
        computes = [p for p in parts if isinstance(p, Compute)]
        by_id = {p.spec.name: p for p in computes}
        router, up, down, msum = by_id["MoeRouterTopK"], by_id["MoeExpertGemm"], by_id["MoeExpertGemmW"], by_id["MoeSum"]
        silu = next(p for p in computes if p.spec.name == "SiluMul")
        M = router.rows
        logits = router.operands[0].view
        hidden = up.operands[0].view                                   # the Gather over hidden rows: the [M,H] view itself
        w13, w2 = up.operands[2].view, down.operands[3].view
        out = msum.outputs[0].view
        vpt = router.spec.bindings["VPT"]                              # the pinned router geometry the pairwise reading bound
        hint = r.hint
        one = Compute(bind(MP.MoeFfnFixedL, E=E, TOPK=TOPK, VPT=vpt, H=H, I=I), M,
                      (PerRow(hidden), PerRow(logits), Shared(w13), Shared(w2)), (PerRow(out),),
                      tuple(sorted(set(router.consumed) | set(up.consumed) | set(silu.consumed) | set(down.consumed) | set(msum.consumed))),
                      hint=f"{hint}/moe_ffn_fixed", kernels=(K_ROUTER, K_FUSED, *silu.kernels, K_SUM), attribution=router.attribution,
                      pattern=self.name, roots=(None, None, _root_name(ctx, w13, "w13"), _root_name(ctx, w2, "w2")))
        six = {id(router), id(up), id(silu), id(down), id(msum)}
        keep = [p for p in parts if id(p) not in six]                 # the scope glue (before) + the alignment noop (+ a TP collective), in order
        tail = [p for p in keep if isinstance(p, Compute)]            # [TP] the in-scope all-reduce stays AFTER the block's one Compute
        keep = [p for p in keep if not isinstance(p, Compute)]
        return Group(tuple(keep) + (one, *tail), r.consumed, hint=hint, pattern=self.name)


#: The declared constructions the fold knows (the frontend's `TargetProfile.MOE_CONSTRUCTIONS` keys); the pattern each folds with.
MOE_CONSTRUCTION_PATTERNS = {"indexed-read": MoEFusedExperts, "padded": MoEFusedExpertsPadded}


def ov_moe_patterns(config: dict[str, Any], *, gemm_constexprs: dict[str, Any] | None = None, gemm_num_sms: int | None = None,
                    moe_construction: str | None = None) -> list:
    """The MoE architecture's patterns, placed BEFORE the base list (the tuned-GEMM specialization must see `o_proj` first; the MoE
    scope pattern matches the custom op no base pattern knows).  `moe_construction` (None = `indexed-read`, MOE-01) selects the
    scope pattern: the workload's DECLARED construction, never inferred (R17-42 route A (4)).  [P6] the tuned GEMM is the
    target-agnostic `gen_llama_patterns.GemmLaunchTuned` (the record target's table; `gemm_constexprs` is accepted and unused: the
    delegate `GemmLaunch` pins the target's tiling)."""
    key = moe_construction or "indexed-read"
    if key not in MOE_CONSTRUCTION_PATTERNS:
        raise ValueError(f"moe_construction {moe_construction!r} is not one of {sorted(MOE_CONSTRUCTION_PATTERNS)}")
    tuned = GemmLaunchTuned() if gemm_num_sms is None else GemmLaunchTuned(num_sms=gemm_num_sms)
    return [tuned, MOE_CONSTRUCTION_PATTERNS[key](config)]
