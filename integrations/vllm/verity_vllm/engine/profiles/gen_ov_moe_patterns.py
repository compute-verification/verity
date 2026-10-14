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
  0. `matmul_kernel_persistent` (gate) -> the profile's own GEMM Compute, when the router's gate GEMM launches inside the scope
                                          (Qwen3-MoE: `FusedMoE(gate=...)`); it must write the logits the router reads
  1. `_moe_C.topk_softmax`             -> Compute(MoeRouterTopK_v1{E,TOPK,VPT}, rows=M): logits[M,E] -> (weights[M,TOPK] f32, ids[M,TOPK] i32)
  2. `_moe_C.moe_align_block_size`     -> Transparent(noop) claiming the kernel: its outputs (sorted_token_ids, expert_ids,
                                          num_tokens_post_padded) are a SCHEDULE — they decide which pairs a CTA processes
                                          together and never enter the arithmetic of an output word (§4 of the wave-1
                                          mechanism note; the Tier A comparison of 3./5. is the check of that claim).
                                          Absent on the naive path (M <= 2: `naive_block_assignment=True`).
  3. `fused_moe_kernel` (w13)          -> Compute(MoeExpertGemm_v1{E,K=H,N=2I}, rows=M*TOPK):
                                          (Gather(hidden, p//TOPK), PerRow(ids.flat), Shared(w13)) -> PerRow(cache1 as [M*TOPK, 2I])
  4. `_C.silu_and_mul`                 -> the base `SiluAndMul` Compute bound as SiluMul_v2{I} (rows=M*TOPK), the constructions' declared activation
  5. `fused_moe_kernel` (w2, weighted) -> Compute(MoeExpertGemmW_v1{E,K=I,N=H}, rows=M*TOPK):
                                          (PerRow(cache2), PerRow(ids.flat), PerRow(weights.flat), Shared(w2)) -> PerRow(cache3 as [M*TOPK, H])
  6. `_moe_C.moe_sum`                  -> Compute(MoeSum_v1{TOPK,H}, rows=M): PerRow(cache3[M,TOPK,H]) -> PerRow(out[M,H])
Everything else under the scope is aten glue (workspace slices / dtype views / empties) handled by `scope_transparents`.

Pins (κ): the router kernel's reduction geometry `VPT` (from the `topkGating<VPT, E, ...>` template instantiation),
`renormalize=False`, no bias, no padding rows; the fused kernel's non-schedule constexprs (`compute_type=bf16`, no
quantisation, `SPLIT_K=1`, `HAS_BIAS=False`, `SWAP_AB=False`) plus — for parity with the base GEMM's κ pins — its tiling
(`FUSED_MOE_PINS`).  Anything else is `Unsupported` with the observed values.

`MoEFusedExpertsFp8` is the scope with block-FP8 experts: both activation quants and the block-FP8 pair forms (its docstring).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from verity.ml.circuits.defs import bind
from verity_vllm.program.registry import act as _ACT
from verity_vllm.program.registry import moe as _MOE
from verity_vllm.program.frontend import targets as T
from verity_vllm.program.registry.quarantine.ov_moe import moe as Q

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (SiluAndMul, bound_args, find_descendants, is_int, is_row_major, op_name, scope_transparents, target_of,
                        triton_name, unsupported)
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.engine.profiles.gen_llama_patterns import GemmLaunchTuned
from verity_vllm.program.frontend.target_profile import TargetProfile

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
#: The batch-invariant GEMM a scope may launch for the router's gate (`Qwen3MoeSparseMoeBlock`: `FusedMoE(gate=...)`).
K_GATE = "matmul_kernel_persistent"

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
    #: the router's statement: None = kernel order, "ordered" = `MoeRouterTopKOrdered[Norm]_v1`, the same function (`MoEFusedExpertsOrdered`);
    #: a class attribute, not a field, so the profile's canonical form (and its fixtures) are unchanged
    router = None
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
    #: the fused kernel's A / B dtype and the scale pointers it must not carry; class attributes like `router`, which the block-FP8
    #: scope (`MoEFusedExpertsFp8`) overrides with `_numeric_pins`
    operand_dtype = "bfloat16"
    absent_ptrs = ("a_scale_ptr", "b_scale_ptr")
    #: numeric pins `_tiling` checks instead (the block-FP8 scope's SWAP_AB, which a tuned tile sets)
    tiling_keys = ()
    kernel_note = "bf16 unquantised kernel"

    def _numeric_pins(self) -> dict[str, Any]:
        return FUSED_MOE_NUMERIC_PINS

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
        gates, own = self._gate_launches(node), self._scope_ops(node)
        for r in (gates, own):
            if isinstance(r, Unsupported):
                return r
        keep = {router.id, up.id, down.id, silu.id, msum.id} | {a.id for a in aligns} | {g.id for g in gates} | {o.id for o in own}
        keep |= set().union(*(set(n.ids()) for n in (router, up, down, silu, msum, *aligns, *gates, *own)))
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
        if self.router == "ordered":
            router_def = _MOE.MoeRouterTopKOrderedNorm if self.renormalize else _MOE.MoeRouterTopKOrdered
        else:
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
        gate_c = [self._gate(node, gates[0], router, logits, ctx)] if gates else []
        if gate_c and isinstance(gate_c[0], Unsupported):
            return gate_c[0]
        parts: list[Compute | Transparent] = list(before) + gate_c + [router_c]

        # 2. alignment: a schedule, no numerical effect ----------------------------------------------------------
        for al in aligns:
            parts.append(Transparent("noop", (al.id,), hint=f"{hint}/moe_align_block_size: pair->expert-block schedule (sorted_token_ids, "
                                     "expert_ids, num_tokens_post_padded); no output word depends on it", kernels=K_ALIGN_KERNELS, pattern=self.name))

        # 3. up/gate GEMM over pairs --------------------------------------------------------------------------------
        P = M * TOPK
        ids_flat = _flat(ids, P)
        try:   # the record target's expert GEMM Definitions (`frontend.targets.moe_expert_gemm_spec`): the same selection the builder makes
            up_spec, down_spec = (T.moe_expert_gemm_spec(target_of(ctx)[0], E, K, N, routed=w) for K, N, w in ((H, 2 * I, False), (I, H, True)))
        except T.UnregisteredTarget as e:
            return unsupported(node, f"target has no registered fused-MoE binding: {e}")
        r = self._fused(node, up, ctx, expect_weighted=False, expect_top_k=TOPK, E=E, K=H, N=2 * I, P=P)
        if isinstance(r, Unsupported):
            return r
        x_up, w13, c1 = r
        if x_up.alloc != hidden.alloc or x_up.shape != (M, H) or not is_row_major(x_up):
            return unsupported(node, "up-GEMM A operand is not the [M,H] hidden_states view", a=repr(x_up), hidden=repr(hidden))
        up_c = Compute(up_spec, P,
                       (Gather(x_up, tuple(p // TOPK for p in range(P))), PerRow(ids_flat), Shared(w13)),
                       (PerRow(_as_rows(c1, P, 2 * I)),), (up.id,), hint=f"{hint}/{_root_name(ctx, w13, 'w13')}", kernels=(K_FUSED,), pattern=self.name)
        parts.append(up_c)

        # 4. silu_and_mul on the pair rows ------------------------------------------------------------------------
        silu_r = SiluAndMul().match(silu, ctx)
        if not isinstance(silu_r, Compute):
            return unsupported(node, "silu_and_mul inside the MoE scope did not resolve as SiluMul",
                               detail=getattr(silu_r, "detail", None), reason=getattr(silu_r, "reason", None))
        if silu_r.rows != P:
            return unsupported(node, "silu_and_mul rows are not the M*TOPK pair rows", rows=silu_r.rows, P=P)
        # the constructions' declared activation (`TargetProfile.MOE_CONSTRUCTIONS`: SiluMul_v2{I}, the Build's `silu_mul` kind)
        parts.append(dataclasses.replace(silu_r, spec=bind(_ACT.SiluMul, I=silu_r.spec.bindings["I"]), hint=f"{hint}/silu_and_mul", pattern=self.name))

        # 5. down GEMM over pairs, routed weight in the epilogue ---------------------------------------------------
        r = self._fused(node, down, ctx, expect_weighted=True, expect_top_k=1, E=E, K=I, N=H, P=P)
        if isinstance(r, Unsupported):
            return r
        x_dn, w2, c3 = r
        wv = bound_args(down).get("topk_weights_ptr")
        if not isinstance(wv, ViewDesc) or wv.alloc != weights.alloc or wv.offset != weights.offset or wv.shape != (M, TOPK):
            return unsupported(node, "down-GEMM routed weights are not the router's topk_weights view", topk_weights_ptr=repr(wv), weights=repr(weights))
        if x_dn.shape != (P, I) or not is_row_major(x_dn):
            return unsupported(node, "down-GEMM A operand is not the [M*TOPK, I] silu output", a=repr(x_dn))
        down_c = Compute(down_spec, P,
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

    def _gate_launches(self, node: TreeNode) -> list[TreeNode] | Unsupported:
        """[Qwen3-MoE on vLLM d9105ea80] `Qwen3MoeSparseMoeBlock` hands `FusedMoE` its `gate` and passes `router_logits=hidden_states`,
        so the router's gate GEMM launches inside the scope, ahead of topk_softmax (OLMoE computes the logits before the call)."""
        gates = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_GATE)
        if len(gates) > 1:
            return unsupported(node, f"MoE scope has more than one {K_GATE} launch (at most the router's gate GEMM)",
                               launches=[g.id for g in gates], census=moe_subtree_census(node))
        return gates

    def _scope_ops(self, node: TreeNode) -> list[TreeNode] | Unsupported:
        """Descendants a subclass resolves itself (`MoEFusedExpertsFp8`: the two activation quants), kept out of the glue walk."""
        return []

    def _gate(self, node: TreeNode, gate: TreeNode, router: TreeNode, logits: ViewDesc, ctx: StepContext) -> Compute | Unsupported:
        """The in-scope router gate GEMM, resolved by the profile's own patterns exactly as the first-match resolver would resolve it
        outside a scope (same tiling pins, same Definition), once it is shown to be the launch that writes the router's logits."""
        if gate.id > router.id:
            return unsupported(node, f"the in-scope {K_GATE} launch follows topk_softmax: it is not the router's gate GEMM", gate_id=gate.id, router_id=router.id)
        c = bound_args(gate).get("c_ptr")
        if not isinstance(c, ViewDesc) or (c.alloc, c.offset, c.shape) != (logits.alloc, logits.offset, logits.shape):
            return unsupported(node, f"the in-scope {K_GATE} launch does not write the router's gating logits", c=repr(c), logits=repr(logits))
        for pt in ctx.profile.patterns:
            if isinstance(pt, MoEFusedExperts):
                continue
            r = pt.match(gate, ctx)
            if r is None:
                continue
            if type(r) is not Compute:
                return unsupported(node, f"{getattr(pt, 'name', type(pt).__name__)} refused the router's gate GEMM inside {op_name(node.op)}",
                                   child_id=gate.id, child_reason=getattr(r, "reason", None), child_detail=getattr(r, "detail", None))
            return r
        return unsupported(node, f"no profile pattern resolves the router's gate GEMM inside {op_name(node.op)}", child_id=gate.id)

    def _tiling(self, node: TreeNode, launch: TreeNode, ctx: StepContext) -> Unsupported | None:
        """None when the launch's tiling is the profile's pinned one (`fused_pins`, warps, stages)."""
        e = launch.enter
        consts = dict(e.constexprs)
        bad = {k: (consts.get(k), v) for k, v in self.fused_pins.items() if consts.get(k) != v}
        if bad or e.num_warps != self.fused_num_warps or e.num_stages != self.fused_num_stages:
            return unsupported(node, "fused_moe_kernel tiling differs from the pinned profile",
                               mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()},
                               num_warps=e.num_warps, num_stages=e.num_stages)
        return None

    def _fused(self, node: TreeNode, launch: TreeNode, ctx: StepContext, *, expect_weighted: bool, expect_top_k: int, E: int, K: int, N: int,
               P: int):
        """Check one `fused_moe_kernel` launch against the pins and return `(a_view, b_view, c_view)` or an Unsupported."""
        e = launch.enter
        consts = dict(e.constexprs)
        bad = {k: (consts.get(k), v) for k, v in self._numeric_pins().items() if k not in self.tiling_keys and consts.get(k) != v}
        if bad:
            return unsupported(node, f"fused_moe_kernel numeric constexprs differ from the modelled {self.kernel_note}",
                               mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
        if (t := self._tiling(node, launch, ctx)) is not None:
            return t
        if consts.get("MUL_ROUTED_WEIGHT") is not expect_weighted or consts.get("top_k") != expect_top_k:
            return unsupported(node, "fused_moe_kernel launch order / role does not match (MUL_ROUTED_WEIGHT, top_k)",
                               observed={"MUL_ROUTED_WEIGHT": consts.get("MUL_ROUTED_WEIGHT"), "top_k": consts.get("top_k")},
                               expected={"MUL_ROUTED_WEIGHT": expect_weighted, "top_k": expect_top_k})
        a = bound_args(launch)
        for k in ("b_bias_ptr", *self.absent_ptrs):
            if a.get(k) is not None:
                return unsupported(node, f"fused_moe_kernel has a {k}", **{k: repr(a.get(k))})
        av, bv, cv = a.get("a_ptr"), a.get("b_ptr"), a.get("c_ptr")
        if not all(isinstance(v, ViewDesc) for v in (av, bv, cv)):
            return unsupported(node, "fused_moe_kernel pointer arguments not bound as expected", args={k: repr(v) for k, v in a.items() if k.endswith("_ptr")})
        if a.get("N") != N or a.get("K") != K or a.get("num_valid_tokens") != P:
            return unsupported(node, "fused_moe_kernel N / K / num_valid_tokens differ from the scope's geometry",
                               observed={"N": a.get("N"), "K": a.get("K"), "num_valid_tokens": a.get("num_valid_tokens")}, expected={"N": N, "K": K, "P": P})
        if bv.shape != (E, N, K) or not _contig(bv) or bv.dtype != self.operand_dtype:
            return unsupported(node, f"fused_moe_kernel B operand is not a contiguous {self.operand_dtype} [E,N,K] expert table", b=repr(bv))
        if (a.get("stride_be"), a.get("stride_bn"), a.get("stride_bk")) != (N * K, K, 1):
            return unsupported(node, "fused_moe_kernel B strides are not [E][N][K] row-major", strides=[a.get("stride_be"), a.get("stride_bn"), a.get("stride_bk")])
        if av.dtype != self.operand_dtype or av.rank != 2 or not is_row_major(av) or av.shape[1] != K:
            return unsupported(node, f"fused_moe_kernel A operand is not a row-major {self.operand_dtype} [rows,K] view", a=repr(av))
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
        if (dot := T.moe_expert_dot(target_of(ctx)[0])) is not None:
            return unsupported(node, f"padded construction on a target whose fused_moe_kernel runs the k16 step {dot.id}: MoeFfnFixedL_v1's "
                               "expert bodies run the Ampere one", dot=dot.id)
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
        keep = [p for p in parts if id(p) not in six]                 # the scope glue (before) + the alignment noop (+ gate GEMM, TP collective), in order
        first = min(router.consumed)
        head = [p for p in keep if isinstance(p, Compute) and min(p.consumed) < first]   # the in-scope gate GEMM writes the logits `one` reads
        tail = [p for p in keep if isinstance(p, Compute) and min(p.consumed) > first]   # [TP] the in-scope all-reduce stays AFTER the block's one Compute
        keep = [p for p in keep if not isinstance(p, Compute)]
        return Group(tuple(keep) + (*head, one, *tail), r.consumed, hint=hint, pattern=self.name)


@dataclass
class MoEFusedExpertsOrdered(MoEFusedExperts):
    """`moe_construction = indexed-read-ordered`: `MoEFusedExperts` with the router stated in the kernel's selection order (`MoeRouterTopKOrdered[Norm]_v1`)."""

    router = "ordered"


#: The declared constructions the fold knows (the frontend's `TargetProfile.MOE_CONSTRUCTIONS` keys); the pattern each folds with.
MOE_CONSTRUCTION_PATTERNS = {"indexed-read": MoEFusedExperts, "padded": MoEFusedExpertsPadded, "indexed-read-ordered": MoEFusedExpertsOrdered}

#: the experts class whose arithmetic the block-FP8 pair forms state (`TargetProfile.moe_backend = "triton"`)
TRITON_EXPERTS = TargetProfile.MOE_BACKEND_KERNELS["triton"]
K_QUANT = "per_token_group_fp8_quant"
#: block 128's fused SiLU and second activation quant (`triton_moe.py:474-485`)
K_SILU_QUANT = "silu_and_mul_per_block_quant"


def header_moe_experts_class(ctx: StepContext) -> str | None:
    """The experts class the record's fused-MoE layers carried (`RunHeader.profile["backends"]["moe_experts_class"]`, written by
    `engine.run_facts` from `moe_experts_class_of(model)` under a declared `moe_backend`), None when the header does not say."""
    try:
        k = ((ctx.header.profile or {}).get("backends") or {}).get("moe_experts_class")
        return k if isinstance(k, str) else None
    except (AttributeError, TypeError):
        return None


def _quant_io(q: TreeNode) -> tuple[Any, Any, Any]:
    """`(input, output_q, output_s)` of a `per_token_group_fp8_quant` call (its first three parameters), or of a
    `silu_and_mul_per_block_quant` call (`input`, `out`, `scales`)."""
    a = bound_args(q)
    if op_name(q.op) == K_SILU_QUANT:
        return a.get("input"), a.get("out"), a.get("scales")
    return tuple(a.get(n, a.get(f"arg{i}")) for i, n in enumerate(("input", "output_q", "output_s")))


def _same(u: Any, v: ViewDesc) -> bool:
    return isinstance(u, ViewDesc) and (u.alloc, u.offset, u.shape) == (v.alloc, v.offset, v.shape)


@dataclass(frozen=True)
class _Scaled:
    """One block-FP8 launch's operands: the quant that wrote A and A_scale, its bf16 input, and the e4m3 / f32 views the kernel read."""

    quant: TreeNode
    source: ViewDesc
    a: ViewDesc
    a_s: ViewDesc
    b: ViewDesc
    b_s: ViewDesc
    KB: int


@dataclass
class MoEFusedExpertsFp8(MoEFusedExperts):
    """The pairwise Group for vLLM d9105ea80's `Fp8MoEMethod` -> `TritonExperts` at block [G, G] (`group`, the rank's MoE block), whose
    activations are quantised per token group inside the scope:

      `_C.per_token_group_fp8_quant` (x)     -> Compute(Fp8GroupQuant_v1{K=H,G}, rows=M): hidden [M,H] bf16 -> (x_q e4m3, x_s [M,H/G] f32)
      `fused_moe_kernel` (w13, fp8)          -> Compute(MoeExpertFp8Gemm_v1{E,K=H,N=2I,G,KB,SWAP,DOT}, rows=M*TOPK):
                                                (Gather(x_q, p//TOPK), Gather(x_s, p//TOPK), PerRow(ids.flat), Shared(w13), Shared(w13_s))
      `_C.silu_and_mul`                      -> SiluMul_v2{I}, as above
      `_C.per_token_group_fp8_quant` (h)     -> Compute(Fp8GroupQuant_v1{K=I,G}, rows=M*TOPK): cache2 bf16 -> (h_q e4m3, h_s f32)
      `fused_moe_kernel` (w2, fp8, weighted) -> Compute(MoeExpertFp8GemmW_v1{E,K=I,N=H,G,KB,SWAP,DOT}, rows=M*TOPK):
                                                (PerRow(h_q), PerRow(h_s), PerRow(ids.flat), PerRow(weights.flat), Shared(w2), Shared(w2_s))

    The router, the alignment schedule, the in-scope gate GEMM, `moe_sum` and the TP collective are `MoEFusedExperts`'.  The launch is
    `use_fp8_w8a8` with `group_n == group_k == G`, A_scale `[rows, K/G]` and B_scale `[E, cdiv(N, G), K/G]` row-major, the tiling pins as
    above, and `BLOCK_SIZE_K` dividing G.  At block 128 vLLM fuses the SiLU with the second quant (`silu_and_mul_per_block_quant`,
    `triton_moe.py:474-485`), which this scope refuses by its shape.  DOT is the record target's e4m3 step (`targets.moe_expert_fp8_dot`);
    a target with none refuses by name.  The record must attest `TritonExperts` (`header_moe_experts_class`): vLLM's FP8 MoE oracle may
    select another experts class, whose arithmetic these Definitions do not state."""

    name = "moe.fused_experts.pairwise.fp8"
    operand_dtype = "float8_e4m3fn"
    absent_ptrs = ()
    tiling_keys = ("SWAP_AB",)
    kernel_note = "block-FP8 kernel (use_fp8_w8a8, group_n == group_k == G)"
    group: int = 0

    def _numeric_pins(self) -> dict[str, Any]:
        return {**FUSED_MOE_NUMERIC_PINS, "use_fp8_w8a8": True, "group_n": self.group, "group_k": self.group}

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or not op_name(node.op).startswith(MOE_OP_PREFIX):
            return None
        got = header_moe_experts_class(ctx)
        if got != TRITON_EXPERTS:
            return unsupported(node, f"the record does not attest {TRITON_EXPERTS} for its fused-MoE layers (RunHeader.profile.backends."
                                     f"moe_experts_class, written under a declared moe_backend); {self.name} states that kernel's arithmetic",
                               moe_experts_class=got)
        r = super().match(node, ctx)
        return self._fp8_parts(node, ctx, r) if isinstance(r, Group) else r

    def _quants(self, node: TreeNode) -> list[TreeNode]:
        return find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == K_QUANT)

    def _scope_ops(self, node: TreeNode) -> list[TreeNode] | Unsupported:
        qs = self._quants(node)
        if len(qs) != 2:
            return unsupported(node, f"block-FP8 MoE scope does not have its two {K_QUANT} calls (the input's, then the SiLU output's)",
                               count=len(qs), census=moe_subtree_census(node))
        return qs

    def _tiling(self, node: TreeNode, launch: TreeNode, ctx: StepContext) -> Unsupported | None:
        """A batch-invariant target's pinned tile (`MoEFusedExperts._tiling`) with SWAP_AB False; otherwise the tuned tile the Build
        accepts (`targets.moe_fp8_tuned_tile_refusal`: cc 9.0, SWAP_AB = `enable_swap_ab`, a wgmma tile)."""
        tp = target_of(ctx)[0]
        e = launch.enter
        c = e.constexprs
        if tp.batch_invariant:
            if c.get("SWAP_AB") is not False:
                return unsupported(node, f"fused_moe_kernel numeric constexprs differ from the modelled {self.kernel_note}",
                                   mismatches={"SWAP_AB": {"observed": c.get("SWAP_AB"), "pinned": False}})
            return super()._tiling(node, launch, ctx)
        why = T.moe_fp8_tuned_tile_refusal(tp, c.get("BLOCK_SIZE_M"), c.get("BLOCK_SIZE_N"), c.get("BLOCK_SIZE_K"), e.num_warps, c.get("SWAP_AB"))
        if why is not None:
            return unsupported(node, f"fused_moe_kernel tuned tile: {why}", child_id=launch.id,
                               tile={k: c.get(k) for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SWAP_AB")},
                               num_warps=e.num_warps, num_stages=e.num_stages)
        return None

    def _fused(self, node: TreeNode, launch: TreeNode, ctx: StepContext, *, expect_weighted: bool, expect_top_k: int, E: int, K: int, N: int,
               P: int):
        """`MoEFusedExperts._fused` under the FP8 pins, then the scales (`_scaled`); returns `(the bf16 input of the quant that wrote A, b,
        c)`, so `match`'s A checks read hidden_states (w13) and the SiLU output (w2) through their quants."""
        r = super()._fused(node, launch, ctx, expect_weighted=expect_weighted, expect_top_k=expect_top_k, E=E, K=K, N=N, P=P)
        if isinstance(r, Unsupported):
            return r
        s = self._scaled(node, launch, E=E, K=K, N=N)
        return s if isinstance(s, Unsupported) else (s.source, r[1], r[2])

    def _scaled(self, node: TreeNode, launch: TreeNode, *, E: int, K: int, N: int) -> _Scaled | Unsupported:
        G = self.group
        a = bound_args(launch)
        av, asv, bv, bsv = a.get("a_ptr"), a.get("a_scale_ptr"), a.get("b_ptr"), a.get("b_scale_ptr")
        if not all(isinstance(v, ViewDesc) for v in (av, asv, bv, bsv)):
            return unsupported(node, "block-FP8 fused_moe_kernel without its a_scale / b_scale views", a_scale_ptr=repr(asv), b_scale_ptr=repr(bsv))
        KB = launch.enter.constexprs.get("BLOCK_SIZE_K")
        if K % G or not isinstance(KB, int) or KB <= 0 or G % KB:
            return unsupported(node, "the block G does not tile K, or BLOCK_SIZE_K does not divide G", K=K, G=G, BLOCK_SIZE_K=KB)
        KG, NB = K // G, -(-N // G)
        a_strides = (a.get("stride_asm"), a.get("stride_ask"))
        if asv.dtype != "float32" or asv.shape != (av.shape[0], KG) or a_strides != (KG, 1):
            return unsupported(node, "a_scale is not the row-major f32 [rows, K/G] group scale of A", a_scale=repr(asv), strides=list(a_strides),
                               expected=[av.shape[0], KG])
        b_strides = (a.get("stride_bse"), a.get("stride_bsn"), a.get("stride_bsk"))
        if bsv.dtype != "float32" or bsv.shape != (E, NB, KG) or not _contig(bsv) or b_strides != (NB * KG, KG, 1):
            return unsupported(node, "b_scale is not the contiguous f32 [E, cdiv(N, G), K/G] block scale of B", b_scale=repr(bsv),
                               strides=list(b_strides), expected=[E, NB, KG])
        writers = [q for q in self._quants(node) if _same(_quant_io(q)[1], av) and _same(_quant_io(q)[2], asv)]
        if len(writers) != 1 or not isinstance(src := _quant_io(writers[0])[0], ViewDesc):
            return unsupported(node, f"no single {K_QUANT} in the scope writes this launch's A and a_scale", a=repr(av), a_scale=repr(asv),
                               writers=[q.id for q in writers])
        return _Scaled(writers[0], src, av, asv, bv, bsv, KB)

    def _fp8_parts(self, node: TreeNode, ctx: StepContext, r: Group) -> Group | Unsupported:
        """`r` with each expert GEMM replaced by its activation quant, then its block-FP8 pair form."""
        from verity_vllm.observe.fold.patterns.fp8 import PerTokenGroupFp8Quant  # noqa: PLC0415  (registry.fp8 loads with an FP8 scope only)
        C, G = self.config, self.group
        E, TOPK, H, I = C["NUM_EXPERTS"], C["TOPK"], C["H"], C["I"]
        up, down = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_FUSED)
        parts = list(r.parts)
        at = {p.consumed: i for i, p in enumerate(parts) if isinstance(p, Compute)}
        silu = next(p for p in parts if isinstance(p, Compute) and p.spec.name == "SiluMul")
        tp = target_of(ctx)[0]
        repl: dict[int, tuple[Compute, Compute]] = {}
        for launch, K, N, routed in ((up, H, 2 * I, False), (down, I, H, True)):
            i = at[(launch.id,)]
            base, P = parts[i], parts[i].rows
            s = self._scaled(node, launch, E=E, K=K, N=N)
            if isinstance(s, Unsupported):
                return s
            q = PerTokenGroupFp8Quant(group=G).match(s.quant, ctx)
            if not isinstance(q, Compute):
                return unsupported(node, f"a {K_QUANT} inside the MoE scope did not resolve as Fp8GroupQuant_v1", child_id=s.quant.id,
                                   child_reason=getattr(q, "reason", None), child_detail=getattr(q, "detail", None))
            if (q.operands[0].view, q.outputs[0].view, q.outputs[1].view) != (s.source, s.a, s.a_s):
                return unsupported(node, f"the {K_QUANT} the launch reads does not resolve to its own views", child_id=s.quant.id)
            if routed and not (_same(silu.outputs[0].view, s.source) and max(silu.consumed) < s.quant.id < launch.id):
                return unsupported(node, "the w2 launch's quant does not read the SiLU output between the SiLU and the launch",
                                   quant_input=repr(s.source), silu_output=repr(silu.outputs[0].view))
            if not routed and not s.quant.id < launch.id:
                return unsupported(node, "the w13 launch's quant runs after the launch", quant_id=s.quant.id, launch_id=launch.id)
            try:
                spec = T.moe_expert_fp8_gemm_spec(tp, E, K, N, G, s.KB, bool(launch.enter.constexprs.get("SWAP_AB")), routed=routed)
            except T.UnregisteredTarget as e:
                return unsupported(node, f"target has no registered block-FP8 fused-MoE binding: {e}")
            if routed:
                ops = (PerRow(s.a), PerRow(s.a_s), base.operands[1], base.operands[2], Shared(s.b), Shared(s.b_s))
            else:
                rows = tuple(p // TOPK for p in range(P))
                ops = (Gather(s.a, rows), Gather(s.a_s, rows), base.operands[1], Shared(s.b), Shared(s.b_s))
            w = "w2" if routed else "w13"
            roots = (None,) * (len(ops) - 2) + (_root_name(ctx, s.b, w), _root_name(ctx, s.b_s, f"{w}_ws"))
            quant = dataclasses.replace(q, hint=f"{r.hint}/{w}_act_quant", attribution=() if routed else q.attribution, pattern=self.name)
            gemm = Compute(spec, P, ops, base.outputs, base.consumed, hint=base.hint, kernels=base.kernels, pattern=self.name, roots=roots)
            repl[i] = (quant, gemm)
        out = [x for i, p in enumerate(parts) for x in repl.get(i, (p,))]
        return Group(tuple(out), r.consumed, hint=r.hint, pattern=self.name)


@dataclass
class MoEFusedExpertsFp8Ordered(MoEFusedExpertsFp8):
    """`moe_construction = indexed-read-ordered` with block-FP8 experts: the router in the kernel's selection order."""

    router = "ordered"


#: The declared constructions the block-FP8 scope folds; `padded` states bf16 expert bodies and has no FP8 form.
MOE_FP8_CONSTRUCTION_PATTERNS = {"indexed-read": MoEFusedExpertsFp8, "indexed-read-ordered": MoEFusedExpertsFp8Ordered}


def ov_moe_patterns(config: dict[str, Any], *, gemm_constexprs: dict[str, Any] | None = None, gemm_num_sms: int | None = None,
                    moe_construction: str | None = None) -> list:
    """The MoE architecture's patterns, placed BEFORE the base list (the tuned-GEMM specialization must see `o_proj` first; the MoE
    scope pattern matches the custom op no base pattern knows).  `moe_construction` (None = `TargetProfile.DEFAULT_MOE_CONSTRUCTION`)
    selects the scope pattern: the workload's DECLARED construction, never inferred (R17-42 route A (4)).  [P6] the tuned GEMM is the
    target-agnostic `gen_llama_patterns.GemmLaunchTuned` (the record target's table; `gemm_constexprs` is accepted and unused: the
    delegate `GemmLaunch` pins the target's tiling)."""
    key = moe_construction or TargetProfile.DEFAULT_MOE_CONSTRUCTION
    if key not in MOE_CONSTRUCTION_PATTERNS:
        raise ValueError(f"moe_construction {moe_construction!r} is not one of {sorted(MOE_CONSTRUCTION_PATTERNS)}")
    tuned = GemmLaunchTuned() if gemm_num_sms is None else GemmLaunchTuned(num_sms=gemm_num_sms)
    return [tuned, MOE_CONSTRUCTION_PATTERNS[key](config)]
