"""The MXFP4 Marlin shared-expert MoE scope: `vllm.moe_forward_shared` as DeepSeek-V4-Pro's H200 TP rank runs it (vLLM d9105ea80,
`DeepseekV4MoE` with `Mxfp4MoEMethod` on the MARLIN backend -> `fused_marlin_moe`, eager, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`), folded
to the Build's rules in `_forward_impl`'s order:

  the gate                                     -> GLM-5.3's (`gen_shared_moe_patterns.scope_gate`), its weight beside tid2eid or the bias
  the shared expert                            -> two dynamic block-FP8 linears around SiluMulClamp_v1{I=SHARED_I,LIMIT}
  a hash layer's router (layer < HASH_LAYERS):
    `_moe_C.topk_softplus_sqrt(w!, ids!, tei!, logits, True, RSF, None, input_ids, tid2eid, is_padding)`
                                               -> GatherI32x{V}_v1 of input_ids[t] and tid2eid's column k, for each token t then slot
                                                  k, then MoeRouterHash_v1{E,SCALE} per token (`TopkSoftplusSqrtHashRule`); tei is dead,
                                                  and is_padding, when given, is the step's False prefix (`transparent.SCRATCH_INT_FILLS`)
  a scored layer's: `_dsv4_topk_kernel` (DSV4_TOPK_PINS) -> MoeRouterSqrtSoftplus_v1{E,SCALE} per token (`Dsv4TopkRule`)
  `_moe_C.moe_align_block_size(ids, E, BM, sorted!, experts!, ntpp!)`
                                               -> MoeAlignRanks_v1{E,P,BM} over the ids, the request's `schedule` witness (a key per
                                                  pair) and the pair indices as constants, then MoeAlignBlocks_v1{P,BM} (`MarlinMoeAlignRule`):
                                                  (blk [P], par), which no served buffer holds, in fold-private scratch
  `_moe_C.moe_wna16_marlin_gemm` (w13)         -> MarlinMoeExpertGemm_v1{E,K=H,N=2I,TK,TN,GRID,DOT} per pair (`MarlinMoeGemmRule`)
  `_C.silu_and_mul_with_clamp`                 -> SiluMulClamp_v1{I,LIMIT} per pair
  `_moe_C.moe_wna16_marlin_gemm` (w2)          -> MarlinMoeExpertGemmW_v1{E,K=I,N=H,...} per pair, the router weight multiplied in
  `_moe_C.moe_sum`                             -> MoeSum_v1{TOPK,H} per token, the op's second output

The experts read their weights Record fields whole (`WeightField`): what Marlin reads is `process_weights_after_loading`'s repacked
codes and permuted scales, no word of which is a leaf of its field.  The (TK, TN, GRID) are `vllm_marlin_moe.served_config`'s at the
launch's tokens and BM `fused_marlin_moe`'s.  One request a step only: `moe_align_block_size` sorts every pair of the launch, so under a
batch a request's blocks read the other requests' routing, which the launch-level schedule unit states.
"""

from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from verity_vllm.program.frontend.rules.vllm_bindings.pins import DSV4_TOPK_PINS
from verity_vllm.program.frontend.rules.vllm_marlin_moe import served_config
from verity.ml.definitions.numerics import marlin_moe_rows as MR
from verity.ml.definitions.registry import marlin_moe as MM

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (SCRATCH_INT_FILLS, SiluAndMulWithClamp, as_float, bound_args, find_descendants, is_row_major,
                                               op_name, target_of, triton_name, unsupported, vocab_kind)
from verity_vllm.observe.fold.resolver import (Compute, Const, Gather, PerRow, Pieces, Resolution, Shared, StepContext, Unsupported, WeightField,
                                               Witness, scratch_view)
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.observe.fold.profiles.gen_ov_moe_patterns import K_ALIGN_KERNELS, K_SUM, _contig, _flat, _same
from verity_vllm.observe.fold.profiles.gen_shared_moe_patterns import (K_DYN_MM, SHARED_OP, _weight_root, scope_gate, scope_group, scope_hidden,
                                                                 shared_expert)

K_HASH_ROUTER = "topk_softplus_sqrt"
K_TOPK = "_dsv4_topk_kernel"
K_CLAMP = "silu_and_mul_with_clamp"
K_MARLIN = "moe_wna16_marlin_gemm"
#: `scalar_types.float4_e2m1f.id` (exponent 2, mantissa 1, signed, finite only), the b_type_id `fused_marlin_moe` passes for MXFP4
MXFP4_TYPE_ID = 562949953487106
#: `ops.moe_wna16_marlin_gemm`'s arguments `fused_marlin_moe` passes as None on W4A16 float4_e2m1f experts without bias, act order or
#: zero points, and the flags it passes there; anything else is another kernel instantiation
MARLIN_NONE = ("b_bias_or_none", "a_scales", "global_scale", "b_zeros_or_none", "g_idx_or_none", "perm_or_none")
MARLIN_FLAGS = {"b_type_id": MXFP4_TYPE_ID, "is_full_k": True, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False,
                "thread_k": -1, "thread_n": -1, "blocks_per_sm": -1}
SCHEDULE = "schedule"
LAYER = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class _Scope:
    gate: TreeNode | None
    mms: tuple[TreeNode, TreeNode]
    silu: TreeNode
    router: TreeNode
    align: TreeNode
    up: TreeNode
    rsilu: TreeNode
    down: TreeNode
    msum: TreeNode


@dataclass
class MoEMarlinSharedExperts:
    """The module docstring's Group.  `config` carries NUM_EXPERTS, TOPK, H, I, SHARED_I, HASH_LAYERS, VOCAB, ROUTED_SCALING_FACTOR and
    SWIGLU_LIMIT; `gate_spec` as `MoEFusedSharedExpertsFp8`'s."""

    config: dict
    gate_spec: Callable[[int, int, int, str], tuple[Any, tuple[str, ...]] | None] | None = None
    name = "moe.marlin_shared_experts.mxfp4"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != SHARED_OP:
            return None
        C = self.config
        ints, floats = ("NUM_EXPERTS", "TOPK", "H", "I", "SHARED_I", "HASH_LAYERS", "VOCAB"), ("ROUTED_SCALING_FACTOR", "SWIGLU_LIMIT")
        if not all(isinstance(C.get(k), int) for k in ints) or any(as_float(C.get(k)) is None for k in floats):
            return unsupported(node, f"profile config lacks {' / '.join(ints + floats)}", config={k: C.get(k) for k in ints + floats})
        io = scope_hidden(node, ctx, C["H"])
        if isinstance(io, Unsupported):
            return io
        hidden, outs = io
        reqs = sorted({r for r, _ in ctx.start.rows})
        if len(reqs) != 1:
            return unsupported(node, "the launch routes several requests' tokens: moe_align_block_size sorts every pair of the launch, so a "
                                     "request's blocks read the others' routing (the launch-level schedule unit)", requests=reqs)
        layer = LAYER.match(f"{node.enter.module or ''}.")
        if layer is None:
            return unsupported(node, f"{SHARED_OP} is not under a decoder layer's module", module=node.enter.module)
        sc = self._census(node)
        if isinstance(sc, Unsupported):
            return sc
        hint = node.enter.module
        hashed = int(layer[1]) < C["HASH_LAYERS"]
        if hashed != (not sc.router.is_triton):
            return unsupported(node, f"layer {layer[1]}'s router is not its kind: {K_HASH_ROUTER} below HASH_LAYERS={C['HASH_LAYERS']}, "
                                     f"{K_TOPK} from it on", router=triton_name(sc.router) if sc.router.is_triton else sc.router.op)
        router = self._hash_router(node, ctx, sc, hidden, hint) if hashed else self._scored_router(node, ctx, sc, hidden, hint)
        if isinstance(router, Unsupported):
            return router
        parts, (scores, beside, weights, ids) = router
        shared = shared_expert(node, ctx, sc.mms, sc.silu, SiluAndMulWithClamp(), {"I": C["SHARED_I"], "LIMIT": float(C["SWIGLU_LIMIT"])},
                               hidden, outs[0], hint, C["H"], self.name)
        if isinstance(shared, Unsupported):
            return shared
        gate = scope_gate(node, ctx, sc.gate, sc.mms[0], op_name(sc.router.op) if not sc.router.is_triton else K_TOPK, scores, beside, hidden,
                          hint, C["NUM_EXPERTS"], C["H"], self.gate_spec, self.name)
        if isinstance(gate, Unsupported):
            return gate
        routed = self._routed(node, ctx, sc, hidden, outs, weights, ids, reqs[0], hint)
        if isinstance(routed, Unsupported):
            return routed
        kernels = [*sc.mms, sc.silu, sc.router, sc.align, sc.up, sc.rsilu, sc.down, sc.msum] + ([sc.gate] if sc.gate is not None else [])
        return scope_group(node, ctx, kernels, [gate, *shared, *parts, *routed], hint, self.name)

    def _census(self, node: TreeNode) -> _Scope | Unsupported:
        def ops(name: str) -> list[TreeNode]:
            return sorted(find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == name), key=lambda n: n.id)

        mms, silus, gates, hashed, aligns, gemms, sums = (ops(k) for k in (K_DYN_MM, K_CLAMP, "mm", K_HASH_ROUTER, "moe_align_block_size",
                                                                            K_MARLIN, "moe_sum"))
        scored = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_TOPK)
        counts = {K_DYN_MM: len(mms), K_CLAMP: len(silus), "mm": len(gates), "router": len(hashed) + len(scored),
                  "moe_align_block_size": len(aligns), K_MARLIN: len(gemms), "moe_sum": len(sums)}
        want = {K_DYN_MM: 2, K_CLAMP: 2, "router": 1, "moe_align_block_size": 1, K_MARLIN: 2, "moe_sum": 1}
        if any(counts[k] != v for k, v in want.items()) or counts["mm"] > 1:
            return unsupported(node, f"{SHARED_OP} scope does not have the MXFP4 Marlin shared-expert shape (2 {K_DYN_MM}, 2 {K_CLAMP}, <=1 mm, "
                                     f"1 {K_HASH_ROUTER} or {K_TOPK}, 1 align, 2 {K_MARLIN}, 1 moe_sum)", counts=counts)
        sc = _Scope(gates[0] if gates else None, (mms[0], mms[1]), silus[0], (hashed + scored)[0], aligns[0], gemms[0], silus[1], gemms[1],
                    sums[0])
        order = [*([sc.gate] if sc.gate is not None else []), *sc.mms[:1], sc.silu, sc.mms[1], sc.router, sc.align, sc.up, sc.rsilu, sc.down,
                 sc.msum]
        if [n.id for n in order] != sorted(n.id for n in order):
            return unsupported(node, f"{SHARED_OP} scope runs out of order (gate, shared expert, router, align, w13, activation, w2, moe_sum)",
                               ids={triton_name(n) if n.is_triton else op_name(n.op): n.id for n in order})
        return sc

    def _hash_router(self, node, ctx, sc: _Scope, hidden: ViewDesc, hint: str):
        """`topk_softplus_sqrt` with a token table: its parts and (scores, the table's name, weights, ids)."""
        C = self.config
        E, TOPK, V, M = C["NUM_EXPERTS"], C["TOPK"], C["VOCAB"], hidden.shape[0]
        ra = bound_args(sc.router)
        w, ids, tei, logits, toks, table = (ra.get(k) for k in ("topk_weights", "topk_indices", "token_expert_indices", "gating_output",
                                                                "input_ids", "tid2eid"))
        if not all(isinstance(v, ViewDesc) for v in (w, ids, tei, logits, toks, table)):
            return unsupported(node, f"{K_HASH_ROUTER} arguments not bound as expected", args={k: repr(v) for k, v in ra.items()})
        rsf = as_float(ra.get("routed_scaling_factor"))
        if ra.get("renormalize") is not True or rsf != float(C["ROUTED_SCALING_FACTOR"]) or ra.get("bias") is not None \
                or ra.get("bias_vl") is not None or ra.get("image_sentinel_lo", 0):
            return unsupported(node, f"{K_HASH_ROUTER} is not the renormalised text-token table route at the profile's routed_scaling_factor "
                                     "(MoeRouterHash_v1)", renormalize=ra.get("renormalize"), routed_scaling_factor=rsf,
                               bias=repr(ra.get("bias")), bias_vl=repr(ra.get("bias_vl")), image_sentinel_lo=ra.get("image_sentinel_lo"))
        if logits.shape != (M, E) or logits.dtype != "float32" or not is_row_major(logits):
            return unsupported(node, f"{K_HASH_ROUTER} gating_output is not a row-major f32 [M,E] view", logits=repr(logits))
        iv = ctx.start.input_ids_view
        if iv is None or (toks.alloc, toks.offset, toks.shape, toks.dtype) != (iv.alloc, iv.offset, (M,), "int32") or not _contig(toks):
            return unsupported(node, f"{K_HASH_ROUTER} input_ids are not the step's i32 input_ids view", input_ids=repr(toks), step=repr(iv))
        tname = _weight_root(ctx, table)
        if table.shape != (V, TOPK) or table.dtype != "int32" or not _contig(table) or tname is None:
            return unsupported(node, f"{K_HASH_ROUTER} tid2eid is not the i32 [VOCAB, TOPK] token table weight", tid2eid=repr(table))
        for nm, v, dt in (("topk_weights", w, "float32"), ("topk_indices", ids, "int32"), ("token_expert_indices", tei, "int32")):
            if v.shape != (M, TOPK) or v.dtype != dt or not _contig(v):
                return unsupported(node, f"{K_HASH_ROUTER} {nm} is not a contiguous {dt} [M,TOPK] view", **{nm: repr(v)})
        pad = self._padding(node, ctx, ra.get("is_padding"), M)
        if isinstance(pad, Unsupported):
            return pad
        P = M * TOPK
        cols = ViewDesc(table.alloc, table.offset, (TOPK, V), (1, TOPK), "int32", table.device)
        try:
            gather, route = vocab_kind(ctx, "gather_i32", V=V), vocab_kind(ctx, "moe_router_hash", E=E, SCALE=rsf)
        except ValueError as e:
            return unsupported(node, str(e))
        at, kern = sc.router.id, ("softplus_sqrt", "SoftplusSqrt")
        parts = [(at, Compute(gather, P, (Gather(toks, tuple(p // TOPK for p in range(P))), Gather(cols, tuple(p % TOPK for p in range(P)))),
                              (PerRow(_flat(ids, P)),), (at,), hint=f"{hint}/router/ids", kernels=kern, pattern=self.name, roots=(None, tname))),
                 (at, Compute(route, M, (PerRow(logits), PerRow(ids)), (PerRow(w),), (at,), hint=f"{hint}/router", kernels=kern,
                              attribution=tuple(ctx.start.rows), pattern=self.name))]
        return parts, (logits, tname, w, ids)

    @staticmethod
    def _padding(node, ctx: StepContext, pad: Any, M: int) -> None | Unsupported:
        """None, or a bool [M] view whose last integer fill this step covered it with False: no token is padding."""
        if pad is None:
            return None
        if not isinstance(pad, ViewDesc) or pad.shape != (M,) or pad.dtype != "bool" or not _contig(pad):
            return unsupported(node, f"{K_HASH_ROUTER} is_padding is not a contiguous bool [M] view", is_padding=repr(pad))
        lo, hi = pad.offset, pad.offset + M
        fills = [(v, x) for v, x in ctx.scratch.get(SCRATCH_INT_FILLS, {}).get(pad.alloc, []) if v.numel and v.offset < hi and lo < v.offset + v.numel]
        if not fills or not (_contig(fills[-1][0]) and fills[-1][0].offset <= lo and hi <= fills[-1][0].offset + fills[-1][0].numel
                             and fills[-1][1] is False):
            return unsupported(node, f"{K_HASH_ROUTER} is_padding is not the step's False prefix (the runner's fill_ of its token rows)",
                               is_padding=repr(pad), fills=[(repr(v), x) for v, x in fills])
        return None

    def _scored_router(self, node, ctx, sc: _Scope, hidden: ViewDesc, hint: str):
        """`_dsv4_topk_kernel[(M,)]` at its pins: its part and (scores, the bias's name, weights, ids)."""
        C = self.config
        E, TOPK, M = C["NUM_EXPERTS"], C["TOPK"], hidden.shape[0]
        e = sc.router.enter
        got = {"kernel": e.fn, "src_sha256": e.src_sha, "constexprs": {k: e.constexprs.get(k) for k in DSV4_TOPK_PINS["constexprs"]},
               "num_warps": e.num_warps}
        if got != DSV4_TOPK_PINS:
            return unsupported(node, f"{K_TOPK} differs from its pins (DSV4_TOPK_PINS)", observed=got, pinned=DSV4_TOPK_PINS)
        ra = bound_args(sc.router)
        x, b, w, ids = (ra.get(k) for k in ("gating_output_ptr", "correction_bias_ptr", "topk_weights_ptr", "topk_ids_ptr"))
        if not all(isinstance(v, ViewDesc) for v in (x, b, w, ids)):
            return unsupported(node, f"{K_TOPK} arguments not bound as expected", args={k: repr(v) for k, v in ra.items()})
        rsf = as_float(ra.get("routed_scaling_factor"))
        if rsf != float(C["ROUTED_SCALING_FACTOR"]) or ra.get("bias_vl_ptr") is not None or e.grid != [M]:
            return unsupported(node, f"{K_TOPK} is not the (M,) launch at the profile's routed_scaling_factor without the vision bias",
                               routed_scaling_factor=rsf, bias_vl_ptr=repr(ra.get("bias_vl_ptr")), grid=e.grid)
        bname = _weight_root(ctx, b)
        if x.shape != (M, E) or x.dtype != "float32" or x.strides != (E, 1) or b.shape != (E,) or b.dtype != "float32" or b.strides != (1,) \
                or bname is None:
            return unsupported(node, f"{K_TOPK} gating_output and correction_bias are not packed f32 [M,E] and the [E] bias weight",
                               gating_output=repr(x), correction_bias=repr(b))
        for nm, v, dt in (("topk_weights_ptr", w, "float32"), ("topk_ids_ptr", ids, "int32")):
            if v.shape != (M, TOPK) or v.dtype != dt or v.strides != (TOPK, 1):
                return unsupported(node, f"{K_TOPK} {nm} is not a packed {dt} [M,TOPK] view", **{nm: repr(v)})
        try:
            spec = vocab_kind(ctx, "moe_router_sqrtsoftplus", E=E, SCALE=rsf)
        except ValueError as e:
            return unsupported(node, str(e))
        at = sc.router.id
        return [(at, Compute(spec, M, (PerRow(x), Shared(b)), (PerRow(w), PerRow(ids)), (at,), hint=f"{hint}/router", kernels=(K_TOPK,),
                             attribution=tuple(ctx.start.rows), pattern=self.name, roots=(None, bname)))], (x, bname, w, ids)

    def _routed(self, node, ctx, sc: _Scope, hidden: ViewDesc, outs, weights: ViewDesc, ids: ViewDesc, req: str,
                hint: str) -> list[tuple[int, Any]] | Unsupported:
        """The align's schedule, w13, the clamped activation, w2 and moe_sum over the (token, slot) pairs."""
        C = self.config
        E, TOPK, H, I, M = C["NUM_EXPERTS"], C["TOPK"], C["H"], C["I"], hidden.shape[0]
        P, tp = M * TOPK, target_of(ctx)[0]
        aa = bound_args(sc.align)
        bufs = tuple(aa.get(k) for k in ("sorted_token_ids", "experts_ids", "num_tokens_post_pad"))
        BM = aa.get("block_size")
        if not _same(aa.get("topk_ids"), ids) or aa.get("num_experts") != E or aa.get("maybe_expert_map") is not None \
                or not all(isinstance(v, ViewDesc) for v in bufs):
            return unsupported(node, "moe_align_block_size does not sort the router's ids over E experts without an expert map",
                               args={k: repr(v) for k, v in aa.items()})
        if BM != MR.moe_block(M, TOPK, E):
            return unsupported(node, f"moe_align_block_size's block {BM!r} is not fused_marlin_moe's at {M} tokens ({MR.moe_block(M, TOPK, E)})")
        ids_flat = _flat(ids, P)
        rank, start, blk, par = (scratch_view(ids_flat, sc.align.id, k, n) for k, n in enumerate((P, P, P, 1)))
        try:
            ranks, blocks = vocab_kind(ctx, "marlin_moe_rank", E=E, P=P, BM=BM), vocab_kind(ctx, "marlin_moe_align", P=P, BM=BM)
        except ValueError as e:
            return unsupported(node, str(e))
        at = sc.align.id
        out: list[tuple[int, Any]] = [
            (at, Compute(ranks, 1, (Shared(ids_flat), Witness(SCHEDULE, (req,)), Pieces((tuple(Const(32, p, "i32", shared=True) for p in range(P)),))),
                         (Shared(rank), Shared(start)), (at,), hint=f"{hint}/moe_align_block_size/rank", kernels=K_ALIGN_KERNELS, pattern=self.name)),
            (at, Compute(blocks, 1, (Shared(ids_flat), Shared(rank), Shared(start)), (Shared(blk), Shared(par)), (at,),
                         hint=f"{hint}/moe_align_block_size", kernels=K_ALIGN_KERNELS, pattern=self.name))]
        prev = hidden
        for launch, K, N, routed in ((sc.up, H, 2 * I, False), (sc.down, I, H, True)):
            if routed:
                act = SiluAndMulWithClamp().match(sc.rsilu, ctx)
                want = {"I": I, "LIMIT": float(C["SWIGLU_LIMIT"])}
                if not isinstance(act, Compute) or act.rows != P or any(act.spec.bindings.get(k) != v for k, v in want.items()):
                    return unsupported(node, f"the routed {K_CLAMP} did not resolve as SiluMulClamp{want} on the pair rows", child_id=sc.rsilu.id,
                                       child_reason=getattr(act, "reason", None))
                x = act.operands[0].view
                if (x.alloc, x.offset) != (prev.alloc, prev.offset):
                    return unsupported(node, f"the routed {K_CLAMP} does not read the w13 output as [P, 2I]", input=repr(x), w13=repr(prev))
                out.append((sc.rsilu.id, Compute(act.spec, P, act.operands, act.outputs, act.consumed, hint=f"{hint}/{K_CLAMP}",
                                                 kernels=act.kernels, pattern=self.name)))
                prev = act.outputs[0].view
            c = self._gemm(node, ctx, launch, prev, K, N, routed, bufs, BM, weights, P, tp)
            if isinstance(c, Unsupported):
                return c
            spec, a, y, wq, ws = c
            ops = (PerRow(a) if routed else Gather(a, tuple(p // TOPK for p in range(P))), PerRow(ids_flat),
                   *((PerRow(_flat(weights, P)),) if routed else ()), PerRow(blk), Shared(par), WeightField((wq,)), WeightField((ws,)))
            out.append((launch.id, Compute(spec, P, ops, (PerRow(y),), (launch.id,), hint=f"{hint}/{wq}", kernels=("Marlin",), pattern=self.name,
                                           roots=(None,) * (len(ops) - 2) + (wq, ws))))
            prev = y
        ma = bound_args(sc.msum)
        y, o = ma.get("input"), ma.get("output")
        if not isinstance(y, ViewDesc) or not isinstance(o, ViewDesc) or ma.get("topk_ids") is not None or ma.get("expert_map") is not None:
            return unsupported(node, "moe_sum is not the modelled (input, output) kernel", args={k: repr(v) for k, v in ma.items()})
        if y.shape != (M, TOPK, H) or not _contig(y) or y.dtype != "bfloat16" or (y.alloc, y.offset) != (prev.alloc, prev.offset):
            return unsupported(node, "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output", input=repr(y), w2=repr(prev))
        if o.shape != (M, H) or not is_row_major(o) or o.dtype != "bfloat16" or o.alloc != outs[1].alloc:
            return unsupported(node, "moe_sum does not write the op's second output as a row-major bf16 [M,H] view", output=repr(o),
                               routed=repr(outs[1]))
        try:
            spec = vocab_kind(ctx, "moe_sum", TOPK=TOPK, H=H)
        except ValueError as e:
            return unsupported(node, str(e))
        out.append((sc.msum.id, Compute(spec, M, (PerRow(y),), (PerRow(o),), (sc.msum.id,), hint=f"{hint}/moe_sum", kernels=(K_SUM,),
                                        attribution=tuple(ctx.start.rows), pattern=self.name)))
        return out

    def _gemm(self, node, ctx, launch: TreeNode, x: ViewDesc, K: int, N: int, routed: bool, bufs: tuple, BM: int, weights: ViewDesc, P: int,
              tp) -> tuple[Any, ViewDesc, ViewDesc, str, str] | Unsupported:
        """One `moe_wna16_marlin_gemm` launch over the align's schedule: its Definition, A, C [P, N] and the weight and scale fields."""
        C = self.config
        E, TOPK, M = C["NUM_EXPERTS"], C["TOPK"], P // C["TOPK"]
        what, a = ("w2" if routed else "w13"), bound_args(launch)
        top_k = 1 if routed else TOPK
        flags = {k: a.get(k) for k in MARLIN_FLAGS} | {"top_k": a.get("top_k"), "mul_topk_weights": a.get("mul_topk_weights"),
                                                     "moe_block_size": a.get("moe_block_size"), "size_m": a.get("size_m"),
                                                     "size_n": a.get("size_n"), "size_k": a.get("size_k")}
        want = MARLIN_FLAGS | {"top_k": top_k, "mul_topk_weights": routed, "moe_block_size": BM, "size_m": P // top_k, "size_n": N, "size_k": K}
        if flags != want or any(a.get(k) is not None for k in MARLIN_NONE):
            return unsupported(node, f"the {what} {K_MARLIN} launch is not fused_marlin_moe's W4A16 float4_e2m1f call", observed=flags,
                               expected=want, none={k: repr(a.get(k)) for k in MARLIN_NONE if a.get(k) is not None})
        if tuple(a.get(k) for k in ("sorted_token_ids", "expert_ids", "num_tokens_past_padded")) != bufs:
            return unsupported(node, f"the {what} launch does not read moe_align_block_size's schedule", align=[repr(v) for v in bufs])
        A, y, wv, sv, g = a.get("a"), (launch.output_views() or [None])[0], a.get("b_q_weight"), a.get("b_scales"), a.get("topk_weights")
        if not all(isinstance(v, ViewDesc) for v in (A, y, wv, sv, g)) or not _same(A, x) or not _same(g, weights):
            return unsupported(node, f"the {what} launch does not read its input and the router's weights", a=repr(A), expected=repr(x),
                               topk_weights=repr(g))
        if y.shape != (P, N) or y.dtype != "bfloat16" or not is_row_major(y) or not _same(a.get("c_or_none"), y):
            return unsupported(node, f"the {what} launch does not write a row-major bf16 [P, N] C", c=repr(y))
        names = [f"{node.enter.module}.routed_experts.{what}_weight{s}" for s in ("", "_scale")]
        served = [(wv, (E, K // 16, 2 * N), "int32"), (sv, (E, K // MM.GROUP, N), "float8_e8m0fnu")]
        if [_weight_root(ctx, v) for v, _, _ in served] != names or any(v.shape != s or v.dtype != d for v, s, d in served):
            return unsupported(node, f"the {what} launch's weights are not the layer's Marlin-repacked {what}_weight and _weight_scale roots",
                               b_q_weight=repr(wv), b_scales=repr(sv), roots=[_weight_root(ctx, v) for v, _, _ in served], expected=names)
        cfg = served_config(E, TOPK, K, N, top_k, M, target=tp)
        if isinstance(cfg, str):
            return unsupported(node, cfg)
        TK, TN, GRID = cfg
        try:
            MM.check_gemm(SimpleNamespace(K=K, N=N, TK=TK, TN=TN))
            spec = vocab_kind(ctx, "marlin_moe_gemm_w" if routed else "marlin_moe_gemm", E=E, K=K, N=N, TK=TK, TN=TN, GRID=GRID, DOT=MM.DOT)
        except ValueError as e:
            return unsupported(node, f"the {what} launch: {e}")
        return spec, A, y, *names


__all__ = ["MARLIN_FLAGS", "MARLIN_NONE", "MXFP4_TYPE_ID", "MoEMarlinSharedExperts", "SCHEDULE"]
