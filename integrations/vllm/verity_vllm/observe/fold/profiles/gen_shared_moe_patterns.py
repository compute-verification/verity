"""The block-FP8 shared-expert MoE scope: `vllm.moe_forward_shared` with a sigmoid, bias-corrected router whose gate the runner holds,
and `TritonExperts` at block [G, G] (GLM-5.3, `GlmMoeDsaForCausalLM` = `DeepseekV32ForCausalLM`, on vLLM d9105ea80 at TP without
expert parallelism, eager, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`).

`moe_forward_shared(hidden [M,H] bf16, router_logits = hidden, shared_experts_input = hidden, input_ids, layer_name, _) -> (shared,
routed)` folds, in `_forward_impl`'s order (fused_moe/runner/moe_runner.py:875-917), to:

  the gate (`GateLinear.forward`, router/gate_linear.py:179-213; the Build's `vllm_bindings.bf16_gemm` Definition at (K=H, N=E, M) on
  the tier `bf16_gemm_check.dispatch_sm90` names, or a `gate_spec(K, N, M, tier)` override):
    tier 4  `aten.mm.dtype(hidden, W.T, f32)` (cuBLAS)    -> Compute(spec, rows=M): (PerRow(hidden), Shared(W)) -> PerRow(scores)
    tier 1  `ll_bf16_gemm` (cuteDSL, M <= 16)             -> the same Compute over the gate weight Root, bound by its output: the kernel
                                                             is no dispatcher event, so only the `aten.empty [M,E] f32` it fills shows;
                                                             the header must attest the tier (`backends.moe_gate_ll_bf16_gemm`)
  the shared expert (`DeepseekV2MLP`, vLLM's default Hopper block-FP8 linear, `fp8.DynamicDeepGemmFp8Linear`):
    `dynamic_flashinfer_deepgemm_blockscale_gemm` (gate_up) -> FlashInferFp8Linear_v1 / DeepGemmFp8Linear_v1{K=H,N=2*SHARED_I}
    `_C.silu_and_mul`                                   -> SiluMul_v2{I=SHARED_I}
    `dynamic_flashinfer_deepgemm_blockscale_gemm` (down) -> {K=SHARED_I,N=H}, writing the op's first output
  `_moe_C.grouped_topk(scores, 1, 1, TOPK, True, RSF, bias, 1)`
                                                        -> MoeRouterSigmoidBias_v1{E,TOPK,SCALE=RSF,KERNEL}: (PerRow(scores), Shared(bias))
                                                           -> (PerRow(weights), PerRow(ids)), KERNEL `registry.moe.router_kernel`
  the routed experts (`MoEFusedExpertsFp8`'s pair forms and tiles):
    `per_token_group_fp8_quant` (hidden)                -> Fp8GroupQuant_v1{K=H,G}
    `moe_align_block_size`                              -> Transparent(noop), absent on the naive path
    `fused_moe_kernel` (w13)                            -> MoeExpertFp8Gemm_v1{E,K=H,N=2I,G,KB,SWAP,DOT}
    `_C.silu_and_mul`                                   -> SiluMul_v2{I} per pair: PerRow(cache1 as [P,2I]) -> PerRow(cache2)
    `per_token_group_fp8_quant` (cache2)                -> Fp8GroupQuant_v1{K=I,G} per pair, the w2 launch's A and a_scale
    `fused_moe_kernel` (w2, weighted)                   -> MoeExpertFp8GemmW_v1{E,K=I,N=H,G,KB,SWAP,DOT}
    `_moe_C.moe_sum`                                    -> MoeSum_v1{TOPK,H}, writing the op's second output

Everything else in the scope is aten glue, each a move or alloc in log order: the weights' `numpy_T`, the column-major scales'
`permute`, `to.dtype`, `empty` / `empty_like` / `full` (the naive path's `num_tokens_post_padded`), and the workspace's slice,
`view.dtype`, reshape, flatten and view chain.  Rank 0's H200 log pins all of it (`tests/engine/test_gen_shared_moe_fp8_recorded.py`).
Outside the op, `shared + routed`
(moe_runner.py:780) is the base `ResidualAddPattern`'s `ResidualAdd_v1{N=H}`; the block has no all-reduce (`reduce_results=False`: the
next layer's fused all-reduce and RMSNorm reduce the sum).
"""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import act as _ACT
from verity.ml.definitions.registry import moe as _MOE
from verity_vllm.program.frontend import targets as T

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (GeometryError, SiluAndMul, as_float, as_weight_view, base_op, bound_args, classify_transparent,
                                               find_descendants, is_row_major, op_name, target_of, triton_name, unsupported)
from verity_vllm.observe.fold.resolver import Compute, Gather, Group, PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode
from verity_vllm.observe.fold.profiles.gen_ov_moe_patterns import (K_ALIGN_KERNELS, K_FUSED, K_QUANT, K_SUM, TRITON_EXPERTS, MoEFusedExpertsFp8,
                                                             _as_rows, _contig, _flat, _root_name, _same, header_moe_experts_class)

SHARED_OP = "moe_forward_shared"
K_DYN_MM = "dynamic_flashinfer_deepgemm_blockscale_gemm"
K_ROUTER_OP = "grouped_topk"
#: the gate tiers this scope binds (`GateLinear.forward`'s tier 4 and tier 1)
GATE_CUBLAS, GATE_CUTEDSL = "cublas", "cutedsl"
#: `ll_bf16_gemm`'s token ceiling (router/gate_linear.py:181)
CUTEDSL_MAX_TOKENS = 16


def header_gate_ll_bf16(ctx: StepContext) -> bool:
    """Whether the record attests that its MoE gates took the cuteDSL `ll_bf16_gemm` tier (`RunHeader.profile["backends"]
    ["moe_gate_ll_bf16_gemm"]`, written by `engine.run_facts` from the gates' `allow_ll_bf16_gemm`)."""
    try:
        return ((ctx.header.profile or {}).get("backends") or {}).get("moe_gate_ll_bf16_gemm") is True
    except (AttributeError, TypeError):
        return False


def _weight_root(ctx: StepContext, view: ViewDesc) -> str | None:
    w = [r for r in ctx.roots_of(view) if r.role == "weight"]
    return w[0].name if w else None


def _glue(node: TreeNode, ctx: StepContext, keep: set[int], pattern: str) -> tuple[list[tuple[int, Any]], Unsupported | None]:
    """`scope_transparents` with each move/alloc keyed by its node id, so the scope's parts run in log order."""
    out: list[tuple[int, Any]] = []

    def walk(n: TreeNode):
        for c in n.children:
            if c.id in keep:
                continue
            yield c
            yield from walk(c)

    for c in walk(node):
        r = classify_transparent(c, ctx)
        if r is None:
            return [], unsupported(node, f"unexplained child {c.op} inside {op_name(node.op)}", child_id=c.id, child=c.op)
        if isinstance(r, Unsupported):
            return [], unsupported(node, f"child {c.op} inside {op_name(node.op)}: {r.reason}", child_id=c.id, child_detail=r.detail)
        if isinstance(r, Transparent) and r.effect in ("move", "alloc"):
            out.append((c.id, dataclasses.replace(r, consumed=(), pattern=pattern)))
        elif isinstance(r, list):
            out.extend((c.id, x) for x in r)
    return out, None


def scope_group(node: TreeNode, ctx: StepContext, kernels: list[TreeNode], parts: list[tuple[int, Any]], hint: str, pattern: str) -> Resolution:
    """The scope's Group: its `parts` (keyed by event id) and the glue around them in log order, every event outside `kernels`' subtrees
    and the parts' own events a move or alloc."""
    keep = set().union(*(set(n.ids()) for n in kernels)) | {i for i, p in parts if isinstance(p, Compute)}
    glue, bad = _glue(node, ctx, keep, pattern)
    if bad is not None:
        return bad
    return Group(tuple(p for _, p in sorted(glue + parts, key=lambda x: x[0])), tuple(sorted(node.ids())), hint=hint, pattern=pattern)


def scope_hidden(node: TreeNode, ctx: StepContext, H: int) -> tuple[ViewDesc, list[ViewDesc]] | Unsupported:
    """`moe_forward_shared`'s hidden states, which are also its router logits and shared-expert input (the runner holds the gate): a
    row-major bf16 [M,H] view over exactly the step's token rows; and its (shared, routed) outputs."""
    sa = bound_args(node)
    hidden, logits, sx = sa.get("hidden_states"), sa.get("router_logits"), sa.get("shared_experts_input")
    if not all(isinstance(v, ViewDesc) for v in (hidden, logits, sx)) or not (_same(logits, hidden) and _same(sx, hidden)):
        return unsupported(node, "moe_forward_shared does not take the hidden states as router_logits and shared_experts_input (the runner "
                                 "holds the gate)", args={k: repr(v) for k, v in sa.items()})
    M = hidden.shape[0]
    if hidden.rank != 2 or hidden.shape[1] != H or hidden.dtype != "bfloat16" or not is_row_major(hidden):
        return unsupported(node, "moe_forward_shared hidden_states is not a row-major bf16 [M,H] view", hidden=repr(hidden), H=H)
    if M != len(ctx.start.rows) or ctx.start.num_tokens != ctx.start.num_tokens_padded:
        return unsupported(node, "MoE rows are not exactly the step's token rows (padding rows would reach the router)",
                           M=M, rows=len(ctx.start.rows), num_tokens=ctx.start.num_tokens, num_tokens_padded=ctx.start.num_tokens_padded)
    outs = node.output_views()
    if len(outs) != 2:
        return unsupported(node, "moe_forward_shared does not return (shared, routed)", outs=[repr(v) for v in outs])
    return hidden, outs


def shared_expert(node: TreeNode, ctx: StepContext, mms: tuple[TreeNode, TreeNode], act: TreeNode, act_pattern: Any, want: dict[str, Any],
                  hidden: ViewDesc, out0: ViewDesc, hint: str, H: int, pattern: str, spec: Any = None) -> list[tuple[int, Any]] | Unsupported:
    """The shared expert: two dynamic block-FP8 linears (`fp8.DynamicDeepGemmFp8Linear`) around its activation `act`, which
    `act_pattern` resolves on the token rows at the statics `want` (I the shared intermediate), as `spec` when given; the down linear
    writes the op's first output `out0`."""
    from verity_vllm.observe.fold.patterns.fp8 import DynamicDeepGemmFp8Linear  # registry.deepgemm on use
    SI = want["I"]
    out: list[tuple[int, Any]] = []
    prev = hidden
    for mm, K, N, what in ((mms[0], H, 2 * SI, "gate_up"), (mms[1], SI, H, "down")):
        if what == "down":
            sr, aop = act_pattern.match(act, ctx), op_name(act.op)
            if not isinstance(sr, Compute) or sr.rows != len(ctx.start.rows) or any(sr.spec.bindings.get(k) != v for k, v in want.items()):
                return unsupported(node, f"the shared expert's {aop} did not resolve as {act_pattern.kind}{want} on the token rows",
                                   child_id=act.id, child_reason=getattr(sr, "reason", None), want=want)
            if not _same(sr.operands[0].view, prev):
                return unsupported(node, f"the shared expert's {aop} does not read the gate_up GEMM's output",
                                   act_input=repr(sr.operands[0].view), gate_up=repr(prev))
            out.append((act.id, dataclasses.replace(sr, spec=spec or sr.spec, hint=f"{hint}/shared/{aop}", pattern=pattern)))
            prev = sr.outputs[0].view
        mr = DynamicDeepGemmFp8Linear().match(mm, ctx)
        if not isinstance(mr, Compute):
            return unsupported(node, f"the shared expert's {what} {K_DYN_MM} did not resolve", child_id=mm.id,
                               child_reason=getattr(mr, "reason", None), child_detail=getattr(mr, "detail", None))
        if not _same(mr.operands[0].view, prev):
            return unsupported(node, f"the shared expert's {what} GEMM does not read its input", gemm_input=repr(mr.operands[0].view),
                               expected=repr(prev))
        if mr.spec.bindings.get("K") != K or mr.spec.bindings.get("N") != N:
            return unsupported(node, f"the shared expert's {what} GEMM is not [K={K}, N={N}]", bindings=dict(mr.spec.bindings))
        out.append((mm.id, dataclasses.replace(mr, hint=f"{hint}/shared/{what}", pattern=pattern)))
        prev = next(o for o in mr.outputs if o is not None).view
    if prev.alloc != out0.alloc:
        return unsupported(node, "the shared expert's down GEMM does not write the op's first output", down=repr(prev), shared=repr(out0))
    return out


def scope_gate(node: TreeNode, ctx: StepContext, gate: TreeNode | None, before: TreeNode, router: str, scores: Any, beside: str | None,
               hidden: ViewDesc, hint: str, E: int, H: int, gate_spec: Callable | None, pattern: str) -> tuple[int, Compute] | Unsupported:
    """The router gate writing the f32 [M,E] `scores` the `router` reads: the in-scope `aten.mm.dtype` `gate` (cuBLAS), or, with no mm,
    the cuteDSL kernel bound by the `aten.empty` it fills ahead of `before` (the shared expert's first GEMM), over `<gate>.weight` of the
    module whose weight `beside` the router reads.  `gate_spec(K, N, M, tier)` overrides the Build's Definition (`served_gate_spec`)."""
    M = hidden.shape[0]
    if not isinstance(scores, ViewDesc):
        return unsupported(node, f"{router}'s scores are not a view", scores=repr(scores))
    if gate is not None:
        tier, at = GATE_CUBLAS, gate
        a = bound_args(gate)
        x, wt, y = a.get("self", a.get("arg0")), a.get("mat2", a.get("arg1")), (gate.output_views() or [None])[0]
        if not (isinstance(x, ViewDesc) and isinstance(wt, ViewDesc) and isinstance(y, ViewDesc)) or not _same(x, hidden):
            return unsupported(node, "the in-scope mm is not the gate over hidden_states", args={k: repr(v) for k, v in a.items()})
        try:
            w = as_weight_view(wt, H, E)
        except GeometryError as e:
            return unsupported(node, f"the gate mm's mat2 is not W.T of a bf16 [E,H] weight: {e}")
        if y.shape != (M, E) or y.dtype != "float32" or not is_row_major(y) or not _same(scores, y):
            return unsupported(node, f"the gate mm does not write the f32 [M,E] scores {router} reads", out=repr(y), scores=repr(scores))
        wname = _weight_root(ctx, w)
    else:
        tier = GATE_CUTEDSL
        if not header_gate_ll_bf16(ctx) or M > CUTEDSL_MAX_TOKENS:
            return unsupported(node, f"{SHARED_OP} has no gate mm, and the record does not attest the cuteDSL ll_bf16_gemm tier at "
                                     f"M <= {CUTEDSL_MAX_TOKENS} (RunHeader.profile.backends.moe_gate_ll_bf16_gemm)", M=M)
        made = [n for n in node.walk() if n is not node and not n.is_triton and any(v.alloc == scores.alloc for v in n.output_views())]
        if len(made) != 1 or base_op(made[0].op) != "aten.empty" or not made[0].id < before.id:
            return unsupported(node, f"the scores {router} reads are not one in-scope aten.empty ahead of the shared expert (the cuteDSL "
                                     "gate's output)", writers=[(n.id, n.op) for n in made])
        at = made[0]
        root = ctx.root_named(f"{beside.rsplit('.', 1)[0]}.weight") if beside else None
        w = root.view if root is not None else None
        wname = root.name if root is not None and root.role == "weight" else None
    if wname is None or w is None or w.shape != (E, H) or w.dtype != "bfloat16" or not is_row_major(w):
        return unsupported(node, f"the gate weight is not a bf16 [E,H] weight Root (<gate>.weight beside the {router} weight)", w=repr(w),
                           beside=beside)
    got = gate_spec(H, E, M, tier) if gate_spec is not None else served_gate_spec(target_of(ctx)[0], H, E, M, tier)
    if got is None:
        return unsupported(node, f"no registered bf16 x bf16 -> f32 gate Definition for the {tier} tier at K={H}, N={E}, M={M} "
                                 f"(the Build's vllm_bindings.bf16_gemm route, or the profile's gate_spec)", tier=tier, K=H, N=E, M=M)
    spec, kernels = got
    return at.id, Compute(spec, M, (PerRow(hidden), Shared(w)), (PerRow(scores),), (at.id,), hint=f"{hint}/gate", kernels=tuple(kernels),
                          attribution=tuple(ctx.start.rows), pattern=pattern, roots=(None, wname))


@dataclass(frozen=True)
class _Scope:
    """The scope's events by role, once the census holds."""

    gate: TreeNode | None
    quants: tuple[TreeNode, TreeNode]
    mms: tuple[TreeNode, TreeNode]
    silu: TreeNode
    router: TreeNode
    aligns: tuple[TreeNode, ...]
    up: TreeNode
    rsilu: TreeNode
    down: TreeNode
    msum: TreeNode


@dataclass
class MoEFusedSharedExpertsFp8(MoEFusedExpertsFp8):
    """The module docstring's Group.  `config` carries NUM_EXPERTS, TOPK, H, I (the rank's routed intermediate), SHARED_I (the rank's
    shared-expert intermediate) and ROUTED_SCALING_FACTOR; `group` is G.  `gate_spec(K, N, M, tier)`, when given, returns a bf16 x
    bf16 -> f32 gate Definition and its kernel tokens for that tier and token count, or None; without it the gate is the Build's
    (`served_gate_spec`)."""

    name = "moe.fused_shared_experts.pairwise.fp8"
    gate_spec: Callable[[int, int, int, str], tuple[Any, tuple[str, ...]] | None] | None = None

    def _quants(self, node: TreeNode) -> list[TreeNode]:
        return find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == K_QUANT)

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != SHARED_OP:
            return None
        got = header_moe_experts_class(ctx)
        if got != TRITON_EXPERTS:
            return unsupported(node, f"the record does not attest {TRITON_EXPERTS} for its fused-MoE layers (RunHeader.profile.backends."
                                     f"moe_experts_class, written under a declared moe_backend); {self.name} states that kernel's arithmetic",
                               moe_experts_class=got)
        C, G = self.config, self.group
        keys = ("NUM_EXPERTS", "TOPK", "H", "I", "SHARED_I")
        if not all(isinstance(C.get(k), int) for k in keys) or as_float(C.get("ROUTED_SCALING_FACTOR")) is None or G <= 0:
            return unsupported(node, "profile config lacks NUM_EXPERTS / TOPK / H / I / SHARED_I / ROUTED_SCALING_FACTOR or the block G",
                               config={k: C.get(k) for k in (*keys, "ROUTED_SCALING_FACTOR")}, group=G)
        io = scope_hidden(node, ctx, C["H"])
        if isinstance(io, Unsupported):
            return io
        hidden, outs = io
        sc = self._census(node)
        if isinstance(sc, Unsupported):
            return sc
        hint = node.enter.module or "mlp.experts"
        parts: list[tuple[int, Any]] = []
        for step in (self._shared, self._router, self._routed):
            r = step(node, ctx, sc, hidden, outs, hint)
            if isinstance(r, Unsupported):
                return r
            parts += r
        gate = self._gate(node, ctx, sc, hidden, hint)
        if isinstance(gate, Unsupported):
            return gate
        parts.append(gate)
        at = [i for i, p in parts if isinstance(p, Compute)]
        if len(at) != len(set(at)) or not {q.id for q in sc.quants} <= set(at):
            return unsupported(node, f"the scope's two {K_QUANT} calls are not one each for the routed w13 and w2 inputs",
                               quants=[q.id for q in sc.quants], resolved=sorted(at))
        kernels = [*sc.quants, *sc.mms, sc.silu, sc.router, *sc.aligns, sc.up, sc.rsilu, sc.down, sc.msum]
        kernels += [sc.gate] if sc.gate is not None else []
        return scope_group(node, ctx, kernels, parts, hint, self.name)

    def _census(self, node: TreeNode) -> _Scope | Unsupported:
        def ops(name: str) -> list[TreeNode]:
            return find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == name)

        quants, mms, silus, gates, routers = ops(K_QUANT), ops(K_DYN_MM), ops("silu_and_mul"), ops("mm"), ops(K_ROUTER_OP)
        aligns, sums = ops("moe_align_block_size"), ops("moe_sum")
        launches = find_descendants(node, lambda n: n.is_triton and triton_name(n) == K_FUSED)
        counts = {K_QUANT: len(quants), K_DYN_MM: len(mms), "silu_and_mul": len(silus), "mm": len(gates), K_ROUTER_OP: len(routers),
                  "moe_align_block_size": len(aligns), K_FUSED: len(launches), "moe_sum": len(sums)}
        want = {K_QUANT: 2, K_DYN_MM: 2, "silu_and_mul": 2, K_ROUTER_OP: 1, K_FUSED: 2, "moe_sum": 1}
        if any(counts[k] != v for k, v in want.items()) or counts["mm"] > 1 or counts["moe_align_block_size"] > 1:
            return unsupported(node, f"{SHARED_OP} scope does not have the block-FP8 shared-expert shape (2 {K_QUANT}, 2 {K_DYN_MM}, "
                                     f"2 silu_and_mul, <=1 mm, 1 {K_ROUTER_OP}, <=1 align, 2 {K_FUSED}, 1 moe_sum)", counts=counts)
        s1, s2 = sorted(mms, key=lambda n: n.id)
        q1, q2 = sorted(quants, key=lambda n: n.id)
        silu, rsilu = sorted(silus, key=lambda n: n.id)
        sc = _Scope(gates[0] if gates else None, (q1, q2), (s1, s2), silu, routers[0], tuple(aligns), launches[0], rsilu, launches[1], sums[0])
        order = [*([sc.gate] if sc.gate else []), s1, sc.silu, s2, sc.router, q1, *sc.aligns, sc.up, sc.rsilu, q2, sc.down, sc.msum]
        if [n.id for n in order] != sorted(n.id for n in order):
            return unsupported(node, f"{SHARED_OP} scope runs out of order (gate, shared expert, router, routed experts, moe_sum)",
                               ids={op_name(n.op) if not n.is_triton else triton_name(n): n.id for n in order})
        return sc

    def _shared(self, node, ctx, sc: _Scope, hidden: ViewDesc, outs, hint) -> list[tuple[int, Any]] | Unsupported:
        """The shared expert: two dynamic block-FP8 linears around `_C.silu_and_mul`, resolved by the dense FP8 pattern."""
        C = self.config
        SI = C["SHARED_I"]
        return shared_expert(node, ctx, sc.mms, sc.silu, SiluAndMul(), {"I": SI}, hidden, outs[0], hint, C["H"], self.name,
                             spec=bind(_ACT.SiluMul, I=SI))

    def _router(self, node, ctx, sc: _Scope, hidden: ViewDesc, outs, hint) -> list[tuple[int, Any]] | Unsupported:
        C = self.config
        E, TOPK, M = C["NUM_EXPERTS"], C["TOPK"], hidden.shape[0]
        ra = bound_args(sc.router)
        scores, bias = ra.get("scores"), ra.get("bias")
        if not isinstance(scores, ViewDesc) or not isinstance(bias, ViewDesc):
            return unsupported(node, "grouped_topk arguments not bound as expected", args={k: repr(v) for k, v in ra.items()})
        got = (ra.get("n_group"), ra.get("topk_group"), ra.get("topk"), ra.get("renormalize"), ra.get("scoring_func"))
        rsf, want = as_float(ra.get("routed_scaling_factor")), float(as_float(C["ROUTED_SCALING_FACTOR"]))
        if got != (1, 1, TOPK, True, 1) or rsf != want:
            return unsupported(node, "grouped_topk is not one group, renormalized, sigmoid, TOPK at the profile's routed_scaling_factor",
                               observed=list(got) + [rsf], expected=[1, 1, TOPK, True, 1, want])
        if scores.shape != (M, E) or scores.dtype != "float32" or not is_row_major(scores):
            return unsupported(node, "grouped_topk scores are not the row-major f32 [M,E] gate output", scores=repr(scores))
        bname = _weight_root(ctx, bias)
        if bias.shape != (E,) or bias.dtype != "float32" or not _contig(bias) or bname is None:
            return unsupported(node, "grouped_topk bias is not the f32 [E] e_score_correction_bias weight", bias=repr(bias))
        router_outs = sc.router.output_views()
        if len(router_outs) != 2:
            return unsupported(node, "grouped_topk does not return (weights, ids)", outs=[repr(v) for v in router_outs])
        weights, ids = router_outs
        if (weights.shape, weights.dtype, ids.shape, ids.dtype) != ((M, TOPK), "float32", (M, TOPK), "int32") \
                or not (_contig(weights) and _contig(ids)):
            return unsupported(node, "grouped_topk outputs are not contiguous f32 / i32 [M,TOPK]", weights=repr(weights), ids=repr(ids))
        kernel = _MOE.router_kernel(E, M, TOPK)
        if kernel is None:
            return unsupported(node, "the grouped_topk launch at this (E, TOPK) is not pinned (registry.moe.router_kernel)", E=E, TOPK=TOPK)
        c = Compute(bind(_MOE.MoeRouterSigmoidBias, E=E, TOPK=TOPK, SCALE=want, KERNEL=kernel), M, (PerRow(scores), Shared(bias)),
                    (PerRow(weights), PerRow(ids)), (sc.router.id,), hint=f"{hint}/router", kernels=(f"single_group_topk_{kernel}_kernel",),
                    attribution=tuple(ctx.start.rows), pattern=self.name, roots=(None, bname))
        return [(sc.router.id, c)]

    def _gate(self, node, ctx, sc: _Scope, hidden: ViewDesc, hint) -> tuple[int, Compute] | Unsupported:
        """The router gate: the in-scope `aten.mm.dtype` (cuBLAS), or, with no mm, the cuteDSL kernel bound by the empty it fills."""
        C, ra = self.config, bound_args(sc.router)
        return scope_gate(node, ctx, sc.gate, sc.mms[0], K_ROUTER_OP, ra.get("scores"), _weight_root(ctx, ra.get("bias")), hidden, hint,
                          C["NUM_EXPERTS"], C["H"], self.gate_spec, self.name)

    def _routed(self, node, ctx, sc: _Scope, hidden: ViewDesc, outs, hint) -> list[tuple[int, Any]] | Unsupported:
        """The input quant, the alignment schedule, w13, the SiLU and its quant, w2 and moe_sum over the (token, slot) pairs."""
        from verity_vllm.observe.fold.patterns.fp8 import PerTokenGroupFp8Quant
        C, G = self.config, self.group
        E, TOPK, H, I, M = C["NUM_EXPERTS"], C["TOPK"], C["H"], C["I"], hidden.shape[0]
        P = M * TOPK
        weights, ids = sc.router.output_views()
        ids_flat, tp = _flat(ids, P), target_of(ctx)[0]
        out: list[tuple[int, Any]] = []
        for al in sc.aligns:
            out.append((al.id, Transparent("noop", (al.id,), hint=f"{hint}/moe_align_block_size: pair->expert-block schedule (sorted_token_ids, "
                                           "expert_ids, num_tokens_post_padded); no output word depends on it", kernels=K_ALIGN_KERNELS,
                                           pattern=self.name)))
        launched = {}
        for launch, K, N, routed in ((sc.up, H, 2 * I, False), (sc.down, I, H, True)):
            r = self._fused(node, launch, ctx, expect_weighted=routed, expect_top_k=1 if routed else TOPK, E=E, K=K, N=N, P=P)
            if isinstance(r, Unsupported):
                return r
            s = self._scaled(node, launch, E=E, K=K, N=N)
            if isinstance(s, Unsupported):
                return s
            if s.quant is not sc.quants[1 if routed else 0]:
                return unsupported(node, f"the {'w2' if routed else 'w13'} launch's A is not written by the routed "
                                         f"{'SiLU output' if routed else 'input'} quant", writer=s.quant.id)
            try:
                c = launch.enter.constexprs
                spec = T.moe_expert_fp8_gemm_spec(tp, E, K, N, G, s.KB, bool(c.get("SWAP_AB")),
                                                  BM=c.get("BLOCK_SIZE_M"), BN=c.get("BLOCK_SIZE_N"),
                                                  WARPS=launch.enter.num_warps, routed=routed)
            except T.UnregisteredTarget as e:
                return unsupported(node, f"target has no registered block-FP8 fused-MoE binding: {e}")
            launched[routed] = (s, r[2], spec)
        (s_up, c1, up_spec), (s_dn, c3, dn_spec) = launched[False], launched[True]
        qr = PerTokenGroupFp8Quant(group=G).match(s_up.quant, ctx)
        if not isinstance(qr, Compute) or (qr.operands[0].view, qr.outputs[0].view, qr.outputs[1].view) != (s_up.source, s_up.a, s_up.a_s):
            return unsupported(node, f"the routed input {K_QUANT} did not resolve to the w13 launch's A and a_scale", child_id=s_up.quant.id,
                               child_reason=getattr(qr, "reason", None))
        if not _same(s_up.source, hidden) or not s_up.quant.id < sc.up.id:
            return unsupported(node, "the routed input quant does not read hidden_states before the w13 launch", quant_input=repr(s_up.source))
        out.append((s_up.quant.id, dataclasses.replace(qr, hint=f"{hint}/w13_act_quant", pattern=self.name)))
        rows = tuple(p // TOPK for p in range(P))
        w13n, w13s = _root_name(ctx, s_up.b, "w13"), _root_name(ctx, s_up.b_s, "w13_ws")
        out.append((sc.up.id, Compute(up_spec, P, (Gather(s_up.a, rows), Gather(s_up.a_s, rows), PerRow(ids_flat), Shared(s_up.b), Shared(s_up.b_s)),
                                      (PerRow(_as_rows(c1, P, 2 * I)),), (sc.up.id,), hint=f"{hint}/{w13n}", kernels=(K_FUSED,), pattern=self.name,
                                      roots=(None, None, None, w13n, w13s))))
        sq = self._silu_quant(node, ctx, sc, c1, s_dn, P, I, G, hint)
        if isinstance(sq, Unsupported):
            return sq
        out += sq
        wv = bound_args(sc.down).get("topk_weights_ptr")
        if not isinstance(wv, ViewDesc) or not _same(wv, weights):
            return unsupported(node, "the w2 launch's routed weights are not grouped_topk's weights", topk_weights_ptr=repr(wv),
                               weights=repr(weights))
        w2n, w2s = _root_name(ctx, s_dn.b, "w2"), _root_name(ctx, s_dn.b_s, "w2_ws")
        out.append((sc.down.id, Compute(dn_spec, P, (PerRow(s_dn.a), PerRow(s_dn.a_s), PerRow(ids_flat), PerRow(_flat(weights, P)), Shared(s_dn.b),
                                                     Shared(s_dn.b_s)), (PerRow(_as_rows(c3, P, H)),), (sc.down.id,), hint=f"{hint}/{w2n}",
                                        kernels=(K_FUSED,), pattern=self.name, roots=(None, None, None, None, w2n, w2s))))
        ma = bound_args(sc.msum)
        y, o = ma.get("input"), ma.get("output")
        if not isinstance(y, ViewDesc) or not isinstance(o, ViewDesc) or ma.get("topk_ids") is not None or ma.get("expert_map") is not None:
            return unsupported(node, "moe_sum is not the modelled (input, output) kernel", args={k: repr(v) for k, v in ma.items()})
        if y.shape != (M, TOPK, H) or not _contig(y) or y.dtype != "bfloat16" or (y.alloc, y.offset) != (c3.alloc, c3.offset):
            return unsupported(node, "moe_sum input is not the contiguous bf16 [M,TOPK,H] w2 output", input=repr(y), cache3=repr(c3))
        if o.shape != (M, H) or not is_row_major(o) or o.dtype != "bfloat16" or o.alloc != outs[1].alloc:
            return unsupported(node, "moe_sum does not write the op's second output as a row-major bf16 [M,H] view", output=repr(o),
                               routed=repr(outs[1]))
        out.append((sc.msum.id, Compute(bind(_MOE.MoeSum, TOPK=TOPK, H=H), M, (PerRow(y),), (PerRow(o),), (sc.msum.id,), hint=f"{hint}/moe_sum",
                                        kernels=(K_SUM,), attribution=tuple(ctx.start.rows), pattern=self.name)))
        return out

    def _silu_quant(self, node, ctx, sc: _Scope, c1: ViewDesc, s_dn, P: int, I: int, G: int, hint: str) -> list[tuple[int, Any]] | Unsupported:
        """`_C.silu_and_mul(cache2!, cache1 as [P, 2I])` then `per_token_group_fp8_quant(cache2)`, the writer of the w2 launch's A and
        a_scale (triton_moe.py's unfused branch)."""
        from verity_vllm.observe.fold.patterns.fp8 import PerTokenGroupFp8Quant
        sr = SiluAndMul().match(sc.rsilu, ctx)
        if not isinstance(sr, Compute) or sr.rows != P or sr.spec.bindings.get("I") != I:
            return unsupported(node, "the routed silu_and_mul did not resolve as SiluMul{I} on the pair rows", child_id=sc.rsilu.id,
                               child_reason=getattr(sr, "reason", None), I=I, P=P)
        x = sr.operands[0].view
        if (x.alloc, x.offset) != (c1.alloc, c1.offset) or x.shape[-1] != 2 * I:
            return unsupported(node, "the routed silu_and_mul does not read the w13 output as [P, 2I]", input=repr(x), cache1=repr(c1))
        q = sc.quants[1]
        qr = PerTokenGroupFp8Quant(group=G).match(q, ctx)
        if not isinstance(qr, Compute) or (qr.outputs[0].view, qr.outputs[1].view) != (s_dn.a, s_dn.a_s):
            return unsupported(node, f"the routed SiLU output's {K_QUANT} did not resolve to the w2 launch's A and a_scale", child_id=q.id,
                               child_reason=getattr(qr, "reason", None))
        if not _same(qr.operands[0].view, sr.outputs[0].view) or not (sc.up.id < sc.rsilu.id < q.id < sc.down.id):
            return unsupported(node, f"the routed {K_QUANT} does not read the SiLU output between the two launches",
                               quant_input=repr(qr.operands[0].view), silu_output=repr(sr.outputs[0].view))
        return [(sc.rsilu.id, dataclasses.replace(sr, spec=bind(_ACT.SiluMul, I=I), hint=f"{hint}/silu_and_mul", pattern=self.name)),
                (q.id, dataclasses.replace(qr, hint=f"{hint}/w2_act_quant", attribution=(), pattern=self.name))]


def served_gate_spec(tp, K: int, N: int, M: int, tier: str) -> tuple[Any, tuple[str, ...]] | None:
    """The Build's gate Definition (`vllm_bindings.bf16_gemm.LlBf16GemmRule` / `CublasRouterRule`) at (K, N, M) on `tier`, or None
    when vLLM dispatches another tier there, the cuBLAS shape and M have no captured route, or the target is not the H200 default."""
    from verity.ml.definitions.registry import bf16_gemm_check, cublas_bf16, ll_bf16

    if tuple(tp.compute_capability) != (9, 0) or tp.batch_invariant:
        return None
    kernel, statics = bf16_gemm_check.dispatch_sm90(M, K, N)
    if (kernel == "cublas") != (tier == GATE_CUBLAS):
        return None
    if kernel == "dotprod":
        return bind(ll_bf16.LlBf16DotprodGemmWord, N=N, K=K, FMA=bf16_gemm_check.DEFAULT_LL_BF16_FMA, **statics), ("ll_bf16_gemm",)
    dot = T.tensor_core_dot(tp)
    if kernel == "splitk":
        return bind(ll_bf16.LlBf16SplitKGemmWord, N=N, K=K, DOT=dot, **statics), ("ll_bf16_gemm",)
    route = bf16_gemm_check.default_cublas_route("router", M, K, N)
    if route is None:
        return None
    if route[0] == "splitk":
        return bind(cublas_bf16.CublasSplitKGemmF32OutWord, K=K, N=N, DOT=dot, **route[1]), ("gemm", "splitK")
    return bind(cublas_bf16.GemmF32OutWord, K=K, N=N, DOT=dot), ("gemm",)


__all__ = ["CUTEDSL_MAX_TOKENS", "GATE_CUBLAS", "GATE_CUTEDSL", "MoEFusedSharedExpertsFp8", "SHARED_OP", "header_gate_ll_bf16",
           "scope_gate", "scope_group", "scope_hidden", "served_gate_spec", "shared_expert"]
