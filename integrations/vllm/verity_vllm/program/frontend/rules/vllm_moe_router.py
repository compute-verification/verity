"""vLLM's grouped sigmoid router (`_moe_C.grouped_topk`, one group), beside the fused-MoE block's leaf kernels of `vllm_moe` (the
`indexed-read` construction): `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token row (numerics: `verity.ml.definitions.registry.moe`).

    _moe_C.grouped_topk(scores [M, E] f32, 1, 1, TOPK, True, scale, bias [E] f32, 1)
        -> (weights f32 [M, TOPK], ids i32 [M, TOPK]): `MoeRouterSigmoidBias{E,TOPK,SCALE=scale,KERNEL}` on each token row, KERNEL the
           launch `router_kernel(E, M, TOPK)` picks (`grouped_topk_kernels.cu` `launch`)

DeepSeek-V3's and GLM's `noaux_tc` routing reaches it through `GroupedTopKRouter` -> `fused_grouped_topk` (vLLM d9105ea80): the
bias is the layer's `e_score_correction_bias` Parameter, and the scale the routed scaling factor, applied inside the kernel where the
model leaves `apply_routed_scale_to_output` False.

DeepSeek-V4's sqrtsoftplus routers (numerics: `verity.ml.definitions.registry.dsv4_moe`), through `FusedTopKBiasRouter` ->
`fused_topk_bias`:

    _moe_C.topk_softplus_sqrt(w!, ids!, tei!, logits [M, E] f32, True, scale, None, input_ids [M] i32, tid2eid [V, 6] i32, None)
        -> per token row the six ids `tid2eid[input_ids[t]]` (`GatherI32x{V}` per slot) and their weights `MoeRouterHash{E,SCALE}`
    verity_cba::triton_launch(_dsv4_topk_kernel; logits [M, E] f32, bias [E] f32, w!, ids!, ...)   (dsv4_topk, the scored layers)
        -> per token row the weights and ids `MoeRouterSqrtSoftplus{E,SCALE}`"""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, array_of, strided_view
from verity.ml.circuits.types import Array
from verity_vllm.program.frontend.derive import Alloc
from verity_vllm.program.frontend.rules.base import InPlaceRule, OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import call_args, from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.triton_iface import launch_meta, launch_tensor, triton_launch
from verity_vllm.program.frontend.rules.vllm_bindings.operands import _strides
from verity_vllm.program.frontend.rules.vllm_bindings.pins import DSV4_TOPK_PINS, _pins_ok
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, DeadOutput, _moe_C_op, _not_owned
from verity.ml.definitions.registry.dsv4_moe import TOPK as DSV4_TOPK
from verity.ml.definitions.registry.moe import router_kernel
from verity.ml.definitions.registry.prims import I32


class GroupedTopkRule(OpRule):
    """`_moe_C.grouped_topk(scores[M,E] f32, n_group=1, topk_group=1, TOPK, renormalize=True, scale, bias[E] f32, SIGMOID)` ->
    `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token row; the node is the pair (weights [M, TOPK] f32, ids [M, TOPK] i32)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GroupedTopk[vllm-moe]", 1, "op-binding",
                    applies_to="_moe_C.grouped_topk(scores, n_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func)",
                    preconditions=("scores is an f32 value [M, E]; bias an f32 value [E] (the layer's e_score_correction_bias)",
                                   "n_group == topk_group == 1, renormalize True, scoring_func 1 (sigmoid); routed_scaling_factor a static float",
                                   "the launch is pinned for (E, M, TOPK) (`registry.moe.router_kernel`)"),
                    evidence_domain="structural; numerics: MoeRouterSigmoidBias_v1 (registry.moe)")

    def targets(self):
        return targets_for("grouped_topk") | _moe_C_op("grouped_topk")

    @staticmethod
    def _args(node) -> dict:
        d = {"scoring_func": 0}
        d.update(zip(("scores", "n_group", "topk_group", "topk", "renormalize", "routed_scaling_factor", "bias", "scoring_func"), node.args))
        d.update(node.kwargs)
        return d

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        a = self._args(node)
        if len(a) != 8:
            return f"unexpected operator arity ({sorted(a)})"
        scores, n_group, topk_group, topk, renorm, scale, bias, func = (a[k] for k in ("scores", "n_group", "topk_group", "topk", "renormalize",
                                                                                      "routed_scaling_factor", "bias", "scoring_func"))
        if (n_group, topk_group, renorm, func) != (1, 1, True, 1):
            return (f"numerics_unregistered: n_group={n_group}, topk_group={topk_group}, renormalize={renorm}, scoring_func={func} "
                    f"(MoeRouterSigmoidBias_v1 is one group, renormalized, sigmoid)")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            return f"routed_scaling_factor {scale!r} is not a static number"
        for nm, t in (("scores", scores), ("bias", bias)):
            if not isinstance(t, torch.fx.Node) or not isinstance(ctx.val(t), Coll) or ctx.dtype_of(t) != torch.float32:
                return f"{nm} is not an f32 value collection"
        ss = shape(ctx, scores)
        if len(ss) != 2 or shape(ctx, bias) != (ss[1],):
            return f"scores {ss} vs bias {shape(ctx, bias)}"
        if not isinstance(topk, int) or not 1 <= topk <= min(32, ss[1]):
            return f"topk {topk!r} vs E={ss[1]}"
        if router_kernel(ss[1], ss[0], topk) is None:
            return f"numerics_unregistered: the grouped_topk launch at E={ss[1]}, top {topk} is not pinned (registry.moe.router_kernel)"
        return True

    def emit(self, node, ctx):
        a = self._args(node)
        scores, TOPK, scale, bias = a["scores"], a["topk"], a["routed_scaling_factor"], a["bias"]
        rows, lead = rows_of(ctx.coll(scores))
        M, E = shape(ctx, scores)
        kernel = router_kernel(E, M, TOPK)
        fn = ctx.kind("moe_router_sigmoid_bias", node, E=E, TOPK=TOPK, RSF=float(scale), KERNEL=kernel)
        b = ctx.coll(bias)
        outs = [ctx.call(fn, r, b, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, [from_rows([o[0] for o in outs], lead), from_rows([o[1] for o in outs], lead)])
        ctx.bind_weight_role(bias, self, "router.e_score_correction_bias", E=E)
        ctx.record(self, node, {"E": E, "TOPK": TOPK, "SCALE": float(scale), "KERNEL": kernel, "rows": M}, emitted=f"{fn.id} x{M}")


def _value(ctx, t, dtype) -> bool:
    return isinstance(t, torch.fx.Node) and isinstance(ctx.val(t), Coll) and ctx.dtype_of(t) == dtype


class TopkSoftplusSqrtHashRule(InPlaceRule):
    """`_moe_C.topk_softplus_sqrt(w!, ids!, tei!, logits[M,E] f32, True, scale, None, input_ids[M] i32, tid2eid[V,6] i32, None)`, the
    hash layers' router (`ops.topk_hash_softplus_sqrt`): per token row the six ids `tid2eid[input_ids[t]]` (`GatherI32x{V}` per slot)
    and their weights `MoeRouterHash{E,SCALE}`; `w` and `ids` are rebound, `tei` is dead.  The kernel's scored path (a bias, no
    table) has no Definition: vLLM routes such a layer through `dsv4_topk` where it can, and its Program is that kernel's."""
    modes = frozenset({"profile"})
    info = RuleInfo("TopkSoftplusSqrtHash[vllm-moe]", 1, "in-place-binding",
                    applies_to="_moe_C.topk_softplus_sqrt(Tensor! w, Tensor! ids, Tensor! tei, logits, renormalize, routed_scaling_factor, "
                               "bias, input_ids, tid2eid, is_padding, bias_vl=None, image_sentinel_lo=0)",
                    preconditions=("logits is an f32 value [M, E]; input_ids an i32 value [M]; tid2eid an i32 value [V, 6]",
                                   "renormalize True; routed_scaling_factor a static number; no bias, padding mask or vision bias",
                                   "w is an allocation [M, 6] f32; ids and tei [M, 6] i32"),
                    evidence_domain="structural; numerics: GatherI32x{V}_v1 and MoeRouterHash_v1 (registry.dsv4_moe: dsv4HashTopkSoftplusSqrt)")

    def targets(self):
        return targets_for("topk_softplus_sqrt") | _moe_C_op("topk_softplus_sqrt")

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        a = call_args(node)
        if a["tid2eid"] is None:
            return ("numerics_unregistered: topk_softplus_sqrt without a hash table (the CUDA scored router); MoeRouterSqrtSoftplus_v1 "
                    "is dsv4_topk's Triton kernel, which fused_topk_bias takes where can_use_dsv4_topk holds")
        if a["bias"] is not None or a["is_padding"] is not None or a["bias_vl"] is not None or a["image_sentinel_lo"]:
            return "numerics_unregistered: a bias, padding mask or vision bias is set (MoeRouterHash_v1 is a text token's table route)"
        if a["renormalize"] is not True:
            return f"numerics_unregistered: renormalize={a['renormalize']!r} (MoeRouterHash_v1 renormalises)"
        scale = a["routed_scaling_factor"]
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            return f"routed_scaling_factor {scale!r} is not a static number"
        if not (ctx.vocab.has("gather_i32") and ctx.vocab.has("moe_router_hash")):
            return "numerics_unregistered: the vocabulary has no gather_i32 / moe_router_hash (GatherI32x{V}_v1, MoeRouterHash_v1)"
        logits, toks, table = a["gating_output"], a["input_ids"], a["tid2eid"]
        if not (_value(ctx, logits, torch.float32) and _value(ctx, toks, torch.int32) and _value(ctx, table, torch.int32)):
            return "logits, input_ids and tid2eid are not f32, i32 and i32 value collections"
        ls, vs = shape(ctx, logits), shape(ctx, table)
        if len(ls) != 2 or shape(ctx, toks) != ls[:1] or len(vs) != 2 or vs[1] != DSV4_TOPK:
            return f"logits {ls}, input_ids {shape(ctx, toks)}, tid2eid {vs} (need [M, E], [M], [V, {DSV4_TOPK}])"
        for nm, dt in (("topk_weights", torch.float32), ("topk_indices", torch.int32), ("token_expert_indices", torch.int32)):
            t = a[nm]
            if not (isinstance(ctx.val(t), (Alloc, Coll)) and ctx.dtype_of(t) == dt and shape(ctx, t) == (ls[0], DSV4_TOPK)):
                return f"{nm} is not an allocation {dt} [{ls[0]}, {DSV4_TOPK}]"
        return True

    def emit(self, node, ctx):
        a = call_args(node)
        logits, toks, table = (ctx.coll(a[k]) for k in ("gating_output", "input_ids", "tid2eid"))
        (M, E), V, SCALE = shape(ctx, a["gating_output"]), shape(ctx, a["tid2eid"])[0], float(a["routed_scaling_factor"])
        gather, route = ctx.kind("gather_i32", node, V=V), ctx.kind("moe_router_hash", node, E=E, SCALE=SCALE)
        cols = [Coll(Array(V, I32), strided_view(table, k, V, V * DSV4_TOPK, DSV4_TOPK, V)) for k in range(DSV4_TOPK)]
        name = ctx.hint_name(node)
        ids = [array_of([ctx.call(gather, toks[t], c, node=node, rule=self, name=f"{name}/ids[{t},{k}]") for k, c in enumerate(cols)])
               for t in range(M)]
        w = [ctx.call(route, logits[t], ids[t], node=node, rule=self, name=f"{name}[{t}]") for t in range(M)]
        ctx.rebind(a["topk_weights"], array_of(w), self, what="topk_weights")
        ctx.rebind(a["topk_indices"], array_of(ids), self, what="topk_ids")
        ctx.rebind(a["token_expert_indices"], DeadOutput("token_expert_indices"), self, what="token_expert_indices")
        ctx.bind_weight_role(a["tid2eid"], self, "router.tid2eid", V=V)
        ctx.record(self, node, {"E": E, "V": V, "SCALE": SCALE, "rows": M}, emitted=f"({gather.id} x{DSV4_TOPK} + {route.id}) x{M}")


_DSV4_TOPK_TENSORS = ("gating_output_ptr", "correction_bias_ptr", "topk_weights_ptr", "topk_ids_ptr")


class Dsv4TopkRule(InPlaceRule):
    """`_dsv4_topk_kernel[(M,)](gating_output, correction_bias, topk_weights, topk_ids, routed_scaling_factor, input_ids, bias_vl,
    image_sentinel_lo, NUM_EXPERTS, BLOCK_N, HAS_VL, launch_pdl, num_warps=1)` launched by `dsv4_topk` (`fused_topk_bias` where
    `can_use_dsv4_topk` holds) -> `MoeRouterSqrtSoftplus{E,SCALE}` per token row; topk_weights and topk_ids (allocations) are rebound.
    Without the vision bias the kernel reads neither input_ids nor image_sentinel_lo, and `launch_pdl` adds the PDL wait and launch
    and no arithmetic."""
    modes = frozenset({"profile"})
    info = RuleInfo("Dsv4Topk[vllm-moe]", 1, "in-place-binding",
                    applies_to="verity_cba::triton_launch of router.dsv4_topk._dsv4_topk_kernel (kernel name + source + constexprs pinned)",
                    preconditions=("kernel qualified name, source sha256, NUM_EXPERTS, BLOCK_N, HAS_VL and num_warps are the pinned ones "
                                   "(DSV4_TOPK_PINS)",
                                   "gating_output a packed f32 value [M, E]; correction_bias a packed f32 value [E]",
                                   f"topk_weights and topk_ids unwritten packed allocations [M, {DSV4_TOPK}], f32 and i32",
                                   "routed_scaling_factor a static number; grid (M,)"),
                    evidence_domain="structural; numerics: MoeRouterSqrtSoftplus_v1 (registry.dsv4_moe: the kernel's cc 9.0 PTX)")

    def targets(self):
        return {triton_launch.default}

    def applies(self, node, ctx):
        meta = launch_meta(node)
        if meta.get("kernel") != DSV4_TOPK_PINS["kernel"]:
            return f"kernel {meta.get('kernel')} is not {DSV4_TOPK_PINS['kernel']}"
        if (r := _pins_ok(meta, DSV4_TOPK_PINS)):
            return r
        if not ctx.vocab.has("moe_router_sqrtsoftplus"):
            return "numerics_unregistered: the vocabulary has no moe_router_sqrtsoftplus (MoeRouterSqrtSoftplus_v1)"
        x, b, w, ids = (launch_tensor(node, meta, n) for n in _DSV4_TOPK_TENSORS)
        if any(t is None for t in (x, b, w, ids)):
            return f"tensor arguments {_DSV4_TOPK_TENSORS} not all present"
        if not (_value(ctx, x, torch.float32) and _value(ctx, b, torch.float32)):
            return "gating_output and correction_bias are not f32 value collections"
        xs, E = shape(ctx, x), meta["scalars"]["NUM_EXPERTS"]
        if len(xs) != 2 or xs[1] != E or shape(ctx, b) != (E,) or _strides(x) != (E, 1) or _strides(b) != (1,):
            return f"gating_output {xs} strides {_strides(x)}, correction_bias {shape(ctx, b)} strides {_strides(b)}: not packed [M, {E}], [{E}]"
        for nm, t, dt in (("topk_weights", w, torch.float32), ("topk_ids", ids, torch.int32)):
            if not (isinstance(ctx.val(t), Alloc) and ctx.dtype_of(t) == dt and shape(ctx, t) == (xs[0], DSV4_TOPK)
                    and _strides(t) == (DSV4_TOPK, 1)):
                return f"{nm} is not an unwritten packed {dt} allocation [{xs[0]}, {DSV4_TOPK}]"
        scale = meta["scalars"].get("routed_scaling_factor")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            return f"routed_scaling_factor {scale!r} is not a static number"
        if meta.get("grid") != [xs[0]]:
            return f"grid {meta.get('grid')} is not ({xs[0]},)"
        return True

    def emit(self, node, ctx):
        meta = launch_meta(node)
        x, b, w, ids = (launch_tensor(node, meta, n) for n in _DSV4_TOPK_TENSORS)
        (M, E), SCALE = shape(ctx, x), float(meta["scalars"]["routed_scaling_factor"])
        fn = ctx.kind("moe_router_sqrtsoftplus", node, E=E, SCALE=SCALE)
        rows, lead = rows_of(ctx.coll(x))
        bias = ctx.coll(b)
        outs = [ctx.call(fn, r, bias, node=node, rule=self, name=f"{ctx.hint_name(node)}[{t}]") for t, r in enumerate(rows)]
        ctx.rebind(w, from_rows([o[0] for o in outs], lead), self, what="topk_weights")
        ctx.rebind(ids, from_rows([o[1] for o in outs], lead), self, what="topk_ids")
        ctx.bind_weight_role(b, self, "router.e_score_correction_bias", E=E)
        ctx.record(self, node, {"E": E, "SCALE": SCALE, "rows": M, "pins": DSV4_TOPK_PINS}, emitted=f"{fn.id} x{M}")


VLLM_MOE_ROUTER_RULES = [GroupedTopkRule(), TopkSoftplusSqrtHashRule(), Dsv4TopkRule()]

__all__ = ["Dsv4TopkRule", "GroupedTopkRule", "TopkSoftplusSqrtHashRule", "VLLM_MOE_ROUTER_RULES"]
