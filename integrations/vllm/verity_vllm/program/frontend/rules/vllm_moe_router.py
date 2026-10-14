"""vLLM's grouped sigmoid router (`_moe_C.grouped_topk`, one group), beside the fused-MoE block's leaf kernels of `vllm_moe` (the
`indexed-read` construction): `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token row (numerics: `verity_vllm.program.registry.moe`).

    _moe_C.grouped_topk(scores [M, E] f32, 1, 1, TOPK, True, scale, bias [E] f32, 1)
        -> (weights f32 [M, TOPK], ids i32 [M, TOPK]): `MoeRouterSigmoidBias{E,TOPK,SCALE=scale,KERNEL}` on each token row, KERNEL the
           launch `router_kernel(E, M, TOPK)` picks (`grouped_topk_kernels.cu` `launch`)

DeepSeek-V3's and GLM's `noaux_tc` routing reaches it through `GroupedTopKRouter` -> `fused_grouped_topk` (vLLM d9105ea80): the
bias is the layer's `e_score_correction_bias` Parameter, and the scale the routed scaling factor, applied inside the kernel where the
model leaves `apply_routed_scale_to_output` False."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_iface import targets_for
from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, _moe_C_op, _not_owned
from verity_vllm.program.registry.moe import router_kernel


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


VLLM_MOE_ROUTER_RULES = [GroupedTopkRule()]

__all__ = ["GroupedTopkRule", "VLLM_MOE_ROUTER_RULES"]
