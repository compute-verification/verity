"""vLLM's grouped sigmoid router (`_moe_C.grouped_topk`, one group) under the batch-invariant profile, beside the fused-MoE block's
leaf kernels of `vllm_moe` (the `indexed-read` construction): `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token row (numerics:
`verity_vllm.program.registry.moe`)."""

from __future__ import annotations

import torch

from verity.ir.refs import Coll
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import from_rows, rows_of, shape
from verity_vllm.program.frontend.rules.vllm_moe import INDEXED_READ, _moe_C_op, _not_owned


def grouped_topk_kernel(E: int, M: int) -> str | None:
    """The launch `_moe_C.grouped_topk` (`grouped_topk_kernels.cu` `launch`, one group) runs for M token rows of E experts: at 256
    experts the block kernel for at most 1024 tokens and the warp kernel for more, at 64 experts the warp kernel at every M; None
    for another E (unmeasured)."""
    if E == 256:
        return "block" if M <= 1024 else "warp"
    if E == 64:
        return "warp"
    return None


class GroupedTopkRule(OpRule):
    """`_moe_C.grouped_topk(scores[M,E] f32, n_group=1, topk_group=1, TOPK, renormalize=True, scale, bias[E] f32, SIGMOID)` ->
    `MoeRouterSigmoidBias{E,TOPK,SCALE,KERNEL}` per token row; the node is the pair (weights [M, TOPK] f32, ids [M, TOPK] i32)."""
    modes = frozenset({"profile"})
    info = RuleInfo("GroupedTopk[vllm-moe]", 1, "op-binding",
                    applies_to="_moe_C.grouped_topk(scores, n_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func)",
                    preconditions=("scores is an f32 value [M, E]; bias an f32 value [E] (the layer's e_score_correction_bias)",
                                   "n_group == topk_group == 1, renormalize True, scoring_func 1 (sigmoid); routed_scaling_factor a static float",
                                   "the launch is known for E (`grouped_topk_kernel`)"),
                    evidence_domain="structural; numerics: MoeRouterSigmoidBias_v1 (registry.moe)")

    def targets(self):
        return _moe_C_op("grouped_topk")

    def applies(self, node, ctx):
        if (r := _not_owned(INDEXED_READ)):
            return r
        if len(node.args) < 8:
            return f"{len(node.args)} args"
        scores, n_group, topk_group, topk, renorm, scale, bias, func = node.args[:8]
        if (n_group, topk_group, renorm, func) != (1, 1, True, 1):
            return (f"numerics_unregistered: n_group={n_group}, topk_group={topk_group}, renormalize={renorm}, scoring_func={func} "
                    f"(MoeRouterSigmoidBias_v1 is one group, renormalized, sigmoid)")
        if not isinstance(scale, (int, float)):
            return f"routed_scaling_factor {scale!r} is not a static number"
        for nm, t in (("scores", scores), ("bias", bias)):
            if not isinstance(t, torch.fx.Node) or not isinstance(ctx.val(t), Coll) or ctx.dtype_of(t) != torch.float32:
                return f"{nm} is not an f32 value collection"
        ss = shape(ctx, scores)
        if len(ss) != 2 or shape(ctx, bias) != (ss[1],):
            return f"scores {ss} vs bias {shape(ctx, bias)}"
        if not 1 <= topk <= min(32, ss[1]):
            return f"topk {topk} vs E={ss[1]}"
        if grouped_topk_kernel(ss[1], ss[0]) is None:
            return f"numerics_unregistered: the grouped_topk launch at E={ss[1]} is unmeasured"
        return True

    def emit(self, node, ctx):
        scores, _ng, _tg, TOPK, _rn, scale, bias = node.args[:7]
        rows, lead = rows_of(ctx.coll(scores))
        M, E = shape(ctx, scores)
        kernel = grouped_topk_kernel(E, M)
        fn = ctx.kind("moe_router_sigmoid_bias", node, E=E, TOPK=TOPK, RSF=float(scale), KERNEL=kernel)
        b = ctx.coll(bias)
        outs = [ctx.call(fn, r, b, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        ctx.define(node, [from_rows([o[0] for o in outs], lead), from_rows([o[1] for o in outs], lead)])
        ctx.bind_weight_role(bias, self, "router.e_score_correction_bias", E=E)
        ctx.record(self, node, {"E": E, "TOPK": TOPK, "SCALE": float(scale), "KERNEL": kernel, "rows": M}, emitted=f"{fn.id} x{M}")


VLLM_MOE_ROUTER_RULES = [GroupedTopkRule()]

__all__ = ["GroupedTopkRule", "VLLM_MOE_ROUTER_RULES", "grouped_topk_kernel"]
