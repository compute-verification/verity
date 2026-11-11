"""The FA2 scope of a model served with logit soft-capping
(`attn_logit_softcapping > 0`: Gemma-2) -> the quarantined `AttentionSoftcap_v1{T,NH,KVH,D,BN,CAP}` (`verity_vllm.program.quarantine.dense.softcap`,
exact `MufuTanh` over all 2^32 sm_89 words), or
`AttentionSoftcap_v2` under FA2's per-iteration `Check_inf`.

vLLM passes the model's `attn_logit_softcapping` to `_vllm_fa2_C.varlen_fwd` as `softcap` (float; `FlashAttentionImpl.__init__`:
`self.logits_soft_cap = logits_soft_cap or 0`).  Inside vllm-flash-attn (`flash_api.cpp::set_params_fprop`) `softcap > 0` selects the
`Is_softcap=true` template of the SAME kernel (`flash_fwd_splitkv_kernel<..., Is_causal=true, Is_local=false, ..., Is_softcap=true>`), and

    params.softcap            = softmax_scale / softcap           acc_s = tanh.approx.f32(acc_s * params.softcap)   (one MUFU.TANH per score)
    params.scale_softmax_log2 = softcap * M_LOG2E                 exp2 stage constant CAP * log2e instead of scale * log2e

so relative to `Attention_v3` the body gains one stage in front of the online softmax (and one changed constant) and nothing else
(`softcap.py` docstring).  This pattern is a MORE SPECIFIC recogniser (no base matcher is widened): it fires only
when the bound launch argument `softcap` is a positive finite float, re-checks every base fact through the base `AttentionScope` (causal,
`softmax_scale == D**-0.5`, bf16, paged KV, `num_splits`, kBlockN, geometry, rows) on a copy of the launch with `softcap` rewritten to
`0.0`, keeps the inert-window rule of `AttentionScopeInertWindow` (Gemma-2's even layers pass `window_size_left = 4095`, inert below 4096
tokens), and then re-binds every resulting attention compute to its softcapped chain (`frontend.targets.softcapped`, the builder's
selection: `Attention_v3{T,NH,KVH,D,BN}` -> `AttentionSoftcap_v1{T,NH,KVH,D,BN,CAP=softcap}` on the accepted target, `Attention_v5` ->
`AttentionSoftcap_v2` under FA2's per-iteration `Check_inf`; any other chain is Unsupported by name) with the same operands / outputs /
attribution / consumed ids.  `softcap` in `(None, 0.0)` is the base pattern's case (returns None); a negative,
NaN or infinite `softcap` is Unsupported here (FA2 takes the softcap path iff `softcap > 0`; the Definition requires `CAP > 0`).

The Definition stays in `quarantine/dense`; G4 admits it through the run's
`quarantine.json` (per-Definition difftest record, `verity_vllm/engine/profiles/quarantine/GEMMA2_2B.json`, inherited by `cov_pod.sh`).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

from verity_vllm.program.frontend import targets as TG

from verity_vllm.observe.fold.patterns import FA2_OP_NAMES, AttentionScope, ComputeGroup, as_float, bound_args, find_descendants, op_name, unsupported
from verity_vllm.observe.fold.profiles.gen_dense2_patterns import AttentionScopeInertWindow, _annotate, _replace_descendant, _rewrite_args
from verity_vllm.observe.fold.resolver import Compute, Resolution, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode


def _is_window(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


@dataclass
class AttentionScopeSoftcap(AttentionScopeInertWindow):
    """`AttentionScope` for launches whose `softcap > 0` (see the module docstring); windowed launches go through the inert-window rule."""

    name: str = "attention.fa2_varlen_fwd.softcap"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_attention_with_output":
            return None
        fas = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) in FA2_OP_NAMES)
        if len(fas) != 1:
            return None                                           # the base pattern reports the malformed scope
        fa = fas[0]
        a = bound_args(fa)
        cap = as_float(a.get("softcap"))
        if cap is None or cap == 0.0:
            return None                                           # no softcap: the inert-window / base patterns' case
        if not (math.isfinite(cap) and cap > 0.0):
            return unsupported(node, "softcap is not a positive finite float (FA2 takes the Is_softcap path iff softcap > 0)", softcap=cap)
        # every other launch fact is the base pattern's to check: hand it the launch with softcap normalised to the no-softcap value
        rewritten = _replace_descendant(node, fa.id, _rewrite_args(fa, {"softcap": 0.0}))
        if _is_window(a.get("window_size_left")):
            r = AttentionScopeInertWindow.match(self, rewritten, ctx)    # None for a real local window: the base pattern reports it
        else:
            r = AttentionScope.match(self, rewritten, ctx)
        if r is None:
            return None
        if isinstance(r, Unsupported):
            return _annotate(r, {"softcap": cap})
        try:
            return self._rebind(r, cap)
        except TG.UnregisteredTarget as e:
            return unsupported(node, f"target has no registered softcapped attention binding: {e}", softcap=cap)

    def _rebind(self, r: Resolution, cap: float) -> Resolution:
        """Every attention compute of the group -> its softcapped chain with `CAP=cap` (`frontend.targets.softcapped`, the selection the
        builder makes: `Attention_v3` -> `AttentionSoftcap_v1`, `Attention_v5` -> `AttentionSoftcap_v2`; any other chain raises
        UnregisteredTarget); operands, outputs, rows, attribution and consumed ids are the base pattern's (the scope's transparent
        children in `.before` are untouched)."""
        def one(c: Resolution) -> Resolution:
            if not isinstance(c, Compute) or getattr(c.spec, "name", None) != "Attention":
                return c
            return dataclasses.replace(c, spec=TG.softcapped(c.spec, cap), pattern=self.name)
        if isinstance(r, ComputeGroup):
            primary = one(r)
            return dataclasses.replace(primary, also=tuple(one(x) for x in r.also))
        return one(r)


def with_softcap_patterns(patterns: list) -> list:
    """The profile's pattern list with the softcap scope inserted AHEAD of the first attention scope it refines (the inert-window
    scope when `with_dense2_patterns` ran first, else the base scope); `bn` / `num_splits` are the base scope's pins."""
    out: list = []
    inserted = False
    for p in patterns:
        if not inserted and isinstance(p, AttentionScope):
            out.append(AttentionScopeSoftcap(bn=p.bn, num_splits=p.num_splits))
            inserted = True
        out.append(p)
    return out
