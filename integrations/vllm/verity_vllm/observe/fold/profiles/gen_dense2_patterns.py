"""MORE SPECIFIC recognisers registered ahead of the base ones by the dense2 profile modules
(`gen_dense2_phi3_mini`, `gen_dense2_gemma2_2b`).  No base matcher is widened; each pattern here states the extra fact it
checks and resolves to an already registered body or to a quarantined one.

`AttentionScopeInertWindow` -- the FA2 scope of a model whose config carries a `sliding_window` the kernel never applies.
    vLLM passes the model-level window to `_vllm_fa2_C.varlen_fwd` as `window_size_left = sliding_window - 1` (Phi-3-mini-4k:
    2046; `FlashAttentionImpl.__init__`: `self.sliding_window = (sliding_window - 1, 0)` for a causal layer).  Inside
    vllm-flash-attn's `mha_varlen_fwd` (flash_api.cpp, upstream FA2 semantics) a window at least as long as the KV extent is
    normalised away BEFORE the kernel is chosen:

        if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
        if (window_size_right >= max_seqlen_k) { window_size_right = -1; }
        params.is_causal = window_size_left < 0 && window_size_right == 0;   // -> the causal, non-local kernel

    so for `is_causal = True` and `window_size_left >= max_seqlen_k` the launched kernel is exactly the one the base
    `AttentionScope` models (`flash_fwd_splitkv_kernel<..., Is_causal=true, Is_local=false, ...>`; the kernel census of the
    Phi-3 run shows that symbol and no `Is_local=true` instantiation).  The pattern checks that inequality on the bound launch
    arguments of THIS call (never the config) and then resolves the scope through the base pattern with the two window
    arguments rewritten to -1, i.e. to the same `Attention_v3{T,NH,KVH,D,BN}` computes.  A window shorter than
    `max_seqlen_k` is left to the base pattern, which reports it as Unsupported (a local kernel is a different body).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from verity_vllm.observe.fold.patterns import FA2_OP_NAMES, AttentionScope, bound_args, find_descendants, op_name, schema_param_names, unsupported
from verity_vllm.observe.fold.resolver import Resolution, StepContext
from verity_vllm.observe.fold.tree import TreeNode


def _rewrite_args(fa: TreeNode, updates: dict[str, Any]) -> TreeNode:
    """A copy of the FA2 launch node with the named schema arguments replaced (positional or keyword, whichever bound them)."""
    names = schema_param_names(fa.enter.schema)
    args = list(fa.enter.args)
    kwargs = dict(fa.enter.kwargs)
    for key, val in updates.items():
        if key in kwargs:
            kwargs[key] = val
        elif key in names and names.index(key) < len(args):
            args[names.index(key)] = val
        else:
            raise KeyError(key)
    return TreeNode(enter=dataclasses.replace(fa.enter, args=args, kwargs=kwargs), exit=fa.exit, children=fa.children)


def _replace_descendant(node: TreeNode, old_id: int, new: TreeNode) -> TreeNode:
    """A copy of `node` whose descendant with event id `old_id` is `new` (ids and every other node are shared, not copied)."""
    if node.id == old_id:
        return new
    children = [_replace_descendant(c, old_id, new) for c in node.children]
    if all(c is o for c, o in zip(children, node.children)):
        return node
    return TreeNode(enter=node.enter, exit=node.exit, children=children)


@dataclass
class AttentionScopeInertWindow(AttentionScope):
    """`AttentionScope` for launches whose `window_size_left >= max_seqlen_k` (see the module docstring)."""

    name: str = "attention.fa2_varlen_fwd.inert_window"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "unified_attention_with_output":
            return None
        fas = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) in FA2_OP_NAMES)
        if len(fas) != 1:
            return None                                   # the base pattern reports the malformed scope
        fa = fas[0]
        a = bound_args(fa)
        left, right, max_k = a.get("window_size_left"), a.get("window_size_right"), a.get("max_seqlen_k")
        if not (isinstance(left, int) and not isinstance(left, bool) and left >= 0):
            return None                                   # no window: the base pattern's case
        causal = a.get("is_causal") if a.get("is_causal") is not None else a.get("causal")
        if causal is not True:
            return unsupported(node, "windowed FA2 launch that is not causal", window_size_left=left, window_size_right=right, causal=causal)
        if not (isinstance(max_k, int) and not isinstance(max_k, bool) and max_k > 0):
            return unsupported(node, "windowed FA2 launch without a bound max_seqlen_k", window_size_left=left, max_seqlen_k=max_k)
        if left < max_k:
            return None                                   # a real local window: the base pattern reports it as Unsupported
        if right not in (None, 0, -1):
            return unsupported(node, "right window on a causal launch", window_size_right=right)
        # window_size_left >= max_seqlen_k and causal: flash_api.cpp sets window_size_left = -1 and (is_causal) window_size_right = 0
        # whatever right value was passed (-1 or 0), i.e. exactly the state of the windowless causal launch `(-1, -1, causal)`
        # that the base pattern models; the rewrite below is that normalisation.
        rewritten = _replace_descendant(node, fa.id, _rewrite_args(fa, {"window_size_left": -1, "window_size_right": -1}))
        r = AttentionScope.match(self, rewritten, ctx)
        if r is None:
            return None
        return _annotate(r, {"inert_window": {"window_size_left": left, "window_size_right": right, "max_seqlen_k": max_k}})


def _annotate(r: Any, extra: dict[str, Any]) -> Any:
    """Record the checked launch facts on an Unsupported's detail; Compute / Group resolutions carry the pattern name already."""
    detail = getattr(r, "detail", None)
    if isinstance(detail, dict):
        try:
            return dataclasses.replace(r, detail={**detail, **extra})
        except (TypeError, ValueError):
            return r
    return r


def with_dense2_patterns(patterns: list, C: dict[str, Any]) -> list:
    """The base pattern list with the more specific dense2 patterns inserted AHEAD of the base ones they refine."""
    out: list = []
    for p in patterns:
        if type(p) is AttentionScope:
            out.append(AttentionScopeInertWindow(bn=p.bn, num_splits=p.num_splits))
        out.append(p)
    return out
