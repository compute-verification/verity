"""Padded-serving rules on vLLM's ops (profile mode; PAD-INT's first-stage requirements, PAD-SPEC v1 §4–§7).

The rule on vLLM's own op, `unified_cache_write`; the rules on ATen ops are `verity_torch.frontend.rules.padding`'s,
and `PADDING_RULES` is both lists, the ATen rules first.

    unified_cache_write(k, v, layer_name, slot)            -> the layer's history row `slot` (the row this step
        appended) now denotes the written values           (`CacheRowWrite`, a StateRule)
"""

from __future__ import annotations

import torch
from verity.circuits.refs import Coll

from verity_torch.frontend.derive import Alloc, refuse
from verity_torch.frontend.rules.base import RuleInfo, StateRule
from verity_torch.frontend.rules.common import shape
from verity_torch.frontend.rules.padding import _PROFILE, PADDING_TORCH_RULES
from verity_vllm.program.frontend.rules.vllm_iface import targets_for


class CacheRowWriteRule(StateRule):
    """`unified_cache_write(k, v, layer_name, slot)`: the K/V row at `slot` of this layer's history — the row this
    step's `unified_attention` appended — now denotes the written values (for every later reader). This is the
    padded engine's cache clear at the step's output boundary (spec §6): the step's own attention has read the raw
    row; later steps read the muxed one. Writing any other row, or a row of a layer this module does not own, is
    refused (`mutation`): the history is explicit state with exactly one writer per row per step."""
    modes = _PROFILE
    info = RuleInfo("CacheRowWrite[vllm]", 1, "state", applies_to="vllm_iface.unified_cache_write(key, value, layer_name, slot) -> ()",
                    preconditions=("k, v are bf16 value collections of one row [1, KVH*D] / [1, KVH, D]",
                                   "layer_name's history exists and is owned by this module path",
                                   "slot is the last row of the history (the row appended by this step); no other row is rewritten",
                                   "this step's attention already read the row (graph order)"),
                    evidence_domain="structural: the rewritten row is the value later Attention calls read")

    def targets(self):
        return targets_for("unified_cache_write")

    def applies(self, node, ctx):
        k, v, layer_name, slot = node.args[:4]
        for nm, a in (("key", k), ("value", v)):
            if not isinstance(a, torch.fx.Node):
                return f"{nm} is not a graph value"
            val = ctx.val(a)
            if isinstance(val, Alloc):
                return f"{nm} is an unwritten allocation"
            if not isinstance(val, Coll):
                return f"{nm} is not a value collection ({type(val).__name__})"
            if ctx.dtype_of(a) != torch.bfloat16:
                return f"numerics_unregistered: {nm} has dtype {ctx.dtype_of(a)}; the cache rows are bf16"
            s = shape(ctx, a)
            if not s or s[0] != 1:
                return f"{nm} shape {s} is not one row"
        if not isinstance(layer_name, str) or not isinstance(slot, int):
            return "layer_name / slot are not static"
        hist = ctx.state.get("kv_history", {})
        if layer_name not in hist:
            return f"no history for {layer_name!r} yet (the write precedes any attention of this layer)"
        return True

    def emit(self, node, ctx):
        k, v, layer_name, slot = node.args[:4]
        mp = ctx.module_path(node)
        owner = ctx.state["kv_owner"].get(layer_name)
        if owner != mp:
            refuse("mutation", node, mp, f"KV history {layer_name!r} is owned by {owner}; {mp} may not rewrite its rows")
        krow_list, vrow_list = ctx.state["kv_history"][layer_name]
        if slot != len(krow_list) - 1:
            refuse("mutation", node, mp, f"unified_cache_write rewrites row {slot} of {layer_name!r} whose history has {len(krow_list)} rows: "
                                         f"only the row this step appended (row {len(krow_list) - 1}) may be rewritten at the step boundary")
        kc, vc = ctx.coll(k), ctx.coll(v)
        row_t = krow_list[slot].type
        if kc.type.leaves != row_t.leaves or vc.type.leaves != row_t.leaves:
            refuse("shape", node, mp, f"written rows have {kc.type.leaves}/{vc.type.leaves} words; the history row has {row_t.leaves}")
        krow_list[slot] = kc.reshape(row_t)
        vrow_list[slot] = vc.reshape(row_t)
        ctx.state.setdefault("kv_rewrites", []).append({"layer_name": layer_name, "slot": slot, "node": node.name})
        ctx.record(self, node, {"layer_name": layer_name, "slot": slot, "row_words": row_t.leaves, "history_rows": len(krow_list)},
                   emitted="(state) history row rebound; no gates")


PADDING_VLLM_RULES = [CacheRowWriteRule()]
PADDING_RULES = PADDING_TORCH_RULES + PADDING_VLLM_RULES

__all__ = ["PADDING_RULES", "PADDING_TORCH_RULES", "PADDING_VLLM_RULES"]
