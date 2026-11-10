"""A request's cached prefix: the key/value rows vLLM's prefix cache serves, carried over from the request that computed them."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import CACHED_PREFIX as STATE, dtype_is, shape
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16


class KvPrefixRule(StateRule):
    """`vllm_iface.kv_prefix(prefix)`: `prefix` bf16 [L, 2, C, KVH*D] is rows 0..C-1 of every attention layer's key and value cache, read
    from the prefix cache (never computed by this request).  Recorded as pending: each attention layer, when its history opens, starts
    it with its C rows (`seed`), layer l of the prefix being the l-th layer to open; the request's own rows follow at positions C...
    No gates."""
    modes = frozenset({"profile"})
    info = RuleInfo("KvPrefix[vllm]", 1, "state", applies_to="vllm_iface.kv_prefix(prefix) -> ()",
                    preconditions=("prefix is a bf16 value collection [L, 2, C, KVH*D] with C >= 1",
                                   "no KV history exists yet and no prefix is pending (the prefix precedes every attention)",
                                   "every one of the L layers opens its history with it, and only the dense attention rule reads it"),
                    evidence_domain="structural: the cached rows are the Program's input, bound by Match to the producing request's rows")

    def targets(self):
        from verity_vllm.program.frontend.rules.vllm_iface import targets_for
        return targets_for("kv_prefix")

    def applies(self, node, ctx):
        prefix = node.args[0]
        if not isinstance(prefix, torch.fx.Node) or not isinstance(ctx.val(prefix), Coll) or not dtype_is(ctx, prefix, BF16):
            return "prefix is not a bf16 value collection"
        s = shape(ctx, prefix)
        if len(s) != 4 or s[1] != 2 or s[2] < 1:
            return f"prefix has shape {s}, not [L, 2, C >= 1, KVH*D]"
        if ctx.state.get("kv_history") or ctx.state.get(STATE):
            return "a KV history or another prefix exists: the cached prefix opens every history"
        return True

    def emit(self, node, ctx):
        L, _, C, rl = shape(ctx, node.args[0])
        coll = ctx.coll(node.args[0]).reshape(Array(L, Array(2, Array(C, Array(rl, Value(16))))))
        ctx.record(self, node, {"layers": L, "rows": C, "row_leaves": rl}, emitted="(pending cached prefix; no gates)",
                   cached_prefix={"C": C, "row_leaves": rl, "layers": []})
        ctx.state[STATE] = {"coll": coll, "L": L, "C": C, "row_leaves": rl, "layers": ctx.applications[-1]["cached_prefix"]["layers"]}


def layout_of(provenance: dict) -> dict | None:
    """The cached prefix a derive read, from its provenance: `{"input": the prefix's user-input index (the wrapper's last input), "C",
    "row_leaves", "layers": attention layer names in the prefix's layer order}`; None for a Program without one."""
    rec = next((a["cached_prefix"] for a in provenance.get("applications") or () if "cached_prefix" in a), None)
    if rec is None:
        return None
    return {"input": len((provenance.get("source") or {}).get("example_inputs") or ()) - 1, **rec}


def seed(ctx, node, layer_name: str, krow_list: list, vrow_list: list, KVH: int, D: int) -> int:
    """Open `layer_name`'s history with its rows of the pending prefix (the next unseeded layer of it); the number of rows (0 without a
    prefix)."""
    p = ctx.state.get(STATE)
    if not p:
        return 0
    mp = ctx.module_path(node)
    l = len(p["layers"])
    if l >= p["L"]:
        refuse("side_input", node, mp, f"layer {layer_name!r} is attention layer {l + 1} of the forward, the cached prefix holds {p['L']}")
    if p["row_leaves"] != KVH * D:
        refuse("shape", node, mp, f"the cached prefix's rows hold {p['row_leaves']} values, layer {layer_name!r} writes KVH*D = {KVH * D}")
    row_t = Array(KVH, Array(D, Value(16)))
    for j in range(p["C"]):
        krow_list.append(p["coll"][l][0][j].reshape(row_t))
        vrow_list.append(p["coll"][l][1][j].reshape(row_t))
    p["layers"].append(layer_name)   # the provenance's `cached_prefix.layers` (`layout_of`)
    return p["C"]


def require_consumed(ctx, node) -> None:
    """Refuse when the prefill opened fewer attention layers than the cached prefix holds (called on every later attention call)."""
    p = ctx.state.get(STATE)
    if p and len(p["layers"]) != p["L"]:
        refuse("side_input", node, ctx.module_path(node), f"the cached prefix holds {p['L']} layers, the prefill opened {len(p['layers'])}")
