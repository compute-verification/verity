"""The padded engine's step-boundary cache clear over implicit KV state."""

from __future__ import annotations

import torch

from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import dtype_is, shape
from verity_vllm.program.frontend.rules.vllm_bindings.operands import BF16


# -- the padded engine's step-boundary cache clear over implicit KV state ----------------------------------------------------


class CachePadRule(StateRule):
    """`vllm_iface.unified_cache_pad(live, zero, slot)`: for EVERY layer whose KV history this engine owns, row `slot` — the
    row the step just appended, which must be the last row of every history — now denotes `batch(SelectBf16_v1, live, row,
    zero-row)` against the step's one broadcast zero (serve3-spec §6 row muxes; the `Clear_OOB_MN` zero-row convention). The
    step's own attention has read the raw row (graph order); every later reader reads the muxed one. The caller must be an
    ancestor module of every history's owner (the engine owns its model's caches). A history with a different row count, a
    non-bf16/non-scalar zero, or a non-bit liveness is refused — nothing is synthesised for a history this call does not fit."""
    modes = frozenset({"profile"})
    info = RuleInfo("CachePad[vllm]", 1, "state", applies_to="vllm_iface.unified_cache_pad(live, zero, slot) -> ()",
                    preconditions=("live is a 1-element bool value collection (or a statically known bool)", "zero is a 1-element bf16 value collection",
                                   "every KV history has exactly slot+1 rows and is owned by a descendant of the caller's module",
                                   "at least one KV history exists"),
                    evidence_domain="structural; numerics: SelectBf16_v1 (exact-model-tested) per K/V word")

    def targets(self):
        from verity_vllm.program.frontend.rules.vllm_iface import targets_for
        return targets_for("unified_cache_pad")

    def applies(self, node, ctx):
        live, zero, slot = node.args[:3]
        from verity_vllm.program.frontend.rules.padding import _bit_value, _static_bit
        if _bit_value(ctx, live) is None and _static_bit(ctx, live) is None:
            return "live is neither a bool value collection nor a statically known bool tensor"
        if not isinstance(zero, torch.fx.Node) or not isinstance(ctx.val(zero), Coll) or not dtype_is(ctx, zero, BF16):
            return "zero is not a bf16 value collection"
        if int(torch.Size(shape(ctx, zero)).numel()) != 1:
            return f"zero has {shape(ctx, zero)} elements, not one"
        if not isinstance(slot, int):
            return "slot is not static"
        hist = ctx.state.get("kv_history", {})
        if not hist:
            return "no KV history exists yet (the cache clear precedes every attention)"
        return True

    def emit(self, node, ctx):
        from verity_vllm.program.frontend.rules.padding import _materialize_bits, _scalar_of
        live, zero, slot = node.args[:3]
        mp = ctx.module_path(node)
        hist = ctx.state["kv_history"]
        owner = ctx.state.get("kv_owner", {})
        c = _materialize_bits(ctx, live, self, node)
        cflat = c.reshape(Array(int(torch.Size(shape(ctx, live)).numel()), Value(1)))
        if cflat.type.n != 1:
            refuse("shape", node, mp, f"live has {cflat.type.n} elements; the step's cache clear is keyed on one bit")
        zc = ctx.coll(zero)
        zscalar = zc.reshape(Array(1, Value(16)))
        fn = ctx.kind("select_bf16", node)
        layers = []
        def _path(p) -> str:  # "a.b.c (Cls)" -> "a.b.c"; "<root> (Cls)" / None -> ""
            p = (p or "").split(" (", 1)[0]
            return "" if p in ("", "<root>") else p

        me = _path(mp)
        for layer_name in sorted(hist):
            own = _path(owner.get(layer_name))
            if not (me == "" or own == me or own.startswith(me + ".")):
                refuse("mutation", node, mp, f"KV history {layer_name!r} is owned by {owner.get(layer_name)!r}, not a descendant of {mp!r}")
            krow_list, vrow_list = hist[layer_name]
            if slot != len(krow_list) - 1:
                refuse("mutation", node, mp, f"unified_cache_pad rewrites row {slot} of {layer_name!r} whose history has {len(krow_list)} rows: "
                                             f"only the row this step appended (row {len(krow_list) - 1}) may be rewritten at the step boundary")
            for which, rows in (("k", krow_list), ("v", vrow_list)):
                row = rows[slot]
                n = row.type.leaves
                flat = row.reshape(Array(n, Value(16)))
                zrow = Coll(Array(n, Value(16)), concat([zscalar[0].refs] * n))
                out = ctx.B.batch(fn, _scalar_of(cflat), flat, zrow, axes=(None, 0, 0), name=f"{ctx.hint_name(node)}/{layer_name}/{which}[{slot}]")
                k = len(ctx.B.nodes) - 1
                ctx.node_hints[k] = f"{ctx.hint_name(node)}/{layer_name}/{which}[{slot}]"
                ctx.node_sources[k] = node.name
                rows[slot] = out.reshape(row.type)
            layers.append(layer_name)
            ctx.state.setdefault("kv_rewrites", []).append({"layer_name": layer_name, "slot": slot, "node": node.name, "pad": True})
        ctx.record(self, node, {"slot": slot, "layers": len(layers), "row_words": hist[layers[0]][0][slot].type.leaves},
                   emitted=f"batch(SelectBf16_v1, live, row, zero-row) x{2 * len(layers)} (one K and one V row per layer; state rows rebound)")
