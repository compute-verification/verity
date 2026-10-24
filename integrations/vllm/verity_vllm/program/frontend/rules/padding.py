"""Padded-serving rules (profile mode; PAD-INT stage-1 requirements, PAD-SPEC v1 §4–§7).

A padded serving module represents the realized generation length as *values*: a per-event live bit derived from
EOS detection, token slots muxed against the pad token, and every decode step's appended K/V row muxed against one
zero gate. These rules bind the ordinary torch ops such a module uses to the `pad-serve@1` kinds exactly as
`verity_vllm/program/registry/serve3.py` uses them — nothing is synthesised: a guard the source does not compute is not
derived, and a mux inside a step body is derived where it is (and then is not `Serve_v3`).

    ones/zeros/full/scalar_tensor (numel 1, bf16 / bool)   -> Const<w>[bits]            (`ConstScalar`)
    eq.Scalar(y: i32 value, EOS)                           -> Const32[EOS], I32Eq       (`EqScalar`)
    bitwise_not / logical_not (bit value)                  -> BitNot                    (`BitNot`)
    __and__ / bitwise_and / logical_and (bit values;       -> BitAnd (+ Const1 for a    (`BitAnd`)
        a statically known bool operand is a constant)         static operand)
    where(live, a: i32, b | scalar)                        -> SelectI32 per element     (`Select`)
    where(live, a: bf16 row, zero)                         -> batch(SelectBf16, live, row, zero-row)
    unified_cache_write(k, v, layer_name, slot)            -> the layer's history row `slot` (the row this step
        appended) now denotes the written values           (`CacheRowWrite`, a StateRule)
"""

from __future__ import annotations

import math

import torch

from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.torch_frontend import _tensor_type, shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo, StateRule
from verity_vllm.program.frontend.rules.common import shape
from verity_vllm.program.frontend.rules.vllm_iface import targets_for

aten = torch.ops.aten

_PROFILE = frozenset({"profile"})


def _numel(shp) -> int:
    return math.prod(shp) if shp else 1


def _leaf_width(c: Coll) -> int:
    return shape_of(c.type)[1]


def _bf16_bits(v: float) -> int:
    return int(torch.tensor(float(v), dtype=torch.bfloat16, device="cpu").view(torch.int16).item()) & 0xFFFF


def _scalar_of(c: Coll) -> Coll:
    """The single leaf of a numel-1 collection as a scalar-typed collection."""
    return Coll(Value(_leaf_width(c)), c.refs)


class ConstScalarRule(OpRule):
    """A tensor of one element made from a literal (`zeros(())`, `ones(1, dtype=bool)`, `full`, `scalar_tensor`) that
    flows into a *value* computation is a constant gate `Const<w>[bits]` (`b1.const`). Integer literals stay static
    structure (`StaticIndex`) unless a value rule needs them (see `BitAnd`); bf16 and bool literals become gates."""
    modes = _PROFILE
    info = RuleInfo("ConstScalar[pad]", 1, "op-binding", applies_to="aten.zeros/ones/full/scalar_tensor with numel 1, dtype bf16",
                    preconditions=("numel is 1", "dtype is bf16 (a 16-bit constant gate); bool/int literals are static structure first"),
                    evidence_domain="structural: Const16[bits] — the profile's constant gate (b1.const)")

    def targets(self):
        return {aten.zeros.default, aten.ones.default, aten.full.default, aten.scalar_tensor.default}

    def applies(self, node, ctx):
        dt = ctx.dtype_of(node)
        shp = ctx.shape_of_node(node)
        if dt != torch.bfloat16:
            return f"dtype {dt}: only bf16 literals become constant gates here"
        if shp is None or _numel(shp) != 1:
            return f"shape {shp}: only one-element literals are constant gates"
        return True

    def emit(self, node, ctx):
        shp = tuple(ctx.shape_of_node(node))
        if node.target is aten.zeros.default:
            v = 0.0
        elif node.target is aten.ones.default:
            v = 1.0
        elif node.target is aten.full.default:
            v = float(node.args[1])
        else:
            v = float(node.args[0])
        bits = _bf16_bits(v)
        fn = ctx.kind("const", node, W=16, BITS=bits)
        c = ctx.call(fn, node=node, rule=self, name=ctx.hint_name(node))
        ctx.define(node, c.reshape(_tensor_type(shp, 16)) if shp else c)
        ctx.record(self, node, {"value": v, "bits": f"0x{bits:04x}", "numel": 1}, emitted=f"Const16[0x{bits:04x}]")


class EqScalarRule(OpRule):
    """`y == EOS` on an i32 token value: `I32Eq_v1(y, Const32[EOS])` per element — the stop test of `PadGuard_v1`
    (spec §4). The literal is one constant gate per `eq` node (per guard, as in `serve3.py`)."""
    modes = _PROFILE
    info = RuleInfo("EqScalar[pad]", 1, "op-binding", applies_to="aten.eq.Scalar(y, int) with y an int32 value collection",
                    preconditions=("y is a value collection with 32-bit integer leaves (a TokenSelect output or an int32 token input)",
                                   "the literal is a Python int (a static EOS id)"),
                    evidence_domain="structural; numerics: I32Eq_v1 (exact-model-tested)")

    def targets(self):
        return {aten.eq.Scalar}

    def applies(self, node, ctx):
        y, lit = node.args[0], node.args[1]
        v = ctx.val(y) if isinstance(y, torch.fx.Node) else None
        if not isinstance(v, Coll):
            return "left operand is not a value collection"
        if ctx.dtype_of(y) not in (torch.int32, torch.int64) or _leaf_width(v) != 32:
            # (a TokenSelect output carries I32 leaves whatever torch's index dtype says)
            return f"left operand dtype {ctx.dtype_of(y)} / {_leaf_width(v)}-bit leaves: I32Eq takes i32 tokens"
        if isinstance(lit, bool) or not isinstance(lit, int):
            return f"literal {lit!r} is not an int"
        return True

    def emit(self, node, ctx):
        y, lit = node.args[0], int(node.args[1])
        yc = ctx.coll(y)
        shp = tuple(shape_of(yc.type)[0])
        n = _numel(shp)
        flat = yc.reshape(Array(n, Value(32))) if shp else yc.reshape(Array(1, Value(32)))
        c = ctx.call(ctx.kind("const", node, W=32, BITS=lit & 0xFFFFFFFF), node=node, rule=self, name=f"{ctx.hint_name(node)}/eos")
        eq = ctx.kind("i32_eq", node)
        outs = [ctx.call(eq, flat[i], c, node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i in range(n)]
        out = array_of(outs)
        ctx.define(node, out.reshape(_tensor_type(shp, 1)) if shp else Coll(Value(1), out.refs))
        ctx.record(self, node, {"literal": lit, "elements": n}, emitted=f"Const32[{lit}] + I32Eq_v1 x{n}")


def _bit_value(ctx, a) -> Coll | None:
    if not isinstance(a, torch.fx.Node):
        return None
    v = ctx.val(a)
    if isinstance(v, Coll) and ctx.dtype_of(a) == torch.bool and _leaf_width(v) == 1:
        return v
    return None


def _static_bit(ctx, a) -> StaticIndex | None:
    """A statically known bool tensor (`torch.ones(1, dtype=bool)`: the initial liveness)."""
    sv = ctx.val(a) if isinstance(a, torch.fx.Node) else None
    return sv if isinstance(sv, StaticIndex) and sv.tensor.dtype == torch.bool else None


def _materialize_bits(ctx, a, rule, node) -> Coll:
    """The constant gates of a statically known bool tensor — emitted ONCE: the fx node is redefined as the value
    collection, so every later reader (the guard's `BitAnd` and `SelectI32` both read `live_0`) shares the gates
    (`live_0 = Const1[0x1]`, one gate, spec §3)."""
    sv = _static_bit(ctx, a)
    if sv is None:
        return ctx.coll(a)
    shp = tuple(sv.shape)
    gates = [ctx.call(ctx.kind("const", node, W=1, BITS=int(bool(b))), node=node, rule=rule, name=f"{ctx.hint_name(a)}/const") for b in sv.flat()]
    out = array_of(gates)
    c = out.reshape(_tensor_type(shp, 1)) if shp else Coll(Value(1), out.refs)
    ctx.define(a, c)
    ctx.record(rule, a, {"static_bits": [int(bool(b)) for b in sv.flat()]}, emitted=f"Const1 x{len(gates)} (a statically known bool read as a value)")
    return c


class BitNotRule(OpRule):
    modes = _PROFILE
    info = RuleInfo("BitNot[pad]", 1, "op-binding", applies_to="aten.bitwise_not / logical_not on a bool value collection",
                    preconditions=("the operand is a value collection of 1-bit leaves (bool)",),
                    evidence_domain="structural; numerics: BitNot_v1 (pad_prims)")

    def targets(self):
        return {aten.bitwise_not.default, aten.logical_not.default}

    def applies(self, node, ctx):
        return True if _bit_value(ctx, node.args[0]) is not None else "operand is not a bool value collection"

    def emit(self, node, ctx):
        x = _bit_value(ctx, node.args[0])
        shp = tuple(shape_of(x.type)[0])
        n = _numel(shp)
        flat = x.reshape(Array(n, Value(1)))
        fn = ctx.kind("bit_not", node)
        outs = [ctx.call(fn, flat[i], node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]" if n > 1 else ctx.hint_name(node)) for i in range(n)]
        out = array_of(outs)
        ctx.define(node, out.reshape(_tensor_type(shp, 1)) if shp else Coll(Value(1), out.refs))
        ctx.record(self, node, {"elements": n}, emitted=f"BitNot_v1 x{n}")


class BitAndRule(OpRule):
    """`live & ~stop`: `BitAnd_v1` per element. A statically known bool operand (`torch.ones(1, dtype=bool)`, the
    initial liveness) becomes the constant gate `Const1[bit]` — `live_0 = Const1[0x1]` (spec §3)."""
    modes = _PROFILE
    info = RuleInfo("BitAnd[pad]", 1, "op-binding", applies_to="aten.__and__.Tensor / bitwise_and.Tensor / logical_and on bool operands",
                    preconditions=("each operand is a bool value collection or a statically known bool tensor (-> Const1 gate)",
                                   "at least one operand is a value", "shapes agree or an operand has one element (broadcast)"),
                    evidence_domain="structural; numerics: BitAnd_v1 (pad_prims, absorbing)")

    def targets(self):
        return {aten.__and__.Tensor, aten.bitwise_and.Tensor, aten.logical_and.default}

    def _operand(self, ctx, a):
        v = _bit_value(ctx, a)
        return v if v is not None else _static_bit(ctx, a)

    def applies(self, node, ctx):
        ops = [self._operand(ctx, a) for a in node.args[:2]]
        if any(o is None for o in ops):
            return "an operand is neither a bool value collection nor a statically known bool tensor"
        if not any(isinstance(o, Coll) for o in ops):
            return "no value operand (a static & static is static structure)"
        shapes = [tuple(shape_of(o.type)[0]) if isinstance(o, Coll) else tuple(o.shape) for o in ops]
        if shapes[0] != shapes[1] and _numel(shapes[0]) != 1 and _numel(shapes[1]) != 1:
            return f"shapes {shapes} neither agree nor broadcast from one element"
        return True

    def emit(self, node, ctx):
        shp = tuple(ctx.shape_of_node(node))
        n = _numel(shp)
        flats = []
        consts = []
        for a in node.args[:2]:
            if _static_bit(ctx, a) is not None:
                consts += [int(bool(b)) for b in _static_bit(ctx, a).flat()]
            o = _materialize_bits(ctx, a, self, node)
            m = _numel(tuple(shape_of(o.type)[0]))
            f = o.reshape(Array(m, Value(1)))
            flats.append([f[i] for i in range(m)] if m == n else [f[0]] * n)
        fn = ctx.kind("bit_and", node)
        outs = [ctx.call(fn, flats[0][i], flats[1][i], node=node, rule=self, name=f"{ctx.hint_name(node)}[{i}]" if n > 1 else ctx.hint_name(node))
                for i in range(n)]
        out = array_of(outs)
        ctx.define(node, out.reshape(_tensor_type(shp, 1)) if shp else Coll(Value(1), out.refs))
        ctx.record(self, node, {"elements": n, "static_operand_bits": consts}, emitted=f"BitAnd_v1 x{n}" + (f" + Const1 x{len(consts)}" if consts else ""))


class SelectRule(OpRule):
    """`torch.where(live, a, b)` keyed by a liveness value: i32 tokens -> `SelectI32_v1` per element (the guard's
    `tok_out = SelectI32(live, y, Const32[0])`); bf16 rows -> `batch(SelectBf16_v1, live, row, zero_row, axes=(None,0,0))`
    against ONE broadcast constant gate (the `Clear_OOB_MN` zero-row convention, spec §6)."""
    modes = _PROFILE
    info = RuleInfo("Select[pad]", 1, "op-binding", applies_to="aten.where.self / where.ScalarOther with a bool value condition",
                    preconditions=("the condition is a bool value collection with one element (broadcast) or the output's shape",
                                   "`a` is an int32 or bf16 value collection of the output's shape",
                                   "`b` is a value collection of the same leaf width with one element or the output's shape, or an int literal (i32)"),
                    evidence_domain="structural; numerics: SelectI32_v1 / SelectBf16_v1 (exact-model-tested)")

    def targets(self):
        return {aten.where.self, aten.where.ScalarOther}

    def applies(self, node, ctx):
        cond, a, b = node.args[:3]
        c = _bit_value(ctx, cond)
        sc = _static_bit(ctx, cond)
        if c is None and sc is None:
            return "condition is neither a bool value collection nor a statically known bool tensor"
        av = ctx.val(a) if isinstance(a, torch.fx.Node) else None
        if not isinstance(av, Coll):
            return "`a` is not a value collection"
        out_shape = tuple(ctx.shape_of_node(node))
        if tuple(shape_of(av.type)[0]) != out_shape:
            return f"`a` shape {shape_of(av.type)[0]} is not the output shape {out_shape} (broadcast of `a` is not bound)"
        cshape = tuple(shape_of(c.type)[0]) if c is not None else tuple(sc.shape)
        if _numel(cshape) != 1 and _numel(cshape) != _numel(out_shape):
            return f"condition shape {cshape} is neither one element nor the output's {out_shape}"
        dt = ctx.dtype_of(a)
        if dt in (torch.int32, torch.int64) and _leaf_width(av) == 32:
            if node.target is aten.where.ScalarOther:
                if isinstance(b, bool) or not isinstance(b, int):
                    return f"i32 select with a non-int literal {b!r}"
                return True
            bv = ctx.val(b) if isinstance(b, torch.fx.Node) else None
            if not isinstance(bv, Coll) or _leaf_width(bv) != 32 or ctx.dtype_of(b) not in (torch.int32, torch.int64):
                return "`b` is not an i32 value collection"
        elif dt == torch.bfloat16:
            if node.target is aten.where.ScalarOther:
                return "bf16 select against a literal: make the pad value one tensor per step (one zero gate shared by the step's muxes)"
            bv = ctx.val(b) if isinstance(b, torch.fx.Node) else None
            if not isinstance(bv, Coll) or _leaf_width(bv) != 16:
                return "`b` is not a bf16 value collection"
        else:
            return f"numerics_unregistered: select on dtype {dt} has no pad-serve kind (SelectI32 / SelectBf16 only)"
        if isinstance(b, torch.fx.Node):
            bs = tuple(shape_of(ctx.val(b).type)[0])
            if _numel(bs) != 1 and _numel(bs) != _numel(out_shape):
                return f"`b` shape {bs} is neither one element nor the output's {out_shape}"
        return True

    def emit(self, node, ctx):
        cond, a, b = node.args[:3]
        c = _materialize_bits(ctx, cond, self, node)
        ac = ctx.coll(a)
        shp = tuple(ctx.shape_of_node(node))
        n = _numel(shp)
        w = _leaf_width(ac)
        aflat = ac.reshape(Array(n, Value(w)))
        cflat = c.reshape(Array(_numel(tuple(shape_of(c.type)[0])), Value(1)))
        if w == 32:
            if isinstance(b, torch.fx.Node):
                bc = ctx.coll(b)
                bflat = bc.reshape(Array(_numel(tuple(shape_of(bc.type)[0])), Value(32)))
                bsel = [bflat[i] if bflat.type.n == n else bflat[0] for i in range(n)]
                emitted_b = "value"
            else:
                pad = ctx.call(ctx.kind("const", node, W=32, BITS=int(b) & 0xFFFFFFFF), node=node, rule=self, name=f"{ctx.hint_name(node)}/pad")
                bsel = [pad] * n
                emitted_b = f"Const32[{int(b)}]"
            fn = ctx.kind("select_i32", node)
            outs = [ctx.call(fn, cflat[i] if cflat.type.n == n else cflat[0], aflat[i], bsel[i], node=node, rule=self,
                             name=f"{ctx.hint_name(node)}[{i}]" if n > 1 else ctx.hint_name(node)) for i in range(n)]
            out = array_of(outs)
            ctx.define(node, out.reshape(_tensor_type(shp, 32)) if shp else Coll(Value(32), out.refs))
            ctx.record(self, node, {"elements": n, "leaf": "i32", "b": emitted_b}, emitted=f"SelectI32_v1 x{n}")
            return
        bc = ctx.coll(b)
        bn = _numel(tuple(shape_of(bc.type)[0]))
        brow = bc.reshape(Array(bn, Value(16)))
        if bn != n:
            brow = Coll(Array(n, Value(16)), concat([brow[0].refs] * n))
        fn = ctx.kind("select_bf16", node)
        if cflat.type.n == n:
            out = ctx.B.batch(fn, cflat, aflat, brow, axes=(0, 0, 0), name=ctx.hint_name(node))
            axes = (0, 0, 0)
        else:
            out = ctx.B.batch(fn, _scalar_of(cflat), aflat, brow, axes=(None, 0, 0), name=ctx.hint_name(node))
            axes = (None, 0, 0)
        k = len(ctx.B.nodes) - 1
        ctx.node_hints[k] = ctx.hint_name(node)
        ctx.node_sources[k] = node.name
        ctx.define(node, out.reshape(_tensor_type(shp, 16)))
        ctx.record(self, node, {"elements": n, "leaf": "bf16", "b_broadcast_from_one": bn != n, "axes": list(map(str, axes))},
                   emitted=f"batch(SelectBf16_v1, n={n}, axes={axes})")


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


PADDING_RULES = [ConstScalarRule(), EqScalarRule(), BitNotRule(), BitAndRule(), SelectRule(), CacheRowWriteRule()]

__all__ = ["PADDING_RULES"]
