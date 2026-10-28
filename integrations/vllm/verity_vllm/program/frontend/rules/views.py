"""View rules: reference rearrangements (no gates) and static structure (index tensors, allocations).

Every rule here emits NO gate. Views reuse the frontend's reference builder (`strided_views._view` & co.), so a
derived view is the very same affine/strided descriptor an authored program would carry.
"""

from __future__ import annotations

import operator

import torch

from verity.ml.circuits.refs import Coll, array_of, concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.strided_views import _dim_len, _expand, _norm_dim, _permute, _select, _slice, _stack, _tensor_type
from verity_vllm.program.frontend.value_types import shape_of, type_of_tensor, width_of
from verity_vllm.program.frontend.derive import Alloc, StaticIndex, refuse
from verity_vllm.program.frontend.rules.base import RuleInfo, ViewRule

aten = torch.ops.aten

_IDENTITY = {aten.alias.default, aten.detach.default, aten.clone.default, aten.contiguous.default, aten.lift_fresh.default,
             aten._assert_tensor_metadata.default}
_RESHAPE = {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints, aten.squeeze.dim, aten.squeeze.default,
            aten.squeeze.dims, aten.unsqueeze.default}
_SPLITS = {aten.split_with_sizes.default, aten.split.Tensor, aten.chunk.default, aten.unbind.int}
_STATIC_MAKERS = {aten.arange.default, aten.arange.start, aten.arange.start_step, aten.full.default, aten.zeros.default, aten.ones.default,
                  aten.scalar_tensor.default, aten.tensor.int, aten.tensor.float}
_STATIC_ARITH = {aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.floor_divide.default, aten.remainder.Tensor, aten.add.Scalar,
                 aten.sub.Scalar, aten.mul.Scalar, aten.neg.default, aten.clamp.default, aten.minimum.default, aten.maximum.default,
                 aten.cumsum.default, aten.to.dtype, aten._to_copy.default, aten.repeat.default, aten.tile.default, aten.flip.default,
                 aten.tril.default, aten.triu.default, aten.bitwise_not.default, aten.logical_not.default, aten.eq.Tensor, aten.ne.Tensor,
                 aten.lt.Tensor, aten.le.Tensor, aten.gt.Tensor, aten.ge.Tensor, aten.eq.Scalar, aten.ne.Scalar, aten.lt.Scalar, aten.le.Scalar,
                 aten.gt.Scalar, aten.ge.Scalar, aten.logical_and.default, aten.logical_or.default, aten.bitwise_and.Tensor, aten.bitwise_or.Tensor,
                 aten.where.self,
                 # transformers 5.x `masking_utils` plumbing on static positions (the `new_ones` / `to.dtype_layout` forms are the same
                 # static-structure class): `find_packed_sequence_indices` = diff -> ne -> cumsum, the
                 # mask functions = `q_idx.new_ones(())`, `__and__`, `packed[b, kv_idx] == packed[b, q_idx]` (index with static indices),
                 # `.to(dtype=bool, layout=strided, device=…)`; `.to(device=…, dtype=…)` of a static mask. All integral/bool.
                 aten.diff.default, aten.__and__.Tensor, aten.__or__.Tensor, aten.__xor__.Tensor, aten.index.Tensor, aten.to.dtype_layout,
                 aten.to.device}
# `x.new_ones(shape) / new_zeros / new_full`: constant makers whose ONE tensor operand only supplies dtype/device defaults (never read)
_STATIC_LIKE_MAKERS = {aten.new_ones.default, aten.new_zeros.default, aten.new_full.default}
_ALLOCS = {aten.empty.memory_format, aten.empty_like.default, aten.new_empty.default, aten.empty_strided.default}
_GATHERS = {aten.index_select.default, aten.index.Tensor, aten.embedding.default}


def _is_int(dt: torch.dtype) -> bool:
    return dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)


def _wrap_views(node, ctx, values):
    """Apply a torch op to StaticIndex operands on real CPU tensors; the result stays static structure."""
    def unwrap(a):
        if isinstance(a, StaticIndex):
            return a.tensor
        if isinstance(a, (list, tuple)):
            return [unwrap(x) for x in a]
        return a
    args = [unwrap(ctx.val(a)) for a in node.args]
    kwargs = {k: unwrap(ctx.val(v)) for k, v in node.kwargs.items()}
    kwargs.pop("device", None)
    kwargs.pop("pin_memory", None)
    with torch.no_grad():
        if node.target is aten.to.device:
            # to.device(self, device, dtype, non_blocking?, copy?): the device is not semantics — only the dtype is applied
            out = args[0].to(dtype=args[2])
        elif node.target is aten.to.dtype_layout:
            # to.dtype_layout(self, *, dtype, layout, device, …): strided CPU tensors here; only the dtype is applied
            out = args[0].to(dtype=kwargs.get("dtype") or args[0].dtype)
        else:
            out = node.target(*args, **kwargs)
    if isinstance(out, (list, tuple)):
        return [StaticIndex(o.cpu()) for o in out]
    return StaticIndex(out.cpu())


class StaticIndexRule(ViewRule):
    """Integer tensors made from constants (`arange`, `full`, ...) or by integer arithmetic/views on such tensors
    are static structure (positions, slot tables): they never become gates. v2 (transformers 5.x plumbing):
    also `diff`, `__and__/__or__/__xor__`, `index.Tensor` with static indices into a static table, `new_ones/new_zeros/new_full`
    of a static tensor, and `to.dtype_layout` / `to.device` (device is not semantics; the dtype is applied) — every tensor
    operand static, the result integral or bool, exactly as before."""
    info = RuleInfo("StaticIndex", 2, "view", applies_to="aten.arange/full/zeros/ones/new_ones/new_zeros/new_full + integer/bool arithmetic "
                                                         "(incl. diff, cumsum, comparisons, __and__/__or__, index with static indices, "
                                                         "to.dtype/to.dtype_layout/to.device) & views over StaticIndex operands",
                    preconditions=("every tensor operand is a StaticIndex (statically known integers)", "the result dtype is integral or bool",
                                   "no operand is a value (Coll) — data never flows into static structure",
                                   "to.device / to.dtype_layout: only the dtype is applied (device/layout are not semantics)"),
                    evidence_domain="structural: the values are computed by torch on real CPU tensors at derivation time")

    def targets(self):
        return _STATIC_MAKERS | _STATIC_LIKE_MAKERS | _STATIC_ARITH | _RESHAPE | _IDENTITY | {aten.slice.Tensor, aten.select.int, aten.cat.default,
                                                                                              aten.stack.default, aten.expand.default, aten.permute.default,
                                                                                              aten.t.default, aten.transpose.int} | _SPLITS

    def applies(self, node, ctx):
        dt = ctx.dtype_of(node)
        if dt is None:
            v = node.meta.get("val")
            if isinstance(v, (list, tuple)) and v and all(_is_int(x.dtype) for x in v):
                pass
            else:
                return "result is not a tensor"
        elif not _is_int(dt):
            return f"result dtype {dt} is not integral"
        tensors = [a for a in node.args if isinstance(a, torch.fx.Node)] + [a for a in node.kwargs.values() if isinstance(a, torch.fx.Node)]
        for a in node.args:
            if isinstance(a, (list, tuple)):
                tensors += [x for x in a if isinstance(x, torch.fx.Node)]
        if node.target in _STATIC_MAKERS:
            if tensors:
                return "constant maker with tensor operands"
            return True
        if not tensors:
            return "no tensor operand"
        for a in tensors:
            v = ctx.val(a)
            if isinstance(v, StaticIndex):
                continue
            if isinstance(v, (list, tuple)) and all(isinstance(x, StaticIndex) for x in v):
                continue
            return f"operand {a.name} is not a StaticIndex"
        return True

    def emit(self, node, ctx):
        out = _wrap_views(node, ctx, None)
        ctx.define(node, out, view_of=None)
        ctx.record(self, node, {"integral": True, "all_static": True}, None, values=(out.flat()[:16] if isinstance(out, StaticIndex) else None))


class IdentityViewRule(ViewRule):
    info = RuleInfo("IdentityView", 1, "view", applies_to="aten.alias/detach/clone/contiguous/lift_fresh/_assert_tensor_metadata",
                    preconditions=("operand is a value collection",), evidence_domain="structural (no gates)")

    def targets(self):
        return _IDENTITY

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if isinstance(v, StaticIndex) and node.target is aten._assert_tensor_metadata.default:
            return True  # a metadata assertion on static structure is not a read
        return True if isinstance(v, Coll) else f"operand is {type(v).__name__}"

    def emit(self, node, ctx):
        v = ctx.val(node.args[0])
        if isinstance(v, StaticIndex):
            ctx.env[node] = v
            return
        x = ctx.coll(node.args[0])
        ctx.define(node, x, view_of=None if node.target is aten.clone.default else node.args[0])


class ReshapeRule(ViewRule):
    info = RuleInfo("Reshape", 1, "view", applies_to="aten.view/reshape/_unsafe_view/flatten/squeeze/unsqueeze",
                    preconditions=("operand is a value collection", "leaf count is preserved"), evidence_domain="structural (no gates)")

    def targets(self):
        return _RESHAPE

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        return True if isinstance(v, Coll) else f"operand is {type(v).__name__}"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        # the Verity width is the collection's own (a TokenSelect output is I32 whatever torch's index dtype says)
        w = shape_of(x.type)[1]
        try:
            ctx.define(node, x.reshape(_tensor_type(tuple(node.meta["val"].shape), w)), view_of=node.args[0])
        except ValueError as e:
            refuse("shape", node, ctx.module_path(node), str(e))


class ViewDtypeRule(ViewRule):
    """v2: also `aten.to.device(x, device, dtype)` — `.to(device=…, dtype=…)` — a device move is not semantics; only the
    dtype is looked at, exactly as for `to.dtype` (same width: a view; bf16<->f32: `CastF32Bf16`; anything else: refused)."""
    info = RuleInfo("ViewDtype", 2, "view", applies_to="aten.view.dtype / aten.to.dtype / aten.to.dtype_layout / aten.to.device / _to_copy between dtypes of one width",
                    preconditions=("operand is a value collection", "source and target dtypes have the same Value<w> width",
                                   "device / layout arguments are ignored (not semantics)"),
                    evidence_domain="structural: a reinterpretation, not a conversion (conversions are numerical kinds)")

    def targets(self):
        return {aten.view.dtype, aten.to.dtype, aten._to_copy.default, aten.to.dtype_layout, aten.to.device}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        src = ctx.dtype_of(node.args[0])
        dst = ctx.dtype_of(node)
        if src is None or dst is None:
            return "dtypes unknown"
        if width_of(src) != width_of(dst):
            # an int64 torch tensor whose *value* already has 32-bit leaves (an I32 token from ArgmaxRef/TokenSelect) cast to int32:
            # the value is what it is — a reinterpretation, not a conversion
            if not dst.is_floating_point and not src.is_floating_point and shape_of(v.type)[1] == width_of(dst):
                return True
            return f"numerics_unregistered: {src} -> {dst} changes the width; a conversion kind is needed"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=node.args[0])


class SliceSelectRule(ViewRule):
    info = RuleInfo("SliceSelect", 1, "view", applies_to="aten.slice.Tensor / aten.select.int / aten.narrow",
                    preconditions=("operand is a value collection", "unit stride"), evidence_domain="structural (no gates)")

    def targets(self):
        return {aten.slice.Tensor, aten.select.int, aten.narrow.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        if node.target is aten.slice.Tensor and len(node.args) > 4 and node.args[4] != 1:
            return "unsupported_view: step != 1"
        return True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        where = ctx.hint_name(node)
        try:
            if node.target is aten.select.int:
                dim = _norm_dim(node.args[1], x.type)
                i = node.args[2]
                i = i + _dim_len(x.type, dim) if i < 0 else i
                out = _select(x, dim, i, where)
            elif node.target is aten.narrow.default:
                dim = _norm_dim(node.args[1], x.type)
                out = _slice(x, dim, node.args[2], node.args[2] + node.args[3], 1, where)
            else:
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                lo = node.args[2] if len(node.args) > 2 else None
                hi = node.args[3] if len(node.args) > 3 else None
                out = _slice(x, dim, lo, hi, 1, where)
        except FrontendError as e:
            refuse("unsupported_view", node, ctx.module_path(node), e.message)
        ctx.define(node, out, view_of=node.args[0])


class SplitRule(ViewRule):
    info = RuleInfo("Split", 1, "view", applies_to="aten.split_with_sizes / split.Tensor / chunk / unbind",
                    preconditions=("operand is a value collection",), evidence_domain="structural (no gates)")

    def targets(self):
        return _SPLITS

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        return True if isinstance(v, Coll) else f"operand is {type(v).__name__}"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        where = ctx.hint_name(node)
        t = node.target
        try:
            if t is aten.unbind.int:
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                parts = [_select(x, dim, i, where) for i in range(_dim_len(x.type, dim))]
            else:
                dim = _norm_dim(node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0), x.type)
                n = _dim_len(x.type, dim)
                if t is aten.split_with_sizes.default:
                    sizes = list(node.args[1])
                elif t is aten.split.Tensor:
                    s = node.args[1]
                    sizes = [s] * (n // s) + ([n % s] if n % s else [])
                else:
                    chunks = node.args[1]
                    s = -(-n // chunks)
                    sizes = [s] * (n // s) + ([n % s] if n % s else [])
                parts, lo = [], 0
                for s in sizes:
                    parts.append(_slice(x, dim, lo, lo + s, 1, where))
                    lo += s
        except FrontendError as e:
            refuse("unsupported_view", node, ctx.module_path(node), e.message)
        ctx.define(node, parts, view_of=node.args[0])


def _cat(parts: list[Coll], dim: int, where: str) -> Coll:
    if dim == 0:
        elem = parts[0].type.elem
        if any(p.type.elem != elem for p in parts):
            raise FrontendError("E_SHAPE", "cat of arrays with different element types", where)
        return Coll(Array(sum(p.type.n for p in parts), elem), concat([p.refs for p in parts]))
    n = parts[0].type.n
    if any(p.type.n != n for p in parts):
        raise FrontendError("E_SHAPE", "cat: leading dims differ", where)
    return array_of([_cat([p[i] for p in parts], dim - 1, where) for i in range(n)])


class CatStackRule(ViewRule):
    info = RuleInfo("CatStack", 1, "view", applies_to="aten.cat / aten.stack",
                    preconditions=("every part is a value collection", "parts agree in shape off the cat axis"),
                    evidence_domain="structural (no gates); inner-axis cat interleaves references (O(rows) descriptor)")

    def targets(self):
        return {aten.cat.default, aten.stack.default}

    def applies(self, node, ctx):
        parts = ctx.val(node.args[0])
        if not isinstance(parts, list) or not all(isinstance(p, Coll) for p in parts):
            return "a part is not a value collection"
        if len({shape_of(p.type)[1] for p in parts}) > 1:
            return "parts have different leaf widths (a cat of values is a view only when the element types agree)"
        return True

    def emit(self, node, ctx):
        parts = [ctx.coll(a) for a in node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        where = ctx.hint_name(node)
        nd = len(shape_of(type_of_tensor(node.meta["val"]))[0])
        if dim < 0:
            dim += nd
        try:
            if node.target is aten.stack.default:
                if dim != 0:
                    parts = [p.reshape(Array(1, p.type)) for p in parts]
                    out = _cat(parts, dim, where)
                else:
                    out = _stack(parts, where)
            else:
                out = _cat(parts, dim, where)
        except FrontendError as e:
            refuse("unsupported_view", node, ctx.module_path(node), e.message)
        ctx.define(node, out, view_of=None)
        for a in node.args[0]:
            if isinstance(a, torch.fx.Node):
                ctx.storage.setdefault(node, ctx.storage_base(a))  # conservatively alias the first part's storage
                break


class PermuteRule(ViewRule):
    info = RuleInfo("Permute", 1, "view", applies_to="aten.permute / t / transpose / numpy_T",
                    preconditions=("operand is a value collection",), evidence_domain="structural (no gates)")

    def targets(self):
        # `aten.numpy_T` is `Tensor.T` (all axes reversed): the FP8 linears pass `weight.T` / `weight_scale_inv.T` to `cutlass_scaled_mm`
        return {aten.permute.default, aten.t.default, aten.transpose.int, aten.numpy_T.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        return True if isinstance(v, Coll) else f"operand is {type(v).__name__}"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        nd = len(shape_of(x.type)[0])
        t = node.target
        if t is aten.permute.default:
            dims = list(node.args[1])
        elif t is aten.t.default:
            dims = [1, 0] if nd == 2 else list(range(nd))
        elif t is aten.numpy_T.default:
            dims = list(range(nd))[::-1]
        else:
            d0, d1 = _norm_dim(node.args[1], x.type), _norm_dim(node.args[2], x.type)
            dims = list(range(nd))
            dims[d0], dims[d1] = dims[d1], dims[d0]
        try:
            ctx.define(node, _permute(x, dims, ctx.hint_name(node)), view_of=node.args[0])
        except FrontendError as e:
            refuse("unsupported_view", node, ctx.module_path(node), e.message)


class ExpandRule(ViewRule):
    info = RuleInfo("Expand", 1, "view", applies_to="aten.expand", preconditions=("operand is a value collection",),
                    evidence_domain="structural (no gates; broadcast = zero stride)")

    def targets(self):
        return {aten.expand.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        return True if isinstance(v, Coll) else f"operand is {type(v).__name__}"

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        try:
            ctx.define(node, _expand(x, list(node.args[1]), ctx.hint_name(node)), view_of=node.args[0])
        except FrontendError as e:
            refuse("unsupported_view", node, ctx.module_path(node), e.message)


class RepeatInterleaveRule(ViewRule):
    """`x.repeat_interleave(r, dim)` (HF `repeat_kv`: every KV head repeated for its query group) is a view: each slice
    along `dim` appears `r` times in order, no gates."""
    info = RuleInfo("RepeatInterleave", 1, "view", applies_to="aten.repeat_interleave.self_int(x, repeats: int, dim: int)",
                    preconditions=("operand is a value collection", "repeats is a static int and dim is given"),
                    evidence_domain="structural (no gates)")

    def targets(self):
        return {aten.repeat_interleave.self_int}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        if len(node.args) < 3 or not isinstance(node.args[1], int) or not isinstance(node.args[2], int):
            return "repeats / dim are not static ints (a flattened repeat_interleave is not a bound view)"
        return True

    def emit(self, node, ctx):
        x = ctx.coll(node.args[0])
        r, dim = int(node.args[1]), int(node.args[2])
        shp = list(shape_of(x.type)[0])
        d = _norm_dim(dim, x.type)
        # bring `dim` to the front, repeat each slice, put it back
        perm = [d] + [i for i in range(len(shp)) if i != d]
        inv = [perm.index(i) for i in range(len(shp))]
        xf = _permute(x, perm, ctx.hint_name(node)) if perm != list(range(len(shp))) else x
        parts = []
        for i in range(shp[d]):
            parts += [xf[i]] * r
        out = array_of(parts)
        out = _permute(out, inv, ctx.hint_name(node)) if inv != list(range(len(shp))) else out
        ctx.define(node, out, view_of=node.args[0])
        ctx.record(self, node, {"repeats": r, "dim": d}, None)


class StaticGatherRule(ViewRule):
    """`table[idx]` with a statically known index tensor is a row selection (a view), e.g. a RoPE table indexed
    by positions or a position-embedding lookup. With a computed index it is NOT this rule (`Embedding` kinds)."""
    info = RuleInfo("StaticGather", 1, "view", applies_to="aten.index_select / index.Tensor / embedding with a StaticIndex index",
                    preconditions=("table is a value collection", "index is a StaticIndex (values statically known)",
                                   "every index is within the table's leading axis"),
                    evidence_domain="structural (no gates): the selected rows are recorded in provenance")

    def targets(self):
        return _GATHERS

    def applies(self, node, ctx):
        t = node.target
        if t is aten.embedding.default:
            table, idx = node.args[0], node.args[1]
        elif t is aten.index_select.default:
            table, idx = node.args[0], node.args[2]
            if node.args[1] not in (0, -len(ctx.shape_of_node(table) or ())):
                return "index_select along a non-leading axis"
        else:
            table, idxs = node.args[0], node.args[1]
            if not isinstance(idxs, (list, tuple)) or len(idxs) != 1 or idxs[0] is None:
                return "index.Tensor with more than one index tensor"
            idx = idxs[0]
        if not isinstance(ctx.val(table), Coll):
            return "table is not a value collection"
        if not isinstance(ctx.val(idx), StaticIndex):
            if isinstance(idx, torch.fx.Node) and ctx.is_input_storage(idx) is not None:
                return "index is a prescribed input value (token ids), not static structure: an Embedding/gather kind must bind it"
            return "dynamic_control_flow: index is a computed value (data-dependent gather), not static structure"
        return True

    def emit(self, node, ctx):
        t = node.target
        if t is aten.embedding.default:
            table, idx = ctx.coll(node.args[0]), ctx.static_index(node.args[1])
        elif t is aten.index_select.default:
            table, idx = ctx.coll(node.args[0]), ctx.static_index(node.args[2])
        else:
            table, idx = ctx.coll(node.args[0]), ctx.static_index(node.args[1][0])
        rows = idx.flat()
        n = table.type.n
        for r in rows:
            if not (0 <= r < n):
                refuse("shape", node, ctx.module_path(node), f"static index {r} outside the table's {n} rows")
        picked = array_of([table[r] for r in rows]) if rows else Coll(Array(0, table.type.elem), concat([]))
        out = picked.reshape(type_of_tensor(node.meta["val"]))
        ctx.define(node, out, view_of=node.args[0])
        w = ctx.weight_of(node.args[0])
        ctx.record(self, node, {"table_is_weight": w.name if w else None, "rows": rows[:64], "n_rows": len(rows)}, None)


class AllocRule(ViewRule):
    """`empty`/`empty_like`: an uninitialised allocation that an in-place rule must write before any read."""
    info = RuleInfo("Alloc", 1, "view", applies_to="aten.empty / empty_like / new_empty / empty_strided",
                    preconditions=("the result is only ever read after an in-place rule wrote it",),
                    evidence_domain="structural: reading an unwritten allocation is refused")

    def targets(self):
        return _ALLOCS

    def applies(self, node, ctx):
        return True

    def emit(self, node, ctx):
        ctx.define(node, Alloc(type_of_tensor(node.meta["val"]), node))


class DropoutEvalRule(ViewRule):
    """`aten.dropout(x, p, train=False)` (an `nn.Dropout` module in eval mode; GPT-2 keeps `drop / attn_dropout / resid_dropout /
    mlp.dropout` in the graph) is the identity: ATen's `dropout` returns `input` itself when `train` is false, whatever `p`.
    No random state is read. `train=True` (or a non-literal `train`) stays `side_input`."""
    info = RuleInfo("DropoutEval", 1, "view", applies_to="aten.dropout.default(x, p, train) with train == False",
                    preconditions=("operand is a value collection", "train is the Python literal False (eval mode) — any p"),
                    evidence_domain="structural (no gates): ATen returns the input unchanged when train is false")

    def targets(self):
        return {aten.dropout.default}

    def applies(self, node, ctx):
        v = ctx.val(node.args[0])
        if not isinstance(v, Coll):
            return f"operand is {type(v).__name__}"
        train = node.args[2] if len(node.args) > 2 else node.kwargs.get("train")
        if train is not False:
            return f"side_input: train={train!r} is not the literal False (a training-mode dropout reads random state)"
        return True

    def emit(self, node, ctx):
        ctx.define(node, ctx.coll(node.args[0]), view_of=node.args[0])
        ctx.record(self, node, {"train": False, "p": node.args[1] if len(node.args) > 1 else node.kwargs.get("p")}, None)


VIEW_RULES = [StaticIndexRule(), IdentityViewRule(), ReshapeRule(), ViewDtypeRule(), SliceSelectRule(), SplitRule(), CatStackRule(), PermuteRule(),
              ExpandRule(), RepeatInterleaveRule(), StaticGatherRule(), AllocRule(), DropoutEvalRule()]

__all__ = ["VIEW_RULES"]
