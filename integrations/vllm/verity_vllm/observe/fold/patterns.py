"""Pattern helpers and the b1 kernel-set patterns (lane L3, DESIGN §6).

Everything here is pure Python over the event trees (`tree.TreeNode`) and the step context
(`resolver.StepContext`); it never reads tensor values.  The patterns are registered by a profile module
(`profiles/vllm_d9105ea80_sm89_eager.py`) which also pins the constexpr / launch-geometry values they accept.

Vocabulary (D68): a pattern maps a raw event subtree to one of the resolver's outcomes: `Compute` (rows of a
registered *specialization*, one *instance* per row), `Transparent` (alias / move / alloc / noop) or `Unsupported`.

Grouped resolutions
-------------------
Several raw events produce more than one resolution: `_C.rotary_embedding` rotates q *and* k (two `RoPE_v1`
specializations from one dispatcher call), `reshape_and_cache_flash` moves k *and* v, the attention op yields one
`Attention_v3{T}` compute per distinct T, and a scope pattern (attention / KV update) may have transparent children
(slices, copies) that the fold must still execute.  `resolver.resolve_step` accepts a single `Resolution` per root, so
this module carries the extra members on the primary resolution (`ComputeGroup.before / .also`,
`TransparentGroup.before / .also`); `members(r)` yields the fold's execution order `before, primary, also`.  Only the
primary carries `consumed` (the whole subtree); the other members have `consumed=()`.  This is a lane-local extension
of the contract — proposed to the integrator as `Resolution.followers` / a `Multi` resolution (see the L3 report);
`resolver.resolve_step` and `ResolutionReport.add` see only the primary until it is lifted.

Geometry helpers
----------------
`views.py` (L2) is a skeleton at the time of writing, so the few view manipulations the patterns need
(`sub_rows`, `rows_operand`, `split_heads`, `as_weight_view`, `as_slot_rows`) live here; they are pure functions on
`ViewDesc` and can move to `views.py` once L2 lands.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ir.defs import bind
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
from verity_vllm.program.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.resolver import (
    Compute,
    Const,
    Gather,
    GatherMany,
    Group,
    Operand,
    Param,
    PerRow,
    Resolution,
    Shared,
    StepContext,
    Transparent,
    Unsupported,
    parts_of,
)
from verity_vllm.observe.fold.tree import TreeNode

# ---------------------------------------------------------------------------------------------------------
# grouped resolutions (lane-local extension; see module docstring)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeGroup(Compute):
    """A `Compute` that carries further resolutions produced by the same raw event subtree.  Execution order for
    the fold: `before` (transparent children of the scope, e.g. slices/copies inside the attention op), the primary,
    then `also` (the k rotation of a RoPE call, the remaining `Attention_v3{T}` groups, ...)."""

    before: tuple[Resolution, ...] = ()
    also: tuple[Resolution, ...] = ()


@dataclass(frozen=True)
class TransparentGroup(Transparent):
    """A `Transparent` that carries further resolutions (the v move of `reshape_and_cache_flash`, ...)."""

    before: tuple[Resolution, ...] = ()
    also: tuple[Resolution, ...] = ()


def members(r: Resolution) -> list[Resolution]:
    """Every resolution a primary stands for, in the fold's execution order (`before`, primary, `also`); the same
    accessor as `resolver.parts_of` (which also understands `resolver.Group`)."""
    return list(parts_of(r))


def is_group(r: Resolution) -> bool:
    return isinstance(r, (ComputeGroup, TransparentGroup, Group))


# ---------------------------------------------------------------------------------------------------------
# op vocabulary and argument binding
# ---------------------------------------------------------------------------------------------------------


def op_parts(op: str) -> tuple[str, str, str]:
    """`'aten.slice.Tensor' -> ('aten', 'slice', 'Tensor')`; `'_C.silu_and_mul.default' -> ('_C', 'silu_and_mul',
    'default')`; a Triton function path `'a.b.matmul_kernel_persistent' -> ('a.b', 'matmul_kernel_persistent', '')`
    is *not* what this is for (use `triton_name`)."""
    parts = op.split(".")
    if len(parts) >= 3:
        return parts[0], parts[1], ".".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return "", op, ""


def base_op(op: str) -> str:
    """`'aten.slice.Tensor' -> 'aten.slice'` (namespace.name without the overload)."""
    ns, name, _ = op_parts(op)
    return f"{ns}.{name}" if ns else name


def op_name(op: str) -> str:
    """The op's name without namespace and overload: `'_C_cache_ops.reshape_and_cache_flash.default' ->
    'reshape_and_cache_flash'`."""
    return op_parts(op)[1]


def triton_name(node: TreeNode) -> str:
    """Last component of a Triton launch's `fn` (`'...batch_invariant.matmul_kernel_persistent' ->
    'matmul_kernel_persistent'`); '' for dispatcher nodes."""
    if not node.is_triton:
        return ""
    return node.enter.fn.rsplit(".", 1)[-1]


def schema_param_names(schema: str) -> list[str]:
    """Parameter names of a dispatcher schema string, in order (`'aten::embedding(Tensor weight, Tensor indices,
    SymInt padding_idx=-1, ...) -> Tensor'` -> `['weight', 'indices', 'padding_idx', ...]`).  Tolerates
    `Tensor(a!)`, `int[]`, `Tensor?`, defaults and the `*` keyword-only marker; returns [] when it cannot parse."""
    if not schema or "(" not in schema:
        return []
    start = schema.index("(")
    depth = 0
    end = None
    for i in range(start, len(schema)):
        c = schema[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return []
    body = schema[start + 1:end]
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for c in body:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur))
    names: list[str] = []
    for p in parts:
        p = p.strip()
        if not p or p == "*":
            continue
        lhs = p.split("=", 1)[0].strip()
        toks = lhs.split()
        if len(toks) < 2:
            continue
        names.append(toks[-1])
    return names


def bound_args(node: TreeNode) -> dict[str, Any]:
    """Arguments by name: Triton launches are recorded by name already; dispatcher calls are bound through their
    schema (positional args -> schema parameter names, then kwargs).  Unnamed positions become `arg<i>`."""
    if node.is_triton:
        return dict(node.enter.args)
    names = schema_param_names(node.enter.schema)
    out: dict[str, Any] = {}
    for i, a in enumerate(node.enter.args):
        out[names[i] if i < len(names) else f"arg{i}"] = a
    for k, v in node.enter.kwargs.items():
        out[k] = v
    return out


def first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def as_float(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


# ---------------------------------------------------------------------------------------------------------
# view geometry
# ---------------------------------------------------------------------------------------------------------

INT_DTYPES = frozenset({"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"})
FLOAT_DTYPES = frozenset({"bfloat16", "float16", "float32", "float64"})


class GeometryError(ValueError):
    pass


def is_int(v: ViewDesc) -> bool:
    return v.dtype in INT_DTYPES


def contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    acc = 1
    for s in reversed(shape):
        out.append(acc)
        acc *= max(int(s), 1)
    return tuple(reversed(out))


def is_row_major(v: ViewDesc) -> bool:
    """Strides are the contiguous ones (dimensions of extent 1 have irrelevant strides)."""
    exp = contiguous_strides(v.shape)
    return all(s == 1 or a == b for s, a, b in zip(v.shape, v.strides, exp))


def sub_rows(v: ViewDesc, lo: int, hi: int) -> ViewDesc:
    """`v[lo:hi]` along the leading axis."""
    if v.rank < 1 or not 0 <= lo <= hi <= v.shape[0]:
        raise GeometryError(f"sub_rows({v!r}, {lo}, {hi})")
    return ViewDesc(v.alloc, v.offset + lo * v.strides[0], (hi - lo,) + v.shape[1:], v.strides, v.dtype, v.device)


def rows_operand(v: ViewDesc, rows: Sequence[int]) -> Operand:
    """The rows `rows` of `v` as an operand: `PerRow` of a leading-axis slice when they are consecutive, else a
    `Gather` with the (public) row indices."""
    rows = list(rows)
    if not rows:
        raise GeometryError("rows_operand: empty row set")
    if rows == list(range(rows[0], rows[0] + len(rows))):
        return PerRow(sub_rows(v, rows[0], rows[-1] + 1))
    return Gather(v, tuple(rows))


def split_heads(v: ViewDesc, nheads: int, d: int) -> ViewDesc:
    """`[M, nheads*d]` (unit inner stride) -> `[M, nheads, d]`; a `[M, nheads, d]` view with strides `(s, d, 1)`
    is returned unchanged."""
    if v.rank == 3:
        if v.shape[1:] == (nheads, d) and v.strides[1:] == (d, 1):
            return v
        raise GeometryError(f"split_heads: {v!r} is not [M,{nheads},{d}] with strides (*, {d}, 1)")
    if v.rank != 2 or v.shape[1] != nheads * d or v.strides[1] != 1:
        raise GeometryError(f"split_heads: {v!r} is not [M,{nheads * d}] with unit inner stride")
    return ViewDesc(v.alloc, v.offset, (v.shape[0], nheads, d), (v.strides[0], d, 1), v.dtype, v.device)


def as_weight_view(b: ViewDesc, K: int, N: int) -> ViewDesc:
    """The GEMM's `b_ptr` is `weight.t()`: a `[K, N]` view with strides `(1, K)` of the `[N, K]` row-major
    parameter.  Returns that parameter view `[N, K]` strides `(K, 1)`."""
    if b.shape == (K, N) and b.strides == (1, K):
        return ViewDesc(b.alloc, b.offset, (N, K), (K, 1), b.dtype, b.device)
    if b.shape == (N, K) and b.strides == (K, 1):
        return b
    raise GeometryError(f"as_weight_view: {b!r} is neither [K={K},N={N}] strides (1,K) nor the [N,K] row-major weight")


def as_slot_rows(cache: ViewDesc) -> ViewDesc:
    """A per-layer FlashAttention KV cache view `[num_blocks, block_size, KVH, D]` reinterpreted as
    `[num_blocks*block_size, KVH, D]`, one row per KV slot.  The slot rows must be uniformly strided
    (`strides[0] == block_size * strides[1]`, so slot `b*bs + o` is at `(b*bs + o) * strides[1]`) with unit head-dim
    stride; heads may be padded (κ's cache is `[blocks][bs][KVH][K:D | V:D]`: strides `(bs*KVH*2D, KVH*2D, 2D, 1)`,
    the value cache at `+D`).  A `[slots, KVH, D]` view with unit inner stride is returned as is."""
    if cache.rank == 4:
        nb, bs, kvh, d = cache.shape
        s_b, s_o, s_h, s_d = cache.strides
        if s_d == 1 and s_h >= d and s_o >= kvh * s_h and s_b == bs * s_o:
            return ViewDesc(cache.alloc, cache.offset, (nb * bs, kvh, d), (s_o, s_h, 1), cache.dtype, cache.device)
        raise GeometryError(f"as_slot_rows: {cache!r} is not a uniformly strided [blocks, block_size, KVH, D] cache")
    if cache.rank == 3 and cache.strides[2] == 1 and cache.strides[1] >= cache.shape[2] and cache.strides[0] >= cache.shape[1] * cache.strides[1]:
        return cache
    raise GeometryError(f"as_slot_rows: unexpected KV cache view {cache!r}")


def kv_group_of(ctx: StepContext, layer: str | None) -> dict | None:
    """[r8 E-cleanup 2026-09-18, hybrid KV cache] The KV-cache group whose block tables / slot mapping address `layer`'s
    KV rows, or None when the step has a single group (the legacy `StepStart.block_tables` / `slot_mapping` are complete).

    vLLM's hybrid KV-cache manager gives layers of different attention kinds separate groups -- Gemma-2: the sliding-window
    (even) layers and the global (odd) layers -- each with its own block tables and slot mapping into the SAME physical KV
    tensors (layers 2k and 2k+1 share one tensor).  Modelling every layer with group 0's tables puts the even layers' rows
    at the odd layers' cells: prefill still replays (reads resolve by provenance to the same step's mis-addressed writes)
    but decode reads the neighbour layer's K/V (Gemma-2 B1 C256 recapture 10:28Z: 18/18 tier-A failures = even-layer
    decode, outputs outside the span of the gathered V).  With several groups a layer whose group is unknown is a
    GeometryError (never a guess)."""
    groups = getattr(ctx.start, "kv_groups", None)
    if not groups or len(groups) <= 1:
        return None
    if not layer:
        raise GeometryError(f"{len(groups)} KV-cache groups but the layer of this KV access is unknown")
    for g in groups:
        if layer in (g.get("layers") or ()):
            return g
    raise GeometryError(f"layer {layer!r} is in none of the {len(groups)} KV-cache groups")


def kv_layer_of(node: TreeNode) -> str | None:
    """The attention layer a KV access belongs to, as vLLM names it in `kv_cache_config.kv_cache_groups[g].layer_names`
    (`model.layers.N.self_attn.attn`): the op's `layer_name` argument when it was recorded as a string, else the observer's
    module-path hint of the dispatch (`DispatchEnter.module`, the `Attention` module whose forward issued the call -- this
    vLLM's `unified_attention_with_output(..., PyObject layer_name, ...)` records the object as `{}`).  A mis-attributed layer
    can only FAIL (its group's slots are the wrong cells: tier A / G5), never pass silently."""
    a = bound_args(node)
    ln = a.get("layer_name")
    if isinstance(ln, str) and ln:
        return ln
    if not node.is_triton:
        mod = getattr(node.enter, "module", None)
        if isinstance(mod, str) and mod:
            return mod
    return None


def kv_slots(ctx: StepContext, request_id: str, T: int, layer: str | None = None) -> tuple[int, ...]:
    """KV slots of positions `0..T-1` of a request from the step's public block table (of `layer`'s KV-cache group when
    the step has several groups; see `kv_group_of`)."""
    g = kv_group_of(ctx, layer)
    bs = ctx.start.block_size if g is None else int(g["block_size"])
    bt = ctx.start.block_tables.get(request_id) if g is None else (g.get("block_tables") or {}).get(request_id)
    if bt is None:
        raise GeometryError(f"no block table for request {request_id!r}")
    if len(bt) * bs < T:
        raise GeometryError(f"block table of {request_id!r} covers {len(bt) * bs} positions < T={T}")
    return tuple(bt[t // bs] * bs + t % bs for t in range(T))


def attribution_for(ctx: StepContext, rows: int, phase: str) -> tuple[tuple[str, int], ...]:
    """(request_id, position) per row when the row count identifies the rows (token rows, or logits rows)."""
    if rows == len(ctx.start.rows) and phase != "sample":
        return tuple(ctx.start.rows)
    if rows == len(ctx.start.logits_rows):
        return tuple(ctx.start.logits_rows)
    if rows == len(ctx.start.rows):
        return tuple(ctx.start.rows)
    return ()


def hint_of(node: TreeNode, *extra: str) -> str:
    mod = node.enter.module
    parts = [p for p in (mod, *extra) if p]
    return "/".join(parts) if parts else node.op


def unsupported(node: TreeNode, reason: str, **detail: Any) -> Unsupported:
    d = {"phase": node.phase, "children": [c.op for c in node.children]}
    d.update(detail)
    return Unsupported(node.id, node.op, reason, d, tuple(sorted(node.ids())))


def _fmt(v: Any) -> Any:
    if isinstance(v, ViewDesc):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return [_fmt(x) for x in v]
    if isinstance(v, dict):
        return {k: _fmt(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------------------------------------
# transparent aten ops (DESIGN §6, rows "view/…", "copy_/…", "empty/…") and integer metadata
# ---------------------------------------------------------------------------------------------------------

ALIAS_OPS = frozenset({
    "aten.view", "aten._unsafe_view", "aten.reshape", "aten.t", "aten.transpose", "aten.permute", "aten.split",
    "aten.split_with_sizes", "aten.slice", "aten.select", "aten.unsqueeze", "aten.squeeze", "aten.expand",
    "aten.expand_as", "aten.alias", "aten.detach", "aten.narrow", "aten.as_strided", "aten.unbind", "aten.unfold",
    "aten.view_as_real", "aten.view_as", "aten.flatten", "aten.unflatten", "aten.lift_fresh", "aten.resolve_conj",
    "aten.resolve_neg", "aten.detach_", "aten.movedim", "aten.chunk", "aten.tensor_split",
    "aten.numpy_T",   # [R15 fp8] `Tensor.T` (all axes reversed): the FP8 linears pass `weight.T` / `weight_scale_inv.T` to cutlass_scaled_mm
})
MOVE_OPS = frozenset({"aten.clone", "aten.contiguous", "aten.copy_", "aten._to_copy", "aten.to", "aten.copy"})
#: alias ops that torch documents as "view if possible, else copy": a fresh output allocation from one of these is the copy case
RESHAPE_COPY_OPS = frozenset({"aten.reshape", "aten.flatten", "aten.unflatten"})
ALLOC_OPS = frozenset({
    "aten.empty", "aten.empty_like", "aten.empty_strided", "aten.zeros", "aten.zeros_like", "aten.ones",
    "aten.ones_like", "aten.full", "aten.full_like", "aten.arange", "aten.new_empty", "aten.new_zeros",
    "aten.new_ones", "aten.new_full", "aten.scalar_tensor", "aten.zero_", "aten.fill_", "aten.fill",
    "aten.empty_permuted", "aten.randn", "aten.rand", "aten.lift_fresh_copy",
})
NOOP_OPS = frozenset({
    "aten._local_scalar_dense", "aten.item", "aten.is_nonzero", "aten.equal", "aten.sym_size", "aten.sym_numel",
    "aten.sym_stride", "aten.sym_storage_offset", "aten.record_stream", "aten.set_", "aten.resize_",
    "aten._has_compatible_shallow_copy_type", "aten.is_same_size", "aten._pin_memory", "aten.pin_memory",
    "aten.is_pinned", "aten._record_function_enter_new", "aten._record_function_enter", "aten._record_function_exit",
    "aten._assert_async", "aten._assert_tensor_metadata", "aten._assert_scalar", "aten.numel", "aten.dim",
    "aten.size", "aten.stride", "aten.storage_offset", "aten.is_contiguous", "aten.is_complex", "aten.is_floating_point",
})

COPY_KERNELS = ("Memcpy", "elementwise_kernel", "vectorized_elementwise_kernel", "unrolled_elementwise_kernel")
FILL_KERNELS = ("FillFunctor", "Memset", "fill_kernel", "vectorized_elementwise_kernel")

SCRATCH_INT_COPIES = "int_copies"   # ctx.scratch key: dst alloc -> (src view, dst view) of integer copies seen this step
SCRATCH_F32_UPCASTS = "f32_upcasts"  # ctx.scratch key: dst alloc -> (src view, dst view) of exact float widenings (bf16 -> f32)


def _copy_endpoints(node: TreeNode) -> tuple[ViewDesc | None, ViewDesc | None]:
    b = base_op(node.op)
    ins = node.input_views()
    outs = node.output_views()
    if b == "aten.copy_":
        args = node.args
        if len(args) >= 2 and isinstance(args[0], ViewDesc) and isinstance(args[1], ViewDesc):
            return args[1], args[0]
        return None, None
    src = ins[0] if ins else None
    dst = outs[0] if outs else None
    return src, dst


def _same_view(a: ViewDesc, b: ViewDesc) -> bool:
    return (a.alloc, a.offset, a.shape, a.strides, a.dtype) == (b.alloc, b.offset, b.shape, b.strides, b.dtype)


def classify_transparent(node: TreeNode, ctx: StepContext) -> Transparent | Unsupported | None:
    """The DESIGN §6 transparent set plus the integer-metadata rule.  Returns None when the op is not transparent
    (a compute pattern must explain it), `Unsupported` when it is one of the transparent ops but violates the rule
    it is admitted under (an alias producing a fresh allocation, a float dtype change, a broadcast copy).

    Integer rule: a tensor whose dtype is integral (positions, slot mapping, block tables, seq_lens, indices, the
    sampled ids) is public step metadata — the fold takes those values from `StepStart` / `StepLink` / `SampleEnd`,
    never from device memory — so an aten op whose tensor inputs and outputs are all integral has no value effect
    the fold must track (`noop`).  Integer copies are additionally remembered in `ctx.scratch["int_copies"]` so that
    the Embedding pattern can map `input_ids.long()` back to the step's `input_ids` view."""
    if node.is_triton:
        return None
    b = base_op(node.op)
    ins = node.input_views()
    outs = node.output_views()
    hint = hint_of(node, node.op)
    if b in NOOP_OPS:
        return Transparent("noop", (node.id,), hint=hint)
    if b in ALIAS_OPS:
        in_allocs = {v.alloc for v in ins}
        stray = [o for o in outs if o.alloc not in in_allocs]
        if stray:
            # [R15 fp8 / Qwen3] `aten.reshape` (and `flatten`) COPY when the requested shape is not expressible as a view of the input's
            # strides -- vLLM's `rms_norm_batch_invariant` does `q.view(M, NH, D).reshape(-1, D)` on the strided q slice of the fused
            # QKV output ([M, NH, D] strides (QKV, D, 1): viewable only at M == 1).  torch's contract for that case is an element-order-
            # preserving copy (`clone(contiguous)` of the logical row-major order), i.e. exactly the "reshaping copy" move below:
            # one input view, one fresh output of the same dtype and element count.  Anything else that leaves its inputs' allocations
            # stays refused.
            if (b in RESHAPE_COPY_OPS and len(ins) == 1 and len(outs) == 1 and outs[0].alloc not in in_allocs
                    and ins[0].numel == outs[0].numel and ins[0].dtype == outs[0].dtype and is_row_major(outs[0])):
                return Transparent("move", (node.id,), src=Shared(ins[0]), dst=Shared(outs[0]), rows=1,
                                   hint=f"{hint}: reshape copied (input strides not viewable as the requested shape)", kernels=COPY_KERNELS)
            return unsupported(node, "alias op produced a view outside its inputs' allocations",
                               outputs=[repr(o) for o in stray], inputs=[repr(v) for v in ins])
        return Transparent("alias", (node.id,), hint=hint)
    if b in ALLOC_OPS:
        if outs and all(is_int(o) for o in outs):
            return Transparent("noop", (node.id,), hint=f"{hint}: integer metadata allocation")
        if outs and all(o.numel == 0 for o in outs):
            return Transparent("noop", (node.id,), hint=f"{hint}: zero-element allocation")
        dst = Shared(outs[0]) if outs else None
        return Transparent("alloc", (node.id,), dst=dst, hint=hint, kernels=FILL_KERNELS if b in ("aten.zeros", "aten.zeros_like", "aten.ones", "aten.ones_like", "aten.full", "aten.full_like", "aten.new_zeros", "aten.new_ones", "aten.new_full", "aten.zero_", "aten.fill_", "aten.fill", "aten.scalar_tensor") else ())
    if b in MOVE_OPS:
        src, dst = _copy_endpoints(node)
        if src is None or dst is None:
            if not ins and not outs:
                return Transparent("noop", (node.id,), hint=hint)
            return unsupported(node, "copy without a tensor source/destination", inputs=[repr(v) for v in ins],
                               outputs=[repr(v) for v in outs])
        if is_int(src) and is_int(dst):
            ctx.scratch.setdefault(SCRATCH_INT_COPIES, {})[dst.alloc] = (src, dst)
            return Transparent("noop", (node.id,), src=PerRow(src), dst=PerRow(dst),
                               rows=dst.shape[0] if dst.rank else 1, hint=f"{hint}: integer metadata copy")
        if src.dtype in ("bfloat16", "float16") and dst.dtype == "float32" and src.shape == dst.shape:
            # exact widening (every bf16/f16 value is an f32 value): the widened copy carries no new values, so a
            # compute reading it is the same function of the narrow source — consumers re-point to `src` via the
            # scratchpad (TokenSelect does); the widened allocation itself receives no refs (an unexplained reader of it
            # is Unresolved in the fold, i.e. fails loudly).
            ctx.scratch.setdefault(SCRATCH_F32_UPCASTS, {})[dst.alloc] = (src, dst)
            return Transparent("noop", (node.id,), src=PerRow(src), dst=PerRow(dst), rows=dst.shape[0] if dst.rank else 1,
                               hint=f"{hint}: exact {src.dtype}->float32 widening (consumers re-point to the source)")
        if src.dtype != dst.dtype:
            return unsupported(node, "dtype change", src=repr(src), dst=repr(dst))
        if _same_view(src, dst):
            return Transparent("alias", (node.id,), hint=f"{hint}: returned self")
        if src.shape == dst.shape:
            return Transparent("move", (node.id,), src=PerRow(src), dst=PerRow(dst), rows=dst.shape[0] if dst.rank else 1,
                               hint=hint, kernels=COPY_KERNELS)
        if src.numel == dst.numel:
            return Transparent("move", (node.id,), src=Shared(src), dst=Shared(dst), rows=1, hint=f"{hint}: reshaping copy",
                               kernels=COPY_KERNELS)
        return unsupported(node, "copy between views of different element counts (broadcast)", src=repr(src), dst=repr(dst))
    if b.startswith("aten.") and (ins or outs) and all(is_int(v) for v in ins) and all(is_int(o) for o in outs):
        return Transparent("noop", (node.id,), hint=f"{hint}: integer metadata op")
    return None


class TransparentAten:
    """DESIGN §6 rows `view/…` → alias, `copy_/…` → move (same dtype), `empty/…` → alloc, plus the integer-metadata
    rule (`classify_transparent`).  Consumes the node only; a child under a transparent op is an anomaly the driver
    reports as `ResolutionError`."""

    name = "transparent.aten"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        r = classify_transparent(node, ctx)
        if r is None or isinstance(r, Unsupported):
            return r
        return dataclasses.replace(r, pattern=self.name)


class TransparentTritonMetadata:
    """A Triton launch whose pointer arguments are all integer / bool tensors is the model runner's input preparation
    (vLLM V2 `gpu/input_batch.py`, `gpu/block_table.py`, `gpu/buffer_utils.py`: `_apply_write_kernel`,
    `_prepare_prefill_inputs_kernel`, `_prepare_pos_seq_lens_kernel`, `_combine_sampled_and_draft_tokens_kernel`,
    `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `_get_num_sampled_and_rejected_kernel`): it computes
    the public step metadata (input ids from the previous step's sampled ids = the StepLink, positions, seq lens,
    block tables, slot mappings) that the fold takes from `StepStart` / `StepLink` / `SampleEnd`.  Same rule as the
    integer aten ops → `Transparent(noop)`.  A launch with any floating-point pointer is not covered (a compute pattern
    must explain it)."""

    name = "transparent.triton_metadata"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton:
            return None
        views = node.input_views()
        if not views or not all(is_int(v) for v in views):
            return None
        return Transparent("noop", (node.id,), hint=f"{triton_name(node)}: integer metadata kernel", kernels=(triton_name(node),),
                           pattern=self.name)


# ---------------------------------------------------------------------------------------------------------
# scope helpers: a wrapper op whose subtree contains the real kernel op plus transparent glue
# ---------------------------------------------------------------------------------------------------------


def _walk_skipping(node: TreeNode, keep: set[int]) -> Iterator[TreeNode]:
    """Descendants of `node` (excluding it) that are not under a kept node."""
    for c in node.children:
        if c.id in keep:
            continue
        yield c
        yield from _walk_skipping(c, keep)


def find_descendants(node: TreeNode, pred) -> list[TreeNode]:
    return [n for n in node.walk() if n is not node and pred(n)]


def scope_transparents(node: TreeNode, ctx: StepContext, keep: set[int], pattern: str) -> tuple[list[Transparent], Unsupported | None]:
    """Explain every descendant of `node` not under a kept node as transparent.  Aliases and noops are simply
    consumed by the scope; moves and allocs are returned (log order, `consumed=()`) so the fold executes them.
    Anything else makes the whole scope Unsupported (naming the child)."""
    emit: list[Transparent] = []
    for c in _walk_skipping(node, keep):
        r = classify_transparent(c, ctx)
        if r is None:
            return [], unsupported(node, f"unexplained child {c.op} inside {op_name(node.op)}", child_id=c.id, child=c.op)
        if isinstance(r, Unsupported):
            return [], unsupported(node, f"child {c.op} inside {op_name(node.op)}: {r.reason}", child_id=c.id, child_detail=r.detail)
        if r.effect in ("move", "alloc"):
            emit.append(dataclasses.replace(r, consumed=(), pattern=pattern))
    return emit, None


# ---------------------------------------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------------------------------------


def _check_pins(node: TreeNode, got: dict[str, Any], pins: dict[str, Any]) -> Unsupported | None:
    bad = {k: (got.get(k), v) for k, v in pins.items() if got.get(k) != v}
    if bad:
        return unsupported(node, "launch constexpr/geometry differs from the pinned profile",
                           mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
    return None


X01_SCRATCH_KEY = "x01_num_sms"


def _targets():
    from verity_vllm.program.registry import targets as T
    return T


def target_of(ctx: StepContext) -> tuple[TargetProfile, bool]:
    """[X-01] The record's `TargetProfile` — the ONE SM-count fact both builder bets and this resolver pin against.  For a
    captured log it is the capture host's own device query, recorded by the GPU phase in the run header
    (`RunHeader.profile["gpu"]`: `verity_vllm.engine.engine_profile.gpu_info` -> `multi_processor_count`, `compute_capability`) and read
    back through `TargetProfile.from_gpu_document`; a header without a GPU document (synthetic / hand-built logs) keeps
    `DEFAULT_TARGET`.  Returns (profile, recorded) where `recorded` says the header carried the fact."""
    try:
        # [R13 hopper] the DECLARED profile when the capture recorded one (`profile["target"]`: exact, incl. `flash_attn_version`), else the
        # device document + the layers' reported FA version (`TargetProfile.from_run_profile`)
        tp = TargetProfile.from_run_profile(ctx.header.profile or {})
    except (AttributeError, TypeError, ValueError):
        tp = None
    return (tp, True) if tp is not None else (DEFAULT_TARGET, False)


def header_gemm_arch_family(ctx: StepContext):
    """[R13 ampere] The tuned-GEMM arch family the CAPTURE host reported (`RunHeader.profile["gpu"]["tuned_gemm_arch_family"]`, written by
    `verity_vllm.engine.engine_profile.gpu_info` from vLLM's own `_get_tuned_matmul_arch_family`), or `...` when the header predates the field
    (the registry then derives it from the capability alone)."""
    try:
        gpu = (ctx.header.profile or {}).get("gpu") or {}
    except AttributeError:
        return ...
    return gpu["tuned_gemm_arch_family"] if isinstance(gpu, dict) and "tuned_gemm_arch_family" in gpu else ...


GEMM_TARGET_SCRATCH_KEY = "r13_gemm_target"


def gemm_target_of(ctx: StepContext):
    """[R13 ampere] The registered GEMM specialization of the record's target (`TargetProfile.gemm_target()` on `target_of(ctx)`, with
    the family the capture host reported): `(GemmTarget, None)` or `(None, refusal)` — the ONE source of the launch geometry every
    `matmul_kernel_persistent` launch must show (`GemmTarget.launch_config`), shared with Bet A (`vllm_bindings.gemm_pins_for`).  The
    selection is recorded once in `ctx.scratch[GEMM_TARGET_SCRATCH_KEY]` for the fold / resolution report."""
    from verity_vllm.program.registry.gemm_targets import try_gemm_target_for
    tp, recorded = target_of(ctx)
    fam = header_gemm_arch_family(ctx)
    gt, refusal = try_gemm_target_for(tp.compute_capability, fam)
    if ctx is None or not isinstance(getattr(ctx, "scratch", None), dict):
        return gt, refusal
    rec = ctx.scratch.setdefault(GEMM_TARGET_SCRATCH_KEY, {"target": gt.key if gt else None, "dot": gt.dot if gt else None, "refusal": refusal,
                                                            "compute_capability": list(tp.compute_capability), "target_recorded": recorded,
                                                            "header_tuned_gemm_arch_family": None if fam is ... else fam,
                                                            "tuned_table_sha256": gt.table_sha256() if gt else None, "launches": 0,
                                                            "basis": "R13 ampere: launch geometry = the registered GEMM target's (verity_vllm.program.registry.gemm_targets)"})
    rec["launches"] += 1
    return gt, refusal


def pinned_num_sms(ctx: StepContext, profile_pin: int) -> int:
    """[X-01] The `NUM_SMS` a persistent-GEMM launch must show: the record target's `num_sms` when the header recorded the device
    (exact — a launch for another GPU on this host is still a pin mismatch), else the profile's own pin (`DEFAULT_TARGET.num_sms`).
    `NUM_SMS` is the persistent grid size (how many CTAs walk the tile list): every output tile is one f32 accumulator advanced over
    the same ascending 64-wide k tiles whatever the grid, so it is a scheduling parameter of the launch, not a numerical one —
    which is why the value comes from the target and not from the numerics profile.  The choice is counted in
    `ctx.scratch[X01_SCRATCH_KEY]` so the fold / resolution report record which target fact was applied."""
    tp, recorded = target_of(ctx)
    if not recorded:
        return int(profile_pin)
    rec = ctx.scratch.setdefault(X01_SCRATCH_KEY, {"pinned": int(tp.num_sms), "source": "RunHeader.profile.gpu.multi_processor_count (TargetProfile.from_gpu_document)",
                                                   "profile_default": int(profile_pin), "compute_capability": list(tp.compute_capability), "launches": 0,
                                                   "basis": "X-01: NUM_SMS = the target's SM count (persistent grid size; scheduling only under BI)"})
    rec["launches"] += 1
    return int(tp.num_sms)


@dataclass
class GemmLaunch:
    """`TritonLaunch …matmul_kernel_persistent` (vLLM batch-invariant GEMM, launched from Python by
    `matmul_persistent(a, b)`; no `aten.mm` in the eager batch-invariant path — the launch is a root-level event
    preceded by the `aten.t` alias of the weight) → `Compute(Gemm_v1{K,N}, rows=M, (PerRow(a), Shared(w)),
    (PerRow(c),))` with `w` the `[N,K]` parameter view behind the transposed `b_ptr`.  Also matches an
    `aten.mm/matmul/linear/addmm` dispatcher node that wraps exactly one such launch (the launch fired inside the
    op's CUDA implementation), consuming the wrapper as scope.

    Constexpr pins (`constexprs`, `num_warps`, `num_stages`) must match exactly.  [R13 ampere] The TILING pinned is the
    record target's launch geometry for (M, N, K) — `gemm_target_of(ctx)`: the registered GEMM specialization of the
    header's capability (`verity_vllm.program.registry.gemm_targets`; `ada` = vLLM's sm_89 tuned table rows else the default, `ampere`
    = the default for every shape); `constexprs` / `num_warps` / `num_stages` given at construction are the profile's default
    expectation for the fields the target does not decide.  A target without a registered specialization is Unsupported
    (`numerics_unregistered`) whatever the launch shows.  The `NUM_SMS` pin is the record target's SM count ([X-01]
    `pinned_num_sms`: one `TargetProfile` fact, read from the run header's device document; `num_sms` is the profile's own
    default when the header has none).  A bound bias (`HAS_BIAS` / non-null `bias_ptr`) is the
    dispatcher-invisible `addmm` path: Unsupported ("bias path") — `Gemm` then `BiasAdd` from one launch is not
    expressible as one Compute (report proposal)."""

    constexprs: dict[str, Any]
    num_warps: int | None = None
    num_stages: int | None = None
    num_sms: int | None = None
    name: str = "gemm.matmul_kernel_persistent"
    kernel: str = "matmul_kernel_persistent"
    wrapper_ops: frozenset = frozenset({"aten.mm", "aten.matmul", "aten.linear", "aten.addmm", "aten.bmm"})

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        before: list[Transparent] = []
        if node.is_triton:
            if triton_name(node) != self.kernel:
                return None
            launch = node
        elif base_op(node.op) in self.wrapper_ops:
            launches = [n for n in node.walk() if n.is_triton and triton_name(n) == self.kernel]
            if len(launches) != 1:
                return unsupported(node, f"{base_op(node.op)} without exactly one batch-invariant {self.kernel} launch",
                                   launches=[n.enter.fn for n in launches])
            launch = launches[0]
            before, bad = scope_transparents(node, ctx, {launch.id}, self.name)
            if bad is not None:
                return bad
        else:
            return None
        a = bound_args(launch)
        consts = dict(launch.enter.constexprs)
        # [R13 ampere] the tiling (BLOCK_SIZE_*, num_warps, num_stages) a launch must show is the record TARGET's launch geometry for
        # (M, N, K) — the registered GEMM specialization's `launch_config` (verity_vllm.program.registry.gemm_targets: vLLM's tuned-table row for the
        # target's arch family, else the default), the same source Bet A pins.  An unregistered target is refused whatever the launch shows.
        gt, refusal = gemm_target_of(ctx)
        if gt is None:
            return unsupported(launch, refusal, compute_capability=list(target_of(ctx)[0].compute_capability))
        expected = dict(self.constexprs)
        exp_warps, exp_stages = self.num_warps, self.num_stages
        Mi, Ni, Ki = a.get("M"), a.get("N"), a.get("K")
        if all(isinstance(v, int) for v in (Mi, Ni, Ki)):
            tconsts, twarps, tstages, _src = gt.launch_config(Mi, Ni, Ki)
            expected.update({k: v for k, v in tconsts.items() if k in expected or k.startswith("BLOCK_SIZE_")})
            if exp_warps is not None:
                exp_warps = twarps
            if exp_stages is not None:
                exp_stages = tstages
        got = {k: first(consts.get(k), a.get(k)) for k in expected}
        bad = _check_pins(launch, got, expected)
        if bad is not None:
            return bad
        geom = {}
        pins = {}
        if exp_warps is not None:
            geom["num_warps"], pins["num_warps"] = launch.enter.num_warps, exp_warps
        if exp_stages is not None:
            geom["num_stages"], pins["num_stages"] = launch.enter.num_stages, exp_stages
        if self.num_sms is not None:   # [X-01] the pin is the record target's SM count (one TargetProfile fact), else the profile's
            geom["NUM_SMS"], pins["NUM_SMS"] = first(consts.get("NUM_SMS"), a.get("NUM_SMS")), pinned_num_sms(ctx, self.num_sms)
        bad = _check_pins(launch, geom, pins)
        if bad is not None:
            return bad
        a_ptr, b_ptr, c_ptr, bias = a.get("a_ptr"), a.get("b_ptr"), a.get("c_ptr"), a.get("bias_ptr")
        M, N, K = a.get("M"), a.get("N"), a.get("K")
        if not all(isinstance(v, ViewDesc) for v in (a_ptr, b_ptr, c_ptr)) or not all(isinstance(v, int) for v in (M, N, K)):
            return unsupported(launch, "matmul launch arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if bias is not None or first(consts.get("HAS_BIAS"), a.get("HAS_BIAS")):
            return unsupported(launch, "bias path (HAS_BIAS / bias_ptr bound): Gemm then BiasAdd from one launch is not one Compute",
                               bias=repr(bias) if isinstance(bias, ViewDesc) else bias)
        if a_ptr.shape != (M, K) or not is_row_major(a_ptr):
            return unsupported(launch, "a_ptr is not a row-major [M,K] view", a=repr(a_ptr), M=M, K=K)
        if c_ptr.shape != (M, N) or not is_row_major(c_ptr):
            return unsupported(launch, "c_ptr is not a row-major [M,N] view", c=repr(c_ptr), M=M, N=N)
        try:
            w = as_weight_view(b_ptr, K, N)
        except GeometryError as e:
            return unsupported(launch, str(e))
        if not (a_ptr.dtype == b_ptr.dtype == c_ptr.dtype == "bfloat16"):
            return unsupported(launch, "Gemm_v1 is the bf16 kernel", dtypes=[a_ptr.dtype, b_ptr.dtype, c_ptr.dtype])
        if K % 16:
            return unsupported(launch, "K is not a multiple of the 16-wide k-step", K=K)
        if not node.is_triton:
            outs = node.output_views()
            if not outs or outs[0].alloc != c_ptr.alloc:
                return unsupported(node, "wrapper output is not the launch's c_ptr allocation")
        roots = ctx.roots_of(w)
        wname = roots[0].name if roots else ""
        rows = M
        # [R13 hopper] the registered GEMM Definition of the record's target (`registry.targets.gemm_spec`: Gemm_v1 on the accepted
        # Ampere-family target, Gemm_v2{DOT=<target step>} otherwise) — the same selection the Bet A builder makes
        try:
            spec = _targets().gemm_spec(target_of(ctx)[0], K, N)
        except _targets().UnregisteredTarget as e:
            return unsupported(launch, f"target has no registered GEMM binding: {e}")
        comp = Compute(spec, rows, (PerRow(a_ptr), Shared(w)), (PerRow(c_ptr),),
                       tuple(sorted(node.ids())), hint=hint_of(launch, wname or "gemm"), kernels=(self.kernel,),
                       attribution=attribution_for(ctx, rows, node.phase), pattern=self.name)
        if before:
            return _group_compute(comp, before=tuple(before))
        return comp


def _group_compute(c: Compute, before: tuple = (), also: tuple = ()) -> ComputeGroup:
    return ComputeGroup(c.spec, c.rows, c.operands, c.outputs, c.consumed, c.hint, c.kernels, c.attribution, c.pattern,
                        before=before, also=also)


def _group_transparent(t: Transparent, before: tuple = (), also: tuple = ()) -> TransparentGroup:
    return TransparentGroup(t.effect, t.consumed, t.src, t.dst, t.rows, t.hint, t.kernels, t.pattern, before=before, also=also)


@dataclass
class BiasAdd:
    """`aten.add.Tensor(y[M,N] bf16, b[N] bf16 weight Root)` — the batch-invariant linear's separate bias add
    (`linear_batch_invariant`: `output + bias`) → `Compute(BiasAdd_v1{N}, rows=M, (PerRow(y), Shared(b)), (PerRow(out),))`.
    Absent for SmolLM2 (`BIAS=False`); present for Qwen2 qkv_proj."""

    name: str = "bias_add.aten_add"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.add":
            return None
        args = node.args
        if len(args) < 2 or not isinstance(args[0], ViewDesc) or not isinstance(args[1], ViewDesc):
            return None
        y, b = args[0], args[1]
        outs = node.output_views()
        if y.rank != 2 or b.rank != 1 or y.shape[1] != b.shape[0] or not outs:
            return None
        if not any(r.role == "weight" for r in ctx.roots_of(b)):
            return None
        alpha = bound_args(node).get("alpha", 1)
        if alpha not in (1, 1.0):
            return unsupported(node, "aten.add with alpha != 1", alpha=alpha)
        if not (y.dtype == b.dtype == outs[0].dtype == "bfloat16") or not is_row_major(y) or not is_row_major(outs[0]):
            return unsupported(node, "BiasAdd_v1 is bf16 over row-major [M,N] views", dtypes=[y.dtype, b.dtype, outs[0].dtype])
        M, N = y.shape
        return Compute(bind(b1.BiasAdd, N=N), M, (PerRow(y), Shared(b)), (PerRow(outs[0]),), (node.id,),
                       hint=hint_of(node, ctx.roots_of(b)[0].name), kernels=("vectorized_elementwise_kernel", "elementwise_kernel"),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class RMSNormTritonLaunch:
    """`TritonLaunch …batch_invariant._rms_norm_kernel` (vLLM `rms_norm_batch_invariant`, the residual-free norm:
    layer 0's input norm) → `Compute(RMSNormTriton_v1{N,EPS}, rows=M, (PerRow(x), Shared(w)), (PerRow(out),))`.
    Pins: `BLOCK_SIZE=1024`, `HAS_WEIGHT=True`, `num_warps=4` (the reduction tree the registered body writes out);
    `eps` must equal the profile's `EPS`."""

    constexprs: dict[str, Any]
    num_warps: int | None = None
    eps: float | None = None
    name: str = "rmsnorm.triton_rms_norm_kernel"
    kernel: str = "_rms_norm_kernel"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if not node.is_triton or triton_name(node) != self.kernel:
            return None
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        got = {k: first(consts.get(k), a.get(k)) for k in self.constexprs}
        bad = _check_pins(node, got, self.constexprs)
        if bad is not None:
            return bad
        if self.num_warps is not None:
            bad = _check_pins(node, {"num_warps": node.enter.num_warps}, {"num_warps": self.num_warps})
            if bad is not None:
                return bad
        x, w, out = a.get("input_ptr"), a.get("weight_ptr"), a.get("output_ptr")
        n_cols, eps = a.get("n_cols"), as_float(a.get("eps"))
        if not all(isinstance(v, ViewDesc) for v in (x, w, out)) or not isinstance(n_cols, int) or eps is None:
            return unsupported(node, "rms_norm launch arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or x.shape[1] != n_cols or not is_row_major(x) or out.shape != x.shape or not is_row_major(out):
            return unsupported(node, "input/output are not row-major [M,N] views", x=repr(x), out=repr(out), n_cols=n_cols)
        if w.shape != (n_cols,) or w.strides != (1,):
            return unsupported(node, "weight is not a contiguous [N] view", w=repr(w))
        if not (x.dtype == w.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "RMSNormTriton_v1 is bf16", dtypes=[x.dtype, w.dtype, out.dtype])
        if self.eps is not None and eps != self.eps:
            return unsupported(node, "eps differs from the profile's EPS", eps=eps, pinned=self.eps)
        M = x.shape[0]
        if node.enter.grid and node.enter.grid[0] != M:
            return unsupported(node, "grid does not launch one program per row", grid=node.enter.grid, M=M)
        roots = ctx.roots_of(w)
        return Compute(bind(b1.RMSNormTriton, N=n_cols, EPS=self.eps if self.eps is not None else eps), M,
                       (PerRow(x), Shared(w)), (PerRow(out),), (node.id,), hint=hint_of(node, roots[0].name if roots else "rms_norm"),
                       kernels=(self.kernel,), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class FusedAddRMSNorm:
    """`_C.fused_add_rms_norm(input!, residual!, weight, epsilon)` (in place: input ← norm(input+residual),
    residual ← input+residual) → `Compute(RMSNormFusedCuda_v2{N,EPS}, rows=M, (PerRow(x), PerRow(res), Shared(w)),
    (PerRow(x), PerRow(res)))`.  The fold reads all operands before writing (the outputs alias the inputs)."""

    eps: float | None = None
    name: str = "rmsnorm.fused_add_rms_norm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "fused_add_rms_norm":
            return None
        a = bound_args(node)
        x, res, w = a.get("input", a.get("arg0")), a.get("residual", a.get("arg1")), a.get("weight", a.get("arg2"))
        eps = as_float(first(a.get("epsilon"), a.get("eps"), a.get("arg3")))
        if not all(isinstance(v, ViewDesc) for v in (x, res, w)) or eps is None:
            return unsupported(node, "fused_add_rms_norm arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or not is_row_major(x) or res.shape != x.shape or not is_row_major(res):
            return unsupported(node, "input/residual are not row-major [M,N] views of one shape", x=repr(x), res=repr(res))
        N = x.shape[1]
        if w.shape != (N,) or w.strides != (1,):
            return unsupported(node, "weight is not a contiguous [N] view", w=repr(w), N=N)
        if not (x.dtype == res.dtype == w.dtype == "bfloat16"):
            return unsupported(node, "RMSNormFusedCuda_v2 is bf16", dtypes=[x.dtype, res.dtype, w.dtype])
        if self.eps is not None and eps != self.eps:
            return unsupported(node, "epsilon differs from the profile's EPS", eps=eps, pinned=self.eps)
        M = x.shape[0]
        roots = ctx.roots_of(w)
        return Compute(bind(b1.RMSNormFusedCuda, N=N, EPS=self.eps if self.eps is not None else eps), M,
                       (PerRow(x), PerRow(res), Shared(w)), (PerRow(x), PerRow(res)), (node.id,),
                       hint=hint_of(node, roots[0].name if roots else "fused_add_rms_norm"), kernels=("fused_add_rms_norm_kernel",),
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


def widened_source(view: ViewDesc, ctx: StepContext) -> ViewDesc | None:
    """The narrow (bf16/f16) view whose exact float32 widening `view` is, when `classify_transparent` saw that widening
    this step and `view` is the widened tensor itself (or a leading-axis row range of it); else None."""
    hit = ctx.scratch.get(SCRATCH_F32_UPCASTS, {}).get(view.alloc)
    if hit is None:
        return None
    src, dst = hit
    if _same_view(view, dst):
        return src
    if (view.rank == dst.rank == 2 and view.shape[1:] == dst.shape[1:] and view.strides == dst.strides
            and is_row_major(dst) and is_row_major(src) and src.shape == dst.shape):
        off = view.offset - dst.offset
        if off >= 0 and off % dst.strides[0] == 0:
            lo = off // dst.strides[0]
            if lo + view.shape[0] <= dst.shape[0]:
                return sub_rows(src, lo, lo + view.shape[0])
    return None


def _positions_ok(view: ViewDesc, M: int, ctx: StepContext) -> str | None:
    """The positions tensor handed to a kernel must be the step's positions buffer (or an integer copy of it): its
    values are the public `StepStart.positions`, never read from memory.  Returns a reason when that cannot be
    established."""
    if not is_int(view):
        return f"positions view is not integral: {view!r}"
    if view.rank != 1 or view.shape[0] < M:
        return f"positions view {view!r} does not cover M={M} rows"
    if len(ctx.start.positions) < M:
        return f"StepStart.positions has {len(ctx.start.positions)} entries < M={M}"
    pv = ctx.start.positions_view
    if pv is None:
        return None
    if view.alloc == pv.alloc:
        return None
    copies = ctx.scratch.get(SCRATCH_INT_COPIES, {})
    src = copies.get(view.alloc)
    if src is not None and src[0].alloc == pv.alloc:
        return None
    return f"positions view {view!r} is neither the step's positions buffer {pv!r} nor an integer copy of it"


@dataclass
class RotaryEmbedding:
    """`_C.rotary_embedding(positions, query!, key!?, head_size, cos_sin_cache, is_neox)` (NeoX, in place) →
    `ComputeGroup(RoPE_v1{NHEADS=NH,D}, rows=M, (PerRow(q as [M,NH,D]), Gather(cos_sin_cache, positions)), (PerRow(q),))`
    with the k rotation (`NHEADS=KVH`) in `.also`.  `positions` are the step's public facts (the tensor's values are
    never read); `cos_sin_cache` rows are gathered by position (the buffer Root, or one of its per-layer aliases)."""

    name: str = "rope.rotary_embedding"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "rotary_embedding":
            return None
        a = bound_args(node)
        pos = a.get("positions", a.get("arg0"))
        q = a.get("query", a.get("arg1"))
        k = a.get("key", a.get("arg2"))
        D = a.get("head_size", a.get("arg3"))
        cs = a.get("cos_sin_cache", a.get("arg4"))
        neox = a.get("is_neox", a.get("arg5"))
        if not isinstance(pos, ViewDesc) or not isinstance(q, ViewDesc) or not isinstance(cs, ViewDesc) or not isinstance(D, int):
            return unsupported(node, "rotary_embedding arguments not bound as expected", args={k_: _fmt(v) for k_, v in a.items()})
        if neox is not True:
            return unsupported(node, "RoPE_v1 is the NeoX rotation (is_neox must be True)", is_neox=neox)
        if cs.rank != 2 or cs.shape[1] != D or not is_row_major(cs) or cs.dtype != "bfloat16":
            return unsupported(node, "cos_sin_cache is not a row-major bf16 [MAX_POS, D] table (partial rotary is not modelled)", cs=repr(cs), D=D)
        M = q.shape[0]
        reason = _positions_ok(pos, M, ctx)
        if reason:
            return unsupported(node, reason)
        positions = tuple(int(p) for p in ctx.start.positions[:M])
        if any(p < 0 or p >= cs.shape[0] for p in positions):
            return unsupported(node, "a position is outside the cos/sin table", positions=positions[:8], rows=cs.shape[0])
        computes: list[Compute] = []
        for which, view in (("q", q), ("k", k)):
            if view is None:
                continue
            if not isinstance(view, ViewDesc):
                return unsupported(node, f"{which} is not a tensor view")
            if view.dtype != "bfloat16":
                return unsupported(node, "RoPE_v1 is bf16", dtype=view.dtype)
            width = view.shape[-1] if view.rank == 2 else (view.shape[1] * view.shape[2] if view.rank == 3 else 0)
            if width % D or width == 0 or view.shape[0] != M:
                return unsupported(node, f"{which} view is not [M, NHEADS*D]", view=repr(view), D=D, M=M)
            nheads = width // D
            try:
                v3 = split_heads(view, nheads, D)
            except GeometryError as e:
                return unsupported(node, str(e))
            computes.append(Compute(bind(b1.RoPE, NHEADS=nheads, D=D), M, (PerRow(v3), Gather(cs, positions)), (PerRow(v3),),
                                    (node.id,) if not computes else (), hint=hint_of(node, f"rope.{which}"),
                                    kernels=("rotary_embedding_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name))
        if not computes:
            return unsupported(node, "rotary_embedding without a query tensor")
        return _group_compute(computes[0], also=tuple(computes[1:]))


def _kv_moves(rc: TreeNode, ctx: StepContext, pattern: str, consumed: tuple[int, ...],
              layer: str | None = None) -> list[Transparent] | Unsupported:
    """The two `Transparent(move)`s of one `reshape_and_cache_flash(key, value, key_cache!, value_cache!,
    slot_mapping, kv_cache_dtype, k_scale, v_scale)`: row i of key/value → KV slot `slot_mapping[i]` (public; the slot
    mapping of `layer`'s KV-cache group when the step has several groups, see `kv_group_of`)."""
    a = bound_args(rc)
    try:
        group = kv_group_of(ctx, layer)
    except GeometryError as e:
        return unsupported(rc, str(e), layer=layer)
    public_slots = ctx.start.slot_mapping if group is None else (group.get("slot_mapping") or [])
    key, value = a.get("key", a.get("arg0")), a.get("value", a.get("arg1"))
    kc, vc = a.get("key_cache", a.get("arg2")), a.get("value_cache", a.get("arg3"))
    sm = a.get("slot_mapping", a.get("arg4"))
    dt = a.get("kv_cache_dtype", a.get("arg5"))
    if not all(isinstance(v, ViewDesc) for v in (key, value, kc, vc, sm)):
        return unsupported(rc, "reshape_and_cache_flash arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
    if dt not in ("auto", "bfloat16"):
        return unsupported(rc, "KV cache dtype is not bf16 (`auto`)", kv_cache_dtype=dt)
    if not (key.dtype == value.dtype == kc.dtype == vc.dtype == "bfloat16"):
        return unsupported(rc, "KV move is bf16 → bf16 only", dtypes=[key.dtype, value.dtype, kc.dtype, vc.dtype])
    try:
        kcs, vcs = as_slot_rows(kc), as_slot_rows(vc)
    except GeometryError as e:
        return unsupported(rc, str(e), key_cache=repr(kc), value_cache=repr(vc))
    kvh, d = kcs.shape[1], kcs.shape[2]
    if vcs.shape != kcs.shape:
        return unsupported(rc, "key/value caches differ in shape", key_cache=repr(kcs), value_cache=repr(vcs))
    try:
        k3, v3 = split_heads(key, kvh, d), split_heads(value, kvh, d)
    except GeometryError as e:
        return unsupported(rc, str(e), key=repr(key), value=repr(value))
    M = k3.shape[0]
    if v3.shape[0] != M:
        return unsupported(rc, "key/value row counts differ", key=repr(k3), value=repr(v3))
    if not is_int(sm) or sm.numel < M:
        return unsupported(rc, "slot_mapping view does not cover the rows", slot_mapping=repr(sm), M=M)
    if len(public_slots) < M:
        return unsupported(rc, "StepStart.slot_mapping shorter than the rows", have=len(public_slots), M=M, layer=layer)
    slots = tuple(int(s) for s in public_slots[:M])
    if any(s < 0 or s >= kcs.shape[0] for s in slots):
        return unsupported(rc, "a slot is outside the cache (or a padding slot -1)", slots=slots[:8], cache_slots=kcs.shape[0])
    k_move = Transparent("move", consumed, src=PerRow(k3), dst=Gather(kcs, slots), rows=M, hint=hint_of(rc, "kv.k"),
                         kernels=("reshape_and_cache_flash_kernel",), pattern=pattern)
    v_move = Transparent("move", (), src=PerRow(v3), dst=Gather(vcs, slots), rows=M, hint=hint_of(rc, "kv.v"),
                         kernels=("reshape_and_cache_flash_kernel",), pattern=pattern)
    return [k_move, v_move]


@dataclass
class KVCacheUpdate:
    """`vllm.unified_kv_cache_update(key, value, layer_name)` ⊃ `reshape_and_cache_flash` (any namespace) →
    `TransparentGroup(move, rows=M, src=PerRow(k as [M,KVH,D]), dst=Gather(k_cache as [blocks*bs,KVH,D], slots))`
    with the v move in `.also`; the outer op is consumed as scope (its other children must be transparent glue,
    e.g. `aten.unbind` of the `[2, blocks, bs, KVH, D]` cache).  A bare `reshape_and_cache_flash` node matches too."""

    name: str = "kv.reshape_and_cache_flash"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        name = op_name(node.op)
        if name == "reshape_and_cache_flash":
            moves = _kv_moves(node, ctx, self.name, (node.id,), layer=kv_layer_of(node))   # bare op: the module-path hint
            if isinstance(moves, Unsupported):
                return moves
            return _group_transparent(moves[0], also=tuple(moves[1:]))
        if name != "unified_kv_cache_update":
            return None
        rcs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "reshape_and_cache_flash")
        if len(rcs) != 1:
            return unsupported(node, "unified_kv_cache_update without exactly one reshape_and_cache_flash child",
                               found=[n.op for n in rcs])
        before, bad = scope_transparents(node, ctx, {rcs[0].id}, self.name)
        if bad is not None:
            return bad
        moves = _kv_moves(rcs[0], ctx, self.name, tuple(sorted(node.ids())), layer=kv_layer_of(node) or kv_layer_of(rcs[0]))
        if isinstance(moves, Unsupported):
            return moves
        return _group_transparent(moves[0], before=tuple(before), also=tuple(moves[1:]))


FA2_OP_NAMES = frozenset({"varlen_fwd", "flash_attn_varlen_func", "fwd"})


def flash_attn_op_version(op: str) -> int | None:
    """[R13 hopper] The FA library a dispatcher op belongs to, from its namespace: `_vllm_fa2_C.varlen_fwd` -> 2, `_vllm_fa3_C.fwd` -> 3
    (vllm.vllm_flash_attn.flash_attn_varlen_func dispatches on `fa_version`); None when the namespace names neither."""
    ns = op_parts(op)[0] or ""
    if "_vllm_fa3_C" in ns or "fa3" in ns:
        return 3
    if "_vllm_fa2_C" in ns or "fa2" in ns:
        return 2
    return None


@dataclass
class AttentionScope:
    """`vllm.unified_attention_with_output(query, key, value, output!, layer_name, …)` ⊃ the FA2 op
    (`_vllm_fa2_C.varlen_fwd`) → one `Compute(Attention_v3{T,NH,KVH,D,BN}, rows=|group|, (rows(q), GatherMany(k_cache,
    slots 0..T-1), GatherMany(v_cache, …)), (rows(out),))` per distinct `T_i = position_i + 1`, grouped as a
    `ComputeGroup` (primary = first group, `.also` = the rest); the scope's transparent children are consumed (moves
    emitted in `.before`); an embedded `reshape_and_cache_flash` (older layouts) is emitted in `.before` too.

    The FA2 arguments must match the registered body: `causal=True`, `softmax_scale = D**-0.5`, no window / alibi /
    softcap / dropout, `num_splits == 1` (the aligned split-KV path whose kBlockN is the pinned `bn`,
    `b1.fa2_kblock_n(D)`), bf16 everywhere.  Anything else is Unsupported."""

    bn: int
    num_splits: int | None = 1
    name: str = "attention.fa2_varlen_fwd"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return None
        if op_name(node.op) != "unified_attention_with_output":
            return None
        fas = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) in FA2_OP_NAMES)
        if len(fas) != 1:
            return unsupported(node, "unified_attention_with_output without exactly one FA2/FA3 op child", found=[n.op for n in fas],
                               descendants=[n.op for n in node.walk() if n is not node][:16])
        fa = fas[0]
        # [R13 hopper] the FA library the launch went to must be the record target's EFFECTIVE version (`TargetProfile.flash_attn_version`,
        # else vLLM's selector default for the capability): the FA2 and FA3 ops bind different registered chains (`registry.targets`).
        tp = target_of(ctx)[0]
        fa_v = tp.effective_flash_attn_version()
        op_v = flash_attn_op_version(fa.op)
        if op_v is not None and op_v != fa_v:
            return unsupported(node, f"attention op is FA{op_v} but the record target's effective flash_attn_version is {fa_v}", op=fa.op,
                               target=tp.to_json())
        if fa_v == 3:
            # FA3-only arguments the registered chain does not model: appended KV, q_v, s_aux, context parallelism.  The FP8 descales
            # (`q_descale`/`k_descale`/`v_descale`) vLLM ALWAYS passes (`layer._q_scale.expand(...)`, stride-0 views of the layer's
            # 1.0 scales) are read by the kernel only under `Is_FP8` (vllm-flash-attn 506341a1 hopper/flash_fwd_kernel_sm90.h:426
            # `if constexpr (Is_FP8 && !Has_softcap) softmax_scale_log2 *= q_descale * k_descale`; mainloop_fwd_sm90_tma_gmma_ws.hpp:1109
            # the softcap variant, :1329 `v_descale = !Is_FP8 || ptr == nullptr ? 1.0f : ...`); for the bf16 launch (checked below:
            # q/k/v/out bf16) they are dead arguments, whatever their values -- so they are not a reason to refuse a bf16 record.
            for key in ("k_new", "v_new", "q_v", "s_aux", "cp_tot_seqused_k", "leftpad_k", "kv_batch_idx", "rotary_cos", "rotary_sin"):
                if a_ := bound_args(fa).get(key):
                    if a_ is not None and a_ != 0:
                        return unsupported(node, f"FA3 op argument {key!r} is bound: not the registered chain", **{key: repr(a_)})
            for key in ("q_descale", "k_descale", "v_descale"):
                a_ = bound_args(fa).get(key)
                if a_ is not None and a_ != 0 and getattr(a_, "dtype", "float32") != "float32":
                    return unsupported(node, f"FA3 op argument {key!r} is not a float32 scale view", **{key: repr(a_)})
            cp = bound_args(fa).get("cp_world_size")
            if cp not in (None, 0, 1):
                return unsupported(node, "FA3 context parallelism is not modelled", cp_world_size=cp)
        rcs = find_descendants(node, lambda n: not n.is_triton and op_name(n.op) == "reshape_and_cache_flash")
        keep = {fa.id} | {n.id for n in rcs}
        before, bad = scope_transparents(node, ctx, keep, self.name)
        if bad is not None:
            return bad
        a = bound_args(fa)
        layer_s = kv_layer_of(node) or kv_layer_of(fa)      # KV-cache group key (hybrid KV cache); NOT the instance hint below
        layer = a.get("layer_name") or bound_args(node).get("layer_name") or ""      # hint text, unchanged (instance names)
        for rc in rcs:
            moves = _kv_moves(rc, ctx, self.name, (), layer=layer_s)
            if isinstance(moves, Unsupported):
                return moves
            before.extend(moves)
        q, k, v, out = a.get("q"), a.get("k"), a.get("v"), a.get("out")
        if not all(isinstance(x, ViewDesc) for x in (q, k, v, out)):
            return unsupported(node, "FA2 op arguments q/k/v/out not bound by name (schema unrecognised)",
                               schema=fa.enter.schema, names=list(a)[:24], views=[repr(x) for x in fa.input_views()][:8])
        causal = first(a.get("is_causal"), a.get("causal"))
        if causal is not True:
            return unsupported(node, "FA2 op is not causal", causal=causal)
        scale = as_float(a.get("softmax_scale"))
        if q.rank != 3 or out.rank != 3:
            return unsupported(node, "q/out are not [M, NH, D] views", q=repr(q), out=repr(out))
        M, NH, D = q.shape
        expect = float(D) ** -0.5
        if scale is None or not math.isclose(scale, expect, rel_tol=1e-9, abs_tol=0.0):
            return unsupported(node, "softmax_scale is not D**-0.5", softmax_scale=scale, expected=expect, D=D)
        for key in ("window_size_left", "window_size_right"):
            wv = a.get(key)
            if wv is not None and isinstance(wv, int) and wv >= 0:
                return unsupported(node, "sliding window is not modelled", **{key: wv})
        softcap = as_float(a.get("softcap"))
        if softcap not in (None, 0.0):
            return unsupported(node, "softcap is not modelled", softcap=softcap)
        if a.get("alibi_slopes") is not None:
            return unsupported(node, "alibi is not modelled")
        pdrop = as_float(first(a.get("p_dropout"), a.get("dropout_p")))
        if pdrop not in (None, 0.0):
            return unsupported(node, "dropout is not modelled", p_dropout=pdrop)
        if self.num_splits is not None:
            # paged (block-table) varlen KV forces `num_splits = 1` inside FA2 (`flash_api.cpp:763`, b1.fa2_kblock_n's
            # D54 note): the aligned split-KV kernel whose kBlockN is the pinned `bn`.  A bound `num_splits` of 0
            # (heuristic) or 1 is that path; anything else is a geometry we have not measured.
            if not isinstance(first(a.get("block_table"), a.get("page_table")), ViewDesc):
                return unsupported(node, "FA op without a block table (non-paged KV): the pinned kBlockN is the paged aligned path's",
                                   names=list(a)[:24])
            ns = a.get("num_splits")
            if ns is not None and ns not in (0, self.num_splits):
                return unsupported(node, "num_splits differs from the pinned aligned split-KV path", num_splits=ns, pinned=self.num_splits)
        if not (q.dtype == k.dtype == v.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "Attention_v3 is bf16", dtypes=[q.dtype, k.dtype, v.dtype, out.dtype])
        try:
            kcs, vcs = as_slot_rows(k), as_slot_rows(v)
        except GeometryError as e:
            return unsupported(node, str(e), k=repr(k), v=repr(v))
        KVH = kcs.shape[1]
        if kcs.shape[2] != D or vcs.shape != kcs.shape or NH % KVH:
            return unsupported(node, "KV cache geometry does not match q", q=repr(q), k_cache=repr(kcs), v_cache=repr(vcs))
        if q.strides[1:] != (D, 1) or out.shape != (M, NH, D) or out.strides[1:] != (D, 1):
            return unsupported(node, "q/out rows are not [NH][D] row-major", q=repr(q), out=repr(out))
        if M != len(ctx.start.rows):
            return unsupported(node, "FA2 query rows differ from the step's token rows", M=M, rows=len(ctx.start.rows))
        if fa_v == 2 and self.bn != b1.fa2_kblock_n(D):
            return unsupported(node, "pinned kBlockN is not the FA2 head-dim dispatch rule's", bn=self.bn, rule=b1.fa2_kblock_n(D), D=D)
        # [R13 hopper] the attention Definition of the record's target (`registry.targets.attention_spec`): Attention_v3{BN} on the
        # accepted target (Ampere step, FA2); Attention_v2{BN, DOT=<target step>, INV} otherwise — forced FA2 on Hopper: BN = fa2_kblock_n(D),
        # INV = Fa2InvSum; native FA3: BN = fa3_kblock_n(D, max_seqlen_q, NH/KVH) (the sm90 tile rule), INV = Fa3InvSum.
        kernel_name = "flash_fwd_splitkv_kernel" if fa_v == 2 else "FlashAttnFwdSm90"
        pattern_name = self.name if fa_v == 2 else "attention.fa3_fwd"
        groups: dict[int, list[int]] = {}
        for i, (req, position) in enumerate(ctx.start.rows):
            T = int(position) + 1
            seq = ctx.start.seq_lens.get(req)
            if seq is not None and T > seq:
                return unsupported(node, "row position exceeds the request's KV length", row=i, request=req, T=T, seq_len=seq)
            groups.setdefault(T, []).append(i)
        computes: list[Compute] = []
        for T, rows in groups.items():
            try:
                slots = tuple(kv_slots(ctx, ctx.start.rows[i][0], T, layer=layer_s) for i in rows)
            except GeometryError as e:
                return unsupported(node, str(e), T=T, layer=layer_s)
            if any(s >= kcs.shape[0] for ss in slots for s in ss):
                return unsupported(node, "a KV slot is outside the cache", T=T, cache_slots=kcs.shape[0])
            try:
                # the launch's `max_seqlen_q` (FA3's tile rule reads it): the most query rows any ONE request has in this step
                max_seqlen_q = max(collections.Counter(req for req, _ in ctx.start.rows).values())
                spec = _targets().attention_spec(tp, T, NH, KVH, D, max_seqlen_q)
            except _targets().UnregisteredTarget as e:
                return unsupported(node, f"target has no registered attention binding: {e}")
            computes.append(Compute(spec, len(rows),
                                    (rows_operand(q, rows), GatherMany(kcs, slots), GatherMany(vcs, slots)),
                                    (rows_operand(out, rows),), (), hint=hint_of(node, str(layer) if isinstance(layer, str) else "attn"),
                                    kernels=(kernel_name,), attribution=tuple(ctx.start.rows[i] for i in rows),
                                    pattern=pattern_name))
        primary = dataclasses.replace(computes[0], consumed=tuple(sorted(node.ids())))
        return _group_compute(primary, before=tuple(before), also=tuple(computes[1:]))


@dataclass
class SiluAndMul:
    """`_C.silu_and_mul(result!, input)` → `Compute(SiluMul_v1{I}, rows=M, (PerRow(gu[M,2I]),), (PerRow(out[M,I]),))`."""

    name: str = "act.silu_and_mul"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "silu_and_mul":
            return None
        a = bound_args(node)
        out = first(a.get("result"), a.get("out"), a.get("arg0"))
        x = first(a.get("input"), a.get("x"), a.get("arg1"))
        if not isinstance(out, ViewDesc) or not isinstance(x, ViewDesc):
            return unsupported(node, "silu_and_mul arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank != 2 or out.rank != 2 or x.shape[0] != out.shape[0] or x.shape[1] != 2 * out.shape[1] or not is_row_major(x) or not is_row_major(out):
            return unsupported(node, "silu_and_mul views are not row-major [M,2I] -> [M,I]", x=repr(x), out=repr(out))
        if not (x.dtype == out.dtype == "bfloat16"):
            return unsupported(node, "SiluMul_v1 is bf16", dtypes=[x.dtype, out.dtype])
        M, I = out.shape
        return Compute(bind(b1.SiluMul, I=I), M, (PerRow(x),), (PerRow(out),), (node.id,), hint=hint_of(node, "silu_and_mul"),
                       kernels=("act_and_mul_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name)


@dataclass
class Embedding:
    """`aten.embedding(weight[V,H], indices[M])` → `Compute(Embedding_v1{V,H}, rows=M, (PerRow(input_ids), Shared(table)),
    (PerRow(out),))`.  The token operand is the step's `input_ids` view (whose rows the fold fills from prompt Roots /
    StepLinks at StepStart) — either the indices view itself or, when vLLM converted it (`input_ids.long()`), the
    integer copy's source recorded by `classify_transparent`."""

    name: str = "embedding.aten_embedding"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.embedding":
            return None
        a = bound_args(node)
        w, idx = a.get("weight", a.get("arg0")), a.get("indices", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(w, ViewDesc) or not isinstance(idx, ViewDesc) or not outs:
            return unsupported(node, "embedding arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if w.rank != 2 or not is_row_major(w) or w.dtype != "bfloat16":
            return unsupported(node, "embedding table is not a row-major bf16 [V,H] view", weight=repr(w))
        if not is_int(idx) or idx.rank != 1:
            return unsupported(node, "embedding indices are not an integer [M] view", indices=repr(idx))
        V, H = w.shape
        M = idx.shape[0]
        out = outs[0]
        if out.shape != (M, H) or not is_row_major(out) or out.dtype != "bfloat16":
            return unsupported(node, "embedding output is not a row-major bf16 [M,H] view", out=repr(out))
        tok = self._token_view(idx, ctx)
        if isinstance(tok, str):
            return unsupported(node, tok, indices=repr(idx))
        roots = ctx.roots_of(w)
        return Compute(bind(b1.Embedding, V=V, H=H), M, (PerRow(tok), Shared(w)), (PerRow(out),), (node.id,),
                       hint=hint_of(node, roots[0].name if roots else "embedding"), kernels=("indexSelectLargeIndex", "indexSelectSmallIndex", "vectorized_gather_kernel"),  # torch 2.13 index_select fast path [L5 census]
                       attribution=attribution_for(ctx, M, node.phase), pattern=self.name)

    @staticmethod
    def _token_view(idx: ViewDesc, ctx: StepContext) -> ViewDesc | str:
        iv = ctx.start.input_ids_view
        if iv is None:
            return idx
        if idx.alloc == iv.alloc:
            return idx
        copies = ctx.scratch.get(SCRATCH_INT_COPIES, {})
        hit = copies.get(idx.alloc)
        if hit is not None:
            src, dst = hit
            if src.alloc == iv.alloc:
                if src.shape == idx.shape:
                    return src
                if src.rank == 1 and src.shape[0] >= idx.shape[0]:
                    return sub_rows(src, 0, idx.shape[0])
        return "embedding indices are neither the step's input_ids view nor an integer copy of it"


SCRATCH_TOKEN_SELECT = "token_select"  # ctx.scratch key: two-stage greedy sampler state (TokenSelectGumbelTwoStage)


def greedy_sampling(ctx: StepContext) -> str | None:
    """None when the run header records greedy sampling (temperature 0) for the whole workload, else the reason."""
    s = (ctx.header.workload or {}).get("sampling") or (ctx.header.profile or {}).get("sampling") or {}
    if s.get("greedy") is not True:
        return "the run header does not record greedy sampling"
    if s.get("temperature") not in (0, 0.0):
        return f"sampling temperature is {s.get('temperature')!r}, not 0"
    if s.get("top_k") not in (None, 0, -1) or s.get("top_p") not in (None, 1, 1.0):
        return f"top_k/top_p sampling ({s.get('top_k')!r}/{s.get('top_p')!r}) is not the registered TokenSelect"
    return None


@dataclass
class TokenSelectGumbelTwoStage:
    """vLLM V2 sampler (`vllm/v1/worker/gpu/sample/gumbel.py`), greedy path.  `gumbel_sample(logits[R,V])` is a two-stage
    argmax:

        _gumbel_sample_kernel[(R, NB)](local_argmax int64[R,NB], local_max f32[R,NB], logits bf16[R,V], seeds, pos, temp, ...)
            # per (row, block): local_max[row,b] = max(logits[row, b*BLOCK:(b+1)*BLOCK]), local_argmax[row,b] = its index
        idx     = aten.argmax(local_max, dim=-1, keepdim=True)     # int64 [R,1]: the winning block
        sampled = aten.gather(local_argmax, -1, idx)               # int64 [R,1]: the token id

    With temperature 0 the kernel adds no Gumbel noise (`temp == 0` branch) and the composite is the first global maximum
    of the row = the registered `TokenSelect_v1{V}` (tl.argmax and torch.argmax both take the first maximum).  Required:
    `APPLY_TEMPERATURE=False`, `IS_DRAFTING=False`, `PER_TOKEN_COL=False`, no logits cache, and a run header recording
    greedy sampling (the kernel branches on the device-side temperature buffer, whose value is the public workload
    sampling parameter; anything else is Unsupported).  Stage 1 (the launch) and stage 2 (the argmax over `local_max`)
    resolve to `Transparent(noop)` and leave their views in `ctx.scratch["token_select"]`; stage 3 (the gather) emits
    `Compute(TokenSelect_v1{V}, rows=R, (PerRow(logits),), (PerRow(sampled),))`."""

    vocab: int | None = None
    block_size: int = 1024
    name: str = "sample.gumbel_two_stage"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return self._stage1(node, ctx) if triton_name(node) == "_gumbel_sample_kernel" else None
        b = base_op(node.op)
        if b == "aten.argmax":
            return self._stage2(node, ctx)
        if b == "aten.gather":
            return self._stage3(node, ctx)
        return None

    def _state(self, ctx: StepContext) -> dict:
        return ctx.scratch.setdefault(SCRATCH_TOKEN_SELECT, {"stage1": {}, "stage2": {}})

    def _stage1(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        why = greedy_sampling(ctx)
        if why is not None:
            return unsupported(node, f"_gumbel_sample_kernel outside greedy sampling: {why}")
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        flags = {k: first(consts.get(k), a.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL")}
        if any(bool(v) for v in flags.values()):
            return unsupported(node, "_gumbel_sample_kernel with drafting / temperature / per-token-column flags", **flags)
        if a.get("logits_cache_ptr") is not None:
            return unsupported(node, "_gumbel_sample_kernel writing a logits cache", logits_cache=_fmt(a.get("logits_cache_ptr")))
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        logits = a.get("logits_ptr")
        local_argmax = a.get("local_argmax_ptr")
        local_max = a.get("local_max_ptr")
        if not all(isinstance(v, ViewDesc) for v in (logits, local_argmax, local_max)):
            return unsupported(node, "_gumbel_sample_kernel pointers not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if logits.rank != 2 or logits.strides[1] != 1 or logits.dtype != "bfloat16":
            return unsupported(node, "TokenSelect_v1 is defined over row-contiguous bf16 [R,V] logits", logits=repr(logits))
        R, V = logits.shape
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "logits width is not the vocabulary", V=V, vocab=self.vocab)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != V:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=V)
        if not isinstance(block, int) or block <= 0:
            return unsupported(node, "BLOCK_SIZE constexpr missing", BLOCK_SIZE=block)
        if block != self.block_size:
            return unsupported(node, "BLOCK_SIZE differs from the pinned sampler block", BLOCK_SIZE=block, pinned=self.block_size)
        nb = -(-V // block)
        if local_argmax.shape != (R, nb) or local_max.shape != (R, nb) or not is_int(local_argmax) or local_max.dtype != "float32":
            return unsupported(node, "local argmax/max buffers are not int64/f32 [R, ceil(V/BLOCK)]", local_argmax=repr(local_argmax),
                               local_max=repr(local_max), R=R, NB=nb)
        if a.get("logits_stride") not in (None, logits.strides[0]):
            return unsupported(node, "logits_stride differs from the logits view", logits_stride=a.get("logits_stride"), view=repr(logits))
        st = self._state(ctx)
        st["stage1"][local_max.alloc] = {"logits": logits, "local_argmax": local_argmax, "local_max": local_max, "kernels": (triton_name(node),)}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 1 (blockwise max/argmax)"), kernels=(triton_name(node),),
                           pattern=self.name)

    def _stage2(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOKEN_SELECT) or {}
        s1 = (st.get("stage1") or {}).get(x.alloc)
        if s1 is None or x.shape != s1["local_max"].shape:
            return None  # not the two-stage sampler's argmax; TokenSelectArgmax may claim it
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "argmax over local_max not along the block axis", dim=dim)
        out = outs[0]
        R = x.shape[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "argmax over local_max does not produce int [R]", out=repr(out))
        st["stage2"][out.alloc] = {"idx": out, **s1}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 2 (winning block)"), kernels=("ArgMaxOps",),
                           pattern=self.name)

    def _stage3(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        idx = first(a.get("index"), a.get("arg2"))
        if not isinstance(src, ViewDesc) or not isinstance(idx, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOKEN_SELECT) or {}
        s2 = (st.get("stage2") or {}).get(idx.alloc)
        if s2 is None or src.alloc != s2["local_argmax"].alloc:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "gather over local_argmax not along the block axis", dim=dim)
        logits: ViewDesc = s2["logits"]
        R, V = logits.shape
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "sampled ids are not an integer [R] view", out=repr(out))
        rows = tuple(ctx.start.logits_rows) if R == len(ctx.start.logits_rows) else ()
        return Compute(bind(b1.TokenSelect, V=V), R, (PerRow(logits),), (PerRow(out),), (node.id,),
                       hint=hint_of(node, "gumbel_sample stage 3 (token id) = argmax(logits)"), kernels=(*s2["kernels"], "ArgMaxOps", "gather_kernel"),
                       attribution=rows, pattern=self.name)


@dataclass
class TokenSelectArgmax:
    """`aten.argmax(logits[R,V], dim=-1)` (phase sample, greedy) → `Compute(TokenSelect_v1{V}, rows=R, (PerRow(logits),),
    (PerRow(out),))`.  The registered `TokenSelect_v1` is the argmax over *bf16* logits; float32 logits are Unsupported
    (reported, not silently accepted)."""

    vocab: int | None = None
    name: str = "sample.argmax_token_select"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.argmax":
            return None
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs:
            return unsupported(node, "argmax arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if x.rank == 1:
            logits = ViewDesc(x.alloc, x.offset, (1, x.shape[0]), (x.shape[0] * x.strides[0], x.strides[0]), x.dtype, x.device)
            if dim not in (None, 0, -1):
                return unsupported(node, "argmax over a 1-D tensor with an unexpected dim", dim=dim)
        elif x.rank == 2 and dim in (1, -1):
            logits = x
        else:
            return unsupported(node, "argmax is not over the last axis of [R,V] logits", view=repr(x), dim=dim)
        R, V = logits.shape
        if logits.strides[1] != 1:
            return unsupported(node, "logits rows are not contiguous", view=repr(logits))
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "argmax width is not the vocabulary", V=V, vocab=self.vocab)
        why = greedy_sampling(ctx)
        if why is not None:
            return unsupported(node, f"argmax over the logits outside greedy sampling: {why}")
        if logits.dtype == "float32":
            # the sampler's `logits.to(torch.float32)`: argmax over the exact widening of the bf16 logits is the argmax
            # over the bf16 logits (the registered `TokenSelect_v1` body); the operand is the bf16 source view
            narrow = widened_source(logits, ctx)
            if narrow is None:
                return unsupported(node, "TokenSelect_v1 is defined over bf16 logits; these float32 logits are not an exact "
                                         "widening of a bf16 view seen this step", view=repr(logits))
            logits = narrow
        if logits.dtype != "bfloat16":
            return unsupported(node, "TokenSelect_v1 is defined over bf16 logits", dtype=logits.dtype)
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "argmax output is not an integer [R] view", out=repr(out))
        return Compute(bind(b1.TokenSelect, V=V), R, (PerRow(logits),), (PerRow(out),), (node.id,), hint=hint_of(node, "argmax"),
                       kernels=("ArgMaxOps",), attribution=tuple(ctx.start.logits_rows) if R == len(ctx.start.logits_rows) else (),
                       pattern=self.name)


SCRATCH_TOPP_SELECT = "topp_token_select"  # ctx.scratch key: the seeded temperature + top-p sampler chain (TokenSelectGumbelTopP)
TOPP_STATS_BLOCK, TOPP_STEP_BLOCK, TOPP_FANOUT, TOPP_ROUNDS = 8192, 2048, 8, 5   # vLLM `_apply_topp_split` pins (registry/topp_split.py)
TEMPERATURE_BLOCK = 8192                                                          # `apply_temperature` BLOCK_SIZE


def header_sampling(ctx: StepContext) -> dict:
    return dict((ctx.header.workload or {}).get("sampling") or (ctx.header.profile or {}).get("sampling") or {})


def request_sampling(ctx: StepContext) -> dict[str, dict]:
    """request id -> {seed, temperature, top_p, top_k} of the run header's request record (children of an n>1 request: the
    recorded fork fact parent seed + index); the workload's sampling block supplies what a record does not pin."""
    wl = ctx.header.workload or {}
    s = header_sampling(ctx)
    per = {p["request_id"]: p for p in s.get("per_request", []) or []}
    out: dict[str, dict] = {}
    for r in wl.get("requests", []) or []:
        p = per.get(r["request_id"], {})
        out[r["request_id"]] = {"seed": None if r.get("seed") is None else int(r["seed"]),
                                "temperature": float(first(r.get("temperature"), p.get("temperature"), s.get("temperature", 0.0))),
                                "top_p": float(first(r.get("top_p"), p.get("top_p"), s.get("top_p", 1.0))),
                                "top_k": int(first(r.get("top_k"), p.get("top_k"), s.get("top_k", 0)) or 0)}
    for c in wl.get("children", []) or []:
        base = dict(out.get(c["parent"], {}))
        base["seed"] = None if c.get("seed") is None else int(c["seed"])
        out[c["request_id"]] = base
    return out


def _f32_bits(x: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclass
class TokenSelectGumbelTopP:
    """[R15 stoch] vLLM V2 sampler (d9105ea80 `Sampler.sample`), the SEEDED temperature + top-p path (an explicit `SamplingParams.seed`
    disables FlashInfer sampling; `top_k` off).  The launch chain of one sampling step, on `logits32 = f32(logits)` (the exact widening
    `classify_transparent` records):

        _temperature_kernel[(R, ceil(V/8192))](logits32, stride, expanded_idx_mapping, temperature[·], V)      logits32[r] /= T[r] in place
        aten.index(top_p[·], expanded_idx_mapping) -> p f32[R]                                                  request fact (header)
        _topp_sb_stats_kernel[(R*S,)] / _topp_sb_step_kernel x5 / _topp_sb_mask_kernel   (HAS_K=False, S, F=8, NUM_ROUNDS=5)  -inf mask in place
        _gumbel_sample_kernel[(R, ceil(V/1024))](local_argmax, local_max, logits32, idx_mapping, seeds[·], pos[·], temperature[·], V;
                                                 APPLY_TEMPERATURE=False, IS_DRAFTING=False, USE_FP64=False, PER_TOKEN_COL=False)
        aten.argmax(local_max) -> winning block ; aten.gather(local_argmax) -> the token id

    Every launch but the last resolves to `Transparent(noop)` that claims its kernel and records the chain in `ctx.scratch["topp_token_select"]`
    (keyed by the fp32 logits allocation); the gather emits, per logits row r of request `rid` at position `pos`,

        Compute(GumbelTopPTokenSelect_v1{V}, 1, (PerRow(logits_bf16[r]), Const32[top_p], Param(seed of rid), Const64[pos], Const32[T], Const32[S]),
                (PerRow(sampled[r]),))

    -- the registry composite (`verity_vllm/program/registry/sampling.py`: TemperatureScale -> TopPMask{V} (topp_split model) -> GumbelSelectF32),
    the same instance the derived Program's `GumbelTopPTokenSelect[vllm]_v1` rule emits: temperature / top_p / pos are Program constants
    (the fold emits their `Const` instances in operand order right before the select), the seed is the request's `seed` PARAMETER.  S
    ([R16 stoch, coord M-0169] the 6th operand, `Const32[S]` "splits") is read from the split kernels' constexpr (the launch geometry:
    live batch x SM count), must agree across the three kernels, and is bound at compare to the request Program's `splits` Input entry
    for the event; the relation S = SplitsFor_v1(|live(t)|, num_SMs) is checked per engine step by GM-01 G6.  Values
    (which seed / T / p the kernels consumed) are NOT read from device memory here: they are the header's request record; the value check
    (`bench.stoch_recompute`) recomputes the served token from the snapshotted logits under the DECLARED seed and fails on a wrong one.

    Unsupported (fail closed, each a named reason): a greedy header (the registered TokenSelect patterns' case, returns None); top-k
    (`_topk_topp_kernel`); min_p / logit bias / penalties / bad words (a write into logits32 before the temperature kernel makes it not the
    exact widening); an unseeded sampled row; a row at T in {0, 1} or p == 1 (other Program families); missing / duplicated / unpinned
    split kernels; the fp64 Gumbel path; a logits cache."""

    vocab: int | None = None
    block_size: int = 1024
    name: str = "sample.gumbel_topp_select"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if header_sampling(ctx).get("greedy") is True:
            return None
        if node.is_triton:
            k = triton_name(node)
            if k == "_temperature_kernel":
                return self._temperature(node, ctx)
            if k in ("_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel"):
                return self._topp(node, ctx, k)
            if k == "_gumbel_sample_kernel":
                return self._stage1(node, ctx)
            if k in ("_topk_topp_kernel", "_min_p_kernel", "_logit_bias_kernel", "_penalties_kernel", "_bad_words_kernel"):
                return unsupported(node, f"{k}: the top-k / min_p / logit-bias / penalties sampler paths have no registered Definition "
                                         "(only the seeded temperature + p-only split pipeline is bound: GumbelTopPTokenSelect_v1)")
            return None
        b = base_op(node.op)
        if b == "aten.argmax":
            return self._stage2(node, ctx)
        if b == "aten.gather":
            return self._stage3(node, ctx)
        if b == "aten.index" and node.phase == "sample":
            return self._state_gather(node, ctx)
        return None

    def _state(self, ctx: StepContext) -> dict:
        return ctx.scratch.setdefault(SCRATCH_TOPP_SELECT, {"rows": {}, "gathers": {}, "stage1": {}, "stage2": {}})

    def _state_gather(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        """`top_p.gpu[expanded_idx_mapping]` (and `top_k.gpu[...]`): a 1-D per-request sampling-state buffer (max_num_reqs entries)
        gathered by the step's request indices -- a request fact of the header, not an activation.  Remembered so the split kernels'
        `p_ptr` can be checked to be exactly this gather."""
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        outs = node.output_views()
        if not isinstance(src, ViewDesc) or src.rank != 1 or src.dtype not in ("float32", "int32", "int64") or not outs:
            return None
        idxs = a.get("indices", a.get("arg1"))
        if not isinstance(idxs, (list, tuple)) or len(idxs) != 1 or not isinstance(idxs[0], ViewDesc) or not is_int(idxs[0]):
            return None
        out = outs[0]
        self._state(ctx)["gathers"][out.alloc] = {"src": src, "out": out, "dtype": src.dtype}
        return Transparent("noop", (node.id,), hint=hint_of(node, f"per-request sampling-state gather ({src.dtype}[{src.numel}]): a request fact"),
                           pattern=self.name)

    def _row_state(self, node: TreeNode, ctx: StepContext, logits: Any) -> dict | Unsupported:
        """The chain record of the fp32 logits allocation `logits` writes / reads, created at first sight from the exact widening."""
        if not isinstance(logits, ViewDesc) or logits.dtype != "float32" or logits.rank != 2 or logits.strides[1] != 1:
            return unsupported(node, "sampler logits are not a row-contiguous float32 [R,V] view", logits=_fmt(logits))
        st = self._state(ctx)["rows"]
        rec = st.get(logits.alloc)
        if rec is None:
            narrow = widened_source(logits, ctx)
            if narrow is None or narrow.dtype != "bfloat16" or narrow.shape != logits.shape:
                return unsupported(node, "the float32 logits are not the exact widening of a bf16 [R,V] view seen this step (a logit-bias / "
                                         "penalties / bad-words / min_p write before the sampler chain?)", logits=repr(logits), narrow=_fmt(narrow))
            rec = {"bf16": narrow, "f32": logits, "temperature": False, "topp": collections.Counter(), "S": None, "kernels": []}
            st[logits.alloc] = rec
        elif logits.shape != rec["f32"].shape or logits.offset != rec["f32"].offset:
            return unsupported(node, "a sampler kernel on a different view of the logits allocation", logits=repr(logits), chain=repr(rec["f32"]))
        return rec

    def _temperature(self, node: TreeNode, ctx: StepContext) -> Resolution:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        rec = self._row_state(node, ctx, a.get("logits_ptr"))
        if isinstance(rec, Unsupported):
            return rec
        if rec["temperature"] or rec["topp"]:
            return unsupported(node, "_temperature_kernel after another sampler kernel on the same logits (chain order)", chain=rec["kernels"])
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        if block != TEMPERATURE_BLOCK:
            return unsupported(node, "_temperature_kernel BLOCK_SIZE differs from the pinned block", BLOCK_SIZE=block, pinned=TEMPERATURE_BLOCK)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != rec["f32"].shape[1]:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=rec["f32"].shape[1])
        rec["temperature"] = True
        rec["kernels"].append(triton_name(node))
        return Transparent("noop", (node.id,), hint=hint_of(node, "temperature: logits32[r] /= T[r] in place (div.full.f32; TemperatureScale_v1)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _topp(self, node: TreeNode, ctx: StepContext, k: str) -> Resolution:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        rec = self._row_state(node, ctx, first(a.get("LOGITS"), a.get("logits_ptr")))     # the split kernels' signature: LOGITS, STATS, PARTS, K, P, ROUND
        if isinstance(rec, Unsupported):
            return rec
        R, V = rec["f32"].shape
        if bool(first(consts.get("HAS_K"), a.get("HAS_K"))):
            return unsupported(node, f"{k} with HAS_K: the top-k + top-p split path has no registered Definition")
        vs = first(consts.get("VOCAB_SIZE"), a.get("VOCAB_SIZE"))
        if vs is not None and int(vs) != V:
            return unsupported(node, "VOCAB_SIZE constexpr differs from the logits width", VOCAB_SIZE=vs, V=V)
        S = first(consts.get("S"), a.get("S"))
        if not isinstance(S, int) or S <= 0 or S & (S - 1) or S > 32:
            return unsupported(node, "S (splits) is not a power of two <= 32", S=S)
        if rec["S"] is not None and rec["S"] != S:
            return unsupported(node, "S differs between the split kernels of one step", S=S, chain_S=rec["S"])
        pins = {"_topp_sb_stats_kernel": {"BLOCK": TOPP_STATS_BLOCK},
                "_topp_sb_step_kernel": {"BLOCK": TOPP_STEP_BLOCK, "F": TOPP_FANOUT, "NUM_ROUNDS": TOPP_ROUNDS},
                "_topp_sb_mask_kernel": {"BLOCK": TOPP_STATS_BLOCK, "F": TOPP_FANOUT, "NUM_ROUNDS": TOPP_ROUNDS}}[k]
        for key, want in pins.items():
            got = first(consts.get(key), a.get(key))
            if got != want:
                return unsupported(node, f"{k} {key} differs from the pinned split pipeline", **{key: got, "pinned": want})
        if k == "_topp_sb_mask_kernel":
            mv = first(consts.get("MASK_VALUE"), a.get("MASK_VALUE"))
            if mv is not None and not (isinstance(mv, float) and math.isinf(mv) and mv < 0):
                return unsupported(node, "_topp_sb_mask_kernel MASK_VALUE is not -inf", MASK_VALUE=mv)
        p = first(a.get("P"), a.get("p_ptr"))
        g = self._state(ctx)["gathers"].get(getattr(p, "alloc", None))
        if not isinstance(p, ViewDesc) or g is None or g["dtype"] != "float32" or p.numel != R:
            return unsupported(node, f"{k} P is not the step's float32 [R] gather of the per-request top_p state", p=_fmt(p), R=R)
        # HAS_K=False launches pass the LOGITS pointer itself as the dummy `K` (`k_ptr = k if k is not None else logits`): any other buffer
        # there is a top-k buffer of a path that is not bound
        kk = first(a.get("K"), a.get("k_ptr"))
        if isinstance(kk, ViewDesc) and kk.alloc != rec["f32"].alloc:
            return unsupported(node, f"{k} with a top-k buffer bound (HAS_K path)", k=_fmt(kk))
        order = {"_topp_sb_stats_kernel": 0, "_topp_sb_step_kernel": 1, "_topp_sb_mask_kernel": 2}[k]
        seen = rec["topp"]
        if order == 0 and seen:
            return unsupported(node, "_topp_sb_stats_kernel after another split kernel", chain=rec["kernels"])
        if order == 1 and (seen["_topp_sb_stats_kernel"] != 1 or seen["_topp_sb_mask_kernel"]):
            return unsupported(node, "_topp_sb_step_kernel outside stats -> steps -> mask", chain=rec["kernels"])
        if order == 1:
            ri = first(a.get("ROUND"), a.get("round_i"))
            if isinstance(ri, int) and ri != seen["_topp_sb_step_kernel"]:
                return unsupported(node, "_topp_sb_step_kernel round out of order", ROUND=ri, expected=seen["_topp_sb_step_kernel"])
        if order == 2 and (seen["_topp_sb_stats_kernel"] != 1 or seen["_topp_sb_step_kernel"] != TOPP_ROUNDS or seen["_topp_sb_mask_kernel"]):
            return unsupported(node, f"_topp_sb_mask_kernel without exactly one stats + {TOPP_ROUNDS} step launches before it", chain=rec["kernels"])
        seen[k] += 1
        rec["S"] = S
        rec["kernels"].append(triton_name(node))
        return Transparent("noop", (node.id,), hint=hint_of(node, f"top-p split pipeline S={S}: {k} (TopPMask_v1{{V}}, splits operand)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage1(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        flags = {k: first(consts.get(k), a.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL", "USE_FP64")}
        if any(bool(v) for v in flags.values()):
            return unsupported(node, "_gumbel_sample_kernel with drafting / in-kernel temperature / per-token-column / fp64 flags", **flags)
        if a.get("logits_cache_ptr") is not None:
            return unsupported(node, "_gumbel_sample_kernel writing a logits cache", logits_cache=_fmt(a.get("logits_cache_ptr")))
        logits = a.get("logits_ptr")
        local_argmax = a.get("local_argmax_ptr")
        local_max = a.get("local_max_ptr")
        if not all(isinstance(v, ViewDesc) for v in (logits, local_argmax, local_max)):
            return unsupported(node, "_gumbel_sample_kernel pointers not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        rec = self._row_state(node, ctx, logits)
        if isinstance(rec, Unsupported):
            return rec
        R, V = rec["f32"].shape
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "logits width is not the vocabulary", V=V, vocab=self.vocab)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != V:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=V)
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        if not isinstance(block, int) or block != self.block_size:
            return unsupported(node, "BLOCK_SIZE differs from the pinned sampler block", BLOCK_SIZE=block, pinned=self.block_size)
        nb = -(-V // block)
        if local_argmax.shape != (R, nb) or local_max.shape != (R, nb) or not is_int(local_argmax) or local_max.dtype != "float32":
            return unsupported(node, "local argmax/max buffers are not int64/f32 [R, ceil(V/BLOCK)]", local_argmax=repr(local_argmax),
                               local_max=repr(local_max), R=R, NB=nb)
        seen = rec["topp"]
        chain_ok = seen["_topp_sb_stats_kernel"] == 1 and seen["_topp_sb_step_kernel"] == TOPP_ROUNDS and seen["_topp_sb_mask_kernel"] == 1
        if not chain_ok:
            return unsupported(node, "_gumbel_sample_kernel without the complete split pipeline (stats, 5 steps, mask) on its logits: rows at "
                                     "top_p == 1 are GumbelTokenSelect_v1 (not bound in this profile)", chain=rec["kernels"])
        rows = list(ctx.start.logits_rows)
        if len(rows) != R:
            return unsupported(node, "logits rows do not match the step's sampled rows", R=R, logits_rows=len(rows))
        rs = request_sampling(ctx)
        specs: list[tuple[str, int, int, float, float]] = []
        for rid, pos in rows:
            q = rs.get(rid)
            if q is None:
                return unsupported(node, "sampled row of a request the run header does not record", request=rid)
            T, p, seed, top_k = q["temperature"], q["top_p"], q["seed"], q["top_k"]
            if top_k not in (0, -1):
                return unsupported(node, "top_k row in a p-only split batch (the header contradicts the launch)", request=rid, top_k=top_k)
            if T in (0.0, 1.0):
                return unsupported(node, "row at temperature 0 / 1 in a stochastic batch: TokenSelect_v1 / an unscaled row (not bound "
                                         "together with GumbelTopPTokenSelect_v1 in this profile)", request=rid, temperature=T)
            if not rec["temperature"]:
                return unsupported(node, "header temperature != 1 but no _temperature_kernel launch on the logits", request=rid, temperature=T)
            if p >= 1.0 or p <= 0.0:
                return unsupported(node, "row at top_p outside (0, 1) in a split-pipeline batch", request=rid, top_p=p)
            if seed is None:
                return unsupported(node, "unseeded sampled row: vLLM's SamplingStates drew the seed on the host (`seed: null` in the request "
                                         "record) -- the RNG stream is not a declared input", request=rid, position=int(pos))
            specs.append((rid, int(pos), int(seed), T, p))
        st = self._state(ctx)
        st["stage1"][local_max.alloc] = {"bf16": rec["bf16"], "local_argmax": local_argmax, "local_max": local_max, "S": int(rec["S"]),
                                          "specs": specs, "kernels": tuple(rec["kernels"]) + (triton_name(node),)}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 1 (noised blockwise max/argmax over the masked row)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage2(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOPP_SELECT) or {}
        s1 = (st.get("stage1") or {}).get(x.alloc)
        if s1 is None or x.shape != s1["local_max"].shape:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "argmax over local_max not along the block axis", dim=dim)
        out = outs[0]
        R = x.shape[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "argmax over local_max does not produce int [R]", out=repr(out))
        st["stage2"][out.alloc] = {"idx": out, **s1}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 2 (winning block)"), kernels=("ArgMaxOps",),
                           pattern=self.name)

    def _stage3(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        idx = first(a.get("index"), a.get("arg2"))
        if not isinstance(src, ViewDesc) or not isinstance(idx, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOPP_SELECT) or {}
        s2 = (st.get("stage2") or {}).get(idx.alloc)
        if s2 is None or src.alloc != s2["local_argmax"].alloc:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "gather over local_argmax not along the block axis", dim=dim)
        logits: ViewDesc = s2["bf16"]
        R, V = logits.shape
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "sampled ids are not an integer [R] view", out=repr(out))
        if out.rank == 1:
            out = ViewDesc(out.alloc, out.offset, (R, 1), (out.strides[0], 1), out.dtype, out.device)
        from verity_vllm.program.registry import sampling as _sampling
        spec = bind(_sampling.GumbelTopPTokenSelect, V=V)
        S = int(s2["S"])        # the split kernels' constexpr: the launch's S -> the 6th operand (bound to the Program's `splits` Input)
        kernels = (*s2["kernels"], "ArgMaxOps", "gather_kernel")
        parts = []
        for r, (rid, pos, seed, T, p) in enumerate(s2["specs"]):
            operands = (PerRow(sub_rows(logits, r, r + 1)), Const(32, _f32_bits(p), "top_p"), Param("seed", (rid,)),
                        Const(64, int(pos) & 0xFFFFFFFFFFFFFFFF, "pos"), Const(32, _f32_bits(T), "temperature"), Const(32, S, "splits"))
            parts.append(Compute(spec, 1, operands, (PerRow(sub_rows(out, r, r + 1)),), (node.id,),
                                 hint=hint_of(node, f"gumbel_topp_token_select row {r}"), kernels=kernels if r == 0 else (),
                                 attribution=((rid, pos),), pattern=self.name))
        primary = parts[0]
        return ComputeGroup(primary.spec, primary.rows, primary.operands, primary.outputs, primary.consumed, hint=primary.hint,
                            kernels=primary.kernels, attribution=primary.attribution, pattern=self.name, also=tuple(parts[1:]))


@dataclass
class LogitsGather:
    """`aten.index.Tensor(hidden[M,H], [logits_indices])` / `aten.index_select(hidden, 0, logits_indices)` with the
    step's public `logits_indices` → `Transparent(move, rows=R, src=Gather(hidden, logits_indices), dst=PerRow(out))`.
    Integer sources are left to the integer-metadata rule."""

    name: str = "sample.logits_gather"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) not in ("aten.index", "aten.index_select"):
            return None
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        if not isinstance(src, ViewDesc) or is_int(src):
            return None
        if base_op(node.op) == "aten.index":
            idxs = a.get("indices", a.get("arg1"))
            if not isinstance(idxs, (list, tuple)):
                return unsupported(node, "aten.index without an index list")
            tensors = [i for i in idxs if isinstance(i, ViewDesc)]
            if len(tensors) != 1 or idxs[0] is None and len(idxs) > 1:
                return unsupported(node, "aten.index with an index pattern other than one leading index tensor", indices=_fmt(idxs))
            idx = tensors[0]
        else:
            dim = a.get("dim", a.get("arg1"))
            idx = a.get("index", a.get("arg2"))
            if dim not in (0,) or not isinstance(idx, ViewDesc):
                return unsupported(node, "index_select not along the leading axis", dim=dim)
        outs = node.output_views()
        if not outs or not is_int(idx) or idx.rank != 1:
            return unsupported(node, "gather index is not an integer [R] view", index=repr(idx))
        R = idx.shape[0]
        li = ctx.start.logits_indices
        if R != len(li):
            return unsupported(node, "gather rows differ from the step's logits_indices", R=R, logits_indices=len(li))
        if src.rank < 1 or any(i < 0 or i >= src.shape[0] for i in li):
            return unsupported(node, "logits_indices outside the gathered source", src=repr(src), logits_indices=list(li)[:8])
        M = len(ctx.start.rows)
        if src.shape[0] not in (M, ctx.start.num_tokens_padded):
            return unsupported(node, "gathered source is not the step's [M, ...] token rows", src=repr(src), M=M)
        out = outs[0]
        if out.shape != (R,) + src.shape[1:]:
            return unsupported(node, "gather output shape does not match", out=repr(out), src=repr(src))
        return Transparent("move", (node.id,), src=Gather(src, tuple(int(i) for i in li)), dst=PerRow(out), rows=R,
                           hint=hint_of(node, "hidden[logits_indices]"), kernels=("index_elementwise_kernel", "indexSelectLargeIndex", "indexSelectSmallIndex"),
                           pattern=self.name)


# ---------------------------------------------------------------------------------------------------------
# labels for reports
# ---------------------------------------------------------------------------------------------------------


def label(r: Resolution) -> str:
    if isinstance(r, Compute):
        return f"Compute {r.spec.id} x{r.rows}"
    if isinstance(r, Transparent):
        return f"Transparent {r.effect}"
    return f"Unsupported: {r.reason}"
