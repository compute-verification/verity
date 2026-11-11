"""The transparent aten ops (DESIGN §6, rows "view/…", "copy_/…", "empty/…") with the integer-metadata rule, and the scope
helpers: a wrapper op whose subtree contains the real kernel op plus transparent glue."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_int, is_row_major, sub_rows
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args, op_name, triton_name
from verity_vllm.observe.fold.patterns.resolutions import hint_of, unsupported
from verity_vllm.observe.fold.resolver import PerRow, Resolution, Shared, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode

ALIAS_OPS = frozenset({
    "aten.view", "aten._unsafe_view", "aten.reshape", "aten.t", "aten.transpose", "aten.permute", "aten.split",
    "aten.split_with_sizes", "aten.slice", "aten.select", "aten.unsqueeze", "aten.squeeze", "aten.expand",
    "aten.expand_as", "aten.alias", "aten.detach", "aten.narrow", "aten.as_strided", "aten.unbind", "aten.unfold",
    "aten.view_as_real", "aten.view_as", "aten.flatten", "aten.unflatten", "aten.lift_fresh", "aten.resolve_conj",
    "aten.resolve_neg", "aten.detach_", "aten.movedim", "aten.chunk", "aten.tensor_split",
    "aten.numpy_T",   # `Tensor.T` (all axes reversed): the FP8 linears pass `weight.T` / `weight_scale_inv.T` to cutlass_scaled_mm
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
SCRATCH_INT_FILLS = "int_fills"     # ctx.scratch key: alloc -> [(view, value)] of the integer `fill_`s seen this step, in order
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
            # `aten.reshape` (and `flatten`) COPY when the requested shape is not expressible as a view of the input's
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
            if b == "aten.fill_":
                ctx.scratch.setdefault(SCRATCH_INT_FILLS, {}).setdefault(outs[0].alloc, []).append((outs[0], bound_args(node).get("value")))
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
