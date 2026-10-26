"""The rotary embedding pattern, and the positions check it shares with the profile patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, is_int, is_row_major, split_heads
from verity_vllm.observe.fold.patterns.ops import bound_args, op_name
from verity_vllm.observe.fold.patterns.resolutions import _fmt, _group_compute, attribution_for, hint_of, unsupported
from verity_vllm.observe.fold.patterns.transparent import SCRATCH_INT_COPIES
from verity_vllm.observe.fold.resolver import Compute, Gather, PerRow, Resolution, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode


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


def step_positions(node: TreeNode, ctx: StepContext, pos: Any, rows: int) -> tuple[int, ...] | Unsupported:
    """The public positions of a kernel's first `rows` rows, once `pos` is the step's positions buffer (`_positions_ok`)."""
    if not isinstance(pos, ViewDesc):
        return unsupported(node, "positions argument is not a tensor view", positions=repr(pos))
    why = _positions_ok(pos, rows, ctx)
    if why:
        return unsupported(node, why)
    out = tuple(int(p) for p in ctx.start.positions[:rows])
    if len(out) != rows:
        return unsupported(node, "StepStart positions do not cover the kernel rows", rows=rows, positions=len(out))
    return out


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
