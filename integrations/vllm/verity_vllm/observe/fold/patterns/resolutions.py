"""Grouped resolutions and the resolution helpers every pattern shares (`unsupported`, `hint_of`, `attribution_for`, report
labels).

Grouped resolutions
-------------------
Several raw events produce more than one resolution: `_C.rotary_embedding` rotates q *and* k (two `RoPE_v1`
specializations from one dispatcher call), `reshape_and_cache_flash` moves k *and* v, the attention op yields one
`Attention_v3{T}` compute per distinct T, and a scope pattern (attention / KV update) may have transparent children
(slices, copies) that the fold must still execute.  `resolver.resolve_step` accepts a single `Resolution` per root, so
this module carries the extra members on the primary resolution (`ComputeGroup.before / .also`,
`TransparentGroup.before / .also`); `members(r)` yields the fold's execution order `before, primary, also`.  Only the
primary carries `consumed` (the whole subtree); the other members have `consumed=()`.
`resolver.resolve_step` and `ResolutionReport.add` see only the primary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.resolver import Compute, Group, Resolution, StepContext, Transparent, Unsupported, parts_of
from verity_vllm.observe.fold.tree import TreeNode


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


def _check_pins(node: TreeNode, got: dict[str, Any], pins: dict[str, Any]) -> Unsupported | None:
    bad = {k: (got.get(k), v) for k, v in pins.items() if got.get(k) != v}
    if bad:
        return unsupported(node, "launch constexpr/geometry differs from the pinned profile",
                           mismatches={k: {"observed": o, "pinned": p} for k, (o, p) in bad.items()})
    return None


def _group_compute(c: Compute, before: tuple = (), also: tuple = ()) -> ComputeGroup:
    return ComputeGroup(c.spec, c.rows, c.operands, c.outputs, c.consumed, c.hint, c.kernels, c.attribution, c.pattern,
                        before=before, also=also)


def _group_transparent(t: Transparent, before: tuple = (), also: tuple = ()) -> TransparentGroup:
    return TransparentGroup(t.effect, t.consumed, t.src, t.dst, t.rows, t.hint, t.kernels, t.pattern, before=before, also=also)


def label(r: Resolution) -> str:
    if isinstance(r, Compute):
        return f"Compute {r.spec.id} x{r.rows}"
    if isinstance(r, Transparent):
        return f"Transparent {r.effect}"
    return f"Unsupported: {r.reason}"
