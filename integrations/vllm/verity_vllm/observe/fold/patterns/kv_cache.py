"""KV-cache addressing (a layer's KV-cache group, the KV slots of a request's positions) and the KV-cache update pattern."""

from __future__ import annotations

from dataclasses import dataclass

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, as_slot_rows, is_int, split_heads
from verity_vllm.observe.fold.patterns.ops import bound_args, op_name
from verity_vllm.observe.fold.patterns.resolutions import _fmt, _group_transparent, hint_of, unsupported
from verity_vllm.observe.fold.patterns.transparent import find_descendants, scope_transparents
from verity_vllm.observe.fold.resolver import Gather, PerRow, Resolution, StepContext, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode


def kv_group_of(ctx: StepContext, layer: str | None) -> dict | None:
    """Hybrid KV cache: the KV-cache group whose block tables / slot mapping address `layer`'s
    KV rows, or None when the step has a single group (the legacy `StepStart.block_tables` / `slot_mapping` are complete).

    vLLM's hybrid KV-cache manager gives layers of different attention kinds separate groups -- Gemma-2: the sliding-window
    (even) layers and the global (odd) layers -- each with its own block tables and slot mapping into the SAME physical KV
    tensors (layers 2k and 2k+1 share one tensor).  Modelling every layer with group 0's tables puts the even layers' rows
    at the odd layers' cells: prefill still replays (reads resolve by provenance to the same step's mis-addressed writes)
    but decode reads the neighbour layer's K/V.  With several groups a layer whose group is unknown is a
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
