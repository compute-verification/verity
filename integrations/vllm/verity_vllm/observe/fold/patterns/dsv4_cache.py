"""DeepSeek-V4's paged caches as its folds read them: a window cache block's NoPE, scale and RoPE rows (`swa_block`, `fp8_rows`), a
compressor's state rows (`state_block`, `state_row`), the tokens' slots in a layer's KV group (`_slots`, `_entry_slots`) and the step's
roots by name.  The DeepSeek-V4 patterns (`dsv4`) and FlashMLA's sparse attention (`dsv4_flashmla`) share them.
"""

from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

import math

from verity_vllm.observe.events import Root, ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError, is_int
from verity_vllm.observe.fold.patterns.kv_cache import kv_group_of
from verity_vllm.observe.fold.patterns.resolutions import unsupported
from verity_vllm.observe.fold.resolver import Shared, StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode


def _module(node: TreeNode) -> str:
    return getattr(node.enter, "module", None) or ""


def _roots(ctx: StepContext) -> dict[str, Root]:
    """The step's roots by name."""
    if "dsv4_roots" not in ctx.scratch:
        ctx.scratch["dsv4_roots"] = {r.name: r for rs in ctx.roots.values() for r in rs}
    return ctx.scratch["dsv4_roots"]


def _weights(node: TreeNode, ctx: StepContext, names: tuple[str, ...]) -> tuple[Shared, ...] | Unsupported:
    rs = _roots(ctx)
    if (missing := [n for n in names if n not in rs]):
        return unsupported(node, "the capture registered no root for weights the call reads", missing=missing)
    return tuple(Shared(rs[n].view) for n in names)


def _is(v: ViewDesc, shape: tuple[int, ...], dtype: str) -> bool:
    return v.shape == shape and v.dtype == dtype


def _slots(node: TreeNode, ctx: StepContext, layer: str, bs: object, sm: ViewDesc, T: int) -> tuple[int, ...] | Unsupported:
    """The T tokens' slots in `layer`'s KV group, once the group's blocks are the launch's `bs` and the slot buffer covers the tokens."""
    try:
        g = kv_group_of(ctx, layer)
    except GeometryError as e:
        return unsupported(node, str(e))
    slots = tuple(int(s) for s in ((g or {}).get("slot_mapping") or ())[:T])
    if g is None or int(g["block_size"]) != bs or len(slots) != T or not is_int(sm) or sm.numel < T or any(s < 0 for s in slots):
        return unsupported(node, "the cache's KV group does not give every token a slot in blocks of the launch's size", layer=layer,
                           block_size=bs, slots=slots[:8])
    return slots


def _entry_slots(node: TreeNode, ctx: StepContext, layer: str, req: str, entries, per_block: int, ratio: int = 1) -> tuple[int, ...] | Unsupported:
    """The slots of `entries` of request `req` in `layer`'s KV group: entry e at table[e // n] * n + e % n, n the group's block size
    over `ratio` (a compressed cache's entries a block, flashmla.py:191), which must be the `per_block` the cache holds."""
    try:
        g = kv_group_of(ctx, layer)
    except GeometryError as e:
        return unsupported(node, str(e))
    bt = ((g or {}).get("block_tables") or {}).get(req)
    n = int((g or {}).get("block_size") or 0) // ratio
    if n != per_block:
        return unsupported(node, "the cache's entries a block are not its KV group's block size over the ratio", layer=layer,
                           per_block=per_block, ratio=ratio, group_block_size=(g or {}).get("block_size"))
    if bt is None or n <= 0 or any(e < 0 or e // n >= len(bt) for e in entries):
        return unsupported(node, "the KV group's block table does not hold every entry", layer=layer, request=req, entries=list(entries)[:8])
    return tuple(int(bt[e // n]) * n + e % n for e in entries)


def swa_block(cache: ViewDesc, block: int, bs: int, DN: int, DR: int, QB: int) -> tuple[ViewDesc, ViewDesc, ViewDesc]:
    """Block `block` of a layer's window cache (uint8 [blocks, block bytes]) as its bs tokens' NoPE [bs, DN] e4m3, scale bytes
    [bs, DN/QB + 1] and RoPE [bs, DR] bf16: the tokens' DN + 2 DR data bytes each, then their scale bytes each
    (`cache_utils.quantize_and_insert_k_cache`)."""
    row, ns = DN + 2 * DR, DN // QB + 1
    if cache.dtype != "uint8" or cache.rank != 2 or cache.strides[1] != 1 or cache.shape[1] != bs * (row + ns) or not 0 <= block < cache.shape[0]:
        raise GeometryError(f"block {block} of a window cache of {bs} rows of {row} + {ns} bytes: {cache!r}")
    base = cache.offset + block * cache.strides[0]
    if (base + DN) % 2:
        raise GeometryError(f"block {block}'s RoPE bytes are not bf16-aligned: {cache!r}")
    return (ViewDesc(cache.alloc, base, (bs, DN), (row, 1), "float8_e4m3fn", cache.device),
            ViewDesc(cache.alloc, base + bs * row, (bs, ns), (ns, 1), "uint8", cache.device),
            ViewDesc(cache.alloc, (base + DN) // 2, (bs, DR), (row // 2, 1), "bfloat16", cache.device))


def block_bytes(cache: ViewDesc) -> ViewDesc:
    """A paged uint8 cache [blocks, ...] whose blocks are contiguous as [blocks, block bytes], `swa_block`'s view."""
    if cache.dtype != "uint8" or cache.rank < 2 or cache.strides[1:] != tuple(math.prod(cache.shape[i + 1:]) for i in range(1, cache.rank)):
        raise GeometryError(f"a paged cache whose blocks are not contiguous bytes: {cache!r}")
    return ViewDesc(cache.alloc, cache.offset, (cache.shape[0], math.prod(cache.shape[1:])), (cache.strides[0], 1), "uint8", cache.device)


def _row(v: ViewDesc, i: int, n: int | None = None) -> ViewDesc:
    """Row i of a 2-D view as a 1-D view of its first n elements (all by default)."""
    return ViewDesc(v.alloc, v.offset + i * v.strides[0], (n or v.shape[1],), (v.strides[1],), v.dtype, v.device)


def fp8_rows(cache: ViewDesc, slots, bs: int, DN: int, DR: int, QB: int) -> list[tuple[ViewDesc, ViewDesc, ViewDesc]]:
    """Each slot's NoPE bytes, DN / QB scale bytes (not the pad byte) and RoPE elements in a paged fp8_ds_mla cache [blocks, block
    bytes] of bs rows a block (`swa_block`): the row `FlashMlaKvRow_v1` and `FlashMlaGatherRow_v1` read."""
    out = []
    for s in slots:
        nope, scale, rope = swa_block(cache, s // bs, bs, DN, DR, QB)
        out.append((_row(nope, s % bs), _row(scale, s % bs, DN // QB), _row(rope, s % bs)))
    return out


def runs(keys: list) -> list[tuple[object, int, int]]:
    """(key, first row, end row) of each run of consecutive rows with one key."""
    out: list[list] = []
    for i, k in enumerate(keys):
        if out and out[-1][0] == k:
            out[-1][2] = i + 1
        else:
            out.append([k, i, i + 1])
    return [(k, lo, hi) for k, lo, hi in out]


def block_runs(slots: tuple[int, ...], bs: int) -> list[tuple[int, int, int]]:
    """(block, first row, end row) of each run of consecutive rows whose slots are in one block."""
    return runs([s // bs for s in slots])


def state_block(cache: ViewDesc, block: int, W: int) -> tuple[ViewDesc, ViewDesc]:
    """Block `block` of a compressor's state cache (f32 [blocks, bs, 2W]) as its bs slots' kv words [bs, W] and score words [bs, W]."""
    if cache.dtype != "float32" or cache.rank != 3 or cache.shape[2] != 2 * W or cache.strides[2] != 1 or not 0 <= block < cache.shape[0]:
        raise GeometryError(f"block {block} of a state cache of rows of 2 x {W} f32 words: {cache!r}")
    base, bs = cache.offset + block * cache.strides[0], cache.shape[1]
    return tuple(ViewDesc(cache.alloc, base + k * W, (bs, W), (cache.strides[1], 1), "float32", cache.device) for k in (0, 1))


def state_row(state: ViewDesc, slot: int, k: int, W: int, at: int, D: int) -> ViewDesc:
    """D words of slot `slot` of a compressor's state cache (f32 [blocks, bs, 2W]) from word `at` of its kv (k = 0) or score (k = 1)
    half."""
    bs = state.shape[1]
    return ViewDesc(state.alloc, state.offset + slot // bs * state.strides[0] + slot % bs * state.strides[1] + k * W + at, (D,), (1,),
                    "float32", state.device)


__all__ = ["block_bytes", "block_runs", "fp8_rows", "runs", "state_block", "state_row", "swa_block"]
