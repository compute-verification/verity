"""GLM-5.3's DSA folds' shared step state (each layer's caches, page table and top-k allocation as its launches establish them) and
the `fp8_ds_mla` latent cache's layout, which `dsa.DsaNormRope` writes and `dsa_flashmla.DsaFlashMlaDecode` reads."""

from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

from dataclasses import dataclass

from verity.device import load_card, number

from verity_vllm.engine.profiles.devices import CARD
from verity_vllm.engine.profiles.engine_classes import CLASS_G
from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import GeometryError
from verity_vllm.observe.fold.patterns.resolutions import unsupported
from verity_vllm.observe.fold.resolver import StepContext, Unsupported
from verity_vllm.observe.fold.tree import TreeNode

RECORD_SMS = int(number(load_card(CARD[CLASS_G["device_name"]])["sm"]["count"]))


def _module(node: TreeNode) -> str:
    return getattr(node.enter, "module", None) or ""


def _root(ctx: StepContext, v: ViewDesc) -> str | None:
    """The checkpoint parameter a weight view reads (`Compute.roots`): GLM-5.3's Triton launches carry no module, so the hint alone
    names one layer's norm weight like every other layer's and the Match's weight map fuses all of them into one overlapping range;
    unnamed, the rotary table's reads (k_pe, q_pe, index-K / Q) map under the hint's `k_pe`, a name the Build has no field for."""
    rs = ctx.roots_of(v)
    return rs[0].name if rs else None


def _state(ctx: StepContext) -> dict:
    return ctx.scratch.setdefault("glm53_dsa", {"modules": {}, "topk": None})


def _module_state(ctx: StepContext, node: TreeNode) -> dict:
    return _state(ctx)["modules"].setdefault(_module(node), {})


def _slots(node: TreeNode, ctx: StepContext, rows: int, limit: int) -> tuple[int, ...] | Unsupported:
    if len(ctx.start.slot_mapping) < rows:
        return unsupported(node, "StepStart slot_mapping does not cover the kernel rows", rows=rows, slots=len(ctx.start.slot_mapping))
    out = tuple(int(s) for s in ctx.start.slot_mapping[:rows])
    if any(s < 0 or s >= limit for s in out):
        return unsupported(node, "a current-token slot is outside the cache", slots=out[:8], cache_slots=limit)
    return out


@dataclass(frozen=True)
class DsMlaCache:
    """The `fp8_ds_mla` latent cache as `_fused_norm_rope_kernel` writes it under ``MLA_CACHE_DS_MLA`` (vLLM d9105ea80
    deepseek_v32/common/kernels.py): slot s's entry of W bytes holds its KVL e4m3 NoPE bytes, then its KVL / TILE f32 scales, then its
    R bf16 RoPE words.  ``nope``, ``scale`` and ``rope`` are the three as one row a slot, over the kernel's ``mla_cache_ptr`` and its
    f32 and bf16 views of the same bytes; FlashMLA reads the entries as ``kv`` [blocks, BS, 1, W] bytes."""

    nope: ViewDesc
    scale: ViewDesc
    rope: ViewDesc
    blocks: int
    BS: int
    W: int

    @property
    def slots(self) -> int:
        return self.nope.shape[0]


def _ds_mla_cache(m: ViewDesc, s: ViewDesc, r: ViewDesc, KVL: int, tiles: int, R: int) -> DsMlaCache:
    """`DsMlaCache` of the kernel's ``mla_cache_ptr`` (e4m3 [blocks, BS, W]), ``mla_cache_ds_scale_ptr`` (the same bytes as f32) and
    ``mla_cache_ds_rope_ptr`` (as bf16)."""
    if m.rank != 3 or m.dtype != "float8_e4m3fn" or m.strides != (m.shape[1] * m.shape[2], m.shape[2], 1):
        raise GeometryError(f"fp8_ds_mla cache is not a row-major e4m3 [blocks, block, entry]: {m!r}")
    nb, BS, W = m.shape
    if KVL + 4 * tiles + 2 * R != W or m.offset % 4:
        raise GeometryError(f"fp8_ds_mla entry of {W} bytes is not {KVL} NoPE bytes, {tiles} f32 scales and {R} bf16 RoPE words: {m!r}")
    for v, dtype, size in ((s, "float32", 4), (r, "bfloat16", 2)):
        if v.alloc != m.alloc or v.dtype != dtype or size * v.offset != m.offset or v.shape != (nb, BS, W // size) \
                or v.strides != (BS * W // size, W // size, 1):
            raise GeometryError(f"fp8_ds_mla {dtype} view {v!r} is not the cache {m!r} as {dtype}")
    n = nb * BS
    return DsMlaCache(ViewDesc(m.alloc, m.offset, (n, KVL), (W, 1), m.dtype, m.device),
                      ViewDesc(s.alloc, s.offset + KVL // 4, (n, tiles), (W // 4, 1), s.dtype, s.device),
                      ViewDesc(r.alloc, r.offset + (KVL + 4 * tiles) // 2, (n, R), (W // 2, 1), r.dtype, r.device), nb, BS, W)


__all__ = ["RECORD_SMS", "DsMlaCache"]
