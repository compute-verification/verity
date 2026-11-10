"""GLM-5.3's index-K cache as the DSA fold reads it, in vLLM d9105ea80's block layout (`_fp8_quant_and_cache_write`,
deepseek_v32/common/kernels.py:63-87): block b of the [blocks, 64, 132] byte cache holds its 64 slots' 128 e4m3 bytes one row after
another, then their 64 f32 scales, so slot s = 64 b + o has its row at byte 8448 b + 128 o and its scale at byte 8448 b + 8192 + 4 o.

* every slot of a cache inside a larger allocation maps to those bytes, the kernel's addressing transcribed here;
* a cache whose geometry or launch arguments differ from that layout is refused by name;
* the fold of a GLM-5.3 rank's step writes each token's `DsaIndexerKRow_v1` row and scale there.  Read as 132-byte slot rows
  ([128 e4m3 | scale] a slot), a block's first slot has the same row bytes and only its scale word elsewhere (bytes 128..131, the
  next slot's first e4m3 bytes): run A8's twin found `DsaIndexerKRow_v1` off the capture at exactly that word.
"""
from __future__ import annotations

import pytest

from tests.check import test_glm53_rank_rehearsal as R
from tests.observe.test_fold_glm53_defaults import view
from verity_vllm.observe.fold.profiles import generic as G
from verity_vllm.observe.events import TritonLaunch
from verity_vllm.observe.fold.fold import fold
from verity_vllm.observe.fold.patterns import dsa as DSA
from verity_vllm.observe.fold.patterns.geometry import GeometryError
from verity_vllm.observe.fold.views import element_runs, row_view

BS, W, D = 64, 132, 128


def _kernel_bytes(base: int, s: int) -> tuple[int, int]:
    """`_fp8_quant_and_cache_write`'s byte addresses of slot s's row and scale in a cache at byte ``base``."""
    block_start = base + (s // BS) * BS * W
    return block_start + (s % BS) * D, block_start + BS * D + (s % BS) * 4


def _caches(alloc: int, blocks: int, base: int):
    return (view(alloc, (blocks, BS, W), "uint8", offset=base),
            view(alloc, (blocks, BS, W // 4), "float32", offset=base // 4))


def test_each_slot_maps_to_the_bytes_the_kernel_writes() -> None:
    blocks, base = 3, 5 * BS * W
    ix = DSA._index_cache(*_caches(7, blocks, base), BS, W, D)
    assert ix.slots == blocks * BS
    for s in range(ix.slots):
        row, scale = _kernel_bytes(base, s)
        assert element_runs(row_view(ix.rows, ix.row(s))) == [(row, D)], s
        sv = row_view(ix.scales, ix.scale(s))
        assert (sv.alloc, sv.dtype, sv.byte_offset, sv.numel) == (7, "float32", scale, 1), s


@pytest.mark.parametrize("change", ["block_size", "stride", "scale_alloc", "scale_offset", "width", "strides"])
def test_a_cache_off_the_layout_is_refused(change: str) -> None:
    k, s = _caches(7, 2, 0)
    block_size, stride = BS, W
    if change == "block_size":
        block_size = 32
    elif change == "stride":
        stride = 136
    elif change == "scale_alloc":
        s = view(8, s.shape, "float32")
    elif change == "scale_offset":
        s = view(7, s.shape, "float32", offset=1)
    elif change == "width":
        k, s, stride = view(7, (2, BS, 130), "uint8"), view(7, (2, BS, 32), "float32"), 130
    elif change == "strides":
        k = view(7, k.shape, "uint8", strides=(BS * W + 4, W, 1))
    with pytest.raises(GeometryError, match="index"):
        DSA._index_cache(k, s, block_size, stride, D)


def test_the_fold_writes_each_tokens_k_row_and_scale_where_the_kernel_does() -> None:
    profile = G.build_profile("GLM53", 8, hf_config=R.CFG, repo="zai-org/GLM-5.3-FP8", revision="aca966e4")
    events = R._log(profile)
    res = fold(events, profile, strict=False, with_program=False)
    assert not res.report.unsupported
    caches = {e.args["indexer_cache_ptr"].alloc: e.args["indexer_cache_ptr"] for e in events
              if isinstance(e, TritonLaunch) and e.fn.endswith(DSA.NORM_KERNEL)}
    krows = [m for m in res.instances if m["spec"].startswith("DsaIndexerKRow_v1")]
    assert len(caches) == len(R.LAYERS) and len(krows) == len(R.LAYERS) * R.M
    for m in krows:
        q, scale = m["output_views"]
        cache = caches[q.alloc]
        row, sb = _kernel_bytes(cache.byte_offset, m["row"])
        assert element_runs(q) == [(row, D)] and (scale.alloc, scale.dtype, scale.byte_offset, scale.numel) == (q.alloc, "float32", sb, 1)
    first = next(m for m in krows if m["row"] == 0)["output_views"][1]
    assert first.byte_offset - caches[first.alloc].byte_offset == BS * D != D
