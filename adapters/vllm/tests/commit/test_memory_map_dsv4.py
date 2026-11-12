"""memory-map/v1 of DeepSeek-V4-Pro's attention caches, for a warm request's: its sliding window, compressor state and
compressed (c4, c128) and indexer caches, which a prefix-cache hit reads from the request that computed them as the dense
attention's K and V (`commit.memory_map`, `cached`).  Neither is on main: no DeepSeek-V4 Program takes a cached prefix (only the
dense attention rule seeds one, `kv_prefix`), and the map lays out none of these caches, so the test is marked to fail until both
land, on the attention layer of `tests/program/test_dsv4_derive.py` at ratio 4 over a 30-row prefill and two decodes."""
from __future__ import annotations

import numpy as np
import pytest

from tests.program import test_dsv4_derive as V4
from tests.program.test_dsv4_derive import flashmla_small, served  # noqa: F401  (DeepSeek-V4's FlashMLA and H200 fixtures)
from verity_vllm.commit import memory_map as MM


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="memory-map/v1 lays out none of DeepSeek-V4's caches, and no DeepSeek-V4 Program takes a cached prefix")
def test_a_warm_deepseek_v4_requests_compressed_caches_are_its_producers(flashmla_small):
    rng = np.random.default_rng(4)
    model = V4._SparseAttn(rng, 4, 28, (30, 1, 1))
    rep = V4._derive_mhc(model, tuple(map(V4._tensor, V4._sparse_inputs(rng, model))))
    assert rep.program is not None, rep.unsupported
    MM.rows(rep.program)
    pytest.fail("memory-map/v1 lays out DeepSeek-V4's caches now: replace this marker with a warm request's test that its window, "
                "compressor state, c4, c128 and indexer pages are its producers' and written once")
