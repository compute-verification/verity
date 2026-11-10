"""`test_rmsnorm_fused`'s sweep of the whole B0 commit store (its bundle under `vllm-poc/bundles`, built on a pod, which no tracked
file holds). Skipped where it is absent, or where `test_rmsnorm_fused`'s numerics can't be built.
"""
from __future__ import annotations

import pytest

from tests.program.test_rmsnorm_fused import _numerics


def test_b0_store_sweep(tmp_path):
    """The whole B0 store: v2 exact on all 2,340 fused-norm rows; v1 misses 13 (the class of the B1 §5.1 row)."""
    _numerics()
    from verity_vllm.program import rmsnorm_fused_sweep as SW
    if not SW.BUNDLE.exists():
        pytest.skip(f"B0 bundle not present: {SW.BUNDLE}")
    commit = SW.extract_bundle(SW.BUNDLE, tmp_path)
    rep = SW.sweep(commit)
    assert rep["rows"] == 2340
    assert rep["trees"]["v2"]["rows_exact"] == 2340
    assert rep["trees"]["v2"]["words_exact"] == rep["trees"]["v2"]["words_compared"] == 2 * 2340 * 576
    assert rep["trees"]["v1"]["rows_exact"] == 2327
    assert rep["v1_vs_v2"]["output_differs_rows"] == 13
    for r in rep["v1_vs_v2"]["rows"]:
        assert r["v2"] == r["captured"] and r["v1"] != r["captured"]
