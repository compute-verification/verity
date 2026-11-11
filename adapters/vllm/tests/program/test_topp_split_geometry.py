"""`TopPMask_v1{V,S}`'s split geometry (registry/topp_split.py), which reads no record:

  1. the per-thread Z group at each split count S = 2, 4, 8, 16, 32 is the upper half of each S-lane group (one Z at S = 1),
     so S = 32 is unchanged (the 431-row validation at S = 32 stays valid): z_group(lane, 32) == (lane >= 16);
  2. the GPU fixture's drain (`topp_split_probe.drain_batches`) crosses every split boundary of a device's SM count.

The model against GPU truth at every S, on the recorded rows, is `test_topp_split_geometry_of_record`.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.definitions.registry import topp_split as TS
from verity.definitions.registry import topp_geometry


def test_zgroup_structure():
    lane = np.arange(32)
    assert np.array_equal(TS.z_group(lane, 32), (lane >= 16).astype(int))          # S = 32: the warp half (the original fit)
    assert np.array_equal(TS.z_group(lane, 16), ((lane % 16) >= 8).astype(int))    # S = 16: half of each 16-lane group
    assert np.array_equal(TS.z_group(lane, 8), ((lane % 8) >= 4).astype(int))
    assert np.array_equal(TS.z_group(lane, 4), ((lane % 4) >= 2).astype(int))
    assert np.array_equal(TS.z_group(lane, 2), (lane % 2).astype(int))
    assert not TS.z_group(lane, 1).any()                                            # S = 1: one Z
    for S in (2, 4, 8, 16, 32):
        assert TS.z_group(lane, S).sum() == 16                                      # the two groups are always half the warp each


@pytest.mark.parametrize("num_sm,added", [(142, {35, 36}), (188, {11, 12, 23, 24, 47, 48}), (132, {16, 33, 34})])
def test_drain_crosses_every_split_boundary_of_the_devices_sm_count(num_sm, added):
    from verity_vllm.pipeline.topp_split_probe import DRAIN_BATCHES, drain_batches
    bs = drain_batches(num_sm)
    assert list(bs) == sorted(bs, reverse=True) and set(bs) - set(DRAIN_BATCHES) == added
    for n in range(1, topp_geometry.SPLIT_MAX_BATCH):
        if topp_geometry.splits_for(n, num_sm) != topp_geometry.splits_for(n + 1, num_sm):
            assert {n, n + 1} <= set(bs), (num_sm, n)
    assert {topp_geometry.splits_for(b, num_sm) for b in bs} == {topp_geometry.splits_for(n, num_sm) for n in range(1, topp_geometry.SPLIT_MAX_BATCH + 1)}
