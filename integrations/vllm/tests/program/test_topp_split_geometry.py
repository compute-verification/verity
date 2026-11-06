"""`TopPMask_v1{V,S}` (registry/topp_split.py) against GPU truth at EVERY split geometry S = 1, 2, 4, 8, 16, 32 --
a model fitted at S = 32 only mis-assigns each thread's Z (and the ladder / pivot /
dup / numkeep / num_preceding derived from it) at S = 16.  CPU only; the kernel outputs come from the committed fixture
(tests/program/data/topp_split_fixture/, produced on an L40S by `verity_vllm.pipeline.topp_split_probe produce`, vLLM
d9105ea80 / Triton 3.7.1 / torch 2.13.0+cu129; `manifest.json` records the environment and every case's parameters).

  1. the ORIGINAL failure: the 3 preserved B8-record rows (bf16 logits row, kernel-consumed masked fp32 row, device facts) -> 0 lanes
     differ (4 / 1 / 7 under the S = 32 grouping);
  2. the per-thread Z group: the S=16 probe's kernel-observed Z at the record's 13 + 20 + 32 pivot-tie lanes == the model's
     `Zh[z_group(lane, 16)]` at every lane, and the OLD grouping (`lane >= 16`) contradicts it (the finding, pinned);
  3. the GPU fixture: 180 rows over S in {1..32} x V in {1024, 3000, 4096, 128256} -- random rows, CONSTRUCTED pivot ties (plateaus of
     identical logits straddling split boundaries, both Z groups, several tie budgets incl. numkeep = 1 and numdup - 1) and the record
     rows at every S -- masked row + stats bit-exact; and the real `_apply_topp_split` entry on a B = 64 -> 1 drain (S flips
     2 -> 4 -> 8 -> 16 -> 32 as a batch empties) == the constexpr-S model row by row.
  4. S = 32 is unchanged (the 431-row validation at S = 32 stays valid): z_group(lane, 32) == (lane >= 16).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from verity.ml.definitions.registry import topp_split as TS
from verity.ml.definitions.registry import topp_geometry

f32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "data", "topp_split_fixture")
RECORD = os.path.join(FIX, "fstoch01_rows.npz")
INTER = os.path.join(FIX, "fstoch01_s16_intermediates.npz")
NEG_INF_BITS = 0xFF800000


def _record():
    z = np.load(RECORD)
    meta = json.load(open(RECORD + ".json"))
    return z, meta


def _x_in(z, meta, key):
    bf = (z[key + "_bf16_logits_bits"].astype(np.uint32) << 16).view(f32)
    d = meta[key]["device"]
    return TS.tl_div(bf, f32(d["temperature"]), TS.Cfg()), float(d["top_p"]), int(meta[key]["S"]), int(meta[key]["V"])


# ------------------------------------------------------------------------------------------------ 1. the original failure


@pytest.mark.parametrize("key", ["step37_row4", "step82_row3", "step92_row3"])
def test_fstoch01_record_row_is_bit_exact_at_s16(key):
    z, meta = _record()
    x_in, p, S, V = _x_in(z, meta, key)
    assert S == 16 and V == 128256
    out, stats, parts, tr = TS.topp_split_row(x_in, p, S)
    obs = z[key + "_masked_f32_bits"]
    diff = np.nonzero(out.view(np.uint32) != obs)[0]
    assert diff.size == 0, f"{key}: {diff.size} lanes differ from the kernel-consumed masked row: {diff[:8].tolist()}"
    assert tr["ties"] is True                       # every one of the 3 rows exercises the duplicate budget
    # the S=16 probe's stats + step-kernel partials (real kernels on the same rows) are reproduced in every executed round
    pr = np.load(INTER)
    ri = ["step37_row4", "step82_row3", "step92_row3"].index(key)
    assert np.array_equal(stats.view(np.uint32), pr["stats"][ri].view(np.uint32))
    fin_round = int(tr["fin"]) // TS.SPLIT_FANOUT
    for r in range(fin_round + 1):
        assert np.array_equal(parts[r].view(np.uint32), pr["parts"][ri][r].view(np.uint32)), f"{key}: round {r} partials differ"


def test_fstoch01_old_grouping_reproduces_the_failure():
    """Negative: with the S=32 grouping (lane >= 16) forced at S=16 the 3 rows differ at exactly the recorded lanes."""
    z, meta = _record()
    expected = {"step37_row4": [10082, 18507, 20705, 64606], "step82_row3": [76847],
                "step92_row3": [15785, 21976, 27202, 28994, 44214, 58835, 62833]}
    orig = TS.z_group
    try:
        TS.z_group = lambda lane, S: (np.asarray(lane) >= 16).astype(np.int64)
        for key, lanes in expected.items():
            x_in, p, S, V = _x_in(z, meta, key)
            out = TS.topp_split_row(x_in, p, S)[0]
            diff = np.nonzero(out.view(np.uint32) != z[key + "_masked_f32_bits"])[0].tolist()
            assert diff == lanes, (key, diff)
    finally:
        TS.z_group = orig


# ------------------------------------------------------------------------------------------------ 2. the per-thread Z group


def test_zgroup_matches_kernel_observed_lane_Z_at_s16_and_old_grouping_does_not():
    z, meta = _record()
    pr = np.load(INTER)
    from verity_vllm.pipeline.topp_split_probe import thread_lane
    any_old_wrong = False
    for ri, key in enumerate(["step37_row4", "step82_row3", "step92_row3"]):
        x_in, p, S, V = _x_in(z, meta, key)
        cfg = TS.Cfg.for_vocab(V, S)
        stats = TS.stats_kernel(x_in, p, S, TS.STATS_BLOCK, cfg)
        M, Zh, mn, nf = TS.row_stats(stats, S, cfg)
        Zw = np.asarray(Zh, f32).view(np.uint32)
        assert Zw[0] != Zw[1], "the two per-thread Z values differ by 1 ulp on these rows (z_nunique == 2 in the probe)"
        lanes = pr[key + "_tie_lanes"]; zobs = pr[key + "_tie_zlane"]
        th = np.array([thread_lane(int(v), V, S, cfg.spt_x) for v in lanes])
        assert np.array_equal(Zw[TS.z_group(th, S)], zobs), key
        any_old_wrong |= not np.array_equal(Zw[(th >= 16).astype(int)], zobs)
    assert any_old_wrong, "the old warp-half grouping must contradict the kernel on at least one tie lane"


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


# ------------------------------------------------------------------------------------------------ 3. the GPU fixture


@pytest.mark.slow
def test_gpu_fixture_every_split_geometry_bit_exact():
    from verity_vllm.pipeline.topp_split_probe import check
    rep = check(FIX, RECORD, verbose=False)
    assert rep["ok"], rep["fail"][:5]
    assert rep["n"] >= 180 and rep["mask_ok"] == rep["n"] and rep["stats_ok"] == rep["n"]
    assert rep["ties_cases"] >= 150 and rep["boundary_tie_cases"] >= 60           # the duplicate budget is what the S = 32 grouping breaks
    assert rep["zgroup_checked"] >= 8 and rep["zgroup_ok"] == rep["zgroup_checked"]
    man = json.load(open(os.path.join(FIX, "manifest.json")))
    assert sorted({c["S"] for c in man["cases"] if c["kind"] != "drain"}) == [1, 2, 4, 8, 16, 32]
    drain = {d["B"]: d for d in rep["drain"]}
    assert {drain[B]["S"] for B in (5, 6, 7, 8)} == {16} and {drain[B]["S"] for B in (1, 2, 3, 4)} == {32}   # the B8 workload's flip
    assert drain[9]["S"] == 8 and drain[18]["S"] == 4 and drain[64]["S"] == 2
    assert all(d["rows_ok"] == d["B"] for d in rep["drain"])


@pytest.mark.parametrize("num_sm,added", [(142, {35, 36}), (188, {11, 12, 23, 24, 47, 48}), (132, {16, 33, 34})])
def test_drain_crosses_every_split_boundary_of_the_devices_sm_count(num_sm, added):
    from verity_vllm.pipeline.topp_split_probe import DRAIN_BATCHES, drain_batches
    bs = drain_batches(num_sm)
    assert list(bs) == sorted(bs, reverse=True) and set(bs) - set(DRAIN_BATCHES) == added
    for n in range(1, topp_geometry.SPLIT_MAX_BATCH):
        if topp_geometry.splits_for(n, num_sm) != topp_geometry.splits_for(n + 1, num_sm):
            assert {n, n + 1} <= set(bs), (num_sm, n)
    assert {topp_geometry.splits_for(b, num_sm) for b in bs} == {topp_geometry.splits_for(n, num_sm) for n in range(1, topp_geometry.SPLIT_MAX_BATCH + 1)}


def test_multi_block_slice_stats_are_fma_accumulated():
    """S <= 8 at V = 128256 spans > 1 BLOCK per slice: `exp_sum * rescale + tl.sum(e)` is fma-contracted (7/7 fixture rows);
    the single-block case (S = 16 / 32) is unaffected -- fma(0, 0, se) == se."""
    z, meta = _record()
    man = json.load(open(os.path.join(FIX, "manifest.json")))
    fx = np.load(os.path.join(FIX, "fixture.npz"))
    checked = 0
    for c in man["cases"]:
        if c["kind"] != "record" or c["S"] > 8:
            continue
        key = list(meta)[c["row"]]
        x_in, p, _, V = _x_in(z, meta, key)
        for fma, want in ((True, True), (False, False)):
            cfg = TS.Cfg.for_vocab(V, c["S"]); cfg.fma_exp_sum = fma
            st = TS.stats_kernel(x_in, p, c["S"], TS.STATS_BLOCK, cfg)
            eq = np.array_equal(st.view(np.uint32), fx[f"c{c['id']}_stats"].view(np.uint32))
            if fma:
                assert eq, (c["id"], c["S"])
            elif not eq:
                checked += 1
    assert checked >= 3, "the non-fma model must differ from the kernel on several multi-block rows (otherwise the test is vacuous)"
