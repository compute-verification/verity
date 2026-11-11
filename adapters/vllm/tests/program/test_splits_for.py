"""`splits_for(n, num_SMs)`, the split count vLLM's own `_apply_topp_split` picks from its live sampler batch, which serving overrides
with `registered_splits(num_SMs)` (`engine.topp_pin`).  CPU, torch-free.

  * the threshold tables at 142 (L40S), 132 (H100) and 188 (RTX PRO 6000) SMs, computed: they differ, so the S an unpinned launch
    takes is a function of (n, num_SMs);
  * range: a power of two in 1..32, non-increasing in n, non-decreasing in num_SMs; `registered_splits` is its value at n = 1 and
    refuses a num_SMs that is no target profile;
  * the real record: the B8 staggered workload served UNPINNED (157 engine steps, 526 sampled rows, 8 requests arriving 0..44 with
    caps 2..128): the mask kernel's constexpr S at every step == splits_for(live count at that step, 142), flipping
    (step 0, live 1, S 32) -> (step 32, live 5, S 16) -> (step 101, live 4, S 32) -- the batch-dependence the pin removes.
"""
from __future__ import annotations

import json
import os

import pytest

from verity.definitions.registry.topp_geometry import SPLIT_MAX_BATCH, registered_splits, splits_for

DATA = os.path.join(os.path.dirname(__file__), "data", "topp_split_fixture")


def _table(num_sms: int, n_max: int = 200) -> list[tuple[int, int, int]]:
    out, start, prev = [], 1, splits_for(1, num_sms)
    for n in range(2, n_max + 1):
        s = splits_for(n, num_sms)
        if s != prev:
            out.append((start, n - 1, prev)); start, prev = n, s
    out.append((start, n_max, prev))
    return out


def test_threshold_tables_l40s_and_h100_are_computed_not_tabled():
    assert _table(142) == [(1, 4, 32), (5, 8, 16), (9, 17, 8), (18, 35, 4), (36, 71, 2), (72, 200, 1)]
    assert _table(132) == [(1, 4, 32), (5, 8, 16), (9, 16, 8), (17, 33, 4), (34, 66, 2), (67, 200, 1)]
    # the two devices disagree at exactly these live counts: a per-class table would be wrong there
    diff = [n for n in range(1, 200) if splits_for(n, 142) != splits_for(n, 132)]
    assert diff == [17, 34, 35, 67, 68, 69, 70, 71]


def test_threshold_table_rtx_pro_6000_188_sms():
    assert _table(188) == [(1, 5, 32), (6, 11, 16), (12, 23, 8), (24, 47, 4), (48, 94, 2), (95, 200, 1)]
    # inside the split pipeline's domain (n_live <= 64) it disagrees with the L40S's 142 at these live counts; S = 1 is unreachable
    diff = [n for n in range(1, SPLIT_MAX_BATCH + 1) if splits_for(n, 188) != splits_for(n, 142)]
    assert diff == [5, 9, 10, 11, 18, 19, 20, 21, 22, 23, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    assert {splits_for(n, 188) for n in range(1, SPLIT_MAX_BATCH + 1)} == {2, 4, 8, 16, 32}


def test_registered_splits_is_the_lone_row_s_and_refuses_a_num_sms_that_is_no_profile():
    assert [registered_splits(m) for m in (1, 2, 16, 31, 32, 132, 142, 188)] == [1, 2, 16, 16, 32, 32, 32, 32]
    assert all(registered_splits(m) == splits_for(1, m) for m in range(1, 300))
    for bad in (0, -142):
        with pytest.raises(ValueError, match=f"num_SMs={bad} is not a target profile"):
            registered_splits(bad)


def test_range_is_a_power_of_two_value6_monotone():
    for m in (1, 2, 8, 64, 108, 132, 142, 144, 256):
        prev = None
        for n in range(1, SPLIT_MAX_BATCH + 1):
            s = splits_for(n, m)
            assert 1 <= s <= 32 and s & (s - 1) == 0
            assert s * n <= m or s == 1                    # the loop invariant: S * n_live <= num_SMs (S = 1 unconditionally)
            assert s == 32 or 2 * s * n > m                # maximality: the next power of two would not fit
            assert prev is None or s <= prev               # non-increasing in the live count
            prev = s
    for n in (1, 5, 9, 17, 36, 64):
        assert all(splits_for(n, m) <= splits_for(n, m + 1) for m in range(1, 300))


def test_r15_b8_unpinned_record_kernel_S_equals_splits_for_of_live_at_every_step():
    j = json.load(open(os.path.join(DATA, "b8_s_schedule_r15_attempt2.json")))
    assert j["engine_steps"] == 157 and j["sampled_rows"] == 526 and j["num_sms"] == 142 and j["every_request_served_to_cap"]
    reqs = j["served_vs_cap"]
    steps = j["steps"]
    assert len(steps) == 157
    flips, prev = [], None
    for st in steps:
        t = st["step"]
        # the declared chronology: n_live_t = #{r : arrive_r <= t <= arrive_r + cap_r - 1}
        n_decl = sum(1 for r in reqs.values() if r["arrive"] <= t <= r["arrive"] + r["cap"] - 1)
        assert n_decl == st["live_declared"] == st["live_realized"] == len(st["requests"])
        S = splits_for(n_decl, 142)
        assert S == st["S_kernel"], (t, n_decl, S, st["S_kernel"])
        if S != prev:
            flips.append((t, n_decl, S)); prev = S
    assert flips == [(0, 1, 32), (32, 5, 16), (101, 4, 32)]
    assert sum(len(st["requests"]) for st in steps) == 526
