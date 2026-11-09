"""`SplitsFor_v1(n_live, num_SMs)` -- the split count S of vLLM's `_apply_topp_split` as a VALUE of the live sampler batch.
CPU, torch-free.

  * the threshold tables at 142 (L40S) and 132 (H100) SMs, computed -- they differ (n = 17, 34..35, 67..71), so S is a function of
    (n, num_SMs), never a per-device-class table;
  * domain: n_live in 1..64, num_SMs >= 1; n_live = 0 (no sampling event) and n_live > 64 (not the split pipeline) REFUSE by name;
  * range: a power of two in 1..32 (Value<6>), non-increasing in n_live, non-decreasing in num_SMs;
  * the real record: the B8 staggered workload (157 engine steps, 526 sampled rows, 8 requests arriving 0..44 with caps 2..128): the
    mask kernel's constexpr S at every step == SplitsFor_v1(live count at that step, 142) -- the live count from the DECLARED
    chronology (arrivals + caps, every request served to its cap) and from the realized one agree at every step; the flips are
    (step 0, live 1, S 32) -> (step 32, live 5, S 16) -> (step 101, live 4, S 32).
"""
from __future__ import annotations

import json
import os

import pytest

from verity.ml.definitions.registry.topp_geometry import SPLIT_MAX_BATCH, SPLITS_FOR_V1_WIDTH, splits_for, splits_for_v1

DATA = os.path.join(os.path.dirname(__file__), "data", "topp_split_fixture")


def _table(num_sms: int, n_max: int = 200) -> list[tuple[int, int, int]]:
    out, start, prev = [], 1, splits_for_v1(1, num_sms)
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
    diff = [n for n in range(1, SPLIT_MAX_BATCH + 1) if splits_for_v1(n, 188) != splits_for_v1(n, 142)]
    assert diff == [5, 9, 10, 11, 18, 19, 20, 21, 22, 23, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    assert {splits_for_v1(n, 188) for n in range(1, SPLIT_MAX_BATCH + 1)} == {2, 4, 8, 16, 32}


def test_lifted_splits_primitives_resolve_at_188_sms():
    from verity.ml.circuits.codec import _lookup_primitive
    from verity.ml.circuits.defs import REGISTRY
    from verity.ml.definitions.registry import lifted as L
    assert _lookup_primitive(REGISTRY, "SplitsForSMS188_v1") is L.splits_for_prim(188)
    assert _lookup_primitive(REGISTRY, "LSplitsForSMS188_v1") is L.l_splits_for(188)
    assert [L.l_splits_for(188).evaluate(n) for n in (1, 5, 6, 11, 12, 23, 24, 47, 48, 64)] == [32, 32, 16, 16, 8, 8, 4, 4, 2, 2]


def test_domain_refuses_by_name():
    with pytest.raises(ValueError, match="n_live=0 outside the domain"):
        splits_for_v1(0, 142)
    with pytest.raises(ValueError, match=f"n_live={SPLIT_MAX_BATCH + 1} outside the domain"):
        splits_for_v1(SPLIT_MAX_BATCH + 1, 142)
    with pytest.raises(ValueError, match="num_SMs=0"):
        splits_for_v1(1, 0)
    assert splits_for_v1(SPLIT_MAX_BATCH, 142) == 2 and splits_for_v1(SPLIT_MAX_BATCH, 132) == 2 and splits_for_v1(1, 1) == 1


def test_range_is_a_power_of_two_value6_monotone():
    for m in (1, 2, 8, 64, 108, 132, 142, 144, 256):
        prev = None
        for n in range(1, SPLIT_MAX_BATCH + 1):
            s = splits_for_v1(n, m)
            assert 1 <= s <= 32 and s & (s - 1) == 0 and s < (1 << SPLITS_FOR_V1_WIDTH)
            assert s * n <= m or s == 1                    # the loop invariant: S * n_live <= num_SMs (S = 1 unconditionally)
            assert s == 32 or 2 * s * n > m                # maximality: the next power of two would not fit
            assert prev is None or s <= prev               # non-increasing in the live count
            prev = s
    for n in (1, 5, 9, 17, 36, 64):
        assert all(splits_for_v1(n, m) <= splits_for_v1(n, m + 1) for m in range(1, 300))


def test_r15_b8_record_kernel_S_equals_splits_for_v1_of_live_at_every_step():
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
        S = splits_for_v1(n_decl, 142)
        assert S == st["S_kernel"], (t, n_decl, S, st["S_kernel"])
        if S != prev:
            flips.append((t, n_decl, S)); prev = S
    assert flips == [(0, 1, 32), (32, 5, 16), (101, 4, 32)]
    assert sum(len(st["requests"]) for st in steps) == 526
