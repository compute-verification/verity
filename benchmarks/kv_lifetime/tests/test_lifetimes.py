from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lifetimes as lt  # noqa: E402
from lifetimes import ALLOC, CACHE, FREE, HIT, RESET  # noqa: E402


class Block:
    def __init__(self, i: int) -> None:
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = i, 0, None, False


class Pool:
    """vLLM's BlockPool's reference counting, without the free queue's order."""

    def __init__(self, n: int) -> None:
        self.blocks = [Block(i) for i in range(n)]
        self.free = list(self.blocks)

    def get_new_blocks(self, n):
        out, self.free = self.free[:n], self.free[n:]
        for b in out:
            b.block_hash = None
            b.ref_cnt += 1
        return out

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0:
                self.free.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free.append(b)

    def reset_prefix_cache(self):
        for b in self.blocks:
            if b.ref_cnt == 0:
                b.block_hash = None
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_recorder_logs_allocations_hits_shares_and_releases():
    pool, clock = Pool(4), Clock()
    rec = lt.Recorder(clock)
    undo = rec.install(pool)
    a, b = pool.get_new_blocks(2)
    a.block_hash = "h"
    clock.now = 1.0
    pool.touch([a])  # shared with a second request
    clock.now = 2.0
    pool.free_blocks([a, b])  # a is still held by the second request
    clock.now = 3.0
    pool.free_blocks([a])
    clock.now = 4.0
    pool.touch([a])  # a prefix hit on the cached block
    clock.now = 5.0
    pool.reset_prefix_cache()
    ev = rec.arrays()
    assert ev["kind"].tolist() == [ALLOC, ALLOC, lt.SHARE, FREE, CACHE, HIT, RESET]
    assert ev["block"].tolist() == [0, 1, 0, 1, 0, 0, -1]
    assert ev["t"].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    undo()
    assert "get_new_blocks" not in vars(pool)


def events(rows):
    t, k, b = zip(*rows)
    return np.array(t, float), np.array(k, np.int8), np.array(b, np.int32)


def test_epochs_split_into_held_segments_and_cached_spans():
    # block 0 unhashed: held [0, 10). block 1 hashed: held [0, 5), cached, hit at 20, held to 30, evicted at 50
    t, k, b = events([(0, ALLOC, 0), (0, ALLOC, 1), (5, CACHE, 1), (10, FREE, 0), (20, HIT, 1), (30, CACHE, 1),
                      (50, ALLOC, 1)])
    ep = lt.parse(t, k, b, t_end=60.0)
    assert ep["a"].tolist() == [0.0, 0.0, 50.0]  # block 0's epoch, then block 1's two
    assert sorted(zip(ep["a"], ep["held_end"], ep["res_end"], ep["censored"])) == [
        (0.0, 10.0, 10.0, False), (0.0, 30.0, 50.0, False), (50.0, 60.0, 60.0, True)]
    assert sorted(zip(ep["seg_start"], ep["seg_end"])) == [(0, 5), (0, 10), (20, 30), (50, 60)]
    assert sorted(zip(ep["cached_start"], ep["cached_end"])) == [(5, 20), (30, 50)]
    assert int(ep["hits"].sum()) == 1


def test_a_reset_ends_every_cached_span():
    t, k, b = events([(0, ALLOC, 0), (2, CACHE, 0), (7, RESET, -1), (9, ALLOC, 0), (12, FREE, 0)])
    ep = lt.parse(t, k, b, t_end=20.0)
    assert sorted(zip(ep["a"], ep["res_end"])) == [(0, 7), (9, 12)]
    assert sorted(zip(ep["cached_start"], ep["cached_end"])) == [(2, 7)]


def test_a_released_block_cannot_be_hit():
    t, k, b = events([(0, ALLOC, 0), (2, FREE, 0), (3, HIT, 0)])
    with pytest.raises(ValueError, match="hit"):
        lt.parse(t, k, b, t_end=5.0)


def test_rekey_counts_the_caps_a_resident_block_outlives():
    # appended at 3 in window 0 of T = 10: caps at 10, 20 (L = 1) and evicted at 25
    t, k, b = events([(3, ALLOC, 0), (25, FREE, 0)])
    ep = lt.parse(t, k, b, t_end=30.0)
    assert lt.policy(ep, 10.0, 1, 0.0, "rekey")["rekeys"].tolist() == [2]
    assert lt.policy(ep, 10.0, 3, 0.0, "rekey")["rekeys"].tolist() == [0]
    ex = lt.policy(ep, 10.0, 1, 0.0, "expire")
    assert ex["rekeys"].tolist() == [0] and ex["cut"].tolist() == [10.0]
    # a phase moves the windows: windows start at 5 + 10 j, so the block is in window -1 and its caps are at 5 and 15
    # (the one at 25 is its eviction)
    assert lt.policy(ep, 10.0, 1, 5.0, "rekey")["rekeys"].tolist() == [2]


def test_the_recorder_carries_a_requests_origin_through_its_prefix_hits():
    pool, clock = Pool(4), Clock()
    rec = lt.Recorder(clock)
    rec.install(pool)
    rec.allocating(0.0)
    (a,) = pool.get_new_blocks(1)
    a.block_hash = "h"
    pool.free_blocks([a])
    clock.now = 10.0
    rec.allocating(10.0)
    pool.touch([a])  # a later turn hits the cached history and inherits its origin
    pool.get_new_blocks(1)
    assert rec.origin == 0.0
    rec.allocating(None)
    clock.now = 20.0
    pool.get_new_blocks(1)
    ev = rec.arrays()
    assert ev["kind"].tolist() == [ALLOC, CACHE, HIT, ALLOC, ALLOC]
    assert ev["origin"].tolist() == [0.0, 0.0, 0.0, 0.0, 20.0]


def test_the_origin_clock_caps_a_block_from_its_contents_origin():
    # appended at 25 from content whose origin is 3, held to 45; T = 10, L = 1
    t, k, b = events([(25, ALLOC, 0), (45, FREE, 0)])
    ep = lt.parse(t, k, b, t_end=50.0, origin=np.array([3.0, 3.0]))
    assert ep["origin"].tolist() == [3.0]
    assert lt.policy(ep, 10.0, 1, 0.0, "expire")["cut"].tolist() == [30.0]
    assert lt.policy(ep, 10.0, 1, 0.0, "expire", "origin")["cut"].tolist() == [25.0]  # past life at its append
    assert lt.policy(ep, 10.0, 1, 0.0, "rekey", "origin")["rekeys"].tolist() == [2]  # recomputed at 30 and 40
    assert lt.policy(ep, 10.0, 1, 0.0, "held", "origin")["rekeys"].tolist() == [2]


def test_held_releases_a_cached_block_at_its_first_idle_cap():
    t, k, b = events([(0, ALLOC, 0), (5, CACHE, 0), (20, HIT, 0), (30, CACHE, 0), (50, ALLOC, 0), (52, FREE, 0)])
    ep = lt.parse(t, k, b, t_end=60.0)
    p = lt.policy(ep, 10.0, 1, 0.0, "held")
    first = int(np.argmin(ep["a"]))
    assert p["rekeys"][first] == 0
    cut = dict(zip(zip(ep["seg_start"], ep["seg_end"]), p["cut"]))
    assert cut[(0.0, 5.0)] == 5.0 and cut[(20.0, 30.0)] == 20.0  # released at 10, so the hit at 20 is uncredited
    # at L = 2 the cap at 20 finds it held (the hit opens a segment at 20): re-keyed, then released at 40
    p = lt.policy(ep, 10.0, 2, 0.0, "held")
    cut = dict(zip(zip(ep["seg_start"], ep["seg_end"]), p["cut"]))
    assert p["rekeys"][first] == 1 and cut[(20.0, 30.0)] == 30.0
    assert lt.policy(ep, 10.0, 2, 0.0, "rekey")["rekeys"][first] == 2  # resident through caps 20 and 40


def test_a_stock_takes_its_bytes_off_every_step():
    steps = lt.steps_arrays([[i, i + 1, 1, 1, 1, 1, 1, 0, 4, 0, 0] for i in range(10)])
    s = np.zeros(4)
    e = np.full(4, 10.0)
    r = lt.step_credit(steps, s, e, s, e, 100.0, [0.0, 100.0, 1000.0])
    assert r["held_bytes_mean"] == 400.0
    assert r["by_stock"] == {"0.0": 1.0, "100.0": 0.75, "1000.0": 0.0}


def test_analyze_reports_lifetimes_occupancy_and_the_frontier():
    rows, rng = [], np.random.default_rng(1)
    for i in range(200):
        a = i * 0.5
        rows += [(a, ALLOC, i), (a + float(rng.choice([1.0, 3.0, 90.0])), FREE, i)]
    rows.sort()
    t, k, b = events(rows)
    steps = lt.steps_arrays([[x, x + 0.5, 1, 1, 1, 1, 1, 0, 1, 0, 0] for x in np.arange(0, 200, 0.5)])
    r = lt.analyze({"t": t, "kind": k, "block": b}, steps, 200.0, 2.0, windows=(10.0,), caps_L=(1, 6), phases=2,
                   taus=(0.0, 5.0), stocks=[0.0])
    assert r["epochs"] == 200
    life = r["held_life_s"]["share_longer_than"]
    assert 0.2 < life["60"] < 0.45 and life["0.5"] == 1.0
    assert r["stock_age_tail"]["60"] > 0.2  # the 90 s blocks hold most of the stock
    fr = r["frontier"]["T=10,L=1"]
    assert fr["rekey"]["credited_before_stock"] == pytest.approx(1.0)
    assert fr["expire"]["credited_before_stock"] < 0.4
    assert fr["rekey"]["rekey_bytes_per_s"] > r["frontier"]["T=10,L=6"]["rekey"]["rekey_bytes_per_s"]
    assert r["delayed"]["5.0"]["append_bytes_per_s"] < r["delayed"]["0.0"]["append_bytes_per_s"]
    assert r["frontier_origin"] == {}  # no origins logged: the block clock only
    o = lt.analyze({"t": t, "kind": k, "block": b, "origin": t - 30.0}, steps, 200.0, 2.0, windows=(10.0,),
                   caps_L=(1,), phases=2, taus=(), stocks=[0.0])
    assert o["frontier"]["T=10,L=1"] == fr
    assert o["frontier_origin"]["T=10,L=1"]["expire"]["credited_before_stock"] == 0.0
    assert o["stock_origin_age_tail"]["20"] == 1.0
