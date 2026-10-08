"""Proposition 2 and the certifier's semantics. By name, the tests mirror Proposition 2's ten statements
(`NetTiming.SecurityProofs.CardFree` and the rest, proved in `NetTiming.SecurityProofs.Occupancy`) and that module's
three proved lemmas without a record, `occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid`. The certifier tests
check the behaviour `NetTiming.Assumptions` takes as hypotheses (W4 packing, W5's untimed status, W6 fail-closed)."""

import itertools
import math
import random

import pytest

from verity.core.protocols.certifier import capacity
from verity.core.protocols.certifier.grid import (
    COMPLETE, GridParams, Reason, Status, Certifier, fill, left_packed, occupancy, released, row_left_packed,
)

SMALL = [(1, 1), (2, 1), (2, 2), (3, 2), (2, 3)]


def frames(n):
    return [bytes([k % 256, k // 256]) for k in range(n)]


def as_row(bits):
    return tuple(b"x" if b else None for b in bits)


@pytest.mark.parametrize("T,r", SMALL)
def test_card_free(T, r):
    patterns = list(itertools.product(itertools.product([False, True], repeat=r), repeat=T))
    assert len(patterns) == capacity.free_patterns(T, r)


@pytest.mark.parametrize("T,r", SMALL)
def test_card_leftPacked(T, r):
    rows = [bits for bits in itertools.product([False, True], repeat=r) if row_left_packed(as_row(bits))]
    assert len(rows) == r + 1
    assert len(rows) ** T == capacity.left_packed_patterns(T, r)


@pytest.mark.parametrize("T,r", SMALL)
def test_occ_fill_leftPacked_and_injective(T, r):
    seen = set()
    for counts in itertools.product(range(r + 1), repeat=T):
        g = fill(frames(T * r), counts, r)
        assert occupancy(g) == counts  # occ_fill
        assert left_packed(g)  # leftPacked_occ_fill
        real = [s for row in g for s in row if s is not None]
        assert real == frames(sum(counts))  # the k-th frame in the k-th real slot
        seen.add(g)
    assert len(seen) == (r + 1) ** T  # SecurityProofs.FillInjective


def test_exists_fill_of_leftPacked():
    rng = random.Random(1)
    for _ in range(200):
        T, r = rng.randint(1, 4), rng.randint(1, 4)
        counts = [rng.randint(0, r) for _ in range(T)]
        g = tuple(tuple(bytes([rng.randrange(256)]) if s < n else None for s in range(r)) for n in counts)
        assert left_packed(g)
        real = [s for row in g for s in row if s is not None]
        assert g == fill(real, occupancy(g), r)


@pytest.mark.parametrize("T,r", SMALL)
def test_card_counts(T, r):
    vectors = list(itertools.product(range(r + 1), repeat=T))
    for M in range(T * r + 1):
        assert sum(sum(n) == M for n in vectors) <= capacity.all_released_patterns(T, M)  # SecurityProofs.CardCountsEqLe
        assert sum(sum(n) <= M for n in vectors) <= capacity.available_patterns(T, M)  # SecurityProofs.CardCountsLeLe


def test_released_self():
    rng = random.Random(2)
    for _ in range(200):
        r = rng.randint(1, 5)
        n = [rng.randint(0, r) for _ in range(rng.randint(1, 8))]
        assert released(r, n) == n


def test_sum_released_le_and_conservation():
    rng = random.Random(3)
    for _ in range(500):
        r = rng.randint(1, 4)
        enq = [rng.randint(0, 6) for _ in range(rng.randint(1, 10))]
        out = released(r, enq)
        assert sum(out) <= sum(enq)  # SecurityProofs.SumReleasedLe
        assert all(0 <= x <= r for x in out)


@pytest.mark.parametrize("T,r", SMALL)
def test_encard_fifoGrids(T, r):
    """Tightness: over all enqueue schedules, the FIFO certifier releases exactly (r + 1)^T grids."""
    grids = {fill(frames(T * (r + 1)), released(r, enq), r)
             for enq in itertools.product(range(r + 2), repeat=T)}
    assert all(left_packed(g) for g in grids)  # leftPacked_fifoGrid
    assert len(grids) == capacity.left_packed_patterns(T, r)


@pytest.mark.parametrize("T,r", SMALL)
def test_encard_fifoGrids_le_of_available(T, r):
    for M in range(T * r + 2):
        grids = {fill(frames(M), released(r, enq), r)
                 for enq in itertools.product(range(M + 1), repeat=T) if sum(enq) <= M}
        assert len(grids) <= capacity.available_patterns(T, M)


def test_certifier_constant_rate_left_packed_fifo():
    rng = random.Random(4)
    p = GridParams(buckets=5, allocation=3, frame_bytes=2)
    w = Certifier("responses", p)
    sent, out = [], []
    for _ in range(3):
        for _ in range(p.buckets):
            for _ in range(rng.randint(0, 5)):
                f = bytes([rng.randrange(256), rng.randrange(256)])
                sent.append(f)
                w.enqueue(f)
            row = w.tick()
            assert len(row) == p.allocation and row_left_packed(row)
        rec = w.close_window()
        assert rec.status == COMPLETE and left_packed(rec.grid)
        out += [s for row in rec.grid for s in row if s is not None]
    assert out and out == sent[: len(out)]  # queue order, across windows


def test_certifier_fail_closed_untimed_status():
    p = GridParams(buckets=4, allocation=2, frame_bytes=1, queue_capacity=3)
    statuses = set()
    for fault_at in range(p.buckets):
        w = Certifier("responses", p)
        for t in range(p.buckets):
            n = 5 if t == fault_at else 1  # 5 frames overflow Q = 3
            for _ in range(n):
                w.enqueue(b"a")
            row = w.tick()
            if t >= fault_at:
                assert row == (None, None)  # the rest of the window is padding
        rec = w.close_window()
        statuses.add(rec.status)
        # the next window recovers
        w.enqueue(b"b")
        w.tick()
        for _ in range(p.buckets - 1):
            w.tick()
        assert w.close_window().status == COMPLETE
    assert statuses == {Status(Reason.OVERFLOW)}  # one untimed symbol, whatever the fault's bucket


def test_certifier_malformed_frame_is_a_fault():
    w = Certifier("responses", GridParams(buckets=2, allocation=1, frame_bytes=4))
    w.enqueue(b"abc")
    w.tick()
    w.tick()
    assert w.close_window().status == Status(Reason.MALFORMED)


def test_unbound_bits_is_about_8_T_r_B():
    assert math.isclose(capacity.unbound_bits(6, 2, 8 * 2), 8 * 6 * 2 * 2, rel_tol=1e-3)
