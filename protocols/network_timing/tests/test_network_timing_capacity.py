"""Theorems 1(b) and 3 on small grids, mirroring the Lean package's pinned statements (`NetTiming.Counting`,
`NetTiming.Advice`).

The accepted observation sequences are the grids an accepted run can show. The tests enumerate every sender behaviour
on a small horizon: every hand-time schedule, and every declared clock sync. They count what an accepted run can
carry, and check it against the bounds."""

import itertools
import math
import random

from verity_network_timing import capacity
from verity_network_timing.audit import Audit
from verity_network_timing.grid import GridParams, Warden, fill, occupancy
from verity_network_timing.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids

P = GridParams(buckets=3, allocation=2, frame_bytes=1)
FRAMES = [LogicalFrame(b"a", 0, 0), LogicalFrame(b"b", 0, 1), LogicalFrame(b"c", 1, 0), LogicalFrame(b"d", 3, 0),
          LogicalFrame(b"e", 4, 0)]


def run(hands, windows, frames=FRAMES):
    """A sender that hands the frames at arbitrary (non-decreasing) buckets: the warden's records."""
    w = Warden("responses", P)
    records, k = [], 0
    for t in range(windows * P.buckets):
        while k < len(frames) and hands[k] == t:
            w.enqueue(frames[k].payload)
            k += 1
        w.tick()
        if (t + 1) % P.buckets == 0:
            records.append(w.close_window())
    return records


def schedules(windows):
    horizon = windows * P.buckets
    for hands in itertools.combinations_with_replacement(range(horizon + 1), len(FRAMES)):
        yield hands


def accepted_observations(sync_set, windows):
    """Every grid sequence of an accepted run, over every sender and every declaration."""
    seen = set()
    for hands in schedules(windows):
        records = run(hands, windows)
        for syncs in itertools.product(list(sync_set), repeat=windows):
            audit = Audit()
            audit.check_link("responses", records, FRAMES, syncs, sync_set, P)
            if audit.accepted:
                seen.add(tuple(r.grid for r in records))
    return seen


def test_tree_lemma():
    """encard_le_prod_of_branching"""
    rng = random.Random(5)
    for _ in range(200):
        n, alpha = rng.randint(1, 4), rng.randint(1, 3)
        seqs = {tuple(rng.randrange(alpha) for _ in range(n)) for _ in range(rng.randint(1, 30))}
        bound = 1
        for t in range(n):
            branching = {}
            for s in seqs:
                branching.setdefault(s[:t], set()).add(s[t])
            bound *= max(len(v) for v in branching.values())
        assert len(seqs) <= bound


def test_accepted_count_le_sync_advice():
    """encard_accSeqs_le_sync: accepted sequences ≤ #Sync^W (content fixed, J so D = 1)."""
    sync_set = SyncSet((0, 1), (1, 2))
    windows = 2
    seen = accepted_observations(sync_set, windows)
    assert 1 <= len(seen) <= sync_set.size ** windows
    assert math.log2(len(seen)) <= capacity.accepted_bound_bits(0, sync_set.bits, windows)
    # and never more than the envelope (the block cap with one block)
    assert len(seen) <= capacity.left_packed_patterns(P.buckets, P.allocation) ** windows


def test_fixed_clock_zero_capacity():
    """encard_accSeqs_le_fixedClock: with a fixed clock, every accepted run shows the same grids."""
    sync_set = SyncSet((1,), (1,))
    seen = accepted_observations(sync_set, 2)
    assert len(seen) == 1
    assert capacity.accepted_bound_bits(0, sync_set.bits, 2) == 0


def test_secret_senders_are_rejected_or_silent():
    """Zero error (encard_obsDecodable_le). A sender encodes a secret in its hand times. Under a fixed clock, every
    accepted secret shows the Program's grids, so a decoder can tell at most one apart; every other secret is rejected."""
    sync = ClockSync(1, 1)
    sync_set = SyncSet((1,), (1,))
    expected = tuple(expected_grids(FRAMES, [sync, sync], P))
    accepted, rejected = set(), 0
    for hands in schedules(2):
        records = run(hands, 2)
        audit = Audit()
        audit.check_link("responses", records, FRAMES, [sync, sync], sync_set, P)
        if audit.accepted:
            accepted.add(tuple(r.grid for r in records))
        else:
            rejected += 1
    assert accepted == {expected} and rejected > 0


def test_free_occupancy_advice_reaches_the_envelope():
    """Without a clock, if the Program took the occupancy itself as timing advice, accepted grids would number
    (r + 1)^T per window: the envelope, T·log2(r + 1) bits."""
    payloads = [bytes([k]) for k in range(P.buckets * P.allocation)]
    grids = {fill(payloads, counts, P.allocation) for counts in itertools.product(range(P.allocation + 1),
                                                                                 repeat=P.buckets)}
    assert len(grids) == capacity.left_packed_patterns(P.buckets, P.allocation)
    assert math.isclose(math.log2(len(grids)), capacity.envelope_bits(P.buckets, P.allocation))
    assert math.isclose(capacity.envelope_bits(600, 1024), 6000.844916635686)


def test_reacting_operator_does_not_add_capacity():
    """The operator sets the second window's request anchor from the occupancy it saw in the first window's last bucket,
    so feedback reaches the Program through ingress. Accepted sequences still number at most #Sync^W (the joint
    witness's shape)."""
    sync_set = SyncSet((0, 1), (1,))
    head = FRAMES[:3]
    seen = set()
    for hands in itertools.combinations_with_replacement(range(2 * P.buckets + 1), len(head) + 1):
        records = run(hands, 2, head + [LogicalFrame(b"z", 0, 0)])
        anchor = P.buckets + occupancy(records[0].grid)[-1]  # the operator reacts to what it observed
        frames = head + [LogicalFrame(b"z", anchor, 0)]
        for syncs in itertools.product(list(sync_set), repeat=2):
            audit = Audit()
            audit.check_link("responses", records, frames, syncs, sync_set, P)
            if audit.accepted:
                seen.add(tuple(r.grid for r in records))
    assert 1 <= len(seen) <= sync_set.size ** 2


def test_block_bound_random_programs():
    """encard_accSeqs_le_blocks. The occupancy of cell (link, bucket) reads only its own block's timing advice and the
    history, and the frames read only content advice and the history: at most 2^A_c · ∏ min(2^A_β, #P_β) sequences."""
    rng = random.Random(6)
    T, links = 2, 2
    for _ in range(1500):
        a_c, adv = rng.randint(0, 2), [rng.randint(0, 3) for _ in range(links)]
        occ_t = [[{} for _ in range(T)] for _ in range(links)]
        cont_t = [[{} for _ in range(T)] for _ in range(links)]

        def look(table, key, hi):
            if key not in table:
                table[key] = rng.randint(0, hi)
            return table[key]

        seqs = set()
        for ac in range(2 ** a_c):
            for a in itertools.product(*[range(2 ** x) for x in adv]):
                hist = ()
                for t in range(T):
                    row = []
                    for ell in range(links):
                        o = look(occ_t[ell][t], (a[ell], ac, hist), 1)
                        row.append(look(cont_t[ell][t], (ac, hist), 3) if o else None)
                    hist += (tuple(row),)
                seqs.add(hist)
        bound = 2 ** a_c
        for x in adv:
            bound *= min(2 ** x, 2 ** T)
        assert len(seqs) <= bound


def test_doc_numbers():
    s = SyncSet(tuple(range(16)), tuple(range(1, 9)))
    assert s.size == 128 and s.bits == 7
    assert math.isclose(capacity.ingress_rate(64), 60.2, rel_tol=1e-3)
    assert math.isclose(capacity.ingress_rate(64, 1000), 6.02, rel_tol=1e-3)
    two_links_hour = (capacity.ingress_rate(64) + capacity.ingress_rate(1024)) * 3600
    assert math.isclose(capacity.k_share(two_links_hour), 0.0063, rel_tol=0.01)
