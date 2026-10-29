"""Theorems 1(b) and 3 on small grids. By name, the tests mirror eight of their statements (`NetTiming.SecurityProofs`,
proved in its `Counting`, `Advice` and `Schedule`): `SecurityProofs.EncardLeProdOfBranching`,
`SecurityProofs.EncardObsDecodableLe`, `SecurityProofs.EncardAccSeqsLeSync`, `SecurityProofs.EncardAccSeqsLeFixedClock`,
and for the constant-rate Program `SecurityProofs.EncardAccSeqsLeConstantRate`,
`SecurityProofs.EncardAccSeqsLeConstantRateFixedClock`, `SecurityProofs.EncardAccSeqsLeConstantRateWindows` and
`SecurityProofs.ExpectedRowCausal`.

The accepted observation sequences are the grids an accepted run can show. The tests enumerate every sender behaviour
on a small horizon: every hand-time schedule, and every declared clock sync. They count what an accepted run can
carry, and check it against the bounds."""

import itertools
import math
import random
from fractions import Fraction
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import capacity
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Certifier, fill, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, eligible, expected_grids

P = GridParams(buckets=3, allocation=2, frame_bytes=1)
FRAMES = [LogicalFrame(b"a", 0, 0), LogicalFrame(b"b", 0, 1), LogicalFrame(b"c", 1, 0), LogicalFrame(b"d", 3, 0),
          LogicalFrame(b"e", 4, 0)]


def run(hands, windows, frames=FRAMES):
    """A sender that hands the frames at arbitrary (non-decreasing) buckets: the network certificates."""
    w = Certifier("responses", P)
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
    """SecurityProofs.EncardLeProdOfBranching"""
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
    """SecurityProofs.EncardAccSeqsLeSync, for the constant-rate Program SecurityProofs.EncardAccSeqsLeConstantRate:
    accepted sequences ≤ #Sync^W (content fixed, J so D = 1)."""
    sync_set = SyncSet((0, 1), (1, 2))
    windows = 2
    seen = accepted_observations(sync_set, windows)
    assert 1 <= len(seen) <= sync_set.size ** windows
    assert math.log2(len(seen)) <= capacity.accepted_bound_bits(0, sync_set.bits, windows)
    # and never more than the envelope (the block cap with one block)
    assert len(seen) <= capacity.left_packed_patterns(P.buckets, P.allocation) ** windows


def test_fractional_rate_count_le_sync_advice():
    """The same bound when ρ is below one frame per bucket: `eligible` spaces a session's frames ⌊λ/ρ⌋ apart."""
    assert [eligible(LogicalFrame(b"", 5, i), ClockSync(1, Fraction(1, 3))) for i in range(4)] == [6, 9, 12, 15]
    assert [eligible(LogicalFrame(b"", 0, i), ClockSync(0, Fraction(2, 3))) for i in range(4)] == [0, 1, 3, 4]
    sync_set = SyncSet((0, 1), (Fraction(1, 2), 1))
    seen = accepted_observations(sync_set, 2)
    assert 1 <= len(seen) <= sync_set.size ** 2


def test_accepted_count_le_sync_per_window_opened():
    """SecurityProofs.EncardAccSeqsLeConstantRateWindows: runs accepted on their first m windows show at most
    #Σ_sync^m grid prefixes, however many windows the advice covers."""
    sync_set = SyncSet((0, 1), (1, 2))
    windows = 2
    for m in range(windows + 1):
        seen = set()
        for hands in schedules(windows):
            records = run(hands, windows)
            for syncs in itertools.product(list(sync_set), repeat=windows):
                verdicts = Audit().check_link("responses", records, FRAMES, syncs, sync_set, P)
                if all(v.accepted for v in verdicts[:m]):
                    seen.add(tuple(r.grid for r in records[:m]))
        assert 1 <= len(seen) <= sync_set.size**m


def test_window_grid_reads_only_earlier_syncs():
    """SecurityProofs.ExpectedRowCausal: window w's grid is the same whatever later windows declare, in Σ_sync or not,
    with fractional rates and frames anchored past the horizon."""
    rng = random.Random(11)
    rhos = (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(2, 3))
    for _ in range(300):
        params = GridParams(buckets=rng.randint(1, 3), allocation=rng.randint(1, 2), frame_bytes=1)
        windows = rng.randint(1, 3)
        frames = [LogicalFrame(bytes([k]), rng.randrange(params.buckets * (windows + 1)), rng.randrange(4))
                  for k in range(rng.randint(0, 6))]
        syncs = [ClockSync(rng.randrange(4), rng.choice(rhos)) for _ in range(windows)]
        grids = expected_grids(frames, syncs, params)
        for w in range(windows):
            later = [ClockSync(rng.randrange(4), rng.choice(rhos)) for _ in range(windows - w - 1)]
            assert expected_grids(frames, syncs[: w + 1] + later, params)[w] == grids[w]


def test_a_negative_logical_index_is_rejected():
    """Lean's index is a ℕ, and SecurityProofs.ExpectedRowCausal needs it. A frame anchored in window 1 with λ = −2 is
    eligible at T + δ − 2 at ρ = 1, so window 0's grid would read window 1's sync: here it holds the frame under δ = 0
    and not under δ = 2. `LogicalFrame` rejects that index."""
    with pytest.raises(ValueError, match="logical index is non-negative"):
        LogicalFrame(b"x", P.buckets, -2)
    unchecked = SimpleNamespace(payload=b"x", anchor=P.buckets, index=-2)
    window0 = {tuple(expected_grids([unchecked], [ClockSync(0, 1), later], P)[0])
               for later in (ClockSync(0, 1), ClockSync(2, 1))}
    assert len(window0) == 2
    assert LogicalFrame(b"x", P.buckets, 0).index == 0


def test_fixed_clock_zero_capacity():
    """SecurityProofs.EncardAccSeqsLeFixedClock, for the constant-rate Program
    SecurityProofs.EncardAccSeqsLeConstantRateFixedClock: with a fixed clock, every accepted run shows the same grids."""
    sync_set = SyncSet((1,), (1,))
    seen = accepted_observations(sync_set, 2)
    assert len(seen) == 1
    assert capacity.accepted_bound_bits(0, sync_set.bits, 2) == 0


def test_secret_senders_are_rejected_or_silent():
    """Zero error (SecurityProofs.EncardObsDecodableLe). A sender encodes a secret in its hand times. Under a fixed
    clock, every accepted secret shows the Program's grids, so a decoder can tell at most one apart; every other secret
    is rejected."""
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
    so feedback reaches the Program through ingress. Accepted sequences still number at most #Sync^W."""
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


def test_sync_set_lists_each_value_once():
    """A repeated δ or ρ (1 and 2/2 are one ρ) is rejected, so iterating Σ_sync yields exactly #Σ_sync values."""
    with pytest.raises(ValueError, match="each delta and each rho once"):
        SyncSet((0, 1, 1), (1,))
    with pytest.raises(ValueError, match="each delta and each rho once"):
        SyncSet((0,), (1, Fraction(2, 2)))
    s = SyncSet((0, 1), (1, Fraction(1, 2)))
    assert len(set(s)) == len(list(s)) == s.size == 4


def test_doc_numbers():
    s = SyncSet(tuple(range(16)), tuple(range(1, 9)))
    assert s.size == 128 and s.bits == 7
    assert math.isclose(capacity.ingress_rate(64), 60.2, rel_tol=1e-3)
    assert math.isclose(capacity.ingress_rate(64, 1000), 6.02, rel_tol=1e-3)
    two_links_hour = (capacity.ingress_rate(64) + capacity.ingress_rate(1024)) * 3600
    assert math.isclose(capacity.k_share(two_links_hour, 9.12e7), 0.0063, rel_tol=0.01)
