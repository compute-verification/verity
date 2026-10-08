"""Network certificates committed by hash, with padding elided: injective on left-packed grids, and the audit on committed
records gives the same verdicts as on raw ones."""

import itertools

import pytest

from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.commitment import commit, record_digest, row_digest
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Reason, Record, Status, Certifier, fill
from verity.core.protocols.certifier.schedule import run_shaper

from test_network_certifier_audit import FRAMES, P, SYNC_SET, SYNCS, honest_ready


def test_elided_digest_is_injective_on_left_packed_grids():
    """Every left-packed grid on T = 2, r = 2 over a 2-frame alphabet has its own digest (as
    `SecurityProofs.FillInjective`)."""
    params = GridParams(buckets=2, allocation=2, frame_bytes=1)
    seen = {}
    for counts in itertools.product(range(3), repeat=2):
        for frames in itertools.product([b"a", b"b"], repeat=sum(counts)):
            grid = fill(list(frames), counts, 2)
            d = record_digest("l", 0, grid, COMPLETE, params)
            assert seen.setdefault(d, grid) == grid
    assert len(seen) == sum(2 ** (a + b) for a, b in itertools.product(range(3), repeat=2))


def test_padding_is_elided_and_packing_enforced():
    params = GridParams(buckets=1, allocation=4, frame_bytes=1)
    assert row_digest((b"a", None, None, None), params) != row_digest((b"a", b"a", None, None), params)
    with pytest.raises(ValueError):
        row_digest((None, b"a", None, None), params)
    with pytest.raises(ValueError):
        row_digest((b"aa", None, None, None), params)


def test_digest_binds_link_window_status_and_params():
    grid = ((b"x", None),)
    params = GridParams(buckets=1, allocation=2, frame_bytes=1)
    base = record_digest("l", 0, grid, COMPLETE, params)
    assert len(bytes.fromhex(base)) == 32
    assert base != record_digest("m", 0, grid, COMPLETE, params)
    assert base != record_digest("l", 1, grid, COMPLETE, params)
    assert base != record_digest("l", 0, grid, Status(Reason.OVERFLOW), params)
    assert base != record_digest("l", 0, grid, COMPLETE, GridParams(buckets=1, allocation=2, frame_bytes=1, bucket_ms=50))


def verdicts(records):
    return [v.violation for v in Audit().check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)]


def test_committed_audit_matches_raw_audit():
    """Honest, late, tampered, faulted and duplicated runs get the same verdicts committed as raw."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    late_ready = honest_ready()
    late_ready[3] = 7
    late = run_shaper(FRAMES, late_ready, SYNCS, Certifier("responses", P))
    rows = [list(row) for row in honest[0].grid]
    rows[0][0] = b"\xff"
    tampered = [Record("responses", 0, tuple(tuple(r) for r in rows), COMPLETE)] + honest[1:]
    w = Certifier("responses", P)
    faulted = run_shaper(FRAMES, honest_ready(), SYNCS[:1], w)
    w.fault(Reason.DROPPED)
    for _ in range(P.buckets):
        w.tick()
    faulted.append(w.close_window())
    duplicated = tampered[:1] + honest
    for records in (honest, late, tampered, faulted, duplicated):
        assert verdicts([commit(r, P) for r in records]) == verdicts(records)
    assert verdicts([commit(r, P) for r in honest]) == [None, None, None]
    assert verdicts([commit(r, P) for r in late])[1] == "mismatch"
    assert verdicts([commit(r, P) for r in duplicated])[0] == "duplicate"


def test_a_status_the_digest_disagrees_with_is_a_mismatch():
    """A certifier that faulted but publishes a complete status in the clear doesn't match the Program's digest."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    faulted_digest = record_digest("responses", 0, honest[0].grid, Status(Reason.DROPPED), P)
    lying = [commit(honest[0], P).__class__("responses", 0, faulted_digest, COMPLETE)] + [commit(r, P) for r in honest[1:]]
    assert verdicts(lying)[0] == "mismatch"
