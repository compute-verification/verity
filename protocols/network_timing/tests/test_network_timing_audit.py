"""The verifier's exact check with record-and-reject, and an honest shaper that meets the logical schedule."""

from verity_network_timing.audit import Audit
from verity_network_timing.grid import GridParams, Reason, Warden
from verity_network_timing.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, run_shaper

P = GridParams(buckets=4, allocation=2, frame_bytes=1)
SYNC_SET = SyncSet((0, 1, 2), (1, 2))
FRAMES = [LogicalFrame(bytes([k]), anchor, index) for k, (anchor, index) in
          enumerate([(0, 0), (0, 1), (0, 2), (5, 0), (5, 1), (9, 0), (9, 1), (9, 2)])]
SYNCS = [ClockSync(1, 2), ClockSync(1, 1), ClockSync(1, 2)]


def honest_ready():
    """Every frame ready one bucket after its anchor, well before its logical bucket."""
    return [f.anchor + 1 for f in FRAMES]


def test_honest_run_is_accepted():
    records = run_shaper(FRAMES, honest_ready(), SYNCS, Warden("responses", P))
    assert [r.grid for r in records] == expected_grids(FRAMES, SYNCS, P)
    audit = Audit()
    audit.check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)
    assert audit.accepted and not audit.violations


def test_record_and_reject_without_halting():
    """A late frame in window 1 is a violation. It is recorded, the run is rejected, and window 2 is still checked."""
    ready = honest_ready()
    ready[3] = 7  # frame 3's logical bucket is 5 + 1 = 6
    records = run_shaper(FRAMES, ready, SYNCS, Warden("responses", P))
    audit = Audit()
    verdicts = audit.check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)
    assert [v.violation for v in verdicts] == [None, "mismatch", None]
    assert not audit.accepted and len(audit.violations) == 1


def test_exact_slot_equality():
    records = run_shaper(FRAMES, honest_ready(), SYNCS, Warden("responses", P))
    grid = [list(row) for row in records[0].grid]
    grid[0][0] = b"\xff"
    tampered = records[0].__class__(records[0].link, 0, tuple(tuple(r) for r in grid), records[0].status)
    audit = Audit()
    assert audit.check_link("responses", [tampered] + records[1:], FRAMES, SYNCS, SYNC_SET, P)[0].violation == "mismatch"


def test_status_advice_and_missing_record_are_violations():
    w = Warden("responses", P)
    records = run_shaper(FRAMES, honest_ready(), SYNCS[:1], w)
    w.fault(Reason.MONITOR_UNAVAILABLE)
    for _ in range(P.buckets):
        w.tick()
    records.append(w.close_window())
    audit = Audit()
    verdicts = audit.check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)
    assert verdicts[1].violation == "status:monitor-unavailable"
    assert verdicts[2].violation == "no-verdict"
    bad = Audit().check_link("responses", records, FRAMES, [ClockSync(7, 1)] + SYNCS[1:], SYNC_SET, P)
    assert bad[0].violation == "advice"
