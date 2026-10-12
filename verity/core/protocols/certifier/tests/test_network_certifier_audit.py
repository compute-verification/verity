"""The verifier's exact check with record-and-reject, an honest shaper that meets the logical schedule, one clock per
spatial unit, and the ingress links' status check."""

from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Reason, Record, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, run_shaper

P = GridParams(buckets=4, allocation=2, frame_bytes=1)
SYNC_SET = SyncSet((0, 1, 2), (1, 2))
FRAMES = [LogicalFrame(bytes([k]), anchor, index) for k, (anchor, index) in
          enumerate([(0, 0), (0, 1), (0, 2), (5, 0), (5, 1), (9, 0), (9, 1), (9, 2)])]
SYNCS = [ClockSync(1, 2), ClockSync(1, 1), ClockSync(1, 2)]


def honest_ready():
    """Every frame ready one bucket after its anchor, well before its logical bucket."""
    return [f.anchor + 1 for f in FRAMES]


def test_honest_run_is_accepted():
    records = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    assert [r.grid for r in records] == expected_grids(FRAMES, SYNCS, P)
    audit = Audit()
    audit.check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)
    assert audit.accepted and not audit.violations


def test_record_and_reject_without_halting():
    """A late frame in window 1 is a violation. It is recorded, the run is rejected, and window 2 is still checked."""
    ready = honest_ready()
    ready[3] = 7  # frame 3's logical bucket is 5 + 1 = 6
    records = run_shaper(FRAMES, ready, SYNCS, Certifier("responses", P))
    audit = Audit()
    verdicts = audit.check_link("responses", records, FRAMES, SYNCS, SYNC_SET, P)
    assert [v.violation for v in verdicts] == [None, "mismatch", None]
    assert not audit.accepted and len(audit.violations) == 1


def test_exact_slot_equality():
    records = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    grid = [list(row) for row in records[0].grid]
    grid[0][0] = b"\xff"
    tampered = records[0].__class__(records[0].link, 0, tuple(tuple(r) for r in grid), records[0].status)
    audit = Audit()
    assert audit.check_link("responses", [tampered] + records[1:], FRAMES, SYNCS, SYNC_SET, P)[0].violation == "mismatch"


def test_status_advice_and_missing_record_are_violations():
    w = Certifier("responses", P)
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


def test_more_than_one_record_for_a_window_is_a_violation():
    """A second record for window 1 makes it a duplicate, whichever comes first and even if the two agree. Another
    link's record for the same window is not a duplicate."""
    records = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    bad = Record("responses", 1, records[0].grid, records[1].status)
    for recs in ([bad] + records, records + [bad], records + [records[1]]):
        verdicts = Audit().check_link("responses", recs, FRAMES, SYNCS, SYNC_SET, P)
        assert [v.violation for v in verdicts] == [None, "duplicate", None]
    other = Record("requests", 1, records[1].grid, records[1].status)
    assert Audit().check_link("responses", records + [other], FRAMES, SYNCS, SYNC_SET, P)[1].accepted


def test_an_out_of_set_sync_flags_only_its_own_window():
    """Window 1 declares δ = 3, outside the set. Only window 1 is an advice violation. Window 0's grid doesn't read
    window 1's sync, and window 2 is checked against the grid the declared syncs give it, so tampering with it is a
    mismatch."""
    declared = [SYNCS[0], ClockSync(3, 1), SYNCS[2]]
    assert declared[1] not in SYNC_SET
    expected = expected_grids(FRAMES, declared, P)
    assert expected[0] == expected_grids(FRAMES, SYNCS, P)[0] and expected[1] != expected_grids(FRAMES, SYNCS, P)[1]
    records = run_shaper(FRAMES, honest_ready(), declared, Certifier("responses", P))
    assert [r.grid for r in records] == expected
    verdicts = Audit().check_link("responses", records, FRAMES, declared, SYNC_SET, P)
    assert [v.violation for v in verdicts] == [None, "advice", None]
    rows = [list(row) for row in records[2].grid]
    rows[2][0] = b"\xff"
    tampered = records[:2] + [Record("responses", 2, tuple(tuple(r) for r in rows), records[2].status)]
    verdicts = Audit().check_link("responses", tampered, FRAMES, declared, SYNC_SET, P)
    assert [v.violation for v in verdicts] == [None, "advice", "mismatch"]


def test_egress_links_with_the_same_syncs_are_accepted():
    """The spatial unit's clock is each window's sync, which every egress link declares alike."""
    audit = Audit()
    for link in ("responses", "tokens"):
        audit.check_link(link, run_shaper(FRAMES, honest_ready(), SYNCS, Certifier(link, P)), FRAMES, SYNCS, SYNC_SET, P)
    assert audit.accepted and len(audit.verdicts) == 6
    assert audit.syncs == dict(enumerate(SYNCS))


def test_a_link_with_another_sync_is_advice_in_that_window_only():
    """A second egress link declares δ = 2 in window 1, in the set but not the unit's: one sync per window for all links
    is the advice `EncardAccSeqsLeSync` counts, `#Σ_sync^W`. Window 1 of the link checked second is an advice
    violation, its other windows are accepted, and the first link checked sets the clock. The same holds committed."""
    other = [SYNCS[0], ClockSync(2, 1), SYNCS[2]]
    assert other[1] in SYNC_SET and other[1] != SYNCS[1]
    responses = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    tokens = run_shaper(FRAMES, honest_ready(), other, Certifier("tokens", P))
    assert all(v.accepted for v in Audit().check_link("tokens", tokens, FRAMES, other, SYNC_SET, P))
    for committed in (False, True):
        resp, tok = ([commit(r, P) for r in recs] if committed else recs for recs in (responses, tokens))
        audit = Audit()
        audit.check_link("responses", resp, FRAMES, SYNCS, SYNC_SET, P)
        audit.check_link("tokens", tok, FRAMES, other, SYNC_SET, P)
        assert [(v.link, v.window, v.violation) for v in audit.violations] == [("tokens", 1, "advice")]
        audit = Audit()
        audit.check_link("tokens", tok, FRAMES, other, SYNC_SET, P)
        audit.check_link("responses", resp, FRAMES, SYNCS, SYNC_SET, P)
        assert [(v.link, v.window, v.violation) for v in audit.violations] == [("responses", 1, "advice")]


def ingress_run(arrivals, faults=None):
    """An ingress certifier's network certificates: `arrivals[w][t]` frames reach it in bucket t of window w, when the outside chooses,
    and window w faults with `faults[w]` from its start when given."""
    certifier = Certifier("requests", P)
    records = []
    for w, counts in enumerate(arrivals):
        if faults and w in faults:
            certifier.fault(faults[w])
        for n in counts:
            for _ in range(n):
                certifier.enqueue(b"q")
            certifier.tick()
        records.append(certifier.close_window())
    return records


def test_ingress_links_are_checked_on_their_statuses_only():
    """Ingress rows are the Program's input, so no grid is compared: two runs with different timing are both accepted,
    raw and committed."""
    early = ingress_run([[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    late = ingress_run([[0, 0, 0, 2], [0, 0, 1, 1], [0, 0, 0, 3]])
    assert [r.grid for r in early] != [r.grid for r in late]
    for records in (early, late, [commit(r, P) for r in early], [commit(r, P) for r in late]):
        assert [v.violation for v in Audit().check_ingress_link("requests", records, 3)] == [None, None, None]


def test_ingress_status_duplicate_and_missing_record_are_violations():
    """An incomplete ingress status is a `status` violation, a second record for a window is a duplicate even if the two
    agree, and a missing record is no verdict, raw or committed. Another link's record for the window doesn't count."""
    records = ingress_run([[1, 0, 0, 0], [0, 2, 0, 0]], faults={1: Reason.DELIVERY_REFUSED})
    assert Audit().check_ingress_link("requests", records, 2)[0].accepted
    recs = [records[0], *records, Record("responses", 2, records[0].grid, COMPLETE)]
    expected = ["duplicate", "status:delivery-refused", "no-verdict"]
    assert [v.violation for v in Audit().check_ingress_link("requests", recs, 3)] == expected
    assert [v.violation for v in Audit().check_ingress_link("requests", [commit(r, P) for r in recs], 3)] == expected


def test_a_run_is_accepted_only_if_every_link_is():
    """Lean's `StatusComplete` needs every boundary link's status complete: with honest egress, the run is accepted when
    the ingress statuses are complete, and rejected, on the ingress window alone, when one is not."""
    egress = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    faulted = [("requests", 2, "status:monitor-unavailable")]
    for faults, violations in ((None, []), ({2: Reason.MONITOR_UNAVAILABLE}, faulted)):
        audit = Audit()
        audit.check_link("responses", egress, FRAMES, SYNCS, SYNC_SET, P)
        audit.check_ingress_link("requests", ingress_run([[1, 0, 0, 1]] * 3, faults), 3)
        assert len(audit.verdicts) == 6 and audit.accepted == (not violations)
        assert [(v.link, v.window, v.violation) for v in audit.violations] == violations
