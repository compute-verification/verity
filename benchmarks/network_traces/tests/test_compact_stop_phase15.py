"""REDTEAM phase 15, target 1 against #3800 (CERT-VARLEN, `cursor/cert-compact-stop-id-061a`): the compact stop is now
`COMPACT_LAST | COMPACT_STOP | id`, `certified_io` refuses a stop on a non-EOS id and an EOS id outside a stop, and
`compact_finish_violations` holds each response to the Commit's sampled ids. The forged-stop cut is a phase-12 regression
(`test_compact_eos_phase12`); this file attacks the parts it didn't:

- the Python/Lean boundary the lead flagged — `active.compact_record_data`'s record-validity and ordinal must agree with
  `CertifierDevice.Frame.compactOrdinal`'s rule (`request == 0 || field / compactStop == 1 → none`), across the whole
  field space including the STOP-without-LAST boundary and the old all-ones `COMPACT_EOS`;
- truncation, extension, pure relabel and finish relabel against `compact_finish_violations`, each a pure-function probe
  on the Commit's sampled ids (the realistic server-side refusals are the phase-12 regressions' `certified_io` cases).
"""
from __future__ import annotations

from verity.core.protocols.certifier import active

B = 16
COMPACT_STOP = active.COMPACT_STOP          # 2^30
COMPACT_LAST = active.COMPACT_LAST          # 2^31


def _lean_compact_ordinal(cell: bytes) -> int | None:
    """`CertifierDevice.Frame.compactOrdinal` transcribed: a cell is a record (and names its ordinal) iff it is at least
    `compactRecordSize` bytes, zero after them, its ordinal is nonzero, and its field is not `compactStop` alone
    (`field / compactStop == 1`, i.e. STOP set and LAST not). The device reads only the ordinal; finish is the server's."""
    size = active.COMPACT_RECORD.size
    if len(cell) < size or any(cell[size:]):
        return None
    request, field = active.COMPACT_RECORD.unpack_from(cell)
    if request == 0 or field // COMPACT_STOP == 1:
        return None
    return request


def _cell(request: int, field: int, *, pad: int = B, tail: int = 0) -> bytes:
    body = active.COMPACT_RECORD.pack(request, field)
    return body + bytes([tail]) + bytes(pad - len(body) - 1) if pad > len(body) else body


def test_phase15_compact_record_data_agrees_with_the_lean_device_rule():
    """Python's `compact_record_data` is a record exactly where Lean's `compactOrdinal` names an ordinal, and reads the
    same ordinal, across every top-two-bit region (none / STOP-only / LAST / LAST|STOP), boundary ids, ordinal 0 and the
    u32 max, the old all-ones `COMPACT_EOS` (now a stop on id 2^30-1), short cells and non-zero padding. No gap."""
    ids = (0, 1, 2, COMPACT_STOP - 1, COMPACT_STOP // 2, 12345)
    his = (0, COMPACT_STOP, COMPACT_LAST, COMPACT_LAST | COMPACT_STOP)   # top-two-bit regions: none, STOP, LAST, LAST|STOP
    requests = (0, 1, 5, (1 << 32) - 1)
    checked = 0
    for request in requests:
        for hi in his:
            for i in ids:
                cell = _cell(request, hi | i)
                py, ln = active.compact_record_data(cell), _lean_compact_ordinal(cell)
                assert (py is None) == (ln is None), (request, hex(hi | i), py, ln)
                if py is not None:
                    assert py[0] == ln == request, (request, hex(hi | i))
                checked += 1
    # the old COMPACT_EOS cell (all ones): both read it as a stop on id 2^30-1 / name its ordinal; it is a real record,
    # refused later by its finish, not by its framing
    eos_cell = active.COMPACT_RECORD.pack(7, 0xFFFF_FFFF)
    assert active.compact_record_data(eos_cell) == (7, COMPACT_STOP - 1, "stop") and _lean_compact_ordinal(eos_cell) == 7
    # STOP without LAST is no record in both; short cell and non-zero padding are no record in both
    assert active.compact_record_data(_cell(7, COMPACT_STOP | 3)) is None and _lean_compact_ordinal(_cell(7, COMPACT_STOP | 3)) is None
    assert active.compact_record_data(b"\0\0\0\7") is None and _lean_compact_ordinal(b"\0\0\0\7") is None
    assert active.compact_record_data(_cell(7, COMPACT_LAST | 3, tail=1)) is None and _lean_compact_ordinal(_cell(7, COMPACT_LAST | 3, tail=1)) is None
    assert checked == len(requests) * len(his) * len(ids)


def _cells(records_by_ordinal):
    return [active.compact_cell(r, B) for recs in records_by_ordinal for r in recs]


def test_phase15_the_tie_accepts_the_commits_own_cells():
    """The baseline: a link whose cells are exactly `compact_records_of` the Commit's sampled ids has no violation."""
    eos = {2}
    sampled = {1: [10, 11, 12], 2: [20, 2]}       # ordinal 2 ends on the EOS id -> its last record is a stop
    cells = _cells([active.compact_records_of(o, sampled[o], eos) for o in sorted(sampled)])
    assert active.compact_finish_violations(sampled, eos, cells) == []


def test_phase15_the_tie_catches_a_truncation():
    """A response cut short (fewer records than the Commit sampled, the kept records unmarked) is a length mismatch — the
    plain truncation beside phase-12's forged-stop cut."""
    eos = {2}
    sampled = {1: [10, 11, 12]}
    short = [active.compact_record(1, 10, None), active.compact_record(1, 11, None)]
    assert active.compact_finish_violations(sampled, eos, _cells([short])) == [(1, "2 records where the Commit sampled 3 ids")]


def test_phase15_the_tie_catches_an_extension():
    """A response run past the Commit (more records than it sampled) is caught — the extra record appended after the
    Commit's own records, whether its finish is length or a stop on the sampled EOS."""
    eos = {2}
    one = {1: [10, 11]}
    longer = active.compact_records_of(1, one[1], eos) + [active.compact_record(1, 99, None)]
    assert active.compact_finish_violations(one, eos, _cells([longer])) == [(1, "3 records where the Commit sampled 2 ids")]
    past = {2: [20, 2]}
    past_stop = active.compact_records_of(2, past[2], eos) + [active.compact_record(2, 20, "length")]
    assert active.compact_finish_violations(past, eos, _cells([past_stop])) == [(2, "3 records where the Commit sampled 2 ids")]


def test_phase15_the_tie_catches_a_pure_relabel():
    """A middle id swapped for another non-EOS id — same length, same finish, internally consistent — is caught by the id
    tie, which the response's framing and length checks alone would miss."""
    eos = {2}
    sampled = {1: [10, 11, 12]}
    relabeled = [active.compact_record(1, 10, None), active.compact_record(1, 77, None), active.compact_record(1, 12, "length")]
    assert active.compact_finish_violations(sampled, eos, _cells([relabeled])) == [(1, "output 1 is id 77 where the Commit sampled 11")]


def test_phase15_the_tie_catches_a_finish_relabel():
    """The Commit's finish, both ways: a stop reported as length (dropping the sampled EOS's mark) and a length reported
    as a stop are each a finish mismatch at the last output, the id unchanged."""
    eos = {2}
    as_length = [active.compact_record(1, 10, None), active.compact_record(1, 2, "length")]   # the EOS stop reported as a length
    assert active.compact_finish_violations({1: [10, 2]}, eos, _cells([as_length])) == [(1, "finish length on output 1 where the Commit's is stop")]
    as_stop = [active.compact_record(2, 20, None), active.compact_record(2, 21, "stop")]        # a length end reported as a stop
    assert active.compact_finish_violations({2: [20, 21]}, eos, _cells([as_stop])) == [(2, "finish stop on output 1 where the Commit's is length")]


def test_phase15_a_stop_without_last_and_a_wrong_ordinal_are_caught():
    """A STOP-without-LAST cell is no record (caught as a non-record cell, and its ordinal left short), and a record under
    an ordinal the Commit never sampled is caught as an unknown ordinal (violations ordinal-sorted)."""
    eos = {2}
    sampled = {1: [10]}
    malformed = active.compact_cell(active.COMPACT_RECORD.pack(1, COMPACT_STOP | 10), B)
    assert active.compact_record_data(malformed) is None
    assert active.compact_finish_violations(sampled, eos, [malformed]) == \
        [(0, "a cell that isn't a compact record"), (1, "0 records where the Commit sampled 1 ids")]
    stray = _cells([[active.compact_record(9, 10, "length")]])
    assert active.compact_finish_violations(sampled, eos, stray) == \
        [(1, "0 records where the Commit sampled 1 ids"), (9, "no request of the Commit is this ordinal")]
