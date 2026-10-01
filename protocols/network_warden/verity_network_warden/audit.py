"""The verifier's check: exact, slot by slot, with record-and-reject.

A link-window is **accepted** if three things hold:
- its status is complete;
- every slot of its grid equals the Program's grid;
- the window's declared clock sync is in the protocol's set.

It must also have exactly one record, settled by the verdict deadline: a link-window with more than one record is a
`duplicate` violation, whatever the records say. Anything else is a **violation**. It is recorded and the run is
rejected, as a policy signal. There is no halt: later windows are still checked and recorded. Every bound in
`capacity` holds on accepted runs only; a rejected window may already have carried up to its unbound capacity before it
was rejected.

The Program's grids are computed from the syncs as declared, out-of-set values included, so a declared sync outside the
set is an `advice` violation in its own window only. A window's grid reads only the syncs of its own and earlier
windows (`schedule.expected_grids`), so an out-of-set sync never changes an earlier window's verdict.

A record may come raw or committed (`commitment`: its digest, with padding elided). A committed record's grid check
compares its digest with the digest of the Program's grid, which is exact up to a SHA-256 collision.

Only egress links are checked: ingress links' statuses, which Lean's `Accepted` also requires complete, are not audited
yet (PROTOCOL.md).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from verity_network_warden.commitment import CommittedRecord, record_digest
from verity_network_warden.grid import COMPLETE, GridParams, Record
from verity_network_warden.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids


@dataclass(frozen=True, slots=True)
class Verdict:
    link: str
    window: int
    # None when accepted; otherwise "status:<reason>", "advice", "mismatch", "duplicate" or "no-verdict"
    violation: str | None

    @property
    def accepted(self) -> bool:
        return self.violation is None


@dataclass
class Audit:
    """Every verdict of a run, in order. The run is accepted only if every link-window is."""

    verdicts: list[Verdict] = field(default_factory=list)

    def check_link(
        self,
        link: str,
        records: Sequence[Record | CommittedRecord],
        frames: Sequence[LogicalFrame],
        syncs: Sequence[ClockSync],
        sync_set: SyncSet,
        params: GridParams,
    ) -> list[Verdict]:
        """Check one link's records for windows 0 … len(syncs) − 1 against the Program's grids under the declared
        syncs."""
        expected = expected_grids(frames, syncs, params)
        by_window: dict[int, list[Record | CommittedRecord]] = defaultdict(list)
        for r in records:
            if r.link == link:
                by_window[r.window].append(r)
        out = []
        for w, sync in enumerate(syncs):
            found = by_window.get(w, [])
            if not found:
                violation = "no-verdict"
            elif len(found) > 1:
                violation = "duplicate"
            elif not (record := found[0]).status.complete:
                violation = f"status:{record.status.reason.value}"
            elif sync not in sync_set:
                violation = "advice"
            elif isinstance(record, CommittedRecord):
                violation = None if record.digest == record_digest(link, w, expected[w], COMPLETE, params) else "mismatch"
            elif record.grid != expected[w]:
                violation = "mismatch"
            else:
                violation = None
            out.append(Verdict(link, w, violation))
        self.verdicts.extend(out)
        return out

    @property
    def accepted(self) -> bool:
        return bool(self.verdicts) and all(v.accepted for v in self.verdicts)

    @property
    def violations(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]
