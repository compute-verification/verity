"""The verifier's check: exact, slot by slot, with record-and-reject.

An egress link-window (`check_link`) is **accepted** if three things hold:
- its status is complete;
- every slot of its grid equals the Program's grid;
- the window's declared clock sync is in the protocol's set, and is the spatial unit's sync for that window.

It must also have exactly one record, settled by the verdict deadline: a link-window with more than one record is a
`duplicate` violation, whatever the records say. Anything else is a **violation**. It is recorded and the run is
rejected, as a policy signal. There is no halt: later windows are still checked and recorded. Every bound in
`capacity` holds on accepted runs only; a rejected window may already have carried up to its unbound capacity before it
was rejected.

A spatial unit has one clock. The audit keeps each window's sync (`Audit.syncs`), set by the first egress link checked
for that window, and a link that declares a different sync for the window is an `advice` violation there. The bound on
accepted runs charges one sync per window for all links, `#Σ_sync^W` values over W windows (Lean
`SecurityProofs.EncardAccSeqsLeSync`, whose timing advice is `Fin W → Sync`): 7 bits per window at 16 × 8. A sync per
link would be advice of `#Σ_sync^(L·W)`, which that bound doesn't cover.

The Program's grids are computed from the syncs as declared, out-of-set values included, so a declared sync outside the
set, or other than the unit's, is an `advice` violation in its own window only. A window's grid reads only the syncs of
its own and earlier windows (`schedule.expected_grids`), so such a sync never changes an earlier window's verdict.

A record may come raw or committed (`commitment`: its digest, with padding elided). A committed record's grid check
compares its digest with the digest of the Program's grid, which is exact up to a SHA-256 collision.

An ingress link's rows are the Program's input, which it reads rather than predicts, so `check_ingress_link` checks only
its statuses: a link-window is accepted if it has exactly one record and its status is complete (Lean `StatusComplete`).
On a committed record it reads the status in the clear, which the digest binds. With no Program grid to recompute the
digest from, it can't catch a published status that disagrees with the digest, as the egress check does.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from verity.protocols.accounting.communication.certifier.commitment import CommittedRecord, record_digest
from verity.protocols.accounting.communication.certifier.grid import COMPLETE, GridParams, Record
from verity.protocols.accounting.communication.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids


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
    """Every verdict of a run, in order, and the spatial unit's clock: each window's sync, set by the first egress link
    checked for that window. The run is accepted only if every link-window is."""

    verdicts: list[Verdict] = field(default_factory=list)
    syncs: dict[int, ClockSync] = field(default_factory=dict)

    def check_link(
        self,
        link: str,
        records: Sequence[Record | CommittedRecord],
        frames: Sequence[LogicalFrame],
        syncs: Sequence[ClockSync],
        sync_set: SyncSet,
        params: GridParams,
    ) -> list[Verdict]:
        """Check one egress link's records for windows 0 … len(syncs) − 1 against the Program's grids under the
        declared syncs. A window whose declared sync differs from the unit's for it, set by an earlier link, is an
        `advice` violation."""
        expected = expected_grids(frames, syncs, params)
        by_window = _by_window(link, records)
        out = []
        for w, sync in enumerate(syncs):
            unit_sync = self.syncs.setdefault(w, sync)
            found = by_window.get(w, [])
            violation = _status_violation(found)
            if violation is None:
                record = found[0]
                if sync not in sync_set or sync != unit_sync:
                    violation = "advice"
                elif isinstance(record, CommittedRecord):
                    if record.digest != record_digest(link, w, expected[w], COMPLETE, params):
                        violation = "mismatch"
                elif record.grid != expected[w]:
                    violation = "mismatch"
            out.append(Verdict(link, w, violation))
        self.verdicts.extend(out)
        return out

    def check_ingress_link(self, link: str, records: Sequence[Record | CommittedRecord], windows: int) -> list[Verdict]:
        """Check one ingress link's records for windows 0 … windows − 1: one record each, with a complete status. No
        grid is compared."""
        by_window = _by_window(link, records)
        out = [Verdict(link, w, _status_violation(by_window.get(w, []))) for w in range(windows)]
        self.verdicts.extend(out)
        return out

    @property
    def accepted(self) -> bool:
        return bool(self.verdicts) and all(v.accepted for v in self.verdicts)

    @property
    def violations(self) -> list[Verdict]:
        return [v for v in self.verdicts if not v.accepted]


def _by_window(link: str, records: Sequence[Record | CommittedRecord]) -> dict[int, list[Record | CommittedRecord]]:
    by_window: dict[int, list[Record | CommittedRecord]] = defaultdict(list)
    for r in records:
        if r.link == link:
            by_window[r.window].append(r)
    return by_window


def _status_violation(found: Sequence[Record | CommittedRecord]) -> str | None:
    """The violations every link-window can have, in order: no record, more than one, or an incomplete status."""
    if not found:
        return "no-verdict"
    if len(found) > 1:
        return "duplicate"
    if not found[0].status.complete:
        return f"status:{found[0].status.reason.value}"
    return None
