"""The verifier's check: exact, slot by slot, with record-and-reject (timing-channel doc §1.2 and §6).

A link-window is **accepted** if three things hold:
- its status is complete;
- every slot of its grid equals the Program's grid;
- the window's declared clock sync is in the protocol's set.

It must also be settled by the verdict deadline. Anything else is a **violation**. It is recorded and the run is
rejected, as a policy signal. There is no halt: later windows are still checked and recorded. Every bound in
`capacity` holds on accepted runs only; a rejected window may already have carried up to its unbound capacity before it
was rejected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from verity_network_timing.grid import GridParams, Record
from verity_network_timing.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids


@dataclass(frozen=True, slots=True)
class Verdict:
    link: str
    window: int
    violation: str | None  # None when accepted; otherwise "status:<reason>", "advice", "mismatch" or "no-verdict"

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
        records: Sequence[Record],
        frames: Sequence[LogicalFrame],
        syncs: Sequence[ClockSync],
        sync_set: SyncSet,
        params: GridParams,
    ) -> list[Verdict]:
        """Check one link's records for windows 0 … len(syncs) − 1 against the Program's grids."""
        expected = expected_grids(frames, syncs, params) if all(s in sync_set for s in syncs) else None
        by_window = {r.window: r for r in records if r.link == link}
        out = []
        for w, sync in enumerate(syncs):
            record = by_window.get(w)
            if record is None:
                violation = "no-verdict"
            elif not record.status.complete:
                violation = f"status:{record.status.reason.value}"
            elif sync not in sync_set or expected is None:
                violation = "advice"
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
