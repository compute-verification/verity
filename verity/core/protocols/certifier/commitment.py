"""The published form of a network certificate: committed by hash, with padding elided.

A raw record is the whole grid, T·r slots of B bytes: 2.3 GiB per link-window at r = 1,024, B = 4 KiB and T = 600.
The certifier publishes instead one digest and the untimed status per link-window.

**Padding is elided.** The certifier is left-packed (W4), so a row is fixed by its count n and its n real frames: slots
n … r − 1 are padding. A row's digest covers only those:

    row(n, f_0 … f_{n−1}) = tagged_sha256("verity-network-warden/row/v1", u32be(n) ‖ f_0 ‖ … ‖ f_{n−1})

The frames all have the fixed size B, so the encoding is unambiguous. A row that isn't left-packed has no encoding.

**The record digest** binds the grid parameters, the link, the window and the status, then the T row digests:

    record = tagged_sha256("verity-network-warden/record/v1", u32be(len(h)) ‖ h ‖ row_0 ‖ … ‖ row_{T−1})

Here `h` is the canonical JSON of {link, window, buckets, allocation, frame_bytes, bucket_ms, status}.

**The check.** The verifier recomputes the digest from the Program's grid, with a complete status. It accepts the
link-window only if the digests are equal. Under the collision resistance of SHA-256 (`cr/sha-256`), equal digests mean
equal records, so the audit stays exact. It never needs the certifier's raw grid, since the Program fixes every egress
frame's content. The capacity bounds on accepted runs then hold up to a SHA-256 collision.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.ml.commitments.identity import Digest, canonical_json_bytes, tagged_sha256
from verity.core.protocols.certifier.grid import Grid, GridParams, Record, Row, Status, row_count, row_left_packed

ROW_TAG = "verity-network-warden/row/v1"
RECORD_TAG = "verity-network-warden/record/v1"


def row_digest(row: Row, params: GridParams) -> bytes:
    """A left-packed row's digest over its count and real frames; padding is elided."""
    if len(row) != params.allocation:
        raise ValueError(f"a row has {params.allocation} slots, not {len(row)}")
    if not row_left_packed(row):
        raise ValueError("a row that isn't left-packed has no encoding (W4)")
    n = row_count(row)
    frames = row[:n]
    if any(len(f) != params.frame_bytes for f in frames):
        raise ValueError(f"frames are {params.frame_bytes} bytes")
    return bytes.fromhex(tagged_sha256(ROW_TAG, n.to_bytes(4, "big") + b"".join(frames)))


def record_digest(link: str, window: int, grid: Grid, status: Status, params: GridParams) -> Digest:
    if len(grid) != params.buckets:
        raise ValueError(f"a grid has {params.buckets} rows, not {len(grid)}")
    header = canonical_json_bytes({
        "link": link, "window": window, "buckets": params.buckets, "allocation": params.allocation,
        "frame_bytes": params.frame_bytes, "bucket_ms": params.bucket_ms,
        "status": "complete" if status.complete else status.reason.value,
    })
    rows = b"".join(row_digest(row, params) for row in grid)
    return tagged_sha256(RECORD_TAG, len(header).to_bytes(4, "big") + header + rows)


@dataclass(frozen=True, slots=True)
class CommittedRecord:
    """What the certifier publishes per link-window: the record digest, and the status in the clear so a violation can be
    classified. The digest binds the status too, so a status that disagrees with it is a mismatch."""

    link: str
    window: int
    digest: Digest
    status: Status


def commit(record: Record, params: GridParams) -> CommittedRecord:
    return CommittedRecord(record.link, record.window,
                           record_digest(record.link, record.window, record.grid, record.status, params), record.status)
