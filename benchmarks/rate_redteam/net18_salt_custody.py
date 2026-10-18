"""Red-team witness (W5, target 3): NET-18's ingress record is a function of the salt at fixed traffic.

Accounting.Assumptions.BoundaryObs (cursor/attrib-ea24) asks that dX's certifiers' records placed during the window be a
function of the block's leak and the boundary's observations. Here the block leaks nothing and the link's traffic is
identical in both runs; only the certifier's salt differs. The ingress records differ (they hold salt_commitment(salt)),
and the frame the certifier sends into the block holds the salt itself, which the record (holding only its SHA-512
commitment) does not determine as a function. So, as implemented, the NET-18 certifier violates BoundaryObs,
SaltFreeJitter (sends a function of records and jitter) and SaltCustody (the certifier's history depends on the salt).

The frame into the block is built with the module's own `salt_frame`, the salt in place of its commitment, as the proxy
writes it on its server connection (active.py's module docstring). Runs on cursor/net18-salt-061a at afe591362:
``PYTHONPATH=<that worktree> python benchmarks/rate_redteam/net18_salt_custody.py``.
"""

from verity.core.protocols.certifier.active import (
    SALT, SALT_FRAME_BYTES, Bucketer, decode_frame, encode_frame, salt_commitment, salt_frame,
)
from verity.core.protocols.certifier.grid import Certifier, GridParams, row_count

P = GridParams(buckets=4, allocation=2, frame_bytes=96)
TRAFFIC = [(0, encode_frame(1, 7, b"prompt", P.frame_bytes)), (2, encode_frame(2, 7, b"end", P.frame_bytes))]


def run(salt: bytes):
    b = Bucketer(Certifier("requests", P), first=lambda w: salt_frame(w, salt_commitment(salt), P.frame_bytes))
    for bucket, frame in TRAFFIC:
        b.arrive(frame, bucket)
    rows, record = [], None
    for _ in range(P.buckets):
        row, rec = b.tick()
        rows.append(row)
        record = rec or record
    released = [decode_frame(f) for row in rows for f in row[: row_count(row)]]
    into_block = [salt_frame(int.from_bytes(p[:8], "big"), salt, SALT_FRAME_BYTES) for k, _, p in released if k == SALT]
    return record, into_block


s1, s2 = bytes(64), bytes([1]) + bytes(63)
r1, b1 = run(s1)
r2, b2 = run(s2)
print("same traffic, same (empty) leak; salts differ")
print("ingress records equal:", r1.grid == r2.grid)
print("record holds salt_commitment(s1):", any(salt_commitment(s1) in (f or b"") for row in r1.grid for f in row))
print("frame sent into the block carries the salt itself:", s1 in b1[0], s2 in b2[0])
