# Network certifier (`verity.protocols.accounting.communication.certifier`)

A spec-level reference for network transparency's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records or records committed by hash (padding elided), and the
  ingress links' status check;
- the capacity counts;
- the calibration of honest runs;
- the online prover (`online`): each window's clock sync declared at its first bucket from what it has seen by then;
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.

It is pure Python. It imports only the standard library and `verity`, for the record digest's `tagged_sha256` and
`canonical_json_bytes`.
`PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`verity/Security/`, area `NetworkCertifier`).
`lean/` is the certifier program, the Lake package `certifier_program` (Lean core only): List forms of the analysis's
definitions, proved equal to them in `verity/Security/Proofs/NetworkCertifier/Code.lean`, and the difftest that holds this
package to them (`lake exe certifier-difftest INSTANCES PYTHON N` on what `lean/scripts/difftest_vectors.py` writes; the audit runs it
from `lean/lean-audit.json`).

~~~python
from verity.protocols.accounting.communication.certifier.audit import Audit
from verity.protocols.accounting.communication.certifier.grid import GridParams, Certifier
from verity.protocols.accounting.communication.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min windows of 100 ms buckets
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per window
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, certifier=Certifier("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

Calibration, from a JSON trace of honest runs:

~~~text
uv run network-certifier-calibrate trace.json --ingress-bucket-ms 100
~~~

Tests: `uv run tools/check/suites.py verity-network-certifier`.
