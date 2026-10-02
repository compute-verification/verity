# Network warden (`verity_network_warden`)

A spec-level reference for network transparency's timing channel:
- the constant-rate, left-packed warden;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records or records committed by hash (padding elided);
- the capacity counts;
- the calibration of honest runs.

It is pure Python. It imports only the standard library and `verity`, for the record digest's `tagged_sha256` and
`canonical_json_bytes`.
`PROTOCOL.md` has the semantics and the correspondence to the Lean package.

~~~python
from verity_network_warden.audit import Audit
from verity_network_warden.grid import GridParams, Warden
from verity_network_warden.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min windows of 100 ms buckets
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per window
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, warden=Warden("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

Calibration, from a JSON trace of honest runs:

~~~text
uv run network-warden-calibrate trace.json --ingress-bucket-ms 100
~~~

Tests: `uv run tools/check/suites.py verity-network-warden`.
