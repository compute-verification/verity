# Network certifier (`verity.core.protocols.certifier`)

A spec-level reference for network accounting's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network certifier's signature over the record's SHA-512 digest, with the window's
  frames in a fixed-depth SHA-512 tree), and the ingress links' status check;
- one link-window's certificates through the proof service (`service`): the network `Spec`, a salted root over the
  certified records, and the check, whose Boolean part is each head's check (`head_check`: the head's hiding opening,
  its fields and its counts against the public occupancy profile), with a host-side verifier of record (no proof of the
  check yet) that compares no grid in the clear: the head check holds the counts on every link-window, and N3 the
  frames on every port-window;
- the capacity counts;
- the calibration of honest runs;
- the online prover (`online`): each window's clock sync declared at its first bucket from what it had seen `lag`
  buckets before;
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.
  With its salt on (`Proxy(salt=active.fresh_salt)`, off by default), each PoUW window (`pouw_window` certifier
  windows) gets one salt of `salt_bits` bits (512 by default), carried into the block only. The ingress record holds a
  salt frame with a salt-free placeholder, so every record is a function of the traffic alone; the salt sits in the
  certifier's one sealed record cell, `_salts[p]`, until `Proxy.reveal(p, registration)` opens it on the PoUW window's
  registration after its last ingress record is published, and `active.check_salts` reads the salt frames. A proxy back
  from maintenance (`Proxy(start=k, position=p)`, off by default) serves from window k on, its first egress frame the
  Program's at position p, and carries nothing from before: a window it was down for has no record.

It is pure Python. It imports only the standard library and `verity`: for the record digests' `tagged_sha256` and
`tagged_sha512` and `canonical_json_bytes`, in `service` the proof service, the hm96 leaves and the check's
Definition, and in `active` the proof service's registration and receipt, which `reveal` reads. The certificate's signature scheme is a parameter: the certifier's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
From `verity`, `head_check` takes the circuits, their evaluator and the commitment gadgets.
`PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`Security/`, area `NetworkCertifier`).
`lean/` is the certifier program, the Lake package `certifier_program` (Lean core only): List forms of the analysis's
definitions, proved equal to them in `Security/Proofs/NetworkCertifier/Code.lean`, and the difftest that holds this
package to them (`lake exe certifier-difftest INSTANCES PYTHON N` on what `lean/scripts/difftest_vectors.py` writes; the audit runs it
from `lean/lean-audit.json`).

The two terms of the charge the capacity bounds leave open are fixed in the trusted text, in the last section of
`Security/Models/NetworkCertifier/Guarantees.lean`'s documentation:
- the content advice `A_c`, which is the schedule. It gets a bound once the deployment declares a deterministic
  scheduler, admission on a step clock and a registered list of N profiles: `log2 N` once per audit, or
  `log2(N · #Σ_sync)` per window when the profile may switch at window boundaries;
- the challenger's coins (D7), an ingress link whose timing is charged on the ingress grid and whose content costs
  0 bits.

~~~python
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

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

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/certifier`.
