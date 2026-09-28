# Network certifier (`verity.core.protocols.certifier`)

A spec-level reference for network accounting's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network certifier's signature over the record's SHA-512 digest, with the window's
  frames in a fixed-depth SHA-512 tree), and the ingress links' status check;
- one link-window's certificates through the proof service (`service`): the network `Spec`, a salted root over the
  certified records, and a host-side verifier of record (no circuit proves the check yet);
- the capacity counts;
- the calibration of honest runs;
- the online prover (`online`): each window's clock sync declared at its first bucket from what it had seen `lag`
  buckets before;
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.

It is pure Python. It imports only the standard library and `verity`: for the record digests' `tagged_sha256` and
`tagged_sha512` and `canonical_json_bytes`, and in `service` the proof service, the hm96 leaves and the check's
Definition. The certificate's signature scheme is a parameter: the certifier's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
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
