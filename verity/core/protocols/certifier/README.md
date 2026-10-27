# Network certifier (`verity.core.protocols.certifier`)

The network certifier is the auditor's device on a server's network link. It sits on the link, carries every byte that
crosses it on a public, constant-rate schedule of buckets, and signs what crossed: each link-window's record, its digest
hidden under a 64-byte salt the certifier draws fresh, so the signature is over a hiding commitment and the record
stays witness. Network accounting reads those certificates. Its check, N2, holds each link-window's per-bucket counts to
the link's public occupancy profile (on a served or advised link, to the counts its ingress and the declared sync or the
advice derive), which puts the link's timing at the profile. The claim is counts only: N3, which would hold the frames
inside the counts to the Program's, is a design and the simulator's behaviour (`benchmarks/network_sim`), not a
protocol.

**Trust.** The certifier is the auditor's hardware. It runs on a host the prover doesn't control, never as a process on
the prover's node, and trust in the device and its host is the named assumption `trusted-host`
(`Verity.Assumptions.TrustedHost`, `Security/Definitions/Core/Assumptions.lean`): the certifier runs its stated code, on
a monotonic clock and a CPU no other tenant's load reaches, and its host never restores an earlier copy of the state it
keeps there. Nothing the prover runs or hosts is trusted.

`PROTOCOL.md` has the semantics, the leak bounds and the correspondence to the Lean analysis, and `protocol/` its design
notes.

## Modules

Each module's docstring has its detail. A Boolean Definition has its reference beside it and a circuit-check sidecar
(`<module>.circuit-check.py`, its pins in `<module>.circuit-check.json`).

The certifier on a link:
- `grid`: the certifier's grid on one link (`GridParams`, `Certifier`): constant-rate, left-packed, FIFO release.
- `schedule`: the Program's logical schedule and its clock-sync advice (`LogicalFrame`, `ClockSync` (δ, ρ), `SyncSet`,
  `run_shaper`).
- `online`: the honest prover's clock sync, declared at each window's first bucket from what it had seen `lag` buckets
  before.
- `active`: the certifier on a live link, an asyncio TCP proxy (`Proxy`) that runs the certifiers and publishes their
  records: the PoUW window's salt and its reveal (`fresh_salt`, `check_salts`, `serve_reveals`, `RemoteCertifier`), the
  weights' root, a served link's admission rule, a proxy back from maintenance, and a sealed pair link between two of
  the auditor's certifiers (`Pair`).
- `warden`: the active warden, a relay that knows bytes only and re-frames each connection into B-byte cells on a fixed
  schedule; a late cell fails its connection closed, with a `FAIL` cell that carries no reason.
- `release`: the release-order headers a certifier puts on the server's copy of each request it releases, which never
  appear in a cell or record.

Records and certificates:
- `certificate`: each link-window's signed record (`Issuer`, `Certificate`, `CertifiedRecord`) and the clock
  certificates (`CertifiedClock`). The signature scheme is a parameter: the `Issuer` takes a `Signer`, the audit a
  `PublicKey`.
- `commitment`: a record committed by hash, with padding elided (`CommittedRecord`).
- `warden_rows`: the warden's cells committed one hm96 row a bucket under a root the warden signs, which a proof block's
  `--registered` port reads (`bucket_port`).
- `out_commit` and `transfer_rows`: the output value the certifier relayed, and every value that crosses a server
  block's boundary, committed as registered values under the certifier's own salts.
- `blocks`: a circuit's server blocks, its server-time block and its transfer gates (`verity/blocks/v1`). `check`
  refuses blocks that don't partition the units, an inference block on two servers, and a boundary row no transfer
  brings; which server runs which units is protocol structure, checked in the clear with no proof block.
- `boundary`: the run's boundary in a window's circuit (`verity/boundary/v3`, `NetTiming.Layout`), and the verifier's
  refusal of a layout that doesn't fit the circuit.

The audit:
- `audit`: the verifier's exact check, slot by slot, with record-and-reject (`Audit`), on raw records, records committed
  by hash or certificates.
- `lean_audit`: `Audit` with every verdict computed by the Lean check (`NetworkCheck.Records`).
- `capacity`: the counts the Lean proves, and the capacity bounds built on them.
- `calibration`: sizing the grid from honest timing traces.

N2 through the proof service:
- `service`: one link-window's certificates through the proof service: the network `Spec`, the salted root over the
  certified records, and the verifier of record, `LinkWindows` (`verify`, and `network_verdict`, the AND over a link's
  link-windows with no ε or share credit).
- `head_check`: each head's check, `NetworkHeadCheck_v2`: its hiding opening, its fields, its counts against the profile,
  and its record digest under the certifier's salt against the digest its certificate signs.
- `served`: a served link's counts, `NetworkServedCounts_v1`, the egress counts derived from the ingress counts under the
  window's declared sync.
- `advised`: an advised link's counts, `NetworkAdvisedCounts_v1`, held to the releases of the prover's advice, with
  each window's sync read from the certifier's clock certificates.
- `tie_service`: the ingress tie through the proof service (`IngressTie`), which builds `advised.IoTieVerdict`.

The step-indexed scan and the I/O tie:
- `ingress_scan`: `IngressScanBucket` (`IngressScanFirst` for bucket 0), one unit a certifier bucket: the warden's
  ingress row and the carried slot state to the next state and its egress view. Programs start at fixed slots a network
  timestep.
- `egress`: `EgressBucket`, one unit a link and bucket, computing the bucket's row of cells from the scan's state, held
  to the warden's egress row (a prototype on a fake upstream, in no Spec).
- `io_tie`: the I/O tie, each egress cell held to the token id the Commit sampled (`IoTieBucket`, `IoTieContent`; a
  prototype on a fake upstream, in no Spec).
- `io_request`: the ingress tie (`IoTieRequest`), each request's prompt length and arrival ordinal held to the warden's
  certified ingress cells.
- `io_ids`: the ingress-ids tie (`IoTieRequestIds`), every prompt id held to the Commit's `c00.input/ids` rows.

Advice and the compile unit:
- `advice`: advice v0, what vLLM's scheduler decided in a session as one canonical bit string, refused unless it is a
  schedule vLLM could have run.
- `advice_check`: `AdviceCheck_v1{L}`, advice v0's validity, the compile unit's first part.
- `replay_step`: `ReplayStep_v1`, one vLLM v1 scheduler step, the compile unit's core (in no proof-service Spec).
- `replay_chain`: the replay's state handed from one step unit to the next under hiding commitments (`ReplayAdvance`,
  `ReplayLoad`, `ReplayClose`, `check_chain`).
- `cache_step` and `cache_step_v2`: vLLM's prefix cache (`CacheStep_v1`), and its block pool with eviction and frees
  (`CacheStep_v2`).
- `arith_encode`: advice v1's arithmetic coder as Boolean Definitions (`ArithCountStep_v1`, `ArithCountFinish_v1`,
  `BookKT_v1`, composed in `ChargeCount_v1`), the window's charge from its decisions.
- `charge_chain` and `window_charge`: the charge's carry from window to window (`ChargeLink_v1`), and the whole window's
  charge (`ChargeCount_v2`, `ChargeLink_v2`).
- `compile_slice`, `compile_slice_v2` and `compile_slice_v3`: the compile unit's parts composed for one window
  (`CompileSlice_v1`; `CompileSlice_v2` with the block pool; `CompileSlice_v3` computing its schedule, so the advice is
  its only free input and the charge its width).
- `served_rows`, `served_tie` and `served_statics`: the step rows a served window ran, registered before the draw, held
  to the positions its steps' Commit holds (`ServedRowsTie_v2`) and to the RoPE statics of the Programs they ran
  (`ServedStaticsTie_v1`).

It is pure Python, importing only the standard library and `verity`.

## Use

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

Live runs and proofs are drivers in `benchmarks/network_traces/`: `certifier_serve.py` runs the active certifier on a
link, `warden_serve.py` the warden, `n2_zk.py` proves a run's link-windows through N2's unit, and `scan_zk.py` the
step-indexed scan. `benchmarks/network_e2e/` is the rehearsal end to end.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/certifier`.

## Lean

- `lean/`: the certifier program, the Lake package `certifier_program` (`NetworkCertifier`, Lean core only): List forms
  of the analysis's definitions, proved equal to them in `Security/Proofs/NetworkCertifier/Code.lean`, and the
  difftests that hold this package's Python to them: `lake exe certifier-difftest INSTANCES PYTHON N` on what
  `lean/scripts/difftest_vectors.py` writes, `certifier-charge-difftest` on what `lean/scripts/charge_vectors.py` writes
  from `arith_encode.reference_charge`, and `certifier-window-difftest` on what `lean/scripts/window_vectors.py` writes
  from `window_charge.reference_window`.
- `verity/core/devices/certifier/lean/`: the certifier device, the Lake package `certifier_device` (`CertifierDevice`):
  `active.Proxy` as a pure state machine and `certificate.Issuer`, in executable Lean, held to them by
  `certifier-device-difftest` and `certifier-issuer-difftest`.
- N2's Lean verifier is `NetworkCheck.LinkWindow` in `verity/core/protocols/network/lean` (below).
- The analysis: `Security/Definitions/NetworkCertifier/` (the definitions, the assumptions and `Guarantees.lean`, whose
  last section fixes the charge's two terms the capacity bounds leave open), `Security/Proofs/NetworkCertifier/` (the
  proofs) and `Security/Properties/NetworkCertifier/LinkWindow.lean` (the Lean verdict's property).

## Zero knowledge: what N2's verdict reads, and what's open

Every protocol is to be zero-knowledge. The network audit's claim is counts only: N2 holds each window's per-bucket
counts to the occupancy profile on every link-window (on a served link, to the ones its ingress counts derive under
the declared sync), beside the proved timing bound, while N3, the content check, is the design (Daniel's provisional
call on ASK-70, 19 Oct).

N2's verifier of record, `LinkWindows.verify`, decides each link-window on a `--zk` proof of N2's registered unit
(`NetworkLinkWindow` v5, served `NetworkServedLinkWindow` v4) and public inputs alone: its own run of C-Flock's Lean
verifier accepts the proof, the proof's public row opens to its own copy of the unit's public arguments, the output
row's verdict and deadline bits are 1, the certificates verify on their public fields (`LinkWindows.certified`), and
the window's sync is declared. It reads no head, salt or sibling. `benchmarks/network_traces/n2_zk.py` proves a
certifier-serve run's link-windows this way through the link's own stream, and scans what the verifier received. The
in-the-clear check, `LinkWindows.clear`, is a debug comparison, off by default. What's open:
- **The certificates aren't a Spec item.** The verifier receives them in the clear from the certifier, the auditor's
  device, beside the proof service, whose public kinds have none for a trusted device's signature over a hiding
  commitment. That is a provisional exception to the four public kinds (decided 24 Oct, option A), not a fifth kind.
- **A run that proves nothing** (a rehearsal, the simulator) is decided in the clear (`LinkWindows.unproved_outcome`,
  its verdict `"decided_by": "clear"`), which is never zero-knowledge.

`LinkWindows.verify`'s verdict of record on a plain, served or warden link is the Lean's, `NetworkCheck.LinkWindow`
(`verity/core/protocols/network/lean`, the `link-window` function of `network-records verdicts`), by the one switch line
`service.OF_RECORD`; an advised link's stays the Python's. `verify` runs both on every proof, and a link-window where
they differ in the accept bit, the public outputs or the digest, or which the Lean refuses, is rejected. The switch went
to `lean` under the 8 Oct ruling (the friction skill), on these grounds:
- the gap list is empty for plain and served links. Everything the Python checks, the Lean checks: the findings prove
  the registered Program and the drawn units with every gate pinned, the public row the verifier derives (root, served
  domain, link, window, each end's grid and profile row, both signed digests, a served link's sync words), both rows'
  openings and the word 0b11, both certificates under the LMS key in the certifier session, and the sync in Σ_sync. Both
  take as inputs what the proof service and the lowering give: the findings of the verifier's own C-Flock Lean run, the
  sessions' SHA-512s, the registration, the draw, and the Program's and partition's digests and the gate ids. On an
  advised link the window view (`advised.view`, from the tie and clock sources) is the Python's and an input to the Lean,
  which is why advised stays `python`. The Lean reads an LMS key only, and rejects any other.
- on a warden link (`service.WardenProfile`) the gap list is empty too: the Lean's warden form (`Kind.warden`) is the
  plain unit's public row with both ends' grids at the warden's parameters digest, which both take as an input (the
  warden's deployment publishes it), and no sync read, as in the Python; it refuses a warden case whose digest isn't 64
  bytes, whose ends have a profile, or whose two grids differ.
- `benchmarks/network_traces/n2_lean_replay.py` replays recorded proving runs of N2 through both, each with injected
  negative controls that both reject.
- the Python keeps running beside the Lean as the reference.
- the Lean's verdict has its property in `Security/Properties/NetworkCertifier/LinkWindow.lean`
  (`Properties.NetworkCertifier.LinkWindowAcceptMeans`, proved): an accepted link-window was proved, opened to the
  word 0b11, certified on both ends and synced within Σ_sync (a warden link's syncs are not asked of), and its public
  outputs are verdict 1 and deadline 1.

`tie_service.IngressTie.verify`, the ingress tie's verifier, is not ported and stays Python.

A proof of the unit is held to its circuit. Each gate names its Definition's circuit digest
(`Gate[NetworkHeadCheck_v2{…}@<digest>]_v2`, `served.gate` likewise), as `head_check.circuit-check.json` and
`served.circuit-check.json` pin it and circuit-check holds the pin to the Definition. So the unit's Program, which the
spec and each registration bind, binds both circuits, and `LinkWindows.verify` refuses a proof that names any other
Program, or a unit of a size class no sidecar pins (`LinkWindows.proved`).

The certificate no longer signs a hash of witness (NET-42). The certifier signs `D = SHA-512("verity/net-cert/v1" ‖
salt_c ‖ R)`, R the head's record digest and `salt_c` 64 bytes the certifier draws fresh for each record
(`certificate.Issuer`'s `coins`) and hands to the prover with the head, as witness only (`CertifiedRecord.salt`). D is
a hiding commitment to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D.
Each end's head check holds R to the head it opens and D to R under `salt_c`, so the certificates' check reads nothing
of the heads; the in-the-clear check also checks that link over the opening (`CertifiedRecord.verify`).
