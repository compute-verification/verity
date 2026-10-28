# Network certifier (`verity.core.protocols.certifier`)

The **network certifier** is the auditor's device on a boundary link of a spatial unit. It shapes each link into a fixed
grid of buckets and frames, records exactly that grid and one status per execution phase, and once per execution phase
whose record is complete signs a **network certificate** over a salted digest of the record. The audit holds what the
certificates sign to the registered Program. The capacity bounds count what a covert sender can move across the boundary
through when frames leave, once the Program fixes every frame's content and order.

## Where it is stated

- **The verifier, in Lean.**
  - The certifier program `lean/` is the Lake package `certifier_program`, library `NetworkCertifier`, Lean core only.
    It holds the grid, the schedule and the charges as code. `Security/Proofs/NetworkCertifier/Code.lean` proves them
    equal to the analysis's definitions, and the difftests hold this package's Python to them (Running, below).
  - The checks of record are `NetworkCheck`, in `verity/core/protocols/network/lean`:
    - `Certificate`: the record head, the frame tree, the record digest and the certificate digest;
    - `Check`: `verity/network-certificate/v4`;
    - `Records`: `verity/network-records/v3`;
    - `LinkWindow`: one execution phase's verdict of record on a link, in its plain, served and network certifier forms;
    - `Period`: `verity/certifier/v0`, onsite's communication component. Its module docstring has both forms and the
      bound K.
  - The device is `CertifierDevice` (`verity/core/devices/certifier/lean`): the issuer, the salted clock certificate,
    the restart, the salt and a served link's admission, with its difftests.
- **Guarantees and properties.**
  - `Security/Definitions/NetworkCertifier` is the timing model `NetTiming`: its named assumptions (`Assumptions.lean`,
    W1 to W6 and J) and its guarantees (`Guarantees.lean` and `Guarantees/`). The last section of `Guarantees.lean`'s
    documentation fixes the content advice A_c and the coins' charge.
  - `Security/Proofs/NetworkCertifier` has the proofs. Its README has the modelling choices.
  - `Security/Properties` states the code's properties:
    - `NetworkCertifier/LinkWindow.lean`: what an accepted execution phase means on a link
      (`Properties.NetworkCertifier.LinkWindowAcceptMeans`);
    - `CertifierDevice.lean` and `CertifierNetwork.lean`: the device's output as the network certificate check reads it;
    - `NetworkRecords.lean`, `NetworkPeriod.lean` and `NetworkScan.lean`.
  - `Security/lean-audit.json` lists each guarantee (its `NetTiming.SecurityProofs.*` entries) and each named
    assumption.
- **Premises outside the Lean.**
  - `trusted-host` (`verity.core.claims`) covers the certifier, its host, its keys' custody and its LMS leaf state.
  - The challenger's `CoinsConfidential` and `OneClock` are in `verity/core/protocols/sampled_proofs/README.md`.
- **Formats and semantics** are in each module's docstring. Each Boolean Definition's `.circuit-check.py` sidecar and
  `.circuit-check.json` pins its circuit.

## Outline

- **The grid and the schedule.**
  - `grid`: the certifier over T, τ_b, r, B and Q, constant rate, left-packed, FIFO and fail-closed.
  - `schedule`: the Program's logical schedule, the clock-sync advice (δ, ρ) and the shaper.
  - `pacer` and `online`.
- **Records and certificates.**
  - `commitment`: the record v1.
  - `certificate`: the record v2, its salted digest, the `Issuer` and the LMS signer and key.
  - `audit`: the exact audit with record-and-reject, and the ingress status check.
  - `lean_audit`: `audit.Audit` with every verdict computed by the Lean check (`NetworkCheck.Records`), the Python
    beside it as the reference.
- **Through the proof system** (N2).
  - `service.LinkWindows`, with `head_check`.
  - `served`: a served link's counts.
  - `advised`: an advised link's counts.
  - `tie_service`: the ingress tie.
- **On live links.**
  - `active`: the active certifier, a TCP proxy with the PoUW salt, sealed pair links and a restart from maintenance.
  - `warden`: the network certifier's relay, which re-frames bytes into cells, with its `fails` record.
  - `warden_rows`, `transfer_rows`, `out_commit` and `release`.
  - `coin_link`: the coin link.
- **The ties and the scan.** `io_tie`, `io_request`, `io_ids`, `ingress_scan`, `egress`, `boundary` and `blocks`.
- **The advice and the compile slice.**
  - `advice`, `advice_check`, `replay_step`, `replay_chain`, `cache_step` and `cache_step_v2`.
  - `compile_slice`, `compile_slice_v2` and `compile_slice_v3`.
  - `served_rows`, `served_tie` and `served_statics`.
  - `arith_encode`, `charge_chain` and `window_charge`.
- **Capacity and calibration.** `capacity` and `calibration`.

It is pure Python. It imports only the standard library and `verity`. The signature scheme is a parameter:
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.

## The certifier device

The network certifier is auditor-trusted. It runs on a host the prover doesn't control (`trusted-host`), runs the grid
and signs what it ran:

- Once per execution phase whose record is complete, it signs an LMS network certificate (SHA-512, n = 64) over a
  salted digest of the record. Any other execution phase gets none (`Issuer.refused`) and fails closed (W6).
- It signs each execution phase's clock sync as a salted clock certificate.
- With the salt on, it draws the PoUW salt and reveals it on the registration of the PoUW execution phase it salts.

What it signs is witness. The record head and the certifier's salts go to the prover, and the auditor gets only the
certificates' public fields. A certifier run on the prover's own machine signs whatever that machine runs, so its
certificates prove nothing.

## Core API

~~~python
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # periods of 600 buckets of 100 ms
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per period
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, certifier=Certifier("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

- **The audit.** `audit.Audit`, with `check_link`, `check_ingress_link` and `active.check_salts`.
- **The device's signatures.**
  - `certificate.Issuer`, with `certify`, `salted_clock` and `refused`.
  - `certificate.LmsSigner` and `certificate.LmsKey`.
  - `CertifiedRecord.verify` and `CertifiedClock.verify`.
- **Through the proof system.**
  - `service.LinkWindows`, with `commit`, `register` and `verify` (the verifier of record).
  - `network_verdict`: the AND over a link's execution phases.
  - `ServedProfile` and `AdvisedProfile` for those links, and `service.WardenProfile` for a network certifier link.
  - `tie_service.IngressTie`: the ingress tie.
- **On a live link.**
  - `active.Proxy`, with `run`, `reveal` and `committed`.
  - `active.serve_reveals` and `active.RemoteCertifier`, the challenger's client.
  - `warden.Warden` and `coin_link.CoinLink`.
- **Calibration.** `calibration.calibrate`.

## Running

~~~text
uv run network-certifier-calibrate trace.json --ingress-bucket-ms 100   # calibration, from a JSON trace of honest runs
uv run tools/verity/check/suites.py verity/core/protocols/certifier      # the tests
~~~

The Lean difftests run from `lean/`, each on the vectors one script writes:

| Difftest | Vectors | Reference |
|---|---|---|
| `lake exe certifier-difftest INSTANCES PYTHON N` | `lean/scripts/difftest_vectors.py` | the grid, the schedule and the charges |
| `lake exe certifier-charge-difftest INSTANCES PYTHON N` | `lean/scripts/charge_vectors.py` | `arith_encode.reference_charge` |
| `lake exe certifier-window-difftest INSTANCES PYTHON N` | `lean/scripts/window_vectors.py` | `window_charge.reference_window` |

The Lean audit runs them from `lean/lean-audit.json`. The runs on live links are in `benchmarks/network_traces`
(`warden_serve.py`, `n2_zk.py`, `scan_zk.py`) and `benchmarks/live_challenger/coin_link.py`.

## Zero knowledge: what N2's verdict reads, and what's open

The network audit's claim is counts only. On every execution phase, N2 holds each end's per-bucket counts to the
occupancy profile. On a served link, it holds them to the counts the ingress counts derive under the phase's clock-sync advice.
Nothing here holds the frames.

N2's verifier of record, `LinkWindows.verify`, decides each execution phase on a `--zk` proof of N2's registered unit and
public inputs alone. It reads no head, salt or sibling. Its verdict of record is the Lean's (`NetworkCheck.LinkWindow`)
wherever `service.OF_RECORD` says so, with the Python beside it as the reference. The in-the-clear check,
`LinkWindows.clear`, is a debug comparison, off by default.

The certificate signs no hash of witness: it signs a hiding digest of the head's record digest under the certifier's
salt. That salt is handed to the prover with the head, as witness only (`CertifiedRecord.salt`). Each end's head check
holds the signed digest to the head it opens. The in-the-clear check also checks that link over the opening
(`CertifiedRecord.verify`).

What's open:
- **The certificates aren't a Spec item.** The verifier receives them in the clear from the certifier, beside the proof
  system, whose public kinds have none for a trusted device's signature over a hiding commitment. That is a
  provisional exception to the four public kinds, not a fifth kind.
- **A run that proves nothing** (a rehearsal, the simulator) is decided in the clear (`LinkWindows.unproved_outcome`,
  its verdict `"decided_by": "clear"`), which is never zero-knowledge.
- **An advised link's verdict of record** is still the Python's, and `tie_service.IngressTie.verify` isn't ported.
