# Network certifier (`verity.core.protocols.certifier`)

The network certifier is the auditor's device on each boundary link of a spatial unit. It releases the link's traffic on
a fixed grid, constant-rate and left-packed: r slots of B bytes at each tick of its clock, T ticks an execution phase. It
records exactly what it sent, and signs a network certificate over a hiding commitment to each record. The audit holds
each record to the registered Program's logical schedule, so what a link's timing can carry beyond the Program is what
the guarantees below bound, and network accounting charges that bound. The network certifier is the auditor's hardware
on a host the prover doesn't control (`trusted-host`); nothing the prover runs or hosts is trusted.

This package is the protocol's Python: the network certifier and its proxy, the records and certificates, the audit's
Python mirror, the proof blocks the prover proves for the network's statements, and the honest side (calibration, the
online prover). The verifier is Lean.

## The verifier and its guarantees

- **The verifier**: `verity/core/protocols/network/lean`, the Lake package `network_check`.
  - `NetworkCheck.Certificate`: a network certificate's encodings: the frame tree, the parameters digest, the record
    head and the digest a certificate signs.
  - `NetworkCheck.Check`: the certificate check, its LMS signature included, `verity/network-certificate/v4`
    (`lake exe network-check`).
  - `NetworkCheck.Records`, `NetworkCheck.RecordsGrid` and `NetworkCheck.RecordsBytes`: the records against the
    Program's grid, `verity/network-records/v3` (`lake exe network-records`). `lean_audit.Audit` is `audit.Audit` with
    every verdict the Lean's.
  - `NetworkCheck.LinkWindow`: one execution phase's transfer blocks through the proof system, the `link-window`
    function of `network-records verdicts`. It is the verdict `service.LinkWindows.verify` returns on a plain, served or
    relay link (`service.OF_RECORD`).
  - `NetworkCheck.Period`: a certification period, `verity/certifier/v0` (`lake exe network-period`), in its clock form
    and its rows form.
- **The certifier program**: `lean/`, the Lake package `certifier_program` (Lean core only).
  `NetworkCertifier.Grid`, `Schedule`, `Ingress`, `Run`, `AdviceV1` and `WindowCharge` are List forms of the analysis's
  definitions, proved equal to them in `Security/Proofs/NetworkCertifier/Code.lean`.
- **The device**: `verity/core/devices/certifier/lean`, the Lake package `certifier_device`: the proxy (`active`) as a
  pure state machine (`CertifierDevice.step`, in `CertifierDevice.Device`), with its certifiers, its online prover, a
  served link's admission rule and the issuer that signs each record.
- **The guarantees**: `Security/Definitions/NetworkCertifier/Guarantees.lean` and `Guarantees/`, each proved in
  `Security/Proofs/NetworkCertifier/` (whose README states the model's choices). The named assumptions are in
  `Assumptions.lean`, `Assumptions/AdviceString.lean` and `Premises.lean` beside them.
- **The properties of the code**: `Security/Properties/NetworkCertifier/LinkWindow.lean` (`LinkWindowAcceptMeans`: an
  accepted execution phase was proved, opened to the word 0b11, certified on both ends and synced within Σ_sync, and its
  public outputs are verdict 1 and deadline 1), `NetworkRecords.lean` (`AcceptMeans`), `NetworkPeriod.lean`
  (`ClockKBoundsCount`, `RowsKBoundsCount`), `NetworkCertifier.lean`, `CertifierDevice.lean` and `CertifierNetwork/`.

What a link's timing can carry, on accepted runs. T ticks an execution phase, r slots a tick, W execution phases, n steps,
D jitter classes, Σ_sync the public clock-sync set and A_c the content advice:

| Quantity | Bound | Guarantee |
|---|---|---|
| Free slot choice | `2^(T·r)` occupancies | `CardFree` |
| Left-packed | `(r + 1)^T` occupancies, attained by the FIFO certifier | `CardLeftPacked`, `EncardFifoGrids` |
| Exactly M released | at most `binom(T + M − 1, M)` | `CardCountsEqLe` |
| At most M available | at most `binom(T + M, M)` grids | `EncardFifoGridsLeOfAvailable` |
| With clock-sync advice | `A_c + W·log2 #Σ_sync + n·log2 D` bits | `EncardAccSeqsLeSync`; at constant rate `EncardAccSeqsLeConstantRate`, over the first m execution phases `EncardAccSeqsLeConstantRateWindows`, per execution phase `EncardAccSeqsLeConstantRatePerWindow`; for a decoder of the wire under W2 and J, `EncardDecodableLeConstantRate` and `EncardDecodableLeConstantRatePerWindow` |
| A fixed clock, under J (D = 1) | 0 bits of timing | `EncardAccSeqsLeFixedClock`, `EncardAccSeqsLeConstantRateFixedClock` |
| The advice string, under a precommitted, deterministic registered compiler | `Σ_w b_w + min(W, m)·log2 #Σ_sync + n·log2 D` bits, `b_w` the public cap charged to execution phase w | `EncardAccSeqsWithinLeConstantRate`; for a decoder of the wire under W2, J and the compiler's two premises, `EncardDecodableWithinLeCompiled` and `EncardDecodableWithinLeCompiledPerWindow`; with advice v1's charge, which needs no premise on the charge, `EncardAccSeqsWithinLeConstantRateV1`, `EncardDecodableWithinLeCompiledV1`, `EncardDecodableWithinLeCompiledV1Counted` and `EncardDecodableWithinLeCompiledV1Whole` (with their per-execution-phase forms), over `EncardChargedLe`, `EncardChargedLeCounted` and `EncardChargedLeWhole` |
| A sampled audit at δ | `c + log2 #(faulty sets) + Σ_{v faulty} committed bits + n·log2 D` bits, except with probability δ + ε_ks + ε_bind | `EncardAccSeqsSampledLe`, for a layout that fits the circuit (`Layout.Fits`); with `SampledAuditAccepted`, `SampledLeakAccepted` |
| The relay | `log2 #schedules + Σ_c log2(cells_c + 1)` bits besides the data | `WardenViewEqClientView`, `EncardWardenViewsLe`; for a decoder of the client's view under `trusted-host` and `BitsCount`, `EncardWardenDecodableLe` |
| Ingress | `log2(r_in + 1)` bits a tick per link under J (D = 1), statuses complete; content adds `log2 #Cs` | `EncardIngressDecodableLe`, `EncardIngressObsDecodableLe`, `EncardIngressObsDecodableLePerWindow`; with content, `EncardIngressDecodableLeContents`, `EncardIngressObsDecodableLeContents`, `EncardIngressObsDecodableLeContentsPerWindow` |
| A certification period | the leak of its accepted execution phases | `PeriodLeakAccepted`; across execution phases, `HorizonWrongUnitsLe` and `HorizonLeakAccepted` |

Every bound holds on accepted runs only. A refused execution phase may already have carried up to its unbound capacity,
about 8·T·r·B bits (`capacity.unbound_bits`); `capacity` computes each bound.

## Premises

Each is named where a guarantee reads it.
- `trusted-host` (`NetworkCertifier.Premises.DeviceIntact`): the network certifier runs its stated code on a host the
  prover doesn't control, its CPU isolated and its clock monotonic, and the host never restores an earlier copy of its
  state (its LMS leaf state and a sealed pair link's secret among it).
- `certifier-coverage`: every uplink of the spatial unit passes through a network certifier, its parameters fixed
  before the run. `latency-ceiling`: every wire from the audited server to its egress network certifier delivers within
  the stated ceiling.
- The network certifier's own (`Assumptions.lean`): `Regenerates` (W1 and W2), `Padded` (W3), `Packed` and `Allocated`
  (W4), `UntimedStatus` (W5), `FailClosed` (W6) and `JitterBounded` (J).
- The registered compiler's (`Assumptions/AdviceString.lean`): `CompilerDeterministic`, `CompilerPrecommitted`, and the
  charge's `BitsCount`, which advice v1's charge meets with no assumption.
- `cr/sha-512` for the commitments and the record tree, `euf-cma/lms-sha512-n64` for the signatures, `prf/sha-512` for a
  sealed pair link's encryption, and `random-oracle` for D hiding R (below).
- `CoinsConfidential` and `OneClock`, the challenger's two premises, prose only
  (`verity/core/protocols/sampled_proofs/README.md`).
- A served link's counts are proved with no premise; that the counts its ingress derives are the Program's grid rests on
  four: every response is F frames; every request is F_in frames, and every ingress frame is a request's but the
  certifier's salt frame heading each execution phase, which `own` counts; requests are in request-major canonical
  order; and an execution phase's anchors share one T, tick length and declared sync.

That honest serving passes is completeness, checked in Python: calibration, the online prover, advice v0's serving scope
and the proxy's admission rule.

## What the Lean doesn't cover yet

- An advised link's verdict, its view (`advised.view`, from the tie and clock sources) and the ingress tie's verifier
  (`tie_service.IngressTie.verify`) are Python.
- The network certificates reach the verifier in the clear beside the proof system: a provisional exception to the four
  public kinds, not a fifth. A run that proves nothing (a rehearsal, the simulator) is decided in the clear
  (`LinkWindows.unproved_outcome`), never in zero knowledge.
- The clock certificate's opening is checked on the host (`advised.SignedClocks`), not in the proof.
- The relay's cell wiring is opened to the verifier (`warden_rows`), which gives away each connection's first row and
  cell count.
- The ties, the replay, the cache steps and the compile slice are in no proof-system Spec, and the verifier computes the
  ties' maps from the opened advice.
- What the advice-string bound needs beyond the compiler's premises is prose: that every value a compiled circuit reads
  has one checked writer, that every advised execution phase's compile slice is proved, that the advice's digest is
  registered before any coin, and that the charge's circuits compute the Lean's charge.
- A sealed pair link's export charge, at the facility boundary only, rests on the egress premise restated there, which
  `Security/` doesn't state.
- The device's properties start from `init`, not from a restart's `initAt`, and the salt's reveal and the weights'
  commitment are Python. The relay has no executable Lean; the device's models the proxy.
- From the ingress theorems to the K charge, reading `log2 #Cs` as the content bits charged and the coin link's content
  exemption are prose, and no audit reads the coin link's record yet.

## How it works

The grid and the audit:
- **The grid** (`grid`): one link's record for one execution phase, T rows of r slots. Each tick emits exactly r frames
  of B bytes, real frames first in FIFO order and padding after. `pacer` writes each egress tick at a time fixed before
  the tick is made.
- **The schedule** (`schedule`): the Program gives every real frame an anchor (the tick of the request it answers) and a
  logical index, and under the execution phase's declared clock sync (δ, ρ), one of Σ_sync, its grid is determined.
  `run_shaper` runs a certifier over a schedule; `online` is the honest prover's online choice of sync, and
  `calibration` sizes r, Σ_sync and the ingress grid from honest runs.
- **The audit** (`audit`): exact, slot by slot, with record-and-reject. An egress execution phase is accepted when its
  status is complete, every slot equals the Program's grid, and its declared sync is in Σ_sync; `check_ingress_link`
  checks an ingress link's status and parameters. Records are raw, committed by hash with padding elided
  (`commitment`), or certified.
- **The certificate** (`certificate`): once an execution phase on each link, the network certifier signs, under LMS,
  `D = SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R)`, R the record's digest and `salt_c` 64 bytes it samples fresh for
  each record and hands to the prover with the head, as witness only (`CertifiedRecord.salt`). D is a hiding commitment
  to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D. The frames sit in a
  fixed-depth SHA-512 tree. A clock certificate (`CertifiedClock`) signs the sync the certifier ran the same way.

Through the proof system:
- **An execution phase's transfer blocks** (`service`, `head_check`): the network `Spec`, a salted root over the
  certified records, and a registered proof block whose Boolean part is each head's check (its hiding opening, its
  fields, its counts against the public occupancy profile, and its record digest under the certifier's salt against the
  digest its certificate signs). The claim is counts only. `LinkWindows.verify` decides each execution phase on a
  zero-knowledge proof and public inputs alone, and reads no head, salt or sibling; the Python runs beside the Lean, and
  a difference refuses. `LinkWindows.network_verdict` is the AND over a link's execution phases in order, with no ε or
  share credit: a refused, late, missing or out-of-order execution phase refuses the link. Each gate names its
  Definition's circuit digest, which the module's `.circuit-check.json` pins.
- **A served link** (`served`): an egress end with no occupancy profile. `NetworkServedCounts_v1` holds the egress
  head's counts to those its ingress head's counts derive under the declared sync. Each execution phase starts empty and
  must end idle, and the proxy's admission rule (`served.Admission`) keeps an honest link accepted.
- **An advised link** (`advised`): a served link under arbitrary traffic. `NetworkAdvisedCounts_v1` holds the egress
  counts to the releases of the prover's advice, each eligible under the sync the certifier ran, and the ingress counts
  to the advice. The prompt lengths are tied to the certified bytes by the ingress tie's verdict (`tie_service`), from
  the verifier's own tie source; without the clock certificates, a tie or a pinned size class every advised execution
  phase is refused.
- **The ties** (`io_tie`, `io_request`, `io_ids`, `tie_service`): prototypes holding certified cells to committed
  values: each egress cell to the token the Commit sampled (`IoTieBucket`), each request cell to the advice's prompt
  lengths (`IoTieRequest`), and each prompt id to the Commit's ids (`IoTieRequestIds`).

The relay:
- **The relay** (`warden`): the network certifier as a relay that knows bytes only, between any client and any server.
  It forwards each direction's bytes unchanged in B-byte cells, r a tick each way per connection; egress cell i leaves
  at tick s + D + ⌊i/r⌋ from its admission s, and a miss fails the connection closed with a `FAIL` cell that carries no
  reason. It records and certifies exactly the cells it sends, and each failure's reason in a third record, `fails`.
- **Its cells as rows** (`warden_rows`): an execution phase's cells committed one hiding row a tick, padding included,
  under a frame-v3-sha512 root it signs beside a salted digest of the wiring, read by a proof block's `--registered`
  port.
- **The step-indexed scan** (`ingress_scan`, `egress`, `boundary`): one proof block a transfer block.
  `IngressScanBucket` reads the tick's ingress row, the slot state and the Programs' row and computes the next state and
  its egress view; `EgressBucket` computes the tick's egress row from the state, every switch computed and none witness,
  and opens only if the row is the certifier's. Programs start at fixed slots every `EVERY` ticks. `boundary` names the
  template that holds each transfer block (`verity/boundary/v3`), and the verifier's `refusal` checks before it samples
  that the layout fits the circuit (`NetTiming.Layout.Fits`).

Servers and their transfers:
- **The transfer rows** (`transfer_rows`, `out_commit`): every value that crosses a server block's boundary (the
  weights' load, an execution phase's request bytes in and generated bytes out, a wire between servers) committed as a
  registered value under the network certifier's own salts, its root in a head it signs (`TransferHead`);
  `check_registration` and `port` are the verifier's read of it.
- **The blocks** (`blocks`, `verity/blocks/v1`): the proof blocks each server runs and its transfer gates. `check`
  refuses blocks whose servers don't partition the circuit, a boundary row no transfer brings, an unpaired wire, and an
  inference block on two servers: a server's ingress and egress carry only its inference blocks' inputs and outputs.
  Which server runs which proof blocks is protocol structure, public and checked in the clear.

The proxy:
- **The active certifier** (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their
  records. With its salt on, each PoUW execution phase gets one fresh salt, sealed in the certifier until `Proxy.reveal`
  opens it once, after that execution phase's last ingress record and within `reveal_deadline`; the weights' root PoUW's
  noise seed reads is recorded before the salt is sampled, signed by the registered prover (`Proxy.commit_weights`).
  Given the challenger's key, both answer only requests the challenger signed for the certifier's run id
  (`serve_reveals`, `RemoteCertifier`). A proxy back from maintenance (`Proxy(start=…, position=…)`) carries nothing
  from before. On a sealed pair link, between two of the auditor's network certifiers, every slot is sealed after its
  record and opened before it (`verity.ml.crypto.aead`). `release` puts the release order on the server's copy of each
  request, outside every cell and record.

The advice and its charge:
- **The advice** (`advice`, `advice_check`): what vLLM's scheduler decided, as one canonical bit string the registered
  compiler reads, refused unless it is a schedule vLLM could have run; `bits` splits it by the earliest execution phase
  that can show each record. `AdviceCheck_v1{L}` is 1 exactly when the committed bits decode to a v0 advice, up to the
  three refusals of `decode` it defers.
- **The compile slice** (`replay_step`, `replay_chain`, `cache_step`, `cache_step_v2`, `compile_slice`,
  `compile_slice_v2`, `compile_slice_v3`): one execution phase's replay of the scheduler as one proof block.
  `ReplayStep_v1` takes the committed queue state, the certified arrivals and the decision bits to the step's rows and
  next state, `replay_chain` hands the state on, and `CacheStep_v1` and `CacheStep_v2` check the prefix cache's hits on a
  table and on vLLM's block pool. `CompileSlice_v3` computes the served rows from the advice, so its only free input is
  the advice, and its charge is that input's width (`free_width`).
- **The served rows** (`served_rows`, `served_tie`, `served_statics`): the step rows a served execution phase ran,
  registered as hiding commitments before the sample, one row a row group of G steps. `ServedRowsTie_v2` holds them to
  the positions the steps' Commit holds, and `ServedStaticsTie_v1` to the RoPE statics of the Programs they ran.
- **The charge** (`arith_encode`, `charge_chain`, `window_charge`): advice v1's arithmetic coder as Boolean
  Definitions. `ChargeCount_v1` counts an execution phase's charge from its decisions, `ChargeLink_v1` holds it from a
  carry-in row to a carry-out row, chained as hiding commitments, and `ChargeCount_v2` and `ChargeLink_v2` charge the
  whole execution phase's record (`Window`) with its hits and order entries.
- **The coin link** (`coin_link`): the one path an execution phase's live coins take from the challenger to the server it
  audits, re-cut into cells and re-clocked on the certifier's own grids both ways. It exempts only the verified,
  expected, once-only coin openings and counts everything else; `benchmarks/live_challenger/coin_link.py` runs it.

## Core API

~~~python
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min execution phases of 100 ms ticks
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per execution phase
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

The difftests that hold this package to the certifier program, each run from `lean/lean-audit.json`:

~~~text
lake exe certifier-difftest INSTANCES PYTHON N          # on what lean/scripts/difftest_vectors.py writes
lake exe certifier-charge-difftest INSTANCES PYTHON N   # advice v1's charge, lean/scripts/charge_vectors.py
lake exe certifier-window-difftest INSTANCES PYTHON N   # its whole execution phase, lean/scripts/window_vectors.py
~~~

It is pure Python and imports only the standard library and `verity`. The signature scheme is a parameter: the
certifier's `certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/certifier`.
