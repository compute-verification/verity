# Network certifier (`verity.core.protocols.certifier`)

A spec-level reference for network accounting's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network certifier's signature over the record's SHA-512 digest hidden under a
  64-byte salt the certifier draws fresh for each record, with the window's frames in a fixed-depth SHA-512 tree), and
  the ingress links' status check;
- one link-window's certificates through the proof service (`service`): the network `Spec`, a salted root over the
  certified records, and the check, whose Boolean part is each head's check (`head_check`: the head's hiding opening,
  its fields, its counts against the public occupancy profile, and its record digest under the certifier's salt
  against the digest its certificate signs), with a host-side verifier of record (no proof of the
  check yet) that compares no grid in the clear. Its claim is counts only: on every link-window the head check holds
  the egress head's counts to the profile (and the ingress head's, where that end declares one), so the link's timing
  is at the profile. Holding the frames, the rest of the egress grid, is N3's, which is a design and the simulator's
  behaviour (`benchmarks/network_sim`), not a protocol: a prototype in the clear, in no proof-service Spec, so the claim
  stays counts only until N3 is a proof-service unit with Selection `All`. The network verdict of record is
  `LinkWindows.network_verdict`, the AND over the link's link-windows with no ε or share credit: every window of the
  egress profile registered in order with an accepted outcome, and a rejected, late, missing or out-of-order window
  rejects the link;
- a served link's counts (`served`, `LinkWindows(served=ServedProfile(F, F_in, windows, own))`), whose egress end has no
  occupancy profile: N2's served unit, `NetworkServedLinkWindow`, adds a Boolean Definition,
  `NetworkServedCounts_v1` (678,795 ANDs at T = 600, F = 1024, F_in = 1, R = 6, R_in = 1), that holds the egress
  head's counts to the ones its ingress head's counts derive under the window's declared sync, the advice the audit
  already charges at log2 #Σ_sync bits a window. Each window starts empty and must end idle, so nothing carries
  between windows, and a request whose frames or response straddle a window's end refuses the window. The claim is
  counts only and rests on no premise. That the derived counts are the Program's grid rests on four: FIXLEN (every
  response F frames), a fixed request size (every request F_in frames, every ingress frame a request's but the
  certifier's salt frame heading each window, which `own` counts), a
  request-major canonical order, and one T, τ_b and declared sync for a window's anchors (`PROTOCOL.md`, "A served
  link"). The proxy's admission rule (`Proxy(served=…)`) admits a request only while its window still ends idle, so
  an honest served link is accepted. An egress end with neither profile gets no verdict (`ValueError`, failing
  closed);
- one vLLM v1 scheduler step (`replay_step`): `ReplayStep_v1`, a Boolean Definition with `reference` its spec, takes the
  committed queue state, the certified arrivals and the committed decision bits to the step's rows (request, kind,
  start, n) and the next state, under sync or async scheduling and with `CacheStep_v1`'s walk entries, and refuses
  what its size class or the prefix layer's counts can't express. It is the compile unit's core, between the committed
  decisions and `compile_steps`' Programs, and is held step for step to the replay
  (`benchmarks/network_e2e/encodings/replay_vectors.py`); it is in no proof-service Spec yet. It takes the circuits
  and their evaluator from `verity`;
- the I/O tie (`io_tie`), a design (`PROTOCOL.md`, "The I/O tie") and its prototype on a fake upstream, in no Spec:
  every cell of an egress link-window is opened, each a pad or the canonical event of one token id on the stream the
  warden set, and each is the output the advice's map names at its place (`window_map`, from advice v0's outputs and
  release buckets; the verifier computes it in the clear from the opened advice, the named gap, until the map is read
  in the proof under the advice's hiding root), with the id the Commit sampled for it (the port: one hiding row of
  ids a step, opened under the Commit's root). The warden commits each bucket's cells as one hiding row. It has the
  reference and the unit's Boolean Definition (`IoTieWindow`: the cells' root by `NetworkRowsRoot`, the port's rows by
  `IoTiePort`, then `IoTieContent`: `IoTieCell` on every cell, the prover's Beneš routing of the port's entries
  `IoTieRoute`, `IoTieMatch`);
- the capacity counts;
- the calibration of honest runs;
- advice v0 (`advice`): what vLLM's scheduler decided in a session (each request's lengths and finish, each step's
  rows with their chunks, prefix-cache hits and the KV they reuse, resumes, and each kept output's egress bucket) as one
  canonical bit string the compiler reads, refused unless it is a schedule vLLM could have run, with its bits split by
  the earliest egress window that can show each record (`bits`). v0 is the committed form; the charge is ENC-CODE's
  compressed length of the advice;
- advice v0's validity (`advice_check`): `AdviceCheck_v1{L}`, the compile unit's first part as a Boolean Definition, 1
  exactly when the committed bits (at most L) decode to a v0 advice, up to the three refusals of `decode` it defers
  (unique ids, a source's KV computed there, release order per request); its AND count at each L is `size(L)`. From
  `verity` it takes the circuits and their evaluator;
- the online prover (`online`): each window's clock sync declared at its first bucket from what it had seen `lag`
  buckets before;
- the prefix cache (`cache_step`): `CacheStep_v1`, a Boolean Definition of one scheduled row's pass over a table of
  whole cached units, which checks the row's hit (the longest cached prefix while no block can have been evicted, a
  cached prefix past that) and owns or appends the units the row computes (`PROTOCOL.md`, "Advice v1"). It takes the
  circuits and their evaluator from `verity`;
- advice v1's arithmetic encoder (`arith_encode`), Boolean Definitions each with its reference as spec. The counting
  variant, which the composition takes: `ArithCountStep_v1{K, BIN, CW}` runs ENC-CODE's 32-bit coder forward on one
  symbol and counts the bits it puts in the window's segment, `ArithCountFinish_v1{CW}` adds the termination and
  holds the window to its declared charge, and `BookKT_v1{C}` is `code_v1`'s KT estimate for a bit's context. `ChargeCount_v1{N}` composes them into one window's charge from its decisions, the compile
  unit's one call a window. The context and the frequencies are the circuit's, from the replay's position and the
  registered book's counts, never witness. The bit-checking variant, the alternative: `ArithEncodeStep_v1{K, BIN}` and
  `ArithEncodeFinish_v1` check each output bit against a committed segment in the decoder's order, with
  `KTContext_v1{C, NW}` ENC-REPLAY's scaled estimate. It is held symbol for symbol to `code.py` and `code_v1` on advice
  v1's windows and on ENC-DERIVED's still-sent fields (`benchmarks/network_e2e/encodings/arith_vectors.py`), and it is
  in no proof-service Spec yet (`PROTOCOL.md`, "Advice v1"). It takes the circuits and their evaluator from `verity`;
- the charge's carry chain (`charge_chain`): `ChargeCount_v1`'s carry committed from window to window as hm96 rows,
  window 0's the book's priors under a public salt, with the verifier's check of how the windows join (`check_chain`)
  and `ChargeLink_v1{N}`, the charge from the carry-in row held to the carry-out row (`protocol/charge-chain.md`). It
  takes the circuits, their evaluator and hm96 from `verity` and the produced rows from sampled proofs;
- the whole window's charge (`window_charge`): `ChargeCount_v2{N, H, O}` and `ChargeLink_v2{N, H, O}` charge advice
  v1's window with its hits and order entries and their counts, coded as `code_v1` codes them, with the order's
  position and rank added to the carry and chained under the value `window-charge` (`protocol/charge-chain.md`, "The
  whole window"). It takes what `charge_chain` takes;
- the active warden (`warden`): a relay that knows bytes only, between any client and any server. It forwards each
  direction's bytes unchanged, in B-byte cells, r a bucket each way per connection, and egress cell i goes out in
  bucket s + D + ⌊i/r⌋ from its admission s. A miss fails the connection closed with a `FAIL` cell that carries no
  reason. It records and certifies exactly the cells it sends, and each failure's reason in a third record, `fails`.
  `PROTOCOL.md`'s "The active warden" has its leak bound, and `benchmarks/network_traces/warden_serve.py` runs it;
- the warden's cells as rows (`warden_rows`): each link-window's cells committed one `hm96-sha512/row/v2` row a cell,
  padding and empty slots included, under a frame-v3-sha512 root the warden signs with the certifier's signer. The
  registration is what a proof unit's `--registered` port reads, so the sender's unit opens a cell as its output and
  the receiver's as its input (`PROTOCOL.md`, "The warden's cells as rows"; `benchmarks/network_traces/warden_zk.py`);
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.
  With its salt on (`Proxy(salt=active.fresh_salt)`, off by default), each PoUW window (`pouw_window` certifier
  windows) gets one salt of `salt_bits` bits (512 by default), carried into the block only. The ingress record holds a
  salt frame with a salt-free placeholder, so every record is a function of the traffic alone; the salt sits in the
  certifier's sealed record cell `_salts[p]` until `Proxy.reveal(p, registration)` opens it on the PoUW window's
  registration after its last ingress record is published, once and only within `reveal_deadline` of the window's
  end on the certifier's clock, and `active.check_salts` reads the salt frames. The weights' root PoUW's noise seed
  reads is recorded before the salt's draw, signed by the registered prover (`Proxy.commit_weights(p, root,
  signature)` under `prover_key`, the sealed cell `_weights[p]`), and given on the same conditions
  (`Proxy.committed(p, registration)`). With `Proxy(challenger_key=…, signer=…)`, both or neither, the challenger's key
  from the auditor's deployment record, the two calls take only a request the auditor's challenger signed under LMS
  for the certifier's key id and run (`Proxy.run_id`, 32 bytes drawn at construction) and answer an `active.Revealed`
  the certifier signed, which names both; `active.serve_reveals` serves them over the network, one JSON line each way,
  to the challenger's `active.RemoteCertifier`, given the run id by the deployment record, which returns the value
  once it verifies for that run. Without the keys they are the in-process calls, for the auditor's side only. A proxy
  back from maintenance (`Proxy(start=k, position=p)`, off by default) serves from window k on, its first egress frame
  the Program's at position p, and carries nothing from before: a window it was down for has no record. On a served
  link (`Proxy(served=ServedProfile(…))`, off by default; with the salt, `own` 1 at a PoUW window of one certifier
  window) its ingress admits whole requests by the admission rule (`served.Admission`). On a sealed pair link
  (`Proxy(pair=active.Pair.of(…))`, off by default), between two of the auditor's certifiers, every slot of the link is
  sealed after its record and opened before it (`verity.ml.crypto.aead`, SHA-512 counter mode then HMAC-SHA512, a
  nonce per slot), so the records are the unsealed run's and a unit that doesn't open fails the link closed.

It is pure Python. It imports only the standard library and `verity`: for the record digests' `tagged_sha256` and
`tagged_sha512` and `canonical_json_bytes`, in `service` the proof service, the hm96 leaves and the check's
Definition, and in `active` the proof service's registration and receipt, which `reveal` reads, and the pair link's
AEAD (`verity.ml.crypto.aead`). The certificate's signature scheme is a parameter: the certifier's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
From `verity`, `head_check` takes the circuits, their evaluator and the commitment gadgets, and `served` the circuits
and their evaluator.
`PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`Security/`, area `NetworkCertifier`).
`lean/` is the certifier program, the Lake package `certifier_program` (Lean core only): List forms of the analysis's
definitions, proved equal to them in `Security/Proofs/NetworkCertifier/Code.lean`, and the difftest that holds this
package to them (`lake exe certifier-difftest INSTANCES PYTHON N` on what `lean/scripts/difftest_vectors.py` writes; the audit runs it
from `lean/lean-audit.json`). Advice v1's charge has its own: `lake exe certifier-charge-difftest INSTANCES PYTHON N` on
what `lean/scripts/charge_vectors.py` writes from `arith_encode.reference_charge`, and its whole window's
`lake exe certifier-window-difftest INSTANCES PYTHON N` on what `lean/scripts/window_vectors.py` writes from
`window_charge.reference_window`.

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

## Not yet zero-knowledge

Every protocol is to be zero-knowledge, and the network audit isn't yet. Its claim today is counts only: N2 holds each
window's per-bucket counts to the occupancy profile on every link-window (on a served link, to the ones its ingress
counts derive under the declared sync), beside the proved timing bound, while N3, the
content check, is the design (Daniel's provisional call on ASK-70, 19 Oct). N2 no longer compares the grid in the clear
(`LinkWindows.verify` runs `check_link(grid=False)`, NET-38). Nothing may call the count check zero-knowledge until
this is closed:
- **N2's unit isn't proved yet.** `LinkWindows.verify`, the verifier of record, runs `head_check.reference` (and on a
  served link `served.reference`) on the host over the opening, so it reads the heads and the certifier's salts.

A proof of the unit is held to its circuit already. Each gate names its Definition's circuit digest
(`Gate[NetworkHeadCheck_v2{…}@<digest>]_v2`, `served.gate` likewise), as `head_check.circuit-check.json` and
`served.circuit-check.json` pin it and circuit-check holds the pin to the Definition. So the unit's Program, which the
spec and each registration bind, binds both circuits, and `LinkWindows.verify`, given a proof's findings, refuses one
that names any other Program, or a unit of a size class no sidecar pins (`LinkWindows.proved`).

The certificate no longer signs a hash of witness (NET-42). The certifier signs `D = SHA-512("verity/net-cert/v1" ‖
salt_c ‖ R)`, R the head's record digest and `salt_c` 64 bytes the certifier draws fresh for each record
(`certificate.Issuer`'s `coins`) and hands to the prover with the head, as witness only (`CertifiedRecord.salt`). D is
a hiding commitment to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D.
Each end's head check holds R to the head it opens and D to R under `salt_c`, so once N2's unit is proved the
certificates' check reads nothing of the heads; until then the verifier of record checks that link on the host too,
over the opening (`CertifiedRecord.verify`).

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/certifier`.
