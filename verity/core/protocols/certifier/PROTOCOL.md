# The network certifier

The **network certifier** is the auditor's device on a boundary link of a spatial unit (the machines behind network
certifiers). It shapes each link into a fixed grid of buckets and frames, records exactly that grid and one status per
sampling block, and once per sampling block signs a **network certificate** over a salted digest of the record. What it bounds
is how much a covert sender can move across the boundary through **when** frames leave, once the content and order of
every frame are fixed by the declared Program.

- **Trust.** The certifier is auditor-trusted: it runs on a host the prover doesn't control, and the auditor trusts it
  to run the grid and to sign what it ran. That trust, with the custody of its keys and of its LMS leaf state, is the
  named assumption `trusted-host`. The challenger's live coins reach it on two more premises, stated with the coins'
  path in `benchmarks/live_challenger/auditor.py` and `verity/core/protocols/sampled_proofs/README.md`:
  `CoinsConfidential` (the challenger's coins reach the certifier only inside auditor-trusted hosts, `trusted-host`, and
  no prover-controlled host reads a coin before the certifier admits it) and `OneClock` (the challenger's release times
  and the certifier's ingress ticks are read on one clock; on two devices, within 1.03 s of each other). No Lean
  statement reads either yet.
- **What it signs.** An LMS signature (SHA-512, n = 64) over a message binding its key, its session, the sampling block and
  the record's digest hidden under a salt the certifier draws ("Network certificates").
- **Public parameters.** Each direction's grid (T, τ_b, r, B and Q, "The certifier (`grid`)"), the set of clock-sync
  values Σ_sync, the certifier's key id and session, and the sampling blocks audited.

Where each part is stated in Lean:
- the timing analysis `NetTiming`, its guarantees and its modelling choices: `Security/Definitions/NetworkCertifier` and
  `Security/Proofs/NetworkCertifier` (whose `README.md` has the modelling choices), with each bound's guarantee in
  "Capacity";
- the certifier program (`lean/`, the Lake package `certifier_program`, Lean core only): the grid, the schedule and the
  charges as code, proved equal to the analysis's definitions and held to this package by its difftests;
- the record and certificate encodings: `NetworkCheck.Certificate` (`verity/core/protocols/network/lean`);
- the audit: `NetworkCheck.Check` (`verity/network-certificate/v4`) and `NetworkCheck.Records`
  (`verity/network-records/v3`);
- one sampling block's verdict of record: `NetworkCheck.LinkWindow`;
- a period, onsite's communication component: `NetworkCheck.Period` (`verity/certifier/v0`).

## Zero knowledge: what ties the checks, and the named gap

N2's count check reads the head leaf of N2's commitment and holds the head's counts to the occupancy profile. Another
check that reads the same leaf is tied to N2's only under three conditions, each of which holds:
1. The root is registered before either check's coins or draws (the firewall's registration, `LinkWindows.commit`).
2. The leaf index, 2 + d, is fixed by the sampling block's direction (`service.HEADS`), which is public structure, not
   something the prover chooses.
3. SHA-512 is collision resistant, as hm96 rows in frame-v3-sha512 trees already assume.

N2 compares no grid, and the network claim is counts only. N2's verifier of record
decides each sampling block on a `--zk` proof of its registered unit and public inputs alone, and reads no head, salt or
sibling ("Its verifier of record" below). What stays open is in the README: the certificates reach the verifier in
the clear beside the proof system, a provisional exception to the four public kinds, and a run that proves nothing is
decided in the clear.

A proof is held to the unit's circuits: each gate names its Definition's pinned circuit digest, so the unit's Program
binds them, and the verifier refuses a proof of any other Program ("The check" below).

The certificate's signature reads no hash of witness. The certifier signs
`D = SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R)`, R the head's record digest:
- `salt_c` is drawn by the certifier (`certificate.Issuer`'s `coins`), 64 bytes, fresh per record, and handed to the
  prover with the head as witness only (`CertifiedRecord.salt`, in the record leaf after the head; the head leaf is
  the head alone);
- each end's head check holds `R` to the digest of the head it opens and `D` to `R` under `salt_c` (part 4 below);
- the LMS check (`Certificate.verify`) reads only the public fields (key id, session, link, direction, window) and
  `D`, a hiding commitment to `R`. The in-the-clear check (`LinkWindows.clear`, a debug comparison) also checks `D`
  against the head and `salt_c` over the opening (`CertifiedRecord.verify`).

## The certifier (`grid`)

Each boundary link has an active certifier with fixed parameters (W1). The protocol programs them from a security
parameter, and the certifier only executes them:
- T buckets per window;
- the bucket width τ_b, 100 ms by default;
- the allocation r, in frames per bucket;
- the frame size B, in bytes;
- the queue capacity Q.

On each link:

- **Constant rate, with a cap, left-packed (W4).**
  - Every bucket emits exactly r frames: `n_t = min(queue, r)` real frames in slots 0 … n_t − 1, in queue order, then
    padding. The I/O tie reads the active warden's grid instead, each live connection's r slots of a bucket by
    ordinal ("The I/O tie", (c)).
  - The **occupancy** is (n_0, …, n_{T−1}), the number of real frames per bucket.
  - A colluding receiver can tell padding from a real frame, so occupancy is the timing channel.
- **FIFO, with the queue carried across windows.** The release count is Lean's `released`, and the grid Lean's `fill`.
- **Fail-closed (W6).**
  - A fault is an overflow beyond Q, a frame of the wrong size, or a reported one (monitor unavailable, drop, delivery
    refused, coverage never established).
  - After a fault, the rest of the window is padding, the queue is dropped, and the first reason is recorded.
- **The record is exactly (grid, status) (W5).** The status is one untimed symbol per sampling block: complete, or not
  established with a reason. It carries no bucket index, time or count.

## The published record (`commitment`)

A raw grid is 2.3 GiB per sampling block at r = 1,024, B = 4 KiB and T = 600. The certifier publishes one digest and the
status instead, which is 32 bytes per sampling block, or about 46 KB per link per day.

- **Padding is elided.** Left-packing (W4) fixes a row by its count n and its n real frames, so a row's digest covers
  only those: `tagged_sha256("verity-network-warden/row/v1", u32be(n) ‖ f_0 ‖ … ‖ f_{n−1})`. A row that isn't
  left-packed has no encoding.
- **The record digest** is `tagged_sha256("verity-network-warden/record/v1", …)`. It covers:
  - the canonical JSON of the link, the window, (T, r, B, τ_b) and the status;
  - then the T row digests.

  `tagged_sha256` is `verity.ml.commitments.identity`'s.
- **The check** recomputes the digest from the Program's grid, with a complete status, and compares. Under `cr/sha-256`,
  equal digests mean equal records, so the audit stays exact. The verifier never needs the raw grid, because the
  Program fixes every egress frame.
- **The status in the clear** is only for classifying a violation. The digest binds it too, so a published status that
  disagrees with the digest is a mismatch.

v1 stays as it is, and readable: `CommittedRecord` and `record_digest` are unchanged, and the audit still takes them.

## Network certificates (`certificate`)

**The trust change.** The **network certifier**, the device on a node's uplink that shapes each link into the grid
(`grid.Certifier`), is auditor-trusted: the auditor trusts it to run the grid and to sign what it ran. A raw or
committed record reaches the auditor as published. Instead, once per sampling block, the certifier signs a **network
certificate**: a signature over a hiding digest of the window's record, salted by the certifier (`certificate.Issuer`).
The record head and the salt are witness. They enter proofs, which open frames of the record in-circuit, and never
reach the auditor in the clear, so the record's own hashes are unsalted. The audit checks the certificate against the
certifier's public key. With an honest certifier every certificate verifies, and the audit's verdicts are those of the
raw records.

**The record (v2)** is still exactly the grid and one status (W5), with padding elided: a fixed binary head
(`RecordHead.encode`: the sampling block, T, r, B and τ_b, the status, the deadline bit, the parameters digest, the
occupancy n_0 … n_{T−1}) ending in the root of a fixed-depth SHA-512 tree over the window's real frames. Its layout,
the frame tree and the record digest are `certificate.py`'s module docstring, stated in Lean by
`NetworkCheck.Certificate` (`RecordHead.encode`, `RecordHead.digest`, `parametersDigest`, `framesRoot`) and held to the
Python by `NetworkCheck.Difftest`.

**The signed digest** hides the record digest R under `salt_c`, 64 bytes the certifier draws from its own
randomness, fresh for each record (`Issuer`'s `coins`), never the prover's:

    D = SHA-512( "verity/net-cert/v1" ‖ salt_c ‖ R )     (`certificate_digest`, Lean `NetworkCheck.certificateDigest`)

The certifier hands `salt_c` to the prover with the head, as witness only (`CertifiedRecord.salt`). R is 64 bytes, so
D fixes R up to a SHA-512 collision (`cr/sha-512`). D hides R as SHA-512 under a secret, uniform 64-byte prefix does,
which rests on `random-oracle`; no Lean guarantee states the hiding.

**The certificate** is D and a signature over a 64-byte message that binds D to the key, the certifier session and the
sampling block, so a certificate can't be replayed onto another window, link, direction, key or session:

    message = SHA-512( [tag] ‖ I ‖ [ [session] ‖ [link] ‖ [direction] ‖ u64be window ] ‖ [D] ),
              tag = "verity/network-certificate-message/v2", direction "egress" or "ingress", [x] = u64be len(x) ‖ x

This is LMS's network-certificate role digest (`verity.ml.crypto.lms.certificate_message`, its tag `lms.ROLES`'
`network-certificate`), the message its in-proof verifier takes, and Lean's `NetworkCheck.messageInput` hashed. I is the
certifier's 16-byte LMS identifier. `session` is the certifier session's id: 32 bytes (`lms.SESSION_BYTES`) the
challenger draws for the session and issues to the certifier, and the audit takes from its parameters, as it takes the
key; the record head and `service.encode` don't carry it. The tag is the message's own domain separation, distinct from
the ids of the checks that read it: the network certificate check `verity/network-certificate/v4` and the audit over
every record `verity/network-records/v3`. The scheme is LMS over SHA-512 (`lms.DEPLOYED`: n = 64, w = 1,
h = 20, the whole SHA-512 output, not truncated), and its unforgeability is `euf-cma/lms-sha512-n64`.
The certifier's `Issuer` holds it as a `Signer` (`key_id`, `sign(message)`; `certificate.LmsSigner`) and the audit as a
`PublicKey` (`key_id`, `verify(message, signature)`; `certificate.LmsKey`), so the scheme is a parameter of both. `LmsSigner`
keeps its next unused leaf and the session ids it has started in an `lms.LeafState`, which it holds from its first use on.
The `Issuer` starts its session on the signer (`Signer.start`) before its first signature, and the state refuses a session
id it has already started, so the device runs once per session. A signer whose state is missing, behind the leaves its
journal has used or its `floor`, or held by another signer, refuses (`SignerRefused`), and so does a refused start. The
certifier then publishes nothing for the sampling block (`Issuer.refused` lists it), which fails closed as a `no-verdict`; it
never signs with a reused leaf, nor in a session it has run before. Both rest on `trusted-host`, where the state
the device keeps on its host is its next LMS leaf and the session ids it has started.

**The check.** The certificate names the head's sampling block, D is the head's record digest under `salt_c`, and the
signature verifies (`CertifiedRecord.verify`; the signature alone is `Certificate.verify`, which reads the public fields
and D). On egress the audit recomputes the record digest from the Program's grid, with a complete status, the deadline
bit set and the parameters digest of `params` and Σ_sync, and compares it with the head's. Under `cr/sha-512`, equal
digests mean equal records, so the audit stays exact. A certified record rests on `cr/sha-512`,
on the signature's unforgeability, and on an honest certifier; not on `cr/sha-256`.

## Through the proof system (`service`)

One sampling block's certificates are registered through the proof system (`verity.core.protocols.sampled_proofs.service`)
as a `Spec` of its own (`service.LinkWindows`, one per link):
- **Committed:** one value, `certificates`, of four leaves: the sampling block's egress and ingress certified records,
  leaves 0 and 1, each `hm96-sha512/row/v2` over `service.encode` (the head, the certifier's salt, then the
  certificate), and their heads,
  leaves 2 and 3, each over the head alone (`RecordHead.encode`). Every leaf has its own fresh salt, in one
  `frame-v3-sha512` tree under one-stage's served domain for the run, the link and the window. A proof that reads only a
  head opens its head leaf in two or three SHA-512 compressions, where the record leaf would cost the certificate's 272
  more. The firewall keeps the records and salts (`Opening`). The challenger receives one salted root and never a
  certificate, a certifier's salt or a head.
- **Selection:** `All`. A registration is one sampling block, one proof block, so an accepted outcome is the exhaustive fact.
  The record's `window` provenance is exactly `{"kind": "served", "window": w}`, and the challenger holds w to the
  registration's order (`LinkWindows.register`): the receipt keeps the order, and the order is the sampling block.
- **The network verdict of record** is `LinkWindows.network_verdict`, the AND over the link's sampling blocks with no ε
  or share credit. Over the stream's registrations and outcome records, it accepts only when, for every window w of
  the egress profile (on a served link, of its served profile), the w-th registration is window w and its outcome of
  record accepted. A rejected or late window,
  a registered window with no outcome yet, a gap in the registrations (a window registered out of order or never, or a
  run that stops short of the profile) and a registration past the profile each reject the link, and `refused` lists
  their windows. A run's network verdict is the AND over its links'. The datacenter model (`benchmarks/datacenter`, a
  Python model, not a verifier of record) builds no `LinkWindows` outcomes, so its account doesn't read this
  aggregate. Its `Account.sound` is the same AND over its own port records: false on any refused port-window, each
  charged at its link's line rate (`refused_bits`).
- **Public:** the population, the window, the value's leaf count, domain and salted root, the parameters digests, the
  charge (log2 #Σ_sync per window, 7 bits at 16 × 8) and the occupancy profile (`LinkWindows.occupancy`: for each end,
  each window's T counts as the Program and the placement fix them, or none for an end the Program reads rather than
  predicts), all bound by the spec's context with the link, the run,
  the certifier's key id and its session id (`certifier_session`), and the check's outputs `verdict` and `deadline`. A rejected sampling block shows only the
  service's `reject`; the violation's kind stays in the verifier. A registration with any other field, a second root,
  a root entry with another key, or anything more in `window` is refused. A served link has no egress profile, and its
  sizes and each window's declared sync are public instead (`PUBLIC_SERVED`, "A served link" below).
- **The check** is `LinkWindows.unit`, `NetworkLinkWindow` v5{EGRESS, INGRESS}, over the two certificates' signed
  digests (``egress``, ``ingress``) and each end's head check's other parameters but its direction, which the unit
  fixes (egress 0, ingress 1). Its output is two bits: ``verdict``, the AND of the two head checks, and ``deadline``,
  the AND of the two heads' deadline bits. The certificates are not in the unit: the verifier checks them on their
  public fields (`LinkWindows.certified`: each verifies under the certifier's key in its session,
  `LinkWindows.certifier_session`, over its public fields and its signed digest, and names the sampling block). A proof
  states the unit on rows ``pub`` (its public parameters), ``wit`` (each end's head, leaf salt, siblings and
  certifier's salt) and the output row ``o0``, one u16 word (`LinkWindows.layout`, `service.OUT_BITS`), its circuit
  `LinkWindows.proof_unit`. The head check is a Boolean Definition,
  `head_check.NetworkHeadCheck_v2{L, T, COUNTS}` (EGRESS and INGRESS are each end's statics), with one output, the ok
  bit, whose `signed` input is its end's signed digest. It opens the head at leaf 2 + direction under the registered
  root, a hiding read (`NetworkHidingRead`), and accepts when the head's link, direction and window are the
  sampling block's, its status is complete, its T, r, B, τ_b and parameters digest are the end's declared grid's
  (`NetworkHeadFields`), on an end with an occupancy profile its T counts are the profile's row for the window
  (`NetworkHeadCounts`; a window past the profile has no row and is a reject), and the head's record digest R
  (`NetworkRecordDigest`) under the witness `salt_c` is the signed digest (`NetworkSignedDigest`), so the certificate
  signs the head the check opened. In the unit's Program each head check is one gate that names the Definition and
  its circuit digest, `Gate[NetworkHeadCheck_v2{L,T,COUNTS}@<digest>]_v2` (`head_check.gate`): the commitment gadgets'
  Boolean bodies encode to about 140 MB, and the service encodes the Program on every registration. The digest is
  core's `program_digest` of the Definition's Program (`head_check.circuit_digest`), the one a C-Flock statement names
  a template by, read from `head_check.circuit-check.json`'s ``digests``, which circuit-check holds to the Definition
  (a lowering pin). The Program's digest is the check's, so the spec, the stream and each registration bind to it, and
  through the gates to the circuits: a proof of another circuit with the same interface names another Program. A size
  class no sidecar pins has a gate naming no digest (`@unpinned`), and no proof of its unit is accepted until a root of
  that class is pinned.
- **Its sizes** (`head_check.size`, L = 9): the count check is an equality of 32·T bits, 32·T − 1 ANDs (19,199 at
  T = 600); the fields are 736 + 8·L − 1 (807); the opening is 775,077 at T = 4 and 1,876,070 at T = 600, nearly all
  of it the head row's hm96 digest; the signed digest is 290,246 at T = 4 and 1,333,292 at T = 600, the record digest's
  ⌈(244 + L + 4T) / 128⌉ compressions (five and 23), the salted digest's two and an equality of 512 bits. With the
  three ANDs that join the four parts a head check is 1,066,260 at T = 4 and 3,229,371 at T = 600. The record digest
  hashes the head a second time, beside the opening's row digest; a record v3 whose digest is the head row's
  sha512/row/v2 digest would share it and save about 1.2M ANDs a head at T = 600. The count check needs no frame's bytes, so it is the part of the egress grid check
  a proof can make without the content, and the only grid check on ingress. It runs on every sampling block: a
  structural check runs in full.
- **N3, the content check,** a prototype in `benchmarks/network_sim` evaluated in the clear and in no proof-system
  Spec, is retired, superseded by the network certifier's step-indexed scan (`ingress_scan`, `egress`), which
  `benchmarks/network_traces/scan_zk.py` proves in zero knowledge, outside this protocol.
- **Its verifier of record** is `LinkWindows.verify`, which decides on the delivered proof (`service.Delivered`) and
  public inputs alone. It accepts when its own run of C-Flock's Lean verifier (`proofs.verify`, `verify --zk`, on the
  circuit file the auditor committed) accepts every session, the sessions prove the link's registered unit (every gate
  names a pinned digest and every session names the registered Program, the partition object's `program`,
  `LinkWindows.proved`) and the drawn unit; the proof's ``pub`` row opens, under the salt delivered, to the verifier's
  own copy (`LinkWindows.public_arguments`: the registered root, the sampling block's public fields and both signed
  digests) and its ``o0`` row to the output word delivered, whose verdict and deadline bits are both 1; the
  certificates verify (`LinkWindows.certified`); and the window's declared sync is one of Σ_sync. It receives the
  registration, the draw, the certificates' public fields and signatures (from the certifier, beside the proof
  system: the service has no public kind for them), the declared syncs, and from the prover the statement's public
  file, its sessions, the circuit file and the salts of ``pub`` and ``o0`` with the output word; never the opening nor
  ``wit``'s salt. The outcome passes the registered root as the session's. A proof after `proof_deadline` is the
  service's `late`. On a plain, served or network certifier link the verdict of record is the Lean's (`NetworkCheck.LinkWindow`,
  `service.OF_RECORD`), with the Python beside it as the reference; the README has the gap list and the replay. The verdict reads no frames, so its claim is counts only: the egress head's counts are the
  profile's (and the ingress head's, where that end declares one), which puts the link's timing at the profile. The
  egress grid is the head's counts filled with its frames in order, and the head's digest binds both, so the frames
  are the rest of the grid, and nothing in this protocol holds them, so it doesn't say the egress grid is the Program's.
  A link whose egress end has neither an occupancy profile nor a served profile has no verdict (`ValueError`, failing
  closed), since nothing would hold its egress timing.
- **The in-the-clear check** is `LinkWindows.clear`, over the opening: the opening reproduces the registered root,
  each end's head check accepts (`head_check.reference`, which the tests hold to the circuit), and the audit accepts
  both records under the certifier's key in its session, comparing no grid (`check_link(grid=False)` on the status,
  the signature and the declared syncs, `check_ingress_link` on the status and parameters); on a served link also the
  served counts' check (`served_check`). It is a debug comparison (`verify(…, opening=…)`, off by default, which
  decides nothing) and the tests' reference. `LinkWindows.unproved_outcome` decides a run that proves nothing (a
  rehearsal, the simulator) on it, its verdict ``"decided_by": "clear"``, never zero-knowledge.
- **The oracle** is `LinkWindows.stand_in`, the in-the-clear verdict: the audit with the Program's egress grid
  compared, and each head's counts the profile's (`LinkWindows.declares`). `clear` accepts every sampling block it
  accepts, and differs only where the record's grid, at the profile's counts, isn't the Program's: in its frames, which
  N2 doesn't hold, or where the declared syncs move the Program's counts off the profile, which N2 holds the counts to
  instead.

What this establishes: the outcome is accept exactly when the Lean verifier accepts a proof of the registered unit
whose public row is the verifier's own and whose output word is 0b11, the certificates verify and the sync is
declared; the outcome record binds the root, the draw and the delivered sessions. The claim is counts only: on every
sampling block of a link whose egress end declares a profile, the egress head's counts are the profile's (and the ingress
head's, where that end declares one), so the link's timing is at the profile. On a served link the egress head's counts
are instead the ones its ingress head's counts derive under the window's declared sync ("A served link" below). An
egress end with neither profile gets no verdict (`ValueError`, failing closed). What it doesn't establish: the head check
leaves the head's deadline byte and frames root free. The unit reads the deadline for its public bit and nothing reads
the frames root, so an accepted outcome says nothing of the frames. The proof opens the head leaves (2, 3)
only: the record leaves (0, 1), each certified record whole, are committed but nothing in the verdict reads them, and
the certificates' signatures are checked on the host from their public fields, not in the proof. One clock per
spatial unit across links is not checked (each registration is one sampling block, alone), and nothing here checks the
frames' consistency with the served computation.

## A network certifier link (`service.WardenProfile`)

Stated in Lean: `NetworkCheck.LinkWindow`'s network certifier form (`Kind.warden`), the plain unit under the active warden's
records, both ends at the network certifier's grid and parameters digest, with no sync read; `service.WardenProfile` is the
Python reference.

## A period, as onsite's communication component (`verity/certifier/v0`)

Stated in Lean: `NetworkCheck.Period` (`verity/core/protocols/network/lean/NetworkCheck/Period.lean`, the
`network-period` executable), whose module docstring has both forms, the clock form and the rows form, their formats
and the bound `K` it publishes.

## A served link (`served`)

A served link's egress counts follow its requests' arrivals and the run-time syncs, so nothing fixes them before the
run and its egress end has no occupancy profile. Declared with
`LinkWindows(served=ServedProfile(F, F_in, windows, own))`, N2 derives them inside the unit instead: on such a link it
holds the egress head's counts to the ones the ingress head's counts derive under the window's declared sync, with no
public egress profile.

- **The unit** is `NetworkServedLinkWindow` v4{EGRESS, INGRESS, SERVED}: `NetworkLinkWindow`'s two head checks
  (neither end with a count check unless its end declares a profile), the window's declared sync as its ``sync``
  parameter, and one gate (`served.gate`)
  naming the Boolean Definition `served.NetworkServedCounts_v1{T, FOUT, FIN, R, RIN, OWN}` and its circuit digest
  pinned in `served.circuit-check.json`, as the head check's gate names its own, over the two heads' counts,
  cut from the heads the head checks open (so witness), and the window's sync (δ, p, q), ρ = p/q. OWN is the ingress
  certifier's own frames at the head of every window's first bucket: 1 with its salt on, else 0. Its one output, the
  ok bit, is 1 when every ingress count is at most R_in, bucket 0's at least OWN, p and q are nonzero, and the egress
  counts are `served.derive`'s, of the ingress counts with bucket 0's OWN own frames taken off:
  - each request's anchor is the bucket whose ingress release completes its F_in frames, as the proxy anchors one;
  - each request is answered by F frames, request after request in canonical order, the k-th frame of a response at
    logical index k, handed by `schedule.handing_buckets` under the sync;
  - the egress certifier releases them FIFO, at most R a bucket, left-packed (`grid.released`);
  - the window ends idle at both ends: the ingress counts sum to a multiple of F_in, and every frame is handed and
    released by bucket T − 1.

  The circuit is a closed form of `derive` (`served`'s docstring), and `served.reference` is its reference, which the
  tests hold to it. Its size (`served.size`) is 678,795 ANDs at T = 600, F = 1024, F_in = 1, R = 6, R_in = 1, and
  688,210 at F_in = 2, R = 16, R_in = 4, beside the head checks' openings (1,876,070 each at T = 600); OWN = 1 adds 63.
- **Public** (`PUBLIC_SERVED`): the items above, with the occupancy profile on ingress only where the placement fixes
  it and none on egress, the served sizes (F, F_in, OWN and the link's windows, fixed before the run), and each window's
  declared sync as an output of the protocol. That sync is the clock-sync advice the audit already reads and charges
  (log2 #Σ_sync bits a window, 7 at 16 × 8): the served counts read it, and nothing else about the run is declared.
- **The verifier of record** is `LinkWindows.verify`, as above: the unit's verdict bit holds the served counts' check
  too, and the verifier's own public row holds the window's declared sync. The in-the-clear check runs `served_check`,
  `served.reference` over the opened heads' counts and the sync. `network_verdict` takes the served profile's windows.
- **Carry-over: none.** Each window's derivation starts with an empty egress queue, no request frame pending and no
  earlier hand, and checks that the window ends the same way. By the check alone, an accepted window's egress counts
  are a function of its own ingress counts and sync. By induction from the run's first window, which starts empty (as
  a certifier back from maintenance does, carrying nothing), on a link whose every window is accepted
  (`network_verdict`'s AND) the Program's run ends every window idle too, so each window's counts are the Program's
  grid under the premises below. It is the smallest sound rule: no state crosses a window's boundary, so no unit needs
  a carry that another window's unit would have to hold it to.
- **Request boundaries.** A request is in one window: its F_in frames are released there, and its F response frames
  are handed and released there. A request whose frames, or whose response, would straddle a window's end makes the
  window refuse, and the next one, whose counts carry frames its ingress counts don't derive.
- **The admission rule** (`active.Bucketer(served=served.Admission(…))`, `active.Proxy(served=…)`) is how an honest
  proxy meets that. It enqueues arrived frames only as whole requests, and only while the window, with the request,
  still ends idle (`served.admits`: the queued frames released at r_in from the next bucket, then `derive` under the
  window's sync); the rest wait, in order, for the next window. The sync is the window's declared one where the proxy
  holds it before the run, else the slowest of Σ_sync (largest δ, smallest ρ): a later hand only lowers the release
  curve, so a window idle under it is idle under any sync of Σ_sync. A served link's ends share T and τ_b (its anchors
  are ingress buckets), its ingress takes the certifier's own frame exactly when the profile counts one (OWN 1: the
  salt on, at a PoUW window of one certifier window, so the salt frame heads every window), and a window must hold
  that frame, one request and its response, or the proxy refuses to start (`ValueError`).
- **Premises** of "the derived counts are the Program's grid", which tie the counts to the served computation; N2's
  claim rests on none of them:
  - FIXLEN: every response is F frames, since every request runs to its `max_tokens` and the server writes one token
    a frame. The lab pads each block's responses to F with empty events.
  - A fixed request size: every request is F_in frames, a power of two, and every ingress frame is a request's but
    the certifier's own salt frame, which heads each window's first bucket when the salt is on (OWN).
  - The Program's canonical order is request-major: each response's frames are consecutive, responses in request
    order. The shaper hands in canonical order, so a server that interleaves responses changes nothing on the wire
    while each frame is ready by its hand.
  - Both ends share T and τ_b (checked by `LinkWindows` and the proxy), and one declared sync applies to all of a
    window's anchors, as each request is admitted in its window.

What this establishes: on every accepted sampling block of a served link, the egress head's counts are `derive`'s of the
ingress head's counts under the declared sync, whatever the frames hold. So the egress timing adds nothing to what
the ingress counts and the charged advice carry, and under the premises it is the Program's. The ingress counts are
the record's own, at most r_in a bucket, as on any end without a profile. What it doesn't establish: the frames
(the salt frame's place among them is the salt audit's, which takes it only first in bucket 0, `check_salts`), and
the premises, which nothing checks.

## The I/O tie (`io_tie`, a design)

N2 holds a window's counts but says nothing of its frames, nor that they carry the tokens the served computation
produced. The I/O tie closes that on a served link's egress. It is one unit per egress sampling block at selection `All`.
It opens every cell of the window and holds each to the output the advice's map names there, and that output's id to
the id the Commit sampled. It is a design with a prototype on the network certifier's own egress (`io_tie`, its cells cut by
`warden.Warden._egress`), in no Spec and unproved. The sampling block's tie is T units, one a bucket row of the network certifier's
registration (WARDEN's v2, a row a bucket), and each outputs 1 exactly when:
1. every cell of its row is one of the network certifier's three: a `PAD` on a stream (no payload; a slot no connection holds is
   B zero bytes, a `PAD` on stream 0), a `DATA` cell (P = B − 9 bytes of payload, one token id's canonical event
   padded with spaces to P) or an `END` (`TRAILER`, the response's last bytes, then zero fill);
2. every cell is its map's: the record's kind on the record's stream q, and a `DATA` cell's id entry e of the
   bucket's port;
3. every opened port row opens at its public step under the root the Commit registered;

and the row is row t of the network certifier's registered root, which the proof system's registered read checks outside the unit.

So a duplicated, dropped, reordered, moved or altered cell, a byte anywhere outside the id, a cell on another request's
stream or in another connection's slots, an output in the wrong bucket, a fabricated output, an early, late or missing
`END`, a `FAIL`, a wrong id and a port row that isn't the Commit's each refuse the window. Nothing is drawn within the
window, so there is no ε: the verdict covers every cell. In the sampled-audit statement (`SampledAuditAccepted`),
a class drawn `All` has error 0. That statement doesn't yet model the transfer gates' tie to the certifier's
signed rows, so this unit's place in it is still to state.

**No value of the prover's says what a cell must be.** v3 held the cells to a table the prover committed, so the table
was a free input: the phase-4 attack forged a wrong id and a fabricated response, each with a
matching table, and both were accepted. Now the map is a function of the advice, which is registered before the draw,
and each id is an entry of the Commit's rows, opened under the Commit's registered root. The prover's values are the
witness that opens those: the cells' and rows' salts, the rows' paths and the routing. Both cases are regressions, refused
(`test_network_certifier_io_tie_phase4.py`), and so is a two-sided tamper of the routing: two ids swapped together
with the switches that route each cell the entry carrying its new id. This also closes a review finding on v3: the
table's binding to the token port was outside the tie and load-bearing, and had to be an `All` unit's egress transfer
gates. v4 has no table. Each id is an entry of the Commit's own rows, which the unit reads at the map's steps as
registered rows under the Commit's root (`io_tie.port_reads`), and `IoTieMatch` holds each routed entry's index to the
one the map names.

**(a) Decision: the token's id in the frame, not its text.** Each egress event carries one sampled id, and the user
decodes it with the public tokenizer.
- **The event** is `data: {…,"choices":[{"index":0,"token_id":%10d,"finish_reason":null}]}\n\n`, `EVENT_HEAD`'s
  object up to the choice, then the id right-aligned in 10 decimal characters. The shim asks vLLM for ids
  (`return_token_ids: true`, which `record.py` already sends and `live_body` deletes; else
  `return_tokens_as_token_ids` with logprobs) and writes one event per id of each chunk's `token_ids`, so neither
  vLLM's chunking nor its UTF-8 holdback reaches the wire. `pad_to`'s empty events and `events`' 64-byte cut go. A
  response is one event per kept output, of any number.
- **The ordinal event** (`active.id_event(request, token_id)`) is this one with the request's ordinal ahead of
  the id (`"request":%10d,"token_id":%10d`): 159 bytes, with the ordinal at frame byte 110 and the id at 132.
  `io_tie.id_event` is at least 138 bytes, with the id at 111. On egress the ordinal is the network certifier's stream, so the
  server's copy of it is redundant, and a field the server writes is a channel unless it is checked. With the ordinal
  event, the cell holds the ordinal to q too: one more 10-digit decode and a compare, at B ≥ 168. The prototype keeps
  the id-only event. The shim writes one of the two, and `IoTieCell` follows it.
- **Ingress** carries ids too: a request is a canonical `/v1/completions` body whose prompt is token ids, each in an
  11-byte field (`[` or `,`, then `%10d`; 11 spaces past the prompt's length, which JSON reads as whitespace), placed so
  no field straddles a frame. The server adds nothing to a token-id prompt, so `c00.input/ids` is the request's ids.
- **The network certifier's cut** is fixed: a `DATA` cell carries exactly P = B − 9 of the server's bytes, and the rest go in the
  `END` (`warden.Warden._egress`). So the shim writes each event padded with spaces (JSON whitespace) ahead of its
  blank line to P bytes, and ends the response with `TRAILER` (`data: [DONE]` and its blank line, 14 bytes, shorter
  than P). The cut is then exactly one event a `DATA` cell and an `END` that holds `TRAILER`; without the trailer, the
  last event would be the `END`'s payload.
- **The cell** is `IoTieCell{B}`, 4,139 ANDs at B = 256 (50,224 at 4,096): ok, data, end, the stream and the id. A
  `DATA` cell is every byte but the stream and the id field against the template (`data_cell`), with the id's
  decode (spaces then digits, no leading zero, below 2^32, so each id has one cell). A `PAD` and an `END` are zero
  past the `END`'s payload, and before it each is its kind's bytes but for the stream.

Text = vocab[id] in the circuit was set aside: each frame would need a hidden read of the vocabulary, a span table,
vLLM's UTF-8 holdback rule and the base64, and the grouping of tokens into chunks would be vLLM's under load, a choice
of up to F − 1 bits a response that no template fixes.

**(b) Decision: every cell, with the advice's map.** The map is a function of the advice: for each slot, the cell
the network certifier must have released there (`window_map`). Advice request i's stream is q = order[i] + 1, its ingress
ordinal on the wire: order[i] is its arrival order from 0 (advice v1's order permutation, (c)), and the wire numbers
from 1, as `active.Requests` streams and the compact record carries it. It was admitted at bucket
s = admitted[i], which the network certifier's FIFO makes non-decreasing in arrival order, and its output k must be released at
s + D + ⌊k / r⌋, so advice v0's release is held to the schedule. Its `END` is its cell n_i, so it holds r slots of
every bucket from s to s + D + ⌊n_i / r⌋, at rank ρ among that bucket's live requests by stream: slot ρ·r + j is a
`PAD` on q before s + D, then its cell (b − s − D)·r + j, an output, the `END` or a `PAD` on q. A slot past the live
requests' is unheld. Each output's entry in its bucket's port is e = σ·P + row: σ is its step's place among the
bucket's steps, sorted and padded to S. The unit reads the map as 8 bytes a slot, q and e as u32be: e = 2^32 − 1 for
a `PAD` on q, 2^32 − 2 for its `END`, and q = 0 for an unheld slot (stream 0's `PAD`, B zero bytes).
- **The advice is private**: it is never published; it enters the circuit builder and is proved in zero
  knowledge. v4 computes the map in the clear from the opened advice v0, as N2's advised mode (`advised`) reads it, and takes the map and the steps as public inputs. That is the **named gap**, the one the compile
  unit carries (the advice's hiding root in the registration, "the advice is opened to the verifier until the
  compile unit exists"). Nothing may call the tie zero-knowledge until it is closed.
- **The target** reads a committed map inside the proof, bound to the advice's hiding root. The map is registered with
  the advice before the draw, as rows of `verity.ml.commitments.advice` (fixed-width bit rows under
  `hm96-sha512/row/v2` leaves in one `frame-v3-sha512` tree, `advice.Registration`), here 4 KiB rows: a window's 8N
  bytes of records and 4S a bucket of steps, at a window-aligned place in a link-day's map. The compile unit holds the
  map's rows to the advice's outputs and release buckets, as `window_map` does in the clear. The window reads its rows
  by a range read (each row's digest, hm96 leaf and `FrameV3Leaf`, the subtree's nodes once and the path above it): 12
  rows at depth 15, 3.07 × 10^7, at the rehearsal, and 1,980 rows at depth 22, 4.77 × 10^9, served. That is about half
  of v3's table (9.86 × 10^9), at 8 bytes a cell over N cells against 16 bytes a slot over M. The steps then come from
  the map's rows, so the port's reads are at hidden positions. `FrameV3Read` already takes its position as a value, so
  they cost what they cost now. The served window is 1.466 × 10^11, 3.4% more than v4.
- **The verifier's cost.** v4's map is a public input of 8 bytes a cell, 8.09 MB a served window, which the verifier
  computes in time linear in the window's capacity. At the target it goes away: the verifier reads the roots, the
  domain ids and the statics, which are O(1) a window.
- **Complete:** every cell is opened and held to its place in the map, so the data cells are exactly the outputs the
  advice releases in the window, with no duplicate and no drop, and responses are of any length. The tie needs none of
  FIXLEN, the window-local arithmetic map or the K draws (FIXLEN stays a premise of N2's served profile, "A served
  link"). Nor does it need a window to start at a request boundary or end idle: a request's outputs fall in the
  windows their release buckets name.
- **The routing** brings each data cell its entry: in each bucket, a Beneš network over M = 2^D ≥ max(R, E)
  positions carries each of the bucket's E = S·P port entries, its id and its index e (W = 32 + D bits), and its
  (2D − 1)·2^(D − 1) switches are witness. Every setting is a permutation, and `IoTieMatch` holds each data cell's
  routed entry to the index the map names, so the switches can only find entries, never choose them.
  `IoTieRoute{W, D}` is built recursively: a column of 2^(D − 1) `IoTieSwitch{W}`es, the two halves'
  `IoTieRoute{W, D − 1}`, and a column. Its ANDs, their order and its lowering's circuit digest are the flat network's
  (`test_the_recursive_route_is_the_flat_one`, the sidecar's pins). Each depth is built once, so the build's memory is
  the wiring's. The flat trace took about 1.9 KB a gate: 15 GB at `IoTieRoute{42, 12}`, which now takes 129 MB.

**(c) Decision: the stream is the network certifier's, and a bucket has one order.**
- **The stream** is the network certifier's framing: the request's ingress ordinal, which the network certifier writes from the connection it
  terminated. The server writes only the id, and a cell on another request's stream is refused by the map.
- **The numbering, under a real engine: advice v1's order permutation.** The ordinal is the network certifier's release order:
  the order in which the requests' last ingress cells land on the link, by (a, ordinal) ("Advice v1"). The trace corpus
  shows that the engine admits in the API server's body-read order, not in the release order: 2,970 inversions on its
  heavy trace and 3 on smallkv. Advice v0 numbers requests by first row, the engine's admission, so on such a link a
  map whose q is the advice's number can't be built: the advice's admissions aren't non-decreasing in its numbering,
  as the network certifier's FIFO makes them in arrival order. So the map puts advice request i on stream order[i] + 1, through
  the permutation advice v1 decodes against the certified record (the replay's requests in arrival order,
  the derived fields' walk, its misses sent as `order` and charged as advice: 2,194 bits on the corpus's heavy
  trace, 30 on smallkv), and `window_map(…, order)` ranks each bucket's connections by order[i]. Every window of the
  heavy trace's honest egress, served through the network certifier's own egress at r = 1, is accepted through it (37 windows
  of 16 buckets, `test_the_heavy_traces_honest_egress_is_accepted_through_its_order`, on
  `verity/core/protocols/certifier/tests/fixtures/warden-heavy-order.json`), and a permutation other than the network
  certifier's is refused (`test_a_request_admitted_ahead_of_an_earlier_arrival_is_refused`).
  In v4 the permutation is public with the map, in the named gap; at the target it is in the committed map's rows,
  which the compile unit holds to the advice. A server that admits in release order (a serve-openai change in vLLM)
  makes it the identity and the charge zero.
- **The order within a bucket.** W4 releases a bucket's frames in queue order, a channel of up to log2 n_t! bits a
  bucket that neither advice v0 nor the timing term charges. The active
  warden (`warden`) instead gives each live connection r slots of every bucket, ranked by ordinal, its cells in
  order, and a `PAD` on its stream where it has none. The map pins every slot, its stream, its kind and a data cell's
  entry: of every order of a bucket's cells the tie accepts the network certifier's, so the order is a function of the
  admissions and the release buckets, which the advice already says and charges. The only orders the map leaves
  free swap byte-identical cells (a connection's pads, one request's repeated id in a bucket, the zero cells of
  unheld slots), so the channel is zero bits.
- **The response's finish, under compact framing: the egress tie's.** A compact record's token field carries
  the finish in band: the id while more outputs follow, 2^31 + id (`COMPACT_LAST`) for the last at `max_tokens`, and
  2^32 − 1 (`COMPACT_EOS`) for a sampled stop token, which carries no id. That is about a trit at each response end
  that the server would choose. The map fixes it: each record also carries its output's finish f, a function of the
  advice (none for output k < output_len − 1; for the last, the request's `finish`, `eos` or `length`, with
  output_len counting the stopping EOS). The unit holds each data cell's field to f and to the port:
  - f none: the field is the routed entry's id, below 2^31 − 1;
  - f `length`: the field is 2^31 + the routed entry's id;
  - f `eos`: the field is 2^32 − 1, and the routed entry's id, the Commit's sampled id, is one of the registered EOS
    ids (public, the model's), so a stop the engine didn't sample is refused.

  One field value is accepted at each cell, so the channel is zero bits. That the advice's finish and output_len
  follow from its ids by the stop rule is the compile unit's (advice v1's `finish`). The map record stays 8 bytes,
  with f in e's top two bits (none 0, `length` 1, `eos` 2), which needs E < 2^30 (2^20 served); the `PAD` and `END`
  markers keep top bits 11, which no f is. A compact record's ordinal is at least 1, so none is an unheld slot's, and
  it is the map's stream, order[i] + 1. The unit is the tie's compact form:
  `IoTieCell`'s parse of the record (ok, data, the ordinal and the field, zero
  fill after the record) and `IoTieMatch`'s check of the field against f, the routed id and the EOS ids. It is built
  when the compact record lands; until then the tie reads id-event cells, whose `finish_reason` is the template's null and which
  carry no finish.

**The two ingress ties.** N2's advised mode derives each request's ingress end from advice v0's prompt lengths. That rests on
two ties between the advice's requests and the network certifier's certified ingress record:
- **Each request's arrival ordinal is order[i]: checked in this unit.** The advice numbers requests in order of first
  row (`advice` validates it), and the map's q for request i is order[i] + 1 ((b), (c)). The network certifier writes the stream
  of every cell in a connection's slots (its pads, its data cells and its `END`) as its request's ingress ordinal on
  the link, from 1, and the unit holds it to q. Every request's `END` is released in one of the session's windows.
  So on a link whose every egress window is accepted, advice request i is the link's arrival order[i] (from 0;
  ordinal order[i] + 1 on the wire). An order other than the network certifier's, or a server that admits a later arrival first
  under v0's identity, puts cells on a stream the map doesn't name there (or leaves no map at all), and the window
  refuses
  (`test_a_request_admitted_ahead_of_an_earlier_arrival_is_refused`). The premise is the network certifier's numbering: the
  proxy forwards whole requests, one connection each, in the order the network certifier numbers them.
- **Each request's prompt_len is its certified ingress record's: checked by the ingress tie, not here.** The advice's
  prompt_len must be the number of prompt fields in the request the network certifier certified on ingress. This unit reads
  egress only; `IoTieRequest` ("The ingress tie" below) reads the ingress cells. The ids' binding to `c00.input/ids`
  is the ingress-ids tie's (`io_ids`, below), reading the Commit's input rows as this unit reads its sampled rows.

**The ingress tie** (`io_request`, `IoTieRequest{B, L, R}`: the reference, the network certifier's own ingress as its fake
upstream (`serve`, which drives `warden.Warden._ingress`) and the Boolean Definitions, with its sidecar). It is one
unit a bucket row of the network certifier's ingress at selection `All`, with one output bit, and it is what N2's advised
verdict requires (the phase-6 attack showed that the advised floor accepts an acausal release under swapped prompt
lengths). B is the cell size (4,096 served), L the network certifier's lanes and R its r, so a row is the bucket's L·R slots.
The row is the registered port (`warden_rows.bucket_port`: unit t reads row t), as the egress unit's is, so the
proof system's registered read holds it to the network certifier's ingress root and no gate hashes it. A row's lanes are its
live connections by ordinal, R slots each (`warden.Warden._ingress`), and an unheld lane is B zero bytes a slot. The
unit outputs 1 exactly when, in each lane of ordinal q:
1. each cell is a `PAD` on q, or a request cell on q: a `DATA` of exactly B − 9 bytes that doesn't end its request,
   or its `END` with the bytes left (`warden.cells`' cut, which the network certifier makes of the client's bytes), and no
   request cell follows a pad or its request's `END`;
2. every request cell is that cell of a canonical id request (`active.id_request` at offset 0, since the cells carry
   the bytes alone). The head and the fixed fields are at their places, a first cell's length is its
   content-length plus `BODY_AT`, the prompt fields are 11 bytes (`[` or `,`, then a u32 in 10 characters), gap
   spaces come only where the next piece would straddle a cell, and `]}` then zero fill ends the request at its last
   byte. A name (`model`, `user`) is printable ASCII but `"` and `\` in quotes: a request whose name needs a JSON
   escape is refused. Each cell parses without its request's prompt length (`parse_frame`), so the count of prompt
   fields comes from the certified bytes;
3. read from the lane's carry-in (bytes left, prompt fields so far), the cells leave its carry-out, with no count
   past 2^32.

- **The lanes** (`lanes`) are public in the named gap, as the egress map is. The verifier computes them from the
  opened advice's prompt lengths, advice request i on connection order[i] + 1 (advice v1's order permutation; v0's
  is the identity), and the network certifier's ingress release (`Release`: each connection's first bucket and its request
  cells a bucket, the wiring's timing), which it takes in the clear until the compile unit exists. Each lane is its
  connection's q and the carries `carries` gives before and after its cells in the bucket, so a connection's lanes
  chain by construction from (0, 0) into its first bucket to (0, n) out of its `END`'s, and a link's verdicts hold
  every request's prompt_len to its cells, whatever buckets it spans. Checked alone, a bucket accepts a carried
  count with any amount added to both its carries (the phase-9 attack (b)); the bucket before holds the count, so the
  chain refuses it. `lanes` refuses a release that isn't the network certifier's schedule (ordinals 1, 2, … admitted in order,
  at most L live, at most R request cells a bucket, the `END` in the last) or isn't the advice's (one connection a
  request, its cells its request's). A connection the network certifier fails stops without an `END`, so a link with one is
  refused until the tie reads the `fails` record. At the target the lanes are under the advice's hiding root.
- **The ordinal is read.** Each cell's header carries its connection's ordinal and the lane names q, so a request
  cell on another connection's stream is refused, as the egress map refuses one. A client that sends its `END`
  after a `DATA` holding its request's last byte (only a request of exactly k·(B − 9) bytes can: 338 tokens at
  B = 4,096) makes the network certifier release an empty `END`, and the tie refuses it, since only an `END` may hold the last
  byte; the shim's cut (`warden.cells`) never does.
- **The circuit.** In each lane, one `IoTieRequestFrame{B}` and one `IoTieRequestStep` a cell from the carry-in,
  then `IoTieRequestEnd` against the carry-out. The frame reads its cell's header (kind, ordinal against q, length)
  and its payload against its request's bytes left both ways, as a first cell and as a continuation, each with its
  prompt slots at fixed places. `]}` sits where the bytes left put it, so each slot is a field, the end or zero
  fill by a comparison of the bytes left with a constant, and the frame outputs whether it is a pad or a request
  cell, whether it ends the request, the bytes left after it and its prompt fields. The step keeps (left, fields,
  over, padded, ok); the count's carry past 2^32 is kept, so a sum that wraps is refused. Public: the lanes, each
  (q, left, fields, left_out, fields_out). Witness: the row.
- **What N2's advised mode takes: the verdict** (`advised.IoTieVerdict`). `advised.ends` takes the prompt-length tie
  only from an `IoTieVerdict(link, registrations, advice, accepted)`: the link, its ingress windows' registrations
  (each one's `registration.digest`, in order), the advice's digest (`advised.digest`) and one bit a registration,
  set when every bucket row of that sampling block is accepted. It is built only from IoTieRequest proofs the verifier
  verified on its own lanes (`lanes` from that advice's prompt lengths and order and the network certifier's release) and given
  only by the verifier's own tie source (`LinkWindows.ties`, `tie_service.Ties`). It refuses a verdict on another
  link or other registrations than the source records, and an advised link whose every ingress window isn't
  accepted. The proof system runs it (`tie_service.IngressTie`, "The tie" below). That makes `advised.ends`'
  premise a check: connection order[i] + 1 is an id request of `prompt_lens[i]` prompt ids. The advised unit itself
  doesn't change.
- **The first negative** is the phase-6 attack's split (`test_redteam_phase6_split_is_refused`). At B = 4,096, prompts
  of 400 and 16 tokens take 2 cells and 1, released one a bucket (r = 1). The forged advice swaps them, keeping the
  total; its lanes aren't the release's, and built anyway they are refused by the unit, which reads 400 fields in
  connection 1's cells. Lengths one off that keep every request's cells, which the release can't see, are refused
  by a bucket's verdict. The circuit reads the same cells (`test_redteam_phase6_cells_read_their_prompt_lengths`:
  338 fields and no end, then 62 and the end, then 16 and the end), and the unit refuses every forgery, cell tamper
  (header, ordinal, payload, fill, order) and lane tamper the reference does (`test_the_unit_is_its_reference`).
  The phase-9 attack's cases are regressions in
  `test_network_certifier_io_request_phase9.py`.
- **Not in it: the prompt ids' values** (the phase-9 attack's first case: the tie counts each request's prompt fields and checks
  each is a u32, but the ids the user sent are tied to nothing). The ingress-ids tie below closes it. Also not in it:
  the advice's output_len ≤ max_tokens (an optional check) and one request a user (`certified_io`'s).
- **The ingress-ids tie** (`io_ids`, `IoTieRequestIds{B, L, R, P, S}`: the reference on a fake upstream Commit
  and the network certifier's own ingress, and the Boolean Definition, with its sidecar), one unit a bucket row like
  IoTieRequest. It is the ingress analog of the egress port's reads: IoTieRequest's scan plus the Commit's
  `c00.input/ids` rows at the map's steps, read as registered rows under the root and domain of the Commit's
  registration, which the verifier holds and no instance names (`io_ids.port_reads`), and every computation of a
  prompt position, and every prefix-cache read in place of one, held to the id the user sent there. It is meant to
  replace IoTieRequest as the ingress unit once the Commit side exists.
  - **What the Commit side must expose** (the fake upstream's layout, `io_ids.commit_rows`):
    `c00.input/ids` of every executed step as one `hm96-sha512/row/v2` row a step under a fresh 192-byte salt: the
    step's input ids, u32be, in the forward's token order (the advice step's rows in batch order, row r's n_r tokens
    at offset Σ_{r' < r} n_{r'}), zero past to a fixed width P (the engine's `max_num_batched_tokens`), in a dense
    `frame-v3-sha512` tree of the session's steps. Its root and domain are registered before the draw and bound to
    the run root and to the compiled circuits' `c00.input/ids` values, which the sampled proofs prove from, as the
    port's sampled rows are (one tree may hold both). The prover keeps each step's salt and path until the tie's
    outcome of record. The verifier also needs each prefix-cache hit's source (the derived fields' `sources`), which advice
    v0's `cache_hit` count doesn't name.
  - **The map** (`bucket_map`, public in the named gap from the advice's steps and sources; at the target under the
    advice's hiding root). The bucket's prompt fields in scan order are its lanes': lane by lane, connection q's
    positions from the carry-in's fields to the carry-out's, q advice request i's. Which cells hold them is the network
    certifier's timing, so the map names fields by ordinal, not by cell. Field f is position p of request i, and its
    class is its root's: (i, p), or, below i's prefix-cache hit, its source's root at p, so a cached position shares
    a class with the computation whose KV it reads, and one opened entry serves every request that reads it.
    `keys[f]` is field f's class, numbered in order of first appearance, then the sink 2^CB − 1. The opened steps are
    every step with a token whose root is a class of the bucket (every scheduled prompt token, a preempted request's
    recomputation included; step 0 for a bucket with no field), and `ekeys[e]` is opened entry e's class, or the
    sink. The verifier refuses an advice whose class no step computes, a source that isn't an earlier request, a step
    past P tokens and more opened steps than S.
  - **In the unit:** each cell's slots (`IoTieIdsSlots{B}`, given the cell's frame on its lane's ordinal) give each
    slot's value (the u32 its 10 characters spell, at its place in a first cell's or a continuation's reading),
    whether it is a field (the cell is a request cell with more fields than its index) and its ordinal (the bucket's
    fields before the cell plus its index). The prover's first Beneš network routes record f, (keys[f], f), to field
    f's slot, and `IoTieIdsKeys` holds each field to the record routed to it, with its ordinal as index and a class
    other than the sink, and gives every other slot the sink. The second sorts the members, each slot (class, 0,
    value), each opened entry (class, 1, its id) and the sink past them, and `IoTieIdsChain` checks the classes
    non-decreasing, one value a class but the sink, and each class starting with a slot (`IoTieIdsFirst` on the first
    member and one batch of `IoTieIdsLink` over each member and its successor, so a binding traces one link; its
    C-Flock lowering is the flat trace's, AND for AND). A Beneš network is a
    permutation under any switches, so every member is in the order and the switches can only sort. On a link whose
    every bucket row is accepted, every computation of request i's prompt position p, and every KV read in place of
    one, is the id the user sent. Public: IoTieRequest's lanes, the Commit's root and domain id, the steps (the
    port's read positions), `keys` and `ekeys`. Witness: the row and the read rows (each with its salt and path, the
    registered reads'), and both routings.
  - **Tests** (`test_network_certifier_io_ids.py`): a link served through the network certifier, of chunked prefill across
    buckets, a recompute after a preemption and a cache hit from another bucket's request or the same bucket's, is
    accepted in every bucket. Refused: the phase-9 attack's first case (one prompt id substituted in the Commit, which
    IoTieRequest accepts), a cache hit on another prefix (either kind), and a Commit that differs at a cache source
    (both buckets). The unit is its reference on those and on map, lane and switch tampers
    (`test_the_unit_is_its_reference`, slow).

**The unit,** `IoTieBucket{B, R, P, S}`, one a bucket row (R = lanes·r slots):
- **The cells** are the network certifier's per-window value, as WARDEN's registration v2 commits them: each bucket's R cells are
  one `hm96-sha512/row/v2` row of R·B bytes under a fresh salt, in a `frame-v3-sha512` tree of T rows (a dense
  domain, its pads to a power of two). The unit's `row` input is the registered port (`warden_rows.bucket_port`: unit t
  reads row t, every slot held or not, with no wiring), so the proof system's registered read checks it under the network
  certifier's root, and no gate of the unit hashes the row or climbs the tree.
- **The port** is the Commit's value: one `hm96-sha512/row/v2` row a step of the session, its P sampled ids (u32be in
  batch-row order, zero past its batch), in a dense `frame-v3-sha512` tree of 2^DS steps whose root and domain the
  Commit registers (`verity.ml.commitments.advice`'s registration of the rows). The unit's `port` input is its
  bucket's S rows at the map's steps, each a registered port (`io_tie.port_reads`: `port[s]` reads the bucket's s-th
  step), so the proof system's registered reads check them under the Commit's root, and no gate of the unit hashes a
  row or climbs the tree. This is the fake upstream's layout. The real port is vLLM's Commit
  (`runner.sampler/sampled_token_ids`, int32 [num_reqs, 1], under the run root's binding map), and the reads follow
  its layout.
- **The content** (`IoTieContent{B, R, P, S}`) parses each cell, routes the bucket's port entries and matches them.
- **Public:** the network certifier's registration (root and domain id), the port's root and domain id, the statics, and in v4
  the bucket's records and steps (the named gap). At the target the map registration's root and domain id replace
  them.
- **Witness:** the row and its salt, and the S port rows with their salts and paths (the registered reads'), and
  the switches.
- **Output:** one bit, ok. A sampling block's tie is its T units'.

So the layout to register is a row a bucket. Row t − w·T holds bucket t's S slots in slot order, slot j at bytes
j·B to (j + 1)·B: WARDEN's row (t − w·T)·S + j becomes words j·B/2 onward of row t − w·T. At WARDEN's schedule that
row is 4 KiB. Built that way: `IoTieBucket` reads its one bucket row and its S Commit step rows as registered ports,
S + 1 reads a unit, and recomputes no root and hashes no row.

**What the map and the port must be.** For the advice's output k = (step, row, request, index), released in the window
(`release[k] // T = w`):
- q is order[`request`] + 1, its request's ingress ordinal on the wire under advice v1's order permutation, which
  the unit holds to the network certifier's stream ("The two ingress ties");
- its cell is in bucket t = `release[k]` − w·T, at slot ρ·r + (`index` mod r), ρ its request's rank by stream among
  the bucket's live requests;
- e = σ·P + `row`, with σ the place of `step` among the bucket's steps;
- the id is port row `step`'s `row`-th id: `runner.sampler/sampled_token_ids` of step `step` at batch row `row`.

So each `release[k]` is output k's frame bucket in the network certifier's certified egress record, which is what N2's
advised floor reads. The unit checks this by position: a cell's bucket is its row in the network certifier's registered tree,
and the map puts output k's cell in row `release[k]` − w·T and nowhere else.

`window_map` computes the map and the steps. The Commit sets the port's layout and its binding to
`c00.input/ids` and `sampled_token_ids`, in zero knowledge what `check/commit_tokens.py` does in the clear. The tie
reads the Commit's root as public.

**The network certifier's rows, exactly.** The network certifier registers its cells as the following root a sampling block, and each bucket's
unit reads its row there by a registered read. The reference is `io_tie.data_cell`, `pad_cell`, `end_cell`,
`bucket_rows`, `domain` and `rows_root`, and the network certifier's own cut is `warden.Warden._egress`.
- **A cell** is B bytes (B = 256; `io_tie` takes B ≥ 147), `HEADER` (`>BII`: the kind, the stream as u32be, the
  payload's length as u32be), the payload, then zero fill to B.
  - A `PAD` (kind 0) on stream q has no payload. A slot no connection holds is stream 0's, B zero bytes.
  - A `DATA` cell (kind 1) carries P = B − 9 bytes, `id_event(id, P)`: `ID_EVENT_HEAD`, the id right-aligned in 10
    decimal characters (`%10d`), `,"finish_reason":null}]}`, spaces to P, then `\n\n`. It has no fill.
  - An `END` (kind 0x20) carries `TRAILER`, `data: [DONE]\n\n`, 14 bytes.
  - The server writes the payloads; the network certifier writes the headers, cuts the payloads and writes the fill.
- **The stream** is the request's ingress ordinal on the link, counted from 1 over the session as `active.Requests`
  numbers its streams: order[i] + 1 for advice request i (order[i] its arrival order from 0), the map's q.
- **A bucket's order** is the network certifier's grid: each live connection's r slots, ranked by ordinal, its cells in order;
  then the slots no connection holds.
- **Row t,** for t in 0 … T − 1, is bucket t's R cells concatenated: R·B bytes, 25,088 at the rehearsal (R = 98) and
  431,616 served (R = 1,686).
- **Its leaf value** is `head_check.hiding_value(row, salt)`, the `hm96-sha512/row/v2` leaf: hm96's (`hm96.SHA512`,
  its default key) tree leaf over `commit_string(key, sha512_row_v2_digest(row, 8·R·B), salt)`. The salt is fresh for
  each row: 192 bytes of the network certifier's own randomness.
- **The tree** is the dense `frame-v3-sha512` domain of T rows, `io_tie.domain(binding, T)`, which is
  `CommitmentDomain(binding, -1, RangeIndexedDomain(0, T), hash="sha512")`. The binding is a 64-byte digest naming the
  link, the window and the cells, which the network certifier's registration fixes; the tie reads the domain id as public. Under
  that domain:
  - leaf t is `leaf_ref(domain_id, "hm96-sha512/row/v2", value_t, t)`;
  - the pads are `_hash(b"pad", domain_id, uint(t))` for t in T … 2^⌈log2 T⌉ − 1;
  - each node is `node_ref(domain_id, level, index, left, right)`.

  This equals `tree_levels` over the domain's `row_leaf`s.
- **Registered:** the root and the domain id, with the window w; bucket t is session bucket w·T + t. The T salts go
  to the prover, privately: they are its witness, and it rebuilds the cells from the map and the port.
- **The port's rows** (the fake upstream's; the real ones are vLLM's) are the same leaves and tree over the session's
  2^DS steps: row s is `port_row` of step s's sampled ids, each under a fresh salt (`rows_levels`, `open_port`:
  `verity.ml.commitments.advice`'s registration of the rows), which each unit reads as registered rows
  (`port_reads`).
- **Not needed by the All tie:** a per-stream sequence number in the header.

**What the prover keeps,** from the window's registration until the tie's outcome of record, by `proof_deadline`: the network
certifier's T salts, and the port rows' salts and paths at the window's steps, which the Commit's prover keeps for its own
openings. At the target, it also keeps the map's rows, salts and paths, which the advice's prover registers. The cells
are a function of the map and the port, so the prover rebuilds them, and it computes the switches when it proves.

**Composition.**
- **N2** (`All`) holds the head's fields and counts and the certificate that signs them.
- **The cells' root** is the network certifier's own per-window value, registered by the auditor's device
  (`trusted-host`). It isn't the head's frames root, which is an unsalted tree that can't be opened outside a
  proof. The head's counts and the cells come from one grid of the network certifier's: bucket t holds n_t data cells. The tie
  doesn't read the head; tying the cells' root into it (a record v3 field) is open.
- **The map** is the advice's, charged as `A_c`. In v4 the verifier computes it from the opened advice v0, the named
  gap. At the target it is registered with the advice before the draw and read in the proof, and the compile unit
  holds it to the advice. **The port** is the Commit's root, and its layout and its binding to the run root are
  the Commit's.

**Reuse.** The new Definitions (`IoTieCell`, `IoTieSwitch`, `IoTieRoute`, `IoTieMatch`, `IoTieContent`,
`IoTieBucket`) are Boolean composites with no hash. No unit hashes a row or climbs a tree: both ties read the network certifier's
registered rows, and the egress and ingress-ids ties the Commit's, whose rows the statement's registered-read slots
hash (`NetworkRowDigest`'s and `Hm96Sha512Leaf`'s work) and whose paths the verifier climbs.

**The prototype** (`io_tie`, on the network certifier's own egress and a fake upstream's port) has the reference
(`io_tie.reference`, in the clear) and the Boolean Definitions above. Its tests hold every circuit to its reference and
refuse each tamper of the cells, the map, the port and the routing.

Still to build: a unit proved on the network certifier's registered rows through `bucket_port` (the tests hold each bucket's row to
the network certifier's registered row and its port to row t a unit; no proof reads them yet), the port's real layout and
registration (vLLM's Commit), the target's map registration and its range read, the ingress tie's view read under the
advice's hiding root, and the real Commit's `c00.input/ids` rows that the ingress-ids tie (`io_ids`) reads.

**Open for proofs.**
- Hidden-position reads (frame and port) under protocol soundness. In v4 every read is at a public position
  (every bucket row, every port row at its public step), and what is hidden is the routing: a Beneš network whose
  switches are witness, which only finds entries, since the match checks each routed entry's index. At the target the
  port's reads are at hidden positions, the steps the private map gives, and the map's rows stay at public ones.

What this establishes, once built: on an egress link whose every window is accepted, every cell the user received is
a pad, `event(id)` or its response's `END`. That id is the Commit's sampled id at the step and batch row advice v0
names for that cell's output. The cell is on its request's stream, at the release bucket and the slot in it that the
advice names, and each request's number in the advice is its arrival ordinal. This holds for every cell with no ε
and for responses of any length. What it doesn't establish:
- zero knowledge for the advice: v4 reads the map and the steps in the clear from the opened advice, the named gap,
  which the target and the compile unit close;
- the advice's prompt lengths against the ingress record, which is the ingress tie's (`IoTieRequest`);
- the port's layout and the registration of its root (the Commit's);
- anything about the text a client shows beyond the public tokenizer's decode.

## The Program's logical schedule and the clock-sync advice (`schedule`)

The Program fixes each real frame's content, its **anchor** (the recorded bucket of the request it answers) and its
**logical index** λ (its position in its session, never negative). A window declares one clock-sync value
`ClockSync(δ, ρ)` from the public set Σ_sync (`SyncSet`; 16 offsets × 8 rates is 7 bits). A frame is **eligible** at
`anchor + δ + ⌊λ/ρ⌋` under the sync of its anchor's window, the shaper **hands** it to the certifier in canonical order at
`h(k) = max(eligible(k), h(k−1))`, and the certifier's grid **releases** it. Stated in Lean:
`Security/Definitions/NetworkCertifier/Schedule.lean` (`NetTiming`'s `eligible`, `handingBuckets`, `expectedRow` and
`constantRate`, the Program whose advice is the content advice and one in-set sync per window), with the certifier
program's `NetworkCertifier.Schedule` as its code; `schedule.py` (`expected_grids`, `run_shaper`) is the Python.

## An advised link (`advised`)

An advised link is a served link under arbitrary traffic: responses of any length, batched, chunked, preempted and
timed as serving chose, carried across windows. Nothing fixes its egress counts before the run and FIXLEN doesn't hold,
so N2 takes them from the prover's advice and holds the advice to the certified ingress head. The advice is
`advised.LinkAdvice`: I1's advice (`advice.Advice`, v0, "Advice v0" below: one per certifier session, after serving,
public in v0 and charged by `advice.bits`) and `received[w]`, the request frames the ingress certifier releases in
window w. It carries no tie: the tie of its requests to the certified ingress, IoTieRequest's verdict
(`advised.IoTieVerdict`), comes only from the verifier's own tie source (`LinkWindows(ties=…)`, `advised.TieSource`).
Declared with `LinkWindows(advised=AdvisedProfile(windows, request_frames, own))`, whose `request_frames` is the link's public
framing: entry n − 1 is the ingress frames a request of an n-token prompt takes (id
requests, whose frames are a function of the prompt's length alone), for every prompt length the link admits.

- **What N2 reads** (`advised.view`):
  - of I1's advice, fields `advice.bits` charges: each request's `prompt_len` and `output_len` and each step's rows'
    `request`, `start` and `n`, through `advice.outputs`, which numbers the kept outputs (request j's λ-th, in step
    then row order); and `release[k]`, the session-global egress bucket in which output k's frame is released. It
    reads no `id`, `finish`, `kind`, `cache_hit`, `preempted` or `sources` beyond `Advice`'s own validity;
  - `received[w]`, which the unit holds to window w's certified ingress head;
  - the tie: the `IoTieRequest` verdict on each of the link's ingress windows, on the request view computed from
    this advice, that the link's j-th certified request is advice request j and its prompt v0's `prompt_len` ids. It
    names the link (`IoTieVerdict.link`), its ingress windows' registrations with the challenger
    (`IoTieVerdict.registrations`, each one's `registration.digest`), the advice (`IoTieVerdict.advice`, its
    `advised.digest`) and each window's verdict (`IoTieVerdict.accepted`), and `view` holds the first two to the link
    and the registrations the verifier's tie source records, so a verdict on another link or ingress record isn't
    reused;
  - the framing, and each window's declared sync, which must be one of Σ_sync, the set the certified egress head's
    parameters digest binds (outside it the window is refused), and the sync the certifier ran that window under,
    which every floor reads and the declared one must be, and which the verifier takes only from its own clock
    source (`LinkWindows.clock_source`, `advised.ClockSource`): the certifier's signed clock sync
    ([protocol/clock-certificate.md](protocol/clock-certificate.md)).
- **The ingress ends are derived** (`advised.ends`), not advised: advice request j is the link's j-th request and its
  prompt length is the certified prompt's, which the tie holds (FCFS admission meets the order, since v0's first-row
  order is then the arrival order), so it ends at request frame `ends[j]` = Σ_{i ≤ j} `request_frames[prompt_len_i −
  1]`, and the last end must be Σ `received`, the link's certified request frames. A link without the tie, a verdict
  on another link, other ingress registrations or another advice or that doesn't accept every ingress window, a
  prompt longer than the framing covers, or ends whose last isn't Σ `received`, refuses the advice. No field but
  v0's says
  where a request ends, so the ends cost nothing beside `advice.bits`.
- **The window's view** (`advised.view`), computed by the verifier in the clear from the advice, the framing and the
  syncs the certifier ran, is the unit's public input:
  - `released[t]`, the frames the advice releases in bucket t, at most R (more refuses the advice);
  - `floor[t]`, the request frames the window's ingress must have released by bucket t, less B_w = Σ_{v<w}
    `received[v]`. Output λ of request j released at b is eligible no earlier than j's anchor + δ + ⌊λ/ρ⌋
    (`schedule.eligible`) under the sync (δ, ρ) the certifier ran the window that holds b under, and is released no
    earlier: so the first `ends[j]` request frames are released by bucket b − δ − ⌊λ/ρ⌋. A frame for which that
    bucket is before 0, or released past the link's windows or in one without the certifier's signed sync, refuses
    the advice. Since δ ≥ 0, no sync lets a frame out before the bucket that releases its request's last frame. A
    release in that bucket is a served clock's, an engine's first output in its request's bucket (a served run's
    advice, `test_network_certifier_advised_served.py`); on a certified path the proxy hands a frame read in bucket t
    on at t + 1, so its honest syncs have δ ≥ 1. A window whose declared sync isn't the one the certifier ran it under
    is refused: a δ below the traffic's would loosen the floor, and the phase-11 attack had an output accepted in its
    request's own bucket under δ = 0 declared for a window run at δ = 1 (regressions in
    `test_network_certifier_advised_phase11_floor.py`). The bucket b − δ − ⌊λ/ρ⌋ can fall in a window before the
    one that releases the frame, so a floor that read a declaration would carry one window's word into another's
    view. No floor reads one, so a wrong declaration refuses its own window and no other (the phase-15 attack), and the
    link's verdict, the AND over its windows, refuses the link. The verifier takes the sync the certifier ran each
    window under only from the certifier's salted clock certificate of the window, opened against the digest it signs
    under its key and session (the sync and the salt are witness, the certificate public), and a window without a
    valid one is refused, and the link with it ([protocol/clock-certificate.md](protocol/clock-certificate.md));
  - `received`, `received[w]`.
- **The unit** is `NetworkAdvisedLinkWindow` v2{EGRESS, INGRESS, ADVISED}: `NetworkLinkWindow`'s checks, and one gate
  (`advised.gate`) naming `advised.NetworkAdvisedCounts_v1{T, RIN, OWN}` and its circuit digest pinned in
  `advised.circuit-check.json`, over the two heads' counts (witness, cut from the heads the head checks open) and the
  view. Its ok bit is 1 when every ingress count is at most R_in and bucket 0's at least OWN; with the OWN own frames
  taken off at bucket 0, the ingress prefix sums are at least `floor` and sum to `received`; and the egress counts are
  `released`. About 105 ANDs a bucket: 63,031 at T = 600, R_in = 1, OWN = 0, and 64,894 at R_in = 4, OWN = 1.
  Those are over circuit-check's whole-Definition budget (60,000 ANDs), so `check` holds the T = 600 roots to the
  reference and their pinned digests without lowering them (`--lower-max-ands 70000` lowers them to 62,986 and
  64,847 with no mismatch); T = 512, R_in = 4, OWN = 1 lowers whole to 55,343 ANDs, which the sidecar pins.
- **Public** (`PUBLIC_ADVISED`): `PUBLIC_SERVED`'s items but the served sizes, with the advised ones (the link's
  windows, the framing and OWN), each window's declared sync, and the advice as a protocol output (`OUTPUTS_ADVISED`:
  the verdict, the deadline bit and the advice's digest). The view is the unit's ``released``, ``floor`` and
  ``received`` parameters (`ADVICE`), in its public row (``pub``); the heads' counts stay witness.
- **The verifier of record** is `LinkWindows.verify(…, advice=A)`, which decides on the delivered proof as on any
  link: it computes `view(A, w)` itself, under the tie from its own tie source alone (`LinkWindows.ties`; no entry
  point takes one beside the advice), and opens the proof's public row only to its own copy, so a proof made on
  another advice's view of the window is refused. Without a `LinkAdvice`, a tie source or a clock source
  (`LinkWindows.clock_source`), or on an advice `view`
  refuses, the unit takes no instance and the window is refused. The in-the-clear check (`clear`, the debug comparison) runs `advised_check`:
  `advised.reference` over the opened heads' counts and `view(A, w)`. Each window reports a third public output,
  `advice`, the SHA-512 of A's canonical
  encoding (`advice.encode`) and `received` (`advised.digest`), so it names the string the compiler reads too.
  `network_verdict` takes the advised profile's windows and refuses a window that names another advice than the first
  accepted window's, so every window is checked against one advice.
- **Carry-over is stated by the advice, not forbidden.** An output sampled in one window may be released in a later
  one, and a response may span any windows: `release` says where each frame went out, so no egress queue is carried.
  The ingress carry is `received`: each window holds its own ingress total to it exactly, so by induction from window
  0 (B_0 = 0) on a link whose every window is accepted, B_w is the request frames released before window w, and each
  window's floor is the link's.
- **Soundness: with the advice fixed, the counts are determined.** On an accepted window w the egress counts are
  `view(A, w).released`, a function of A's `release` alone: the ingress counts, `received` and the syncs only accept
  or refuse and change no count. So on a link whose every window is accepted, all naming one advice, the egress
  timing of accepted runs carries at most `advice.bits` (the accepted-runs bound's A_c) beside the log2 #Σ_sync bits
  a window the sync charge counts, and nothing else is left free: not the frames' timing within the counts, not the
  ingress timing, not when the server wrote, and not where a request ends. The advice is held to the ingress head: no
  frame is released before the ingress certifier released `ends[j]` request frames, δ + ⌊λ/ρ⌋ buckets earlier,
  under a sync in Σ_sync. Those are request j's and every earlier request's frames because v0's prompt
  lengths are the link's requests', in arrival order: the tie, below. Without it, a prover that splits the same
  request frames among the requests otherwise (a short and a long prompt swapped between an early and a later
  request) lowers the early request's end, and so the floor, and an output released before its request's frames were
  in is accepted on the same certified record, under an advice `advice.bits` charges no more (the phase-6 attack, a
  regression in `test_network_certifier_advised_phase6.py`).
- **The tie is IoTieRequest's verdict, and without it every advised link is refused** (fail closed).
  The certified prompts are witness, so the verifier can't derive the prompt lengths and order on
  the host, and a tie the prover states is no tie: the phase-7 attack showed that one stating the swapped split reopens
  phase 6's acausal release. Nor is a verdict the prover delivers: the phase-10 attack built one for
  the forged advice's digest, every window accepted, and the verifier of record, which read it off the delivered
  advice, accepted the swapped split (regressions in `test_network_certifier_advised_phase10.py` and
  `test_network_certifier_advised_phase6.py`). So the advice carries no tie, and `advised.ends` takes it only as
  the in-proof `IoTieRequest`:
  its verdict on every ingress window of this link's registrations, for the same advice digest, from proofs the
  verifier of record verified itself (`advised.IoTieVerdict`, naming the link, the registrations and the advice),
  given by the verifier's own tie source (`LinkWindows.ties`). The tie service (`tie_service.IngressTie`) builds it,
  and `tie_service.Ties` is that source. A registration is one ingress sampling block, the network certifier's salted root of its
  T bucket rows, held to the window's order, and T proof blocks, unit t bucket row t. The unit is
  `NetworkIngressTieBucket`: one gate naming IoTieRequest and its pinned circuit digest
  (`io_request.circuit-check.json`), its witness row the bucket row itself, read off the network certifier's root by the
  verifier's own registered read (`warden_rows.bucket_port`). The statement is a shared-row file whose `wit` table is
  the registered port. The verifier of record decides each window on the proof alone, each unit's `pub` row opened to
  its own lanes (`io_request.lanes`, from the advice it holds and the network certifier's release, taken in the clear), and
  reports the advice's digest. `IngressTie.tie` reads the verdict from the stream's outcomes of record alone, on the
  link and the stream's registrations. Window w is accepted when its outcome is, decided on the proof for the
  advice's digest. Every window is refused unless the stream registers exactly the advice's windows in order, the
  release is the network certifier's schedule for the advice and puts `received`'s request cells in each window, and the advised
  profile's framing is the network certifier's cut (`tie_service.framing`) with no own frame. A connection's lanes chain from
  (0, 0) into its first bucket to (0, n) out of its `END`'s, so the units together hold each request's count across
  the buckets it spans. A size class with no pinned digest has no accepted proof, and at the served size none is
  pinned (1,500,947 ANDs a bucket row, about 9.0 × 10^8 a 600-bucket window), so there every advised link is still
  refused. `tests/test_network_certifier_tie_service.py` holds the phase-9 attack's carried field count, its forged
  advice and phase 6's split, refused through the verifier, on stand-in proofs (`tests/standin.py`), and
  `benchmarks/network_traces/n2_tie_zk.py` proves them live. The advised tests stand in for the source with its
  reference in the clear (`tests/advised_traffic.py`, `StandInTie`).
- **Bucket 0 is capped at R_in, which holds the own frame.** The ingress certifier releases min(queue, r_in) frames a
  bucket (`grid.Certifier`) with its own frame queued at the head of the queue, ahead of bucket 0's arrivals and of any
  request frames carried from the window before (`active.Bucketer`), and a head with a count past its allocation
  doesn't exist (`certificate.RecordHead`): with R_in request frames arriving in bucket 0 the certified count is R_in,
  one of them the own frame, and the last request frame goes out in bucket 1.
- **The ends are where the signed counts put them**: request k ends in the first bucket where the
  ingress head's signed counts, less one own frame a window, reach `ends[k]`, with `request_frames[n − 1]`
  `id_request_frames(n, B)`. The floor reads the same prefix sums, so it holds each output to the bucket the signed
  counts end its request in. The rule is exact, since each window's own frame leaves in its first bucket.
- **Named gap: the record doesn't hold the prompt lengths the unit reads.** The unit
  takes v0's prompt lengths, held to the certified ones only by IoTieRequest's verdict (the tie above), because the
  ingress head doesn't sign them. Design: the ingress head signs each bucket's completed requests inside
  `RecordHead.encode`, and the unit reads them from the head it opens. It changes the head encoding that Lean pins, so
  it lands with the head check's and the Lean model's change, not before. With the network certifier's FIFO admission the
  completed requests follow from the signed counts and the certified prompt lengths, so on ingress what's left is
  holding v0's prompt lengths to the certified bodies. Each window's sync is signed: the certifier's salted clock
  certificate beside its egress record, so it needs no head change
  ([protocol/clock-certificate.md](protocol/clock-certificate.md)).
- **Named gap: the clock's opening.** A window's clock certificate signs its sync salted
  (`certificate.clock_commitment`), and until N2's advised unit opens it the verifier of record opens it on the host
  (`advised.SignedClocks`), so on a refused window it reads the sync the certifier ran. Closing it is a circuit change
  to N2's advised unit, with its `circuit-check` report, not a Lean one: the design is in
  [protocol/clock-certificate.md](protocol/clock-certificate.md) ("The opening is on the host").
- **Premises** of "an honest advised link is accepted", on which N2's claim doesn't rest:
  - `release` is the certified egress record's: each kept output is one egress frame, in the link's canonical order,
    which the I/O tie fixes (within a bucket the order doesn't change the counts);
  - the releases fall in the link's windows, each of which declares a sync in Σ_sync, the one the certifier ran it
    under, whose clock certificate the certifier signed and the verifier holds (without it, no advised link is
    accepted);
  - `received` is the ingress record's;
  - the link's j-th request is advice request j (FCFS admission: IoTieRequest's ordinals are the verifier's, so an
    advice numbered in another order, as a runner's batch order numbers v0 and the walk's `walk.order` maps
    back, is refused), an id request of its prompt length;
  - IoTieRequest runs in the proof system and accepts every ingress window of the link (until it does, no advised
    link is accepted);
  - each window's declared sync has a δ and ρ that every frame released in it meets: a small δ, and ρ at least a
    stream's frames a bucket (`schedule.eligible` reads the anchor's window's sync, the view the release's, which the
    verifier sees);
  - both ends share T and τ_b (checked by `LinkWindows`).

What this establishes: on every accepted sampling block of an advised link, the egress head's counts are the advice's
releases, and the advice's releases are eligible after the ingress head's releases under the declared sync. What it
doesn't: the frames (the I/O tie's), that the advice is the computation's (the compiler's and PoUW's), the
tie itself (IoTieRequest's, through the tie service; no IoTieRequest digest is pinned at the served size class, so
there no advised link is accepted), the syncs the certifier ran (its report, which no head signs yet), and the
premises.

## The audit (`audit`)

The audit (`Audit.check_link` on every egress link, `Audit.check_ingress_link` on every ingress link) is stated in
Lean as a core protocol: `NetworkCheck.Check` (`verity/network-certificate/v4`, certified records) and
`NetworkCheck.Records` (`verity/network-records/v3`, raw, committed and certified records). Their module docstrings give
the acceptance conditions and the violations, the first that applies naming it (`no-verdict`, `duplicate`, `signature`,
`status:<reason>`, `advice`, `mismatch`), and `verity/core/protocols/network/tests/test_network_check_protocol.py` and
`test_network_records_protocol.py` hold them to `audit.py`. A violation is recorded and the run rejected; there is no
halt, and later windows are still checked.

**One clock per spatial unit.** Every egress link of the unit reads the same sync in a window. The audit keeps each
window's sync (`Audit.syncs`), set by the first egress link checked for that window. A link that declares a different
sync for the window is an `advice` violation in that window, even if its sync is in Σ_sync. The reason is the bound:
the accepted-runs bound charges one sync per window for all links. The advice of `SecurityProofs.EncardAccSeqsLeSync`
is `Ac × (Fin W → Sync)`, so its 7 bits per window at 16 × 8 are `#Σ_sync^W` values over W windows. A sync per link
would be advice of `#Σ_sync^(L·W)` over L egress links, L times the bits, and an audit that took each link's syncs on
their own would accept it.

A declared sync outside Σ_sync, or other than the unit's, is an `advice` violation in its own window only. The
Program's grids are computed from each link's syncs as declared, out-of-set values included (any δ ≥ 0 and ρ > 0 gives
a schedule), and every other window is checked against them. A window's grid reads only the syncs of its own and
earlier windows: a frame is eligible no earlier than its anchor, hand times only rise in canonical order, and the queue
carries forward (`SecurityProofs.ExpectedRowCausal`, for any clock-sync values, in the set or not). So a window before
the violating one gets the same verdict whatever that window declared, and a later one is checked against the grid the
declared advice gives it.

**Ingress links are audited on their statuses** (`Audit.check_ingress_link(link, records, windows)`). Lean's acceptance
needs every boundary link's status to be complete, ingress links included, since an ingress status is itself an egress
channel (`Security/Proofs/NetworkCertifier/README.md`, modelling choice 6). An ingress link's rows are the Program's input, which it reads rather
than predicts, so no grid is compared. An ingress sampling block is accepted if it has exactly one record, present by the
deadline, and its status is complete; otherwise it is a `no-verdict`, `duplicate` or `status:<reason>` violation, as on
egress. A committed record's status is read in the clear, and the digest binds it. The verifier has no Program grid to
recompute an ingress digest from, though, so unlike on egress it can't catch a published status that disagrees with
the digest: that rests on the certifier publishing its own status (W5). A certified ingress record's status is the signed
head's, so it rests on the certifier instead, and the audit (`check_ingress_link(..., key=key, params=params)`) also
requires the head's T, r, B, τ_b and parameters digest to be `params`'s, or it is a `mismatch`. A run is accepted only if every sampling block of
every link the audit checks is, so an accepted verdict on every link of ∂X, with `check_link` on egress and
`check_ingress_link` on ingress, means every boundary link's status was complete in every window audited.

Every capacity bound holds on accepted runs only. A rejected window may already have carried up to its unbound capacity,
about 8·T·r·B bits (`capacity.unbound_bits`). On committed records, the bounds also assume `cr/sha-256`; on certified
records, `cr/sha-512`, the signature's unforgeability and an honest network certifier.

## Capacity (`capacity`)

| Quantity | Value | Lean |
|---|---|---|
| Free slot choice | `2^(T·r)` occupancies | `SecurityProofs.CardFree` |
| Left-packed | `(r + 1)^T` occupancies, attained by the FIFO certifier | `SecurityProofs.CardLeftPacked`, `SecurityProofs.EncardFifoGrids` |
| Exactly M released | at most `binom(T + M − 1, M)` | `SecurityProofs.CardCountsEqLe` |
| At most M available | at most `binom(T + M, M)` grids | `SecurityProofs.EncardFifoGridsLeOfAvailable` |
| Envelope, with no clock | `T·log2(r + 1)` bits per sampling block | Proposition 2 |
| Accepted runs, with clock-sync advice | `A_c + W·log2 #Σ_sync + n·log2 D` bits | `SecurityProofs.EncardAccSeqsLeSync`; for the shipped schedule, `SecurityProofs.EncardAccSeqsLeConstantRate` |
| The shipped schedule, over the first m windows | `A_c + min(W, m)·log2 #Σ_sync + n·log2 D` bits, for n ≤ min(W, m)·T steps | `SecurityProofs.EncardAccSeqsLeConstantRateWindows`; for a decoder of the wire, under W2 and J, `SecurityProofs.EncardDecodableLeConstantRate` |
| The shipped schedule, per window | `A_c` once, then `log2 #Σ_sync + T·log2 D` bits per window over the run's first m whole windows (n = m·T ≤ W·T): 7 bits per window at 16 × 8 sync values under J. Cumulative: a later window may carry more, since content advice and an earlier window's sync can first show in it | `SecurityProofs.EncardAccSeqsLeConstantRatePerWindow`; for a decoder of the wire, under W2 and J, `SecurityProofs.EncardDecodableLeConstantRatePerWindow` |
| The circuit's Program under a sampled audit of its step units, at δ | `c + log2 #(faulty sets) + Σ_{v faulty} committed bits + n·log2 D` bits, where the inputs other than the ingress take at most `2^c` values and a faulty set is one `RAcc δ` allows (in `M_δ`, no unit drawn `All`). It holds on the runs an accepted audit gives, except with probability δ + ε_ks + ε_bind over the selection (`SampledAuditAccepted`), with c the unregistered ones, since `RAcc` pins the inputs to the trace. At δ ≥ miss(1) no unit may be faulty, and it is `c + n·log2 D` | `SecurityProofs.EncardAccSeqsSampledLe`, for a layout that fits the circuit (`Layout.Fits`); given `SampledAuditAccepted`, `SecurityProofs.SampledLeakAccepted` |
| The network certifier (design v3): every byte re-framed into cells on a public schedule, a late cell fails its connection | `log2 #schedules + Σ_c log2(cells_c + 1)` bits besides the data, one fail choice per cell of each connection's lane or none. Over a monitoring period of sampled audits, plus each window's sampled charge (the row above) at the network certifier's rows | for a decoder of the client's view, under `trusted-host`, `Claims.Sampling.KS` and `Commitment.Binds`, `SecurityProofs.PeriodLeakAccepted` |
| A fixed clock, under J (D = 1) | 0 bits of timing | `SecurityProofs.EncardAccSeqsLeFixedClock`; for the shipped schedule, `SecurityProofs.EncardAccSeqsLeConstantRateFixedClock` |
| Ingress, where no clock applies | `log2(r_in + 1)/τ_in` bit/s per link, under J (D = 1), statuses complete | `SecurityProofs.EncardIngressDecodableLe`, `SecurityProofs.EncardIngressObsDecodableLe` |
| Ingress, content plus timing | `log2 #Cs` bits plus the ingress timing | `SecurityProofs.EncardIngressDecodableLeContents`, `SecurityProofs.EncardIngressObsDecodableLeContents` |
| Ingress, per sampling block (the K charge's unit) | `T·log2(r_in + 1)` bits per sampling block over the first m whole windows, plus `T·log2 D` of jitter per window, statuses complete. Unlike egress, each step's bound does not depend on what came before | `SecurityProofs.EncardIngressObsDecodableLePerWindow`; content plus timing, `SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow`; for a decoder of the wire, W2 and `SecurityProofs.EncardIngressDecodableLe` at n = m·T |

**The advice's charge.** The advice string is the circuit's Program's content advice, so its charge is the sampled row's
`c`. The sampled leak bound takes `c` as declared and public, with the premise that at most `2^c` advice values are
charged `c` bits or fewer (`hAs` in `Security/Definitions/NetworkCertifier/Guarantees/SampledCharge.lean`), as for any
prefix-free code (Kraft). Advice v1's charge is `code_v1`'s segment length for the whole window, every count and value
inside the arithmetic coder, and its strings are prefix-free, so at most `2^c` advice values (each window's decisions,
hits and order entries) are charged `c` bits or fewer (`protocol/charge-chain.md`, "The bound"). Lean mirrors the
charge's references one for one: `ChargeCount_v1{N}`'s (`arith_encode.py`'s `reference_charge`,
`Security/Definitions/NetworkCertifier/AdviceV1.lean`) and `ChargeCount_v2{N, H, O}`'s (`window_charge.py`'s
`reference_window`, `Security/Definitions/NetworkCertifier/WindowCharge.lean`). Which slots ask (`en`) is not charged
because it is not advice: the compile slice, its charge included, is proved on every advised window and never sampled,
so `en` is pinned by that slice's scan, and the carry across windows is chained from the book's priors
(`charge_chain.check_chain`). That the circuits compute Lean's windows (`ChargeCount_v1`'s `AdviceV1.window`,
`ChargeCount_v2`'s `WindowCharge.window`) is held by tests: circuit-check holds each circuit to its Python reference,
and the Lean is held to those references by the charge difftests: the certifier program's `NetworkCertifier.AdviceV1`
and `NetworkCertifier.WindowCharge`, proved equal to the definitions (`CodeChargesV1`, `CodeChargesWhole`), give what
`reference_charge` and `reference_window` give on 4,000 random instances each (`runs` in `lean/lean-audit.json`, their
data checked by `verity/core/protocols/certifier/tests/test_network_certifier_charge_lean_difftest.py` and
`verity/core/protocols/certifier/tests/test_network_certifier_window_lean_difftest.py`); it is not proved in Lean.

The bound charges the declared `c`, not the realized lengths, since a run counts if any advice within it explains it.
Each run's egress must equal the Program's, which needs every egress frame tied, and which the tie gives only when no
step unit can be faulty: every step unit drawn `All`, or a sampled audit at δ ≥ miss(1). At a smaller δ an accepted
sampled audit gives `Accepted` only with the faulty proof blocks' committed gates held as advice
(`SampledAuditAccepted`), and the charge adds the faulty proof blocks' committed bits and log2 of the number of possible
faulty sets, except with probability δ + ε_ks + ε_bind over the selection (`SecurityProofs.SampledLeakAccepted`).
Charging a window only for the windows up to it would need a causality assumption about the Program, which the statement
doesn't take.

**The network certifier's row.** The network certifier terminates both connections and sends every cell in its bucket: a connection admitted
at bucket `s` sends its egress cell `j` in bucket `s + D + ⌊j/r⌋` (`Security/Definitions/NetworkCertifier/Warden.lean`).
Admission is the network certifier's decision on its own schedule. A cell, or the end of the stream, that reaches the network certifier after
its bucket fails the connection there with one fail cell that carries no reason, and the lane stays held to its
scheduled end. So the client's view of the server side is the schedule, each connection's fail cell or none, and the
data in the cells: the server's accept time, send timing, ACKs and flow control reach it only through the fail cells.
One run of the network certifier's code gives both the cells the client receives and its record of them
(`Security/Definitions/NetworkCertifier/Tap.lean`, under `trusted-host`). NCI's transfer gates are these cells' bits,
whose number per window the schedule fixes. The schedules are charged `log2 #schedules`, so a schedule that came to
depend on the server is still counted. The content term comes from the audits, not from a premise:
`SecurityProofs.PeriodLeakAccepted` composes each window's sampled leak bound at the network certifier's rows with the
client's view read back from its record, over a monitoring period of sampled audits. Each window adds its unregistered
inputs' bits, log2 of its possible faulty sets, their committed bits and the timing, and each window's audit fails with
probability at most δ + ε_ks + ε_bind + Pr[B_tap].

The inference-only policy charges an isolation block the ingress covert capacity from the last wipe before it starts,
plus what survived that wipe. `capacity.k_share` expresses bits as a share of K, which the caller passes in bits
(`calibrate(k_bits=…)`, `--k-bits`): 9.12e7 under NCI's inference-only policy (α·Γ = 9.6e−11 × 9.5e17). The
calibration report records the K it used.

Every statement in the table's Lean column is a guarantee in `Security/lean-audit.json`, frozen as no-function: each is
stated over the model, not the network check's code (`Security/Proofs/NetworkCertifier/README.md`, "The records"). Four
are the bounds `calibration.calibrate` reports: egress over a monitoring period
(`capacity.egress_bits_per_day_one_clock`, `SecurityProofs.EncardDecodableLeConstantRate`) and per window
(`clock_sync.bits_per_window`, `SecurityProofs.EncardDecodableLeConstantRatePerWindow`), and ingress over a monitoring period (the
K charge, `k_charge.share_of_k_by_span` from each link's `rate_bps`, `SecurityProofs.EncardIngressObsDecodableLe`) and
per sampling block (`envelope_bits_per_window`, `SecurityProofs.EncardIngressObsDecodableLePerWindow`, whose interaction
is any link's). The steps of their proofs are lemmas (`Security/Proofs/NetworkCertifier/Lemmas.lean`): proved and
audited, with no record.

The shipped schedule's egress bounds are proved in Lean for `constantRate`, the Program of `schedule`, with no
dataflow hypothesis: for every frame function, sync set and operator, over a monitoring period within the W windows with advice
(n ≤ W·T), since those are the windows the audit checks. The clock-sync bound is 7 bits per window at 16 × 8 sync
values, summed from the run's start (`SecurityProofs.EncardAccSeqsLeConstantRate`). Under J with one sync
value it is 0 bits of timing (`SecurityProofs.EncardAccSeqsLeConstantRateFixedClock`). A monitoring period within the first m
windows is charged only those m windows (`SecurityProofs.EncardAccSeqsLeConstantRateWindows`), and a decoder of the
wire recovers no more (`SecurityProofs.EncardDecodableLeConstantRate`).

The ingress bound is proved in Lean, for a decoder in X that reads the wire (`SecurityProofs.EncardIngressDecodableLe`)
or the records (`SecurityProofs.EncardIngressObsDecodableLe`). It counts `(r_in + 1)^T` occupancies per ingress
sampling block (times `D^T` for jitter, 1 under J), given the frames' content, on runs whose statuses are all complete
(`StatusComplete`). The reference audits ingress statuses (see above), so an accepted verdict on every link of ∂X
establishes `StatusComplete` for the windows audited. That is the acceptance hypothesis of every ingress guarantee:
`SecurityProofs.EncardIngressObsDecodableLe`, `SecurityProofs.EncardIngressDecodableLe`, their `Contents` versions and
the two per-window forms. Over the first m whole windows of a run (n = m·T; for the K charge, the run from the last
wipe) the count factors per sampling block: `(r_ℓ + 1)^T` on each link in each window, `T·log2(r_in + 1)` bits, which is
the K charge's unit (`SecurityProofs.EncardIngressObsDecodableLePerWindow`, and
`SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow` for the `Contents` version).

The `Contents` versions drop the one given content: messages whose frames follow some content assignment in a set `Cs`
number at most `#Cs` times that count. So Lean proves, for the set `Ms` of messages counted, `log2 #Ms ≤ log2 #Cs` plus
the timing bits (`SecurityProofs.EncardIngressObsDecodableLeContents`, and
`SecurityProofs.EncardIngressDecodableLeContents` for the wire). The K charge adds content bits, charged as reads, to
timing bits. Taking `log2 #Cs` to be those content bits is an accounting step, not part of the theorem. It is exact
under fixed-length frames (W3): N frames of B bytes have `2^(8·B·N)` contents, so for `Cs` all of them, `log2 #Cs` is
the `8·B·N` bits read.

## The schedule's declared bound and the coins

The two terms of the charge that the capacity bounds leave open, the content advice `A_c` and the coins (D7), are fixed in
the trusted text: the last section of `Security/Definitions/NetworkCertifier/Guarantees.lean`'s documentation. The coin
link that carries the coins: [protocol/coin-link.md](protocol/coin-link.md).

## Advice for arbitrary serving: what it can carry

The goal is network accounting for arbitrary vLLM usage: no FIXLEN, no fixed request size, and
real EOS. The registered thing is a precommitted ahead-of-time compiler. After serving, the prover hands it an
**advice** string that encodes everything serving made unpredictable: in advice v0, each request's lengths and finish,
each step's batch rows, and each step's release bucket. The compiler turns the advice into each step's circuit, and the
certifier certifies the I/O. Sampled proofs check two things: that the committed values pass the compiled circuits (the
step units), and that they agree with the certified I/O (the tie units). N2 holds each window's egress counts to the
ingress counts and the advice. The advice's length is the leak, charged as `A_c` in the accepted-runs bound.

This section attacks that claim through its parts: the advice codec (`advice.py`, with `encode`,
`decode` and `bits`), N2's advice mode (`advised`), the I/O tie unit (`io_tie`) and the compiler
(`verity_vllm.program.predict.compile_steps`). It gives the argument, every assumption the argument rests on, what
each attack costs under the design, and the adversarial cases each part's tests must carry.

### The argument

The notation is for one audit of W windows:
- I is the certified ingress: the requests' bytes and counts. It is charged on ingress, as reads and timing (the K
  charge), and nothing below charges it again.
- w is the weights, under a root registered before any coin.
- `compile` is `compile_steps(registered, ·)`. It is registered with the codec, a cap c on the advice's charged bits,
  Σ_sync and (T, r, B, τ_b, Q), all before the audit's first window.
- a is the advice the prover registers, and 𝒜_c is the set of advice strings whose charge `bits(a)` is at most c.
- The units are the step units of the compiled circuits, the tie units, and N2's units (one per sampling block).

**The claim.** Given the ingress, the registered items fix a set of egress observation sequences with at most

    #𝒜_c · #Φ · #Σ_sync^W · D^n   elements,   that is   c + 1 + I_ε + W·log2 #Σ_sync + n·log2 D bits,

and the audit accepts a run whose sequence lies outside it with probability at most δ_audit. Here Φ is the influence
of the faulty proof blocks (step 3 below), I_ε = log2 #Φ, and δ_audit = δ + ε_ks + ε_bind + δ_in is the error of
`audit_influence` (`Security/Proofs/Flock/Soundness/Audit/Influence.lean`).

**Why it holds.**
1. **Fix a.** Then every unit's circuit is fixed, because `compile` is a function of the registered items and of a. So
   for each a the audit is the one-stage audit of a fixed circuit, and `audit_influence` applies: the audit accepts
   while the values it delivers lie outside Φ_a with probability at most δ_audit. Φ_a is the set of values reachable
   when the faulty proof blocks form a set whose acceptance probability is at least δ, and it is fixed before the draw.
2. **With no faulty proof block, the egress is a function of I and a.** Every committed value is the forward evaluation of the
   compiled circuits from I, w and a, because every value a circuit reads has exactly one writer (A2). Each tie unit
   then holds its frame to the canonical frame of the committed token at the (request, position) the advice maps it to.
   N2 holds each window's counts to the counts that the ingress counts and the advice derive. So the egress rows are
   the Program `constantRate T W ss frames` with content advice a: `frames ins a ℓ` gives each frame's payload, anchor
   and logical index.
3. **A faulty proof block adds its influence.** Take the content advice of step 2 to be (a, φ), where φ is the faulty
   proof blocks and
   what they let through, so that Ac = 𝒜_c × Φ.
   - A wrong step unit lets through its outputs: its public output, at most 16 bits, and its committed ones, their
     bits, plus the choice of which unit is wrong. That is `card_influenceSet_le_harm`'s `H + log2 Σ_{j ≤ K} C(n, j)`.
   - A wrong tie unit lets through the frame it was to hold, not its one-bit verdict. That is 8·B bits, or about
     log2 V for the token alone if an exhaustive check holds every frame to the canonical form (attack 8).
4. **Count.** `SecurityProofs.EncardAccSeqsLeConstantRate` with this Ac gives `#Ac · #Σ_sync^W · D^n`, so no new
   counting guarantee is needed. What is new is the bridge from steps 1 to 3: that an audit accepted by sampled proofs
   is, except with probability δ_audit, `Accepted` for some (a, φ). `Accepted` is the exact binding, in which every
   recorded row is the Program's. An audit that recomputes the grid gets it, and a sampled audit does not.
   For the circuit's Program, `SampledAuditAccepted` (proved) is that bridge with φ the faulty proof blocks' committed
   gates, and `SecurityProofs.EncardAccSeqsSampledLe` is its count: φ costs those gates' bits plus log2 of the number
   of possible faulty sets. `SecurityProofs.SampledLeakAccepted` composes the two, with a pinned at the registered inputs.
5. **The advice set.** `#𝒜_c ≤ 2^(c+1) − 1`. `encode` is injective (A5), so distinct advice values are distinct bit
   strings of at most c bits. A prefix-free code gives `#𝒜_c ≤ 2^c` by Kraft, with the charge the decoder's own length
   of the symbols it decoded, whichever of a message's codewords the prover sent.

The prover's choice of a is free inside 𝒜_c. Acceptance is existential (`AcceptedUpTo`: some advice explains the run),
so the bound counts every string the audit would admit, whenever and however it was chosen. That makes the advice's
length the right charge. It is also why advice that is not what serving did costs nothing more (attack 7).

### What it rests on

Each assumption, and what breaks without it:
- **A1, a deterministic compiler.** `compile` is a pure function of the registered items and the advice, pinned by
  digest before the run. It reads nothing the prover chooses later.
  - The row's pin and the device pin (`predict.compiled`'s kernel configurations, autotune winners, cuBLAS routes and
    fused-MoE configurations) are registered items. The prover chooses them before the run, so they cost their size
    once, like a hidden profile list (`Security/Definitions/NetworkCertifier/Guarantees.lean`, "The declaration"), unless the
    auditor fixes them from a reference device of its own.
  - The circuits are the exact silicon semantics. There is no tolerance: a band of t accepted values on a checked value
    frees log2 t bits there. No input names a reduction order, an atomic's interleaving or an autotune choice.
  - A kernel that is nondeterministic on silicon makes honest runs fail. That is a completeness failure, and its fix
    is a deterministic kernel or a charged advice field, never a witness.
- **A2, the Commit covers every value the circuit reads, each with one writer.** A value a compiled circuit reads is one
  of these:
  - a certified input, through an exhaustive tie. The prompt's token ids are the registered tokenizer and chat template
    applied to the certified request body. The request's parameters (temperature, top-p, seed, maximum tokens, stops)
    are read from the body;
  - the weights' committed root;
  - a constant the advice fixes: positions, shapes and the wiring;
  - the committed output of exactly one unit earlier in the advice's order. A decode row's input id is the previous
    position's sampled token. A KV entry at (request, position, layer) is its writer's output. A cache hit names its
    donor, whose token prefix equals the request's.

  `compile` refuses a read with no writer. Without A2, a value nobody checks feeds the circuits, and every draw passes
  (attacks 4 and 6).
- **A3, the tie and the step units are drawn at ε at δ, not all.** N2's units are `All`, and so is the compile slice
  on every advised window, never sampled: its carry chain binds only proved units and its scan pins `en`
  ([protocol/compile-slice.md](protocol/compile-slice.md)). The step units add I_ε at
  probability δ_audit. A wrong tie unit adds 8·B bits unless the tie is `All` too (attack 8). The bound holds on
  accepted audits only, as every capacity bound here does: a rejected window may already have carried up to its unbound
  capacity.
- **A4, the advice is bound to the run before the coins.** The registration that the challenger receipts before its
  draw holds the advice's canonical digest (or every compiled Program's digest), with the run, the link and the windows
  in its context. Every consumer (the compiler, N2 and the tie) reads that one advice. If the advice can follow the
  draw, the prover picks, after the draw, an a under which the drawn units pass. The error then grows to as much as
  #𝒜_c · δ, which is vacuous. Each window's advice must precede its own units' draw. It may follow an earlier window's
  draw, since an earlier window's faulty proof blocks are counted in Φ by their exits, whatever later circuits read them.
- **A5, the charge is the encoding every consumer decodes.**
  - `bits` counts the canonical, self-delimiting encoding of every field any consumer reads, and no consumer reads
    anything else. A field read from elsewhere (the Commit's row metadata, a default, a field an encoding stopped
    sending) is a free channel, unless the consumer recomputes it or checks it against the decoded advice.
  - The charge is a function of the decoded advice: its canonical length, or the decoder's own length of the symbols
    it decoded. Then the count holds whatever bytes the prover sent. Wherever the raw string itself is consumed (the
    digest the registration binds, or a seed or coin derived from it), `decode` must refuse whatever `encode` would
    not produce: padding bits, a trailing byte, a non-minimal integer, an unused enumeration value. Or the auditor's
    side re-encodes the decoded advice rather than forwarding the prover's bytes, as the Flock firewall relay
    re-encodes a round.
  - Each window's part is self-delimiting, and the windows' charges sum to the whole encoding.
  - Each field is charged to a window no later than the first window whose rows it can change. Then the first m
    windows' charges bound the first m windows, which the per-window form needs, as `ExpectedRowCausal` gives it for
    the syncs. A prefill step that releases nothing is still charged.
- **A6, the certifier's clock.** The buckets are the certifier's, under `trusted-host`, W1 to W6 and J (D = 1
  is no jitter), and the advice's release buckets are certifier buckets. Nothing else in the accounting reads a time.
  The challenger's readings of when a registration or a proof arrived (`received_at`) are the prover's choice within
  their deadlines, so they stay out of anything published, or they are charged.
- **A7, the advice set is fixed before the run.** The codec and the cap c are registered before the audit's first
  window, and the audit refuses advice over the cap, as it refuses a sync outside Σ_sync. `A_c = log2 #Ac` in the
  trusted text needs the set fixed before the run. Without a registered cap, the realized charge is sound only in the
  indexed form: for every c, the runs whose charge is at most c number at most 2^(c+1) times the rest.

The claim does not say three things: that the advice is what serving did (attack 7), anything about rejected windows,
and the ingress terms, which stay the K charge's.

### Attacks

Moved to [protocol/arbitrary-serving-attacks.md](protocol/arbitrary-serving-attacks.md), with the adversarial cases each
part's tests carry.

## Calibration (`calibration`)

Moved to [protocol/calibration.md](protocol/calibration.md).

## Advice v0 (`advice`)

Moved to [protocol/advice-v0.md](protocol/advice-v0.md).

## The replay's state chain (`replay_chain`)

Under advice v1 (the next section's end) the compile unit replays the scheduler one step unit per engine step. Each
unit is `ReplayStep_v1` (`replay_step`): the committed queue state, the next A certified arrivals and the
next A decisions go in, and the step's rows and next state come out. The chain hands the state from unit to unit as
the state rows of one produced value a window (`one_stage.produced`, value `replay-state`, `hm96-sha512/row/v2` bit
rows), beside a window row (value `replay-window`). Each unit reads its rows at public positions, and no gate hashes a
row: the Lean verifier checks each read's path.

**The step's cursors are private.** The step reads its arrivals and decisions at the scan's cursors, which move by its
`taken` and `used`, and where a step stands in the certified arrivals or the window's decisions is the schedule
itself. A registered read sits at a position the verifier derives, so no unit can read the arrival record at the
cursor. So:
- once a window, `ReplayLoad` writes the window row: the certified arrivals not yet taken, then the window's own (K of the network
  certifier's slots), as `queue` (at most Q), and the window's decoded decisions as `dec` (at most D), all read at public
  positions;
- every step unit reads that one row, and its state row carries the cursors as private heads `qh` and `dh`.
  `ReplayAdvance` checks the step's A arrivals and A decision bits are the window row's at the heads. It finds them with a
  log shifter that computes only the A entries it returns: about Q + A·bitlen(Q) muxes an element bit, where moving
  the whole queue would take Q·bitlen(A).

**The rows'** layouts are `replay_chain`'s (`row_fields`, `window_fields`). The state row's state group is
ReplayStep_v1's state input and next-state output, bit for bit, at the shape's AS and EW (`Shape`'s two statics):
with EW, `run_pfx R×EW` after `run_comp`; with AS, `run_fin R×1 | run_cap R×1 | wait_cap W×1` after `free`. It adds
the heads `qh` and `dh` and `steps`, the window's executed steps still to come. The window row is the queue (at most
Q) and the decisions ``dec`` (at most D) with ``left``. With AS an arrival carries its cap (O = max_tokens): `q_cap
Q×1` after `q_O`, as the step's `arr_cap` and the warden slots' `a_cap`. In every row the valid elements of each
group are a prefix and the rest are zero bits, and `decode` and `decode_window` refuse any other row.

**A window** is the state rows S_0 … S_{P+1}, with P the public cap on its steps, and two window rows: the last
window's and its own. Unit 0, `ReplayLoad`, reads state rows 0 and 1 and both window rows. Unit u ≥ 1 is step u:
ReplayStep_v1 on state row u, with `ReplayAdvance` checking the step's arrivals and bits against window row 1 and
checking state row u + 1. Steps past the window's executed count are idle: the next row is the row, and
`ReplayAdvance`'s `active` = 0 drops the unit's rows.
- The load refuses a state row 0 that leaves steps to run or has read past its window row's decisions, a queue past
  Q, more than D decisions (the carried ones and the window's) or P steps. A step refuses arrivals taken past what is
  left or past A, and decisions read past what is left or past A.
- **The carry.** Window w's state row 0 is window w − 1's last state row, and its window row 0 is window w − 1's window
  row 1, each under the same salt (`commit_window(carry=…)`), so the verifier checks that their openings b‖c are equal
  (`check_chain`, `C-carry`). Decisions a window leaves unread carry: the load writes the window's after the last
  window row's unread ones, as it does the queue's arrivals. A5 ("The windows", below) charges a decision to the
  earlier of its step's window and its arrival's ingress window, up to two windows before its step's: at T = 20, 5 of
  async-kv-pressure's 17 decisions and 5 of sync-kv-pressure's 15, which the chain carries from window 0 to their
  steps in windows 1 and 2 (`benchmarks/network_e2e/encodings/tests/test_v1.py`).
- **The end.** `ReplayClose` reads the session's last state row and window row and refuses a session that ends with a
  step to run or a decision unread (`close`). The compile unit charges each decision in the window that commits it,
  A5's, in commit order, which is the order the scan reads them in. So a carried decision is charged where it is
  committed and read where its step is, and none is left uncharged or unread
  ([protocol/compile-slice.md](protocol/compile-slice.md)).
- **The start.** The session's first rows are the empty engine (`empty`, with NB − 1 free blocks) and the empty
  window, under the public salts `empty_salts`, whose openings the verifier computes (`C-first`).

**Cost** (`size`). A step costs `ReplayAdvance`'s ANDs: the two fronts, about (Q + A·QW)·(1 + IW + 2·LW) and
D + A·DW, and about 2·width for the idle select and the equality. It also costs three reads and one state row. A
window costs P steps and one `ReplayLoad`, P + 2 state rows and two window rows, and 3P + 4 reads, each a frame-v3
path of depth ⌈log2(P + 2)⌉, or 1 for a window row. A session adds one `ReplayClose` (13 ANDs at the sync
KV-pressure trace's shape) and its two reads.

## Advice v0's validity (`advice_check`)

The compile unit's first part ("The charge, and what is committed", above), that the committed bit string decodes to
a well-formed v0 advice, is the Boolean Definition

    AdviceCheck_v1{L} (bits: Array<L, Value<1>>, length: Array<LW, Value<1>>) -> Value<1>

with L the public cap in bits, a multiple of 40, and LW = bitlen(L / 8). `bits` is the string in stream order and
`length` its byte count, least significant bit first; bits past 8·length aren't read. The output is 1 exactly when
`advice_check.reference` holds: length ≤ L / 8, and `decode` takes the bytes up to three refusals the circuit doesn't
make yet (`DEFERRED`). It makes every other refusal of `decode`: the grammar (version, minimal `uvar`s, the least
widths, no truncation, zero padding under a byte, no trailing byte), UTF-8 ids, the lengths ≥ 1, first-row order, every
request admitted, a first row that doesn't resume and starts at its hit, hits only on admitting rows and their sources
summing to them, adjacent sources of different requests, own KV only on a resume, sources admitted by their step's end,
continuity, no row past the known tokens (so at most one past the last output), the derived `kind`, no resume of a
finished request, no request twice in a step, and exactly `output_len` kept outputs.

**Deferred**, refused by `decode` and not yet by the circuit:
- **Unique ids.** Exact equality of variable-length strings needs the ids sorted lexicographically (prefix doubling, a
  sort per doubling of the longest id), or a digest, which would hold only computationally.
- **A source's KV computed there** (`advice._covered`): whether a run of positions lies in one interval its request
  had computed by that step, a query over position and time per request.
- **Release buckets non-decreasing per request.** The k-th release belongs to the k-th kept output in stream order,
  which only the replay knows: matching them takes a second sort, by position, and up to five releases can end in one
  block.

**Under advice v1 this isn't what the compile unit proves.** At 8.7 × 10^10 ANDs for a 60 s window's 10.6 Mbit, v0's
validity is too costly to be the compile unit's proof. The replay runs vLLM's scheduler exactly on every corpus trace,
and advice v1 is what the certified record, the replay and the derived fields leave, a residual of a few hundred bits
a window coded prefix-free, and `v1.decode(bytes, certified record)` gives I1 v0. So the compile unit
takes v1 and produces v0 by replay, and most of v0's validity holds by construction. The three deferred rules are then:
- unique ids are certified, by the network certifier's admission order and the request records;
- a source's KV is the derived fields' canonical source rule, checked against the tokens;
- release order is the network certifier's fixed schedule.

`AdviceCheck` stays as v0's reference check, and as the fallback for advice that comes without a replay.

**The circuit**, three stages:
1. `AdviceParse` scans the bits five at a time through a bit-serial parser. Its state is the field being read, its bits
   left and value so far, the header's counts and widths, the counters of the records left, and the in-stream checks.
   A row is at least five bits, so each block ends at most one request record or row, which it emits as an event.
2. Batcher's odd-even merge sort orders the L / 5 events by (request, position), one batch of `AdviceSwap` per layer:
   the network on the next power of two, less the comparators that reach past the events.
3. `AdviceReplay` scans the sorted events with one request's state, as `advice._replay` replays it: each request's
   record, then its rows in stream order.

**Cost** (`size`): ANDs = (L / 5)·(parse + replay) + C(L / 5)·swap + the ends, with C(n) ≈ n·log2(n)² / 4
comparators: 470,116 at L = 240 and 1.49 × 10^8 at 40,000. The 60 s llama8b-shape window's v0 advice, 10.6 Mbit,
would cost 8.67 × 10^10, 70% of it the sort. A public cap on the rows, the sort's permutation as a private input or a
narrower w_len would each cut that, each a change to what the Definition states.

## Advice v1 (`benchmarks/network_e2e/encodings/v1.py`)

v1 is the advice the compile unit takes: I1 v0 sent as only what the certified record, the scheduler replay and the
derived rules leave. `v1.encode(advice, record, registered) -> bytes` and `v1.decode(bytes, record, registered) -> I1
v0` with `decode(encode(a)) == a`. `decode` is total: it refuses by name every string that isn't the canonical
encoding of an advice under the record, and a string past the cap ℓ unread. It stays in the encodings until its circuit
lands.

**Certified, 0 bits**: the network certifier's signed records under design v3 and the id I/O they carry,
which the compile unit holds as witness.
- The ordinals: the k-th connection to arrive has ordinal k (the network certifier's admission, every cell carrying its ordinal).
- Each request's ingress bytes, one id request (`active.id_request_fields`): `user` (I1's `id`), `prompt`
  (prompt_len and the ids), `max_tokens`, `temperature`, `top_p` and `seed`. Its release is a, the bucket of its
  connection's last ingress cell: the server holds the whole body from a on.
- Each output's id, the `token_id` of the id event that carries it (`active.id_event`, tagged by its request's
  ordinal). These give `output_len`, and through the registered stop ids, `finish`.
- Each output's release bucket, that of the egress cell carrying its event. Cell i of the connection released at a
  is in bucket a + D + ⌊i/r⌋, a function of a alone. v1 doesn't read the `fails` record: a failed connection has no
  advice.
- *The reference order* is the network certifier's release order: the requests by (a, ordinal) (`v1.release_order`), each
  request's last ingress cell landing, which the network certifier's signed record holds. The admission order (by s) is the
  client's send order, and the engine reorders far more against it: behind the network certifier, the trace corpus has 13,424
  inversions of 499,500 pairs against admission order on its heavy trace and 590 on small KV, against 2,970 and 3
  against release order.

**Replayed, 0 bits**: the replay (`replay`) runs vLLM's scheduler under the registered engine, its KV pool
`num_gpu_blocks` included, over the requests in the engine's arrival order (sent, below). It gives every step's rows
(request, start, n), the new and resumed requests, the cache hits (its block pool's LRU, eviction included) and the
preemptions. It needs one decision per open step: whether the next request had arrived.

**Derived, 0 bits**: the derived fields' walk (`derived`) recomputes every start and n and checks them against the replay. It
gives the batch order (Model Runner V2's stable sort by n), I1's numbering by first row, `kind`, the canonical
`sources`, and the request records (`prompt_len`, `output_len`, `finish`). While the cumulative pool bound holds (sync
scheduling, and ⌈(P + O)/B⌉ blocks for each request admitted through the step, that step's included, at most
NB − 1), nothing has been evicted, and every hit is the longest cached prefix, at 0 bits. The resumes and the new
requests' order that the walk would otherwise read are the replay's.

**The arrival order**, recovered from the advice's rows (`v1.arrival_order`). FCFS admits new requests in arrival
order, so arrivals go by first step, and within a step by the scheduler's order. Under the scheduler's row order that
is the step's own. Under the batch order (a stable sort by n) a step's new requests of different n don't show it, so
the order is the topological one that:
- puts a request after any of the step's new requests its hit's sources name, and the step's partial prefill last;
- keeps each equal-n chain's batch order, in the step itself and in every later step, among those of the step's new
  requests still running since. The scheduler keeps its running requests in admission order and puts a resumed one at
  the end, so equal-n decode rows show it. It reads later steps until the order is unique or fewer than two are left.
- takes the earliest in release order where the rows leave a choice, so an order the schedule never shows costs
  nothing.

**Sent**, as v1's windows (`code_v1.Window`), and nothing else:
- *The decisions*, one per open step: whether the next request had arrived.
- *The order.* The engine admits in the API server's body-read order, which comes after the release and overtakes it
  under load: requests overtake each other by length, and the HTTP layer reorders on top. v1 sends the permutation
  from release order to the arrival order as its deviations (`v1._deviations`). For each arrival p that isn't the
  earliest released request not yet arrived, it sends p and its rank r ≥ 1 among those, as code_v1's order entry
  (p, (r,)): positions gap-coded and increasing, ranks zigzag-coded, `order_len` registered at 1 (`v1.CAPS`). The sum
  of the ranks is the inversion count. The decoder (`v1._arrival`) refuses rank 0, a position or rank past the
  requests left, and a repeated position.
- *The hits past the bound.* Past the bound, eviction may shorten a hit, so the hit is the prover's choice, and v1
  sends it. It sends every prefix-cache lookup the replay's scheduler reads at a step past the bound, in the order it
  reads them. Those are each waiting candidate its loop reaches (the one whose allocation ends the loop included) and
  the next arrival `would_admit` checks before a decision is asked. Each goes as its shortfall: the longest prefix a
  cache that never evicts holds, less the hit, in blocks. That is 0 unless the replay evicted. Every lookup is then
  derived or sent, so no prover choice goes uncharged, and the encoding is injective on the whole schedule.

*Not sent: the KV pool.* `num_gpu_blocks` is public and registered before the run, in `Registered.engine`, never in
the string. It used to be a header the prover set, about 20 bits, and that undercharged: a bigger declared pool keeps
the no-eviction bound longer and buys free derived hits (the phase-7 attack: 103 hits charged at a pool of 80 blocks
against 100 at 160). The same string under another registered pool decodes to another advice or is refused.

*The order's tie.* `v1.decode_full` returns `Decoded(advice, streams, arrival)`. `streams[q]` is the network certifier's ordinal of
I1 request q's connection: the order permutation (`walk.order`) through the arrival order, a stable named field.
The I/O tie's egress tie checks each cell's stream against its request q through it. That replaces v4's stream == q, which
refuses honest traffic wherever the engine's order isn't the network certifier's.

*The windows* (A5). Step s's window is the earliest window that releases an output at step s or later (the last window
when none follows): the first window whose rows a value read at step s can change. Arrival i's ingress window is the
earliest of ⌊a_j / T⌋ over it and every later arrival j. That never decreases, even where the arrivals deviate from
release order.
- A sent hit at step s goes to step s's window.
- A decision about arrival i at step k goes to the earlier of step k's window and i's ingress window.
- An order entry for arrival p goes to the earlier of p's ingress window and the window of the first step that reads
  p: a decision about it, a lookup of it, or its admission.
- Each min is never later than either rule, so it is at least as conservative for the leak bound as A5's, and every
  value about a request is charged in or before the window the request came in.
- Each field is nondecreasing in the replay's order. So each field's values, window after window, are in the order
  the replay reads them, and the decoder replays them straight through, with no routing.

*The string*, MSB first: `code_v1.encode(windows, book)`, then zero bits to a byte. W is the record's last bucket // T
+ 1.
- `code_v1` codes each window as one terminated arithmetic segment, its counts before its values. So its decoder finds
  each window's end, and v1 sends no length prefix (Elias-gamma(ℓ_w + 1) cost 5 to 22 bits a string).
- A window's charge is its segment's bits (`code_v1.Encoded.lens`, `bits_by_window`), a function of the string alone.
- The codebook is `Registered.book`, a `code_v1.Book` fit on other runs, never this one. Its digest (SHA-512 of its
  canonical JSON) is registered before the first window.
- `decode` runs `code_v1.decode`, which takes only the canonical segments, then the order entries' inverse, then the
  replay, which consumes the decisions in order. It refuses the string unless its windows are the replay's residual
  (every hit, and every value's window) and its order is the one `arrival_order` recovers from the advice it decodes
  to.

The compile unit commits the values themselves, not this string, and the string's segments are only their charge (the
composition below): what the bound counts is the charge, not the committed length.

- *The bound.* `code_v1`'s segments are prefix-free, so the strings are prefix-free for every W, and at most
  2^c strings are charged c bits or fewer (Kraft): the sampled leak bound's premise `hAs`.
  `code_v1.encode` is injective on the windows (`code_v1.decode` inverts it), and the order entries' inverse, the
  replay and the walk give the advice from the windows, so at most 2^c residuals, and so at most that many advices,
  are accepted at a charge of c or less.
- *v0's per-window parts.* Each window's part is self-delimited.
**Composition**, all in one ZK proof under the advice's hiding root:
1. *The values are the committed advice.* The advice's hiding rows hold each window's residual raw, as
   `replay_chain`'s window row: its decisions as bits in the scan's order, at most D a window (4,391 on the corpus'
   largest), in `dec` with their count `left`; its sent shortfalls, at most H, in `hits` with `hleft` ("The rows"
   below); and its order entries (position, rank). No decoder runs in the circuit: the scan reads each field at a
   private head of the chain's (`dh`, `hh`).
2. The scan of `ReplayStep_v1` over the session's steps, from the empty engine. The window row holds a queue of the
   next Q released requests not yet arrived, in (a, ordinal) order, Q a public cap like H, and arrival p takes the one
   at its order entry's rank (0 where none names p; a rank of Q or more is refused). Q = 128 holds the network certifier's traces'
   largest, 75 (heavy; p50 7) and 2 (small KV). The chain fills the step's A arrival slots from the queue, each
   `arr_id` its arrival ordinal, so `ReplayStep_v1` needs no change for the ranks. The decisions are the scan's bits,
   and each lookup's hit is h* less the committed shortfall. The chain reads every committed value and none past
   them, so the values are exactly the asks, the deviations and, past the bound, the lookups. Under async scheduling
   (AS) the certified O forces each lag row, at no decision (`ReplayStep_v1`'s "Async scheduling").
3. *The charge.* The registered code runs forward over the same values: per window, `code_v1`'s window over
   the values charged to it, in `code_v1._window`'s order, on the encoder ("The encoder in the circuit"
   below). A decision's context is min(run, 2), run the 0s since the last 1 in the decision stream, across windows:
   a function of the committed decisions, which the encoder keeps in its own state. The window's coded length ℓ_w
   (`code_v1.Encoded.lens`) must be at most its declared charge c_w, a protocol output. No codeword is committed, so
   nothing reads a segment at a data-dependent cursor, no ask forks the replay, and the bound above holds as it is.
   `CompileSlice_v1` is 1–3 for one window, proved in ZK, with its stand-ins (Open).
4. The walk over the rows, which gives I1 v0.
5. `AdviceCheck_v1{L}` specifies what 1–4 give. The tests hold every advice v1 decodes, random and
   bit-flipped strings' too, to I1's `advice.decode`, whose refusals that circuit makes. So it holds wherever 2 and 4
   derive their rows rather than take them as witness, and needn't run at v0's length.
6. `compile_steps(registered, advice, ingress)` turns the advice into the Programs, with `ingress` the certified
   bodies by id.

**The encoder in the circuit** (`arith_encode`: `ArithCountStep_v1{K, BIN, CW}`, `ArithCountFinish_v1{CW}` and
`BookKT_v1{C}`, with their references as specs; its docstring has the mechanics).
- *The charge is the coded length.* The scan reads the committed values (item 2). The encoder runs `code_v1`'s code
  forward over them, once a window from `Tally()`, and counts the bits the window's segment would hold: each symbol's
  scalings, taken at once, and the termination's 2 when used. The total must be at most c_w. No codeword is committed
  and no segment is compared. `encode` is injective on the values and its strings are prefix-free, so at most 2^c value
  lists are accepted at a charge of c or less ("The bound" above).
- *A window's run*, in `code_v1._window`'s order: n_arrived = `left`; each decision slot j < D at context min(run, 2),
  its (f0, f1) from `BookKT_v1{3}`, coded by `ArithCountStep_v1{18, 1, 20}`; n_hits = `hleft` and each shortfall;
  n_resumes (0 under v1); n_order and each order entry's position gap and rank step; n_lag (0 under v1); then
  `ArithCountFinish_v1{20}`. Slots past a count run with en = 0. A refusal (a symbol
  code.py refuses, more than K scalings, a count past 2^20, a charge past c_w) fails the unit, and K = 18 never refuses
  a symbol.
- *The context and the frequencies are the circuit's, never witness.* The context is a 2-bit run counter carried from
  the replay's position: an arrival resets it, and a 0 raises it, saturating. (f0, f1) come from counts that start at
  the registered book's public priors and that each coded decision raises in the circuit, across windows. An integer's
  (cum, freq, total) come from the book's public bin tables. A context or a frequency of the prover's choosing would
  pick the code per value and charge any decisions next to nothing. The compile slice computes both in the
  circuit too, at its own model (Open), and charges the position after `ArithEncodeFinish_v1`, which counts the same
  bits.
- *Integers.* `code_v1` codes an integer as `code.Binned` does: a bin symbol from the book's table (or the escape and a
  uniform one), then a uniform offset, each a general (cum, freq, total) that `ArithCountStep_v1{18, 0, 20}` counts.
  `ChargeCount_v1{N}` lowers a decisions-only window's counts, the slice's one call (docstring), and
  `ChargeCount_v2{N, H, O}` (`window_charge`) the whole window's, its hits and order entries with their counts too.
- *Held to the coders* (`benchmarks/network_e2e/encodings/arith_vectors.py`): `code_v1` over v1's windows
  (`v1.residual`, a book fit at `v1.TABLE_CAPS`) and the walk's still-sent fields. After every symbol the count's
  interval and `used` are code.py's and its count is the bits consumed. Each window's finish gives v1's charge and
  refuses one bit less, each bit's estimate is the book's model's at context min(run, 2), and the bit-checking variant
  refuses every flipped or cut segment. On the 14 corpus traces at T = 600, read with the network certifier's ingress and with the
  client's send as the release (order entries sent), and 6 traces' fields: no circuit case mismatches. The certifier's vectors (the replay's step
  traces, the trace corpus's fixture) hold no order entry.
- *Cost* (ANDs, pinned): a decision 2,580 (the step's 2,243 and `BookKT_v1{3}`'s 337), 2.8% of a
  `ReplayStep_v1{16,64,16}` step's 92,146; an integer's symbol 3,963; a window's finish 41. A window pays for all its
  fixed-width slots. The corpus's largest: light's, 4,396 decisions, 11.4M; smallkv's, 1,466
  decisions and 5,686 integer symbols, 26.3M before the integers' lowering; async heavy's with the client's send, 1,000
  decisions and 1,761 integer symbols, 9.6M.
- *The bit-checking variant* (`ArithEncodeStep_v1{K, BIN}`, `ArithEncodeFinish_v1`, `KTContext_v1{C, NW}`) is the
  alternative, where the committed advice is the segments and the encoder's output must equal each, bit for bit. Its
  decision step costs 3,244, and it needs a reader at a data-dependent cursor, which isn't built.

**The charge's carry across windows** (`charge_chain`, `ChargeLink_v1{N}`): [protocol/charge-chain.md](protocol/charge-chain.md).

**The whole window's charge** (`window_charge`, `ChargeCount_v2{N, H, O}` and `ChargeLink_v2{N, H, O}`, the coding of
the hits, the order entries and their counts): [protocol/charge-chain.md](protocol/charge-chain.md), "The whole window".

**The prefix cache in the scan.** `CacheStep_v1` (`cache_step`) is the walk's prefix rules (`Prefixes`) as a Boolean
Definition: one row's pass over a table of at most N cached units (a unit's U tokens, its parent entry and its owner,
the lowest-numbered request that computed it), compared on tokens, with no hash in the circuit. Each engine step of the
scan (item 2) is one `ReplayStep_v1` and the `CacheStep_v1` calls on its rows, which carry the table from call to call
in `ReplayStep_v1`'s scheduling order:
- *The calls.* Each running row comes first, from its slot's walk entry `pfx` over the units from its start's: it owns
  or appends the units it completes.
  - Each candidate (the W waiting slots, then the A arrivals) then gets a lookup from the root (`admit`, nins = 0),
    always in derive mode (the call's own `sent` = 0). It checks that its wh is the longest cached prefix h*.
  - The candidate's hit h, the input `ReplayStep_v1` reads with U = B, is h* less its shortfall. The shortfall is 0
    while the bound holds, and past it the next sent one, at the window row's head `hh`.
  - An admitted candidate gets a second call, which owns or appends the units its row completes. It starts from its
    hit's entry: the lookup's `trail` entry at unit h − 1, or ROOT when h = 0.
  - So a later candidate's lookup reads an earlier one's units, as vLLM caches a block when it allocates it.
  - The scan skips an empty slot's call: it keeps the table and doesn't read that call's ok. That is R + 2·(W + A)
    calls a step.
- *The owner.* A call's r is the row's request id, the arrival ordinal `ReplayStep_v1` carries (RW = IW). FCFS admits
  new requests in arrival order, so the lowest id is the earliest-admitted request, the one I1 numbers lowest. A hit's
  canonical sources are its lookup's trail entries' owners in the table at the step's end (`cache_step.sources`), where
  an owner may have dropped to a lower request later in the step. The walk (item 4) reads them as I1's `sources`.
- *The derive bit.* Each step takes one bit `sent`: 0 while the no-eviction bound holds, when every shortfall is 0;
  1 past it, when the step reads one sent shortfall per lookup. The bound counts the step's own admissions,
  ⌈(P + O)/B⌉ blocks per admitted request, while the lookups that size those admissions read the bit. So the bit is
  witness, and the step's end checks it against the count: sent = 0 iff the count through this step is at most NB − 1.
  The count only grows, so the bit never returns to 0. Async scheduling sends from the first step.
- *The rows.* These are the exact fields the two owners add, each little-endian, with EW = bitlen(N),
  HW = bitlen(⌈L / B⌉) (`CacheStep_v1`'s), H the sent shortfalls a window (a public cap, like D) and
  HCW = bitlen(H).
  - The replay step (`ReplayStep_v1`'s "Walk entries"): the state gains `run_pfx R×EW` after `run_comp`, the step's one
    walk input is `row_pfx R×EW`, each row's entry after its last `CacheStep_v1` call, and it outputs
    `looked (W + A)×1` after `used`, the candidates whose lookup it read, which `replay_vectors --EW` holds to the
    replay's `_KV.hit` calls.
  - `replay_chain`'s state row gains `hh 1×HCW` after its heads' `steps`, and then one group in
    `CacheStep_v1`'s `_in_fields` order:

        cache    c_valid N×1 | c_parent N×EW | c_tok N×(U·TW) | c_owner N×RW | filled 1×BW | sent 1×1

  - The window row gains one group after `dec`:

        hits     hits H×HW | hleft 1×HCW

  - `hh` is how many of the window row's shortfalls the window's steps so far read. `ReplayAdvance` checks that the
    step's W + A shortfalls are the window row's at `hh`, with the log shifter `dh` uses, and `hh` moves by sent·Σ
    `looked`. `ReplayLoad` writes the window's decoded shortfalls after the last window row's unread ones, as it does
    `dec`, and refuses a state row 0 that left any unread.
  - The empty state's groups are all zero bits. While the bound holds, nothing leaves the cache, so N = NB − 1 entries
    hold every unit computed: 149,212 bits of table a state row at smallkv's (511, 16, 17, 10).
- *Past the bound.* The table never evicts, so it holds every unit the session computed, not NB − 1: 961 units on
  corpus-smallkv's fixture, whose pool is 100 blocks. An evicting cache never hits past what a cache that never evicts
  holds, so a shortfall is at least 0, and v1's encoder refuses a replay hit past h*. `CacheStep_v1`'s own `sent` = 1
  mode, a hit checked only as a cached prefix, is then unused.
- *Cost* (`cache_step.size`): a call walks ⌈L / U⌉ units at about N·(2·(U·TW + EW) + 2·RW) ANDs each. That is 5,856
  ANDs at (N, U, TW, RW, L) = (8, 4, 8, 4, 32), and 38.4M at smallkv's (511, 16, 17, 10, 2048): about 1.1G a step at
  (R, W, A) = (4, 8, 4), where `ReplayStep_v1` is 12,235. The linear table is a slice's size class, and its difftest
  (`cache_vectors`) runs it over two traces' slices at N = 128, against `Prefixes` over the whole traces. A static
  window would let a continuing row's call walk only the units its chunk touches, not all ⌈L / U⌉.

**Scaling: offline memory checking** (the design to build; nothing of it is built). The linear table compares every
entry at every unit, and past the bound N is the session's units. On corpus-smallkv's fixture (NB = 100, L = 4,096,
1,934 steps) N = 961, which makes a unit about 561K ANDs and a call (256 units) 144M. That is about 4G a step at
(R, W, A) = (4, 8, 4), and 7.8e12 for the session. Offline memory checking replaces the table with the session's
memory operations as records, sorted once, as `advice_check` sorts its events.
- *The argument: a sorting network.* A Batcher network in the circuit sorts the records by (key, time), so the sort
  is a permutation by construction, and needs no coin. One linear pass over the sorted records then checks two
  things: each read returns the latest write before it under its key (or none, for a miss), and each append finds
  none. Two alternatives cost more:
  - A permutation argument (a grand product of γ − fingerprint over the step units' records and over the sorted ones)
    needs a live coin γ drawn after both lists are committed, which is a second round through the firewall per
    session. Its fingerprint takes about 20K ANDs a record a side, 3.3e8 at E = 2^13, which is no cheaper than the
    sort.
  - A multiset hash (each record's SHA-512, summed) needs no coin, but costs a compression of about 62K ANDs a
    record a side: about 1e9 at E = 2^13, five times the sort.
- *What each step unit commits.* `ReplayStep_v1` and its `CacheStep_v1` calls stop reading a table. Each unit a
  lookup reads (its hit units and the miss after them) is a read record: key, claimed value and time. The claimed
  value is witness the step uses at once; the lookup's hit is its count of read records that found a value. Each unit
  a row completes is a write record (an append, or an owner dropping to a lower request).
  - A record is the key (parent entry and U tokens, EW + U·TW = 282 bits), the value (entry and owner) and the time,
    about 320 bits.
  - The time is a counter in the scan's state row. The step places its records in the window row's record buffer, a
    public cap a window like H, with the log shifter `dh` uses. So the records are values under the advice's hiding
    root, committed with the window rows the scan already commits.
- *Where the sort runs.* It runs in units of its own after the scan, under the same root: `MemSort_v1{E, w}`, Batcher's
  network at E records of w bits. It is log E·(log E + 1)/2 layers of at most E/2 comparators. A unit runs a block of
  layers on committed records and commits its output; the last unit runs the pass and outputs one bit, ok. At
  corpus-smallkv's size the never-evicting table has E = 8,098: 1,518 lookups reading 5,378 hit units plus the unit
  after each, and 1,202 unit completions, padded to 2^13. That is about 315K comparators at about 700 ANDs each,
  2.2e8 ANDs for the session, plus a 5.7M-AND pass: some 35,000 times less than the linear table. At 2.9M ANDs a layer
  (E/2 comparators), it fits as 91 layer units or as one unit, whichever size class the service runs.
- *The block pool, in the same records.* The same mechanism replays the pool's LRU, which the circuit doesn't run
  yet:
  - The free queue becomes a doubly linked list over block ids, a node per block holding prev, next and its reference
    count, about 50 bits a record keyed by BW = bitlen(NB) bits.
  - The hash map becomes the table above, with a delete record when a block is evicted. An entry's id is the time of
    its insert, so a key evicted and computed again gets a new entry.
  - On corpus-smallkv's fixture the replay makes 2,642 queue operations (1,247 pops, 74 takes of a free block on a
    hit, 1,321 pushes), about 4.5 node records each, about 12K in all. It also makes 7,024 hash reads, 1,202 inserts
    and at most 1,247 deletes, 9,473 in all.
  - That is two sorts at 2^14: about 5.3e8 ANDs for the hash records and 0.9e8 for the nodes, about 6.2e8 for the
    session, 2.8 times the never-evicting table's.
  - It buys the hits. Each hit past the bound becomes the real pool's, derived at 0 bits, and v1 stops sending
    shortfalls. On the smallkv traces, almost all of v1's charge is shortfalls.
- *The gap it closes* (the phase-7 attack's second case). Past the bound the scan holds a hit only to the never-evicting table: the
  hit is h* less the sent shortfall. So any hit up to the longest donor prefix passes, whatever the real pool evicted.
  The attack's probe holds this: a hit no donor computed is refused, and a shorter one is accepted. v1's decoder holds
  each shortfall to the replay's LRU, and the circuit can't, so the circuit accepts strings v1 refuses. The choice is
  charged (each shortfall is sent), so the leak bound holds. But the schedule the circuit accepts needn't be the one
  vLLM's pool gives, and the shortfalls are most of smallkv's charge. Replaying the pool makes the circuit's hit the
  replay's.

**Open.**
- *The charge's pieces* (composition item 3, "The encoder in the circuit").
  - The counting form and `code_v1`'s model (`ArithCountStep_v1`, `ArithCountFinish_v1` and `BookKT_v1`).
    Lowering an integer (the window's counts, each order entry and each shortfall) to its `Binned` symbols in
    the book's public tables isn't built, so the circuit charges the decisions only.
- `ReplayStep_v1` takes the prefix layer (each hit, its evictable blocks, each slot's returns) as witness it bounds.
  The hits are fixed: `CacheStep_v1` derives them while the no-eviction bound holds, and past it each is the longest
  cached prefix less v1's sent, charged shortfall. The evictable counts and the returns stay bounded witness. A
  reference count per entry in the table would fix both while the bound holds. Past it they need the replay's LRU
  block pool in the circuit (offline memory checking's block pool, above), which also closes the hit's gap.
- Past the bound the table holds the session's units, so the scan at a whole session's size needs offline memory
  checking (above).
- *The order's floor.* A server that admitted in release order at the next bucket boundary, a change to vLLM's
  serving, would send no order entries (`replay`'s `ingress="warden"` is that floor on the corpus).
- The 14 corpus traces have no network certifier. The client's send stands in for each request's release (`replay`'s
  `ingress="client"`), and the release buckets are the engine's. The two network certifier traces' `.released` views carry the network
  certifier's (a, ordinal).

## The compile slice (`compile_slice`)

`CompileSlice_v1` is the compile unit's pieces ("Advice v1" above) composed whole for one window, each hit
ReplayStep_v1 reads derived by `CacheStep_v1`'s calls on the certified tokens from the empty cache. Its circuit, verdict
flags, reference (`check`), stand-ins and the phase-9 attack's cases (`phase9`) are `compile_slice`'s docstring. Its step
rows are held to the committed ``srv`` row, which the verifier holds by b‖c to the served rows the Commit side
registers before the draw (`served_rows`), so the phase-10 attack's six passes are refused; at G < P, the served rows'
blocks, its P / G rows (`CompileSliceBlocks_v1`, [protocol/served-statics.md](protocol/served-statics.md)). Its charge is
`ChargeLink_v1{D}` over the decisions the window commits, from its carry-in to its carry-out in the session's carry
chain. In [protocol/compile-slice.md](protocol/compile-slice.md): the charge and what binds its inputs, why every
advised window is proved, a served window at its engine's class (w212 under its AsyncScheduler, 1,021,302 ANDs, and
857,866 at the sync class), and its `--zk` runs through `benchmarks/network_traces/compile_slice_zk.py`.

## The replay at the served shape

In [protocol/served-shape.md](protocol/served-shape.md): a served window as the slice's input (vLLM 8eb1's w212), and
the replay's cost at served shapes and its split into units.

## The active certifier (`active`)

`active.Proxy` runs the certifiers on one boundary link of a live byte stream. It is an asyncio TCP proxy, the I/O shell
around `grid.Certifier`, `schedule`'s handing rule, `commitment` and `audit`, and it adds no record format. Its records are
`grid.Record`s, published raw or through `commitment.commit`, and `Audit.check_link` and `check_ingress_link` read
them.

- **The link** is one TCP connection to the peer outside. Every frame on it is B bytes: a header (kind u8, stream u32,
  length u32), the payload, then zero fill. Padding is B zero bytes. Inside, each stream has its own connection to the
  server.
- **Egress:**
  - **Framing.** The server's bytes on a stream are cut after every `\n\n`, the end of a server-sent event, so a
    frame's content is fixed by the bytes and never by how they were read. Its `id:` is its position in the Program's
    canonical order. An id event (`id_event`, 159 bytes, a 256-byte frame with its header) is one sampled token's
    request ordinal and id in decimal fields of a fixed JSON template.
  - **Compact records** (`Proxy(compact=True)`, which takes `ordinals`). After its HTTP response head
    the server writes one 12-byte unit a sampled token (`compact_unit`, cut by `CompactUnits`): u32be its position in
    the canonical order, then its 8-byte record (`compact_record`), u32be the request's ingress ordinal (≥ 1, its
    stream on the link) and u32be the token field:

    | token field | the output | finish |
    |---|---|---|
    | id, below 2^30 | a token, more follow | none |
    | 2^31 + id (`COMPACT_LAST`) | the response's last, at `max_tokens` | `length` |
    | 2^31 + 2^30 + id (`COMPACT_LAST`, `COMPACT_STOP`) | a sampled stop token, its id the EOS id sampled | `stop` |

    A field from 2^30 to 2^31 − 1 is no record's. The frame is the record zero-filled to B, with no header
    (`compact_cell`), so B can be 8. Its template
    (`compact_placement`) is the record at bytes 0 … 8 and zero after it, and no record is padding, since the ordinal
    is at least 1. The proxy faults the window `malformed` on a unit whose record isn't one (`compact_record_data`) or
    names another request than its connection's, and on bytes left uncut at the stream's end. A cell's ordinal is
    then fixed by the traffic, and its 32-bit token field is all the server chooses in it: under a K-draw tie an
    unchecked cell carries at most 32 bits, against the 247 bytes after the header of a 256-byte event frame. The
    certifier still sees only bytes. The client renders OpenAI's completions SSE from the cells with the registered
    tokenizer (`benchmarks/network_traces/compact_render.py`: vLLM V1's incremental detokenization, byte-equal to
    vLLM's stream text).
    - **Numbering.** The ordinal is at least 1 on the wire, and the I/O tie reads advice v0's request q as ordinal
      q + 1.
    - **The finish is in band.** `COMPACT_LAST` and `COMPACT_STOP` are flags of well-formed records, which the
      certifier commits whatever they say. The certified I/O and the I/O tie hold each response's length and finish
      to the Commit's sampled ids, a client's abort only where the Commit says so
      ([protocol/compact-finish.md](protocol/compact-finish.md)).
    - **What covers it.** The Lean device port models ordinal headers and compact records (`CertifierDevice`'s
      `Config.ordinals` and `Config.compact`), and its difftest holds it to this proxy on honest, salted and restarted
      compact runs at B of 8, 12 and 48 and on each malformed form. The device's Lean properties are stated for every
      configuration, compact ones included. The head model takes a frame as B opaque bytes, which a compact cell is.
      The gate's rehearsal runs compact framing on the auditor's host
      (`certifier-rehearse --traffic varied --compact B`), so with this coverage a compact run there gives a verdict of
      record.
  - **Anchor, index and ready bucket.** A frame is anchored at the egress bucket in which ingress released its
    request's last frame. Its logical index is its position in its stream. Read at time s, it is ready at ⌊s/τ_b⌋ + 1.
  - **Handing.** `Shaper` hands it at `max(h(k), ready(k), last)`, `run_shaper`'s rule under the declared sync of its
    anchor's window. The egress `Certifier` releases min(queue, r) frames per bucket, left-packed, and r − n padding
    frames follow them on the wire.
- **Ingress:** a frame read at time s is enqueued at bucket ⌈s/τ_in⌉ and released by the ingress `Certifier` at r_in
  (`Bucketer`, `calibration.size_ingress_ms`'s model). On a served link (`Proxy(served=…)`, off by default) it is
  enqueued only with its request's other frames, and only while the window, with the request, still ends idle: the
  admission rule ("A served link" above).
- **The salt** (`Proxy(salt=…)`) is off by default, and with it off every record, wire and verdict is as above.
  - **What.** `salt_bits` bits (512 by default, `SALT_BITS`) per PoUW window from the certifier's own randomness
    (`fresh_salt`). PoUW window p is the `pouw_window` = k certifier windows p·k … p·k + k − 1 (k = 1 by default),
    and its salt is drawn at its first ingress tick, t_s. A source of coins handed to the certifier, such as the
    memory challenger's, takes the same place: a function from the PoUW window and the width in bytes to that many
    bytes. A proxy back from maintenance in the middle of a PoUW window has no salt for it.
  - **Into the block only.** At that tick the certifier puts its own salt frame at the head of the queue, ahead of the
    bucket's arrivals and of any queue carried from the window before (`Bucketer(first=…)`, `Certifier.enqueue(…,
    first=True)`), so it leaves in slot 0 of that bucket, r_in ≥ 1, and every other frame keeps its FIFO order. It is
    kind `SALT` (3) on `SALT_STREAM` (2^32 − 1), and its payload is u64be p ‖ `PLACEHOLDER`, 64 zero bytes
    (`SALT_FRAME_BYTES`, 81 bytes, the least ingress frame a salted proxy takes). It takes one of the window's ingress
    slots like any crossing, so r_in carries honest ingress plus one frame per PoUW window. Released first, it makes a
    salted window's request frames in each bucket exactly its counts, less one in bucket 0, so where requests end,
    derived from the signed counts and each request's frames (`id_request_frames`), is the bucket that releases its
    last frame even when a carried queue fills bucket 0; queued behind that queue, the salt would leave a bucket or
    more later and move those ends late by one. The head's counts don't read the order, so its encoding, `head_check`
    and the Lean head model are unchanged. The `Bucketer` follows the salt frame by its place, never by its bytes
    (`Bucketer.own`), so a frame of kind `SALT` read from the link, even one of the same bytes, is never taken for the
    certifier's, and one a fault drops is never released. When the ingress certifier releases it, the proxy writes the
    frame's header, p and the salt (no fill; the header's length is 8 plus the salt's bytes) on a server connection of
    its own. The proxy never reads that connection, so nothing the server writes there reaches egress.
    The egress side never holds the salt or anything drawn from it: the egress grid, its padding and the wire are the
    unsalted run's.
  - **In the record: nothing of the salt.** The salt frame's bytes and its slot are fixed by the window and the traffic,
    so every record, before t_d and after, is a function of the traffic alone, and two salts at the same traffic give
    the same records. The record's formats are unchanged: the v1 digest and the v2 frame tree bind the salt frame like
    any other frame. Neither the Lean analysis nor the Lean ports change.
  - **The sealed slots.** The certifier's record cells are its grids and two sealed slots per PoUW window:
    `_salts[p]`, which holds the salt and which only `reveal` opens, and `_weights[p]` (next item). So the record's
    open cells (`Accounting.Attribution.openCells`, the record minus those slots) are the grids, and what
    the certifier sends into the block, the salt frame's release included, is a function of its record, the sealed
    slots included, and its jitter, as
    `Accounting.TrustedUnits` (W2) asks of a certifier's sends, in `Security/Definitions/Accounting/Physical.lean`.
  - **The weights before the salt.** PoUW's noise seed reads the weights' root, so a credited window needs the
    weights fixed before its salt. A second sealed record cell per PoUW window, `_weights[p]`, holds that order: the
    prover's registration path hands `Proxy.commit_weights(p, root, signature)` the 64-byte frame-v3-sha512 root
    (`WEIGHTS_ROOT_BYTES`) once, signed under the prover key registered with the certifier (`Proxy(prover_key=…)`, any
    `certificate.PublicKey`, LMS in deployment) over `weights_message(link, p, root, context)`: the tag
    `verity-network-certifier/weights/v1`, the certifier's ingress link, p, the root and the deployment's binding of the
    run (`prover_context`, an epoch's stream, say), so a message signed for one run is never taken in another. The
    certifier refuses (`RuntimeError`) without a prover key, on a signature that doesn't verify, once p's first ingress
    tick, the salt's draw, has run or was missed (a proxy back from maintenance after it), and a second root for p, so
    only the registered prover fixes p's root. Nothing the certifier sends depends on it. `Proxy.committed(p,
    registration)` gives it to the verifier on `reveal`'s conditions (as often as asked: it is a hiding commitment), and
    refuses a window whose weights weren't recorded. The order is the certifier's word, under `trusted-host`,
    like the salt.
  - **Revealed on the window's registration, after t_d.** `Proxy.reveal(p, registration)` gives PoUW window p's salt.
    The verifier calls it with the registration the proof system returned for p (`Registration`, from
    `sampled_proofs.service`), and the certifier fails closed (`RuntimeError`) unless the registration's receipt is a
    registration receipt (`one_stage.registration.RECEIPT`) whose digest is the record's own and the registration's,
    and `registered` reads p off the registration: by default `served_window`, which reads the provenance
    `{"kind": "served", "window": p}`, and otherwise the reader the protocol gives (`Proxy(registered=…)`). It also
    refuses while p's last ingress record, window p·k + k − 1's, is unpublished (t_d); once the certifier's own clock
    is more than `reveal_deadline` past p's end (`Proxy(reveal_deadline=…)`, one PoUW window by default), so a
    registration made after the window's work gets no salt and that work is timed; for a PoUW window whose salt was
    never drawn; and for one whose salt it has revealed already, so a second registration naming p, by another stream
    or epoch, gets none. Nothing else publishes the salt.
  - **Asked for and answered under LMS** (`Proxy(challenger_key=…, signer=…)`, both or neither). Deployed, the
    certifier runs on the link, off the prover's node, and the proof system's challenger, which is the
    auditor's, on an auditor host. `challenger_key` comes from the auditor's deployment record, never from the
    prover's configuration, and `trusted-host` covers that record's custody. Each proxy draws a run id at
    construction, 32 bytes of the certifier's own randomness (`Proxy.run_id`; `run_nonce`, `secrets.token_bytes` by
    default, is the tests' hook), and the record gives it to the challenger beside the certifier's key, written when
    the auditor starts the proxy, since a restart draws a new one. The record, not a signed `hello`, because a hello
    would either let any caller spend the certifier's LMS leaves or need a signed request of its own, and the record
    is already where the client gets the certifier's key. With the keys, `reveal(p, registration, request)` and
    `committed(p, registration, request)` take only a request the challenger signed under `challenger_key` (LMS:
    `lms.DEPLOYED`, n = 64) over `reveal_request_message(key_id, run_id, link, p, digest, what)`: the tag
    `verity-network-certifier/reveal-request/v1`, then u16be len(key_id) ‖ key_id, the certifier signer's, the run id
    (32 bytes), u16be len(link) ‖ link, the certifier's ingress link, u64be p, u32be len(digest) ‖ digest, the
    registration's, and u8 len(what) ‖ what, `salt` or `weights`. A request that doesn't verify is refused
    (`RuntimeError`) before anything else is read, and spends nothing: no third party can spend p's one reveal, a
    request for another certifier or another run is never taken, and since the digest binds the registration's record,
    a registration counts only if the challenger signed it, whatever its receipt. The answer is a `Revealed(window,
    value, signature, key_id, run_id)`, signed by the certifier's own key (`signer`, the network certifier's, as its
    `Issuer` signs with) over `reveal_response_message`, the request's fields then u32be len(value) ‖ value under
    `verity-network-certifier/reveal-response/v1`. `Revealed.verify(key, run_id, link, digest, what)` holds only if it
    names that key's id and that run and the signature verifies, so a consumer can key p on the certifier run, (key
    id, run id, link), which `Proxy.certifier_run` and `RemoteCertifier.certifier_run` give alike (link is the ingress
    link the messages bind): two proxies with one link name, whose p's collide, or two runs of one certifier, never
    answer for each other. A signer that refuses leaves p's reveal spent and gives nothing. The salt needs
    authenticity, not confidentiality: after t_d the block's server already holds it, and the verifier fills
    `pouw.salt` as a public input, so nothing is encrypted. A refusal (`{"refused": reason}`) isn't signed: forged on
    the path, it costs the window as dropping the answer would; so does a request copied off the path and sent
    first, which spends the reveal on a salt that is no longer secret. `serve_reveals(proxy, host, port)` serves both
    calls beside `run`, one JSON line per request and per answer, each call off the loop; `RemoteCertifier(host, port, certifier_key, challenger_signer, link, run_id)` is
    the challenger's client, given the certifier's key and run id by the deployment record, whose `reveal` and
    `committed` return the value they verified, as a keyless proxy's do in process, and raise `RuntimeError` on a
    refusal, an answer of another run, one that doesn't verify, or no answer. t_d, the deadline and the once rule are
    unchanged. What still rests on `trusted-host` is the certifier's key custody, its signing key and leaf state
    on its host (`LmsSigner`, one leaf at a time across the threads that share it) and its word on what it signs, and
    the deployment record's custody: the keys and the run id it gives each side. The challenger's key is the
    auditor's, on its own host. A proxy with neither key is the in-process call the tests and replays make
    (`reveal(p, registration)`, which returns the salt): it takes no request, and holds only while called on the
    auditor's side.
  - **The audit's read** (`check_salts(audit, link, records, windows, salts, pouw_window=k, salt_bits=…)`). PoUW
    window p is accepted, at its first certifier window w = p·k, if w has exactly one raw record, which holds exactly
    one salt frame of p, carrying the placeholder, and p's revealed salt has `salt_bits` bits. Anything else, a missing
    salt included, is a `salt` violation at w, recorded with the audit's verdicts. No record binds the salt: the
    revealed salt is the certifier's word, under `trusted-host`. A committed or certified record doesn't show
    its frames, so there a proof opens the salt frame.
  - **Its place, pinned.** The audit takes the salt frame only as the first frame of w's bucket 0, slot 0, where the
    proxy releases it ("Into the block only"); anywhere else it is a `salt` violation at w. The served counts
    (`served.derive`, `NetworkServedCounts_v1`) take bucket 0's first OWN frames for the salt, so the same counts with
    the salt in a later bucket would derive other request ends and still pass the circuit; the audit's pin is what
    ties that premise to the record. A proof that opens the salt frame of a committed record opens it at that slot.
  - **What it costs.** The protocol fixes the salt frame's place in the queue and its bytes, so it adds nothing to the
    ingress count `(r_in + 1)^T`. Its release into the block carries the certifier's coins, which no outside party
    chooses and nothing X sends can change, so it costs 0 bits by the argument the trusted text makes for the
    challenger's coins (D7, `Security/Definitions/NetworkCertifier/Guarantees.lean`). That text states the exemption for coin
    openings on a coin link, and doesn't name the certifier's salt yet.
- **Back from maintenance** (`Proxy(start=k, position=p)`) is off by default, and with it off every record, wire and
  verdict is as above.
  - **What.** A certifier that was down serves from window k of the trusted clock on: each grid's first tick is window
    k's first bucket, its first records are window k's, and the first egress frame it takes is the Program's at
    position p, the frames before p being the earlier windows'. It carries nothing from before it went down: no queue,
    stream, fault or salt.
  - **The windows it was down for** have no record, and the audit gives each of their sampling blocks `no-verdict`, as for
    any missing record. The certifier's audit has no maintenance: excusing those windows is the verifier's plan's, in
    the accounting layer (`benchmarks/datacenter`'s `Maintenance`), which charges everything the block held instead.
  - **Going down.** `windows` ends a proxy's run early: `Proxy(windows=w)` serves windows 0 … w − 1, publishes their
    records and closes the link. A run whose window w is the outage is then `Proxy(windows=w)`, nothing for window w,
    and `Proxy(start=w + 1, position=p)`. Its later windows are the full run's if nothing crosses into window w.
  - **Its limits.** An online prover declares from window 0's frames on, so a proxy back from maintenance takes its
    syncs. The certifier's Lean device port (`verity/core/devices/certifier/lean/`) models the restart (`initAt`), the
    salt, a served link's admission, ordinal headers and compact records; the device's Lean properties (`Security/Properties/CertifierDevice.lean`) are
    stated from window 0 (`init`), so they don't cover a proxy back from maintenance yet.
- **Advice:** the declared syncs, one per window, are given before the run, or an online prover (`Proxy(prover=…)`,
  lag 1 … T − 1) declares each at its window's first tick, before anything is handed, from the frames the shaper has
  taken. The proxy never chooses on a tick: right after tick w·T − lag (window 0's before the first tick) it hands
  `Prover.snapshot()` to a worker process (`active.worker()`, off its event loop and GIL), and the tick at w·T
  declares the result if it is done, else the envelope, logged as unready (`LaggedChoices`). The proxy shuts the
  worker down when its run ends.
- **Faults fail closed** (W6). Each one maps to a status:

  | Fault | Status |
  |---|---|
  | A tick that runs more than `slack` after its boundary, read when it reads its link (the ticks due at one boundary run in one step, ingress first: `active.TICK_ORDER`). Its bucket is still emitted, as padding, so no bucket slips into the next | `monitor-unavailable` |
  | More than a bucket left unwritten on the link at a tick (with a pacer, more than one bucket past its time whose write hasn't ended) | `delivery-refused` |
  | With a pacer, a bucket that reached it after its time, or whose write started more than `slack` after it | `monitor-unavailable`, on the bucket's window: the open one, or the record held until its last bucket's write is reported |
  | With a pacer, a bucket the link refused, and every one after it | `delivery-refused`, on the bucket's window as above |
  | A server that refuses a stream, or the salt's connection | `delivery-refused` (ingress) |
  | A released request not written to its server connection by `slack` after its release tick, nor by the end of its anchor's egress bucket | `delivery-refused`: ingress, on its window if still open; egress, from the check until the stream has ended and every frame that arrived before its end is handed |
  | The shaper's buffer or the queue past Q | `overflow` |
  | A declared sync outside Σ_sync | `coverage-never-established` |
  | An event without a position, with one already seen, longer than a frame, or on a stream with no released request; bytes left unframed at a stream's end | `malformed` |
- **A late forward.** The server answers a request from when it receives it and numbers its events in the order it
  writes them. So a request it receives a bucket late is answered a bucket late, and its events and the ones numbered
  after them aren't the Program's: unfaulted, the window would be a `mismatch` though no tick was late. Its status:
  - `delivery-refused`, as for a link left unwritten or a refused stream. The proxy can't tell its own pause from a
    slow server connection, and `monitor-unavailable` is the ticker's rule.
  - The deadline is the tick's own, `slack` after the release tick, since a forward past the middle of the bucket can
    already reorder a server that writes there; and never past the anchor's egress bucket.
  - The proxy checks it at every egress tick before handing. The first frame a late forward can change is ready at the
    anchor's next bucket at the earliest, so it is never handed into an unfaulted window.
  - A request released in an ingress window's last bucket has its window published before the deadline, so its
    lateness shows on egress only.
- **A dropped queue.** A fault drops the queue, but the Program's grid carries queued frames into the next window (FIFO
  across windows). So an egress window that closes faulted while its queue would still hold frames faults the next
  one too, with the same reason, until a faulted window closes with none.
- **The slack** bounds how late a tick may run: 25 ms by default, a quarter bucket. A deployment sets it from the jitter
  its timing bounds assume (J).
- **The pacer** (`Proxy(paced=…)`, `pacer.Pacer`). One loop runs the ticks and reads the link and every server
  connection, so when the server writes within a bucket moves when the tick runs, and a bucket written at its tick leaves
  then: the prover's offset within a bucket reaches the client, against J. With `paced`, a time past the slack, the
  egress tick hands its bucket to a process of its own, which writes it to the link `paced` after the bucket's boundary
  on the host's monotonic clock, a time fixed before the bucket is made. A thread would share the loop's interpreter
  lock, which the loop holds while it reads. So a tick late by up to the slack never moves when a bucket leaves, and a
  bucket that can't leave at its time fails its window closed (above). `benchmarks/network_traces/certifier_serve.py`
  paces at 50 ms (`PACED_MS`).

The clock and the sleep are injected (`VirtualClock` serves a run driven event by event). `Shaper` and `Bucketer` do no
I/O, so the tests drive real sockets without reading the wall clock. `benchmarks/network_traces/active_replay.py`
replays recorded traces through the proxy in real time, and on an injected clock.

## Sealed pair links (`active.Pair`, `verity.ml.crypto.aead`)

The leaf and spine switches between blocks are the prover's, so they can copy every frame that crosses from one block's
certifier to another's. On a sealed pair link (`Proxy(pair=active.Pair.of(…))`, off by default, and with it off every
record, wire and verdict is as above) the two ends are both the auditor's certifiers, and every frame between them is
sealed under an AEAD built from SHA-512 alone.

- **Where it sits.** The sending certifier seals each egress slot after its egress certifier records it, and the
  receiving certifier opens each unit before its ingress certifier takes it. Every slot is sealed, padding too, so the
  link carries r units of B + 64 bytes a bucket whatever the occupancy. The records, the occupancy, the commitments and
  the certificates are the unsealed run's, so N2 and the timing bound read the same grids
  (`test_a_sealed_pair_link_carries_only_ciphertext_and_every_record_is_the_unsealed_runs`).
- **The construction** (`verity.ml.crypto.aead`, the standard library's SHA-512 with its whole output; no new hash).
  H is SHA-512, HMAC is HMAC-SHA512 (RFC 2104), and [x] is u16be len(x) ‖ x.
  - Keys, by HKDF-Expand (RFC 5869) with one output block. S is 64 uniform bytes, so there is no extract step:
    `K_enc = HMAC(S, "verity-network-certifier/pair/v1/enc" ‖ [σ] ‖ [ℓ] ‖ 0x01)`, and `K_mac` is the same under
    `…/mac`. σ is the session id and ℓ the direction's name (`"a>b"`), so each direction of each pair in each session
    has keys of its own, and a unit reflected back to its sender doesn't open.
  - The nonce of slot j of bucket t of window w of the sender's egress grid is `N = u64be w ‖ u32be t ‖ u32be j`.
  - A frame f is sealed as follows: `KS = H(K_enc ‖ N ‖ u64be 0) ‖ H(K_enc ‖ N ‖ u64be 1) ‖ …`, cut to B bytes;
    `C = f ⊕ KS`; `tag = HMAC(K_mac, N ‖ C)`. The unit is `C ‖ tag`.
  - A unit is opened by recomputing its tag, comparing it in constant time, and giving `C ⊕ KS`, or nothing.
  - Why each piece is standard:
    - Each keystream input is 88 bytes, inside one compression block, with the key in it. That is SHA-512's compression
      function as a dual PRF keyed by the secret bytes of its first block: the claim `prf/sha-512`
      (`verity.core.claims`), on which `verity.ml.randomness`'s keyed streams already rest. Every input has one
      length, so length extension doesn't arise.
    - Counter mode over a PRF is IND-CPA while no nonce repeats under a key (Bellare, Desai, Jokipii and Rogaway, 1997).
    - HMAC-SHA512 is a PRF under the same assumption (Bellare, 2006; FIPS 198-1), so it is a strongly unforgeable MAC.
    - Encrypt-then-MAC with the nonce under the tag is IND-CCA and INT-CTXT (Bellare and Namprempre, 2000; scheme N2 of
      Namprempre, Rogaway and Shrimpton, 2014).
    - Every unit under one key has one length, so `N ‖ C` needs no length field. No new claim id is needed.
- **Keys.** There is one secret S per certifier pair per session: 64 bytes of one certifier host's own randomness
  (`fresh_pair_secret`), drawn again for each new session σ.
  - SHA-512 alone gives no key agreement, so S reaches the pair's other certifier host the way the certifier's keys and
    run id reach their consumers: through the auditor's deployment channel.
  - S and its keys live only in the two certifier processes and on their hosts, never in a record, a log or the
    prover's configuration.
  - They are protocol keys, which seal frames, not access keys.
  - Their custody joins `trusted-host`, which now also covers each pair's secret on its two hosts and the
    deployment channel that installs it.
  - A leaked S opens that pair's link for that session. It changes no record and no verdict: the account only loses
    the premise below, and the link is charged as an unsealed one.
- **Randomness and leakage.** The only coins are S, and nonces are public structure. Each nonce is used once under its
  key:
  - the sender refuses a slot at or before the last it sealed (`Sealer`);
  - a certifier back from maintenance keeps its sealer;
  - each session and direction has keys of its own.

  The leakage argument: a nonce is a function of the link, the window, the bucket and the slot, which the path sees
  already, so it adds nothing. With each nonce used once under keys the path never holds, every unit is
  indistinguishable from B + 64 uniform bytes (`prf/sha-512`). Since every slot is sealed, padding too, at r units a
  bucket, what the path sees is a function of public parameters and time alone: neither the frames nor which slots
  are padding.
- **Fail closed.** The receiver opens the k-th unit it reads as slot k of its peer's grid, from the peer's start
  window on (`Opener`). A unit that doesn't open (tampered, replayed, reordered, missing, which shifts every unit after
  it, or sealed under another session's or direction's keys) faults its ingress window `malformed`. The opener stays
  refused, so every later window is `malformed` too, until a new session. Nothing is retried or redrawn.
- **The account.** Once every link between two blocks' certifiers is sealed, a crossing from one block to another is
  ciphertext off the two certifiers. So the export charge (λ_out) applies at the facility boundary only. In
  `benchmarks/datacenter` this switches at `Datacenter(sealed=True)` (off by default), which `Placement._exported`
  reads to count only the crossings to a reader off every block.
- **The analysis.** `Security/` doesn't change. The egress model's A1 (EgressMediated(s))
  restated with encryption as its premise is a claim, not a proof:
  - **The restatement (A1 at the facility boundary, sealed):** every bit that leaves the facility is an egress slot
    value certified equal to the committed output at a certifier on the facility boundary, or one of s side-channel
    bits per window.
  - **The premise:** every link between two spatial units is a sealed pair link between two of the auditor's
    certifiers. The AEAD's security rests on `prf/sha-512`, and its keys' custody on `trusted-host`.
  - **What it changes:** A1's units are then the facility, not each block.
## Tests

`uv run tools/verity/check/suites.py verity/core/protocols/certifier`. The tests enumerate small monitoring periods, so they
check the Lean statements on examples and prove nothing. A test that mirrors a statement of the analysis carries its
name, and each test module's docstring says what it holds.

**The difftest** checks the definitions the analysis's statements are about, not the statements. On 10,000 random instances,
the certifier program (`lean/`, `NetworkCertifier`), evaluated, gives what this package gives. The Lean side is
`expectedRow`, with `Accepted` of `constantRate` and `StatusComplete` decided per prefix of whole windows. The Python
side is `expected_grids`, `check_link`'s verdicts with the first differing bucket, `check_ingress_link`, and each
prefix's acceptance. `lean/scripts/difftest_vectors.py` writes the instances and this package's results, and the Lean
audit runs `lean/NetworkCertifier/DifftestMain.lean` on them (`runs` in `lean/lean-audit.json`).
`test_network_certifier_lean_difftest.py` checks that the data the record pins is still this package's. The program's
definitions are List forms of `NetTiming`'s `Finset` ones, and four guarantees prove them equal on everything the
difftest compares (`NetTiming.SecurityProofs.Code`): `CodeExpectedRow`, `CodeAcceptedConstantRate`,
`CodeStatusComplete` and `CodeSyncSet`.

## The active warden (`warden`)

The wire is client ↔ network certifier ↔ server, with the network certifier responsible only for limiting network steganography and
certifying exactly the bits it sees. So the network certifier knows bytes only: nothing in it belongs to vLLM
or to any workload. `warden.py`'s docstring has its cells, admission, schedule and failures. This section says what
its wire can carry.

**The schedule's public parameters** are `WardenParams`, fixed before the run:
- B (`cell_bytes`), r (`slots`), τ (`bucket_ms`), D (`delay`), the lanes, the queue and the window T;
- the bounds `ingress_cells`, `ingress_buckets` and `egress_cells`.

`WardenParams.digest` binds them, and every record head's `parameters` is that digest. The ticker's slack is the active
certifier's.

**What the wire is a function of.** On connection q the network certifier sends r cells of B bytes each way in every bucket, from
q's admission s to its last bucket. The kinds and buckets of q's egress cells are a function of the parameters and
three numbers, and its ingress cells of s and the client's bytes:
- s: the bucket the network certifier admits q in, on its own schedule: of the arrivals, which are the client's, and of the
  admission rule. The rule is FIFO over the lanes, and a lane frees when an earlier connection ends, so s also depends
  on the earlier connections' n and f below.
- n: q's egress cells, max(1, ⌈L/P⌉) for a response of L bytes.
- f: the egress cell at which q failed, if it did.

**Egress is based on admission, never on response completion.** Egress cell i is in bucket s + D + ⌊i/r⌋. Nothing
else moves that base: not the bucket the client's ingress ends in (a), not when the server accepts its connection,
and not when its response starts or completes. Ingress is released on the client's bytes whether or not the server's
connection has opened, and what it releases before then is buffered for the server, at most the ingress bound's
`ingress_cells` cells (past them the connection fails closed). A server that hasn't sent cell i's bytes by its bucket
fails q there, which is f.

Every `DATA` and `END` cell's content is the bytes, cut at fixed offsets (byte k is in cell ⌊k/P⌋), never by when or
how they were read.

**The leak bound** is what a server can signal on the wire beyond the response it sends:
- **0 size bits.** Every cell is B bytes, whatever its kind (`PAD`, `DATA`, `END`, `FAIL` or `REFUSED`).
- **0 timing bits, at bucket resolution.**
  - Given n, every cell's bucket is fixed. When the server writes and how it chunks its bytes reach the wire only by
    failing the connection.
  - Within a bucket the jitter is the ticker's, bounded by the slack as for the active certifier (its J). A tick past
    the slack faults every record's window.
  - With `paced` (`Warden(paced=…)`), the tick hands each connection's cells to the egress pacer, as the active
    certifier's pacer (above), which writes them to the connection's socket `paced` after the bucket's boundary, a
    time fixed before the bucket is made. A bucket that reaches the pacer after that time, or whose release starts
    past the slack, faults every record's window: the open one, or the records held until each of its buckets'
    releases is reported. `benchmarks/network_traces/warden_serve.py` paces at 50 ms by default (`--paced-ms`).
- **The advice is not on the wire.** The network certifier sends no advice. Whatever the proofs need beyond the committed values
  travels on the proof side and is charged there (`A_c`).
- **log2 of the schedule's public parameters**, once per deployment: log2 of the number of parameter sets it may
  choose from, which is 0 when one set is registered.
- **log2(cells + 1) per failed connection.**
  - A server that fails connection q picks the egress cell it fails at, or not to fail: at most n + 1 ≤
    `egress_cells` + 1 outcomes.
  - Every failure sends the same `FAIL` cell, with no payload: a missed deadline, a stream that hasn't ended by the
    last cell, a broken server connection, and the client's own two. The reason is a frame of the network certifier's `fails`
    record, in the bucket of the `FAIL`: witness, certified with the others and never on the wire.
  - A failure the client causes (`ingress`, `malformed`) fails at once, and only returns the client's own bits to it.
  - A failed connection frees its lane at its `FAIL`'s bucket, so f also moves later connections' s. That carries no
    bits beyond f's, which this term charges, though they reach another client.

n is content, not transport: it is the length of the protocol's output, which the proofs bind. It is charged with the
content, at ε times the output bits of unproved units at δ. So on a run in which no connection failed, the transport
leaks log2 of the schedule's parameters, and each failed connection adds log2(cells + 1). Lean states the bound over
the network certifier's schedule as `Security/Definitions/NetworkCertifier/Warden.lean` models it, for a decoder of the
client's view read back from the certifier's record (`SecurityProofs.PeriodLeakAccepted`; "Capacity"), not over
`warden.py`'s code.

**Deployment requirement (`trusted-host`).** The bound holds only on a host that gives the network certifier's process:
- CPU isolation from every co-tenant (its own cores), so that no co-resident process, the prover's above all, can
  make a tick late and so choose which windows fault (about 1 bit a window otherwise);
- a monotonic clock that nothing the prover runs can step or slew by more than the slack: buckets are read from it.

Both are part of the named assumption `trusted-host` (the auditor's host and its provider are trusted). Neither is
checked here.

**What it attests.** Each direction's records hold exactly the cells sent that way, padding included, so |T| is the
wire's bits. The third record, `fails`, holds one reason a failed connection.
- Each window's three records are certified as the certifier's are (record v2 and `certificate.Issuer`), with the network
  certifier's parameters digest in the head. A head is witness, so no reason reaches the client.
- `audit` reads the records in the clear and refuses a window whose cells aren't the schedule's, whose `FAIL` cells
  and reasons don't match one for one, or in which a connection failed.
- `benchmarks/network_traces/warden_serve.py` runs the network certifier live and checks that each client got exactly its egress
  record's cells, each in its bucket. The auditor gate's `warden-rehearse` (`benchmarks/live_challenger/auditor.py`)
  runs the same rehearsal on the auditor's host.
- The server's copy of each request carries its release order (`release`), so that `serve-openai --admit
  release-order` adds requests to the engine in the order they were released: `x-verity-release: N`, N its `END`'s
  index among the ingress records' `END` cells, and `x-verity-release-run:` the network certifier's `run_id` (in
  `deployment.json`). The server gets a request whole at its `END`'s release, and nothing of a connection that fails
  before it. The headers are on the server's side only. No cell or record holds them, so the cell counts, the prompt
  lengths and the I/O tie's request frames are unchanged. They reach the server, never the client, so the leak bound
  is unchanged. The active certifier's served proxy adds the same headers (`active_replay.ReleaseProxy`), with
  N = s − 1 for request stream s.

**Named gaps:**
- TLS isn't terminated: both sides are plain TCP.
- Every recorded run had the network certifier on the server's host, and the server gets the ingress bytes over loopback, with no
  shim on its side. `warden-rehearse` puts the network certifier and its client on the auditor's host and reaches the server
  through the gate's ssh tunnel to the peer's loopback. No run of it is recorded yet, and the tunnel's far end is still
  loopback on the prover's host.
- **The cells' wiring is open to the verifier.** A unit's read is (q, k), connection q's
  cell k of the sampling block. The verifier places it at its bucket's row and slot with the network certifier's wiring
  (`Wiring.place`), which the signed head names by a salted digest, so the reads are the network certifier's and never the
  prover's (the phase-8 attack's finding, closed). Which cell each unit reads is the Program's: the units' wiring
  (`Units`, a (q, k) a unit) is fixed with the Program from the schedule, the ordinals and the signed wiring alone,
  never the cells' bytes, and the port's value names its digest, which Lean's registered reads hold the statement's
  header to, so a statement made for another units' wiring is refused even at the same rows (the phase-11 attack's pad
  for a data cell, closed; no gate). But the wiring is opened to the verifier, and it and the rows the units
  read give away each connection's first row and cell count, both the prover's. A permutation argument closes it, with
  no new verifier form. Before the window's coins, the prover registers V′, a value of its own: the sampling block's
  N = T·S cells as pairs (k, cell k of V, the network certifier's value), reordered into unit order, a fixed number of positions
  per consuming unit (its cells in wire order, then empty cells), so the population's shape is public and its
  occupancy isn't. Under two live coins α and γ, a population reads every row of V (a bucket's S cells) and of V′
  through two registered ports, both in public order. It carries ∏(γ − f(k, V_k)) and ∏(γ − f(V′_m)) over the cells
  from unit to unit as committed values, and the last unit's one output bit says whether they are equal. Here
  f(k, x) = k + α·x(α), with x(α) a cell's B/8 words of 64 bits as a polynomial over GF(2^64): at a public α it is XOR
  gates alone, and each cell costs one GF(2^64) product. Each consuming unit then reads its own positions of V′, which
  are public, and checks in-circuit that its cells' k are its connection's schedule (bucket s + D + ⌊i/r⌋ for egress
  cell i), that its cells' stream field is its ordinal and that its spare positions hold B zero bytes, so the wiring
  stays witness. A wrong V′ passes with probability at most (N·B/8 + N)/2^64 (Schwartz–Zippel). An unproved unit
  could forge its running product, so every unit of the permutation population is proved, not a sample. The steps,
  each a PR: (1) V′'s commitment and registration in `warden_rows`; (2) the permutation unit's Definition, its
  circuit-check sidecar and its cost at the served schedule; (3) the demonstration on `warden_zk.py`, with a wrong
  wiring refused; (4) the consuming units read V′, the wiring is no longer opened, and this gap closes. The I/O tie,
  which reads every row of the window in order, needs no wiring.
- A record's certificate signs certificate v2's SHA-512 frame leaves, one per cell. The same cells are also committed
  as v3's hm96 rows, a row a bucket (8·B transfer gates a cell), under a root the network certifier signs ("The warden's cells as
  rows").

## The warden's cells as rows (`warden_rows`)

Design v3 makes the transfer gates the cells' bits: the sender's proof block opens a cell's row as an output and the
receiver's opens the same row as an input, under a root the network certifier signs, through the Lean verifier's registered
reads (`Flock.Registered.checkPort`, `--registered`), with no new verifier form. `warden_rows.py`'s docstring has the
encodings; this section says what is committed, signed and read.

**The rows.** Each sampling block of the wire's two records (link, direction, window w) is T rows, a row a bucket of its
grid whatever the load (the I/O tie's layout, "The I/O tie"):
- row t − w·T is bucket t's S = lanes·r slots in slot order, slot j at bytes j·B to (j + 1)·B, so at words j·B/2
  onward; at the served schedule a row is 4 KiB;
- a bucket's cells are sorted by stream, the connection's ordinal, each connection's r cells together and in order,
  its padding cells among them; a slot no connection holds is B zero bytes, so T and the row's size are the
  schedule's;
- a row is `hm96-sha512/row/v2` of its 8·S·B bits, bit 8k + i being bit i of byte k: S·B/2 little-endian u16 words
  at 16 bits, which is how a C-Flock port of 8·S·B bits lays out the row.

**The wiring** says which connection holds which slots: connection q, live in `cells`/r rows from row `first`, holds
slots ρ·r … ρ·r + r − 1 of each, ρ its rank by ordinal among the row's live connections, so its cell k of the
sampling block is in row first + ⌊k/r⌋ at slot ρ·r + (k mod r) (`Wiring.place`). The network certifier reads it off the record
(`wiring`, which refuses a record not laid out so) and commits it, as (q, first, cells) by q, under a 64-byte salt of
its own.

**The commitment** is a registered value's: row k's `b‖c` is hm96's commit string of its `sha512/row/v2` digest under
salt k, 192 bytes the network certifier draws from the operating system for that row alone, and the leaves form a
frame-v3-sha512 tree. Its domain is the network certifier's own, binding the parameters digest, the certifier session, the link,
direction and window, B, S, T and the row format, so a root opens under one sampling block of one session's schedule.
The rows, their salts and the wiring's are witness, handed to the prover alone. What the network certifier keeps for the prover
is the records and, for each sampling block, the rows' salts and the wiring's (`cells.salts`, `Cells.witness`): the
rows and the wiring are read off the records again. The auditor's gate keeps all of it under the run's `witness/`.

**The signed head** names the root and the wiring: link, direction, window, the parameters digest, B, S, T, the
wiring's digest, the domain and the root, under its own tag (`CellsHead`). The network certifier signs it with the certifier's
signer (`Issuer.sign`: the certifier message for the key, the session and the sampling block, one LMS leaf). The record's
own certificate of that sampling block signs its salted record digest, so neither verifies as the other.

**The registration** (`registration`, `verity-network-warden/registration/v2`) is the head, the rows' format and order
(`layout`: T, S, r and the first bucket w·T) and the certificate, as one public JSON object. `check_registration` is
the verifier's read of it: every field but the wiring's digest and the root is the one the schedule and the
sampling block give, and the certificate verifies over the head under the network certifier's key in the session.

**Ordinals count from 1.** A cell's stream field is its connection's ordinal, and so is the wiring's q: the k-th
connection to reach the network certifier, from 0, is ordinal k + 1, the wire's ingress ordinal, as the active certifier's
`Requests` streams the k-th request and the I/O tie's map puts advice request i on stream order[i] + 1. Ordinal 0 is no
connection's: a `REFUSED` cell carries it.

**How a unit reads it.** The verifier's own `--registered` port (`port`, `verity/registered-reads/v1`) names the value
`<link>/<direction>/<window>/units/<digest>`, the digest of the Program's units' wiring (`Units`), the root, T leaves of
S·B/2 words of 16 bits, the domain, the row format and `reads`, the row each unit of the population opens. A unit's
read is its (q, k) in the units' wiring, and the verifier places it with the wiring: the
row is the cell's bucket, and the cell is at its slot's words. The verifier takes a wiring only when it is the one the
head names and the schedule's (ordinals and first rows in admission order, r cells a row, at most `lanes` connections
a row), and refuses a read of a connection the wiring doesn't hold, of a cell past the connection's, or of a cell
another read opens. So no read opens an empty slot, a foreign connection's cell or a cell twice. The statement is a
shared-row file whose port's table rows are those rows under the network certifier's salts, its header's `registered` entry their
positions and paths (`Cells.statement`). An output port reads them as the sender's unit's output (its table is the
instances' output rows), an input port as the receiver's input. A port reads one row a unit, so a unit that reads
cells of k buckets has k registered ports, and reading one cell costs its whole row. The statement's header names
the port's value, so it is made for the Program's units' wiring and no other. The cells' wiring is opened to the
verifier: the network certifier's named gap "The cells' wiring is open to the verifier", which has the plan that closes it.

**How the other readers of the certifier's rows differ:**
- N2's head checks: the prover commits each record head as one row (`LinkWindows.commit`: one-stage's served domain,
  fresh salts). The unit opens it in-circuit (`FrameV3Read`) and recomputes the certificate's salted digest in-circuit.
- The I/O tie's prototype (`io_tie.py`): these rows (`window_rows`), the network certifier's own cells, with a `PAD` on the
  connection's stream in its own slots. Its unit is one a bucket (`IoTieBucket`): unit t reads row t whole, every slot
  held or not, as a registered port with no wiring (`bucket_port`), so no gate of it hashes the row. Its port names
  the head's value and no `Units`: the rows alone fix its reads, and the registered reads hold each row to its unit.
- Here the network certifier commits one row per bucket under its own domain and signs the head, so the certificate names the
  root in the clear. The opening is the verifier's registered-read check, and no gate of the unit hashes.

**Demonstration** (`benchmarks/network_traces/warden_zk.py`). A toy unit, `WardenRowIdentity{B}` at B a row's S·B
bytes, takes one row's bits through two NOTs. Each unit reads one of a connection's cells, so its row is that cell's
bucket. Its port is registered against the network certifier's egress root of a `warden_serve` run, proved by C-Flock `prove --zk`
behind the firewall, and held by Lean `verify --zk --registered`. The members:
- the sender's output and the receiver's input are accepted;
- rows under the prover's own root are refused, and so is the prover's registration of it under the network certifier's
  certificate;
- a row one bit off the network certifier's, in the unit's cell, is refused.

No salt and no row content reaches the verifier.
