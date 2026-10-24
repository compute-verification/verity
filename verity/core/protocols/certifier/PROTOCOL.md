# Network accounting: the bucketed timing channel

A spatial unit is the machines behind network certifiers. The question is how much a covert sender can move across its
boundary through **when** frames leave, once the content and order of every frame are fixed by the declared Program.
This package is the reference for the answer, in the terms of its Lean analysis `NetTiming` (`Security/`: the
areas `Definitions/NetworkCertifier`, `Models/NetworkCertifier` and `Proofs/NetworkCertifier`, whose `README.md` has the modelling choices). The
tests mirror some of its statements by name; the Tests section lists which. The certifier program (`lean/`, the Lake
package `certifier_program`, Lean core only) is the code the difftest runs, proved equal to the analysis's definitions.

## Zero knowledge: what ties the checks, and the named gap

N2's count check and N3 read the same head leaf of N2's commitment. N3 holds the window's N frames to the head's root,
and so to the head's Σ n_t, and N2 holds those counts to the occupancy profile. The two are tied only under three
conditions, each of which holds:
1. The root is registered before either check's coins or draws (the firewall's registration, `LinkWindows.commit`).
2. The leaf index, 2 + d, is fixed by the link-window's direction (`service.HEADS`), which is public structure, not
   something the prover chooses.
3. SHA-512 is collision resistant, as hm96 rows in frame-v3-sha512 trees already assume.

N2 no longer compares the grid in the clear: `LinkWindows.verify` runs `check_link(grid=False)` (NET-38), and the
network claim is counts only until N3 is a proof-service unit on every window (ASK-70, counts-first). One gap keeps the
count check from being zero-knowledge, and nothing may call it that until it is closed (the README states it too):
- **N2's unit isn't proved yet:** its verifier of record runs `head_check.reference` on the host over the opening, the
  heads and the certifier's salts. When it is proved, the head-check gate binds `NetworkHeadCheck_v2`'s circuit digest
  and a proof of any other circuit is refused (NET-43).

The certificate's signature no longer reads a hash of witness (NET-42). The certifier signs
`D = SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R)`, R the head's record digest:
- `salt_c` is drawn by the certifier (`certificate.Issuer`'s `coins`), 64 bytes, fresh per record, and handed to the
  prover with the head as witness only (`CertifiedRecord.salt`, in the record leaf after the head; the head leaf N3
  reads is the head alone);
- each end's head check holds `R` to the digest of the head it opens and `D` to `R` under `salt_c` (part 4 below);
- the LMS check (`Certificate.verify`) reads only the public fields (key id, session, link, direction, window) and
  `D`, a hiding commitment to `R`. Until N2's unit is proved, the verifier of record checks `D` against the head and
  `salt_c` on the host too (`CertifiedRecord.verify`).

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
    padding.
  - The **occupancy** is (n_0, …, n_{T−1}), the number of real frames per bucket.
  - A colluding receiver can tell padding from a real frame, so occupancy is the timing channel.
- **FIFO, with the queue carried across windows.** The release count is Lean's `released`, and the grid Lean's `fill`.
- **Fail-closed (W6).**
  - A fault is an overflow beyond Q, a frame of the wrong size, or a reported one (monitor unavailable, drop, delivery
    refused, coverage never established).
  - After a fault, the rest of the window is padding, the queue is dropped, and the first reason is recorded.
- **The record is exactly (grid, status) (W5).** The status is one untimed symbol per link-window: complete, or not
  established with a reason. It carries no bucket index, time or count.

## The published record (`commitment`)

A raw grid is 2.3 GiB per link-window at r = 1,024, B = 4 KiB and T = 600. The certifier publishes one digest and the
status instead, which is 32 bytes per link-window, or about 46 KB per link per day.

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
committed record reaches the auditor as published. Instead, once per link-window, the certifier signs a **network
certificate**: a signature over a hiding digest of the window's record, salted by the certifier (`certificate.Issuer`).
The record head and the salt are witness. They enter proofs, which open frames of the record in-circuit, and never
reach the auditor in the clear, so the record's own hashes are unsalted. The audit checks the certificate against the
certifier's public key. With an honest certifier every certificate verifies, and the audit's verdicts are those of the
raw records.

**The record (v2)** is still exactly the grid and one status (W5), with padding elided, in a fixed binary layout
(`RecordHead.encode`), all integers big-endian:

| field | bytes | |
|---|---|---|
| `len(link)`, `link` | 2, len | UTF-8 |
| direction | 1 | 0 egress, 1 ingress |
| window | 8 | |
| T, r, B, τ_b | 4 each | τ_b in ms |
| status | 1 | 0 complete; 1 monitor-unavailable, 2 overflow, 3 malformed, 4 dropped, 5 delivery-refused, 6 coverage-never-established (`STATUS_CODES`) |
| deadline | 1 | 1 when the certifier settled the record by the verdict deadline |
| parameters | 64 | `parameters_digest`: `tagged_sha512("verity-network-certifier/parameters/v1", ·)` of the canonical JSON of every `GridParams` field, Q included, and of Σ_sync (δs, and ρs as fraction strings; null on ingress) |
| n_0 … n_{T−1} | 4 each | the occupancy |
| root | 64 | the frame tree's root |

The record digest is `tagged_sha512("verity-network-certifier/record/v2", R)` (`verity.ml.commitments.identity`).
Every field has a fixed width or a length before it, so R parses one way, and the counts fix where each frame sits in
the grid (W4). The schedule check reads the counts exhaustively; the frames are in the tree.

**The frame tree** is a binary SHA-512 Merkle tree over the window's N = Σ n_t real frames, in the order `grid.fill`
places them, each of B bytes:

    leaf(f)    = SHA-512( block("verity-network-certifier/frame/v2\0" ‖ u32be B) ‖ f )
    E_0        = SHA-512( block("verity-network-certifier/empty/v2\0") )
    node(a, b) = SHA-512( block("verity-network-certifier/node/v2\0") ‖ a ‖ b ),   E_{l+1} = node(E_l, E_l)

`block(x)` is x zero-padded to one 128-byte SHA-512 block, so each of the three hashes starts from a constant
midstate, and the tags keep leaves, nodes and empty subtrees apart. The depth is fixed, D = ⌈log2(T·r)⌉: leaf k is
frame k for k < N and E_0 after, so padding slots never enter the tree, and opening frame k (`open_frame`,
`verify_frame`) costs D siblings at every occupancy, which is the shape a sampled in-circuit opening needs.

frame-v3's SHA-512 tree (`frame-v3-sha512`) doesn't fit, for three reasons:
- its `CommitmentDomain` takes a 32-byte SHA-256 `binding`, and its positions compute a SHA-256 identity eagerly, so the
  record would rest on `cr/sha-256` as well;
- its `owner` is a weights, boundary or replay unit, not a link-window;
- it pads its leaves to the next power of two of N, so its depth and opening shape change with the occupancy.

**The signed digest** hides the record digest R under `salt_c`, 64 bytes the certifier draws from its own
randomness, fresh for each record (`Issuer`'s `coins`), never the prover's:

    D = SHA-512( "verity/net-cert/v1" ‖ salt_c ‖ R )     (`certificate_digest`, Lean `NetworkCheck.certificateDigest`)

The certifier hands `salt_c` to the prover with the head, as witness only (`CertifiedRecord.salt`). R is 64 bytes, so
D fixes R up to a SHA-512 collision (`cr/sha-512`). D hides R as SHA-512 under a secret, uniform 64-byte prefix does,
which rests on `random-oracle`; no Lean guarantee states the hiding.

**The certificate** is D and a signature over a 64-byte message that binds D to the key, the certifier session and the
link-window, so a certificate can't be replayed onto another window, link, direction, key or session:

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
certifier then publishes nothing for the link-window (`Issuer.refused` lists it), which fails closed as a `no-verdict`; it
never signs with a reused leaf, nor in a session it has run before. Both rest on `trusted-nebius-host`, where the state
the device keeps on its host is its next LMS leaf and the session ids it has started.

**The check.** The certificate names the head's link-window, D is the head's record digest under `salt_c`, and the
signature verifies (`CertifiedRecord.verify`; the signature alone is `Certificate.verify`, which reads the public fields
and D). On egress the audit recomputes the record digest from the Program's grid, with a complete status, the deadline
bit set and the parameters digest of `params` and Σ_sync, and compares it with the head's. Under `cr/sha-512`, equal
digests mean equal records, so the audit stays exact. A certified record rests on `cr/sha-512`,
on the signature's unforgeability, and on an honest certifier; not on `cr/sha-256`.

## Through the proof service (`service`)

One link-window's certificates are registered through the proof service (`verity.core.protocols.sampled_proofs.service`)
as a `Spec` of its own (`service.LinkWindows`, one per link):
- **Committed:** one value, `certificates`, of four leaves: the link-window's egress and ingress certified records,
  leaves 0 and 1, each `hm96-sha512/row/v2` over `service.encode` (the head, the certifier's salt, then the
  certificate), and their heads,
  leaves 2 and 3, each over the head alone (`RecordHead.encode`). Every leaf has its own fresh salt, in one
  `frame-v3-sha512` tree under one-stage's served domain for the run, the link and the window. A proof that reads only a
  head (N3's, which is designed to hold the window's frames to the head's root) opens its head leaf in two or three
  SHA-512 compressions, where the record leaf would cost the certificate's 272 more. The firewall keeps the records and
  salts (`Opening`). The challenger receives one salted root and never a certificate, a certifier's salt or a head.
- **Selection:** `All`. A registration is one link-window, one proof unit, so an accepted outcome is the exhaustive fact.
  The record's `window` provenance is exactly `{"kind": "served", "window": w}`, and the challenger holds w to the
  registration's order (`LinkWindows.register`): the receipt keeps the order, and the order is the link-window.
- **The network verdict of record** is `LinkWindows.network_verdict`, the AND over the link's link-windows with no ε
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
  the certifier's key id and its session id (`certifier_session`), and the check's outputs `verdict` and `deadline`. A rejected link-window shows only the
  service's `reject`; the violation's kind stays in the verifier. A registration with any other field, a second root,
  a root entry with another key, or anything more in `window` is refused. A served link has no egress profile, and its
  sizes and each window's declared sync are public instead (`PUBLIC_SERVED`, "A served link" below).
- **The check** is `LinkWindows.unit`, `NetworkLinkWindow` v3{EGRESS, INGRESS}, over the two certificates' signed
  digests: the AND of the certificates' check and each end's head check. The certificates' check is one opaque
  primitive over the two signed digests, with no gates: each certificate verifies under the certifier's key in its
  session (`LinkWindows.certifier_session`) over its public fields and its signed digest, and names the link-window,
  and its head's deadline bit is set. The head check is a Boolean Definition,
  `head_check.NetworkHeadCheck_v2{L, T, COUNTS}` (EGRESS and INGRESS are each end's statics), with one output, the ok
  bit, whose `signed` input is its end's signed digest. It opens the head at leaf 2 + direction under the registered
  root, the hiding read N3 makes of the same leaf (`NetworkHidingRead`), and accepts when the head's link, direction
  and window are the link-window's, its status is complete, its T, r, B, τ_b and parameters digest are the end's
  declared grid's (`NetworkHeadFields`), on an end with an occupancy profile its T counts are the profile's row for the
  window (`NetworkHeadCounts`; a window past the profile has no row and is a reject), and the head's record digest R
  (`NetworkRecordDigest`) under the witness `salt_c` is the signed digest (`NetworkSignedDigest`), so the certificate
  signs the head the check opened. In the unit's Program each head check is one gate that names the Definition
  (`head_check.gate`): the commitment gadgets' Boolean bodies encode to about 140 MB, and the service encodes the Program
  on every registration. The Program's digest is the check's, so the spec, the stream and each registration bind to it.
- **Its sizes** (`head_check.size`, L = 9): the count check is an equality of 32·T bits, 32·T − 1 ANDs (19,199 at
  T = 600); the fields are 736 + 8·L − 1 (807); the opening is 775,077 at T = 4 and 1,876,070 at T = 600, nearly all
  of it the head row's hm96 digest; the signed digest is 290,246 at T = 4 and 1,333,292 at T = 600, the record digest's
  ⌈(244 + L + 4T) / 128⌉ compressions (five and 23), the salted digest's two and an equality of 512 bits. With the
  three ANDs that join the four parts a head check is 1,066,260 at T = 4 and 3,229,371 at T = 600. The record digest
  hashes the head a second time, beside the opening's row digest; a record v3 whose digest is the head row's
  sha512/row/v2 digest would share it and save about 1.2M ANDs a head at T = 600. The count check needs no frame's bytes, so it is the part of the egress grid check
  a proof can make without the content, and the only grid check on ingress. It runs on every link-window (RT-NET29
  item 3: a structural check runs in full).
- **N3, the content check,** is a design and the simulator's behaviour, not part of this protocol: a prototype unit in
  `benchmarks/network_sim` ("N3 in zero knowledge"), evaluated in the clear, in no proof-service Spec and not proved.
  Its design runs it on every port-window that recorded frames, holding its N frames to the head's frames root, so to
  Σ n_t, and to the committed layout's frames. A port-window the profile leaves empty passes from public occupancy,
  with no N3 unit. That holds because the profile is fixed by the Program and the placement before the run and bound
  by the spec's context: nothing declared at run time sets it, and a head with frames where the profile says 0 fails
  N2's count check. An end without a profile has no such backing, since its occupancy is its record's own.
- **Its verifier of record** is `LinkWindows.verify`, on the host, which reads the opening. It accepts when the opening
  reproduces the registered root, each end's head check accepts (`head_check.reference`, which the tests hold to the
  circuit), and the audit accepts both records under the certifier's key in its session, comparing no grid:
  `check_link(grid=False)` on the status, the signature and the declared syncs, `check_ingress_link` on the status and
  parameters. `verify` reads no frames, so its claim is counts only: the egress head's counts are the profile's (and
  the ingress head's, where that end declares one), which puts the link's timing at the profile. The egress grid is
  the head's counts filled with its frames in order, and the head's digest binds both, so the frames are the rest of
  the grid, and nothing in this protocol holds them: until N3 is a proof-service unit with Selection `All`, it doesn't
  say the egress grid is the Program's. On a served link it also runs the served counts' check (`served_check`, "A
  served link" below). A link whose egress end has neither an occupancy profile nor a served profile has no verdict
  (`ValueError`, failing closed), since nothing would hold its egress timing. The outcome passes the opening's root as the session's, which one-stage's audit holds to the registered one. An
  unset deadline bit is a `no-verdict` there, so a reject; a proof after `proof_deadline` is the service's `late`.
- **The oracle** is `LinkWindows.stand_in`, the in-the-clear verdict: the audit with the Program's egress grid
  compared, and each head's counts the profile's (`LinkWindows.declares`). `verify` accepts every link-window it
  accepts, and differs only where the record's grid, at the profile's counts, isn't the Program's: in its frames, which
  N2 doesn't hold (N3's design would reject them), or where the declared syncs move the Program's counts off the
  profile, which N2 holds the counts to instead.

What this establishes: the outcome is accept exactly when the opening seals, both head checks accept and the audit
accepts the registered link-window's records, and the outcome record binds the root, the draw and the session. The
claim is counts only: on every link-window of a link whose egress end declares a profile, the egress head's counts are
the profile's (and the ingress head's, where that end declares one), so the link's timing is at the profile. It stays
counts only until N3 is a proof-service unit with Selection `All`. On a served link the egress head's counts are
instead the ones its ingress head's counts derive under the window's declared sync ("A served link" below). An egress
end with neither profile gets no verdict (`ValueError`, failing closed). What it doesn't establish: the verifier
of record reads the witness, so it is not zero-knowledge, and it must run where the heads and the certifier's salts
may go until the unit is proved. The head check leaves the head's deadline byte and frames root free. The certificates' check reads the
deadline, and N3's design would read the frames root, so an accepted outcome says nothing of the frames. One clock per
spatial unit across links is not checked (each registration is one link-window, alone), and the frames' consistency
with the served computation is left to N3.

## A served link (`served`)

A served link's egress counts follow its requests' arrivals and the run-time syncs, so nothing fixes them before the
run and its egress end has no occupancy profile. Declared with
`LinkWindows(served=ServedProfile(F, F_in, windows, own))`, N2 derives them inside the unit instead: on such a link it
holds the egress head's counts to the ones the ingress head's counts derive under the window's declared sync, with no
public egress profile.

- **The unit** is `NetworkServedLinkWindow` v2{EGRESS, INGRESS, SERVED}: `NetworkLinkWindow`'s certificates' check and
  two head checks (neither end with a count check unless its end declares a profile), and one gate (`served.gate`)
  naming the Boolean Definition `served.NetworkServedCounts_v1{T, FOUT, FIN, R, RIN, OWN}` over the two heads' counts,
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
- **The verifier of record** is `LinkWindows.verify`, which on a served link also runs `served_check`:
  `served.reference` over the opened heads' counts and the window's declared sync. `network_verdict` takes the served
  profile's windows.
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
    a frame (#3051). The lab pads each block's responses to F with empty events.
  - A fixed request size: every request is F_in frames, a power of two, and every ingress frame is a request's but
    the certifier's own salt frame, which heads each window's first bucket when the salt is on (OWN).
  - The Program's canonical order is request-major: each response's frames are consecutive, responses in request
    order. The shaper hands in canonical order, so a server that interleaves responses changes nothing on the wire
    while each frame is ready by its hand.
  - Both ends share T and τ_b (checked by `LinkWindows` and the proxy), and one declared sync applies to all of a
    window's anchors, as each request is admitted in its window.

What this establishes: on every accepted link-window of a served link, the egress head's counts are `derive`'s of the
ingress head's counts under the declared sync, whatever the frames hold. So the egress timing adds nothing to what
the ingress counts and the charged advice carry, and under the premises it is the Program's. The ingress counts are
the record's own, at most r_in a bucket, as on any end without a profile. What it doesn't establish: the frames
(N3's, the salt frame's place among them included: the counts see only that bucket 0 holds OWN frames more), and the
premises, which nothing checks.

## The I/O tie (`io_tie`, a design)

N2 holds a window's counts, and N3's design holds its frames to the head's frames root and a template, but neither says
the frames carry the tokens the served computation produced: N3's open item is its outbox's agreement with the values.
The I/O tie is the unit that closes it on a served link, on draws. It is a design: nothing below is implemented, in a
Spec or proved. For each drawn (request, token) it
1. opens the head as N2 does, leaf 2 + direction of the link-window's registered `certificates` value, by
   `NetworkHidingRead`;
2. opens frame k under the head's frames root (`certificate.open_frame`'s path, in the circuit);
3. reads the token in the frame at the canonical template's place;
4. opens the Commit's value of that token: request g's i-th prompt id (`c00.input/ids`) on ingress, its j-th sampled id
   (`runner.sampler/sampled_token_ids`) on egress, through the token port below;
5. checks that the two are equal.

It reads the head leaf N2 and N3 read, under the same root, so the three conditions at the top of this document tie it
to both.

**(a) Decision: the token's id in the frame, not its text.** Each egress event carries one sampled id, and the user
decodes it with the public tokenizer.
- **The event** is `data: {…,"choices":[{"index":0,"token_id":%10d,"finish_reason":null}]}\n\n`, `EVENT_HEAD`'s
  object up to the choice, then the id right-aligned in 10 decimal characters. The shim asks vLLM for ids
  (`return_token_ids: true`, which `record.py` already sends and `live_body` deletes; else
  `return_tokens_as_token_ids` with logprobs) and writes one event per id of each chunk's `token_ids`, so neither
  vLLM's chunking nor its UTF-8 holdback reaches the wire. `pad_to`'s empty events and `events`' 64-byte cut go: a
  response is its F ids, F events, and the shim refuses one with any other number.
- **Ingress** carries ids too: a request is a canonical `/v1/completions` body whose prompt is token ids, each in an
  11-byte field (`[` or `,`, then `%10d`; 11 spaces past the prompt's length, which JSON reads as whitespace), placed so
  no field straddles a frame. The server adds nothing to a token-id prompt, so `c00.input/ids` is the request's ids.
- **N3's template** trades the text field (88 base64 characters at byte 119) for the 10-character field, and its outbox
  holds a u32 a frame instead of up to 64 bytes.
- **What it proves:** at the drawn positions, the bytes the user receives are `event(id)` for the id the Commit sampled
  there, so the text the user's client shows is the public tokenizer's decode of committed ids, holdback included,
  and a function of committed values alone.
- **What it costs:** `IoTieEvent` is 33,470 ANDs at B = 4,096: an equality over the frame's other 8B − 80 bits
  (32,687), which makes the tie stand alone and which it drops once N3 is a unit on `All`, and the id's decode, 783
  (spaces then digits, no leading zero, below 2^32, so each id has one frame).

Measured on canonical frames (the real-vLLM rehearsal `r20261024-021333-0ca6`: 40 requests of F = 16, 640 events a
profile; the GLM-4.5 tokenizer, V = 151,329):

| vLLM chunks that merged tokens | honest | late | extra |
|---|---|---|---|
| served-40 | 9 | 10 | 12 |
| planned-40 | 19 | 23 | 19 |

A merged chunk is one frame of several tokens' text and one padding frame, so 2.8% (served, honest) and 5.9%
(planned, honest) of the frames aren't one token's text. The rehearsal had no empty interior event and no invalid
UTF-8 one, but 1,077 of the tokenizer's tokens (0.71%) are invalid UTF-8 alone, which vLLM holds back and merges into
the next chunk, and 124 are longer than an event's 64 bytes (the longest 512), which `events` cuts over several frames.

Text = vocab[id] in the circuit was set aside. Each frame would need a read of the vocabulary at a hidden id (151,329
entries of up to 512 bytes, depth 18, about 3.6M ANDs a token), times the tokens its span covers, a read of a committed
span table saying which ids a frame holds, the holdback's UTF-8 rule and the base64: about 9M to 15M a draw, against
about 33.5K. Worse, the grouping is vLLM's under load, not the Program's: a prover that chooses which tokens merge and
where the padding goes has up to F − 1 bits a response in the frames that no template fixes, and the frame-to-token
map stops being arithmetic, which (b) needs.

**(b) Decision: fixed lengths and a window-local map.** Egress frame k (0 ≤ k < N, N = Σ n_t of the head) is token
k mod F of the window's request ⌊k / F⌋; ingress frame k (OWN ≤ k < N_in) is frame (k − OWN) mod F_in of request
⌊(k − OWN) / F_in⌋. With F and F_in powers of two the map is wiring. It is sound against duplicate and missing frames:
N2's served check fixes N = F·R_w for the R_w requests the ingress counts derive, every window starts at a request
boundary and ends idle (carry-over none, request boundaries), and the canonical order is request-major. So the frames'
count is exact and the map is a bijection onto the window's (request, token) pairs. A duplicated, dropped or
reordered frame shifts the frames after it, and every shifted position whose id differs is a mismatch the draws count;
one whose id is the same carries the same bytes. The tie measures exactly the share of positions whose bytes aren't
the committed ones, and it checks FIXLEN and the request-major order at its draws, which today are unchecked premises.

| Map | Against duplicate or missing frames | Cost a draw |
|---|---|---|
| (stream, index) in each event, written by the server | Unsound at sampled cost: a duplicate pair is a birthday event, and showing there is none takes a sort or permutation check over all N frames, an `All` unit of N3's size | 64 header bits |
| Each request's count, or prefix sums, in the head, written by the certifier | Sound on the certifier's word, but the lengths vary, so N2's served derive must charge each as the prover's choice (log2 F_max bits a request) | 4 head bytes a request (about 4 KB and 1.8M ANDs more a head opening at R_max = 987, in N2's too), and a hidden search of about 100K |
| FIXLEN with the window-local map (chosen) | Sound: N2 fixes N, and the map is a bijection | about 0 |

No record field is added. A run-wide map would need the window's first canonical position in the head (a record v3
field) and a run-wide port, which waits for the run's end and is deeper. If lengths ever vary, the second row is the
sound one.

**The Commit's value: a token port.**
- **Variant P, recommended.** Before the window's coins, the prover commits through the proof service one value per
  link-window, `tokens`: request g's F sampled ids, then its P prompt ids, 32 u32be ids a row (128 bytes); request g's
  slot s is row g·C + ⌊s / 32⌋, column s mod 32, C ≥ (F + P) / 32 a power of two. Each row is an
  `hm96-sha512/row/v2` leaf under its own fresh salt, in one `frame-v3-sha512` tree under the served-domain rule
  (`registration.served_domain`). It has R_max·C rows, R_max the window's public request capacity, so its leaf count
  says nothing of R_w; rows past R_w·C are zero rows, salted like the rest and never read. The tie reads the row at a
  hidden position (`NetworkHidingRead{1024, DT}`, DT = log2(R_max·C)) and its column by a 32-way mux (`IoTieColumn`,
  992 ANDs). With F ≥ 32 and R_max = 2^(D − log2 F), the most requests the window's tree can hold, egress frame k's
  row and column are k's own bits, so the position is wiring; the rehearsal's F = 16 needs a small adder. The port
  must equal the Commit:
  its prompt slots `c00.input/ids`, its output slots `runner.sampler/sampled_token_ids` (int32 [num_reqs, 1] a step)
  of the steps that served request g. That tie is compute's, in zero knowledge what `check/commit_tokens.py` does in
  the clear by identity (step, op path, output member, invocation), and its share adds to this unit's.
- **Variant H, the step rows directly.** The tie reads a committed token index at (g, s) for its step and offset, then
  the `rows-sha512` step row's 4,096-byte chunk that holds the id: about 10.4M ANDs more a draw than P, with no second
  unit.

**The draw and the unit.**
- **Selection `All`** over a link's windows, one unit per link-window and side, `NetworkIoTie_v1{SIDE, L, T, B, D, K,
  F, FIN, OWN, P, DT, C}`, with K draw slots. Each slot is one 64-bit live coin u_i, drawn by the challenger after the
  link-window's N2 registration and the port's. The circuit takes frame k_i = ⌊u_i·N / 2^64⌋ with N from the head, a
  bias of at most N / 2^64 a draw. A window with N = 0, or OWN frames only on ingress, accepts.
- **Public:** N2's `certificates` root and domain, the port's root and domain, the K coins and the statics.
- **Witness:** the head (`RecordHead.encode`), its leaf's salt and 2 siblings; for each draw, the frame and its D
  siblings, the port row, its salt and DT siblings.
- **Output:** one bit, ok.

**ε at δ.** A window's draws are independent and uniform over its N positions, and N2's counts with FIXLEN make N =
F·R_w, every token of every request of the window, so none is outside the population. If a share ε_w of window w's
positions is wrong, all of the link's draws pass with probability Π_w (1 − ε_w)^K ≤ exp(−K·Σ_w ε_w). An accept at δ
therefore means Σ_w ε_w < ln(1/δ) / K, and the wrong tokens number fewer than N_max·ln(1/δ) / K, N_max a public bound
on a window's tokens (T·r at most). N_w is witness on a served link, so this sum form is what the unit surfaces. On a
link of near-equal windows (the served shape's busy windows hold 34,499 to 35,386 frames), the wrong share of all
tokens is ε when the K·W draws of W windows meet (1 − ε)^(K·W) ≤ δ:

| ε | K·W at δ = 2^−40 | K·W at δ = 2^−20 |
|---|---|---|
| 10% | 264 | 132 |
| 1% | 2,759 | 1,380 |
| 0.1% | 27,713 | 13,857 |

At the served shape's 1,440 windows a day, K = 20 is 28,800 draws: ε = 0.1% at δ = 2^−40 per link-day. With N3 on
`All` (every frame a canonical event, only the id free), the frames carry at most ε·N·log2 V bits (17.2 a frame)
beyond the committed ids, at δ.

**Cost a draw** in ANDs, by part. The rehearsal is L = 6, T = 60, r = 98 (D = 13), with a port of R_max = 119 requests
of P ≤ 48 and F = 16 (DT = 8). Served is `SERVED`'s shape, L = 9, T = 600, r = 1,686 (D = 20), with a port of R_max
= 987 requests of P = 2,048 and F = 1,024 (DT = 17). B = 4,096 at both ends.

| Part | Rehearsal | Served | From |
|---|---|---|---|
| Head opening, once a unit | 890,971 | 1,876,070 | `NetworkHidingRead` at D = 2, `head_check.size` |
| Position | ~7K | ~25K | Σ n_t (32·T bits), u·N, the map (estimate) |
| Frame opening | 3,426,040 | 4,240,882 | (33 + 2D) compressions of 57,947, 512·D for the path's order, 511 for the root's equality |
| Frame parse | 33,470 | 33,470 | `IoTieEvent`: 8B − 80 bits of template, the id's decode |
| Commit opening, P | 1,774,919 | 3,283,992 | `NetworkHidingRead{1024, DT}` (1,773,927 at 8, 3,283,000 at 17: the row's digest, `Hm96Sha512Leaf`, `FrameV3Read`), and `IoTieColumn` |
| Join | 2 | 2 | |
| The first draw | ≈ 6.13M | ≈ 9.46M | |
| Each further draw | ≈ 5.23M | 7,558,346 | the head shared; served is `IoTieDraw{4096, 20, 1024, 128}`, measured |

Per link-day at the served shape and K = 20, that is 1,440 × (9.46M + 19 × 7.56M) ≈ 2.2 × 10^11 ANDs: about 24 times
N2's 9.30 × 10^9, and 1/2,000 to 1/4,000 of N3's 4.9 to 9.2 × 10^14. Variant H adds about 10.4M a draw. An egress B of
256 (the id event fits) cuts the frame opening to (3 + 2D) compressions, 1,687,630 and 2,502,472, saving about 1.7M a
draw. An ingress draw at the rehearsal's D_in = 7 is about 5.43M: the head, the frame (2,727,604), its template and an
M-way field mux (about 80·M, 10K at M = 128), and the port.

**What the prover keeps,** from the window's registration until the tie's outcome of record, by `proof_deadline`:
N2's opening (the firewall's `Opening`), the frame tree's leaves (64 bytes a frame, 2.3 MB at 35,386 frames, with
the frames rebuilt from the ids and the template, where raw frames at B = 4,096 would be 145 MB a window and 209 GB a
day), the port's rows and salts, and the Commit's step rows the port's binding reads. The coins come after both
registrations.

**Composition.** N2 (`All`) holds the head's fields and counts and that the certificate signs it, frames root
included. N3 (`All`, a design) holds every frame under that root to its template and outbox, whose egress outbox
becomes ids. The tie (sampled) holds each drawn frame's id to the Commit's, which is N3's outbox agreement with the
values on a served link. The three read one head leaf, 2 + direction, under one registered root.

**Reuse.** `NetworkHidingRead` (the head, and the port's row at N_BITS = 1,024), `Sha512Compress`, `Hm96Sha512Leaf`,
`FrameV3Read` and `LmsEqual`. The new Definitions are composites of those, with no new hash.

**The prototype** (`io_tie`, on a fake upstream) has the reference (`io_tie.reference`: the head's opening, the draws,
the frames, the ids and the port's rows, in the clear) and one draw's circuit, `IoTieDraw{B, D, FOUT, C}`:
`NetworkFrameRead{B, D}` (the frame tree's leaf, path and root, with `NetworkFrameSwap` ordering each level),
`IoTieEvent{B}`, the port row's `NetworkHidingRead` and `IoTieColumn`. Its tests hold the circuits to the reference,
on honest windows and on a duplicated frame, swapped responses, a wrong port id, a zero-padded id, another path and
another root. Still to build: `IoTiePosition{T, FOUT}` (k = ⌊u·N / 2^64⌋ and k < N, N the head's counts' sum), the
head's opening inside the unit and the K draws' AND, and the ingress side (`IoTieRequest{B, M}`).

**Open for proofs.**
- Variant P or H, and who owns the port's binding to the step rows.
- Hidden-position reads (frame and port) under the one soundness theorem.
- A unit on `All` with K live coins mapped through a witness N, against a public population of draws.
- The ε form the integrity profile takes: the sum over windows, or the share of tokens on near-equal windows.
- An egress B of 256 while ingress keeps 4,096.

What this establishes, once built: on every accepted link-window, the wrong share of drawn positions is the stated ε
at δ, so the bytes the user receives are `event(id)` of the committed ids, and its prompts the committed inputs, at
that share; FIXLEN and the request-major order are checked at the draws. What it doesn't: the port's agreement with
the step rows (compute's, in variant P), any frame not drawn, and anything about the text a client shows beyond the
public tokenizer's decode.

## The Program's logical schedule and the clock-sync advice (`schedule`)

The Program fixes each real frame's content, its **anchor** and its **logical index** λ. The anchor is the recorded
bucket of the request it answers. λ is its position within its session, from the canonical gate order, so it is never
negative: `LogicalFrame` rejects a negative index, as Lean's ℕ does. With one, a frame anchored in window 1 could be
eligible in window 0, and window 0's grid would read window 1's sync.

A window declares one clock-sync value `ClockSync(δ, ρ)`, from the protocol's set `SyncSet`: for example 16 offsets ×
8 rates, which is 7 bits. δ is a whole number of buckets. ρ is a positive fraction of frames per bucket, below 1 for a
session slower than one frame per bucket (a POUS band forward takes about 13 buckets). A frame then moves through three
steps:
- it is **eligible** at `anchor + δ + ⌊λ/ρ⌋`, using the sync of its anchor's window;
- the shaper **hands** it to the certifier in canonical order, at `h(k) = max(eligible(k), h(k−1))`;
- the certifier's own rule then **releases** it.

`expected_grids` is the Program's grid, and `run_shaper` drives a certifier as an honest sender does. Lean's
`NetTiming.Protocol.Schedule` defines the same objects (`eligible`, `handingBuckets`, `expectedRow`), and the Program
`constantRate` whose advice is the content advice and one in-set sync per window (`Security/Proofs/NetworkCertifier/README.md`,
modelling choices 13–17).

## The audit (`audit`)

An egress link-window (`Audit.check_link`) is accepted if:
- its status is complete;
- every slot equals the Program's grid (for a committed record: its digest equals the Program grid's digest; for a
  certified record: its head's record digest does);
- its declared clock sync is in the set, and is the spatial unit's sync for the window;
- it has exactly one record, present by the verdict deadline;
- for a certified record, its certificate verifies under the certifier's key (`key`), names the record's link,
  direction and window, and signs its record digest under the certifier's salt (`certificate_digest`).

Anything else is a violation, and the first that applies names it:
- `no-verdict`: no record by the deadline (for a certified record, also one whose deadline bit is 0);
- `duplicate`: more than one record for the link-window, whatever they say (the audit never picks one);
- `signature`: a certified record's certificate doesn't verify, or names another link-window, or signs another digest;
- `status:<reason>`: the status is not complete;
- `advice`: the declared clock sync is outside Σ_sync, or differs from the unit's for the window;
- `mismatch`: a slot differs from the Program's grid (for a committed record, the digest differs).

A violation is **recorded, and the run is rejected**, as a policy signal. There is no halt: later windows are still
checked.

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
than predicts, so no grid is compared. An ingress link-window is accepted if it has exactly one record, present by the
deadline, and its status is complete; otherwise it is a `no-verdict`, `duplicate` or `status:<reason>` violation, as on
egress. A committed record's status is read in the clear, and the digest binds it. The verifier has no Program grid to
recompute an ingress digest from, though, so unlike on egress it can't catch a published status that disagrees with
the digest: that rests on the certifier publishing its own status (W5). A certified ingress record's status is the signed
head's, so it rests on the certifier instead, and the audit (`check_ingress_link(..., key=key, params=params)`) also
requires the head's T, r, B, τ_b and parameters digest to be `params`'s, or it is a `mismatch`. A run is accepted only if every link-window of
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
| Envelope, with no clock | `T·log2(r + 1)` bits per link-window | Proposition 2 |
| Accepted runs, with clock-sync advice | `A_c + W·log2 #Σ_sync + n·log2 D` bits | `SecurityProofs.EncardAccSeqsLeSync`; for the shipped schedule, `SecurityProofs.EncardAccSeqsLeConstantRate` |
| The shipped schedule, over the first m windows | `A_c + min(W, m)·log2 #Σ_sync + n·log2 D` bits, for n ≤ min(W, m)·T steps | `SecurityProofs.EncardAccSeqsLeConstantRateWindows`; for a decoder of the wire, under W2 and J, `SecurityProofs.EncardDecodableLeConstantRate` |
| The shipped schedule, per window | `A_c` once, then `log2 #Σ_sync + T·log2 D` bits per window over the run's first m whole windows (n = m·T ≤ W·T): 7 bits per window at 16 × 8 sync values under J. Cumulative: a later window may carry more, since content advice and an earlier window's sync can first show in it | `SecurityProofs.EncardAccSeqsLeConstantRatePerWindow`; for a decoder of the wire, under W2 and J, `SecurityProofs.EncardDecodableLeConstantRatePerWindow` |
| A fixed clock, under J (D = 1) | 0 bits of timing | `SecurityProofs.EncardAccSeqsLeFixedClock`; for the shipped schedule, `SecurityProofs.EncardAccSeqsLeConstantRateFixedClock` |
| Ingress, where no clock applies | `log2(r_in + 1)/τ_in` bit/s per link, under J (D = 1), statuses complete | `SecurityProofs.EncardIngressDecodableLe`, `SecurityProofs.EncardIngressObsDecodableLe` |
| Ingress, content plus timing | `log2 #Cs` bits plus the ingress timing | `SecurityProofs.EncardIngressDecodableLeContents`, `SecurityProofs.EncardIngressObsDecodableLeContents` |
| Ingress, per link-window (the K charge's unit) | `T·log2(r_in + 1)` bits per link-window over the first m whole windows, plus `T·log2 D` of jitter per window, statuses complete. Unlike egress, each step's bound does not depend on what came before | `SecurityProofs.EncardIngressObsDecodableLePerWindow`; content plus timing, `SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow`; for a decoder of the wire, W2 and `SecurityProofs.EncardIngressDecodableLe` at n = m·T |

The inference-only policy charges an isolation unit the ingress covert capacity from the last wipe before it starts,
plus what survived that wipe. `capacity.k_share` expresses bits as a share of K, which the caller passes in bits
(`calibrate(k_bits=…)`, `--k-bits`): 9.12e7 under NCI's inference-only policy (α·Γ = 9.6e−11 × 9.5e17). The
calibration report records the K it used.

Every statement in the table's Lean column is a guarantee in `Security/lean-audit.json`, frozen as no-function: each is
stated over the model, not the network check's code (`Security/Proofs/NetworkCertifier/README.md`, "The records"). Four
are the bounds `calibration.calibrate` reports: egress over a horizon
(`capacity.egress_bits_per_day_one_clock`, `SecurityProofs.EncardDecodableLeConstantRate`) and per window
(`clock_sync.bits_per_window`, `SecurityProofs.EncardDecodableLeConstantRatePerWindow`), and ingress over a horizon (the
K charge, `k_charge.share_of_k_by_span` from each link's `rate_bps`, `SecurityProofs.EncardIngressObsDecodableLe`) and
per link-window (`envelope_bits_per_window`, `SecurityProofs.EncardIngressObsDecodableLePerWindow`, whose interaction
is any link's). The steps of their proofs are lemmas (`Security/Proofs/NetworkCertifier/Lemmas.lean`): proved and
audited, with no record.

The shipped schedule's egress bounds are proved in Lean for `constantRate`, the Program of `schedule`, with no
dataflow hypothesis: for every frame function, sync set and operator, over a horizon within the W windows with advice
(n ≤ W·T), since those are the windows the audit checks. The clock-sync bound is 7 bits per window at 16 × 8 sync
values, summed from the run's start (`SecurityProofs.EncardAccSeqsLeConstantRate`). Under J with one sync
value it is 0 bits of timing (`SecurityProofs.EncardAccSeqsLeConstantRateFixedClock`). A horizon within the first m
windows is charged only those m windows (`SecurityProofs.EncardAccSeqsLeConstantRateWindows`), and a decoder of the
wire recovers no more (`SecurityProofs.EncardDecodableLeConstantRate`).

**The block cap** (`SecurityProofs.EncardAccSeqsLeBlocks`) sharpens these only on a small grid or a horizon of a few
buckets. Its conditions fail for blocks of one window, since δ and ⌊λ/ρ⌋ carry a frame into later windows and the
queue carries across them (`constantRate_not_windowBlocks`). So for this schedule it applies only with the horizon as
one block, whose timing advice is every window's sync. In its count form (`SecurityProofs.EncardAccSeqsLeBlocksCounts`)
a cell is one link's bucket and its occupancy a count from 0 to r, and the one block caps a horizon of n buckets on L
egress links at `#Ac · D^n · min(#Σ_sync^W, (r + 1)^(L·n))`: the lesser of the advice cap and the left-packed envelope.
The count form's first condition holds only when each bucket's counts fix its rows, as when every frame has the same
payload; with two payloads it fails (`constantRate_not_framesRead_counts`).

So for a horizon of n buckets within the first m windows, the one-block cap is below the advice cap `#Σ_sync^m`
exactly when the envelope is: `(r + 1)^(L·n) < #Σ_sync^m`. On whole windows (n = m·T) that is
`(r + 1)^(L·T) < #Σ_sync`, and since L ≥ 1 it can hold only if `T·log2(r + 1) < log2 #Σ_sync`: a window's grid shows
fewer patterns than Σ_sync has values.
- **At T = 600 with 16 × 8 sync values** the one-block cap is below the advice cap only on a horizon shorter than
  `7/(L·log2(r + 1))` buckets, inside the first window (7 buckets on one link at r = 1). One link's grid carries at
  least 600 bits per window against the sync's 7, and a horizon that reaches window m ≥ 2 has more than (m − 1)·600
  buckets, each worth at least one bit, against 7m.
- **On a small grid it does sharpen them:** at T = 2, r = 1 and W = 3 on one link, the one-block cap is 2^6 = 64,
  against the advice cap's 128^3 = 2,097,152.

The ingress bound is proved in Lean, for a decoder in X that reads the wire (`SecurityProofs.EncardIngressDecodableLe`)
or the records (`SecurityProofs.EncardIngressObsDecodableLe`). It counts `(r_in + 1)^T` occupancies per ingress
link-window (times `D^T` for jitter, 1 under J), given the frames' content, on runs whose statuses are all complete
(`StatusComplete`). The reference audits ingress statuses (see above), so an accepted verdict on every link of ∂X
establishes `StatusComplete` for the windows audited. That is the acceptance hypothesis of every ingress guarantee:
`SecurityProofs.EncardIngressObsDecodableLe`, `SecurityProofs.EncardIngressDecodableLe`, their `Contents` versions and
the two per-window forms. Over the first m whole windows of a run (n = m·T; for the K charge, the run from the last
wipe) the count factors per link-window: `(r_ℓ + 1)^T` on each link in each window, `T·log2(r_in + 1)` bits, which is
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
the trusted text: the last section of `Security/Models/NetworkCertifier/Guarantees.lean`'s documentation.

## Calibration (`calibration`)

`network-certifier-calibrate TRACE [--ingress-bucket-ms 100]` reads honest timing traces (the format is in the module
docstring) and sizes:

- **Σ_sync:**
  - δ from 0 up to the p99.99 first-frame latency, in 16 steps;
  - ρ geometrically from the p0.01 production rate up to the p50 rate, in 8 values, each rounded down to a fraction
    with denominator at most 1024 (rounding down keeps a value lenient). Both rates are taken per configuration first:
    the low end is the p0.01 of the configurations' p0.01 rates, and the high end the median of their medians over
    windows. A configuration's windows are the windows its egress frames span, 1 + its largest anchor // T, and the
    high end is the least median rate t such that the configurations with a median rate at most t span at least
    ⌈W/2⌉ of all W windows. So a run cut into more configurations, at a dropped window say, keeps its high end.
- **The honest prover's choice:** one clock per spatial unit, so per window one value for every egress link of the
  configuration: the value met on all of them with the least total hand time over the window's frames on every link
  (their total shaping delay), with ties to the smaller δ, then the larger ρ. Taking the least δ first would pick a
  slow ρ to save one bucket, and hold every later frame for seconds. Each link's frames are in the Program's canonical
  order, which is step-major for a batching engine, so a long session interleaves with later windows' frames.
  While a window is chosen, later windows take their most eager value; that only lowers hand times, so a window found
  met stays met.
- **The online prover** (`online.Prover`): that choice is in hindsight, from all of a window's frames. A live prover
  declares window w's sync at its first bucket, w·T: the rule on the frames ready by w·T − lag (the canonical prefix
  it has read, each anchored earlier) and the earlier declarations, and nothing else. A rule (`online.rule`) sees, for
  each recent window, the least δ per ρ that would have met the frames seen of it with the other windows as declared.
  `cover-K-M` takes the value of least estimated hand time whose δ is M buckets past every one of the last K windows'
  least δ at its ρ, `-fallback` declares the envelope (the largest δ and smallest ρ the calibration's prover declared)
  while one of them was late under its own declaration, `-rhoR` declares the chosen δ at the ρ R steps slower, and
  `last` is `cover-1-0`. `sticky-K-M-N[-rhoR]` declares `cover-K-M[-rhoR]` only in windows 0, 1, 1+N, ...; in between
  it keeps its value while that still has M buckets to spare, and otherwise escalates to a value fixed by the kept one.
  A window with no history declares the envelope. A configuration cut from one run at a dropped window may start from
  the previous one's history (`Prover(prior=…)`).
- **The lag:** live, 1 … T − 1 buckets. It lets the choice be made off the proxy's tick, since from bucket w·T − lag on
  nothing changes it, and a choice not ready at w·T is declared as the envelope. `online.declare(lag=…)` replays the
  declarations with every choice ready, so it gives a live run's wherever its choices were ready; lag 0, the choice at
  the tick itself, is the reference replay's.
- **What the prover's choice changes:** the bound charges log2 #Σ_sync bits per window whatever chose the value, so it
  holds under any rule, any lag, and an unready choice: the envelope is in Σ_sync, and the audit checks a
  declaration, as advice, only for membership in Σ_sync. What changes is who can use those bits: a sync chosen before
  the run, or from the trace's model, doesn't depend on the sender, but an online prover reads the sender's timing, so
  the sender steers every declaration after its first (in the injected-clock replay, the burst, spread and bits
  senders each move the declared δ, and the wire differs from honest's in every window but the first). Live, whether
  a choice is ready by its tick depends on the proxy's load as well, which the sender can move too, and the same
  bits are charged for it. `Rule.choices` bounds how many values a sender could make a rule declare in a window:
  about #Σ_sync for `cover`, at most 2 between a `sticky` rule's free windows.
- **The honest miss rate:** a window that no value meets on every egress link is a miss, which is a violation. A
  larger δ or a smaller ρ only raises hand times, so a window misses exactly when the most lenient value misses on some
  link. A window is the configuration's, so the rate is the configurations' missed windows over their windows
  (1 + largest anchor // T each), and a link's `honest_windows` and `honest_misses` are those of its configurations.
  One clock can miss where a choice per link would not: the value met on every link can hand one link's frames earlier
  than that link's own choice, and a later window whose frames were met only behind them then misses. The target is
  1e-4.
- **The egress allocation r:** the least r at or above twice the p99.9 of per-bucket demand whose FIFO replay has no
  overflow and a p99 queueing delay of at most one bucket. Both improve monotonically in r, so the search doubles and
  then bisects. If one bucket's demand exceeds Q, every r overflows: the tool exits with an error naming the demand and
  Q, for the egress and the ingress grid alike.
- **The ingress grid:** the smallest r_in that carries honest ingress, at τ_in. τ_in is **100 ms**, decided on 29 Sep
  2026 from a calibration of honest traces. The grid stays configurable, so `--ingress-bucket-ms` still compares
  alternatives. With arrival times (`arrival_ms`), τ_in may be finer than the trace bucket, and the report adds the
  delivery delay: a frame waits for the next ingress bucket boundary, then for its FIFO slot. The p99 FIFO delay is at
  most one ingress bucket by default; `--ingress-max-delay-ms` sets the budget in ms instead, so different τ_in are
  compared at the same latency.

The report has the per-link bandwidth, the utilization and padding share, the clock-sync set and its bits, the (δ, ρ)
range the honest prover declared, the first-frame latency it declared (the p50 and p99 of its declared δ, in ms), the
shaping delay (hand time minus ready time, over every frame), the capacity per day, and the K charge by span.

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
    canonical order.
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
  - **Into the block only.** At that tick the certifier enqueues its own salt frame ahead of the bucket's arrivals, and
    behind any queue carried from the window before (`Bucketer(first=…)`). It is kind `SALT` (3) on `SALT_STREAM`
    (2^32 − 1), and its payload is u64be p ‖ `PLACEHOLDER`, 64 zero bytes (`SALT_FRAME_BYTES`, 81 bytes, the least
    ingress frame a salted proxy takes). It takes one of the window's ingress slots like any crossing, so r_in carries
    honest ingress plus one frame per PoUW window. The `Bucketer` follows it by its place in the queue, never by its
    bytes (`Bucketer.own`), so a frame of kind `SALT` read from the link, even one of the same bytes, is never taken
    for the certifier's, and one a fault drops is never released. When the ingress certifier releases it, the proxy
    writes the frame's header, p and the salt (no fill; the header's length is 8 plus the salt's bytes) on a server
    connection of its own. The proxy never reads that connection, so nothing the server writes there reaches egress.
    The egress side never holds the salt or anything drawn from it: the egress grid, its padding and the wire are the
    unsalted run's.
  - **In the record: nothing of the salt.** The salt frame's bytes and its slot are fixed by the window and the traffic,
    so every record, before t_d and after, is a function of the traffic alone, and two salts at the same traffic give
    the same records. The record's formats are unchanged: the v1 digest and the v2 frame tree bind the salt frame like
    any other frame, and N3 checks it against its public payload. Neither the Lean analysis nor the Lean ports change.
    A placeholder in the commitment's slot was the smaller of the two changes compute accounting offered (18 Oct): the
    other, the commitment in the record published at t_d, changes the record's format, its digest and certificate,
    and N3's public input, and with the certifier trusted it binds nothing the reveal doesn't.
  - **The sealed slots.** The certifier's record cells are its grids and two sealed slots per PoUW window:
    `_salts[p]`, which holds the salt and which only `reveal` opens, and `_weights[p]` (next item). So the record's
    open cells (`Accounting.Attribution.openCells`, the record minus those slots, ATTRIB-R4) are the grids, and what
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
    refuses a window whose weights weren't recorded. The order is the certifier's word, under `trusted-nebius-host`,
    like the salt.
  - **Revealed on the window's registration, after t_d.** `Proxy.reveal(p, registration)` gives PoUW window p's salt.
    The verifier calls it with the registration the proof service returned for p (`Registration`, from
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
    certifier runs on the link, off the prover's node, and the proof service's challenger (#2573), which is the
    auditor's, on an auditor host. `challenger_key` comes from the auditor's deployment record, never from the
    prover's configuration, and `trusted-nebius-host` covers that record's custody. Each proxy draws a run id at
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
    unchanged. What still rests on `trusted-nebius-host` is the certifier's key custody, its signing key and leaf state
    on its host (`LmsSigner`, one leaf at a time across the threads that share it) and its word on what it signs, and
    the deployment record's custody: the keys and the run id it gives each side. The challenger's key is the
    auditor's, on its own host. A proxy with neither key is the in-process call the tests and replays make
    (`reveal(p, registration)`, which returns the salt): it takes no request, and holds only while called on the
    auditor's side.
  - **The audit's read** (`check_salts(audit, link, records, windows, salts, pouw_window=k, salt_bits=…)`). PoUW
    window p is accepted, at its first certifier window w = p·k, if w has exactly one raw record, which holds exactly
    one salt frame of p, carrying the placeholder, and p's revealed salt has `salt_bits` bits. Anything else, a missing
    salt included, is a `salt` violation at w, recorded with the audit's verdicts. No record binds the salt: the
    revealed salt is the certifier's word, under `trusted-nebius-host`. A committed or certified record doesn't show
    its frames, so there a proof opens the salt frame.
  - **What it costs.** The protocol fixes the salt frame's place in the queue and its bytes, so it adds nothing to the
    ingress count `(r_in + 1)^T`. Its release into the block carries the certifier's coins, which no outside party
    chooses and nothing X sends can change, so it costs 0 bits by the argument the trusted text makes for the
    challenger's coins (D7, `Security/Models/NetworkCertifier/Guarantees.lean`). That text states the exemption for coin
    openings on a coin link, and doesn't name the certifier's salt yet.
- **Back from maintenance** (`Proxy(start=k, position=p)`) is off by default, and with it off every record, wire and
  verdict is as above.
  - **What.** A certifier that was down serves from window k of the trusted clock on: each grid's first tick is window
    k's first bucket, its first records are window k's, and the first egress frame it takes is the Program's at
    position p, the frames before p being the earlier windows'. It carries nothing from before it went down: no queue,
    stream, fault or salt.
  - **The windows it was down for** have no record, and the audit gives each of their link-windows `no-verdict`, as for
    any missing record. The certifier's audit has no maintenance: excusing those windows is the verifier's plan's, in
    the accounting layer (`benchmarks/datacenter`'s `Maintenance`), which charges everything the block held instead.
  - **Going down.** `windows` ends a proxy's run early: `Proxy(windows=w)` serves windows 0 … w − 1, publishes their
    records and closes the link. A run whose window w is the outage is then `Proxy(windows=w)`, nothing for window w,
    and `Proxy(start=w + 1, position=p)`. Its later windows are the full run's if nothing crosses into window w.
  - **Its limits.** An online prover declares from window 0's frames on, so a proxy back from maintenance takes its
    syncs. The certifier's Lean device port (`verity/core/devices/certifier/lean/`) models a proxy from window 0 with
    the salt off.
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
  | More than a bucket left unwritten on the link at a tick | `delivery-refused` |
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

The clock and the sleep are injected (`VirtualClock` serves a run driven event by event). `Shaper` and `Bucketer` do no
I/O, so the tests drive real sockets without reading the wall clock. `benchmarks/network_traces/active_replay.py`
replays recorded traces through the proxy in real time, and on an injected clock.

## Tests

`uv run tools/verity/check/suites.py verity/core/protocols/certifier`. The tests enumerate small
horizons, so they check the Lean statements on examples and prove nothing. By name, they mirror 19 of the analysis's 35
statements (in `Security/`, not the certifier program's `lean/`)
`NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, 12 of them guarantees (`Security/lean-audit.json`'s `guarantees`)
and seven lemmas, and three more proved lemmas:
- **`test_network_certifier_grid.py`:**
  - Proposition 2's ten statements, by enumeration: `SecurityProofs.CardFree`, `SecurityProofs.CardLeftPacked`,
    `SecurityProofs.FillInjective`, `SecurityProofs.ExistsFillOfLeftPacked`, `SecurityProofs.CardCountsEqLe`,
    `SecurityProofs.CardCountsLeLe`, `SecurityProofs.ReleasedSelf`, `SecurityProofs.SumReleasedLe`,
    `SecurityProofs.EncardFifoGrids` and `SecurityProofs.EncardFifoGridsLeOfAvailable`;
  - the lemmas `occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid` (`NetTiming.SecurityProofs.Occupancy`);
  - the certifier's semantics, which Lean takes as hypotheses (W4 packing, W5's untimed status, W6 fail-closed);
- **`test_network_certifier_capacity.py`:**
  - `SecurityProofs.EncardLeProdOfBranching`, the tree lemma;
  - `SecurityProofs.EncardAccSeqsLeSync`: accepted grids at most `#Σ_sync^W`, over every sender and declaration on a
    small horizon, with ρ below 1, and with a reacting operator;
  - `SecurityProofs.EncardAccSeqsLeFixedClock`: the fixed clock's single accepted grid;
  - `SecurityProofs.EncardObsDecodableLe`: under a fixed clock, accepted secrets all show one grid sequence;
  - `SecurityProofs.EncardAccSeqsLeBlocks`: the block bound on random Programs;
  - `SecurityProofs.EncardAccSeqsLeConstantRate` and `SecurityProofs.EncardAccSeqsLeConstantRateFixedClock`: the two
    tests above, for the shipped schedule;
  - `SecurityProofs.EncardAccSeqsLeConstantRateWindows`: runs accepted on their first m windows show at most
    `#Σ_sync^m` grid prefixes;
  - `SecurityProofs.ExpectedRowCausal`: a window's grid is unchanged when later windows' syncs are drawn again, and a
    negative logical index, which would break it, is rejected;
- **`test_network_certifier_audit.py`:** record-and-reject and each violation, which no Lean statement names; one clock
  per spatial unit, the advice `SecurityProofs.EncardAccSeqsLeSync` counts; and the ingress links' status check, Lean's
  `StatusComplete`;
- **`test_network_certifier_commitment.py`:**
  - the elided digest is injective on left-packed grids, by enumeration;
  - it binds the link, window, status and parameters;
  - the committed audit's verdicts equal the raw audit's;
- **`test_network_certifier_certificate.py`:**
  - record v2's byte layout, rebuilt field by field and pinned by one digest; the status codes;
  - the head is injective on left-packed grids, by enumeration, and binds every field; padding never enters the tree,
    and every frame opens at depth D;
  - the certified audit's verdicts equal the raw and committed audits', per link-window, on egress and ingress;
  - a certificate moved to another window, link or direction, relabelled, forged, checked under another key or signed
    by the same key in another of the certifier's sessions (the cross-session replay) is a `signature` violation; an unset deadline bit is a `no-verdict`, another Σ_sync or Q a `mismatch`;
  - a signer whose state is missing or behind refuses, and that link-window is a `no-verdict`; it signs again once its
    state is restored, never reusing a leaf. A second signer on held state, or one whose floor is past the state, refuses
    too. The signer is the certifier's `LmsSigner` on a tree of 2^6 leaves (`tests/signers.py`), not the deployed 2^20;
  - a second run of the key under a session its state has started is refused at its start, and its link-windows are
    `no-verdict`s;
  - the certificate message is LMS's network-certificate role digest, and binds the key, the session, the link-window
    and the digest;
- **`test_network_certifier_service.py`:** one link-window through the proof service: the round trip to an exhaustive
  accept, nothing of a certificate or a violation's kind in what the challenger keeps, each refused registration (an
  extra field, a certificate in the clear, a window other than the order), and each reject (a bad signature, a digest
  mismatch, another window or link, a certificate from another session, an unset deadline bit, a late proof, an opening of
  another root), each held to the stand-in; the unit's id and its gates by size class; N2 compares no grid, so a window
  whose frames differ from the Program's at the profile's counts (in content, or in order within a bucket) is accepted
  where the stand-in rejects it, its frames being left to N3, and an egress head under another parameters digest is
  rejected; a link whose egress end has neither an occupancy profile nor a served profile has no verdict; a head of
  another size class is a reject;
- **`test_network_certifier_served.py`:** a served link: honest served traffic is accepted in every window; a window
  whose response is handed late, or carries a frame more or one fewer, is refused while the audit accepts it; a
  request straddling a window's end refuses that window and the next, and behind the admission rule the same arrivals
  are accepted; behind the admission rule, under random dense load, and on random idle runs, each window's derived
  counts are the Program's grid (`expected_grids`); a window that doesn't end idle has no derivation; the circuit
  equals its reference on four size classes, sync edge cases included; the served unit's id, public items and spec;
- **`test_network_certifier_head_check.py`:** N2's head check: the count check is 32·T − 1 ANDs and the fields
  736 + 8·L − 1; each part equals its reference on honest heads and refuses each tampered input where it belongs (link,
  direction, window, parameters digest or status: the fields; another profile row or edited counts: the counts; another
  root, salt or path: the opening), and the deadline byte and frames root are left free; the whole Definition equals its
  reference (slow);
- **`test_network_certifier_calibration.py`:** the calibration tool, and the honest prover's choice against its rule by
  brute force, on one link and on several under one clock;
- **`test_network_certifier_active.py`:** the active certifier over localhost sockets on an injected clock:
  - `Shaper` is `run_shaper`'s rule on random instances;
  - an honest run's records, hand times and wire are the Program's, accepted raw and committed;
  - senders that burst, spread, or pick their write times at random or by bits give the same wire, byte for byte;
  - a frame held past every sync is a mismatch;
  - a late tick, an ingress tick read early (the online prover's refusal), an overrun, a refused stream, a request
    forwarded a bucket late, a fault that drops a carried queue and out-of-set advice each fail closed;
  - with the salt on, each window's salt reaches the server on its own connection and nothing else carries it: the
    wire and the egress records are the unsalted run's, each ingress record holds one salt frame, the placeholder's,
    at the window's first bucket, and two runs on different salts publish the same records. With PoUW windows of two
    certifier windows and 256-bit salts, one 256-bit salt is drawn per PoUW window. `reveal` refuses a window
    without its registration (none, another window's, a receipt or digest that isn't the record's, a provenance its
    reader doesn't read as the window), until its last ingress record is published, and for a salt never drawn;
    `check_salts` accepts the revealed salts and rejects a missing one, one of another width and a second salt frame.
    The bucketer puts the certifier's own frame first in each window, behind the carried queue, and knows it by its
    place: a link frame of the same bytes is never taken for it, and one a fault drops is never released;
  - back from maintenance at window 2 of a Program with nothing in window 1, a proxy publishes exactly the full run's
    window-2 records and wire, and hands each frame when the full run does; so do a shaper and a bucketer on random
    instances. A proxy can't start past its windows or with an online prover;
  - the served bucketer admits whole requests while the window ends idle, holds the rest for the next window, and
    every record it publishes derives; it refuses an own frame and a window that can't hold one request;

No test mirrors the other eight statements of Theorems 1(b) and 3 (`SecurityProofs.EncardAcceptedRunsLe`,
`SecurityProofs.EncardDecodableLe`, `SecurityProofs.NoFaultOfAccepted`, `SecurityProofs.RecordedRowEqFill`,
`SecurityProofs.EncardAccSeqsLeAdvice`, `SecurityProofs.EncardAccSeqsLeBlocksCounts`,
`SecurityProofs.EncardDecodableLeBlocks`, `SecurityProofs.EncardDecodableLeConstantRate`), nor the ingress bound's four
(`SecurityProofs.EncardIngressObsDecodableLe`, `SecurityProofs.EncardIngressDecodableLe`,
`SecurityProofs.EncardIngressObsDecodableLeContents`, `SecurityProofs.EncardIngressDecodableLeContents`), nor the four
per-window forms (`SecurityProofs.EncardAccSeqsLeConstantRatePerWindow`,
`SecurityProofs.EncardDecodableLeConstantRatePerWindow`, `SecurityProofs.EncardIngressObsDecodableLePerWindow`,
`SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow`), and the sixteen witnesses (thirteen in
`NetTiming.SecurityProofs.Sanity`, three in `NetTiming.SecurityProofs.Schedule`) have no
Python counterpart.

**The difftest** checks the definitions those statements are about, not the statements. On 10,000 random instances,
the certifier program (`lean/`, `NetworkCertifier`), evaluated, gives what this package gives. The Lean side is
`expectedRow`, with `Accepted` of `constantRate` and `StatusComplete` decided per prefix of whole windows. The Python
side is `expected_grids`, `check_link`'s verdicts with the first differing bucket, `check_ingress_link`, and each
prefix's acceptance. `lean/scripts/difftest_vectors.py` writes the instances and this package's results, and the Lean
audit runs `lean/NetworkCertifier/DifftestMain.lean` on them (`runs` in `lean/lean-audit.json`).
`test_network_certifier_lean_difftest.py` checks that the data the record pins is still this package's. The program's
definitions are List forms of `NetTiming`'s `Finset` ones, and four guarantees prove them equal on everything the
difftest compares (`NetTiming.SecurityProofs.Code`): `CodeExpectedRow`, `CodeAcceptedConstantRate`,
`CodeStatusComplete` and `CodeSyncSet`.
