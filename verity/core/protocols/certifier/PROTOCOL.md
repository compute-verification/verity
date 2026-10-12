# Network accounting: the bucketed timing channel

A spatial unit is the machines behind network certifiers. The question is how much a covert sender can move across its
boundary through **when** frames leave, once the content and order of every frame are fixed by the declared Program.
This package is the reference for the answer, in the terms of its Lean analysis `NetTiming` (`Security/`: the
areas `Definitions/NetworkCertifier`, `Models/NetworkCertifier` and `Proofs/NetworkCertifier`, whose `README.md` has the modelling choices). The
tests mirror some of its statements by name; the Tests section lists which. The certifier program (`lean/`, the Lake
package `certifier_program`, Lean core only) is the code the difftest runs, proved equal to the analysis's definitions.

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
certificate**: a signature over the SHA-512 digest of the window's record (`certificate.Issuer`). A certificate is
witness. It enters proofs, which open frames of its record in-circuit, and never reaches the auditor in the clear, so
its hashes are unsalted. The audit checks it against the certifier's public key. With an honest certifier every
certificate verifies, and the audit's verdicts are those of the raw records.

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

**The certificate** is the record digest and a signature over a 64-byte message that binds it to the key and the
link-window, so a certificate can't be replayed onto another window, link, direction or key:

    message = SHA-512( [tag] ‖ I ‖ [ [link] ‖ [direction] ‖ u64be window ] ‖ [record digest] ),
              tag = "verity/network-certificate/v1", direction "egress" or "ingress", [x] = u64be len(x) ‖ x

This is LMS's network-certificate role digest (`verity.ml.crypto.lms.certificate_message`), the message its in-proof
verifier takes. I is the certifier's 16-byte LMS identifier. The scheme is LMS over SHA-512 (`lms.DEPLOYED`: n = 64, w = 1,
h = 20, the whole SHA-512 output, not truncated), and its unforgeability is `euf-cma/lms-sha512-n64`.
The certifier's `Issuer` holds it as a `Signer` (`key_id`, `sign(message)`; `certificate.LmsSigner`) and the audit as a
`PublicKey` (`key_id`, `verify(message, signature)`; `certificate.LmsKey`), so the scheme is a parameter of both. `LmsSigner`
keeps its next unused leaf in an `lms.LeafState`, which it holds from its first signature on. A signer whose state is missing,
behind the leaves its journal has used or its `floor`, or held by another signer, refuses (`SignerRefused`). The certifier then
publishes nothing for the link-window (`Issuer.refused` lists it), which fails closed as a `no-verdict`; it never signs with a
reused leaf.

**The check.** On egress the audit recomputes the record digest from the Program's grid, with a complete status, the
deadline bit set and the parameters digest of `params` and Σ_sync, and compares it with the certificate's. Under
`cr/sha-512`, equal digests mean equal records, so the audit stays exact. A certified record rests on `cr/sha-512`,
on the signature's unforgeability, and on an honest certifier; not on `cr/sha-256`.

## Through the proof service (`service`)

One link-window's certificates are registered through the proof service (`verity.core.protocols.sampled_proofs.service`)
as a `Spec` of its own (`service.LinkWindows`, one per link):
- **Committed:** one value, `certificates`: the link-window's egress and ingress certified records, leaves 0 and 1, each
  `hm96-sha512/row/v2` over `service.encode` (the head, then the certificate) with a fresh salt, in one `frame-v3-sha512`
  tree under one-stage's served domain for the run, the link and the window. The firewall keeps the records and salts
  (`Opening`). The challenger receives one salted root and never a certificate.
- **Selection:** `All`. A registration is one link-window, one proof unit, so an accepted outcome is the exhaustive fact.
  The record's `window` provenance is exactly `{"kind": "served", "window": w}`, and the challenger holds w to the
  registration's order (`LinkWindows.register`): the receipt keeps the order, and the order is the link-window.
- **Public:** the population, the window, the value's leaf count, domain and salted root, the parameters digests and
  the charge (log2 #Σ_sync per window, 7 bits at 16 × 8), both bound by the spec's context with the link, the run and
  the certifier's key id, and the check's outputs `verdict` and `deadline`. A rejected link-window shows only the
  service's `reject`; the violation's kind stays in the verifier. A registration with any other field, a second root,
  a root entry with another key, or anything more in `window` is refused.
- **The check** is `service.UNIT`, `NetworkLinkWindow` v0: one opaque primitive from the two leaves' committed digests to
  the verdict bit. It has no gates. Its Program's digest is the check's, so the spec, the stream and each registration
  bind to it, but nothing proves it in a circuit. Its verifier of record is a host-side stand-in, `LinkWindows.verify`,
  that reads the opening: it accepts when the opening reproduces the registered root and the audit accepts both
  records under the certifier's key (`check_link` against the Program's grid under the declared syncs,
  `check_ingress_link` on the status and parameters). The outcome passes the opening's root as the session's, which
  one-stage's audit holds to the registered one. An unset deadline bit is a `no-verdict` there, so a reject; a proof
  after `proof_deadline` is the service's `late`.

What this establishes: the outcome is accept exactly when the audit accepts the registered link-window's records, and
the outcome record binds the root, the draw and the session. What it doesn't: the stand-in reads the witness, so it is
not zero-knowledge, and the verifier of record must run where the certificates may go until the check is a Program with a
proof. One clock per spatial unit across links is not checked (each registration is one link-window, alone), and the
frames' consistency with the served computation is N3.

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
  certified record: the digest its certificate signs does);
- its declared clock sync is in the set, and is the spatial unit's sync for the window;
- it has exactly one record, present by the verdict deadline;
- for a certified record, its certificate verifies under the certifier's key (`key`), names the record's link,
  direction and window, and signs its record digest.

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
around `grid.Certifier`, `schedule`'s handing rule, `commitment` and `audit`, and it adds no format. Its records are
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
  (`Bucketer`, `calibration.size_ingress_ms`'s model).
- **Advice:** the declared syncs, one per window, are given before the run, or an online prover (`Proxy(prover=…)`,
  lag 1 … T − 1) declares each at its window's first tick, before anything is handed, from the frames the shaper has
  taken. The proxy never chooses on a tick: right after tick w·T − lag (window 0's before the first tick) it hands
  `Prover.snapshot()` to a worker process (`active.worker()`, off its event loop and GIL), and the tick at w·T
  declares the result if it is done, else the envelope, logged as unready (`LaggedChoices`). The proxy shuts the
  worker down when its run ends.
- **Faults fail closed** (W6). Each one maps to a status:

  | Fault | Status |
  |---|---|
  | A tick that runs more than `slack` after its boundary. Its bucket is still emitted, as padding, so no bucket slips into the next | `monitor-unavailable` |
  | More than a bucket left unwritten on the link at a tick | `delivery-refused` |
  | A server that refuses a stream | `delivery-refused` (ingress) |
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
  - a certificate moved to another window, link or direction, relabelled, forged or checked under another key is a
    `signature` violation; an unset deadline bit is a `no-verdict`, another Σ_sync or Q a `mismatch`;
  - a signer whose state is missing or behind refuses, and that link-window is a `no-verdict`; it signs again once its
    state is restored, never reusing a leaf. A second signer on held state, or one whose floor is past the state, refuses
    too. The signer is the certifier's `LmsSigner` on a tree of 2^6 leaves (`tests/signers.py`), not the deployed 2^20;
  - the certificate message is LMS's network-certificate role digest;
- **`test_network_certifier_service.py`:** one link-window through the proof service: the round trip to an exhaustive
  accept, nothing of a certificate or a violation's kind in what the challenger keeps, each refused registration (an
  extra field, a certificate in the clear, a window other than the order), and each reject (a bad signature, a digest
  mismatch, another window or link, an unset deadline bit, a late proof, an opening of another root);
- **`test_network_certifier_calibration.py`:** the calibration tool, and the honest prover's choice against its rule by
  brute force, on one link and on several under one clock;
- **`test_network_certifier_active.py`:** the active certifier over localhost sockets on an injected clock:
  - `Shaper` is `run_shaper`'s rule on random instances;
  - an honest run's records, hand times and wire are the Program's, accepted raw and committed;
  - senders that burst, spread, or pick their write times at random or by bits give the same wire, byte for byte;
  - a frame held past every sync is a mismatch;
  - a late tick, an ingress tick read early (the online prover's refusal), an overrun, a refused stream, a request
    forwarded a bucket late, a fault that drops a carried queue and out-of-set advice each fail closed;

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
