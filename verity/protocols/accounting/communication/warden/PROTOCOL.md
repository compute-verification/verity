# Network transparency: the bucketed timing channel

A spatial unit is the machines behind network wardens. The question is how much a covert sender can move across its
boundary through **when** frames leave, once the content and order of every frame are fixed by the declared Program.
This package is the reference for the answer, in the terms of its Lean package `network-timing` (`lean/`). The tests
mirror some of its statements by name; the Tests section lists which.

## The warden (`grid`)

Each boundary link has an active warden with fixed parameters (W1). The protocol programs them from a security
parameter, and the warden only executes them:
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

A raw grid is 2.3 GiB per link-window at r = 1,024, B = 4 KiB and T = 600. The warden publishes one digest and the
status instead, which is 32 bytes per link-window, or about 46 KB per link per day.

- **Padding is elided.** Left-packing (W4) fixes a row by its count n and its n real frames, so a row's digest covers
  only those: `tagged_sha256("verity-network-warden/row/v1", u32be(n) ‖ f_0 ‖ … ‖ f_{n−1})`. A row that isn't
  left-packed has no encoding.
- **The record digest** is `tagged_sha256("verity-network-warden/record/v1", …)`. It covers:
  - the canonical JSON of the link, the window, (T, r, B, τ_b) and the status;
  - then the T row digests.

  `tagged_sha256` is `verity.primitives.commitments.identity`'s.
- **The check** recomputes the digest from the Program's grid, with a complete status, and compares. Under `cr/sha-256`,
  equal digests mean equal records, so the audit stays exact. The verifier never needs the raw grid, because the
  Program fixes every egress frame.
- **The status in the clear** is only for classifying a violation. The digest binds it too, so a published status that
  disagrees with the digest is a mismatch.

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
- the shaper **hands** it to the warden in canonical order, at `h(k) = max(eligible(k), h(k−1))`;
- the warden's own rule then **releases** it.

`expected_grids` is the Program's grid, and `run_shaper` drives a warden as an honest sender does. Lean's
`NetTiming.Protocol.Schedule` defines the same objects (`eligible`, `handingBuckets`, `expectedRow`), and the Program
`constantRate` whose advice is the content advice and one in-set sync per window (`lean/README.md`, modelling choices
13–17).

## The audit (`audit`)

An egress link-window (`Audit.check_link`) is accepted if:
- its status is complete;
- every slot equals the Program's grid (for a committed record: its digest equals the Program grid's digest);
- its declared clock sync is in the set, and is the spatial unit's sync for the window;
- it has exactly one record, present by the verdict deadline.

Anything else is a violation, and the first that applies names it:
- `no-verdict`: no record by the deadline;
- `duplicate`: more than one record for the link-window, whatever they say (the audit never picks one);
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
channel (`lean/README.md`, modelling choice 6). An ingress link's rows are the Program's input, which it reads rather
than predicts, so no grid is compared. An ingress link-window is accepted if it has exactly one record, present by the
deadline, and its status is complete; otherwise it is a `no-verdict`, `duplicate` or `status:<reason>` violation, as on
egress. A committed record's status is read in the clear, and the digest binds it. The verifier has no Program grid to
recompute an ingress digest from, though, so unlike on egress it can't catch a published status that disagrees with
the digest: that rests on the warden publishing its own status (W5). A run is accepted only if every link-window of
every link the audit checks is, so an accepted verdict on every link of ∂X, with `check_link` on egress and
`check_ingress_link` on ingress, means every boundary link's status was complete in every window audited.

Every capacity bound holds on accepted runs only. A rejected window may already have carried up to its unbound capacity,
about 8·T·r·B bits (`capacity.unbound_bits`). On committed records, the bounds also assume `cr/sha-256`.

## Capacity (`capacity`)

| Quantity | Value | Lean |
|---|---|---|
| Free slot choice | `2^(T·r)` occupancies | `SecurityProofs.CardFree` |
| Left-packed | `(r + 1)^T` occupancies, attained by the FIFO warden | `SecurityProofs.CardLeftPacked`, `SecurityProofs.EncardFifoGrids` |
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

`lean-audit.json` lists as guarantees only the four whose bounds `calibration.calibrate` reports: egress over a horizon
(`capacity.egress_bits_per_day_one_clock`, `SecurityProofs.EncardDecodableLeConstantRate`) and per window
(`clock_sync.bits_per_window`, `SecurityProofs.EncardDecodableLeConstantRatePerWindow`), and ingress over a horizon (the
K charge, `k_charge.share_of_k_by_span` from each link's `rate_bps`, `SecurityProofs.EncardIngressObsDecodableLe`) and
per link-window (`envelope_bits_per_window`, `SecurityProofs.EncardIngressObsDecodableLePerWindow`, whose interaction
is any link's). The table's other statements are lemmas: proved and audited, with no record.

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

Under the proof service the warden is the network certifier and its records are network certificates. This section
fixes the two terms of the charge that the capacity bounds leave open: the content advice `A_c` and the coins (D7).
Nothing in it is implemented yet. The numbers are at the llama8b-node1 calibration (`art:962e249a`,
`calibrations/llama8b-node1.json`): T = 600 buckets of 100 ms, B = 4,096, one egress link at r = 1,686, one ingress
link (`requests`) at r_in = 34 and τ_in = 100 ms, Σ_sync of 16 × 8 values, and J (D = 1).

### What `A_c` counts

`constantRate T W ss frames` takes the advice `Ac × (Fin W → Σ_sync)`, and `frames ins c ℓ` gives every egress frame
(its payload, anchor, logical index and canonical position) from the ingress `ins` and the content advice `c`.
Acceptance is existential: a run is accepted if some advice value explains it (`AcceptedUpTo`). The proof
(`encard_accSeqs_le_advice`) maps each accepted sequence to an advice value that accepts it, plus its jitter code. So
`A_c = log2 #Ac` counts every content-advice value the audit would admit as an explanation, whenever and however it was
chosen. It is charged once per run of W windows, not per window. The number means something only if the set `Ac` and
the function `frames` are fixed before the run, independently of the sender's secret, and the audit refuses advice
outside `Ac`, as it refuses a sync outside Σ_sync.

For a served Program the content advice is the schedule: the per-step batch composition from which the circuit is
built. The anchors and logical indices are read from the ingress and the session. The schedule reaches the egress
frames through their canonical order, which is step-major, through how many tokens each event holds, and through any
batch-variant numerics. Today the schedule is whatever the workload manifest declares: each request's `arrive_step`
and the engine's scheduler settings. So `#Ac` has no bound, and the charge has no number. A monotone map from buckets
to steps that is free at up to s_max steps per bucket is worth `T·log2(s_max + 1)` bits per window. At s_max = 15
that is 2,400 bits per window, 343 times the sync's 7.

### The declaration

- **A deterministic scheduler.** The pinned engine environment includes its scheduler. Given a capacity profile p and
  each request's admission step, the scheduler fixes the per-step schedule: `s = Sched_E(p, arrivals)`. Bit-exact
  replay already relies on this, and `verity_vllm` compares each request's realized arrival step with the declared one.
- **Admission on a step clock.** The profile sets κ, a whole number of engine steps per bucket. Request q is admitted
  at step `κ · anchor(q)`. Its anchor is the bucket in which the ingress certifier released its last frame, read from
  the ingress certificate. Requests released in one bucket are admitted in the ingress release order. The engine never
  starts step κ·b before the releases of bucket b are in. When it runs slower, it lags, and a lag past the declared sync
  is an honest miss, as now. Given p, the schedule is then a function of the ingress certificates, and p is the only
  free choice. `verity_vllm.engine.arrivals` already admits each request right before engine step `arrive_step`, on
  the step counter and never on wall time. The declaration sets `arrive_step = κ · anchor`. The module's rule that an
  idle engine admits the next arrival at once is itself a function of the schedule.
- **A registered list of profiles.** `P_cap = (p_0, …, p_{N−1})`. Each profile holds κ and every scheduler setting the
  schedule reads: the batch caps (sequences and tokens per step), the chunked-prefill threshold, the KV block count,
  which fixes preemption, and the output lag of asynchronous scheduling. The list is a public parameter of the audit,
  registered with Σ_sync, (T, r, B, τ_b, Q), r_in and τ_in before the audit's first window and before any coin. Like
  Σ_sync, it is the deployment's structure.
  - If the developer wants the profiles hidden, the list is a frame-v3-sha512 root over hm96-sha512 rows, one per
    profile, with N public. A sender inside X could then pick the list itself, so a hidden list also costs its own size
    once per registration: at most N·b_p bits for b_p-bit rows. A public list costs nothing.
- **The profile in use.** Its index i < N goes under the schedule root, with the per-step schedule rows. That root is
  a frame-v3-sha512 tree of hm96-sha512 rows (circuits' X3), hidden, and registered before any coin that depends on it.
  The registration statement proves that the rows are `Sched_E(p_i, arrivals)` for the arrivals the ingress
  certificates give, and that i < N. The count holds whenever i was chosen, since acceptance takes any value in the set.
  The rule that the root precedes the coins is there for the sampled proofs, so that the circuit can't be shaped around
  the draw.

**The count.** With one profile per audit, `#Ac = N` and `A_c = log2 N`, once. Switching profiles at window
boundaries, declared at each window's first bucket with the sync, makes the advice one (profile, sync) pair per window:
`log2(N · #Σ_sync)` bits per window. `SecurityProofs.EncardAccSeqsLeConstantRate` with `Ac = Fin N`, or with
`Fin W → Fin N` for switching, states both, so no new guarantee is needed.

**At today's parameters.** The recordings use one scheduler configuration. κ needs a grid that covers the honest step
rate, calibrated as ρ's 8 values are, which is not done yet. Taking 8 values of κ gives N = 8:

| Charge | Per window | Per audit of W = 1,440 windows (a day) |
|---|---|---|
| Egress, one profile per audit | 7 bits, plus 3 bits once | 3 + 1,440 · 7 = 10,083 bits |
| Egress, a profile per window | 7 + 3 = 10 bits | 14,400 bits |
| Egress today, with `A_c` left out | 7 bits | 10,080 bits (`egress_bits_per_day_one_clock`) |

### The coins (D7)

The challenger's coins reach the developer's site on a certified link, so they are ingress. Their timing is charged on
the ingress grid, and their content is exempt.

- **Timing.** The coin link is one more ingress link, at τ_in and an allocation r_coin. It adds a factor
  `(r_coin + 1)^T` per window to the ingress count, which is `T·log2(r_coin + 1)` bits per link-window, the K charge's
  unit (`SecurityProofs.EncardIngressObsDecodableLePerWindow`). r_coin isn't calibrated. At r_coin = 1, one 4 KiB frame
  per bucket, it is 600 bits per link-window (10 bit/s): 864,000 bits a day, 0.0095 of K = 9.12e7. That is on top of
  the `requests` link's 3,077.6 bits per link-window (51.3 bit/s, 0.0486 of K a day), for 0.0581 of K a day in all.
- **Content: 0 bits.** `SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow` charges `log2 #Cs` for the
  content, where `Cs` is one set of content assignments covering every message counted. Conditioned on the
  challenger's coins, which is how the model treats a randomized outside party, the coin link's content is one fixed
  stream `c_coin`: the challenger's coin tree, read in order. So `Cs = Cs_requests × {c_coin}`, and the coins add
  `log2 1 = 0` to the content. Charged as reads, they would cost up to 8·B·r_coin/τ_in = 327,680 bit/s.
- **Why the exemption is sound.**
  - The coins are the challenger's. No outside party can choose their content: every accepted frame on the coin link
    opens against the challenger's committed coin root. This is a declared item of the consistency relation between
    the circuit and the network certificates (`note:network-accounting/20261006T0633Z-draft-circuit-certificate-consistency`), and any other frame on the link is unexplained, so the window is
    rejected.
  - They are fixed before the developer sees them. The coin tree is committed before the audit, so no coin depends on
    anything X sends, and `c_coin` is the same for every message the bound counts.
  - Only the count is free. The frames on the coin link are a prefix of `c_coin`, in order, since a retry never
    redraws. How far the prefix has reached at each bucket is timing, and the grid charges it. Which coins the firewall
    asks for, and when, is egress, which the egress grid charges.
- **What it doesn't cover.** The exemption is for coin openings only. Anything else the challenger sends on the link
  must be fixed by the public parameters, or it is charged as content.

## Calibration (`calibration`)

`network-warden-calibrate TRACE [--ingress-bucket-ms 100]` reads honest timing traces (the format is in the module
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
  declares window w's sync at its first bucket, w·T, from the frames ready by then (the canonical prefix it has read,
  each anchored earlier) and the earlier declarations, and nothing else. A rule (`online.rule`) sees, for each recent
  window, the least δ per ρ that would have met the frames seen of it with the other windows as declared. `cover-K-M`
  takes the value of least estimated hand time whose δ is M buckets past every one of the last K windows' least δ at its
  ρ, `-fallback` declares the envelope (the largest δ and smallest ρ the calibration's prover declared) while one of
  them was late under its own declaration, and `last` is `cover-1-0`. A window with no history declares the envelope.
  A configuration cut from one run at a dropped window may start from the previous one's history
  (`Prover(prior=…)`). The bound charges log2 #Σ_sync bits per window whatever chose the value, so it holds under
  any rule. What changes is who can use those bits: a sync chosen before the run, or from the trace's model, doesn't
  depend on the sender, but an online prover reads the sender's timing, so the sender steers every declaration after
  its first (in the injected-clock replay, the burst, spread and bits senders each move the declared δ, and the wire
  differs from honest's in every window but the first).
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

## The active warden (`active`)

`active.Proxy` runs the wardens on one boundary link of a live byte stream. It is an asyncio TCP proxy, the I/O shell
around `grid.Warden`, `schedule`'s handing rule, `commitment` and `audit`, and it adds no format. Its records are
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
    anchor's window. The egress `Warden` releases min(queue, r) frames per bucket, left-packed, and r − n padding
    frames follow them on the wire.
- **Ingress:** a frame read at time s is enqueued at bucket ⌈s/τ_in⌉ and released by the ingress `Warden` at r_in
  (`Bucketer`, `calibration.size_ingress_ms`'s model).
- **Advice:** the declared syncs, one per window, are given before the run, or an online prover (`Shaper(prover=…)`)
  declares each at its window's first tick, before anything is handed, from the frames the shaper has taken.
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

`uv run tools/check/suites.py verity-network-warden`. The tests enumerate small horizons, so they check the Lean
statements on examples and prove nothing. By name, they mirror 19 of the Lean package's 35 statements
`NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, none of them among its four guarantees (`lean-audit.json`'s
`guarantees`), and three more proved lemmas:
- **`test_network_warden_grid.py`:**
  - Proposition 2's ten statements, by enumeration: `SecurityProofs.CardFree`, `SecurityProofs.CardLeftPacked`,
    `SecurityProofs.FillInjective`, `SecurityProofs.ExistsFillOfLeftPacked`, `SecurityProofs.CardCountsEqLe`,
    `SecurityProofs.CardCountsLeLe`, `SecurityProofs.ReleasedSelf`, `SecurityProofs.SumReleasedLe`,
    `SecurityProofs.EncardFifoGrids` and `SecurityProofs.EncardFifoGridsLeOfAvailable`;
  - the lemmas `occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid` (`NetTiming.SecurityProofs.Occupancy`);
  - the warden's semantics, which Lean takes as hypotheses (W4 packing, W5's untimed status, W6 fail-closed);
- **`test_network_warden_capacity.py`:**
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
- **`test_network_warden_audit.py`:** record-and-reject and each violation, which no Lean statement names; one clock
  per spatial unit, the advice `SecurityProofs.EncardAccSeqsLeSync` counts; and the ingress links' status check, Lean's
  `StatusComplete`;
- **`test_network_warden_commitment.py`:**
  - the elided digest is injective on left-packed grids, by enumeration;
  - it binds the link, window, status and parameters;
  - the committed audit's verdicts equal the raw audit's;
- **`test_network_warden_calibration.py`:** the calibration tool, and the honest prover's choice against its rule by
  brute force, on one link and on several under one clock;
- **`test_network_warden_active.py`:** the active warden over localhost sockets on an injected clock:
  - `Shaper` is `run_shaper`'s rule on random instances;
  - an honest run's records, hand times and wire are the Program's, accepted raw and committed;
  - senders that burst, spread, or pick their write times at random or by bits give the same wire, byte for byte;
  - a frame held past every sync is a mismatch;
  - a late tick, an overrun, a refused stream, a request forwarded a bucket late, a fault that drops a carried queue
    and out-of-set advice each fail closed;

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
`NetTiming`'s definitions, evaluated, give what this package gives. The Lean side is `expectedRow`, with `Accepted` of
`constantRate` and `StatusComplete` decided per prefix of whole windows. The Python side is `expected_grids`,
`check_link`'s verdicts with the first differing bucket, `check_ingress_link`, and each prefix's acceptance.
`lean/scripts/difftest_vectors.py` writes the instances and this package's results, and the Lean audit runs
`lean/NetTimingDifftest.lean` on them (`runs` in `lean/lean-audit.json`). `test_network_warden_lean_difftest.py`
checks that the data the record pins is still this package's.
