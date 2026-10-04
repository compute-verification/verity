# Network transparency: the bucketed timing channel

A spatial unit is the machines behind network wardens. The question is how much a covert sender can move across its
boundary through **when** frames leave, once the content and order of every frame are fixed by the declared Program.
This package is the reference for the answer, in the terms of its Lean package `network-timing` (`protocols/network_warden/lean/`). The tests
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

  `tagged_sha256` is `verity.commitments.identity`'s.
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
`constantRate` whose advice is the content advice and one in-set sync per window (`protocols/network_warden/lean/README.md`, modelling choices
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
channel (`protocols/network_warden/lean/README.md`, modelling choice 6). An ingress link's rows are the Program's input, which it reads rather
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
plus what survived that wipe. `capacity.k_share` expresses bits as a share of `K = 9.12e7` bits.

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

## Calibration (`warden_calibration`, in `tools/catalog/warden`)

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
- **Advice:** the declared syncs, one per window, are given before the run.
- **Faults fail closed** (W6). Each one maps to a status:

  | Fault | Status |
  |---|---|
  | A tick that runs more than `slack` after its boundary. Its bucket is still emitted, as padding, so no bucket slips into the next | `monitor-unavailable` |
  | More than a bucket left unwritten on the link at a tick | `delivery-refused` |
  | A server that refuses a stream | `delivery-refused` (ingress) |
  | The shaper's buffer or the queue past Q | `overflow` |
  | A declared sync outside Σ_sync | `coverage-never-established` |
  | An event without a position, with one already seen, longer than a frame, or on a stream with no released request; bytes left unframed at a stream's end | `malformed` |
- **The slack** bounds how late a tick may run: 25 ms by default, a quarter bucket. A deployment sets it from the jitter
  its timing bounds assume (J).

The clock and the sleep are injected (`VirtualClock` serves a run driven event by event). `Shaper` and `Bucketer` do no
I/O, so the tests drive real sockets without reading the wall clock. `benchmarks/network_traces/active_replay.py`
replays recorded traces through the proxy in real time, and on an injected clock.

## Tests

`uv run tools/check/suites.py verity-network-warden`. The tests enumerate small horizons, so they check the Lean
statements on examples and prove nothing. By name, they mirror 19 of the Lean package's 51 pins, each a guarantee
`NetTiming.SecurityProofs.G : NetTiming.Guarantees.G`, and its three proved but unpinned lemmas:
- **`test_network_warden_grid.py`:**
  - Proposition 2's ten pins, by enumeration: `SecurityProofs.CardFree`, `SecurityProofs.CardLeftPacked`,
    `SecurityProofs.FillInjective`, `SecurityProofs.ExistsFillOfLeftPacked`, `SecurityProofs.CardCountsEqLe`,
    `SecurityProofs.CardCountsLeLe`, `SecurityProofs.ReleasedSelf`, `SecurityProofs.SumReleasedLe`,
    `SecurityProofs.EncardFifoGrids` and `SecurityProofs.EncardFifoGridsLeOfAvailable`;
  - the unpinned `occ_fill`, `leftPacked_occ_fill` and `leftPacked_fifoGrid` (`NetTiming.SecurityProofs.Occupancy`);
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
- **`test_network_warden_calibration.py`** (in `tools/catalog/warden/tests`)**:** the calibration tool, and the honest prover's choice against its rule by
  brute force, on one link and on several under one clock;
- **`test_network_warden_active.py`:** the active warden over localhost sockets on an injected clock:
  - `Shaper` is `run_shaper`'s rule on random instances;
  - an honest run's records, hand times and wire are the Program's, accepted raw and committed;
  - senders that burst, spread, or pick their write times at random or by bits give the same wire, byte for byte;
  - a frame held past every sync is a mismatch;
  - a late tick, an overrun, a refused stream and out-of-set advice each fail closed;

No test mirrors the other eight pins of Theorems 1(b) and 3 (`SecurityProofs.EncardAcceptedRunsLe`,
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
`protocols/network_warden/lean/scripts/difftest_vectors.py` writes the instances and this package's results, and the Lean audit runs
`protocols/network_warden/lean/NetTimingDifftest.lean` on them (`runs` in `protocols/network_warden/lean/lean-audit.json`). `test_network_warden_lean_difftest.py`
checks that the data the record pins is still this package's.
