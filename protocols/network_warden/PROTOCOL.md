# Network transparency: the bucketed timing channel

A spatial unit is the machines behind network wardens. The question is how much a covert sender can move across its
boundary through **when** frames leave, once the content and order of every frame are fixed by the declared Program.
This package is the reference for the answer, in the terms of the Lean package `network-timing`. That package lives
outside this repository for now, while the repo's Lean organization is planned. It pins every statement the tests
mirror.

## The warden (`grid`)

Each boundary link has an active warden with fixed parameters (W1). The protocol programs them from a security
parameter, and the warden only executes them:
- T buckets per window;
- the bucket width τ_b, 100 ms by default;
- the allocation r, in frames per bucket;
- the frame size B;
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

## The Program's logical schedule and the clock-sync advice (`schedule`)

The Program fixes each real frame's content, its **anchor** and its **logical index** λ. The anchor is the recorded
bucket of the request it answers. λ is its position within its session, from the canonical gate order.

A window declares one clock-sync value `ClockSync(δ, ρ)`, from the protocol's set `SyncSet`: for example 16 offsets ×
8 rates, which is 7 bits. δ is a whole number of buckets. ρ is a positive fraction of frames per bucket, below 1 for a
session slower than one frame per bucket (a POUS band forward takes about 13 buckets). A frame then moves through three
steps:
- it is **eligible** at `anchor + δ + ⌊λ/ρ⌋`, using the sync of its anchor's window;
- the shaper **hands** it to the warden in canonical order, at `h(k) = max(eligible(k), h(k−1))`;
- the warden's own rule then **releases** it.

`expected_grids` is the Program's grid, and `run_shaper` drives a warden as an honest sender does.

## The audit (`audit`)

A link-window is accepted if:
- its status is complete;
- every slot equals the Program's grid;
- its declared clock sync is in the set;
- its record is present by the verdict deadline.

Anything else is a violation: `status:<reason>`, `advice`, `mismatch` or `no-verdict`. It is **recorded, and the run is
rejected**, as a policy signal. There is no halt: later windows are still checked.

Every capacity bound holds on accepted runs only. A rejected window may already have carried up to its unbound capacity,
about T·r·B bits (`capacity.unbound_bits`).

## Capacity (`capacity`)

| Quantity | Value | Lean |
|---|---|---|
| Free slot choice | `2^(T·r)` occupancies | `card_free` |
| Left-packed | `(r + 1)^T` occupancies, attained by the FIFO warden | `card_leftPacked`, `encard_fifoGrids` |
| Exactly M released | at most `binom(T + M − 1, M)` | `card_counts_eq_le` |
| At most M available | at most `binom(T + M, M)` grids | `encard_fifoGrids_le_of_available` |
| Envelope, with no clock | `T·log2(r + 1)` bits per link-window | Proposition 2 |
| Accepted runs, with clock-sync advice | `A_c + W·log2 #Σ_sync + n·log2 D` bits | `encard_accSeqs_le_sync` |
| A fixed clock, under J (D = 1) | 0 bits of timing | `encard_accSeqs_le_fixedClock` |
| Ingress, where no clock applies | `log2(r_in + 1)/τ_in` bit/s per link | the envelope on the ingress grid |

The inference-only policy charges an IU the ingress covert capacity from the last wipe before it starts, plus what
survived that wipe. `capacity.k_share` expresses bits as a share of `K = 9.12e7` bits.

## Calibration (`calibration`)

`network-warden-calibrate TRACE [--ingress-bucket-ms 100]` reads honest timing traces (the format is in the module
docstring) and sizes:

- **Σ_sync:**
  - δ from 0 up to the p99.99 first-frame latency, in 16 steps;
  - ρ geometrically from the p0.01 production rate up to the p50 rate, in 8 values, each rounded down to a fraction
    with denominator at most 1024 (rounding down keeps a value lenient).
- **The honest prover's choice:** per window, the met value with the least latency. Frames are in the Program's
  canonical order, which is step-major for a batching engine, so a long session interleaves with later windows' frames.
  While a window is chosen, later windows take their most eager value; that only lowers hand times, so a window found
  met stays met.
- **The honest miss rate:** a window that no value meets is a miss, which is a violation. The target is 1e-4.
- **The egress allocation r:** the least r at or above twice the p99.9 of per-bucket demand whose FIFO replay has no
  overflow and a p99 queueing delay of at most one bucket. Both improve monotonically in r, so the search doubles and
  then bisects.
- **The ingress grid:** the smallest r_in that carries honest ingress, at τ_in. τ_in is decision D, 100 ms by default,
  and a parameter. With arrival times (`arrival_ms`), τ_in may be finer than the trace bucket, and the report adds the
  delivery delay: a frame waits for the next ingress bucket boundary, then for its FIFO slot. The p99 FIFO delay is at
  most one ingress bucket by default; `--ingress-max-delay-ms` sets the budget in ms instead, so different τ_in are
  compared at the same latency.

The report has the per-link bandwidth, the utilization and padding share, the clock-sync set and its bits, the (δ, ρ)
range the honest prover declared, the added first-frame latency, the shaping delay (hand time minus ready time, over
every frame), the capacity per day, and the K charge by span.

## Tests

`uv run pytest protocols/network_warden`. The tests mirror the Lean package's pinned statements:
- **`test_network_warden_grid.py`:** the counts, `fill`, tightness and the fixed-M bounds, by enumeration, plus the warden's
  semantics;
- **`test_network_warden_capacity.py`:**
  - the tree lemma;
  - accepted grids at most `#Σ_sync^W`, over every sender and declaration on a small horizon;
  - the fixed clock's single accepted grid;
  - a reacting operator;
  - the block bound on random Programs;
- **`test_network_warden_audit.py`:** record-and-reject;
- **`test_network_warden_calibration.py`:** the calibration tool.
