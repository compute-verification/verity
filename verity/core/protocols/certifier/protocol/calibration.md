Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

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
- **The ingress grid:** the smallest r_in that carries honest ingress, at τ_in. τ_in is **100 ms**, set
  from a calibration of honest traces. The grid stays configurable, so `--ingress-bucket-ms` still compares
  alternatives. With arrival times (`arrival_ms`), τ_in may be finer than the trace bucket, and the report adds the
  delivery delay: a frame waits for the next ingress bucket boundary, then for its FIFO slot. The p99 FIFO delay is at
  most one ingress bucket by default; `--ingress-max-delay-ms` sets the budget in ms instead, so different τ_in are
  compared at the same latency.

The report has the per-link bandwidth, the utilization and padding share, the clock-sync set and its bits, the (δ, ρ)
range the honest prover declared, the first-frame latency it declared (the p50 and p99 of its declared δ, in ms), the
shaping delay (hand time minus ready time, over every frame), the capacity per day, and the K charge by span.
