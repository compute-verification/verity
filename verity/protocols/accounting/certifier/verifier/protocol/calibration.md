Part of the certifier's [PROTOCOL.md](../PROTOCOL.md).

## The prover's clock syncs (`calibration.choose_syncs`, `online`)

Σ_sync, the (δ, ρ) values a window may declare, is fixed by the protocol (`schedule.SyncSet`). Given it:

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
  least δ at its ρ, `-fallback` declares the envelope (the largest δ and smallest ρ the honest prover declared)
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
  (1 + largest anchor // T each).
  One clock can miss where a choice per link would not: the value met on every link can hand one link's frames earlier
  than that link's own choice, and a later window whose frames were met only behind them then misses.
