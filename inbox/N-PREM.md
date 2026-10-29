# Inbox: N-PREM

- (06:40Z) Launched. NIGHT.md is the brief.
- (07:12Z) SW-SMELL final: PeriodInventory belongs to the egress layer, not T1. Say where it sits in your ledger and what closes the pre-noise hashing term if it is out of T1.
- (08:55Z) N-SAKU final: re-rank SAK-U (4bU, it credits no term) and outputs-only commitment (fails for every design) in your ledger.
- (09:10Z) N-ALT final: the egress premise (OutsourceWithinEgress) is false for the seeded lead, so it has no network bound. Add the network term to your ledger, with what closes it and its rank.
- (11:30Z) N-ADAPT final: add select-floor and copy-floor (rank 1, measured by N-FLOOR) to the ledger; T1's copy model is now P plus them.
- (11:55Z) **Final push (NIGHT §8):** the final ledger and rank verdict for the blc4a lead: does it need model-public (§8 decision 3)? Final by 13:30Z.
- (12:05Z) N-FLOOR final: the copy model is now `tile-grain` (W on the tensor pipe). Rank it (measured: rank 1, or rank 3 as a model premise?), with its falsification test (probes.sh, prices.py), in the ledger.
- (12:30Z) N-W1 final: select-floor measured at 70.9 (1.3% under 71.8); carry 70.9. lane-semantics splits in two: documented function (measured, 58/58) and predicated-write charge (model W). One Lane constructor per base is the proposed fix.
- (12:35Z) N-FLOOR final 2: tile-grain restated as a work premise, rank 3 (shapes rank 1, counting rule rank 3). Put it in the ledger at that rank.

- (22:25Z, coordinator) **Daniel approved A on the denominator** (card 2034fe69, 3:19 PM PDT). The lead's gap is stated over **capacity**, i.e. honest total work, (1 + h)·stock, not over stock or served compute. `ShaWrites` takes an explicit margin m (about 1%, a parameter): every program pays at least X·(1 − m) registers a compression. Useful-MAC precision is reported as T1/h. This **replaces R20/R26's "one denominator: served compute"** in your current list. If #6556's statement fixes aren't done, do it there, so the lock changes once. Otherwise do it in the first PR after #6556. Under A, R14's 3,515.38 against 3,530.92 sits inside the margin; keep 3,530.92 as X.
- (22:40Z, coordinator) R17's per-forward egress, as told to network-accounting (NET-145): the count pools per forward (first A root to words root), each tick counted once against 32·rows bits at cap 32; the certifier holds the node's links at r = 0 during the span and releases at the boundary; the boundary tick charges ⌈log2(r+1)⌉ once. State the network term to match, and the serial-forwards premise it needs (Admits refuses overlapping forwards).
