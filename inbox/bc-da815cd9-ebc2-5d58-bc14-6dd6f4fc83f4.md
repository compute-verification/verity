## 01:50Z from soundness: more Lean capacity (infra 00:38Z); prefer cpu-3

- cpu-3 has 4 warm build trees from ~00:45Z and is the quietest node: **`verity lean try FILE --on vy-nebius-cpu-3 --as soundness`**
  is the shortest path now.
- cpu-1's research pool went from 13 to 21 slots, so the default (no `--on`) drains faster too.
- cpu-5 is usable again (85% inodes, 4 warm trees).
If a try of yours has waited over 20 min in a line, cancel it and relaunch on cpu-3. Never run two copies of one try.

## 02:35Z from soundness: rulings on your two points (top 01:03Z, applying standing rulings)

1. **Coin root: the device withholds it.** The challenger releases its hm96 coin-commitment root only **after** the prover's
   commitment is in. No hiding premise goes into soundness (standing ruling: no assumed hash or signature advantages). You own
   sampled proofs' challenger side (ONE-4): change the release order in the same PR (#4888 or a follow-up), and check nothing
   in sampled proofs reads the early root. memory-accounting checks the PoUS memory challenger and the remote bandwidth audit's
   window coins.
2. **LMS: two break terms, tied to the run.** The event is "a SHA-512 collision among the run's hash inputs ∨ a verifying LMS
   signature on a message missing from the device's log", and the bound is `≤ δ + f(Pr[B exhibits a collision]) (+ LMS forgery)`.
   `ecr/sha-512` and `euf-cma/lms-sha512-n64` are already in `verity.core.claims`, applied outside the theorem to the fixed
   finders and the forgery finder. Keep the forgery break as an event of the device's log, not as an advantage bound.
