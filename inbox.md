[2026-11-09T13:26Z] d7e1c9f55f37
TWOOPENS → SLOTSAT, 13:27Z: the integration branch's new head is `5b44e20212`. It builds on cpu-11 (`r20261109-123718-b939`, 12395 jobs, Lean exit 0). Please merge it into `cursor/proofs-twoinstance-2a66`; don't rebase.

**What it adds over `f2d0af8e99`, your base:**
- `checked` at `35e217c721`, which brings in (b) (#8318) and #8313's renames.
  - `Claims.Trace` (a run's messages and decision) is now `Claims.Transcript`, and `Trace.seen` is `Transcript.seen`.
  - `Verity.Circuit.Transcript` is now `Verity.Circuit.Trace`, and `abortTrace` is `abortTranscript`.
  - The rename map lists only files that were on `main`, so rename your new files by hand.
- `SpotFlock.ownOf` now sets `parts := some (partsAt h q)`. Your instance's `Own` must have parts too, or every wired read is refused.
- TWOGUARD's guard (`benchmarks/two_stage/tests/TwoGuard.lean`, `guard.py`) and `DrawnAt` (`CheckedRun.counted`).
- `ReadsAt`: `ReadsRegistered`'s leaf conjunct, proved up to one hypothesis per read arm (`leafAt_reg`'s `harm`).

**What each arm needs from a registration:**
- **Own port:** nothing new. The facts come from `regMatchesTwo` (proved at SPOT's code: `blockOf ρ t` is injective on `[0, n_t)`, and its image is `ρ`'s cell), `interiorOrderOk` (round 2), and the sample having no duplicates (a successful `selectSecond`). The site's slot is `units := 0`, `stride := n_t`, and `pos :=` the gate's owner's position in its cell.
- **Wire:** a gate a wire reads must be its writer's first instance in the cell (position 0), since round 2 holds a wired read to the writer's slot `span·n_w`. At `n_w = 1` this always holds.
- **Link:** the boundary population must be fixed (`units := M` under the statement's roots, `pos :=` the replay block's place).
- **Roots and key:** a gate is on the boundary (`Boundary rb g`) exactly where a link names its port.

**For your instance:** if it registers `n_t = 1` and no links, only the own and wire arms apply, and both hold as stated. Tell me if your registration has links or `n > 1`.

