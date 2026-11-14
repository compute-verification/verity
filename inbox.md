[2026-11-14T16:37Z] 71871d2b03c1
## Coordinator, Sat 14 Nov 16:37Z: PERIODS: #9595 held by ci

From the proofs lead, Sat 14 Nov. ci's reader HOLDS #9595 at `c2defac3ca95`: merged with `main` its Lean does not build. `auditedOpen_eq_auditedIn` (`Spot/ChallengerSession.lean:786`) is no longer `rfl`, because `main` rewrote `auditedOpen`'s period check to `p.cap.isNull || p.circuit.isNull` and the PR's `auditedIn` still says `== Json.null`. Merge `main` into #9595's branch, write `auditedIn`'s line as `main`'s, build on a node (Lean fast path + quick tier at the merged head), push, and mark again under rule 9 (or a ready note if a lock line's text changes). Do this before anything else left in your turn. The rename itself read clean.

