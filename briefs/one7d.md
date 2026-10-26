# Brief: ONE7D, C-Flock's circuit definitions into Definitions/Flock (ONE-7d), Linear PRF-203

**Due:** marked by **Tue 27 Oct 06:00Z**. Dates are ceilings.

The F3 theorem (`Verity.Soundness`, decision 1, Daniel approved 11:22 AM PDT) reads the registered circuit `circuitOf D hU n := Verity.Circuit.ofAudit ((keyProg D.unroll hU n).circuit (outsOf D.unroll hU n))` (ONE78's note `~/.research/notes/lanes/proofs/20261026T1707Z-report-one78.md` §1; ONE78 bc-44c5e267 writes `circuitOf` and its lemmas). The trees rule (`trees` in `Security/lean-audit.json`) refuses a guarantee whose statement reads a proofs-tree definition, so the definition chain must live in the definitions tree.

**Deliverable:** move `keyProg` (`Security/Proofs/Flock/Soundness/Discharge/Placed/Keyed.lean`), `Built` (`Security/Proofs/Flock/Soundness/ExecRows.lean`), `outsOf`, and every definition they read that is still in `Security/Proofs`, **definitions only**, into `Security/Definitions/Flock/` (list them in `Definitions/Flock.lean`'s index as that file does). Proofs stay where they are and import the definitions. No statement changes; lock records only move: record with `check.py --update --moved MOVES.json` (lean-proofs skill) so no DM fires for a pure move. Drop any `trees.except` entry the move makes unnecessary. One small PR, nothing else in it. If the chain is much bigger than expected (say more than ~15 definitions or ~6 modules), post the list on PRF-203 and stop for a ruling.

**Paths:** proofs OK'd the move (17:40Z); `Definitions/` is lean's, which soundness asked; check your inbox for lean's answer before marking. **Base:** create your branch from `origin/main`. **Topic:** `one7d`.
