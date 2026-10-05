import FlockSoundness.Protocol.LigeritoLevels

/-!
# A level's folds against a list (PROTOCOL.md §13.2, §13.3 step 4)

`foldRounds` folds the lowest variable first, one `K` challenge per round, and the running claim
is a sum `Σ_y u(y)·v(y)` of the level's (folded) table `u` against the claim's weight `v`. For a
list of tables (the codewords near the level's commitment), a claim that is false for every one
of them stays false for every one after `k` rounds, except with probability `k·2L/|K|`
(`value_foldRounds_sum_le`): per round, each list element's true quadratic differs from the
verifier's, which sums to the claim by construction (§13.2). Together with the rounds' MCA events
(`value_foldRounds_mca_le`) this is the accounting's `Level.foldError`.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- Fold the lowest variable by each challenge in turn (`foldRounds`' order). -/
def foldLoL {L : Type} [Field L] : List L → (ℕ → L) → (ℕ → L)
  | [], u => u
  | r :: rs, u => foldLoL rs (foldLo r u)

end FlockSoundness
