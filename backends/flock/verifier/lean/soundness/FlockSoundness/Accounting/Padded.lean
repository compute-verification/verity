import FlockSoundness.Accounting.Bound
import FlockSoundness.Protocol.Accounting.Padded

/-!
# The bound with a padded level 0 (M1)

M1 (lane `flock-zk`, `backends/flock/live/PROTOCOL.md` §7) hides level 0 two ways, and both change
level 0's soundness terms:

* **Per-lane padding.** Every lane's codeword encodes its message and `t` uniform coefficients, so
  the level-0 code is `RS[2^d, 2^cols + t]`, of rate `ρ' = (2^cols + t)/2^d`. Its radius, list
  size, queries and mutual-correlated-agreement numerator are the base level's at `ρ'`
  (`Level.padRadius`, `padListBound`, `padQueryError`, `padMcaA`).
* **Extra lanes.** Each rep folds `e` extra uniform lanes into its level-0 fold with fresh coins
  `ρ₁ … ρ_e`, after the lane rounds. Each coin is one more affine line for mutual correlated
  agreement (`a/2^256`) and one more coin on the restarted claim (`L/2^256`) (`padFoldError`).

Everything else in the rep is unchanged, with the padded list bound for level 0 (`padRepError`).
The bound is a function of `t` and `e`, so a change to either is a parameter, not a proof.
-/

namespace FlockSoundness.Accounting

namespace Level

end Level

/-- **One table with the padded level 0**: the link points, then two reps on independent coins. -/
noncomputable def padTableError (s : Schedule) (st : Stmt) (t e : ℕ) : ℝ :=
  padLinkError s st t + padRepError s st t e ^ 2

/-- A schedule with level 0's query count replaced (M1 sizes it for the padded rate). -/
def Schedule.withQueries₀ (s : Schedule) (q₀ : ℕ) : Schedule :=
  { s with levels := s.levels.modifyHead fun l => { l with queries := q₀ } }

end FlockSoundness.Accounting
