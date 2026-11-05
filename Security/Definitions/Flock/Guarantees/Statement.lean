import Definitions.Flock.Statement

/-!
# C-Flock's statement model: its guarantees (trusted)

What every witness satisfying a statement's block R1CS (`Definitions.Flock.Statement`) holds, over the model alone.
The proof is `Flock.SecurityProofs.ZeroRowForced : Flock.Guarantees.ZeroRowForced`.
-/

namespace Flock.Guarantees

open FlockSoundness.Model

/-- **A zero row of `A₀` forces its position to 0**: in every witness satisfying a statement, a position `c` whose row
of `A₀` is zero (`Statement.ZeroRowA₀`) holds 0 in every block, since that block's constraint at `c` reads
`(A₀ z)_c · (B₀ z)_c = z_c` and its left side is 0. A circuit's input column whose rows are empty is such a position: it
holds a constant, not a free bit. -/
def ZeroRowForced : Prop :=
  ∀ (S : Statement) (z : Witness S.m), S.Satisfies z → ∀ c, S.ZeroRowA₀ c → ∀ o, S.block z o c = 0

end Flock.Guarantees
