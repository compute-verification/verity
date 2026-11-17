module

public import Proofs.Core.Game.Rewinding
public import Definitions.Flock.Rewinding
public import Mathlib.Analysis.Real.Sqrt
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Combinatorics.Enumerative.DoubleCounting

/-! Aliases of Core's `Proofs.Core.Game.Rewinding`, so Flock's soundness proofs and every other protocol's read one game
library. -/

@[expose] public section

namespace FlockSoundness.Rewinding

export Verity.Rewinding (bad_sq_le exists_plurality off_eq off_le_row off_sq_le openedTo_subset prBad_le
  sum_conflictAt_le)

end FlockSoundness.Rewinding
