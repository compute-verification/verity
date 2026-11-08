module

public import Definitions.Flock.Game.Expect
public import Proofs.Core.Game.Runs
public import Proofs.Flock.Soundness.Game.Expect
import Mathlib.Algebra.CharP.Defs
import Mathlib.Tactic.Positivity.Finset

/-!
# Flock's game library is Core's: `Runs`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Runs`, so Flock's soundness proofs
and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (expectN expectN_add expectN_const expectN_const_mul expectN_mono expectN_nonneg expectN_prod_map
  expectN_sum expectN_sum_map)

end FlockSoundness.Game
