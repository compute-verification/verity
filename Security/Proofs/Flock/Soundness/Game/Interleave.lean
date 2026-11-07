module

public import Proofs.Core.Game.Interleave
public import Definitions.Flock.Game.Interleave
public import Proofs.Flock.Soundness.Game.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Inv

/-!
# C-Flock's game library is Core's: `Interleave`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Interleave`, so C-Flock's soundness
proofs and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (value_and_const value_bind_ret value_congr value_const_and value_false value_interleave_le
  value_interleaveWith_le value_map)

end FlockSoundness.Game
