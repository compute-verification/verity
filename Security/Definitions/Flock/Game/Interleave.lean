module

public import Definitions.Flock.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# Flock's game vocabulary is Core's: `Interleave`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game`, so Flock's spec and every
other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (interleave interleaveWith)

end FlockSoundness.Game
