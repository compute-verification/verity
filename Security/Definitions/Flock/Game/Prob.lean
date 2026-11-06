module

public import Definitions.Flock.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real
public import Definitions.Core.Game.Prob

/-!
# C-Flock's game vocabulary is Core's: `Prob`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Prob`, so C-Flock's spec and every
other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game.Strategy

export Verity.Game.Strategy (ofBind ofMap)

end FlockSoundness.Game.Strategy
