module

import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Positivity.Finset
public import Definitions.Core.Rewinding

/-!
# Flock's game vocabulary is Core's: `Rewinding`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Rewinding`, so Flock's spec and every
other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Rewinding

export Verity.Rewinding (bad conflict ConflictAt IsPlurality Off opened openedTo)

end FlockSoundness.Rewinding
