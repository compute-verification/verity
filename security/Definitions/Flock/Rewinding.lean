module

import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Positivity.Finset
public import Definitions.Core.Rewinding

/-! Flock's `Rewinding` vocabulary, aliases of Core's `Definitions.Core.Rewinding`. -/

@[expose] public section

namespace FlockSoundness.Rewinding

export Verity.Rewinding (bad conflict ConflictAt IsPlurality Off opened openedTo)

end FlockSoundness.Rewinding
