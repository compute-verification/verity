import Definitions.Core.Rewinding

/-!
# C-Flock's game vocabulary is Core's: `Rewinding`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Rewinding`, so C-Flock's spec and every
other protocol's read one game library.
-/

namespace FlockSoundness.Rewinding

export Verity.Rewinding (bad conflict ConflictAt IsPlurality Off opened openedTo)

end FlockSoundness.Rewinding
