import Definitions.Core.Game.Prob
import Definitions.Flock.Game.Interleave

/-!
# C-Flock's game vocabulary is Core's: `Prob`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Prob`, so C-Flock's spec and every
other protocol's read one game library.
-/

namespace FlockSoundness.Game.Strategy

export Verity.Game.Strategy (ofBind ofMap)

end FlockSoundness.Game.Strategy
