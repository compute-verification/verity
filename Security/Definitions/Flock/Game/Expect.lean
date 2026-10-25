import Definitions.Core.Game.Expect
import Definitions.Flock.Game
import Definitions.Flock.Rewinding

/-!
# C-Flock's game vocabulary is Core's: `Expect`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Expect`, so C-Flock's spec and
every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (expect expect₂ ind)

end FlockSoundness.Game

namespace FlockSoundness.Rewinding

export Verity.Rewinding (IsPluralityE)

end FlockSoundness.Rewinding
