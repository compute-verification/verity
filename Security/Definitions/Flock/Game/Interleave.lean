import Definitions.Flock.Game

/-!
# C-Flock's game vocabulary is Core's: `Interleave`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game`, so C-Flock's spec and every
other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (interleave interleaveWith)

end FlockSoundness.Game
