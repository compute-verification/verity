import Definitions.Core.Game.Seq
import Definitions.Flock.Game.Expect

/-!
# C-Flock's game vocabulary is Core's: `Seq`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Seq`, so C-Flock's spec and every
other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (drawStrat runsG runsStrat)

end FlockSoundness.Game

namespace FlockSoundness.Game.Strategy

export Verity.Game.Strategy (bind map)

end FlockSoundness.Game.Strategy
