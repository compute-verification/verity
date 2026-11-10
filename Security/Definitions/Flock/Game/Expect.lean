module

public import Definitions.Core.Game.Expect
public import Definitions.Flock.Game
public import Definitions.Flock.Rewinding

/-!
# Flock's game vocabulary is Core's: `Expect`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Expect`, so Flock's spec and
every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (expect expect₂ ind)

end FlockSoundness.Game

namespace FlockSoundness.Rewinding

export Verity.Rewinding (IsPluralityE)

end FlockSoundness.Rewinding
