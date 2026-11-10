module

public import Definitions.Core.Game.Batch
public import Definitions.Flock.Game.Lock

/-!
# Flock's game vocabulary is Core's: `Batch`

Every name here is an alias (`export`) of Core's definition in `Definitions.Core.Game.Batch`, so Flock's spec and
every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (batch)

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (batchLock leftSame lockWithL lockWithSame mapL rightSame sameThen)

end FlockSoundness.Game.Lock
