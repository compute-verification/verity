module

public import Proofs.Core.Game.Batch
public import Definitions.Flock.Game.Batch
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Game.Basic
import Proofs.Flock.Soundness.Game.Interleave
import Proofs.Flock.Soundness.Game.Lock

/-!
# C-Flock's game library is Core's: `Batch`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Batch`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (value_batch_getD_le)

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (value_withDummy)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (batchLock leftSame lockWithL lockWithSame map mapL mono rightSame same)

end FlockSoundness.Game.Lock.Aligned
