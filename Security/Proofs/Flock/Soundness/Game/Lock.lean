import Proofs.Core.Game.Lock
import Definitions.Flock.Game.Lock
import Proofs.Flock.Soundness.Game.Expect
import Proofs.Flock.Soundness.Game.Interleave

/-!
# C-Flock's game library is Core's: `Lock`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Lock`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (avg_sqrt_le' expect_le_coupled expect_withDummy forkE_congr forkE_nonneg leafE_add leafE_const
  leafE_fixed leafE_mono leafE_mono_rel leafE_off_le leafE_sum)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (bind embedLeft embedRight interleaveLock lockWith refl)

end FlockSoundness.Game.Lock.Aligned
