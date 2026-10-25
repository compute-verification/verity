import Proofs.Core.Game.Lock
import Proofs.Flock.Soundness.Game.Expect
import Proofs.Flock.Soundness.Game.Interleave

/-!
# C-Flock's game library is Core's: `Lock`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Lock`, so C-Flock's soundness
proofs and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (Lock)

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (Aligned Chooser Disjoint RPair avg_sqrt_le' bind casesOn coin dropDummy embedLeft
  embedRight expect_le_coupled expect_withDummy forkE forkE_congr forkE_nonneg getPair getPair_fresh₁ getPair_fresh₂
  getPair_keep₁ getPair_keep₂ getPair_set₁ getPair_set₂ interleaveLock leafE leafE_add leafE_const leafE_fixed
  leafE_mono leafE_mono_rel leafE_off_le leafE_sum lockWith map mono oracleStrat rec recOn ret same send table
  withDummy)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (bind casesOn coin embedLeft embedRight interleaveLock lockWith rec recOn refl ret
  send table)

end FlockSoundness.Game.Lock.Aligned

namespace FlockSoundness.Game.Lock.Disjoint

export Verity.Game.Lock.Disjoint (casesOn left mk rec recOn right)

end FlockSoundness.Game.Lock.Disjoint
