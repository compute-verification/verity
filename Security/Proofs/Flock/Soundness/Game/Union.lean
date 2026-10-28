import Proofs.Core.Game.Union
import Proofs.Flock.Soundness.Game.Interleave

/-!
# C-Flock's game library is Core's: `Union`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Union`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (instLawfulMonad value_bind_le_of_forall value_exists_finset_le value_mapM_exists_le
  value_mapM_map_ne value_or_le)

end FlockSoundness.Game
