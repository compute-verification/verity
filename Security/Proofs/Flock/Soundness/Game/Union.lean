module

public import Definitions.Flock.Game
public import Proofs.Core.Game.Union
public import Proofs.Flock.Soundness.Game.Interleave
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real
import Proofs.Flock.Soundness.Game.Basic

/-!
# C-Flock's game library is Core's: `Union`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Union`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (instLawfulMonad value_bind_le_of_forall value_exists_finset_le value_mapM_exists_le
  value_mapM_map_ne value_or_le)

end FlockSoundness.Game
