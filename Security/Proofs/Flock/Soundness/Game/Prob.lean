module

public import Definitions.Core.Game
public import Proofs.Core.Game.Prob
public import Definitions.Flock.Game.Prob
public import Mathlib.Data.Fintype.Pi
public import Proofs.Flock.Soundness.Game.Union
import Mathlib.Basic.ENNReal.BigOperators
import Proofs.Flock.Soundness.Game.Basic

/-!
# C-Flock's game library is Core's: `Prob`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Prob`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (avg_ite_eq_prCoin avg_pi_prod prCoin_false prCoin_pi_forall prob_and_const prob_bind_le
  prob_bind_of_const prob_coin prob_false prob_le_one prob_map prob_mono prob_or_le prob_send prob_true)

end FlockSoundness.Game
