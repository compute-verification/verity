import Proofs.Core.Game.Poly
import Proofs.Flock.Soundness.Game.Basic

/-!
# C-Flock's game library is Core's: `Poly`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Poly`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (prCoin_eval_eq_eval_le prCoin_eval_eq_zero_le prCoin_mvEval_eq_zero_le)

end FlockSoundness.Game
