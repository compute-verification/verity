import Proofs.Core.Game.Interleave
import Proofs.Flock.Soundness.Game.Basic

/-!
# C-Flock's game library is Core's: `Interleave`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Interleave`, so C-Flock's soundness
proofs and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (interleave interleaveWith value_and_const value_bind_ret value_congr value_const_and value_false
  value_interleaveWith_le value_interleave_le value_map)

end FlockSoundness.Game
