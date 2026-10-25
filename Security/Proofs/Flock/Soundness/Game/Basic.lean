import Proofs.Core.Game.Basic

/-!
# C-Flock's game library is Core's: `Basic`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Basic`, so C-Flock's soundness
proofs and every other protocol's read one game library.
-/

namespace FlockSoundness

export Verity (Game)

end FlockSoundness

namespace FlockSoundness.Game

export Verity.Game (Strategy avg avg_add avg_const avg_mono avg_mul_const bind bind_assoc bind_coin bind_eq bind_ret
  bind_ret_right bind_send casesOn coin draw instMonad map prCoin prCoin_eq_card prCoin_exists_le prCoin_mono prob
  prob_le_value pure_eq rec recOn recv ret send value value_bind_le value_coin value_coin_le value_coin_ret
  value_le_one value_mono value_ret value_round_le value_send)

end FlockSoundness.Game

namespace FlockSoundness.Game.ret

export Verity.Game.ret (inj injEq)

end FlockSoundness.Game.ret

namespace FlockSoundness.Game.send

export Verity.Game.send (inj injEq)

end FlockSoundness.Game.send

namespace FlockSoundness.Game.coin

export Verity.Game.coin (inj injEq)

end FlockSoundness.Game.coin
