import Proofs.Core.Game.Expect
import Definitions.Flock.Game.Expect
import Proofs.Flock.Soundness.Game.Basic
import Proofs.Flock.Soundness.Rewinding

/-!
# C-Flock's game library is Core's: `Expect`

Every name here is an alias (`export`) of Core's declaration in `Proofs.Core.Game.Expect`, so C-Flock's soundness proofs
and every other protocol's read one game library.
-/

namespace FlockSoundness.Game

export Verity.Game (expect_add expect_const expect_const_mul expect_ind_le_one expect_mono expect_nonneg expect_sub
  expect_sum ind_le_one ind_mono ind_nonneg prob_toReal)

end FlockSoundness.Game

namespace FlockSoundness.Rewinding

export Verity.Rewinding (exists_pluralityE expect_bad_le ind_isSome_eq_sum)

end FlockSoundness.Rewinding
