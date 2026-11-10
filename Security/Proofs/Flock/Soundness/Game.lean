module

public import Definitions.Flock.Game
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Data.Fintype.Pi
public import Proofs.Core.Game.Basic
public import Proofs.Core.Game.Batch
public import Proofs.Core.Game.Expect
public import Proofs.Core.Game.Interleave
public import Proofs.Core.Game.Lock
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Runs
public import Proofs.Core.Game.Seq
public import Proofs.Core.Game.Union
public import Proofs.Flock.Soundness.Rewinding

/-!
# Flock's game library is Core's

Every name here is an alias (`export`) of Core's lemma in `Proofs.Core.Game`'s `Basic`, `Batch`, `Expect`,
`Interleave`, `Lock`, `Prob`, `Runs`, `Seq` and `Union`, so Flock's soundness proofs and every other protocol's read
one game library. Core's `Poly` (a polynomial's root at a coin) is imported and named as core's (`Verity.Game`) by the
phases that need it, so this library carries no polynomials. What Flock proves of games itself is in
`Proofs.Flock.Soundness.Game`'s `Tape` (the bind tape and `seqG`), `LockChoose` (aligned tables and the fork's choice),
and `LockLin` and `LockOff` (the link finder).
-/

@[expose] public section

namespace FlockSoundness.Game

export Verity.Game (avg_add avg_const avg_ite_eq_prCoin avg_mono avg_mul_const avg_pi_prod bind_assoc bind_coin bind_eq
  bind_ret bind_ret_right bind_send expect_add expect_bind expect_const expect_const_mul expect_ind_le_one expect_map
  expect_mono expect_nonneg expect_runsG expect_sub expect_sum expectN expectN_add expectN_const expectN_const_mul
  expectN_mono expectN_nonneg expectN_prod_map expectN_sum expectN_sum_map ind_le_one ind_mono ind_nonneg
  instLawfulMonad prCoin_eq_card prCoin_exists_le prCoin_false prCoin_mono prCoin_pi_forall prob_and_const prob_bind_le
  prob_bind_of_const prob_coin prob_false prob_le_one prob_le_value prob_map prob_mono prob_or_le prob_send prob_toReal
  prob_true pure_eq value_and_const value_batch_getD_le value_bind_le value_bind_le_of_forall value_bind_ret value_coin
  value_coin_le value_coin_ret value_congr value_const_and value_exists_finset_le value_false value_interleave_le
  value_interleaveWith_le value_le_one value_map value_mapM_exists_le value_mapM_map_ne value_mono value_or_le value_ret
  value_round_le value_send)

end FlockSoundness.Game

namespace FlockSoundness.Game.Lock

export Verity.Game.Lock (avg_sqrt_le' expect_le_coupled expect_withDummy forkE_congr forkE_nonneg leafE_add leafE_const
  leafE_fixed leafE_mono leafE_mono_rel leafE_off_le leafE_sum value_withDummy)

end FlockSoundness.Game.Lock

namespace FlockSoundness.Game.Lock.Aligned

export Verity.Game.Lock.Aligned (batchLock bind embedLeft embedRight interleaveLock leftSame lockWith lockWithL
  lockWithSame map mapL mono refl rightSame same)

end FlockSoundness.Game.Lock.Aligned

namespace FlockSoundness.Rewinding

export Verity.Rewinding (exists_pluralityE expect_bad_le ind_isSome_eq_sum)

end FlockSoundness.Rewinding
