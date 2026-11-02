import Proofs.Flock.Soundness.Soundness
import Proofs.Flock.Level3

/-!
# The executable's arithmetic satisfies `Arith.Correct`

`execArith` is the arithmetic the executable verifier computes with (lane `flock-verifier`,
package `level3`): `F = GF(2^128)` in GHASH form, `K = GF(2^256)`, the Lagrange weights of the
univariate-skip nodes, the pinned zerocheck coordinates, the packing basis `x^v`, the embedding
`F ⊂ K` and `u`, the Reed–Solomon domain `ω_q` and novel basis `Ŵ_i`, and `lo`. Every field of
`Arith.Correct` is a theorem of `FlockLevel3` about those functions (`execArith_correct`), so
the soundness theorems hold for the executable's arithmetic with no arithmetic hypothesis
(`table_sound_exec`).
-/

namespace FlockSoundness

open Model Accounting

/-- **The executable's arithmetic.** -/
noncomputable def execArith : Arith FlockLevel3.GF128 FlockLevel3.GF256 where
  lagS := FlockLevel3.lagS
  lagΛ := FlockLevel3.lagΛ
  combW := FlockLevel3.combW
  pinned := FlockLevel3.pinnedF
  pack := FlockLevel3.pack
  unpack := FlockLevel3.unpack
  embed := FlockLevel3.embed
  u := FlockLevel3.u
  omega := FlockLevel3.omega
  What := FlockLevel3.What
  lo := FlockLevel3.lo

theorem eqAt_pinned (b : ℕ) : eqAt execArith.pinned b = FlockLevel3.eqAtPinned b := by
  unfold FlockLevel3.eqAtPinned eqAt
  rfl

/-- **`Arith.Correct` holds for the executable's arithmetic.** -/
theorem execArith_correct : execArith.Correct where
  card_F := FlockLevel3.card_F
  card_K := FlockLevel3.card_K
  charTwo := FlockLevel3.charTwo
  charTwoK := FlockLevel3.charTwoK
  unpack_pack := FlockLevel3.unpack_pack
  pack_unpack := FlockLevel3.pack_unpack
  unpack_add := FlockLevel3.unpack_add
  nodes := FlockLevel3.nodes
  pinned_indep c h := FlockLevel3.pinned_indep c (by simpa only [eqAt_pinned] using h)
  pinned_ne_one := FlockLevel3.pinned_ne_one
  omega_injective := FlockLevel3.omega_injective
  xhat_poly := FlockLevel3.xhat_poly
  ofLimbs_bijective := FlockLevel3.ofLimbs_bijective
  lo_balanced := FlockLevel3.lo_balanced

/-- **One table's session is sound, for the executable's arithmetic.** `table_sound` with
`execArith_correct`, so it has no named assumption. -/
theorem table_sound_exec (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (σ : Game.Strategy (table execArith S sch hm ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch hm ptLocal mPts) σ ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) :=
  table_sound execArith execArith_correct S sch hsch hm ptLocal mPts hlay σ

/-- `table_sound_exec` for `m ∈ [22, 33]` under the `fast100` schedule: at most `2^-205`. -/
theorem table_sound_exec_fast100 (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm1 : 22 ≤ S.m) (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ)
    (hshape : (S.shape mPts).InRange) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Game.Strategy (table execArith S sch (by omega) ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch (by omega) ptLocal mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  table_sound_fast100 execArith execArith_correct S sch hsch hm1 hm2 ptLocal mPts hshape hlay σ

/-- `table_sound_exec` for `m ∈ {34, 35}`: at most `2^-205`. -/
theorem table_sound_exec_fast100_34_35 (S : Statement)
    (sch : Schedule) (hsch : fast100 S.m = some sch) (hm1 : 34 ≤ S.m) (hm2 : S.m ≤ 35)
    (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Game.Strategy (table execArith S sch (by omega) ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch (by omega) ptLocal mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  table_sound_fast100_34_35 execArith execArith_correct S sch hsch hm1 hm2 ptLocal mPts hshape
    hlay σ

end FlockSoundness
