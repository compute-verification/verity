module

public import Proofs.Flock.Soundness.Soundness
public import Proofs.Flock.Level3

/-! The executable's arithmetic (`execArith`) satisfies `Arith.Correct` (`execArith_correct`), so the soundness theorems
hold for it with no arithmetic hypothesis. -/

@[expose] public section

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

theorem table_sound_exec (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (σ : Game.Strategy (table execArith S sch hm ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch hm ptLocal mPts) σ ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) :=
  table_sound execArith execArith_correct S sch hsch hm ptLocal mPts hlay σ

end FlockSoundness
