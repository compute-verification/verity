import Definitions.Claims.Recursion
import Proofs.Flock.Claims.RedTeam.ZK
import Proofs.Flock.Claims.RedTeam.Complete

/-!
# Red team: the recursion draft's zero knowledge and completeness

* `half_le_of_hm96Hiding_const`: at the constant mask `fun _ : Unit => true` (and the salt hash `id`), hm96's hiding holds
  only at `δ₁ ≥ 1/2`: the mask is `true` surely, a uniform bit half the time.
* `recursionZK_const`: the recursion's `ZK` at that mask, at `ε_O = 0` and every outer simulator, holds for every honest
  prover wherever the inner verifier takes a round: `Hm96Hiding` forces `2·k_I·δ₁ ≥ 1`, and `ZKmalR` holds at any error
  `≥ 1` (`RedTeam.zkmalR_of_one_le`). The mask, the salt hash and `ε_O` are the claim's parameters, so whoever
  discharges it picks them.
* `recursionZKPriv_collSim`: the private variant's `ZKPriv` at `ε_zk = 0`, at any masked outer record and any hash with
  a collision, at the colliding simulator (`RedTeam.zkmalR_collSim`): its simulator is a parameter.
* `recursionComplete_empty`: the recursion's `Complete` at `δ_c = 0`, at a prover with an empty tape
  (`RedTeam.complete_empty`).
-/

namespace Claims.Recursion.RedTeam

open Claims.RedTeam
open scoped ENNReal NNReal

/-- **At a constant mask, hm96's hiding needs `δ₁ ≥ 1/2`.** -/
theorem half_le_of_hm96Hiding_const {δ₁ : ℝ≥0∞} (h : Hm96Hiding (fun _ : Unit => true) (id : Unit → Unit) δ₁) :
    2⁻¹ ≤ δ₁ := by
  have h1 := h fun p => p.1 = true
  simp [PrENNReal, Fintype.sum_prod_type] at h1
  rw [← ENNReal.inv_two_add_inv_two] at h1
  exact (ENNReal.add_le_add_iff_left (ENNReal.inv_ne_top.2 two_ne_zero)).1 h1

/-- **The recursion's `ZK` holds** at the constant mask, `ε_O = 0` and every outer simulator, for every honest prover,
wherever the inner verifier takes a round. -/
theorem recursionZK_const {Coin W Col T Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Fintype Ω] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T)
    (hzk : DO.zk = true) (P : W → Prover Ω) (SO : Msg → ℕ → OSim)
    (hk : ∀ x w, R D I x w → 1 ≤ I.V.k (xI D x)) :
    Recursion.ZK H D I DO P SO (fun _ : Unit => true) (id : Unit → Unit) fun _ _ => 0 := by
  refine ⟨hzk, fun δ₁ hδ => zkmalR_of_one_le H D.V.k P _ _ _ fun x w hR => ?_⟩
  have h3 : ((2⁻¹ : ℝ≥0) : ℝ≥0∞) ≤ (δ₁ : ℝ≥0∞) := by
    rw [ENNReal.coe_inv_two]
    exact half_le_of_hm96Hiding_const hδ
  have h4 : (2⁻¹ : ℝ) ≤ (δ₁ : ℝ) := by exact_mod_cast ENNReal.coe_le_coe.1 h3
  have h5 : (1 : ℝ) ≤ I.V.k (xI D x) := by exact_mod_cast hk x w hR
  simp only [Finset.sum_const_zero, zero_add]
  nlinarith

/-- **The private variant's `ZKPriv` holds at `ε_zk = 0`** at any masked outer record and any hash with a collision,
for every honest prover, at the colliding simulator. -/
theorem recursionZKPriv_collSim {Coin W Col T Ω Y B Cy : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [Fintype Ω] [Fintype Y] [Fintype B] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (hzk : DO.zk = true) (P : W → Prover Ω) (My : Y → B) (cy : Y → Cy)
    {u v : Msg} (h : Collision H u v) : Recursion.ZKPriv H D I DO P (collSim u v) My cy fun _ _ => 0 :=
  ⟨hzk, fun _ _ => zkmalR_collSim H D.V.k P _ h⟩

/-- **The recursion's `Complete` holds at `δ_c = 0`** at a prover with an empty tape. -/
theorem recursionComplete_empty {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg)
    (D : Deploy Coin) (I : Inner Coin W) : Recursion.Complete H D I noTape 0 :=
  complete_empty H D.V noTape _ _

end Claims.Recursion.RedTeam
