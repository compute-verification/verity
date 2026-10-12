import Proofs.Flock.Soundness.Model.Statement
import Proofs.Flock.Soundness.Refine.Tapes
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckReplay

/-!
# The honest `final_c` is the replay's `c` claim's value

`finalC_eq_eval`: the honest zerocheck's clear `final_c = Σ_x eq(r, x)·ĉ(z, x)` (`honest`, over `n = m − 6` residual
variables, with `c` the witness cast) is the model's value of the claim `(0, L_s(z), r)`, written as `claimOf`'s fields
(`⟨fOf value, vecOf 64 skip, vecOf (m − 6) x⟩`, so `claimOf m c` unfolds to it): `L_s(ζ)` is the executable's
`skipWeights ζ`, and address `a = s + 64·x`.

`finalC_masked_eval`: with `replay_zerocheck`'s `c` claim, the masked proof's `final_c` (`Masks.finalC`) is that claim's
value under the pad `h_fc`, the hypothesis the ring switch's completeness takes (`fOf p.zerocheck.finalC =
(claimOf st.m rp.claims[1]!).eval z + fOf h[lay.fc]!`).
-/

namespace FlockSoundness.Discharge.Composed.ZerocheckMasked

open Flock Flock.Zk FlockLevel3 FlockSoundness.Refine FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

theorem mul_sum_fin (R : 𝔽) (f : Fin 64 → 𝔽) : R * ∑ s, f s = ∑ s, R * f s := Finset.mul_sum _ _ _

theorem eqR_eq_eqAt (r : Array F128) (n x : ℕ) : eqR (coin r) 0 n x = Model.eqAt (vecOf n r) x := by
  unfold eqR Model.eqAt
  rw [← Fin.prod_univ_eq_prod_range (fun j => if x.testBit j then coin r (0 + j) else 1 + coin r (0 + j)) n]
  refine Finset.prod_congr rfl fun j _ => ?_
  simp only [zero_add]
  rfl

/-- **The honest `final_c` is the `c` claim's value**: `Σ_x eq(r, x)·ĉ(z, x)` over the witness `w` is the model's
evaluation of the claim with skip weights `skipWeights z` at the point `r`. -/
theorem finalC_eq_eval (m : ℕ) (hm : 6 ≤ m) (a b c : ℕ → 𝔽) (w : Model.Witness m)
    (hc : ∀ i : Fin (2 ^ m), c i.val = ((w i).val : 𝔽)) (r rho : Array F128) (z : F128) :
    fOf (honest (m - 6) a b c r rho z).finalC =
      (⟨fOf 0, vecOf 64 (skipWeights z), vecOf (m - 6) r⟩ : Model.Claim 𝔽 m).eval w := by
  classical
  have hpow : 2 ^ m = 2 ^ (m - 6) * 64 := by
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]
  let e : Fin (2 ^ (m - 6)) × Fin 64 ≃ Fin (2 ^ m) := finProdFinEquiv.trans (finCongr hpow.symm)
  have he : ∀ q, (e q).val = q.2.val + 64 * q.1.val := fun q => by
    simp [e, finProdFinEquiv]
  show (∑ x ∈ Finset.range (2 ^ (m - 6)), eqR (coin r) 0 (m - 6) x * skipAt c (GF128.of z) x) = _
  unfold Model.Claim.eval
  rw [← e.sum_comp, Fintype.sum_prod_type, ← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [skipAt, mul_sum_fin]
  refine Finset.sum_congr rfl fun s _ => ?_
  have hmod : (e (x, s)).val % 64 = s.val := by simp only [he]; omega
  have hdiv : (e (x, s)).val / 64 = x.val := by simp only [he]; omega
  have hsk : vecOf 64 (skipWeights z) ⟨(e (x, s)).val % 64, Nat.mod_lt _ (by norm_num)⟩ = lagS (GF128.of z) s := by
    show fOf (skipWeights z)[(e (x, s)).val % 64]! = _
    rw [hmod]
    rfl
  dsimp only
  rw [hsk, hdiv, ← hc, he, eqR_eq_eqAt]
  ring

/-- **The masked `final_c` is the replay's `c` claim's value under its pad**, when the proof masks the honest
zerocheck at the replay's coins (`claimOf st.m rp.claims[1]!` unfolds to the claim on the right). -/
theorem finalC_masked_eval (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) {t t' : Transcript}
    {rp : Replayed} (hm : 14 ≤ st.m) (hrun : Zk.replay st lay p root t = .ok (rp, t')) (h : Array F128) (n : ℕ)
    (a b c : ℕ → 𝔽) (w : Model.Witness st.m) (hc : ∀ i : Fin (2 ^ st.m), c i.val = ((w i).val : 𝔽))
    (rho : Array F128) (hM : Masks lay h (honest (st.m - 6) a b c (pinned ++ rOuterAt t) rho (zAt t)) p.zerocheck n) :
    fOf p.zerocheck.finalC =
      (⟨fOf rp.claims[1]!.value, vecOf 64 rp.claims[1]!.skip, vecOf (st.m - 6) rp.claims[1]!.x⟩ :
        Model.Claim 𝔽 st.m).eval w + fOf h[lay.fc]! := by
  obtain ⟨-, -, hcl, -⟩ := replay_zerocheck st lay p root hm hrun
  have hg := finalC_eq_eval st.m (by omega) a b c w hc (pinned ++ rOuterAt t) rho (zAt t)
  rw [hcl, hM.finalC]
  exact (of_add _ _).trans (congrArg (· + fOf h[lay.fc]!) hg)

end FlockSoundness.Discharge.Composed.ZerocheckMasked
