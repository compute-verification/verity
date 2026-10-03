import FlockSoundness.Model.Statement
import FlockSoundness.Refine.Tapes
import FlockSoundness.Discharge.Composed.Inner.ZerocheckReplay

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

/-- **The honest `final_c` is the `c` claim's value**: `Σ_x eq(r, x)·ĉ(z, x)` over the witness `w` is the model's
evaluation of the claim with skip weights `skipWeights z` at the point `r`. -/
theorem finalC_eq_eval (m : ℕ) (hm : 6 ≤ m) (a b c : ℕ → 𝔽) (w : Model.Witness m)
    (hc : ∀ i : Fin (2 ^ m), c i.val = ((w i).val : 𝔽)) (r rho : Array F128) (z : F128) :
    fOf (honest (m - 6) a b c r rho z).finalC =
      (⟨fOf 0, vecOf 64 (skipWeights z), vecOf (m - 6) r⟩ : Model.Claim 𝔽 m).eval w := by
  sorry

/-- **The masked `final_c` is the replay's `c` claim's value under its pad**, when the proof masks the honest
zerocheck at the replay's coins (`claimOf st.m rp.claims[1]!` unfolds to the claim on the right). -/
theorem finalC_masked_eval (st : Setup) (lay : PadLayout) (p : Proof) (root : ByteArray) {t t' : Transcript}
    {rp : Replayed} (hm : 14 ≤ st.m) (hrun : Zk.replay st lay p root t = .ok (rp, t')) (h : Array F128) (n : ℕ)
    (a b c : ℕ → 𝔽) (w : Model.Witness st.m) (hc : ∀ i : Fin (2 ^ st.m), c i.val = ((w i).val : 𝔽))
    (rho : Array F128) (hM : Masks lay h (honest (st.m - 6) a b c (pinned ++ rOuterAt t) rho (zAt t)) p.zerocheck n) :
    fOf p.zerocheck.finalC =
      (⟨fOf rp.claims[1]!.value, vecOf 64 rp.claims[1]!.skip, vecOf (st.m - 6) rp.claims[1]!.x⟩ :
        Model.Claim 𝔽 st.m).eval w + fOf h[lay.fc]! := by
  sorry

end FlockSoundness.Discharge.Composed.ZerocheckMasked
