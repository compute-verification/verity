import Definitions.Accounting.Guarantees.ComposeWindow
import Proofs.Accounting.Compose

/-!
# PoUW's credit, window by window: proof

`CircuitCreditPerWindow` is `CircuitCredit` at each window's play, with a change of measure: under the uniform law,
`Nci.Dampener.prob` is the card ratio `Pouw.RelativeSecurity.prob` takes (`dampenerProb_uniform`), and an event of the
play's coins pulled back along the window's coins `Ω' ≃ A × R` has the same ratio (`prob_fst_equiv`). A window that
carries no PoUW play is never wrong, so its probability is 0.
-/

namespace Accounting.SecurityProofs.Compose

open Accounting.Compose

/-- Under the uniform law on a finite type, `Nci.Dampener.prob` is `Pouw.RelativeSecurity.prob`. -/
theorem dampenerProb_uniform {Ω : Type} [Fintype Ω] [Nonempty Ω] (E : Ω → Prop) :
    Nci.Dampener.prob (PMF.uniformOfFintype Ω) E = Pouw.RelativeSecurity.prob E := by
  classical
  have hu : ∀ ω : Ω, (PMF.uniformOfFintype Ω ω).toReal = (Fintype.card Ω : ℝ)⁻¹ := fun ω => by
    simp [PMF.uniformOfFintype_apply]
  rw [Pouw.RelativeSecurity.prob_eq_card]
  unfold Nci.Dampener.prob
  simp_rw [hu]
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]

/-- An event of the coins `Ω'` that reads only the first part of `e : Ω' ≃ A × R` has the probability of that part's
event. -/
theorem prob_fst_equiv {Ω' A R : Type} [Fintype Ω'] [Nonempty Ω'] [Fintype A] [Fintype R] (e : Ω' ≃ A × R)
    {E : Ω' → Prop} {F : A → Prop} (hE : ∀ a, E a ↔ F (e a).1) :
    Pouw.RelativeSecurity.prob E = Pouw.RelativeSecurity.prob F := by
  have : Nonempty R := ⟨(e (Classical.arbitrary Ω')).2⟩
  rw [Pouw.RelativeSecurity.prob_equiv e E, ← Pouw.RelativeSecurity.prob_fst (β := R) F]
  congr 1
  funext b
  exact propext ((hE (e.symm b)).trans (by rw [Equiv.apply_symm_apply]))

/-- **PoUW's credit, window by window** (`Accounting.Guarantees.Compose.CircuitCreditPerWindow`). -/
theorem CircuitCreditPerWindow : Accounting.Guarantees.Compose.CircuitCreditPerWindow := by
  intro T Ω _ _ wrong p hp hw t ξ
  rcases hw t ξ with h | h
  · obtain ⟨ι, K, P, T', L, U, F, X, C, Ξ, V, h⟩ := h
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, h⟩ := h
    obtain ⟨Sc, Cd, Sh, Z, H, Np, Kd, hN, hr, hK, h⟩ := h
    obtain ⟨W, Pl, pr, folded, lam, h⟩ := h
    obtain ⟨ρ, p', p0, wmax, Λ, δs, δbind, δks, win, h⟩ := h
    obtain ⟨R, _, e, hpre, hE, hδ⟩ := h
    obtain ⟨hS, hρ, hρN, hwm, hlk, hΛ, hSR, hgame, hOf, hMoved, hJS, hPins, hMu, hRB, hG, hUF, hFits, hMS, hSplit,
      hClock, hDisp, hTen, hPipe, hSmem, hWords, hSem, hFold, hFill⟩ := hpre
    have hc := Accounting.SecurityProofs.Compose.CircuitCredit Sc Cd Sh Z H hN hr hK W Pl pr folded lam ρ p' p0 wmax Λ
      δs δbind δks win hS hρ hρN hwm hlk hΛ hSR hgame hOf hMoved hJS hPins hMu hRB hG hUF hFits hMS hSplit hClock hDisp
      hTen hPipe hSmem hWords hSem hFold hFill
    unfold Certificate.Holds at hc
    rw [dampenerProb_uniform]
    refine le_trans (le_of_eq ?_) (hc.trans hδ)
    exact prob_fst_equiv e hE
  · unfold Nci.Dampener.prob
    simpa [h] using hp

end Accounting.SecurityProofs.Compose
