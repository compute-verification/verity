import Definitions.Accounting.Guarantees.ComposeWindow
import Proofs.Accounting.Compose

/-!
# PoUW's credit, window by window: proof

`CircuitCreditPerWindow` is `CircuitCredit` at each window's play, with a change of measure: under the uniform law,
`Nci.Dampener.prob` is the card ratio `Pouw.RelativeSecurity.prob` takes (`dampenerProb_uniform`); an event of the
window's coins `Ω'` inside an event pulled back along `Ω' ≃ B` has at most its probability (`prob_le_equiv`); and the
play's failure over `B = Paths … × R`, the play at each value `r` of `R` failing with probability at most `δ` by
`CircuitCredit`, has probability at most `δ` (`Pouw.RelativeSecurity.prob_le_of_slices`, `creditPlayOn_prob_le`). A
window that carries no PoUW play is never wrong, so its probability is 0.
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

/-- An event of the coins `Ω'` inside an event `G` of `B` pulled back along `e : Ω' ≃ B` has at most `G`'s
probability. -/
theorem prob_le_equiv {Ω' B : Type} [Fintype Ω'] [Fintype B] (e : Ω' ≃ B) {E : Ω' → Prop} {G : B → Prop}
    (hE : ∀ a, E a → G (e a)) : Pouw.RelativeSecurity.prob E ≤ Pouw.RelativeSecurity.prob G := by
  refine (Pouw.RelativeSecurity.prob_mono hE).trans (le_of_eq ?_)
  rw [Pouw.RelativeSecurity.prob_equiv e]
  congr 1
  funext b
  rw [Equiv.apply_symm_apply]

/-- **A `CreditPlayOn` play's failure has probability at most `δ ≥ 0`**: `CircuitCredit` at the play of each value of
the coins `R` it doesn't read, averaged over them. -/
theorem creditPlayOn_prob_le {Ω' : Type} [Fintype Ω'] {E : Ω' → Prop} {δ : ℝ} {c : Accounting.Compose.Certificate}
    (hδ0 : 0 ≤ δ) (h : Accounting.Guarantees.Compose.CreditPlayOn Ω' E δ c) : Pouw.RelativeSecurity.prob E ≤ δ := by
  obtain ⟨ι, K, P, T, L, U, F, X, C, Ξ, V, h⟩ := h
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, h⟩ := h
  obtain ⟨Sc, Cd, Sh, Z, H, Np, Kd, hN, hr, hK, W, h⟩ := h
  obtain ⟨R, _, Pl, pr, folded, lam, h⟩ := h
  obtain ⟨ρ, p, p0, wmax, Λ, δs, δbind, δks, win, e, hpre, hc, hE, hδ⟩ := h
  refine (prob_le_equiv e (G := fun b => (Pl b.2).accepts b.1 ∧ ((Pl b.2).run b.1).run.time < (1 - c.eps) * c.credit)
    hE).trans (Pouw.RelativeSecurity.prob_le_of_slices _ hδ0 fun r => ?_)
  subst hc
  obtain ⟨hS, hρ, hρN, hwm, hlk, hΛ, hSR, hgame, hOf, hMoved, hJS, hPins, hMu, hRB, hG, hUF, hFits, hMS, hSplit,
    hClock, hDisp, hTen, hPipe, hSmem, hWords, hSem, hFold, hFill⟩ := hpre r
  have hcc := Accounting.SecurityProofs.Compose.CircuitCredit Sc Cd Sh Z H hN hr hK W (Pl r) (pr r) (folded r) lam ρ p
    p0 wmax Λ δs δbind δks win hS hρ hρN hwm hlk hΛ hSR hgame hOf hMoved hJS hPins hMu hRB hG hUF hFits hMS hSplit
    hClock hDisp hTen hPipe hSmem hWords hSem hFold hFill
  unfold Certificate.Holds at hcc
  exact hcc.trans hδ

/-- **PoUW's credit, window by window** (`Accounting.Guarantees.Compose.CircuitCreditPerWindow`). -/
theorem CircuitCreditPerWindow : Accounting.Guarantees.Compose.CircuitCreditPerWindow := by
  intro T Ω _ _ wrong cert p hp hw t ξ
  rcases hw t ξ with h | h
  · rw [dampenerProb_uniform]
    exact creditPlayOn_prob_le hp h
  · unfold Nci.Dampener.prob
    simpa [h] using hp

end Accounting.SecurityProofs.Compose
