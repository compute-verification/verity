import Definitions.Nci.Guarantees.RemoteEndToEnd
import Proofs.Nci.RemotePeriod
import Proofs.Accounting.Overhead

/-!
# The remote protocol end to end: the proof

`RemoteEndToEnd` is the overhead core's event decomposition, `Accounting.SecurityProofs.Overhead.EndToEnd`, in the
experiment that draws the period's coins `ξ` as one uniform coin (`Verity.Game.draw`, whose probabilities are the
uniform law's: `prob_eq_draw`). Its events:

* the claims and the class are both `DampenerRemotePeriod`'s event `G`: the compute efficiency is bounded and the
  hardware multiple at the run's own count `M` is at least `ρ`. Its complement has probability at most `δ`, so accepted
  runs outside it do too, and no run is in the claims and outside the class;
* the cost is the hardware multiple at the task's `N` less `ρ`, and `s_min = 0`: the multiple falls as the count grows
  (`hardwareMultiple_anti`), so on `G` a run with `M ≥ N` has its multiple at `N` at least `ρ`;
* the coverage exception is `Coverage`'s, which bounds the runs that reach the task with fewer than `N` whatever the
  class.

`remoteEndToEnd_of_period` is the argument from `DampenerRemotePeriod`; `RemoteEndToEnd` applies it to that guarantee's
proof.
-/

namespace Nci.SecurityProofs

open Verity Nci.Dampener

/-- Under the uniform law, `prob` is the probability of the game that draws one uniform coin. -/
theorem prob_eq_draw {Ξ : Type} [Fintype Ξ] [Nonempty Ξ] (E : Ξ → Prop) (s : Game.Strategy (Game.draw Ξ)) :
    prob (PMF.uniformOfFintype Ξ) E = (Game.prob E (Game.draw Ξ) s).toReal := by
  rw [prob_uniform_eq]
  rfl

/-- A bound on `prob` under the uniform law is one on the drawing game's probability. -/
theorem draw_le_ofReal {Ξ : Type} [Fintype Ξ] [Nonempty Ξ] {E : Ξ → Prop} (s : Game.Strategy (Game.draw Ξ)) {x : ℝ}
    (h : prob (PMF.uniformOfFintype Ξ) E ≤ x) : Game.prob E (Game.draw Ξ) s ≤ ENNReal.ofReal x := by
  rw [prob_eq_draw E s] at h
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (Game.prob_le_one E _ s))]
  exact ENNReal.ofReal_le_ofReal h

/-- **The hardware multiple falls as the count grows**: at `0 < N ≤ M`, `max(1, MFU₀/η(M)) ≤ max(1, MFU₀/η(N))`. -/
theorem hardwareMultiple_anti {mfu κ C U N M : ℝ} (hmfu : 0 < mfu) (hκ : 0 < κ) (hS : 0 < C + U) (hN : 0 < N)
    (hNM : N ≤ M) : hardwareMultiple mfu κ M C U ≤ hardwareMultiple mfu κ N C U := by
  unfold hardwareMultiple efficiency
  refine max_le_max le_rfl (div_le_div_of_nonneg_left hmfu.le ?_ ?_)
  · exact div_pos (mul_pos (div_pos two_pos hκ) hN) hS
  · exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hNM (div_pos two_pos hκ).le) hS.le

/-- **`RemoteEndToEnd` from `DampenerRemotePeriod`**: `EndToEnd` at the period's coins, with the dampener's event as
the claims and the class. -/
theorem remoteEndToEnd_of_period (h : Guarantees.DampenerRemotePeriod) : Guarantees.RemoteEndToEnd := by
  intro T Ω _ _ c wrong p ε β ι _ bad drawn detected win I idW τw w ks K Φ Δ H q B εW κ C U χ mfu W M Mu Mr Mx R _
    r d Th Γ α b_w b_a Λ dd e accept success N η hmodel hpouw hsp hN hcov
  obtain ⟨⟨hp, hε, hc, hcC, hq0, hq1⟩, ⟨hord, hI, hH, hΔ, hΦ, hTD, hcap, hw, hB, hwB⟩,
    ⟨hκ, hC, hU, hmfu, hmfu1, hα, hba, hΛ, hthr, hbound, hsize⟩, ⟨hM0, hM, hu, hres, hx⟩⟩ := hmodel
  obtain ⟨hbadP, hdrawnA, hmiss, hks⟩ := hsp
  have hD := h T Ω c wrong p ε β ι bad drawn detected win I idW τw w ks K Φ Δ H q B εW κ C U χ mfu W M Mu Mr Mx R r
    d Th Γ α b_w b_a Λ dd e hp hε hc hcC hpouw hbadP hdrawnA hq0 hq1 hmiss hks hord hI hH hΔ hΦ hTD hcap hw hB hwB hκ
    hC hU hmfu hmfu1 hα hba hΛ hthr hbound hsize hM0 hM hu hres hx
  let μ := PMF.uniformOfFintype ((t : Fin T) → Ω t)
  -- the dampener's argument, whose bound `ρ` is `remoteDampener` of it
  let x : ((t : Fin T) → Ω t) → ℝ := fun ξ => (1 - unmonitoredShare C U) *
      (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
    declaredShare κ (rhoT α b_a Λ) C U + χ
  -- the dampener's event
  let G : ((t : Fin T) → Ω t) → Prop := fun ξ =>
    efficiency κ (M ξ) C U ≤
        unmonitoredShare C U +
            (1 - unmonitoredShare C U) *
              (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
          declaredShare κ (rhoT α b_a Λ) C U + χ ∧
      remoteDampener mfu (unmonitoredShare C U) (x ξ) ≤ hardwareMultiple mfu κ (M ξ) C U
  let δ := p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t
  let s : Game.Strategy (Game.draw ((t : Fin T) → Ω t)) := fun _ => PUnit.unit
  have hnn : ∀ E : ((t : Fin T) → Ω t) → Prop, 0 ≤ prob μ E := fun E => by
    rw [prob_eq_draw E s]
    exact ENNReal.toReal_nonneg
  have hGδ : prob μ (fun ξ => ¬ G ξ) ≤ δ := by
    have h1 := prob_add_not μ G
    have h2 : 1 - δ ≤ prob μ G := hD
    linarith
  have hδ0 : 0 ≤ δ := (hnn _).trans hGδ
  have hη0 : 0 ≤ η := (hnn _).trans hcov
  have hE := Accounting.SecurityProofs.Overhead.EndToEnd ((t : Fin T) → Ω t) (Game.draw ((t : Fin T) → Ω t))
    (fun _ => accept) (fun _ => G) (fun _ => G) (fun _ => success) (fun _ ξ => (M ξ : ℝ))
    (fun _ ξ => hardwareMultiple mfu κ N C U - remoteDampener mfu (unmonitoredShare C U) (x ξ)) N 0
    (fun _ => ENNReal.ofReal δ) (fun _ => 0) (fun _ => ENNReal.ofReal η)
    (fun s => draw_le_ofReal s ((prob_mono μ fun ξ h => h.2).trans hGδ))
    (fun s => (Game.prob_mono (fun _ h => h.2 h.1) _ s).trans_eq (Game.prob_false _ s))
    (fun _ ξ hg hNM => sub_nonneg.2 (hg.2.trans (hardwareMultiple_anti hmfu hκ (by linarith)
      (Nat.cast_pos.2 hN) hNM)))
    (fun s => draw_le_ofReal s ((prob_mono μ fun ξ h => ⟨h.2.1, Nat.cast_lt.1 h.2.2⟩).trans hcov)) s
  rw [prob_eq_draw _ s, ← ENNReal.toReal_ofReal (add_nonneg hδ0 hη0), ENNReal.ofReal_add hδ0 hη0]
  refine ENNReal.toReal_mono (by simp) (le_trans ?_ (hE.trans_eq (by rw [add_zero])))
  exact Game.prob_mono (fun ξ h => ⟨h.1, h.2.1, sub_neg.2 h.2.2⟩) _ s

/-- **The remote protocol, end to end** (`Nci.Guarantees.RemoteEndToEnd`). -/
theorem RemoteEndToEnd : Guarantees.RemoteEndToEnd := remoteEndToEnd_of_period DampenerRemotePeriod

end Nci.SecurityProofs
