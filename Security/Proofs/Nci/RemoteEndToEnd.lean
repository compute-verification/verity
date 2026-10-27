import Definitions.Nci.Guarantees.RemoteEndToEnd
import Proofs.Nci.Remote

/-!
# The remote protocol end to end: the proof from the remote dampener over a period

`remoteEndToEnd_of_period` is the whole argument: `DampenerRemotePeriod` at the period's coins, its two protocol legs
read off `PouwPerWindow` and `SampledPerWindow`, gives an event `G` of probability at least `1 − δ` on which the
hardware multiple at the run's own count `M` is at least `ρ`; the multiple falls as the count grows
(`hardwareMultiple_anti`), so on `G` a run with `M ≥ N` has its multiple at the task's `N` at least `ρ` too, and the
cheap runs that reach the task lie in `¬G` or in `Coverage`'s exception (`prob_cheap_le`).

`prob_cheap_le` is the spine's Step 1 event decomposition at one good event; the overhead core's generic `EndToEnd`
replaces it when it lands.
-/

namespace Nci.SecurityProofs

open Verity Nci.Dampener

/-- **The cheap-success decomposition** (the spine's Step 1, at one good event; the overhead core's `EndToEnd` replaces
it). If an event `G` has probability at least `1 − δ`, runs that reach the task with fewer than `N` counted
multiplications have probability at most `η`, and no run in `G` with at least `N` is cheap, then runs that are accepted,
reach the task and are cheap have probability at most `δ + η`. -/
theorem prob_cheap_le {Ω : Type} [Fintype Ω] (μ : PMF Ω) {accept success cheap good : Ω → Prop} {M : Ω → ℕ} {N : ℕ}
    {δ η : ℝ} (hgood : 1 - δ ≤ prob μ good) (hcov : prob μ (fun ω => success ω ∧ M ω < N) ≤ η)
    (hinv : ∀ ω, good ω → N ≤ M ω → ¬ cheap ω) :
    prob μ (fun ω => accept ω ∧ success ω ∧ cheap ω) ≤ δ + η := by
  have h1 : prob μ (fun ω => accept ω ∧ success ω ∧ cheap ω) ≤
      prob μ (fun ω => ¬ good ω ∨ (success ω ∧ M ω < N)) := by
    refine prob_mono μ fun ω h => ?_
    by_cases hg : good ω
    · exact Or.inr ⟨h.2.1, not_le.1 fun hN => hinv ω hg hN h.2.2⟩
    · exact Or.inl hg
  have h2 := prob_or_le μ (fun ω => ¬ good ω) (fun ω => success ω ∧ M ω < N)
  have h3 := prob_add_not μ good
  linarith

/-- **The hardware multiple falls as the count grows**: at `0 < N ≤ M`, `max(1, MFU₀/η(M)) ≤ max(1, MFU₀/η(N))`. -/
theorem hardwareMultiple_anti {mfu κ C U N M : ℝ} (hmfu : 0 < mfu) (hκ : 0 < κ) (hS : 0 < C + U) (hN : 0 < N)
    (hNM : N ≤ M) : hardwareMultiple mfu κ M C U ≤ hardwareMultiple mfu κ N C U := by
  unfold hardwareMultiple efficiency
  refine max_le_max le_rfl (div_le_div_of_nonneg_left hmfu.le ?_ ?_)
  · exact div_pos (mul_pos (div_pos two_pos hκ) hN) hS
  · exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hNM (div_pos two_pos hκ).le) hS.le

/-- **`RemoteEndToEnd` from `DampenerRemotePeriod`**: the dampener's event, coverage and the multiple's monotonicity. -/
theorem remoteEndToEnd_of_period (h : Guarantees.DampenerRemotePeriod) : Guarantees.RemoteEndToEnd := by
  intro T Ω _ _ c wrong p ε β ι _ bad drawn detected win I idW τw w ks K Φ Δ H q B εW κ C U χ mfu W M Mu Mr Mx R _
    r d Th Γ α b_w b_a Λ dd e accept success N η hp hε hc hcC hpouw hq0 hq1 hsp hord hI hH hΔ hΦ hTD hcap hw hB hwB
    hκ hC hU hmfu hmfu1 hα hba hΛ hthr hbound hsize hM0 hM hu hres hx hN hcov
  obtain ⟨hbadP, hdrawnA, hmiss, hks⟩ := hsp
  have hD := h T Ω c wrong p ε β ι bad drawn detected win I idW τw w ks K Φ Δ H q B εW κ C U χ mfu W M Mu Mr Mx R r
    d Th Γ α b_w b_a Λ dd e hp hε hc hcC hpouw hbadP hdrawnA hq0 hq1 hmiss hks hord hI hH hΔ hΦ hTD hcap hw hB hwB hκ
    hC hU hmfu hmfu1 hα hba hΛ hthr hbound hsize hM0 hM hu hres hx
  refine prob_cheap_le _ hD hcov fun ξ hg hNM hcheap => ?_
  have hanti := hardwareMultiple_anti hmfu hκ (by linarith) (Nat.cast_pos.2 hN) (Nat.cast_le.2 hNM)
    (C := C) (U := U)
  exact absurd (hcheap.trans_le (hg.2.trans hanti)) (lt_irrefl _)

end Nci.SecurityProofs
