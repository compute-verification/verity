module

public import Definitions.Core.InProofs

/-!
# Proofs of `Verity.InProofs`' shape

`Gives` is monotone in the relation it counts against: a relation that accepts less fails more units, so `Gives` at
`right` gives it at any `ok` with `right ≤ ok` (`Gives.mono`). An event inside another has no greater probability
(`prPMF_mono`).
-/

@[expose] public section

namespace Properties.Accounting.InProofs

open Verity.InProofs Verity.Game
open scoped ENNReal

theorem prPMF_mono {α : Type} (μ : PMF α) {E F : α → Prop} (h : ∀ x, E x → F x) : prPMF μ E ≤ prPMF μ F :=
  MeasureTheory.measure_mono fun x hx => h x hx

theorem failing_card_le {Ω : Type} {n : ℕ} {V : Fin n → Type} (r : Run Ω n V) {right ok : (u : Fin n) → V u → Prop}
    (h : right ≤ ok) (ω : Ω) : (r.failing ok ω).card ≤ (r.failing right ω).card := by
  classical
  refine Finset.card_le_card fun u hu => ?_
  simp only [Run.failing, Finset.mem_filter, Finset.mem_univ, true_and] at hu ⊢
  exact fun hr => hu (h u _ hr)

theorem Gives.mono {Ω : Type} {n : ℕ} {V : Fin n → Type} {μ : PMF Ω} {r : Run Ω n V}
    {right ok : (u : Fin n) → V u → Prop} {K₀ : ℕ} {ε : ℝ≥0∞} (h : right ≤ ok) (g : Gives μ r right K₀ ε) :
    Gives μ r ok K₀ ε :=
  le_trans (prPMF_mono μ fun ω hω => ⟨hω.1, hω.2.trans (failing_card_le r h ω)⟩) g

end Properties.Accounting.InProofs
