module

public import Definitions.Pouw.PearlC.W1

/-! The gap property over W1's priced sum follows from the property over W1 itself at the same `g` and `δ`. -/

@[expose] public section

namespace Pouw.PearlC.W1Priced

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.W1

variable {Q R S : Type}

theorem gGap_of_cost_le [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM₁ CM₂ : CostModel Q R S)
    (f : CM₂.Prog → CM₁.Prog)
    (hrun : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.run (f p) H U act σ s = CM₂.run p H U act σ s)
    (hcalls : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.calls (f p) H U act σ s ≤ CM₂.calls p H U act σ s)
    (hcost : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.cost (f p) H U act σ s ≤ CM₂.cost p H U act σ s)
    (P : Protocol Q R S) (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) (h : GGap CM₁ P D g δ) : GGap CM₂ P D g δ := by
  intro U hU T hT q hq A hA
  let A₁ : Adversary CM₁ := ⟨A.pre, f A.online, A.act⟩
  have hA₁ : A₁.Bounded U T q := fun H s =>
    ⟨(hcost _ _ _ _ _ _).trans (hA H s).1, (Nat.add_le_add_left (hcalls _ _ _ _ _ _) _).trans (hA H s).2⟩
  have e : ∀ x : (Q → R) × S, A₁.trace U x.1 x.2 = A.trace U x.1 x.2 := fun x =>
    hrun A.online x.1 U A.act (A.pre.eval x.1) x.2
  have := h U hU T hT q hq A₁ hA₁
  simpa only [e] using this

theorem w1_cost_le_priced (dv : W1Device) (hf : 0 ≤ dv.prices.fetch) (P : Protocol Q R S) (salt : S → List ℕ)
    (pr : CodeImage) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S) :
    (w1 dv P salt).cost pr H U act σ s ≤ (w1Priced dv salt).cost pr H U act σ s := by
  simp only [w1, w1Priced, w1Read]
  rw [Rat.cast_le]
  refine add_le_add le_rfl (mul_le_mul_of_nonneg_left ?_ hf)
  exact_mod_cast List.length_filter_le _ _

theorem gGap_w1Priced [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1Device) (hf : 0 ≤ dv.prices.fetch)
    (P : Protocol Q R S) (salt : S → List ℕ) (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ)
    (h : GGap (w1 dv P salt) P D g δ) : GGap (w1Priced dv salt) P D g δ :=
  gGap_of_cost_le (w1 dv P salt) (w1Priced dv salt) (fun p => p) (fun _ _ _ _ _ _ => rfl)
    (fun _ _ _ _ _ _ => le_rfl) (w1_cost_le_priced dv hf P salt) P D g δ h

theorem w1Priced_cost (dv : W1Device) (salt : S → List ℕ) (pr : CodeImage) (H : Q → R) (U : Workload)
    (act : ℕ → Codes) (σ : State) (s : S) :
    (w1Priced dv salt).cost pr H U act σ s =
      ((final dv (salt s) pr U act σ).cost : ℝ) + (dv.prices.fetch : ℝ) * (w1Read dv (salt s) pr U act σ : ℝ) := by
  simp [w1Priced]

end Pouw.PearlC.W1Priced
