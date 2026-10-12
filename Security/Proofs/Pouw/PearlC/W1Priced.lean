import Definitions.Pouw.PearlC.W1

/-!
# The γ-gap property over W1's priced sum

Daniel's item 9 splits W1's charge between two accounts: a run's compute (the instructions' prices) to the compute
account, and every word it fetches first to the bandwidth account, with no refund for a correct unit's operands
(`w1Priced`). The γ-gap property over the priced sum, compute plus `fetch` per word read, follows from the property over
W1 itself at the same `γ` and `η`, with no new assumption:

- `gγ_of_cost_le`: a model that charges at least another's, with the same runs and at most its calls, inherits its
  γ-gap property. An adversary within budget `T` under the dearer model is within it under the cheaper one, and commits
  the same transcript.
- `w1_cost_le_priced`: W1's refund only waives fetches, so W1 charges at most the priced sum when `fetch ≥ 0`.
- `gγ_w1Priced`: hence `Gγ (w1 dv P salt) → Gγ (w1Priced dv salt)`.
- `w1Priced_cost`: the priced cost is compute plus `fetch` times `w1Read`, the two accounts' shares.

The honest reference's cost `Wref` has no fetch term, so `γ` keeps its meaning: the priced sum only adds to what every
program, the reference included, is charged.
-/

namespace Pouw.PearlC.W1Priced

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.W1

variable {Q R S : Type}

/-- **A dearer model keeps the γ-gap property**: if every program of `CM₂` has a program of `CM₁` with the same run, at
most its calls and at most its cost, then `CM₁`'s γ-gap property gives `CM₂`'s at the same `γ` and `η`. -/
theorem gγ_of_cost_le [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM₁ CM₂ : CostModel Q R S)
    (f : CM₂.Prog → CM₁.Prog)
    (hrun : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.run (f p) H U act σ s = CM₂.run p H U act σ s)
    (hcalls : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.calls (f p) H U act σ s ≤ CM₂.calls p H U act σ s)
    (hcost : ∀ (p : CM₂.Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
      CM₁.cost (f p) H U act σ s ≤ CM₂.cost p H U act σ s)
    (P : Protocol Q R S) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) (h : Gγ CM₁ P D γ η) : Gγ CM₂ P D γ η := by
  intro U hU T hT q hq A hA
  let A₁ : Adversary CM₁ := ⟨A.pre, f A.online, A.act⟩
  have hA₁ : A₁.Bounded U T q := fun H s =>
    ⟨(hcost _ _ _ _ _ _).trans (hA H s).1, (Nat.add_le_add_left (hcalls _ _ _ _ _ _) _).trans (hA H s).2⟩
  have e : ∀ x : (Q → R) × S, A₁.transcript U x.1 x.2 = A.transcript U x.1 x.2 := fun x =>
    hrun A.online x.1 U A.act (A.pre.eval x.1) x.2
  have := h U hU T hT q hq A₁ hA₁
  simpa only [e] using this

/-- **W1 charges at most the priced sum**: its refund only waives first fetches. -/
theorem w1_cost_le_priced (dv : W1Device) (hf : 0 ≤ dv.prices.fetch) (P : Protocol Q R S) (salt : S → List ℕ)
    (pr : Prog) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S) :
    (w1 dv P salt).cost pr H U act σ s ≤ (w1Priced dv salt).cost pr H U act σ s := by
  simp only [w1, w1Priced, w1Read]
  rw [Rat.cast_le]
  apply add_le_add_left
  refine mul_le_mul_of_nonneg_left ?_ hf
  exact_mod_cast List.length_filter_le _ _

/-- **The γ-gap property over the priced sum** follows from W1's, at the same `γ` and `η`. -/
theorem gγ_w1Priced [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1Device) (hf : 0 ≤ dv.prices.fetch)
    (P : Protocol Q R S) (salt : S → List ℕ) (D : Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ)
    (h : Gγ (w1 dv P salt) P D γ η) : Gγ (w1Priced dv salt) P D γ η :=
  gγ_of_cost_le (w1 dv P salt) (w1Priced dv salt) (fun p => p) (fun _ _ _ _ _ _ => rfl)
    (fun _ _ _ _ _ _ => le_rfl) (w1_cost_le_priced dv hf P salt) P D γ η h

/-- **The priced cost splits between the accounts**: compute, plus `fetch` per word read. -/
theorem w1Priced_cost (dv : W1Device) (salt : S → List ℕ) (pr : Prog) (H : Q → R) (U : Workload)
    (act : ℕ → Codes) (σ : State) (s : S) :
    (w1Priced dv salt).cost pr H U act σ s =
      ((final dv (salt s) pr U act σ).cost : ℝ) + (dv.prices.fetch : ℝ) * (w1Read dv (salt s) pr U act σ : ℝ) := by
  simp [w1Priced]

end Pouw.PearlC.W1Priced
