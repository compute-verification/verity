import Models.Nci.Assumptions.Training

/-!
# NCI's guarantees on a reference training circuit, under one assumption

`ThreadedThrottled` and `ContainedTraining` restated with `TrainingIncompressible` as their only named assumption, in
place of `WeightsIncompressible` for the first and `ActivationsSupport` and `PartialSumsSupport` for the second. Each
statement names both the chain and the threading the assumption charges, and uses one of them.
-/

namespace Nci.Guarantees

open Verity

/-- **High NCI means padding, under training's incompressibility** (`ThreadedThrottled` with `TrainingIncompressible`): a
declaration `d` that contains a `(dd, e)`-threaded reference circuit `c` and complies at `(Γ, α)` has at least
`b_a / (αΛ)` gates per counted multiplication of `c`, where `Λ` is the thread bound at `S = αΓ`. -/
def IncompressibleThreadedThrottled : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (ch : Chain c.D) (T : Threading c) (dd e : ℕ),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → c.Contains d → Complies Γ α d →
    T.Threaded Γ (α * Γ) dd e → Assumptions.TrainingIncompressible c ch T Γ (α * Γ) b_w b_a →
      b_a * T.M.card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Training is slowed by the partial-sum law, under training's incompressibility** (`ContainedTraining` with
`TrainingIncompressible`): when the products of a chain `ch` that `c` runs, at 2 flops each, are a fraction `κ` of `c`'s
size, every declaration that contains `c` and complies at `(Γ, α)` is larger by a factor of at least `κ F`. `dd` is the
chain's smallest inner dimension. -/
def IncompressibleContainedTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : Declaration) (ch : Chain c.D) (T : Threading c) (dd : ℕ),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → c.Contains d → Complies Γ α d →
    Assumptions.TrainingIncompressible c ch T Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * ch.G →
      κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed ≤ d.D.size d.D.computed

end Nci.Guarantees
