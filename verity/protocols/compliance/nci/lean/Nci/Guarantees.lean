import Nci.Assumptions

/-!
# NCI's guarantees

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`Nci.SecurityProofs.G : Nci.Guarantees.G`.
-/

namespace Nci.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains `c` and complies at `(Γ, α)` has at least `Reads_Γ(c) / α` gates.
Lower bounds on NCI become lower bounds on size when the declared circuit computes `c` the way `c` does. -/
def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (c d : Declaration), c.Contains d → Complies Γ α d →
    (Reads Γ c : ℝ) ≤ α * d.D.size d.D.computed

/-- **Inference complies as declared**: a declaration divided into sessions of at most `Γ` gates, each reading from
outside itself only baseline information and its own prompt `prompt u`, with at most `α` bits of prompt per gate of the
session, complies at `(Γ, α)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (d : Declaration) (n : ℕ) (P : Partition d.D n) (prompt : Fin n → Finset (Fin d.D.N)),
    (∀ u, d.D.size (P.gates u) ≤ Γ) → (∀ u, d.D.into (P.gates u) ⊆ d.θ ∪ prompt u) →
    (∀ u, (d.D.width (prompt u) : ℝ) ≤ α * d.D.size (P.gates u)) →
      Complies Γ α d ∧ slowdown (Complies Γ α) d ≤ 1

/-- **Updated matrix multiplications are throttled** (the draft's theorem, "Throttling updated matrix
multiplications"): under compute-bounded incompressibility, every declared circuit `D ⊨ 𝒫` that runs updated matrix
multiplications is larger than those multiplications by a factor of at least `F`. The multiplications have about `2G`
gates: `G` products, and about as many additions. `K = αΓ` is the most an IU reads, `b_w` and `b_a` are the widths of a
weight entry and an activation entry, and `d` is the smallest inner dimension (a smaller `d` gives a smaller `F`). -/
def UpdatedMatMul : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (D : Declaration) (c : Chain D.D) (d : ℕ),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, d ≤ c.d ℓ) → Complies Γ α D →
    Assumptions.ActivationsIncompressible D c Γ (α * Γ) b_w b_a →
    Assumptions.PartialSumsIncompressible D c Γ (α * Γ) b_w b_a →
      F α (α * Γ) b_w b_a d * (2 * c.G) ≤ D.D.size D.D.computed

/-- **Frontier training is throttled** (the draft, "From updated matrix multiplications to training"): let `T` be a
training circuit in the class `𝒯`, so that every declared circuit that implements `T` and complies runs updated matrix
multiplications, which take up a fraction `κ` of `size(T)` (the coverage `κ(T)`). If compute-bounded incompressibility
holds for the multiplications each of them runs, then `ρ_𝒫(T) ≥ κ F`. -/
def Training : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (T : Declaration) (d : ℕ),
    0 < α → 0 < b_w → 0 < b_a → 0 < T.D.size T.D.computed →
    (∀ D, T.Implements D → Complies Γ α D → ∃ c : Chain D.D,
      (∀ ℓ, d ≤ c.d ℓ) ∧ κ * T.D.size T.D.computed ≤ 2 * c.G ∧
      Assumptions.ActivationsIncompressible D c Γ (α * Γ) b_w b_a ∧
      Assumptions.PartialSumsIncompressible D c Γ (α * Γ) b_w b_a) →
    ENNReal.ofReal (κ * F α (α * Γ) b_w b_a d) ≤ slowdown (Complies Γ α) T

end Nci.Guarantees
