import Nci.Protocol

/-!
# NCI's guarantees

Each is a `def G : Prop` over `Protocol`'s definitions, proved as `Nci.SecurityProofs.G : Nci.Guarantees.G`.
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

end Nci.Guarantees
