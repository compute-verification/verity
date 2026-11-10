module

public import Definitions.Nci.Assumptions
public import Definitions.Nci.Threads
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.Nci

/-!
# The inference-only policy's lemmas

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`Nci.SecurityProofs.G : Nci.Guarantees.G`.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains the reference circuit `r` (`M`) and complies at `(Γ, α)`
(`CompliesRate`), with its registered information free of the reference circuit's fresh input (`ThetaRegistered`), has
at least `Reads_Γ(r) / α` gates. Lower bounds on the fresh input a computation reads become lower bounds on size when the
declared circuit computes `r` the way `r` does. -/
def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (r d : RegisteredCircuit) (M : Containment r.D d.D), CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    (Reads Γ r : ℝ) ≤ α * d.D.size d.D.computed

/-- **Inference complies as declared**: a Boolean declaration divided into sessions of exactly `c` gates, padding
included, each reading from outside itself only registered information and its own prompt `prompt u` of at most `k`
bits, complies at `(c, k)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (c k : ℕ) (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)),
    d.D.Boolean → Assumptions.Sessions d P prompt c k → Complies c k d ∧ slowdown (Complies c k) d ≤ 1

end Nci.Guarantees
