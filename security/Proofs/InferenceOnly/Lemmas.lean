module

public import Definitions.InferenceOnly.Assumptions
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.InferenceOnly

/-!
# The inference-only policy's lemmas

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each is a `def G : Prop` over `Protocol`'s definitions and the assumptions, proved as
`InferenceOnly.SecurityProofs.G : InferenceOnly.Guarantees.G`.
-/

@[expose] public section

open Slowdown

namespace InferenceOnly.Guarantees

open Verity
open scoped ENNReal

/-- **The read bound**: a declaration that contains the reference training algorithm `r` (`M`) and complies at `(Γ, α)`
(`CompliesRate`), with its registered information free of the reference training algorithm's fresh input (`ThetaRegistered`), has
at least `Reads_Γ(r) / α` gates. Lower bounds on the fresh input a computation reads become lower bounds on size when the
declared circuit computes `r` the way `r` does. -/
def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (r d : RegisteredCircuit) (M : Containment r.D d.D), CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    (Reads Γ r : ℝ) ≤ α * d.D.size d.D.computed

end InferenceOnly.Guarantees
