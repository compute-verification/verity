module

public import Definitions.InferenceOnly.Assumptions
public import Definitions.Slowdown.Threads
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.InferenceOnly

/-! The inference-only policy's lemmas: steps of proofs, which no record of the lock pins. -/

@[expose] public section

open Slowdown

namespace InferenceOnly.Guarantees

open Verity
open scoped ENNReal

def ReadBound : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (r d : RegisteredCircuit) (M : Containment r.D d.D), CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    (Reads Γ r : ℝ) ≤ α * d.D.size d.D.computed

end InferenceOnly.Guarantees
