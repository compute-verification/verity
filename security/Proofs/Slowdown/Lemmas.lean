module

public import Definitions.InferenceOnly.Assumptions
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.InferenceOnly
public import Definitions.Slowdown.Factor

/-! The slowdown bound's lemmas: steps of proofs, each a `def G : Prop` proved as `Slowdown.SecurityProofs.G` and
pinned by no record. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity
open scoped ENNReal

def Inference : Prop :=
  ∀ (c k : ℕ) (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)),
    d.D.Boolean → Assumptions.Sessions d P prompt c k → Complies c k d ∧ slowdown (Complies c k) d ≤ 1

end Slowdown.Guarantees
