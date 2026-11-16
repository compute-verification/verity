module

public import Definitions.InferenceOnly.Assumptions
public import Definitions.Core.Circuit
public import Definitions.Core.RegisteredCircuit
public import Definitions.Core.Partition
public import Definitions.InferenceOnly
public import Definitions.Slowdown.Factor

/-!
# The slowdown bound's lemmas

These are steps of proofs, not guarantees: they sit with the proofs, and no record of the lock pins them.

Each is a `def G : Prop` over the policy's definitions and the assumptions, proved as
`Slowdown.SecurityProofs.G : Slowdown.Guarantees.G`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity
open scoped ENNReal

/-- **Inference complies as declared**: a Boolean declaration divided into sessions of exactly `c` gates, padding
included, each reading from outside itself only registered information and its own prompt `prompt u` of at most `k`
bits, complies at `(c, k)`, and its slowdown factor is at most 1. -/
def Inference : Prop :=
  ∀ (c k : ℕ) (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (prompt : Fin B → Finset (Fin d.D.N)),
    d.D.Boolean → Assumptions.Sessions d P prompt c k → Complies c k d ∧ slowdown (Complies c k) d ≤ 1

end Slowdown.Guarantees
