module

public import Definitions.Nci.Guarantees.Activations

/-!
# NCI's activations bound over all of training ([ACTIVATIONS-BOUND]): lemmas

The statements of `Definitions.Nci.Guarantees.Activations` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity

/-- **The premise over all counted multiplications gives the chain's**: where a chain `ch` is part of the threading
(`ChainIn`), `ActivationsIncompressibleAll` gives `ActivationsIncompressible` for `ch` at the same caps, for nonnegative
rates: a chain's weight entries and input entries are distinct weight and activation gates, and an input entry whose
activation is whole in `R` is linked to an entry whole in `R`. -/
def ActivationsChain : Prop :=
  ∀ (r : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (L : T.Links) (c : ℕ) (k b_w b_a : ℝ),
    ChainIn ch T L → 0 ≤ b_w → 0 ≤ b_a →
    Assumptions.ActivationsIncompressibleAll r T L c k b_w b_a →
      Assumptions.ActivationsIncompressible r ch c k b_w b_a

end Nci.Guarantees
