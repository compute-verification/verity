module

public import Definitions.Accounting.Guarantees.ServingBlocks.Witnesses

/-!
# The serving-block bound: honest and nontrivial lemmas: lemmas

The statements of `Definitions.Accounting.Guarantees.ServingBlocks.Witnesses` that are steps of its guarantees' proofs
rather than guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem
of the same name, and audited with the proofs: no record pins it.
-/

@[expose] public section

namespace Accounting.Guarantees.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- **The elementary uncertified cap's premise holds** at `Honest`: its serving blocks and dark remainder refine
themselves. -/
def UncertifiedElementary_honest : Prop :=
  Nonempty (Honest.P.Refines Honest.P)

/-- **The elementary uncertified cap is false without refinement**: at `Counter`, the parent map `away` sends the one
inference block, which holds the counted operation's gate, to the unit of `P₂` that holds no gate, so it is no
refinement, and the operation's weight `1`, in `P₂`'s first unit with no inference block mapped there, is above the cap
`split = 0`. -/
def UncertifiedElementary_nontrivial : Prop :=
  ¬ (∀ g, Counter.P₂.unit g = (Counter.P.unit g).map Counter.away) ∧
  ¬ UncertifiedCap Counter.J (Counter.P₂.gates 0) (fun i : {i // Counter.away i = 0} => Counter.P.gates i)
    (Counter.J.split (Counter.P₂.gates 0) (fun i : {i // Counter.away i = 0} => Counter.P.gates i))

end Accounting.Guarantees.ServingBlocks
