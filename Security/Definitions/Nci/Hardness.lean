import Definitions.Nci

/-!
# Priced reads: the vocabulary

Daniel's one schema for the hardness premises (26 Oct): a set of at most `c` operation gates is handed, besides the
registered inputs, at least the price of the operand values of the counted operations it computes whole, less a deficit.
The premise is `Assumptions.Priced`; this is what it reads.

* `Operations`: counted operations, each with its gates and the operand values it reads.
* `Operations.whole`, `Operations.ops`: the operations a set computes whole, and their operand values `Ops(S)`.
* `Operations.ofActs`: operations that each read one activation occurrence, the fan-out instance's.
* `Operations.WholeIn`, `Operations.FanOut`: every counted operation lies whole in one block of a partition, and each
  value is an operand of at most `e` of them.
-/

namespace Nci.Hardness

open Verity

/-- **Counted operations** of `C`: the operations `J`, each with its gates `gates p` and the operand values
`operands p` it reads, values of a type `V`. A value is a label, not a gate: an updated parameter that many products
use is one value, and each activation occurrence is its own. -/
structure Operations (C : Circuit) (ι V : Type*) where
  J : Finset ι
  gates : ι → Finset (Fin C.N)
  operands : ι → Finset V

namespace Operations

variable {C : Circuit} {ι V : Type*} (O : Operations C ι V)

/-- The counted operations that `S` computes whole: all of their gates are in `S`. -/
def whole (S : Finset (Fin C.N)) : Finset ι :=
  O.J.filter fun p => O.gates p ⊆ S

/-- `Ops(S)`: the operand values of the counted operations `S` computes whole, each once. -/
def ops [DecidableEq V] (S : Finset (Fin C.N)) : Finset V :=
  (O.whole S).biUnion O.operands

/-- Every counted operation lies whole in one block of `P`. -/
def WholeIn {n : ℕ} (P : Partition C n) : Prop :=
  ∀ p ∈ O.J, ∃ u, O.gates p ⊆ P.gates u

/-- **Fan-out at most `e`**: each value is an operand of at most `e` counted operations. -/
def FanOut [DecidableEq V] (e : ℕ) : Prop :=
  ∀ v, (O.J.filter fun p => v ∈ O.operands p).card ≤ e

end Operations

/-- **Operations of `C` that each read one activation occurrence** `act p`, the fan-out instance's priced operand. A
product's parameter, priced `0` there, adds nothing to the price and is left out. -/
def Operations.ofActs (C : Circuit) {ι V : Type*} (J : Finset ι) (gates : ι → Finset (Fin C.N)) (act : ι → V) :
    Operations C ι V :=
  ⟨J, gates, fun p => {act p}⟩

end Nci.Hardness
