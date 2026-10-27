/-
`Flock.Test.ToyPubFiles`' program descriptor and template query, as a registration names them: the descriptor parses
(`Program.parse`), and its layout under the query is defined with one replay unit, unit 0 (`Slots.popOk … 1`), which is
what `Bridge.PopMatches` states of a registration of one unit.
-/
import SampledProofs.Slots
import Flock.Test.ToyPubFiles

namespace SampledProofs.ToyPop

open Flock.Test.ToyPubFiles

#guard match Program.parse ⟨progDesc.data.toList.toArray⟩ with
  | .ok p => Slots.popOk p progQuery 1
  | .error _ => false

end SampledProofs.ToyPop
