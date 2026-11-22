module

public import PouwPearlPlus.Load
public import PouwPearlPlus.Check
meta import PouwPearlPlus.Load
meta import PouwPearlPlus.Check

/-!
# One guard per refusal code Pearl+'s check makes

Each `#guard` builds the smallest input that reaches one of Pearl+'s own refusal codes and asserts the code, through the
function that makes it. The codes it shares with PoUW's check (`manifest`, `coin`, …) are guarded in
`PouwVerifier.Refusals`.
-/

namespace PouwVerifier.PearlPlus.Refusals

open Lean (Json)

/-- A pass's manifest under salt `s` holding calls at `indices`. -/
def pass (indices : List Int) : Json :=
  Json.mkObj [("salt", .str "s"),
    ("matmuls", Json.arr (indices.map fun i => Json.mkObj [("index", Lean.toJson i)]).toArray)]

#guard (passesDistinct [pass [-1]]).map (·.1) == some "call-range"
#guard (passesDistinct [pass [0, 0]]).map (·.1) == some "repeated-call"
#guard (passesPlaced [pass [0], pass [1]] (some [0])).map (·.1) == some "call-place"

/-- A run whose coin record was drawn on another run's commitment than its passes'. -/
def drawnElsewhere : Input where
  coins := .inproc ⟨0, .empty⟩
  manifests := []
  draws := 0
  held := some { commitment := .empty, source := .empty }

#guard (match afterTie drawnElsewhere [] none with
  | .error r => r.refused.map (·.1)
  | .ok _ => none) == some "after-coin"

end PouwVerifier.PearlPlus.Refusals
