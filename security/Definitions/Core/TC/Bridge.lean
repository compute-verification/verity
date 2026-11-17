module

public import Definitions.Core.TC.Spec
public import Definitions.Core.TC.Relation

/-! The one place the two transcriptions meet: the relation's canonical state of a semantics value (`stateOf`). -/

@[expose] public section

namespace Verity.TC.Proofs

open Verity.TC.Spec Verity.TC.Relation

/-- The relation's state of a value: what `decode_state ∘ pack_value` gives on a group's output. -/
def stateOf (Z : Int) : Value → State
  | .nan => ⟨1, 0, 0, Z, 0⟩
  | .inf n => ⟨0, 1, if n then 1 else 0, Z, 0⟩
  | .fin t => if t.magnitude = 0 then ⟨0, 0, 0, Z, 0⟩
      else ⟨0, 0, if t.negative then 1 else 0, t.exponent, t.magnitude⟩

/-- The values a group takes as its accumulator: an FP32 word's value, or a group's output. -/
def Admissible : Value → Prop
  | .fin t => t.magnitude < 2 ^ 24 ∧
      (t.magnitude ≠ 0 → -126 ≤ t.exponent ∧ t.exponent ≤ 127 ∧ (2 ^ 23 ≤ t.magnitude ∨ t.exponent = -126))
  | _ => True

end Verity.TC.Proofs
