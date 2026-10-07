import Definitions.Core.TC.Spec
import Definitions.Core.TC.Relation

/-!
# A semantics value as a relation state

The one place the two independent transcriptions meet: `stateOf` is the relation's canonical state of a value of the
semantics, and `Admissible` the values a group takes as its accumulator. `pack_sound_complete` is stated over both.
-/

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
