import FlockSoundness.Discharge.ScopeW.Chain
import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.Hidden.Rows

set_option Elab.async false

/-!
# A row's value and the bytes its digest hashes

A drawn unit's row value is a layout value (`Layout.widthOf`, `Layout.padBits`) of a row's `16 · words` bits. Its
encoding under the row's schema (`rowEncOf`) is the message the row's `hm96` slot commits to (`rowMsgOf`) of the row's
bytes:

* `rowPorts_bits`, `parse_bits`: a bit row's length is below `2^64` (the parse bounds its compressions by
  `HmRow.MAX_COMPS`);
* `bitShaped_row`: a layout value of fewer than `2^64` bits is a bit row's (`BitShaped`);
* `rowEncOf_row`: at a row of a widened shape (`rowShapeW`), `rowEncOf` of the value of bits `b` is `rowMsgOf` of `b`'s
  bytes.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3

/-- **`HmRow.rowPorts`' bit rows are of fewer than `2^64` bits**: their compressions are at most `HmRow.MAX_COMPS`. -/
theorem rowPorts_bits {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = true → p.bits < 2 ^ 64 := by
  sorry

/-- `rowPorts_bits` at a parsed circuit's rows. -/
theorem parse_bits {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : ∀ p ∈ c.ports.toList, p.v2 = true → p.bits < 2 ^ 64 := by
  sorry

/-- **A layout value of fewer than `2^64` bits is a bit row's.** -/
theorem bitShaped_row {W K : ℕ} (hKW : K ≤ W) (hK : K < 2 ^ 64) (b : Fin K → Bool) :
    BitShaped (Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) := by
  sorry

/-- **A row value's encoding is its row's message**: at a row of a widened shape (`rowShapeW`), whose bit row is of
fewer than `2^64` bits, `rowEncOf` of the layout value of `16 · words` bits `b` is `rowMsgOf` of `b`'s `2 · words`
bytes. -/
theorem rowEncOf_row {p : Port} (hs : rowShapeW p = true) (hb : p.v2 = true → p.bits < 2 ^ 64) {W K : ℕ}
    (hK : K = 16 * p.words) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowEncOf p (Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) =
      rowMsgOf p (List.ofFn fun j : Fin (2 * p.words) => Binding.byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩) := by
  sorry

end FlockSoundness.Discharge.ScopeW
