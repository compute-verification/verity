import FlockSoundness.Binding.Rows
import FlockSoundness.Discharge.Layout.Readers
import FlockSoundness.Protocol.Discharge.Layout.Pins
import FlockSoundness.Audit.FlockLinked

/-!
# A layout and its rows from three pinning facts

`Prog.flock_headline` takes the layout's facts as `lay : Binding.Layout` and `rs : Binding.HmRows lay`. This file builds
both from fixed readers (`Readers`: each commit string's row, salt and `b ‖ c` addresses in a drawn unit's table, which
commit strings are `hm96-sha512/row/v1` rows, and what the registration holds at each: the public `b ‖ c` of a row, the
value of any other commit string) and three facts about every satisfying witness of a drawn unit's table:

* `Readers.Copies`: each column of the unit on a committed wire holds the bit the witness has at the wire's row address
  (Δ's copy into the row's message bit for an input; the column itself for an output);
* `Readers.BcPinned`: at a row, the 128 bytes at `bcAt` are the registration's `b ‖ c` (the statement's regions pin the
  `hm96` slots' outputs to the public `b ‖ c`);
* `Readers.ValPinned`: at another commit string, the bits at `rowAt` are the registration's value.

The registered leaf (`Readers.regLeaf`) is `SHA-512(leaf_prefix ‖ b ‖ c)` of the registration's `b ‖ c` at a row, the
leaf the verifier computes from the public `b ‖ c` (`HmRow.checkPublic`), and `SHA-512(enc v)` of the registration's
value elsewhere. That the serving roots commit to these leaves is the serving roots' rewinding (`StrictCR.TreeRewind`),
paid once in the headline as `δ_tree`; nothing here is cryptographic.
-/

namespace FlockSoundness.Discharge.Layout

open Game Model Audit Binding

variable {C : Circuit} {n : ℕ} {Reg : Type} {mPts : ℕ}

namespace Readers

variable {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (rd : Readers C plan tab) {P : Audit.Partition C n}
  (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

/-- The layout's commit string of a wire is the readers'. -/
theorem layout_pos (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (g : Fin C.N) :
    (rd.layout place encO hcopy).pos g = rd.pos g := rfl

/-- The layout's registered leaf is the readers'. -/
theorem layout_regLeaf (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (R : Reg) (q : rd.Pos) :
    (rd.layout place encO hcopy).regLeaf R q = rd.regLeaf encO R q := rfl

/-- **The value a message holds at a commit string** is the readers' `valAt`, at the commit string's width. -/
theorem layout_row (encO : BitVal rd.W → List UInt8) (hcopy : rd.Copies place) (S : Finset (Fin n)) (R : Reg)
    (u : Fin n) (M : Binding.Msg plan tab S R u) (q : rd.Pos) :
    (rd.layout place encO hcopy).row S R u M q = (widthOf rd.W (rd.rowBits q), rd.valAt S R u M q) := rfl

end Readers

end FlockSoundness.Discharge.Layout
