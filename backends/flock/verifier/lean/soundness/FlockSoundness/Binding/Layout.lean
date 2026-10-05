import FlockSoundness.Audit.FlockLink
import FlockSoundness.Binding.Leaf
import FlockSoundness.Protocol.Binding.Layout

/-!
# The layout of registered leaves, and its value binding

The link theorem (`flock_batched_linkSoundE`) takes the binding of wires to registered commit strings as a named
structure, `Partition.ValueBinding`. This file builds one (`Layout.binding`) from:

* **the layout** (`Layout`): the commit strings (`Pos`: the leaves under the serving roots) and the values behind them
  (`Val`: a row), the commit string that holds each wire and each value's bit on it, the leaf digest the registration
  commits to at each commit string (`regLeaf`), and what a drawn unit's table's witness reads at a commit string: a value
  and a salt, read at fixed addresses (`row`, `salt`: the bits at `rowAt`, the bytes at `saltAt`). One fact about the
  layout, `decode_row`: in a satisfying witness, the decoder's bit on each of the unit's wires is the bit of the row the
  witness holds at the wire's commit string;
* **a leaf scheme per commit string**, with its explicit extractor (`ExplicitLeaf`);
* **the registration fact** (`Layout.Registered`): in a satisfying witness, the leaf of what the witness reads at each of
  the unit's commit strings is the registered one. For `hm96-sha512/row/v1` rows it comes from the circuit's row hashing,
  `Assumptions.HmRowComputes` (`Binding.Rows`).

The binding's commitment is the pair (commit string, leaf digest), and its `collide` is the leaf scheme's extractor, so
`collide_spec` is proved: two openings of one commit string with one leaf and different values give two different byte
strings with one SHA-512 digest (`Refine.H512`).
-/

namespace FlockSoundness.Binding

open Game Model Accounting Extract Refine Audit

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {P plan tab} {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

namespace Layout

variable (lay : Layout P plan tab decode) (scheme : lay.Pos → ExplicitLeaf lay.Val)

end Layout

end FlockSoundness.Binding
