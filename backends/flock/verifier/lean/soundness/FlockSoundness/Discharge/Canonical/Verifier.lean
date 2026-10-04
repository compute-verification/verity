import FlockSoundness.Discharge.Canonical.Format

set_option Elab.async false

/-!
# The verifier's obligation: `HmOut.parse` accepts canonical circuits only

`parse_formatOk` is the one statement about the executable verifier the canonical headline (`Headline.lean`) adds: an
untyped circuit file `HmOut.parse` accepts is in the canonical format (`formatOk`).

**It is false at main.** Main's `HmOut.parse` accepts `sha512/row/v1` rows, `sha512/row-seg/v1` rows, zero-padded
output rows, and a tail's circuit with any `vus_per_block`. It becomes true with verifier change V1 (`PROTOCOL.md`
§16.15): `HmOut.parse` refuses, after `HmOut.checkRows`, every untyped circuit `formatOk` refuses, with one `throw` per
conjunct, and `flock-verify`'s Rust twin refuses the same.

**Proof work, after V1.** Mechanical: unfold `HmOut.parse` to the new check (`ite_throw_cond`, as `parse_untyped`
reads `checkRows`), one conjunct per `throw`.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock

/-- **Every untyped circuit `HmOut.parse` accepts is canonical** (after V1; false at main). -/
theorem parse_formatOk {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (ht : tags.typed = false) (h : HmOut.parse tags cf tables = .ok c) : formatOk c = true := by
  sorry

end FlockSoundness.Discharge.Canonical
