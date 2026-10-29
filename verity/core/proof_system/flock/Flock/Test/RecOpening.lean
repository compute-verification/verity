/-
`RecOpening.check` on a unit named `rec-opening/m22-c2-p0` (m = 22 has a `fast100` schedule, claims 2): any number of
units per VU but one is refused, before the structure is computed or any net built; at one it goes on to the part's bits
(this unit has no input bits, so its claims exceed them). A unit of another name passes at any number.
-/
import Flock.RecOpening

namespace Flock.Test.RecOpening

def circuit (unit : String) (upv : Nat) : Circuit :=
  { (default : Circuit) with
    name := "t", upv, nets := #[("unit", { (default : Net) with name := unit })], unitNet := 0 }

def refusal (unit : String) (upv : Nat) : Option String :=
  match Flock.RecOpening.check (circuit unit upv) with
  | .error e => some e
  | .ok () => none

def opening : String := "rec-opening/m22-c2-p0"

#guard refusal opening 2 == some s!"circuit t: unit {opening} has 2 units per VU, not 1"
#guard refusal opening 0 == some s!"circuit t: unit {opening} has 0 units per VU, not 1"
#guard refusal opening 1 == some s!"circuit t: unit {opening}'s claims exceed its 0 input bits"
#guard refusal "rec-residuals/m22-k6-c2-p0" 2 == none

end Flock.Test.RecOpening
