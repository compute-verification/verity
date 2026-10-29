/-
`RecOpen.check` on one of R's levels, a unit named `rec-open-stacked/l<LANES>-h<H>`: refused as a `rec-open/` unit is,
before any net is built, for its bits (this unit has none) at heights `RecOpen_v4` builds and for a `LANES` or `H` it does
not; refused by name in any other form; a unit of another name passes. The honest unit and one a row off are read by the
built verifier (`test_rec_outer.py`).
-/
import Flock.RecOpen

namespace Flock.Test.RecOpen

def circuit (unit : String) : Circuit :=
  { (default : Circuit) with name := "t", nets := #[("unit", { (default : Net) with name := unit })], unitNet := 0 }

def refusal (unit : String) : Option String :=
  match Flock.RecOpen.check (circuit unit) with
  | .error e => some e
  | .ok () => none

def bits (unit : String) (opening : Nat) : Option String :=
  some s!"circuit t: unit {unit} has 0 input and 0 output bits, not rec-open's {opening} and 2048"

def height : Option String := some "circuit t: rec-open's LANES is a positive multiple of 8 and H is 0 to 32"

#guard refusal "rec-open-stacked/l8-h0" == bits "rec-open-stacked/l8-h0" 6144
#guard refusal "rec-open-stacked/l8-h17" == bits "rec-open-stacked/l8-h17" 14336
#guard refusal "rec-open/l8-h0" == bits "rec-open/l8-h0" 6144
#guard refusal "rec-open-stacked/l8-h33" == height && refusal "rec-open-stacked/l12-h0" == height
#guard refusal "rec-open-stacked/l8-h01" ==
  some "circuit t: unit rec-open-stacked/l8-h01 is not rec-open-stacked/l<LANES>-h<H>"
#guard refusal "rec-open-stacked/l8" == some "circuit t: unit rec-open-stacked/l8 is not rec-open-stacked/l<LANES>-h<H>"
#guard refusal "rec-open-stacked" == none && refusal "rec-opening/m22-c2-p0" == none

end Flock.Test.RecOpen
