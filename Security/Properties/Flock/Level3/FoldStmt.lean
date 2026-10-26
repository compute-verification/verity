import Flock.Circuit
import Mathlib.Data.Nat.Notation

/-!
Which columns a range of the lincheck fold covers (`covers`). The fold's facts (`fold_get`, `baseOf_get`) are
`Proofs.Flock.Level3.FoldStmt`.
-/

open Flock

namespace FlockLevel3

/-! ### The ranges, with the slot-type cache -/

/-- `c` lies in one of `r`'s slots. -/
def covers (r : Range) (c : ℕ) : Prop := r.pos0 ≤ c / 2 ^ r.slotLog ∧ c / 2 ^ r.slotLog < r.pos0 + r.count

instance instDecidableCovers (r : Range) (c : ℕ) : Decidable (covers r c) := inferInstanceAs (Decidable (_ ∧ _))

end FlockLevel3
