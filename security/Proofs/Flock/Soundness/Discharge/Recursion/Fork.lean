module

import Proofs.Flock.Soundness.Game
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.Fork`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR

open scoped ENNReal

/-- **Openings of a binding commitment**, as the stage binding reads them: an opening `o` opens the digest `com o` to the
value `val o`, and two openings of one digest to different values are a collision of `H` (`collide`), as
`ValueBindingZ`'s are. -/
structure Openings {D : Type} (H : List UInt8 → D) (V : Type) : Type 1 where
  Op : Type
  Dg : Type
  com : Op → Dg
  val : Op → V
  collide : Op → Op → List UInt8 × List UInt8
  collide_spec : ∀ o o', com o = com o' → val o ≠ val o' →
    (collide o o').1 ≠ (collide o o').2 ∧ H (collide o o').1 = H (collide o o').2

end FlockSoundness.Discharge.Recursion
