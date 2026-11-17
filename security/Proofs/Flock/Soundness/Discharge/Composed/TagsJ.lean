module

public import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.CategoryTheory.Category.Init
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Refine.Walk
import VCVio.Prelude.Core

/-! Table `j`'s tags differ from the session's in the identity alone, which `HmRow.parse` and `HmRow.loadPublic` do not read. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Flock Exec

theorem parse_identity (tags : Tags) (x : Lean.Json) (b : ByteArray) (t : List (String × ByteArray)) :
    HmRow.parse { tags with identity := x } b t = HmRow.parse tags b t := by
  kernel_rfl

theorem parse_tagsAt (tags : Tags) (count j : ℕ) (b : ByteArray) (t : List (String × ByteArray)) :
    HmRow.parse (tagsAt tags count j) b t = HmRow.parse tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · exact parse_identity _ _ _ _

theorem loadPublic_tagsAt (tags : Tags) (count j : ℕ) (c : Flock.Circuit) (b : ByteArray) :
    HmRow.loadPublic (tagsAt tags count j) c b = HmRow.loadPublic tags c b := by
  unfold tagsAt
  split_ifs <;> rfl

end FlockSoundness.Discharge.Composed
