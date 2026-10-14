import Proofs.Flock.Soundness.Discharge.Exec.Setup

/-!
# Table `j`'s tags: the identity alone differs, which the circuit and public parses do not read

`tagsAt tags count j` (`Exec/Setup.lean`) is `tags` at one table and `tags` with `Tags.tableIdentity count j` as its
identity at several. `HmRow.parse` and `HmRow.loadPublic` read no identity, so they agree at every table's tags.
-/

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
