import Proofs.Accounting.Tag.Basic
import Proofs.Accounting.Tag.PolyHash
import Proofs.Accounting.Tag.PartialRead

/-!
# The block tags' guarantees

Each statement of `Models/Accounting/Guarantees/Tag.lean`, proved:

* `Tag/Basic.lean`: `p = 2^31 − 1` is prime, uniform probabilities as counts, the sparse root count;
* `Tag/PolyHash.lean`: the polynomial hash's non-collision, unpredictability, fold, digest and Horner loop;
* `Tag/PartialRead.lean`: a partially read block's unread words are each guessed with probability `2^-32`.
-/

namespace Accounting.SecurityProofs.Tag

theorem PolyNonCollision : Accounting.Guarantees.Tag.PolyNonCollision := Accounting.TagProofs.polyNonCollision
theorem PolyNonCollisionKeyed : Accounting.Guarantees.Tag.PolyNonCollisionKeyed :=
  Accounting.TagProofs.polyNonCollisionKeyed
theorem PolyUnpredictable : Accounting.Guarantees.Tag.PolyUnpredictable := Accounting.TagProofs.polyUnpredictable
theorem PartialReadGuess : Accounting.Guarantees.Tag.PartialReadGuess := Accounting.TagProofs.partialReadGuess
theorem FoldNonCollision : Accounting.Guarantees.Tag.FoldNonCollision := Accounting.TagProofs.foldNonCollision
theorem DigestNonCollision : Accounting.Guarantees.Tag.DigestNonCollision := Accounting.TagProofs.digestNonCollision
theorem HornerIsBlockHash : Accounting.Guarantees.Tag.HornerIsBlockHash := Accounting.TagProofs.hornerIsBlockHash

end Accounting.SecurityProofs.Tag
