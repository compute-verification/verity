import Proofs.Accounting.Tag.Basic
import Proofs.Accounting.Tag.PolyHash
import Proofs.Accounting.Tag.PartialRead
import Proofs.Accounting.Tag.Signs
import Proofs.Accounting.Tag.LittlewoodOfford
import Proofs.Accounting.Tag.SketchExact

/-!
# The block tags' guarantees

Each statement of `Definitions/Accounting/Guarantees/Tag.lean` and `Definitions/Accounting/Guarantees/Sketch.lean`, proved:

* `Tag/Basic.lean`: `p = 2^31 − 1` is prime, uniform probabilities as counts, the sparse root count;
* `Tag/PolyHash.lean`: the polynomial hash's non-collision, unpredictability, fold, digest and Horner loop;
* `Tag/PartialRead.lean`: a partially read block's unread words are each guessed with probability `2^-32`;
* `Tag/Signs.lean`: `k` uniform ±1 rows collide on two different integer vectors with probability at most `2^-k`, and
  pass a wrong committed product with probability at most `2^-k`;
* `Tag/SketchExact.lean`: the ±1 sketch on sm_120's E4M3 tensor cores is exact, and so collides as rarely;
* `Tag/LittlewoodOfford.lean`: Erdős's anti-concentration bound for one row under partial reads.
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

namespace Accounting.SecurityProofs.Sketch

theorem NonCollision : Accounting.Guarantees.Sketch.NonCollision := Accounting.TagProofs.nonCollision
theorem SmallCode : Accounting.Guarantees.Sketch.SmallCode := Accounting.TagProofs.smallCode_value
theorem Exact : Accounting.Guarantees.Sketch.Exact := Accounting.TagProofs.sketchExact
theorem TensorCoreNonCollision : Accounting.Guarantees.Sketch.TensorCoreNonCollision :=
  Accounting.TagProofs.tensorCoreNonCollision
theorem CommitThenProject : Accounting.Guarantees.Sketch.CommitThenProject := Accounting.TagProofs.commitThenProject
theorem LittlewoodOfford : Accounting.Guarantees.Sketch.LittlewoodOfford := Accounting.TagProofs.littlewoodOfford

end Accounting.SecurityProofs.Sketch
