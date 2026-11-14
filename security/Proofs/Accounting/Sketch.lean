module

public import Proofs.Accounting.Tag.Signs
public import Proofs.Accounting.Tag.LittlewoodOfford
public import Proofs.Accounting.Tag.SketchExact

/-!
# The exact ±1 sketch's guarantees

Each statement of `Definitions/Accounting/Guarantees/Sketch.lean`, proved:

* `Tag/Signs.lean`: `k` uniform ±1 rows collide on two different integer vectors with probability at most `2^-k`, and
  pass a wrong committed product with probability at most `2^-k`;
* `Tag/SketchExact.lean`: the ±1 sketch on sm_120's E4M3 tensor cores is exact, and so collides as rarely;
* `Tag/LittlewoodOfford.lean`: Erdős's anti-concentration bound for one row under partial reads.

`Tag/Basic.lean` holds what they share: uniform probabilities as counts.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Sketch

theorem NonCollision : Accounting.Guarantees.Sketch.NonCollision := Accounting.TagProofs.nonCollision
theorem SmallCode : Accounting.Guarantees.Sketch.SmallCode := Accounting.TagProofs.smallCode_value
theorem Exact : Accounting.Guarantees.Sketch.Exact := Accounting.TagProofs.sketchExact
theorem TensorCoreNonCollision : Accounting.Guarantees.Sketch.TensorCoreNonCollision :=
  Accounting.TagProofs.tensorCoreNonCollision
theorem CommitThenProject : Accounting.Guarantees.Sketch.CommitThenProject := Accounting.TagProofs.commitThenProject
theorem LittlewoodOfford : Accounting.Guarantees.Sketch.LittlewoodOfford := Accounting.TagProofs.littlewoodOfford

end Accounting.SecurityProofs.Sketch
