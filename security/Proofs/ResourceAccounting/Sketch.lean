module

public import Proofs.Poub.Tag.Signs
public import Proofs.Poub.Tag.LittlewoodOfford
public import Proofs.Poub.Tag.SketchExact

/-!
# The exact ±1 sketch's guarantees

Each statement of `Definitions/Poub/Guarantees/Sketch.lean`, proved:

* `Tag/Signs.lean`: `k` uniform ±1 rows collide on two different integer vectors with probability at most `2^-k`, and
  pass a wrong committed product with probability at most `2^-k`;
* `Tag/SketchExact.lean`: the ±1 sketch on sm_120's E4M3 tensor cores is exact, and so collides as rarely;
* `Tag/LittlewoodOfford.lean`: Erdős's anti-concentration bound for one row under partial reads.

`Tag/Basic.lean` holds what they share: uniform probabilities as counts.
-/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Sketch

theorem NonCollision : ResourceAccounting.Guarantees.Sketch.NonCollision := ResourceAccounting.TagProofs.nonCollision
theorem SmallCode : ResourceAccounting.Guarantees.Sketch.SmallCode := ResourceAccounting.TagProofs.smallCode_value
theorem Exact : ResourceAccounting.Guarantees.Sketch.Exact := ResourceAccounting.TagProofs.sketchExact
theorem TensorCoreNonCollision : ResourceAccounting.Guarantees.Sketch.TensorCoreNonCollision :=
  ResourceAccounting.TagProofs.tensorCoreNonCollision
theorem CommitThenProject : ResourceAccounting.Guarantees.Sketch.CommitThenProject := ResourceAccounting.TagProofs.commitThenProject
theorem LittlewoodOfford : ResourceAccounting.Guarantees.Sketch.LittlewoodOfford := ResourceAccounting.TagProofs.littlewoodOfford

end ResourceAccounting.SecurityProofs.Sketch
