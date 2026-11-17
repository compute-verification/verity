module

public import Proofs.Poub.Tag.Signs
public import Proofs.Poub.Tag.LittlewoodOfford
public import Proofs.Poub.Tag.SketchExact

/-! The exact ±1 sketch's guarantees, each statement of `Definitions/Poub/Guarantees/Sketch.lean` proved in a file
under `Tag/`. -/

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
