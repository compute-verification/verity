module

public import Definitions.Pouw.PerRun

/-! `acceptAllRun`, a run on a device with no capacity whose every audit accepts, used only by the satisfiability witnesses. -/

@[expose] public section

namespace ResourceAccounting.PouwWitness

/-- A device with no capacity. -/
def nullDevice : Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- The step that charges nothing. -/
def idleStep : Step Unit Unit := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

/-- A run on `nullDevice` whose schedule is idle and whose every audit accepts. -/
def acceptAllRun : Run nullDevice Unit Unit Unit Unit where
  acct _ := ()
  win _ := ⟨0, 0⟩
  schedule _ _ := idleStep
  fits _ := by
    intro t
    simp [Step.Fits, idleStep, nullDevice]
  accepts _ _ := True

end ResourceAccounting.PouwWitness
