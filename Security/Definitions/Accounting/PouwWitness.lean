module

public import Definitions.Accounting.PerRun

/-!
# The run PoUW's compute witnesses fail at (trusted definitions; used only by the satisfiability witnesses)

`acceptAllRun`: on a device with no capacity (`nullDevice`) and an idle schedule (`idleStep`), a run whose every audit
accepts. It does no work, so its compute audit accepts short of any positive certificate: what
`Accounting.Guarantees.ComputeOfPouw` and `ComputeOfPouwPerRun` rule out.
-/

@[expose] public section

namespace Accounting.PouwWitness

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

end Accounting.PouwWitness
