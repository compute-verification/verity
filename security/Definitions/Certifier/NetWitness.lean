module

public import Definitions.Compose.JointPeriod
public import Definitions.Pouw.PouwWitness

/-! The period run the network attester's accounting bridge fails at, which only the satisfiability witnesses use. -/

@[expose] public section

namespace ResourceAccounting.NetWitness

open PouwWitness

/-- The circuit with no gate. -/
def emptyCircuit : Verity.Circuit where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅

/-- A period run on `nullDevice` over `emptyCircuit`, idle, whose every audit accepts at every coin and at which the
network attester's claim never holds. -/
def acceptAllNet : PeriodRun nullDevice Unit emptyCircuit (fun _ : Unit => Unit) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own _ := ()
  acct _ := ()
  win := ⟨0, 1⟩
  schedule _ _ := idleStep
  fits _ := by
    intro t
    simp [Step.Fits, idleStep, nullDevice]
  trace _ g := g.elim0
  commWithin _ _ := False
  accepts _ _ := True

end ResourceAccounting.NetWitness
