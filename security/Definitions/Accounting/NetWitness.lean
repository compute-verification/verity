module

public import Definitions.Accounting.JointPeriod
public import Definitions.Accounting.PouwWitness

/-!
# The period run the network certifier's accounting bridge fails at (trusted definitions; used only by the
satisfiability witnesses)

`acceptAllNet`: on a device with no capacity (`PouwWitness.nullDevice`), over the circuit with no gate
(`emptyCircuit`) and an idle schedule (`PouwWitness.idleStep`), a period run whose every audit accepts at every coin
while the network certifier's claim never holds: what `Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted` rules
out.
-/

@[expose] public section

namespace Accounting.NetWitness

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
network certifier's claim never holds. -/
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

end Accounting.NetWitness
