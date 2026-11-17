module

public import Properties.Flock.Circuit.CircuitOf
public import Properties.Flock.Circuit.ProgRows

/-! A unit of the key program as an instance of the circuit's rows. -/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

/-- **Unit `u` of the key program as an instance of its rows**: the circuit's template instance rows, wired as the
program wires them. -/
noncomputable def unitInst (u : Fin n) : (keyPartition D hU n).IsRowsUnit u (Placed.circuitRows D.unroll hU) :=
  (Const.Prog.castW (rows_keyProg D hU n u) (Const.Prog.rowsWireC (Placed.keyProg D.unroll hU n) u)).toIsRowsUnit _

/-- **Unit `u`'s constant gate**: the gate on the stack's constant column, which every copy's constant row reads. -/
noncomputable def constGate (u : Fin n) : Fin (circuitOf D hU n).N :=
  (unitInst D hU n u).wire (Placed.circuitRows D.unroll hU).one

end Flock.Bridge
