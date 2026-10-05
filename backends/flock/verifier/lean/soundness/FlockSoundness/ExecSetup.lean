import FlockSoundness.ExecCircuit
import FlockSoundness.ExecCheck
import FlockSoundness.ExecLookup
import FlockSoundness.Protocol.ExecSetup
import FlockSoundness.ExecPlacement

open Flock FlockLevel3

namespace FlockSoundness


/-! ## Nets, ranges and the pin -/

/-! ## Port lists and the rows' slots -/

/-! ## Every net of an accepted circuit -/

/-! ## Δ's later pairs are input bits -/


/-! ## The accepted statement's layout -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A drawn unit's place, from the executable's accept step** (#145's `UnitPlace`, which `loweringSoundB_of_place` and
#171's `_placed` forms take): VU `g`'s stacked unit slots in block `o` of the accepted statement, given the unit's
instance in the audit circuit, whose inputs on one gate carry one value there (`Layout.Aliased`). -/
noncomputable def unitPlace_of_setupH (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hal : (setupH_layout ht h).Aliased regs inst o hg) :
    UnitPlace P ((setupH_layout ht h).model regs) u :=
  (setupH_layout ht h).unitPlace regs inst o hg hal

end FlockSoundness
