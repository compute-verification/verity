import FlockSoundness.ExecDelta
import FlockLevel3.Placed
import FlockSoundness.Protocol.ExecPlacement

/-!
# An instance's rows, placed by the statement the verifier accepted

The verifier's level-0 matrices are `placedA st` and `placedB st` (flock-verifier's `FlockLevel3.Placed`, PR #156): column
`j`'s entries come from the last range covering `j`, rows in the same slot only, plus Δ's pairs. This file proves
`Placement` for an instance's stacked rows (`Rows.stack`, `placement_stack`) from the layout facts `Layout` names:
- **the ranges:** their spans are pairwise disjoint (`HmRow.check`'s pairwise test), a net range's slots are its net's
  `2^unitLog` rows, and every net's constant and input rows fit its slot;
- **the unit range:** `vus_per_block × units_per_vu` slots of the unit net, a net `Net.ofRows` built (a text net, or a
  typed flat class's `Typed.netOfD`) whose constant row reads itself and whose input rows end before its computed rows;
- **the pin:** the first slot's constant of a range with a slot, whose net's constant row reads itself;
- **Δ:** every net slot's constant other than the pin gets `(k, k)` and `(k, pin)`, and every other pair's row is an input
  bit of a net slot (`HmRow.delta`, `delta_split`).

Then at each unit copy's computed rows the matrices are the net's rows shifted to the slot, and Δ adds nothing there; at
each copy's constant row they read the pin; and so does the pin's row. `ExecSetup.lean` derives `Layout` from
`Stmt.setupH`.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Spans and covers -/

/-! ## The last range covering a column -/

/-! ## Counting -/

/-! ## `Sparse`'s entries are its rows' counts -/

/-! ## Δ's first loop, and input bits -/

/-! ## The layout facts -/

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

end Layout

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

end Layout

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

/-- **A drawn unit's place in the accepted statement** (#145's `UnitPlace`, which `loweringSoundB_of_place` and #171's
`_placed` forms take): VU `g`'s stacked unit slots in block `o`, given the unit's instance in the audit circuit
(`inst`, the circuit side), whose inputs on one gate carry one value there (`hal`). -/
noncomputable def unitPlace (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n} (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv))
    (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) (hal : L.Aliased regs inst o hg) :
    UnitPlace P (L.model regs) u where
  R := (unitRows st L.unit_built).stack st.c.upv
  inst := inst
  o := o
  col := stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
    (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg)
  placed := L.placement regs hg
  aliased := hal

end Layout

end FlockSoundness
