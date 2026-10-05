import FlockSoundness.Discharge.Aliased.Rows
import FlockSoundness.Protocol.Discharge.Aliased.Copies
import FlockSoundness.ExecFlatCopies

/-!
# `Layout.Aliased` from Δ's copy positions (`pp.aliased`)

At a unit slot's input port bit the block rows are the net's self row and Δ's one copy pair (`Layout.placed_port`, from
`HmRow.check`'s exactly-once wiring), so they copy the column Δ copies there (`flatSrcAt`), or are zero when that
column lies outside the block. Each unit slot's row past its useful rows is a forced-zero row (`Layout.placed_zero`):
the net's row there is empty (`parse_rows`), no slot constant is there, and Δ's other pairs are at input bits. So in
every witness satisfying the accepted statement:
- `copyPos_block`: each of VU `g`'s stacked unit inputs carries what its copy position (`copyPos`) carries;
- `zeroPos_block`: each unit slot's forced-zero row (`zeroPos`) carries 0. Δ copies these into zero leaves (the slot's
  own) and into cut bits past a row word (the range's first slot's).

`Layout.Aliased` quantifies over any instance of the unit's rows in any audit circuit, so it is not a fact of the
statement alone: two inputs an instance puts on one gate can copy different sources. `aliased_of_copies` derives it from
`Copies`, which relates the instance's wiring to the copy positions: inputs on one gate copy one position, or each copies
a forced-zero row. `setupH_aliased` and `setupH_unitPlace` instantiate it at the untyped statement `Stmt.setupH`
accepted.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : UnitFacts tags st)

end copies

/-! ## At the untyped statement `Stmt.setupH` accepted -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A drawn unit's place, from the executable's accept step** (`unitPlace_of_setupH` with `setupH_aliased`): VU `g`'s
stacked unit slots in block `o`, for an instance that wires their inputs as Δ copies them. -/
noncomputable def setupH_unitPlace (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies (setupH_layout ht h) regs (setupH_unitFacts ht h) inst hg) :
    UnitPlace P ((setupH_layout ht h).model regs) u :=
  unitPlace_of_setupH ht h regs inst o hg (setupH_aliased ht h regs inst o hg hcp)

end FlockSoundness.Discharge.Aliased
