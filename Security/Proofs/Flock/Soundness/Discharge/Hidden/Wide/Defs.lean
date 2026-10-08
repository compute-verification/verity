import Proofs.Flock.Soundness.Discharge.Hidden.Defs

/-!
# `OutputsCommitted` and `OutputsBinding`, widened: the frozen statements

`Defs.lean`'s two statements cover a circuit whose unit returns the outputs (`outNet = unitNet`), `sha512/row/v1` rows and
a population file without shared rows. These cover more of the sessions the prover writes, each with the conclusion of
the one before it, read at the same `outBytes` (the output words where the circuit computes them, `HmOut.outCol`: the
unit, or a tail's last stage):

* `…Tail`: a tail that returns the outputs too (GEMM, attention: META `out_net` a stage). The circuit has a unit per
  template instance at least (`0 < upv`), in place of `outNet = unitNet`;
* `…V2`: `sha512/row/v2` rows too (META `row_schemas`), every row's prefix its own (`HmRow.prefixOf`), for output rows
  one per output port, each of whole SHA-512 blocks (a run of ports in one row, or an exact-length row whose last block
  is partial and so in its `hm96` slot, is not covered);
* `…Wide`: a population file with shared rows too (header `shared_rows`, `pop.tables ≠ none`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

/-! ## A tail that returns the outputs -/

/-- **`OutputsCommittedTail`**: `OutputsCommitted` for a circuit whose unit or tail returns the outputs, with a unit per
template instance at least (`0 < upv`). In a drawn statement `Stmt.setupHidden` accepts, in every satisfying witness of
its block R1CS at regions containing the executable's, at block `o` and template instance `g` (below `2^⌊log₂ g⌋`),
output row `j`'s `b ‖ c` in the registered population's public file, at the unit the template instance proves, is the
commit string of the row prefix and the output words the template instance computes (`outBytes`), under the salt the
witness holds at the row's `hm96` slot. For a population file without shared rows, a draw whose proved units are in the
population, and `sha512/row/v1` rows (`hv1`).
-/
def OutputsCommittedTail : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
          outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`OutputsBindingTail`** (under `cr/sha-512`): `OutputsBinding` under `OutputsCommittedTail`'s conditions. Two drawn
statements `Stmt.setupHidden` accepts on one circuit file, public file and tables, and satisfying witnesses of their
block R1CS, that put one registered unit at block `o`, template instance `g` and at block `o'`, template instance `g'`:
when the output words the two template instances compute for output row `j` differ, `commitPair` of their rows' bytes
and salts is a SHA-512 collision. -/
def OutputsBindingTail : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
        outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.rowPrefix (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words).data.toList ++
        outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

/-! ## `sha512/row/v2` rows -/

/-- **`OutputsCommittedV2`**: `OutputsCommittedTail` for `sha512/row/v1` or `sha512/row/v2` rows, the row prefix
the row's own (`HmRow.prefixOf`: `sha512/row/v1`'s for its words, or a bit row's `P(bits)`). For unsegmented rows (no
`segs`) whose output rows are one per output port, each of whole SHA-512 blocks (a multiple of 64 words), a population
file without shared rows and a draw whose proved units are in the population. -/
def OutputsCommittedV2 : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`OutputsBindingV2`** (under `cr/sha-512`): `OutputsBindingTail` under `OutputsCommittedV2`'s conditions, each
row's prefix its own (`HmRow.prefixOf`). -/
def OutputsBindingV2 : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

/-! ## Shared rows -/

/-- **`OutputsCommittedWide`**: `OutputsCommittedV2` for a population file with or without shared rows (header
`shared_rows`: an instance's input row a row of its port's table, `Public.digest`), for a draw whose proved units are in
the population. -/
def OutputsCommittedWide : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`OutputsBindingWide`** (under `cr/sha-512`): `OutputsBindingV2` under `OutputsCommittedWide`'s conditions. -/
def OutputsBindingWide : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    0 < st.c.upv →
    (∀ p ∈ st.c.ports.toList, p.segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

end FlockSoundness.Discharge.Hidden
