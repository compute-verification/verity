import FlockSoundness.Discharge.Canonical.Commit
import FlockSoundness.Discharge.Hidden.Wide.Defs

set_option Elab.async false

/-!
# The rows and outputs a canonical statement commits, at the statement level

`Hidden/Wide/Defs.lean`'s `OutputsCommittedWide` and `OutputsBindingWide` at the canonical format, in place of their
conditions (`0 < upv`, no segments, one output row per output port of whole SHA-512 blocks, `g < 2^⌊log₂ g⌋`), and
`RowsCommittedC` for every row. These read the computed outputs where the circuit computes them (`HmOut.outCol`: the
unit, or a tail's last stage), so they cover what the session headline's `keyProg` (the unit net's rows) does not: a
tail's outputs.

* `RowsCommittedC`: every row `p`'s registered `b ‖ c`, at the unit a VU proves, is the commit string of `P(ℓ_p)` and
  the `ℓ_p` bits the witness holds at the row's message columns (`rowBits`), packed.
* `outBitsC`, `OutputsCommittedC`: output row `j` (a run of output ports, exactly `16 · words` bits) holds the run's
  output words as the circuit computes them.
* `OutputsBindingC` (under `cr/sha-512`): two VUs proving one unit that compute different words for an output row give a
  SHA-512 collision.

**Proof work.**
* `rows_committedC`: moderate, and it is `Commit.lean`'s work restated: `outputs_committed_wide'`'s route (the flat
  twin, `Hidden/Wide/RowCommit.lean`, `RowBc.lean`) with `row_bc_C` for `Hidden.row_bc_tgt` (any length, partial last
  block) and `Hidden.bc_drawnW` for the registered side (shared rows or not; the VU bound from `formatOk_vu_lt`).
* `outputs_committedC`: moderate. `rows_committedC` at `p = nIn + j`, then Δ's output pairs (`Hidden.outPairs`, one per
  output word into its run's row, `Hidden.msg_copy`) rewrite `rowBits` at output row `j` to `outBitsC`; `outRowsExact`
  leaves no forced-zero bit in the row. At a tail the copied column is the last stage's (`HmOut.outCol`'s `outNet`
  branch), which `outputs_committed_tail'` already handles.
* `outputs_bindingC`: small. Both sides equal one registered `b ‖ c` (`outputs_committedC`), the prefixes are equal (one
  circuit file), so `Binding.commitString`'s collision extractor gives `commitPair` unless the packed bits are equal,
  and `FlockLevel3.RowV2.packBits_inj` (equal lengths) turns that into equal `outBitsC`
  (`outputs_binding_wide_of`'s shape).
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-- **Output row `j`'s bits at VU `g`**, under the bits `w`, at a canonical circuit: run `j`'s `16 · words` bits, bit
`16 k + t` bit `t` of the run's word `k` (output word `start + k`) as the circuit computes it (`HmOut.outCol`). -/
def outBitsC (w : ℕ → Bool) (c : Circuit) (g j : ℕ) : List Bool :=
  List.ofFn fun i : Fin (16 * (HmOut.outGroup c j).2.2.2) =>
    w (HmOut.outCol c g ((HmOut.outGroup c j).2.2.1 + i.val / 16) + i.val % 16)

/-- **`RowsCommittedC`**: in a drawn statement `Stmt.setupHidden` accepts on a canonical circuit, in every satisfying
witness of its block R1CS at regions containing the executable's, at block `o` and VU `g`, row `p`'s `b ‖ c` in the
registered population's public file (shared rows or not), at the unit the VU proves, is the commit string of `P(ℓ_p)`
and the row's `ℓ_p` bits at the VU's message columns, packed, under the salt the witness holds at the row's `hm96`
slot. -/
def RowsCommittedC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    formatOk st.c = true →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ}, g < st.c.g → p < st.c.ports.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++
          HmRow.packBits (rowBits (Hm.blockBit (L.model regs) z o) st.c g p))
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g p))

/-- **`OutputsCommittedC`**: `OutputsCommittedWide` at the canonical format. Under `RowsCommittedC`'s conditions,
output row `j`'s `b ‖ c`, at the unit the VU proves, is the commit string of `P(16 · words)` and the run's output words
the VU computes (`outBitsC`), packed. With or without a tail. -/
def OutputsCommittedC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    formatOk st.c = true →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → j < st.c.outRowPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.rowPrefixV2 (16 * (HmOut.outGroup st.c j).2.2.2)).data.toList ++
          HmRow.packBits (outBitsC (Hm.blockBit (L.model regs) z o) st.c g j))
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`OutputsBindingC`** (under `cr/sha-512`): `OutputsBindingWide` at the canonical format. Two drawn statements
`Stmt.setupHidden` accepts on one canonical circuit file, public file and tables, and satisfying witnesses of their
block R1CS that put one registered unit at block `o`, VU `g` and at block `o'`, VU `g'`: when the bits the two VUs
compute for output row `j` differ, `commitPair` of their rows' bytes and salts is a SHA-512 collision. -/
def OutputsBindingC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    formatOk st.c = true →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : FlockSoundness.Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g' < st'.c.g → j < st.c.outRowPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBitsC (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBitsC (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (Hidden.commitPair
      ((HmRow.rowPrefixV2 (16 * (HmOut.outGroup st.c j).2.2.2)).data.toList ++
        HmRow.packBits (outBitsC (Hm.blockBit (L.model regs) z o) st.c g j))
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.rowPrefixV2 (16 * (HmOut.outGroup st'.c j).2.2.2)).data.toList ++
        HmRow.packBits (outBitsC (Hm.blockBit (L'.model regs') z' o') st'.c g' j))
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))

theorem rows_committedC : RowsCommittedC := by
  sorry

theorem outputs_committedC : OutputsCommittedC := by
  sorry

theorem outputs_bindingC : OutputsBindingC := by
  sorry

end FlockSoundness.Discharge.Canonical
