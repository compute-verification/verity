import Proofs.Flock.Soundness.Discharge.Canonical.Commit
import Proofs.Flock.Soundness.Discharge.Canonical.OutCopy
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Defs
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv

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

Every declaration here is proved from `Commit.row_bc_C`, whose proof is `Commit.lean`'s:
* `rows_committedC`: `outputs_committed_wide'`'s route. A satisfying witness of the hidden statement satisfies its flat
  twin (`OutCopy.sat_flatC`), whose row `p`'s `b ‖ c` is `row_bc_C`'s, and `Hidden.bc_drawnW` reads the registered side
  (shared rows or not; the VU bound from `formatOk_vu_lt`).
* `outputs_committedC`: `rows_committedC` at `p = nIn + j`, then Δ's output pairs (`OutCopy.msg_copyC`, one per output
  word into its run's row) rewrite `rowBits` at output row `j` to `outBitsC` (`rowBits_outC`); `formatOk`'s exact rows
  leave no forced-zero bit in the row.
* `outputs_bindingC`: both sides equal one registered `b ‖ c` (`outputs_committedC`), the prefixes are equal (one circuit
  file), so `Hidden.commitPair_spec'` gives a collision unless the packed bits are equal, and
  `FlockLevel3.RowV2.packBits_inj` (equal lengths) turns that into equal `outBitsC`.
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
  intro tags cf pf coins tables d partition program st ht h pop hpop D hD hlt hfmt L regs hregs z hz o g p hg hp hr
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L0 : FlockSoundness.Layout (Hidden.flatTwin st) :=
    layout_of (st := Hidden.flatTwin st) (parse_facts hpa) s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (Hidden.flatTwin st) := Hidden.unitFacts_of_parse (st := Hidden.flatTwin st) hpa rfl
  have R : Hm.RowRanges st.c := Hm.RowRanges.ofL (st := Hidden.flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hU := unit_built F0.parse F0.check
  have hrows := (outRowsC_of hU hpa hcr hfmt).1
  have hupv : 0 < st.c.upv := Hidden.setupHidden_upv_pos ht h
  have hbc : Hm.bytesOf (Hm.blockBit (L.model regs) z o) 128 (Hm.bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++
        HmRow.packBits (rowBits (Hm.blockBit (L.model regs) z o) st.c g p))
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g p)) :=
    row_bc_C L0 F0 rfl hupv hrows.1 (sat_flatC L L0 F0 hda hrows hz) o hg hp
      ((formatOk_spec hU hfmt).1 _ (getElem_mem_toList' hp)).1
      (v2_of_parse hpa hp ((formatOk_spec hU hfmt).1 _ (getElem_mem_toList' hp)).1
        ((formatOk_spec hU hfmt).1 _ (getElem_mem_toList' hp)).2).2
  rw [← hbc, ← Hidden.bytesAt_bc L R regs z o hg hp]
  exact (Hidden.bc_drawnW h hpop hD hlt L regs hregs hz o hg (formatOk_vu_lt hfmt hg) hp hr).symm

/-- **Output row `j`'s bits are its run's output words**, when each of the row's message bits holds its word's bit. -/
theorem rowBits_outC {c : Circuit} (w : ℕ → Bool) {g j : ℕ}
    (hbits : c.ports[HmOut.nIn c + j]!.bits = 16 * (HmOut.outGroup c j).2.2.2)
    (hcopy : ∀ k < (HmOut.outGroup c j).2.2.2, ∀ t < 16,
      w (Typed.msgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
        w (HmOut.outCol c g ((HmOut.outGroup c j).2.2.1 + k) + t)) :
    rowBits w c g (HmOut.nIn c + j) = outBitsC w c g j := by
  apply List.ext_getElem (by rw [rowBits_length, hbits]; simp [outBitsC])
  intro i h1 h2
  have hi : i < 16 * (HmOut.outGroup c j).2.2.2 := by simpa [outBitsC] using h2
  simp only [rowBits, outBitsC, List.getElem_ofFn]
  have := hcopy (i / 16) (by omega) (i % 16) (Nat.mod_lt _ (by norm_num))
  have e1 : 2 * (i / 16) + i % 16 / 8 = i / 8 := by omega
  have e2 : i % 16 % 8 = i % 8 := by omega
  rw [e1, e2] at this
  exact this

theorem outputs_committedC : OutputsCommittedC := by
  intro tags cf pf coins tables d partition program st ht h pop hpop D hD hlt hfmt L regs hregs z hz o g j hg hj hr
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L0 : FlockSoundness.Layout (Hidden.flatTwin st) :=
    layout_of (st := Hidden.flatTwin st) (parse_facts hpa) s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (Hidden.flatTwin st) := Hidden.unitFacts_of_parse (st := Hidden.flatTwin st) hpa rfl
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  obtain ⟨hrows, hwords⟩ := outRowsC_of (unit_built F0.parse F0.check) hpa hcr hfmt
  obtain ⟨hp, -⟩ := hrows.2 j hj
  have hbits : st.c.ports[HmOut.nIn st.c + j]!.bits = 16 * (HmOut.outGroup st.c j).2.2.2 := by
    rw [Hidden.getElem!_eq_getD']
    exact formatOk_out hfmt hj
  have hrb : rowBits (Hm.blockBit (L.model regs) z o) st.c g (HmOut.nIn st.c + j) =
      outBitsC (Hm.blockBit (L.model regs) z o) st.c g j := by
    refine rowBits_outC _ hbits fun k hk t ht => ?_
    have hc := msg_copyC L L0 F0 hda hrows hz o hg hj (by rw [hwords j hj]; exact hk) ht
    rw [Hidden.outSrc, ite_eq_left hk] at hc
    rw [Hidden.msgCol_delta]
    exact hc
  rw [rows_committedC ht h hpop hD hlt hfmt L regs hregs hz o hg hp hr, hbits, hrb]

theorem outputs_bindingC : OutputsBindingC := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hs hs' pop hpop D D' hD
    hDp hD' hDp' hfmt L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg' hj hr hr' hidx hne
  have hcc : st.c = st'.c := Hidden.setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have hfmt' : formatOk st'.c = true := by rw [← hcc]; exact hfmt
  have hj' : j < st'.c.outRowPorts.size := by rw [← hcc]; exact hj
  have e1 := outputs_committedC ht hs hpop hD hDp hfmt L regs hregs hz o hg hj hr
  have e2 := outputs_committedC ht hs' hpop' hD' hDp' hfmt' L' regs' hregs' hz' o' hg' hj' hr'
  have hpre : HmRow.rowPrefixV2 (16 * (HmOut.outGroup st.c j).2.2.2) =
      HmRow.rowPrefixV2 (16 * (HmOut.outGroup st'.c j).2.2.2) := by rw [hcc]
  refine Hidden.commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hpre] at e
    refine hne (FlockLevel3.RowV2.packBits_inj ?_ (List.append_cancel_left e))
    simp only [outBitsC, List.length_ofFn, hcc]
  · rw [hidx, hcc]

end FlockSoundness.Discharge.Canonical
