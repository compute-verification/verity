import FlockSoundness.Discharge.Hm.Bc
import FlockSoundness.Protocol.Discharge.Hidden.Wide.Copies
import FlockSoundness.Protocol.Discharge.Hidden.Wide.RowCommit

/-!
# A row's `b ‖ c` under its own prefix, for `sha512/row/v1` and `sha512/row/v2` rows

`Hm.row_bc_flat` for any rows of the untyped statement (`row_bc_tgt`), at a row of `16 · words` bits in
`2 · words = 128 · chunks` bytes (an output row, `sha512/row/v1` or `sha512/row/v2`): Δ's one copy at each of the
row's `cv` and `pad` input bits (`inBit_copyAll`, from `flatAll_count` and the filters of `Wide.Copies`) gives
`port_commit_rowsP` its four facts (`row_copies_tgt`), with the row's own prefix `HmRow.prefixOf`. A row of no
compression (`words = 0`) is hashed by its `hm96` slot alone, its `cv` the prefix's midstate (`row_copies_tgt0`,
`port_commit_rows0`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **Δ's pairs at a row net's input bit `x`**, without a typed template and for any rows: `(x, x), (x, src)` for each
copy `(x, src)` of the rows' constant bits or the rows' wires there; the slot constants, the leaf and cut bits and the
other wires write nothing there. -/
theorem flatAll_count (hrw : RowWires st.c) {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD) (j : ℕ) :
    (flatPairsAll st.c st.pin).count (x, j) =
      (((rowCopiesAll st.c st.pin ++
          wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
        (fun e => e.1 = x)).map fun e => (if j = x then 1 else 0) + (if j = e.2 then 1 else 0)).sum := by
  have R := RowRanges.ofL L F.parse F.check
  have P := F.parse
  have C := F.check
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, -⟩ := R.range hnn
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  have hun : st.c.unitNet ≠ n := by
    rcases hn with rfl | rfl
    · exact F.sha
    · exact F.hm
  obtain ⟨hum, huk, huc⟩ := unit_range_facts P C
  have hunit : ∀ s₀ b, s₀ < st.c.unitRange.count → b < 2 ^ st.c.unitRange.slotLog →
      st.c.unitRange.slot s₀ + b ≠ x := fun s₀ b hs₀ hb he =>
    atSlot_ne L F hnn P.unitNet hun hat (show AtSlot st.c st.c.unitNet s₀ b x from ⟨hs₀, hb, he.symm⟩)
  have hC : ∀ j, (constPairs st.c st.pin).count (x, j) = 0 := fun j => by
    rw [constPairs_count R.disj R.slotLog R.fit hm hk hat j, ite_eq_right (fun h => by have := h.1; omega)]
  have hleaf : ∀ j, (Flat.leafPairs st.c).count (x, j) = 0 := by
    intro j
    refine List.count_eq_zero.2 fun hmem => ?_
    simp only [Flat.leafPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨k, hk', jj, hjj, tt, htt, he⟩ := hmem
    have hx : Flat.leafCol st.c k jj tt = x := by rcases he with he | he <;> exact (Prod.mk.inj he).1.symm
    have hupv : 0 < st.c.upv := by
      rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
      · rw [L.unit_count, h0, Nat.mul_zero] at hk'; omega
      · exact h0
    have hmod := Nat.mod_lt k hupv
    have hlt : k % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact hmod
    have hsz : (st.c.leavesIn.getD (k % st.c.upv) #[]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
        ← getElem!_pos st.c.leavesIn _ hlt]
      exact C.leaves.2 _ hmod
    rw [hsz] at hjj
    obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjj
    have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
      C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hjj)
    have htw : tt < (portCols st.c.unit.inGroups)[jj]!.2 := by rw [hw, h16]; exact htt
    refine hunit k _ hk' (L.port_lt_slot hjs htw) ?_
    rw [← hx]
    unfold Flat.leafCol
    rw [hcol, Nat.add_assoc]
  have hcut : ∀ j, (Flat.cutPairs st.c).count (x, j) = 0 := by
    intro j
    refine List.count_eq_zero.2 fun hmem => ?_
    simp only [Flat.cutPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨g, hg, e, he, tt, htt, hpe⟩ := hmem
    have hx : Flat.cutCol st.c g e + tt = x := by rcases hpe with h | h <;> exact (Prod.mk.inj h).1.symm
    obtain ⟨he1, he2⟩ := C.cuts e he
    have hpv := perVu_unit C hum huk huc (by omega)
    refine hunit (g * st.c.upv + e.1) _ (by rw [huc]; exact Rows.mul_add_lt hg he1) (L.port_lt_slot he2 htt) ?_
    rw [← hx]
    unfold Flat.cutCol Circuit.slotCol Circuit.unitRange
    rw [hpv, Nat.add_assoc]
  have hwire : (wireCopies st.c).filter (fun e => e.1 = x) =
      (wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter (fun e => e.1 = x) := by
    obtain ⟨rest, hws, hrest⟩ := hrw
    unfold wireCopies wireCopiesOf
    rw [hws, List.filter_flatMap, List.filter_flatMap]
    refine List.flatMap_congr fun g hg => ?_
    rw [List.flatMap_append, List.filter_append]
    refine (congrArg (HAppend.hAppend _) (filter_nil_of _ x ?_)).trans (List.append_nil _)
    intro e hmem
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hmem
    obtain ⟨w, hw, tt, htt, rfl⟩ := hmem
    obtain ⟨hd1, hd2, hd3⟩ := C.wires w (by rw [hws]; exact List.mem_append_right _ hw)
    obtain ⟨hwsha, hwhm⟩ := hrest w hw
    have hne : w.dst.1 ≠ n := by
      rcases hn with rfl | rfl
      · exact hwsha
      · exact hwhm
    obtain ⟨hm', hk', -⟩ := rangeOf_spec (C.one_range _ hd1)
    have hq := getD_mem_toList hd3 (d := default)
    have hle := (nets_ok P w.dst.1).1.2.2 _ hq
    have hfit := (L.fit w.dst.1).2
    intro he
    refine atSlot_ne L F hnn hd1 hne hat (s' := g * st.c.perVu.getD w.dst.1 0 + w.dst.2.1)
      (off' := ((portCols (st.c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1 + tt)
      ⟨by rw [C.count _ hm' _ hk']; exact Rows.mul_add_lt (List.mem_range.1 hg) hd2,
        by rw [L.slotLog _ hm' _ hk']; unfold Flat.wireWidth at htt; omega, ?_⟩
    rw [← he]
    unfold Flat.wireDst Circuit.slotCol
    rw [Nat.add_assoc]
  have hf' : (rowCopiesAll st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) =
      (rowCopiesAll st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter (fun e => e.1 = x) := by
    rw [List.filter_append, hwire, ← List.filter_append]
  simp only [flatPairsAll, rowPairsAll_eq, wirePairs_eq, List.count_append, hC, hleaf, hcut, Nat.zero_add,
    Nat.add_zero]
  rw [← List.count_append, ← List.flatMap_append, count_pair2, hf']

/-- **An input bit of a row net's slot holds its one copy**, in the untyped statement and for any rows. At a self row
`x` of a `sha512x3` or `hm96` slot below the net's input rows, where the rows' copies and wires write exactly one
`(x, src)`, every block of a satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copyAll (hda : st.da = (HmRow.delta st.c st.pin).1) (hrw : RowWires st.c) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopiesAll st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src := by
  have R := RowRanges.ofL L F.parse F.check
  have hF : ∀ j, (flatPairsAll st.c st.pin).count (x, j) = (if j = x then 1 else 0) + (if j = src then 1 else 0) :=
    fun j => by
      rw [flatAll_count L F hrw hn hat hl hoff, hf]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]
  have hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
    intro j
    rw [L.delta_b, deltaEntry_eq_count, hda, delta_flat_all _ _ F.parse.typed hu, hF]
    exact ⟨rfl, rfl⟩
  have hx : x < 2 ^ (L.model regs).kLog := hat.lt R.bound (R.range (R.rowNet_lt hn)).1
  exact block_of_rows hz o hx fun j => inBit_rows R hn hat hl hself hx hδ j

end Bc

end FlockSoundness.Discharge.Hidden
