import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Copies
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.RowCommit
import Proofs.Flock.Soundness.Discharge.Hm.Bc

/-!
# A row's `b ‖ c` under its own prefix, for `sha512/row/v1` and `sha512/row/v2` rows

`Hm.row_bc_flat` for any rows of the untyped statement (`row_bc_tgt`), at a row of `16 · words` bits whose bytes fill
its compressions' blocks and end, with their padding, in its `hm96` slot's block (`sha512/row/v1` or `sha512/row/v2`):
Δ's one copy at each of the row's `cv` input bits and at each constant bit of its `pad` (`inBit_copyAll`, from
`flatAll_count` and the filters of `Wide.Copies`) gives `port_commit_rowsP` its facts (`row_copies_tgt`,
`pad_const_tgt`), with the row's own prefix `HmRow.prefixOf`. A row of no compression is hashed by its `hm96` slot
alone, its `cv` the prefix's midstate (`row_copies_tgt0`, `port_commit_rows0`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

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

/-- **The three chain copies into template instance `g`'s row `p`** of a compression or more, in every block of a
satisfying witness of the untyped statement: the first compression's `cv` is the row prefix's midstate
(`HmRow.prefixOf`), each later compression's `cv` the previous one's output, and the `hm96` slot's `cv` the last
compression's output. -/
theorem row_copies_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (h0 : 0 < st.c.ports[p]!.chunks) :
    (∀ t < 512, blockBit (L.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (L.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t)) ∧
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (L.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  refine ⟨fun t ht => ?_, fun k hk t ht => ?_, fun t ht => ?_⟩
  · have hs : shaSub st.c p 0 < 3 := (rowSlot_lt hp h0).2
    rw [shaIn_pos R hu hp h0]
    have hf := cv0_filterAll R hu (pin := st.pin) hg hp h0 ht
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inl rfl) (shaPos_at R hu hg hp h0 (off := t) (by omega))
      (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero
  · have hs : shaSub st.c p (k + 1) < 3 := (rowSlot_lt hp hk).2
    have hf := chain_filterAll R hu (pin := st.pin) (k := k + 1) hg hp (by omega) hk ht
    rw [chainDst_eq R.rowNets hs] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inl rfl) (R.shaIn_at hg hp hk ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    rfl
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := hmcv_filterAll R hu (pin := st.pin) hg hp h0 ht
    have hx : Flat.wireDst st.c g (hmWire st.c.ports st.c.shaNet st.c.hmNet p) = st.c.slotCol st.c.hmNet g p :=
      hmWireDst_eq R.rowNets g p
    rw [hx] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
      (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    unfold Flat.wireSrc hmWire shaBase shaOut shaSub
    simp only [rowSlots_size hp]

/-- **The `cv` copy into template instance `g`'s row `p` of no compression**, in every block of a satisfying witness of
the untyped statement: the `hm96` slot's `cv` is the row prefix's midstate. -/
theorem row_copies_tgt0 (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (h0 : st.c.ports[p]!.chunks = 0) :
    ∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  intro t ht
  rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
  have hf := hmcv0_filterAll R hu (pin := st.pin) hg hp h0 ht
  rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
    (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
  exact blockBit_constSrc _ _ hpin hzero

/-- **The constant bits of template instance `g`'s row `p`'s `pad`**, in every block of a satisfying witness of the
untyped statement: wherever `HmRow.constBit` gives the row's last block a constant, Δ's one copy holds it there. -/
theorem pad_const_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    ∀ t < 1024, ∀ v, HmRow.constBit st.c.ports[p]! (HmRow.tailOf st.c.ports[p]!) st.c.ports[p]!.chunks t = some v →
      blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) = v := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  intro t ht v hv
  rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
  have hf := pad_filterAll R hu (pin := st.pin) hg hp ht
  rw [rowSlots_size hp, hv] at hf
  have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
    rw [padCol_eq R.rowNets, Nat.add_assoc]
  rw [hx] at hf
  simp only [bitCopy] at hf
  rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
    (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
  cases v
  · exact hzero
  · exact hpin

/-- **A row's `b ‖ c` under its own prefix**, in every block of a satisfying witness of the untyped statement, for any
rows: at a row of `16 · words` bits whose bytes fill its compressions' blocks and end, with their padding, in its `hm96`
slot's block (`sha512/row/v1` or `sha512/row/v2`), the commit string of the row's prefix `HmRow.prefixOf` and the bytes
`Typed.msgCol` reads, under the salt its `hm96` slot reads. -/
theorem row_bc_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hb : st.c.ports[p]!.bits = 16 * st.c.ports[p]!.words)
    (h1 : 128 * st.c.ports[p]!.chunks ≤ 2 * st.c.ports[p]!.words)
    (h2 : 2 * st.c.ports[p]!.words + 17 ≤ 128 * (st.c.ports[p]!.chunks + 1))
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.prefixOf st.c.ports[p]!).data.toList ++
        bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hhm := hmSlot_flat L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))
  have hs := hu _ (getElem_mem_toList' hp)
  have hconst := pad_const_tgt L F hda hupv hu hz o hg hp
  rcases Nat.eq_zero_or_pos st.c.ports[p]!.chunks with h0 | h0
  · rw [h0] at hconst h2
    exact port_commit_rows0 st.c _ g p (size_prefixOf _) hs hb (by rw [rowSlots_size hp, h0]) h2 R.rowNets hhm
      (row_copies_tgt0 L F hda hupv hu hz o hg hp h0) hconst
  · obtain ⟨hcv0, hchain, hcvh⟩ := row_copies_tgt L F hda hupv hu hz o hg hp h0
    exact port_commit_rowsP st.c _ g p (size_prefixOf _) hs hb (rowSlots_size hp) h0 h1 h2 R.rowNets
      (fun k hk => (rowSlot_lt hp hk).2)
      (fun k hk => shaSlot_flat L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)) hhm hcv0 hchain hcvh
      hconst

end Bc

end FlockSoundness.Discharge.Hidden
