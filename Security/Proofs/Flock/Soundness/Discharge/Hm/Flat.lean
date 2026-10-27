/-
A row's `b ‖ c` in a satisfying witness of the untyped statement `Stmt.setupH` accepted (`row_bc_setupH`). Without a typed
template Δ is its slot constants' pairs, the rows' constant bits, the units' leaf and cut bits and the wires' bits, and
nothing after (`delta_flat_eq`). Past a row net's input rows Δ writes only slot constants (`Layout.not_inBit`), so the
`sha512x3` and `hm96` slots hold their nets' rows (`shaSlot_flat`, `hmSlot_flat`). At their `cv` and `pad` input bits the
leaf and cut bits are elsewhere (the unit is neither row net), and so are the wires after the rows' own (`RowWires`,
`HmRow.parse`'s wire check): Δ's one copy there is `rowCopies`' or the rows' wires' (`inBit_copyL`). The constant bits
read the pin and the unit range's forced-zero row (`Aliased.zeroPos_block`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Copy
import Proofs.Flock.Soundness.Discharge.Aliased

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

/-- **Δ without a typed template, pair by pair**: the slot constants' pairs, the rows' constant bits, the units' leaf
and cut bits and the wires' bits (`flatPairs`). -/
theorem delta_flat_eq (c : Circuit) (pin : ℕ) (ht : c.typed = none)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hw : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) :
    (HmRow.delta c pin).1.toList = flatPairs c pin := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d.toList ++ rowPairs c pin) ?_ ?_
  · rw [← rowPairsAll_v1 c pin hv1 hw hs]
    refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rw [segs_none_of_v1 hv1 hs hp]
    refine ⟨_, rfl, ?_⟩
    rw [rowPairsOf, ← List.append_assoc, ← List.append_assoc]
    refine Eq.trans (forIn_range'_append (fun k => (List.range 1024).flatMap fun t => bitPairs c pin
        (msgColOf c g p k + t) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)) _ _
        (fun k _ _ d => ⟨_, rfl, ?_⟩) _) ?_
    · refine forIn_range_append _ _ (fun t _ d => ?_) d
      split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = none :=
          Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine Eq.trans (forIn_range_append (fun t => bitPairs c pin (padCol c g p + t)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)) 1024
        (fun t _ d => ?_) _) ?_
    · split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          none := Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d0 hd0
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  simp only [ht]
  show d3.toList = _
  rw [hd3, hd2, hd1, hd0, hd]
  rfl

/-- **`HmRow.parse`'s wire check, on the rows' nets**: the circuit's wires begin with the rows' chains and `hm96` inputs
(`HmRow.rowWires`), and no later wire ends in a `sha512x3` or `hm96` slot. -/
def RowWires (c : Circuit) : Prop :=
  ∃ rest, c.wires.toList = (HmRow.rowWires c.ports c.shaNet c.hmNet).toList ++ rest ∧
    ∀ w ∈ rest, w.dst.1 ≠ c.shaNet ∧ w.dst.1 ≠ c.hmNet

section Flat

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **The block rows at a row net's row past its input rows**, in the untyped statement: the net's row, placed in the
slot, and the slot constants' pairs there (Δ's other pairs are input bits). -/
theorem rowNet_entryL {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ off) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (j : ℕ) :
    rangesEntry st (slotEntryA st) x j + deltaEntry st.da x j =
        ((((l.ra.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) x j + deltaEntry st.db x j =
        ((((l.rb.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) := by
  have R := RowRanges.ofL L F.parse F.check
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, hsl⟩ := R.range hnn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le R.bound hm) R.kLog
  have hw : 2 ^ (st.c.nets.getD n default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) h32
  obtain ⟨-, ha, hb, -, -, -, -, -, -, -, -, -, hlk, -⟩ := Net.ofRows_ok hl
  have hd : deltaEntry st.da x j = (((constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) := by
    obtain ⟨rest, hsplit, hrest⟩ := L.delta
    have h0 : rest.count (x, j) = 0 := List.count_eq_zero.2 fun hmem => by
      have hin := hrest _ hmem
      simp only [hat.2.2] at hin
      exact L.not_inBit hm hk hat.1 hat.2.1 hlo hin
    rw [deltaEntry_eq_count, hsplit, List.count_append, h0, Nat.add_zero]
  have hre := fun e => rangesEntry_at R.disj hm hk hat e j
  have hla := ofRows_cols_lt ha off
  have hlb := ofRows_cols_lt hb off
  refine ⟨?_, ?_⟩
  · rw [hre, hd]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
    rw [sparseEntry_ofRows ha, ofRows_row ha hw, count_slot _ (by rw [hsl]; exact hla) j]
    push_cast; ring
  · rw [hre, L.delta_b, hd]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
      lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
    rw [hlk]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ + 0 else 0) + _ = _
    rw [add_zero, sparseEntry_ofRows hb, ofRows_row hb hw, count_slot _ (by rw [hsl]; exact hlb) j]
    push_cast; ring

/-- **A row net's slot column is no unit input row** (`UnitIn`): the ranges are apart, and the unit is neither row
net. -/
theorem not_unitIn_of_rowNet {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) :
    ¬ UnitIn st.c x := by
  rintro ⟨s', off', hat', -⟩
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hm, hk, -, -⟩ := R.range (R.rowNet_lt hn)
  have he := (hat.inj L.disj hm hk L.unit_mem L.unit_kind hat').1
  rcases hn with rfl | rfl
  · exact F.sha he.symm
  · exact F.hm he.symm

/-- **A unit slot's forced-zero row is no unit input row** (`UnitIn`): it is past the unit's constant row, which is
past its inputs. -/
theorem not_unitIn_zero (s₀ : ℕ) : ¬ UnitIn st.c (st.c.unitRange.slot s₀ + st.c.unit.useful) := by
  rintro ⟨s, off, ⟨-, hoff, hx⟩, hlt⟩
  have hb := built_spec (FlockSoundness.unit_built F.parse F.check)
  have hz : st.c.unit.useful < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact F.check.unit_zero
  have := (slot_add_inj (r := st.c.unitRange) hz hoff hx).2
  have h1 := hb.2.1
  have h2 := hb.2.2.2.1
  omega

/-- **A row net's slot holds its net's rows** in every block of a witness that satisfies every row but the unit input
rows (`SatEx`, `UnitIn`): each computed row past the input rows holds, and the slot's constant column is 1. -/
theorem laidRows_flat_ex {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) (UnitIn st.c) z) (o : Fin (2 ^ (L.model regs).nbl)) {n s : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hs : s < (st.c.rangeOf n).count) {l : Flock.HmNets.Laid}
    {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2) :
    LaidRows l ((st.c.nets.getD n default).2.inWords * Flock.WORD)
      fun j => blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, hsl⟩ := R.range hnn
  obtain ⟨-, ha, hb, -, hcp, -, -, hin, hca, hcb, -, -, -, -⟩ := Net.ofRows_ok hl
  have hfit : l.constPos < 2 ^ (st.c.rangeOf n).slotLog := by rw [hsl, ← hcp]; exact R.fit n
  have hat : ∀ off < 2 ^ (st.c.rangeOf n).slotLog, AtSlot st.c n s off ((st.c.rangeOf n).slot s + off) :=
    fun off ho => ⟨hs, ho, rfl⟩
  have hxlt : ∀ off < 2 ^ (st.c.rangeOf n).slotLog, (st.c.rangeOf n).slot s + off < 2 ^ st.c.kLog :=
    fun off ho => (hat off ho).lt R.bound hm
  refine ⟨?_, fun off hlo hoff => ?_⟩
  · have hx := hxlt _ hfit
    show blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + l.constPos) = true
    rw [blockBit_of_lt _ z o hx]
    by_cases hp : (st.c.rangeOf n).slot s + l.constPos = st.pin
    · have hpin : (⟨(st.c.rangeOf n).slot s + l.constPos, hx⟩ : Fin (2 ^ (L.model regs).kLog)) = (L.model regs).pin :=
        Fin.ext hp
      rw [hpin, hz.2.1 o]
      rfl
    · have hlo' : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ l.constPos := hin
      have hE := rowNet_entryL L F hn (hat _ hfit) hlo' hl
      have hC := fun j => constPairs_count (pin := st.pin) R.disj R.slotLog R.fit hm hk (hat _ hfit) j
      have hrow : ∀ (r : Array (Array ℕ)), r.getD l.constPos #[] = #[l.constPos] →
          ∀ j : Fin (2 ^ (L.model regs).kLog),
          ((((r.getD l.constPos #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j.val +
            (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + l.constPos, j.val) : ℕ) : ZMod 2) =
            if j = (L.model regs).pin then 1 else 0 := by
        intro r hr j
        rw [hr, hC j.val, ite_eq_left ⟨hcp.symm, hp⟩]
        have hj : (j = (L.model regs).pin) ↔ j.val = st.pin := Fin.ext_iff
        by_cases h1 : j.val = (st.c.rangeOf n).slot s + l.constPos
        · have h2 : ¬ j.val = st.pin := fun h => hp (h1.symm.trans h)
          rw [ite_eq_right (fun h => h2 (hj.1 h)), ite_eq_left h1, ite_eq_right h2]
          simp only [List.map_cons, List.map_nil, List.count_singleton, h1, beq_self_eq_true, ite_true]
          decide
        · rw [ite_eq_right h1]
          have h3 : List.count j.val [(st.c.rangeOf n).slot s + l.constPos] = 0 :=
            List.count_eq_zero.2 (by simp [h1])
          simp only [List.map_cons, List.map_nil, h3, zero_add]
          by_cases h2 : j.val = st.pin
          · rw [ite_eq_left h2, ite_eq_left (hj.2 h2)]; rfl
          · rw [ite_eq_right h2, ite_eq_right (fun h => h2 (hj.1 h))]; rfl
      have hcopy : CopyRow (L.model regs) ⟨(st.c.rangeOf n).slot s + l.constPos, hx⟩ (L.model regs).pin :=
        ⟨fun j => ((hE j.val).1.trans (hrow l.ra hca j)), fun j => ((hE j.val).2.trans (hrow l.rb hcb j))⟩
      rw [hcopy.block_eq_ex hz (not_unitIn_of_rowNet L F hn (hat _ hfit)) o, hz.2.1 o]
      rfl
  · have ho : off < 2 ^ (st.c.rangeOf n).slotLog := lt_trans hoff hfit
    have hx := hxlt _ ho
    have hE := rowNet_entryL L F hn (hat _ ho) hlo hl
    have hC : ∀ j, (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + off, j) = 0 := fun j => by
      rw [constPairs_count R.disj R.slotLog R.fit hm hk (hat _ ho) j, ite_eq_right (fun h => by have := h.1; omega)]
    have hcols : ∀ (r : Array (Array ℕ)) (m : Sparse), Sparse.ofRows r (2 ^ (st.c.nets.getD n default).2.unitLog) = .ok m →
        ∀ y ∈ (r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (L.model regs).kLog := by
      intro r m hr y hy
      obtain ⟨y', hy', rfl⟩ := List.mem_map.1 hy
      exact hxlt _ (by rw [hsl]; exact ofRows_cols_lt hr off y' hy')
    have hmv : ∀ (r : Array (Array ℕ))
        (M : Matrix (Fin (2 ^ (L.model regs).kLog)) (Fin (2 ^ (L.model regs).kLog)) (ZMod 2)),
        (∀ y ∈ (r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (L.model regs).kLog) →
        (∀ j : Fin (2 ^ (L.model regs).kLog), M ⟨(st.c.rangeOf n).slot s + off, hx⟩ j =
          ((((r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j.val +
            (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + off, j.val) : ℕ) : ZMod 2)) →
        FlockSoundness.bit (M.mulVec ((L.model regs).block z o) ⟨(st.c.rangeOf n).slot s + off, hx⟩) =
          xorRow (fun j => blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + j)) (r.getD off #[]) := by
      intro r M hlt hM
      simp only [Matrix.mulVec, dotProduct, hM, hC, Nat.add_zero]
      rw [sum_count_nat _ hlt]
      exact bit_sum_row _ _ _
    have h := congrArg FlockSoundness.bit (hz.1 o ⟨(st.c.rangeOf n).slot s + off, hx⟩
      (not_unitIn_of_rowNet L F hn (hat _ ho)))
    rw [bit_mul] at h
    show _ = blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + off)
    rw [blockBit_of_lt _ z o hx]
    refine Eq.trans ?_ h
    exact congrArg₂ (· && ·) (hmv l.ra (L.model regs).A₀ (hcols _ _ ha) fun j => (hE j.val).1).symm
      (hmv l.rb (L.model regs).B₀ (hcols _ _ hb) fun j => (hE j.val).2).symm

/-- **Template instance `g`'s `sha512x3` slot `i` holds the verifier's `sha512x3` rows**, past its 8192 input rows, in
every block of a satisfying witness of the untyped statement. -/
theorem shaSlot_flat_ex {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) (UnitIn st.c) z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.shaNet 0) :
    LaidRows (Flock.HmNets.shaLaid ()) 8192 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.shaNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows_flat_ex L F hz o (Or.inl rfl) (R.slot_lt_count (Or.inl rfl) hg hi) (shaNet_ofRows R.rowNets)
  rw [shaNet_inRows R.rowNets] at h
  exact h

theorem shaSlot_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.shaNet 0) :
    LaidRows (Flock.HmNets.shaLaid ()) 8192 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.shaNet g i + j) :=
  shaSlot_flat_ex L F (hz.ex _) o hg hi

/-- **Template instance `g`'s `hm96` slot `i` holds the verifier's `hm96` rows**, past its 4096 input rows. -/
theorem hmSlot_flat_ex {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) (UnitIn st.c) z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.hmNet 0) :
    LaidRows (Flock.HmNets.hmLaid ()) 4096 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows_flat_ex L F hz o (Or.inr rfl) (R.slot_lt_count (Or.inr rfl) hg hi) (hmNet_ofRows R.rowNets)
  rw [hmNet_inRows R.rowNets] at h
  exact h

/-- **Template instance `g`'s `hm96` slot `i` holds the verifier's `hm96` rows**, past its 4096 input rows. -/
theorem hmSlot_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.hmNet 0) :
    LaidRows (Flock.HmNets.hmLaid ()) 4096 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g i + j) :=
  hmSlot_flat_ex L F (hz.ex _) o hg hi

/-- A column at a slot of a net other than `n` is not at `n`'s slots. -/
theorem atSlot_ne {n n' s s' off off' x : ℕ} (hn : n < st.c.nets.size) (hn' : n' < st.c.nets.size) (hne : n' ≠ n)
    (hat : AtSlot st.c n s off x) (hat' : AtSlot st.c n' s' off' x) : False := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (F.check.one_range n hn)
  obtain ⟨hm', hk', -⟩ := rangeOf_spec (F.check.one_range n' hn')
  exact hne (hat'.inj L.disj hm' hk' hm hk hat).1

/-- **Δ's flat pairs at a row net's input bit `x`**: where `rowCopies` and the rows' wires write one copy `(x, src)`,
the slot constants, the leaf and cut bits and the other wires write nothing. -/
theorem flatPairs_countL (hrw : RowWires st.c) {n s off x src : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hf : (rowCopies st.c st.pin ++ wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) (j : ℕ) :
    (flatPairs st.c st.pin).count (x, j) = (if j = x then 1 else 0) + (if j = src then 1 else 0) := by
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
  have hf' : (rowCopies st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) = [(x, src)] := by
    rw [List.filter_append, hwire, ← List.filter_append]
    exact hf
  exact flatPairs_count hC hleaf hcut hf' j

/-- **An input bit of a row net's slot holds its one copy**, in the untyped statement. At a self row `x` of a
`sha512x3` or `hm96` slot below the net's input rows, where `rowCopies` and the rows' wires write exactly one
`(x, src)`, every block of a satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copyL (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hrw : RowWires st.c)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopies st.c st.pin ++ wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src := by
  have R := RowRanges.ofL L F.parse F.check
  have hF := flatPairs_countL L F hrw hn hat hl hoff hf
  have hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
    intro j
    rw [L.delta_b, deltaEntry_eq_count, hda, delta_flat_eq _ _ F.parse.typed hv1 F.parse.portWords F.parse.v1Segs, hF]
    exact ⟨rfl, rfl⟩
  have hx : x < 2 ^ (L.model regs).kLog := hat.lt R.bound (R.range (R.rowNet_lt hn)).1
  exact block_of_rows hz o hx fun j => inBit_rows R hn hat hl hself hx hδ j

end Flat

end FlockSoundness.Discharge.Hm
