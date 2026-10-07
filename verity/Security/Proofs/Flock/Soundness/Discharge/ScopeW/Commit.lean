import Proofs.Flock.Soundness.Discharge.ScopeW.Chain
import Proofs.Flock.Soundness.Discharge.ScopeW.Copies
import Proofs.Flock.Soundness.Discharge.Hm.Site
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat

set_option Elab.async false

/-!
# A row's `b ‖ c` at the widened scope

`Hm.row_bc_flat` and its forms at a drawn unit's site, for rows of the four shapes `rowShapeW` admits: the `b ‖ c` at
a row's `hm96` output is the commit string of `rowMsgOf` of the bytes at the row's message columns (`Typed.msgCol`),
under the salt at `saltCol`. Each is stated once at lane B's context (`L`, the flat twin's `Aliased.UnitFacts` `F`, its
layout `Lf`, and `RowDelta`), so the flat, hidden and public-input forms are instances:

* `Slot`: the row nets' slots hold their nets' rows, at a statement with its twin's circuit facts;
* `row_bc_flatW`: the copies at each shape (`row_copies_wholeW`, `row_copies_shortW`, `row_copies_segW`) into the
  shape's commit (`Hidden.port_commit_rowsP`, `port_commit_shortW`, `port_commit_segW`);
* `row_bc_setupHW`, `acc_row_bcW`, `bc_site_colW`, `bc_site_commitW`: at `Stmt.setupH` (`rowDelta_flat`), and at an
  accepted statement's site;
* `outRowsS_of_checkRows`, `HiddenExec.AcceptedH.row_bcW`, `bc_site_colHW`, `bc_site_commitHW`: at `Stmt.setupHidden`
  (`rowDelta_hidden`), and at an accepted hidden-output statement's site.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm Compose Flock.DeriveAll Types.Dag

/-! ## The row nets' slots -/

namespace Slot

section Slot

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
include L F

/-- `Hm.rowNet_entryL` at a statement, with its flat twin's circuit facts. -/
theorem rowNet_entry {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
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

/-- `Hm.laidRows_flat` at a statement: a row net's slot holds its net's rows. -/
theorem laidRows {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s : ℕ}
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
      have hE := rowNet_entry L F hn (hat _ hfit) hlo' hl
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
      rw [hcopy.block_eq hz o, hz.2.1 o]
      rfl
  · have ho : off < 2 ^ (st.c.rangeOf n).slotLog := lt_trans hoff hfit
    have hx := hxlt _ ho
    have hE := rowNet_entry L F hn (hat _ ho) hlo hl
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
    have h := congrArg FlockSoundness.bit (hz.1 o ⟨(st.c.rangeOf n).slot s + off, hx⟩)
    rw [bit_mul] at h
    show _ = blockBit (L.model regs) z o ((st.c.rangeOf n).slot s + off)
    rw [blockBit_of_lt _ z o hx]
    refine Eq.trans ?_ h
    exact congrArg₂ (· && ·) (hmv l.ra (L.model regs).A₀ (hcols _ _ ha) fun j => (hE j.val).1).symm
      (hmv l.rb (L.model regs).B₀ (hcols _ _ hb) fun j => (hE j.val).2).symm

/-- `Hm.shaSlot_flat` at a statement. -/
theorem shaSlot {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.shaNet 0) :
    LaidRows (Flock.HmNets.shaLaid ()) 8192 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.shaNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows L F hz o (Or.inl rfl) (R.slot_lt_count (Or.inl rfl) hg hi) (shaNet_ofRows R.rowNets)
  rw [shaNet_inRows R.rowNets] at h
  exact h

/-- `Hm.hmSlot_flat` at a statement. -/
theorem hmSlot {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.hmNet 0) :
    LaidRows (Flock.HmNets.hmLaid ()) 4096 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows L F hz o (Or.inr rfl) (R.slot_lt_count (Or.inr rfl) hg hi) (hmNet_ofRows R.rowNets)
  rw [hmNet_inRows R.rowNets] at h
  exact h

end Slot

end Slot

/-! ## A row's `b ‖ c`, at lane B's context -/

section Bc

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {extra : List (ℕ × ℕ)} {keep : ℕ → Bool} (hΔ : RowDelta st extra keep) (Lf : FlockSoundness.Layout (Hidden.flatTwin st))
include L F hΔ Lf

/-- **A row's `b ‖ c`, for a row of any widened shape**, in every block of a satisfying witness: the commit string of
`rowMsgOf` of the bytes at the row's message columns, under the salt its `hm96` slot reads. -/
theorem row_bc_flatW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hs : rowShapeW st.c.ports[p]! = true) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      (rowMsgOf st.c.ports[p]!
        (bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p)))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hhm := Slot.hmSlot L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))
  have hsha : ∀ k < st.c.ports[p]!.chunks, LaidRows (Flock.HmNets.shaLaid ()) 8192
      fun j => blockBit (L.model regs) z o (shaBase st.c g p k + j) :=
    fun k hk => Slot.shaSlot L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)
  obtain hw | hsh | hsg : RowWhole st.c.ports[p]! ∨ RowV2Short st.c.ports[p]! ∨ RowSegWhole st.c.ports[p]! := by
    rcases rowShapeW_spec hs with h | h | h | h
    exacts [Or.inl (rowWhole_of_v1 F.parse (getElem_mem_toList' hp) h), Or.inl (rowWhole_of_v2 h),
      Or.inr (Or.inl h), Or.inr (Or.inr h)]
  · obtain ⟨hcv0, hchain, hcvh, hpad⟩ := row_copies_wholeW L F hΔ Lf hupv hz o hg hp hw
    obtain ⟨hu, -, hn, hw2⟩ := hw
    rw [rowMsgOf_unseg hu]
    exact Hidden.port_commit_rowsP st.c _ g p (Hidden.size_prefixOf _) hu hn hw2 (rowSlots_size hp).ge R.rowNets
      (fun k hk => (rowSlot_lt hp hk).2) hsha hhm hcv0 hchain hcvh hpad
  · obtain ⟨hcvh, hpad⟩ := row_copies_shortW L F hΔ Lf hupv hz o hg hp hsh
    exact port_commit_shortW st.c _ g p hp hsh R.rowNets hhm hcvh hpad
  · obtain ⟨hscv, hschain, hspad, hocv, hochain, hdig, htail⟩ := row_copies_segW L F hΔ Lf hupv hz o hg hp hsg
    exact port_commit_segW st.c _ g p hp hsg R.rowNets hsha hhm hscv hschain hspad hocv hochain hdig htail

end Bc

section Plain

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
  (hda : st.da = (HmRow.delta st.c st.pin).1)
include L F hda

/-- `row_bc_flatW` at a statement whose Δ is `HmRow.delta`'s (`Stmt.setupH`). -/
theorem row_bc_flatW_flat (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hs : rowShapeW st.c.ports[p]! = true) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      (rowMsgOf st.c.ports[p]!
        (bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p)))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) :=
  row_bc_flatW L (twin_of_flat L F hda).1 (rowDelta_flat hda) (twin_of_flat L F hda).2 hupv hz o hg hp hs

end Plain

/-! ## At `Stmt.setupH` and an accepted statement's site -/

section Setup

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A row's `b ‖ c` at the untyped statement `Stmt.setupH` accepted**, for rows of the widened shapes (`hW`). -/
theorem row_bc_setupHW (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (ho : st.c.outNet = st.c.unitNet) (hW : ∀ p ∈ st.c.ports.toList, rowShapeW p = true)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness ((setupH_layout ht h).model regs).m}
    (hz : ((setupH_layout ht h).model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      (rowMsgOf st.c.ports[p]!
        (bytesOf (blockBit ((setupH_layout ht h).model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p)))
      (bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 192 (saltCol st.c g p)) :=
  row_bc_flatW_flat (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) (setupH_spec ht h).2.2.1
    (upv_pos (Aliased.setupH_unitFacts ht h).check ho) hz o hg hp (hW _ (getElem_mem_toList' hp))

end Setup

section Site

open Model

variable {T : Model.Statement}

/-- `row_bc_setupHW` in an accepted statement `T`. -/
theorem acc_row_bcW (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      (rowMsgOf a.st.c.ports[p]! (bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p)))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  exact row_bc_setupHW ht h ho hW hz o hg hp

/-- **A row's `b ‖ c` at a drawn unit's site**, for rows of the widened shapes (`hW`): at block `o`, VU `g` and row
port `p`, the 128 bytes at the `hm96` slot's output (`bcCol`) are the commit string of `rowMsgOf` of the `2 · words`
bytes at the message columns (`Typed.msgCol`), under the 192 bytes at `saltCol`. -/
theorem bc_site_colW (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString (rowMsgOf a.st.c.ports[p]!
          (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
            FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have R := acc_rowRanges a
  rw [bytesAt_site a z o 128 (bcCol a.st.c g p) (fun j hj t ht => bcCol_lt R hg hp hj ht),
    bytesAt_site a z o 192 (saltCol a.st.c g p) (fun j hj t ht => saltCol_lt R hg hp hj ht),
    ofFn_site a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse (setupH_spec a.ht a.h).1 hg hp j t ht)]
  exact acc_row_bcW a ho hW hz o hg hp

/-- `bc_site_colW` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitW (a : Placed.Accepted T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString (rowMsgOf a.st.c.ports[p]!
          (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
            FlockSoundness.bit (z (Layout.siteAddr a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => Layout.siteAddr a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => Layout.siteAddr a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => Layout.siteAddr a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (perVu_hm_pow2 (acc_rowRanges a))]
    rfl
  rw [hbc]
  exact bc_site_colW a ho hW hz o hg hp

end Site

/-! ## At `Stmt.setupHidden` and an accepted hidden-output statement's site -/

/-- **`OutRowsS` from the parse**: the rows of the widened shapes (`hW`), and the output rows after the input rows
(`HmOut.checkRows`). -/
theorem outRowsS_of_checkRows {c : Circuit} (hW : ∀ p ∈ c.ports.toList, rowShapeW p = true)
    (h : HmOut.checkRows c = .ok ()) : OutRowsS c :=
  ⟨hW, fun _ hj => (Hidden.checkRows_out h hj).1⟩

section SiteH

open Model HiddenExec

variable {T : Model.Statement}

/-- **A row's `b ‖ c` in an accepted hidden-output statement**, for rows of the widened shapes (`hW`): `row_bc_flatW`
at the statement's own layout, its Δ `HmOut.delta`'s (`rowDelta_hidden`). -/
theorem _root_.FlockSoundness.Discharge.HiddenExec.AcceptedH.row_bcW (a : AcceptedH T)
    (ho : a.st.c.outNet = a.st.c.unitNet) (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    bytesOf (blockBit T z o) 128 (bcCol a.st.c g p) = Binding.commitString
      (rowMsgOf a.st.c.ports[p]! (bytesOf (blockBit T z o) (2 * a.st.c.ports[p]!.words) (Typed.msgCol a.st.c g p)))
      (bytesOf (blockBit T z o) 192 (saltCol a.st.c g p)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hW hg hp ⊢
  have s := Hidden.setupHidden_spec' h
  have F0 := flat_unitFacts ht h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  exact row_bc_flatW L F0 (rowDelta_hidden L F0 (outRowsS_of_checkRows hW (Hidden.parse_untyped ht s.parse).2) hda)
    (flat_layout ht h) (Hm.upv_pos F0.check ho) hz o hg hp (hW _ (getElem_mem_toList' hp))

/-- **A row's `b ‖ c` at a drawn unit's site in a hidden-output table**, for rows of the widened shapes (`hW`): at block
`o`, VU `g` and row `p`, the 128 bytes at the `hm96` slot's output (`bcCol`) are the commit string of `rowMsgOf` of the
`2 · words` bytes at the message columns (`Typed.msgCol`), under the 192 bytes at `saltCol`. -/
theorem bc_site_colHW (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (bcCol a.st.c g p j.val t.val)) =
      Binding.commitString (rowMsgOf a.st.c.ports[p]!
          (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
            FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have R := a.rowRanges
  rw [bytesAt_siteH a z o 128 (bcCol a.st.c g p) (fun j hj t ht => Hm.bcCol_lt R hg hp hj ht),
    bytesAt_siteH a z o 192 (saltCol a.st.c g p) (fun j hj t ht => Hm.saltCol_lt R hg hp hj ht),
    ofFn_siteH a z o _ (Typed.msgCol a.st.c g p)
      (fun j _ t ht => Layout.msgCol_lt_of_parse a.parse hg hp j t ht)]
  exact a.row_bcW ho hW hz o hg hp

/-- `bc_site_colHW` at the address the regions pin (`Layout.bcAddr`). -/
theorem bc_site_commitHW (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hW : ∀ p ∈ a.st.c.ports.toList, rowShapeW p = true)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p : ℕ} (hg : g < a.st.c.g)
    (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      Binding.commitString (rowMsgOf a.st.c.ports[p]!
          (List.ofFn fun j : Fin (2 * a.st.c.ports[p]!.words) => Binding.byteOf fun t =>
            FlockSoundness.bit (z (siteAddrH a o (Typed.msgCol a.st.c g p j.val t.val)))))
        (Binding.bytesAt z fun (j : Fin 192) (t : Fin 8) => siteAddrH a o (saltCol a.st.c g p j.val t.val)) := by
  have hbc : (fun (j : Fin 128) (t : Fin 8) => siteAddrH a o (Layout.bcAddr a.st.c g p j.val t.val)) =
      fun j t => siteAddrH a o (bcCol a.st.c g p j.val t.val) := by
    funext j t
    rw [Layout.bcAddr_eq_slotCol _ _ _ _ _ (Hm.perVu_hm_pow2 a.rowRanges)]
    rfl
  rw [hbc]
  exact bc_site_colHW a ho hW hz o hg hp

end SiteH

end FlockSoundness.Discharge.ScopeW
