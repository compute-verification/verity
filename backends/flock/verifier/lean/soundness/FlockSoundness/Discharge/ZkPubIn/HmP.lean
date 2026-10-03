import FlockSoundness.Discharge.ZkPubIn.Flat
import FlockSoundness.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# A row's `b ‖ c` in a statement with the verifier's public inputs

`Hm.row_bc_flat` (`Hm/Bc.lean`, `Hm/Flat.lean`) at the statement itself, whose Δ is `HmIn.delta`'s, rather than at its
flat twin. A witness of the statement need not satisfy the twin, which copies the forced zero into a public leaf's
bits; but the row nets' slots read only their own rows, the pin and the forced zero, and at a `cv` or `pad` input bit
Δ is `HmRow.delta`'s: the bit is no public column (a row net's slot is no unit slot, `publicCol_spec`) and no output
row's message bit (`HmRow.delta` copies into the one, `flatPairs_countL`, and not the other, `flat_count_msg`). The
circuit's facts are the twin's (`Aliased.UnitFacts` at `Hidden.flatTwin st`, which reads only the circuit and its pin).

* `rowNet_entryLP`, `laidRows_flatP`, `shaSlot_flatP`, `hmSlot_flatP`: the row nets' slots hold their rows;
* `zero_bitP`: the forced zero (`HiddenExec.zeroPos_blockH`);
* `inBit_copyLP`, `row_copies_flatP`, `row_bc_flatP`: Δ's one copy at a `cv` or `pad` input bit, and a row's `b ‖ c`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Hm Compose Flock.DeriveAll Types.Dag

section HmP

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
include L F

/-- `Hm.rowNet_entryL` at the statement, with the twin's circuit facts. -/
theorem rowNet_entryLP {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
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

/-- `Hm.laidRows_flat` at the statement: a row net's slot holds its net's rows. -/
theorem laidRows_flatP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
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
      have hE := rowNet_entryLP L F hn (hat _ hfit) hlo' hl
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
    have hE := rowNet_entryLP L F hn (hat _ ho) hlo hl
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

/-- `Hm.shaSlot_flat` at the statement. -/
theorem shaSlot_flatP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.shaNet 0) :
    LaidRows (Flock.HmNets.shaLaid ()) 8192 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.shaNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows_flatP L F hz o (Or.inl rfl) (R.slot_lt_count (Or.inl rfl) hg hi) (shaNet_ofRows R.rowNets)
  rw [shaNet_inRows R.rowNets] at h
  exact h

/-- `Hm.hmSlot_flat` at the statement. -/
theorem hmSlot_flatP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g i : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD st.c.hmNet 0) :
    LaidRows (Flock.HmNets.hmLaid ()) 4096 fun j => blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g i + j) := by
  have R := RowRanges.ofL L F.parse F.check
  have h := laidRows_flatP L F hz o (Or.inr rfl) (R.slot_lt_count (Or.inr rfl) hg hi) (hmNet_ofRows R.rowNets)
  rw [hmNet_inRows R.rowNets] at h
  exact h

variable (Lf : FlockSoundness.Layout (Hidden.flatTwin st))
include Lf

/-- `Hm.zero_bit` at the statement (`HiddenExec.zeroPos_blockH`). -/
theorem zero_bitP (hupv : 0 < st.c.upv) (hg : 0 < st.c.g) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    blockBit (L.model regs) z o (zeroCol st.c) = false := by
  have hs₀ : 0 < st.c.unitRange.count := by rw [L.unit_count]; exact Nat.mul_pos hg hupv
  have h0 := HiddenExec.zeroPos_blockH L Lf regs F hs₀ hz o
  unfold zeroCol
  rw [blockBit_of_lt _ z o (y := st.c.unitRange.slot 0 + st.c.unit.useful) (Aliased.zeroPos Lf regs F hs₀).isLt]
  show FlockSoundness.bit ((L.model regs).block z o (Aliased.zeroPos Lf regs F hs₀)) = false
  rw [h0]
  rfl

variable {pi : HmIn.Ports} (lp : LeavesPublic st.c pi) (hrows : Hidden.OutRows st.c)
  (hda : st.da = ((HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray).filter
    (fun x => !HmIn.publicCol st.c pi x.1))
include lp hrows hda

/-- **`Hm.inBit_copyL` at the statement**: at a `cv` or `pad` input bit `x` of a row net's slot, Δ is `HmRow.delta`'s:
`x` is no public column (no unit slot's bit, `publicCol_spec`) and no output row's message bit (`HmRow.delta` copies
into `x`, `flatPairs_countL`, and into no such bit, `Hidden.flat_count_msg`). -/
theorem inBit_copyLP (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hrw : RowWires st.c)
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
  have hF := flatPairs_countL Lf F hrw hn hat hl hoff hf
  have hun : st.c.unitNet ≠ n := by
    rcases hn with rfl | rfl
    · exact F.sha
    · exact F.hm
  have hpc : HmIn.publicCol st.c pi x = false := by
    cases hc : HmIn.publicCol st.c pi x with
    | false => rfl
    | true =>
      exfalso
      obtain ⟨k, hk, j, -, t, ht, hp, hw, he, -⟩ := publicCol_spec L F.check lp hc
      exact atSlot_ne Lf F (R.rowNet_lt hn) F.parse.unitNet hun hat
        (show AtSlot st.c st.c.unitNet k _ x from ⟨hk, L.port_lt_slot hp (by rw [hw]; exact ht), he⟩)
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ x := by
    intro q hq hqx
    simp only [Hidden.outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil,
      or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    rw [hrows.size] at hj
    obtain ⟨hp, hb, ht8⟩ := Hidden.outRow_bound hrows hj hk ht
    have h0 := Hidden.flat_count_msg Lf F F.parse.rowWires hrows.1 hpos hg hp hb ht8 x
    rw [← hq1, hqx, hF x] at h0
    simp at h0
  have hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
    intro j
    have e : deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
      rw [hda, deltaEntry_filter (fun j' => by simp [hpc]) j, Hidden.deltaEntry_append_out hout, deltaEntry_eq_count,
        delta_flat_eq _ _ F.parse.typed hv1 F.parse.portWords F.parse.v1Segs, hF]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  have hx : x < 2 ^ (L.model regs).kLog := hat.lt R.bound (R.range (R.rowNet_lt hn)).1
  exact block_of_rows hz o hx fun j => inBit_rows R hn hat hl hself hx hδ j

/-- `Hm.row_copies_flat` at the statement: the four copies into VU `g`'s row `p`. -/
theorem row_copies_flatP (hupv : 0 < st.c.upv)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    (∀ t < 512, blockBit (L.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (HmRow.midstate st.c.ports[p]!.words) t) ∧
    (∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (L.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t)) ∧
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (L.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t)) ∧
    (∀ t < 1024, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  have hseg : ∀ q < st.c.ports.size, st.c.ports[q]!.segs = none := fun q hq =>
    F.parse.v1Segs _ (getElem_mem_toList' hq) (hv1 _ (getElem_mem_toList' hq)).1
  have h0 := hpos p hp
  have hpin := pin_bit L hz o
  have hzero := zero_bitP L F Lf hupv (by omega) hz o
  refine ⟨fun t ht => ?_, fun k hk t ht => ?_, fun t ht => ?_, fun t ht => ?_⟩
  · have hs : shaSub st.c p 0 < 3 := (rowSlot_lt hp h0).2
    have hf : (rowCopies st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
          (fun e => e.1 = shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
        [(shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t,
          constSrc st.c st.pin (HmRow.midstate st.c.ports[p]!.words) t)] := R.cv0_filter hpos hseg hg hp ht
    rw [inBit_copyLP L F Lf lp hrows hda hv1 hrw hz o (Or.inl rfl) (R.shaIn_at hg hp h0 ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero
  · have hs : shaSub st.c p (k + 1) < 3 := (rowSlot_lt hp hk).2
    have hf := R.chain_filter hpos hseg (pin := st.pin) (k := k + 1) hg hp (by omega) hk ht
    rw [chainDst_eq R.rowNets hs] at hf
    rw [inBit_copyLP L F Lf lp hrows hda hv1 hrw hz o (Or.inl rfl) (R.shaIn_at hg hp hk ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    rfl
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := R.hmcv_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : Flat.wireDst st.c g (hmWire st.c.ports st.c.shaNet st.c.hmNet p) = st.c.slotCol st.c.hmNet g p :=
      hmWireDst_eq R.rowNets g p
    rw [hx] at hf
    rw [inBit_copyLP L F Lf lp hrows hda hv1 hrw hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    unfold Flat.wireSrc hmWire shaBase shaOut shaSub
    simp only [rowSlots_size hp]
  · rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
    have hf := R.pad_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
      rw [padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx] at hf
    rw [inBit_copyLP L F Lf lp hrows hda hv1 hrw hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero

/-- **A row's `b ‖ c`** (`Hm.row_bc_flat`) in every block of a satisfying witness of the statement with the
verifier's public inputs. -/
theorem row_bc_flatP (hupv : 0 < st.c.upv)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hcv0, hchain, hcvh, hpad⟩ := row_copies_flatP L F Lf lp hrows hda hupv hv1 hz o hg hp
  obtain ⟨hv, -, hn⟩ := hv1 _ (getElem_mem_toList' hp)
  refine port_commit_rows st.c _ g p (F.parse.v1Segs _ (getElem_mem_toList' hp) hv) hn
    (F.parse.portWords _ (getElem_mem_toList' hp) hv) (rowSlots_size hp).ge R.rowNets
    (fun k hk => (rowSlot_lt hp hk).2) (fun k hk => ?_) ?_ hcv0 hchain hcvh hpad
  · exact shaSlot_flatP L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)
  · exact hmSlot_flatP L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))

end HmP

end FlockSoundness.Discharge.ZkPubIn
