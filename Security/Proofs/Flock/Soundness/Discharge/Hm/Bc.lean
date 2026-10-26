/-
A row's `b ‖ c` in a satisfying witness of the untyped statement `Stmt.setupH` accepted (`row_bc_setupH`): the commit
string of the row and salt its slots read. Δ's one copy at each `cv` and `pad` input bit (`inBit_copyL`) gives
`port_commit_rows` its four facts (`row_copies_flat`); the constant bits read the pin (`pin_bit`) and the unit range's
forced-zero row (`zero_bit`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Flat

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)

/-- The pin carries 1, in every block of a witness that satisfies the statement off some rows (`SatEx`). -/
theorem pin_bit_ex {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {D : ℕ → Prop}
    {z : Model.Witness (L.model regs).m} (hz : SatEx (L.model regs) D z) (o : Fin (2 ^ (L.model regs).nbl)) :
    blockBit (L.model regs) z o st.pin = true := by
  rw [blockBit_of_lt _ z o L.pin_lt]
  have hpin : (⟨st.pin, L.pin_lt⟩ : Fin (2 ^ (L.model regs).kLog)) = (L.model regs).pin := Fin.ext rfl
  rw [hpin, hz.2.1 o]
  rfl

/-- The pin carries 1, in every block of a satisfying witness. -/
theorem pin_bit {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    blockBit (L.model regs) z o st.pin = true :=
  pin_bit_ex L (hz.ex fun _ => False) o

include F in
/-- The unit range's first slot's forced-zero row carries 0, in every block of a witness that satisfies every row but
the unit input rows (`SatEx`, `UnitIn`), when the range has a slot. -/
theorem zero_bit_ex (hupv : 0 < st.c.upv) (hg : 0 < st.c.g) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : SatEx (L.model regs) (UnitIn st.c) z)
    (o : Fin (2 ^ (L.model regs).nbl)) :
    blockBit (L.model regs) z o (zeroCol st.c) = false := by
  have hs₀ : 0 < st.c.unitRange.count := by rw [L.unit_count]; exact Nat.mul_pos hg hupv
  have h0 := Aliased.zeroPos_block_ex L regs F hs₀ hz (not_unitIn_zero L F 0) o
  unfold zeroCol
  rw [blockBit_of_lt _ z o (y := st.c.unitRange.slot 0 + st.c.unit.useful) (Aliased.zeroPos L regs F hs₀).isLt]
  show FlockSoundness.bit ((L.model regs).block z o (Aliased.zeroPos L regs F hs₀)) = false
  rw [h0]
  rfl

include F in
/-- The unit range's first slot's forced-zero row carries 0, in every block of a satisfying witness, when the range has
a slot. -/
theorem zero_bit (hupv : 0 < st.c.upv) (hg : 0 < st.c.g) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    blockBit (L.model regs) z o (zeroCol st.c) = false :=
  zero_bit_ex L F hupv hg (hz.ex _) o

include F in
/-- **The four copies into template instance `g`'s row `p`**, in every block of a satisfying witness of the untyped statement: the first
compression's `cv` is the row prefix's midstate, each later compression's `cv` the previous one's output, the `hm96`
slot's `cv` the last compression's output, and its `pad` the padding block. -/
theorem row_copies_flat (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
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
  have hzero := zero_bit L F hupv (by omega) hz o
  refine ⟨fun t ht => ?_, fun k hk t ht => ?_, fun t ht => ?_, fun t ht => ?_⟩
  · have hs : shaSub st.c p 0 < 3 := (rowSlot_lt hp h0).2
    have hf : (rowCopies st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
          (fun e => e.1 = shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
        [(shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t,
          constSrc st.c st.pin (HmRow.midstate st.c.ports[p]!.words) t)] := R.cv0_filter hpos hseg hg hp ht
    rw [inBit_copyL L F hv1 hda hrw hz o (Or.inl rfl) (R.shaIn_at hg hp h0 ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero
  · have hs : shaSub st.c p (k + 1) < 3 := (rowSlot_lt hp hk).2
    have hf := R.chain_filter hpos hseg (pin := st.pin) (k := k + 1) hg hp (by omega) hk ht
    rw [chainDst_eq R.rowNets hs] at hf
    rw [inBit_copyL L F hv1 hda hrw hz o (Or.inl rfl) (R.shaIn_at hg hp hk ht) (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) (shaLaid_self (by omega)) hf]
    rfl
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := R.hmcv_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : Flat.wireDst st.c g (hmWire st.c.ports st.c.shaNet st.c.hmNet p) = st.c.slotCol st.c.hmNet g p :=
      hmWireDst_eq R.rowNets g p
    rw [hx] at hf
    rw [inBit_copyL L F hv1 hda hrw hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
      (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    unfold Flat.wireSrc hmWire shaBase shaOut shaSub
    simp only [rowSlots_size hp]
  · rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
    have hf := R.pad_filter hpos hseg (pin := st.pin) hg hp ht
    have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
      rw [padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx] at hf
    rw [inBit_copyL L F hv1 hda hrw hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero

include F in
/-- **A row's `b ‖ c`**, in every block of a satisfying witness of the untyped statement: the commit string of the row
prefix and message its `sha512x3` slots read, and the salt its `hm96` slot reads, for `sha512/row/v1` rows (`hv1`). -/
theorem row_bc_flat (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hcv0, hchain, hcvh, hpad⟩ := row_copies_flat L F hda hupv hv1 hz o hg hp
  obtain ⟨hv, -, hn⟩ := hv1 _ (getElem_mem_toList' hp)
  refine port_commit_rows st.c _ g p (F.parse.v1Segs _ (getElem_mem_toList' hp) hv) hn
    (F.parse.portWords _ (getElem_mem_toList' hp) hv) (rowSlots_size hp).ge R.rowNets
    (fun k hk => (rowSlot_lt hp hk).2) (fun k hk => ?_) ?_ hcv0 hchain hcvh hpad
  · exact shaSlot_flat L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)
  · exact hmSlot_flat L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))

end Bc

/-- A unit that returns the outputs has a slot per template instance at least: the check's power of two of unit slots.
-/
theorem upv_pos {c : Circuit} (C : CheckFacts c) (ho : c.outNet = c.unitNet) : 0 < c.upv := by
  have h := C.unit_pow2 ho
  rw [C.unit_range.2] at h
  unfold isPow2 at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact Nat.pos_of_mul_pos_left h.1

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **A row's `b ‖ c` at the untyped statement `Stmt.setupH` accepted**: in every block of a satisfying witness, template instance `g`'s
row `p`'s `hm96` slot outputs the commit string of the row prefix and message its `sha512x3` slots read, under the salt
its `hm96` slot reads, when the unit returns the outputs (`ho`) and the rows are `sha512/row/v1` rows (`hv1`). -/
theorem row_bc_setupH (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (ho : st.c.outNet = st.c.unitNet)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness ((setupH_layout ht h).model regs).m}
    (hz : ((setupH_layout ht h).model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.rowPrefix st.c.ports[p]!.words).data.toList ++
        bytesOf (blockBit ((setupH_layout ht h).model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit ((setupH_layout ht h).model regs) z o) 192 (saltCol st.c g p)) :=
  row_bc_flat (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) (setupH_spec ht h).2.2.1
    (upv_pos (Aliased.setupH_unitFacts ht h).check ho) hv1 hz o hg hp

end FlockSoundness.Discharge.Hm
