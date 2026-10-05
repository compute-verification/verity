import FlockSoundness.Discharge.Hidden.Wide.Copies
import FlockSoundness.Discharge.Hidden.Wide.RowCommit
import FlockSoundness.Discharge.Hm.Bc
import FlockSoundness.Protocol.Discharge.Hidden.Wide.RowBc

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

open Hm Compose Flock.DeriveAll Types.Dag

/-- A row of `16 · words` bits has `2 · words` bytes, `sha512/row/v1` or `sha512/row/v2`. -/
theorem bytes_of_bits {p : Port} (hb : p.bits = 16 * p.words) : p.bytes = 2 * p.words := by
  unfold Port.bytes
  split <;> omega

/-- `Hm.constBit_v1` for any row of `2 · words` bytes: its constant bits past its `n` blocks are the padding block's. -/
theorem constBit_row {p : Port} {n t : ℕ} (hs : p.segs = none) (hby : p.bytes = 2 * p.words) (hb : p.bits = 16 * p.words)
    (hw : 2 * p.words = 128 * n) (ht : t < 1024) :
    HmRow.constBit p (HmRow.tailOf p) n t = some (wordsBit (HmRow.padWords p.words) t) := by
  have hj : (HmRow.blockBit t).1 = 8 * (t / 64) + 7 - t % 64 / 8 := rfl
  have hb8 : (HmRow.blockBit t).2 = t % 8 := rfl
  unfold HmRow.constBit HmRow.tailOf
  simp only [hby, hb, hs]
  rw [ite_eq_right (fun h => absurd h (by rw [hj, hb8]; omega)), padTail_eq (by omega), HmRow.padWords,
    wordsBit_eq _ (n := 16) (by omega), decide_eq_true (by omega), Bool.true_and,
    show 128 * n + (HmRow.blockBit t).1 - 2 * p.words = 8 * (t / 64) + 7 - t % 64 / 8 by rw [hj]; omega, hb8, wbit,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some, Option.getD_some,
    wordsBE_getElem! (by omega), getLsbD_wordL _ _ _ (by omega), get!_eq_getD,
    show 0 + 8 * (t / 64) + (7 - t % 64 / 8) = 8 * (t / 64) + 7 - t % 64 / 8 by omega,
    show t % 64 % 8 = t % 8 by omega]
  congr 1
  generalize ((HmRow.padBlock (128 + 2 * p.words)).data.toList.getD (8 * (t / 64) + 7 - t % 64 / 8) 0).toNat = x
  rw [Nat.testBit, Nat.one_and_eq_mod_two]
  rcases Nat.mod_two_eq_zero_or_one (x >>> (t % 8)) with h | h <;> simp [h]

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **The four copies into VU `g`'s row `p`**, for a row of `16 · words` bits in `2 · words = 128 · chunks` bytes and a
compression or more, in every block of a satisfying witness of the untyped statement: the first compression's `cv` is
the row prefix's midstate (`HmRow.prefixOf`), each later compression's `cv` the previous one's output, the `hm96`
slot's `cv` the last compression's output, and its `pad` the padding block. -/
theorem row_copies_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hb : st.c.ports[p]!.bits = 16 * st.c.ports[p]!.words)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks) (h0 : 0 < st.c.ports[p]!.chunks) :
    (∀ t < 512, blockBit (L.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
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
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  refine ⟨fun t ht => ?_, fun k hk t ht => ?_, fun t ht => ?_, fun t ht => ?_⟩
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
  · rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
    have hf := pad_filterAll R hu (pin := st.pin) hg hp ht
    rw [rowSlots_size hp, constBit_row (hu _ (getElem_mem_toList' hp)) (bytes_of_bits hb) hb hw ht] at hf
    have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
      rw [padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx] at hf
    simp only [bitCopy] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero

/-- **The two copies into VU `g`'s row `p` of no compression**, in every block of a satisfying witness of the untyped
statement: the `hm96` slot's `cv` is the row prefix's midstate, and its `pad` the padding block. -/
theorem row_copies_tgt0 (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hb : st.c.ports[p]!.bits = 16 * st.c.ports[p]!.words)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks) (h0 : st.c.ports[p]!.chunks = 0) :
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ t < 1024, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) := by
  have R := RowRanges.ofL L F.parse F.check
  have hrw : RowWires st.c := F.parse.rowWires
  have hpin := pin_bit L hz o
  have hzero := zero_bit L F hupv (by omega) hz o
  refine ⟨fun t ht => ?_, fun t ht => ?_⟩
  · rw [(hmIn_eq R.rowNets).1, Nat.add_zero]
    have hf := hmcv0_filterAll R hu (pin := st.pin) hg hp h0 ht
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := t) (by omega)) (hmNet_ofRows R.rowNets)
      (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero
  · rw [(hmIn_eq R.rowNets).2.1, Nat.add_assoc]
    have hf := pad_filterAll R hu (pin := st.pin) hg hp ht
    rw [rowSlots_size hp, constBit_row (hu _ (getElem_mem_toList' hp)) (bytes_of_bits hb) hb hw ht] at hf
    have hx : padCol st.c g p + t = st.c.slotCol st.c.hmNet g p + (512 + t) := by
      rw [padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx] at hf
    simp only [bitCopy] at hf
    rw [inBit_copyAll L F hda hrw hu hz o (Or.inr rfl) (R.hmIn_at hg hp (off := 512 + t) (by omega))
      (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) (hmLaid_self (by omega)) hf]
    exact blockBit_constSrc _ _ hpin hzero

/-- **A row's `b ‖ c` under its own prefix**, in every block of a satisfying witness of the untyped statement, for any
rows: at a row of `16 · words` bits in `2 · words = 128 · chunks` bytes (`sha512/row/v1` or `sha512/row/v2`), the
commit string of the row's prefix `HmRow.prefixOf` and the message its `sha512x3` slots read, under the salt its `hm96`
slot reads. -/
theorem row_bc_tgt (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hb : st.c.ports[p]!.bits = 16 * st.c.ports[p]!.words)
    (hw : 2 * st.c.ports[p]!.words = 128 * st.c.ports[p]!.chunks) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      ((HmRow.prefixOf st.c.ports[p]!).data.toList ++
        bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  have R := RowRanges.ofL L F.parse F.check
  have hhm := hmSlot_flat L F hz o hg (by rw [R.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _))
  rcases Nat.eq_zero_or_pos st.c.ports[p]!.chunks with h0 | h0
  · obtain ⟨hcvh, hpad⟩ := row_copies_tgt0 L F hda hupv hu hz o hg hp hb hw h0
    have hw0 : st.c.ports[p]!.words = 0 := by omega
    rw [hw0] at hpad ⊢
    exact port_commit_rows0 st.c _ g p (size_prefixOf _) R.rowNets hhm hcvh hpad
  · obtain ⟨hcv0, hchain, hcvh, hpad⟩ := row_copies_tgt L F hda hupv hu hz o hg hp hb hw h0
    exact port_commit_rowsP st.c _ g p (size_prefixOf _) (hu _ (getElem_mem_toList' hp)) h0 hw (rowSlots_size hp).ge R.rowNets
      (fun k hk => (rowSlot_lt hp hk).2)
      (fun k hk => shaSlot_flat L F hz o hg (by rw [R.perVu_sha]; exact (rowSlot_lt hp hk).1)) hhm hcv0 hchain hcvh
      hpad

end Bc

end FlockSoundness.Discharge.Hidden
