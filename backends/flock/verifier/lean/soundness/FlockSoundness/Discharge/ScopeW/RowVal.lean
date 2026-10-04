import FlockSoundness.Discharge.ScopeW.Chain
import FlockSoundness.Discharge.ScopeW.RowEnc
import FlockSoundness.Discharge.Hidden.Rows

set_option Elab.async false

/-!
# A row's value and the bytes its digest hashes

A drawn unit's row value is a layout value (`Layout.widthOf`, `Layout.padBits`) of a row's `16 · words` bits. Its
encoding under the row's schema (`rowEncOf`) is the message the row's `hm96` slot commits to (`rowMsgOf`) of the row's
bytes:

* `rowPorts_bits`, `parse_bits`: a bit row's length is below `2^64` (the parse bounds its compressions by
  `HmRow.MAX_COMPS`);
* `bitShaped_row`: a layout value of fewer than `2^64` bits is a bit row's (`BitShaped`);
* `rowEncOf_row`: at a row of a widened shape (`rowShapeW`), `rowEncOf` of the value of bits `b` is `rowMsgOf` of `b`'s
  bytes.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3

set_option maxHeartbeats 1000000 in
/-- **`HmRow.rowPorts`' bit rows are of fewer than `2^64` bits**: their compressions are at most `HmRow.MAX_COMPS`. -/
theorem rowPorts_bits {mj : Lean.Json} {typed : Bool} {ports : Array Port} (h : HmRow.rowPorts mj typed = .ok ports) :
    ∀ p ∈ ports.toList, p.v2 = true → p.bits < 2 ^ 64 := by
  unfold HmRow.rowPorts at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) v2 one at h
  obtain ⟨s, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp1] at h
  obtain ⟨listedPorts, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  obtain ⟨lengths, h⟩ : ∃ l, jp2 l = .ok ports := by
    split at h
    · obtain ⟨s, -, h⟩ := except_bind_ok h
      extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpv at h
      replace h := ite_throw_ok h
      simp (config := { zeta := false }) only [jpv] at h
      obtain ⟨l, -, h⟩ := except_bind_ok h
      exact ⟨l, h⟩
    · exact ⟨_, h⟩
  simp (config := { zeta := false }) only [jp2] at h
  obtain ⟨ports', hmap, h⟩ := except_bind_ok h
  obtain ⟨listed, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) bad jp3 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp3] at h
  obtain ⟨u, -, h⟩ := except_bind_ok h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h
  replace h := ite_throw_ok h
  simp (config := { zeta := false }) only [jp4] at h
  cases h
  intro p hp hv
  obtain ⟨i, -, hf⟩ := array_mapM_ok hmap p hp
  obtain ⟨a, -, hf⟩ := except_bind_ok hf
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpa at hf
  replace hf := ite_throw_ok hf
  simp (config := { zeta := false }) only [jpa] at hf
  obtain ⟨name, -, hf⟩ := except_bind_ok hf
  obtain ⟨words, -, hf⟩ := except_bind_ok hf
  obtain ⟨comps, -, hf⟩ := except_bind_ok hf
  split at hf
  · extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpn at hf
    replace hf := ite_throw_ok hf
    simp (config := { zeta := false }) only [jpn] at hf
    cases hf
    cases hv
  · rename_i len seg _
    extract_lets (config := { underBinder := false, onlyGivenNames := true }) serial segs want jpw at hf
    obtain ⟨hbad, hf⟩ := ite_throw_ok' hf
    simp (config := { zeta := false }) only [jpw] at hf
    cases hf
    simp only [want, Bool.or_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, not_or, Decidable.not_not,
      Prod.mk.injEq] at hbad
    obtain ⟨hmax, -, -⟩ := hbad
    have hser : serial ≤ HmRow.MAX_COMPS := by
      rcases hsg : segs with _ | ⟨s', o⟩
      · simp only [hsg] at hmax
        cases seg
        · simpa using hmax
        · simp at hmax
      · simp only [segs] at hsg
        split at hsg
        · rename_i hc
          simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
          exact hc.2
        · cases hsg
    simp only [serial, HmRow.MAX_COMPS] at hser
    show len < 2 ^ 64
    omega

/-- `rowPorts_bits` at a parsed circuit's rows. -/
theorem parse_bits {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : ∀ p ∈ c.ports.toList, p.v2 = true → p.bits < 2 ^ 64 := by
  obtain ⟨⟨mj, typed, hr⟩, -⟩ := Hidden.parse_rowPorts h
  exact rowPorts_bits hr

/-- **A layout value of fewer than `2^64` bits is a bit row's.** -/
theorem bitShaped_row {W K : ℕ} (hKW : K ≤ W) (hK : K < 2 ^ 64) (b : Fin K → Bool) :
    BitShaped (Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) := by
  have hw : (Layout.widthOf W K).val = K := by simp only [Layout.widthOf]; omega
  refine ⟨by rw [hw]; exact hK, fun i hi => ?_⟩
  change (Layout.widthOf W K).val ≤ i.val at hi
  rw [hw] at hi
  simp only [Layout.padBits, show ¬ i.val < K by omega, ↓reduceDIte]

namespace RowVal

theorem testBit_bitsVal : ∀ (bs : List Bool) (t : ℕ), (HmRow.bitsVal bs).testBit t = bs.getD t false
  | [], t => by simp [HmRow.bitsVal]
  | b :: bs, 0 => by
    rw [HmRow.bitsVal, Nat.testBit_zero]
    cases b <;> simp
  | b :: bs, t + 1 => by
    rw [HmRow.bitsVal, Nat.testBit_succ, show ((if b then 1 else 0) + 2 * HmRow.bitsVal bs) / 2 = HmRow.bitsVal bs by
      split <;> omega, testBit_bitsVal bs t]
    simp

/-- **Eight bits' value is their byte.** -/
theorem bitsVal_byte {bs : List Bool} (hl : bs.length = 8) :
    (HmRow.bitsVal bs).toUInt8 = Binding.byteOf fun t => bs.getD t.val false := by
  have hlt : HmRow.bitsVal bs < 2 ^ 8 := hl ▸ RowV2.bitsVal_lt bs
  apply UInt8.toNat_inj.1
  rw [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt hlt]
  apply Nat.eq_of_testBit_eq
  intro t
  by_cases ht : t < 8
  · rw [testBit_bitsVal]
    exact (Layout.byteOf_bit (fun t : Fin 8 => bs.getD t.val false) ⟨t, ht⟩).symm
  · rw [Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (by omega))),
      Nat.testBit_eq_false_of_lt (lt_of_lt_of_le (UInt8.toNat_lt _) (Nat.pow_le_pow_right (by norm_num) (by omega)))]

/-- **`8 n` bits pack to the `n` bytes whose byte `i` holds bits `8 i` to `8 i + 7`.** -/
theorem packBits_eq {bs : List Bool} {L : List UInt8} (hl : bs.length = 8 * L.length)
    (hL : ∀ i (hi : i < L.length), L[i] = Binding.byteOf fun t => bs.getD (8 * i + t.val) false) :
    HmRow.packBits bs = L := by
  apply List.ext_getElem (by rw [RowV2.packBits_length, hl]; omega)
  intro i hi hi'
  rw [hL i hi']
  simp only [HmRow.packBits, List.getElem_map, List.getElem_range]
  rw [bitsVal_byte (by rw [List.length_take, List.length_drop]; omega)]
  congr 1
  funext t
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt t.isLt, List.getElem?_drop]

theorem length_valBits_row {W K : ℕ} (hKW : K ≤ W) (b : Fin K → Bool) :
    (valBits ((Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) : Layout.BitVal W)).length = K := by
  rw [length_valBits]; simp only [Layout.widthOf]; omega

theorem getD_valBits_row {W K : ℕ} (hKW : K ≤ W) (b : Fin K → Bool) {x : ℕ} (hx : x < K) :
    (valBits ((Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) : Layout.BitVal W)).getD x false =
      b ⟨x, hx⟩ := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by rw [length_valBits_row hKW]; exact hx),
    Option.getD_some]
  simp only [valBits, List.getElem_ofFn, Layout.padBits, hx, ↓reduceDIte]

end RowVal

open RowVal

/-- **A row value's encoding is its row's message**: at a row of a widened shape (`rowShapeW`), with no segments at a
`sha512/row/v1` row (`ParseFacts.v1Segs`) and a bit row of fewer than `2^64` bits (`parse_bits`), `rowEncOf` of the
layout value of `16 · words` bits `b` is `rowMsgOf` of `b`'s `2 · words` bytes. -/
theorem rowEncOf_row {p : Port} (hs : rowShapeW p = true) (hu : p.v2 = false → p.segs = none)
    (hb : p.v2 = true → p.bits < 2 ^ 64) {W K : ℕ} (hK : K = 16 * p.words) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowEncOf p (Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) =
      rowMsgOf p (List.ofFn fun j : Fin (2 * p.words) => Binding.byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩) := by
  have hbits := rowShapeW_bits hs
  have hlen := length_valBits_row hKW b
  have hw : (Layout.widthOf W K).val = K := by simp only [Layout.widthOf]; omega
  rcases hv : p.v2 with _ | _
  · have hsn := hu hv
    rw [rowMsgOf_unseg hsn]
    simp only [rowEncOf, HmRow.prefixOf, hsn, hv, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
    exact Layout.rowEnc_row hK hKW b
  have hsh := bitShaped_row hKW (by have := hb hv; omega) b
  rcases hsg : p.segs with _ | segs
  · rw [rowMsgOf_unseg hsg]
    simp only [rowEncOf, HmRow.prefixOf, hsg, hv, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, rowEncV2, hsh,
      RowV2.enc, hlen]
    congr 1
    · rw [hK, hbits]
    · refine packBits_eq (by rw [hlen, List.length_ofFn]; omega) fun i hi => ?_
      rw [List.length_ofFn] at hi
      rw [List.getElem_ofFn]
      congr 1
      funext t
      have := t.isLt
      rw [getD_valBits_row hKW b (by omega)]
  have hseg : RowSegWhole p := by
    rcases rowShapeW_spec hs with h | h | h | h
    · rw [h.1] at hv; cases hv
    · rw [h.2.1] at hsg; cases hsg
    · rw [h.2.1] at hsg; cases hsg
    · exact h
  have hS : segsOf p = (HmRow.segShape p.bits).1 := by simp [segsOf, hseg.2.1]
  have hcount : segCount ((Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) : Layout.BitVal W) =
      (segsOf p).size := by
    rw [segCount, hw, hS, Chain.segShape_size, hK, hbits, HmRow.ROW_SEG_BITS]
    omega
  simp only [rowEncOf, rowMsgOf, HmRow.prefixOf, hsg, Option.isSome_some, ↓reduceIte, rowEncSeg, hsh]
  rw [hcount]
  refine congrArg₂ (· ++ ·) (by rw [hw, hK, hbits]) (List.flatMap_congr fun j hj => ?_)
  · rw [List.mem_range] at hj
    obtain ⟨-, hm1, hm2, hm3, -⟩ := Chain.seg_geom hseg hj
    have hm : ((segsOf p)[j]!).1 = min (p.bits - 8192 * j) 8192 := by
      have := Chain.segShape_get p.bits (j := j) (by rw [← hS]; exact hj)
      rw [← hS] at this
      exact this
    have hml : (segBits ((Layout.widthOf W K, (Layout.padBits b : Fin W → Bool)) : Layout.BitVal W) j).length =
        ((segsOf p)[j]!).1 := by
      rw [segBits, List.length_take, List.length_drop, hlen, hm, HmRow.ROW_SEG_BITS]
      omega
    have hdl : (segData p (List.ofFn fun j : Fin (2 * p.words) => Binding.byteOf fun t =>
        b ⟨8 * j.val + t.val, by omega⟩) j).length = ((segsOf p)[j]!).1 / 8 := by
      rw [segData, List.length_take, List.length_drop, List.length_ofFn]
      omega
    refine congrArg (fun m => Binding.digBytes (Refine.H512 m)) ?_
    rw [segMsg, segMsgOf, hcount, hml]
    refine congrArg₂ (· ++ ·) rfl (packBits_eq (by rw [hml, hdl]; omega) fun i hi => ?_)
    rw [hdl] at hi
    simp only [segData, List.getElem_take, List.getElem_drop, List.getElem_ofFn]
    congr 1
    funext t
    have := t.isLt
    rw [segBits, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt (by rw [HmRow.ROW_SEG_BITS]; omega),
      List.getElem?_drop, ← List.getD_eq_getElem?_getD, getD_valBits_row hKW b (by rw [HmRow.ROW_SEG_BITS]; omega)]
    exact congrArg b (Fin.ext (by simp only [HmRow.ROW_SEG_BITS]; omega))

end FlockSoundness.Discharge.ScopeW
