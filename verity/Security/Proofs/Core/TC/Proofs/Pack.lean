import Proofs.Core.TC.Proofs.Words

/-!
# The unit boundary: `pack_relation` is `pack_value` then `cvt.rn.bf16.f32`

For a canonical final state, `pack_relation`'s predicates force the FP32 word to be `pack_value` of the state's value
(`pack_word`) and the BF16 word to be `cvt_rn_bf16_f32` of it (`pack_sound`), and some columns satisfy them
(`pack_complete`). None of this depends on the pipeline: the floor enters only through states whose magnitude is zero.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- `cvt.rn.bf16.f32`'s arithmetic on a word below `2^32`, as division and remainder. -/
theorem cvt_arith (W b : Nat) (hb : b = W / 65536 % 2) :
    (W >>> 16) &&& 1 = b ∧ ((W + 0x7FFF + b) >>> 16) &&& 0xFFFF = (W + 0x7FFF + b) / 65536 % 65536 := by
  refine ⟨?_, ?_⟩
  · rw [Nat.shiftRight_eq_div_pow, show (1 : Nat) = 2 ^ 1 - 1 by decide, and_mask]; simp [hb]
  · rw [Nat.shiftRight_eq_div_pow, show (0xFFFF : Nat) = 2 ^ 16 - 1 by decide, and_mask]

/-- The rounding half of `pack_relation`: its BF16 word is the RNE rounding of its FP32 word, or `0x7FFF` for a NaN. -/
theorem cvt_part (nan : Int) (hnan : nan = 0 ∨ nan = 1) (c : PackCols) (hW0 : 0 ≤ c.word) (hW1 : c.word < 2 ^ 32)
    (h10 : RANGE c.lo 16) (h11 : RANGE c.hi 16) (h12 : LIN (c.hi * 65536 + c.lo - c.word)) (h13 : BOOL c.b16)
    (h14 : RANGE c.hi2 15) (h15 : LIN (c.hi2 * 2 + c.b16 - c.hi)) (h16 : BOOL c.carry) (h17 : RANGE c.rem 16)
    (h18 : LIN (c.carry * 65536 + c.rem - (c.lo + 0x7FFF + c.b16))) (h19 : BOOL c.wrap) (h20 : RANGE c.rounded 16)
    (h21 : LIN (c.rounded + c.wrap * 65536 - c.hi - c.carry)) (h22 : MUL nan (0x7FFF - c.rounded) (c.bf16 - c.rounded)) :
    c.bf16 = if nan = 1 then (0x7FFF : Int)
      else ((((c.word.toNat + 0x7FFF + ((c.word.toNat >>> 16) &&& 1)) >>> 16) &&& 0xFFFF : Nat) : Int) := by
  have hb := bool_of _ h13
  have hc := bool_of _ h16
  have hw := bool_of _ h19
  unfold RANGE at h10 h11 h14 h17 h20
  unfold LIN at h12 h15 h18 h21
  unfold MUL at h22
  obtain ⟨W, hW⟩ : ∃ W : Nat, c.word = W := ⟨c.word.toNat, (Int.toNat_of_nonneg hW0).symm⟩
  rw [hW] at h12 hW1 ⊢
  simp only [Int.toNat_natCast]
  obtain ⟨e1, e2⟩ := cvt_arith W (W / 65536 % 2) rfl
  rw [e1, e2]
  rcases hnan with rfl | rfl
  · simp only [Int.zero_mul] at h22
    simp only [show ¬ ((0 : Int) = 1) by decide, ite_false]
    omega
  · simp only [ite_true]
    omega

/-- A group's output packs to a word whose class is NaN exactly when the value is NaN. -/
theorem pack_class (V : Value) (hV : Admissible V) : fp32Class (packValue V) = .nan ↔ V = .nan := by
  cases V with
  | nan => simp only [iff_true]; decide
  | inf n => cases n <;> simp only [reduceCtorEq, iff_false] <;> decide
  | fin t =>
    simp only [reduceCtorEq, iff_false]
    simp only [Admissible] at hV
    obtain ⟨hlt, hnz⟩ := hV
    by_cases h0 : t.magnitude = 0
    · simp only [packValue, packFp32, h0, ite_true]; decide
    · obtain ⟨he1, he2, _⟩ := hnz h0
      have hsg : (if t.negative then 1 else 0 : Nat) < 2 := by split <;> decide
      have hx : t.magnitude &&& 0x7FFFFF < 2 ^ 23 := by
        rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by decide, and_mask]; exact Nat.mod_lt _ (by decide)
      simp only [packValue, packFp32, h0, ite_false, FP32_MAX_EXPONENT, Int.not_lt.mpr he2]
      have hE : (if t.magnitude &&& 1 <<< 23 = 0 then t.exponent + 127 - 1 else t.exponent + 127).toNat < 255 := by
        split <;> omega
      rw [pack_fields _ _ _ hsg (by omega) hx]
      generalize hW : (if t.negative then 1 else 0) * 2 ^ 31 +
        (if t.magnitude &&& 1 <<< 23 = 0 then t.exponent + 127 - 1 else t.exponent + 127).toNat * 2 ^ 23 +
        (t.magnitude &&& 0x7FFFFF) = W
      have hWlt : W < 2 ^ 32 := by rw [← hW]; omega
      have hE' : W / 2 ^ 23 % 2 ^ 8 =
          (if t.magnitude &&& 1 <<< 23 = 0 then t.exponent + 127 - 1 else t.exponent + 127).toNat := by
        rw [← hW]; omega
      obtain ⟨_, f2, _, _⟩ := fields W hWlt
      unfold fp32Class
      simp only []
      rw [f2]
      have : W / 2 ^ 23 % 2 ^ 8 ≠ 0xFF := by omega
      simp only [this, ite_false]
      split <;> simp

/-- The packing half of `pack_relation`: its FP32 word is `pack_value` of the state's value. -/
theorem pack_word (Z : Int) (V : Value) (hV : Admissible V) (c : PackCols) (h : packWordHolds (stateOf Z V) c) :
    c.word = (packValue V : Int) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := h
  have hh := bool_of _ h1
  unfold MUL at h4 h5 h6 h7 h8
  unfold LIN at h9
  unfold RANGE at h2
  unfold ISZERO RANGE at h3
  cases V with
  | nan =>
    simp only [stateOf] at h2 h3 h4 h5 h6 h7 h8 h9
    have hfz : c.fin_nz = 0 := by rw [← h6]; simp
    rw [hfz, Int.zero_mul] at h7
    simp only [Int.sub_self, Int.sub_zero, Int.zero_mul] at h8
    show c.word = 2147483647
    omega
  | inf n =>
    simp only [stateOf] at h2 h3 h4 h5 h6 h7 h8 h9
    have hfz : c.fin_nz = 0 := by rw [← h6]; simp
    rw [hfz, Int.zero_mul] at h7
    simp only [Int.sub_self, Int.zero_sub, Int.zero_mul, Int.sub_zero] at h8
    cases n
    · show c.word = 2139095040
      simp at h9; omega
    · show c.word = 4286578688
      simp at h9; omega
  | fin t =>
    simp only [Admissible] at hV
    obtain ⟨hlt, hnz⟩ := hV
    by_cases h0 : t.magnitude = 0
    · simp only [stateOf, h0, ite_true] at h2 h3 h4 h5 h6 h7 h8 h9
      simp only [packValue, packFp32, h0, ite_true]
      have hz : c.z = 1 := by simpa using h3.2
      rw [hz] at h6
      have hfz : c.fin_nz = 0 := by rw [← h6]; simp
      rw [hfz, Int.zero_mul] at h7
      rcases hh with hh | hh <;> rw [hh] at h2 h8 <;> simp at h2 h8 <;> omega
    · obtain ⟨he1, he2, hn⟩ := hnz h0
      simp only [stateOf, h0, ite_false] at h2 h3 h4 h5 h6 h7 h8 h9
      have hz : c.z = 0 := by simpa [h0] using h3.2
      rw [hz] at h4 h6
      have hfz : c.fin_nz = 1 := by rw [← h6]; simp
      rw [hfz, Int.one_mul] at h7
      have hbit := bit23 t.magnitude hlt
      have hsg : (if t.negative then 1 else 0 : Nat) < 2 := by split <;> decide
      have hx : t.magnitude &&& 0x7FFFFF = t.magnitude % 2 ^ 23 := by
        rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by decide, and_mask]
      have hng : ((if t.negative then 1 else 0 : Nat) : Int) = if t.negative then 1 else 0 := by split <;> rfl
      simp only [packValue, packFp32, h0, ite_false, FP32_MAX_EXPONENT, Int.not_lt.mpr he2]
      have hE : (if t.magnitude &&& 1 <<< 23 = 0 then t.exponent + 127 - 1 else t.exponent + 127).toNat < 2 ^ 8 := by
        split <;> omega
      rw [pack_fields _ _ _ hsg hE (by rw [hx]; exact Nat.mod_lt _ (by decide)), hx]
      rcases hh with hh | hh <;> rw [hh] at h2 h4 h7 h8
      · -- no hidden bit: a subnormal
        have hsub : t.magnitude < 2 ^ 23 := by omega
        have hb : t.magnitude &&& 1 <<< 23 = 0 := hbit.mpr hsub
        simp only [hb, ite_true]
        simp only [Int.sub_zero, Int.one_mul] at h4
        rw [← h4, Int.one_mul] at h5
        have : (t.exponent + 127 - 1).toNat = 0 := by omega
        rw [this]
        omega
      · have hnorm : ¬ t.magnitude < 2 ^ 23 := by omega
        have hb : ¬ t.magnitude &&& 1 <<< 23 = 0 := fun e => hnorm (hbit.mp e)
        simp only [hb, ite_false]
        omega

/-- `pack_relation` is `pack_value` then `cvt_rn_bf16_f32`, for every admissible value. -/
theorem pack_sound (Z : Int) (V : Value) (hV : Admissible V) (c : PackCols) (h : packHolds (stateOf Z V) c) :
    c.word = (packValue V : Int) ∧ c.bf16 = (cvtRnBf16F32 (packValue V) : Int) := by
  have hw := pack_word Z V hV c h.1
  refine ⟨hw, ?_⟩
  obtain ⟨h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22⟩ := h.2
  have hlt := pack_lt V hV
  have hnan : (stateOf Z V).nan = 0 ∨ (stateOf Z V).nan = 1 := by cases V <;> simp [stateOf] <;> split <;> simp
  have := cvt_part _ hnan c (by rw [hw]; omega) (by rw [hw]; exact_mod_cast hlt) h10 h11 h12 h13 h14 h15 h16 h17 h18
    h19 h20 h21 h22
  rw [this, hw, Int.toNat_natCast]
  unfold cvtRnBf16F32
  have hc := pack_class V hV
  cases V with
  | nan =>
    have : fp32Class (packValue .nan) = .nan := by decide
    simp [stateOf, this]
  | inf n =>
    have : ¬ fp32Class (packValue (.inf n)) = .nan := fun e => by simpa using hc.mp e
    simp [stateOf, this]
  | fin t =>
    have : ¬ fp32Class (packValue (.fin t)) = .nan := fun e => by simpa using hc.mp e
    simp only [this, ite_false]
    simp only [stateOf]
    split <;> simp

/-- The columns `pack_relation` computes from the final state `st`. -/
def honestPack (st : State) : PackCols :=
  let hidden : Int := if st.M ≥ 2 ^ 23 then 1 else 0
  let low := st.M - hidden * 2 ^ 23
  let z : Int := if st.M = 0 then 1 else 0
  let fin := 1 - st.nan - st.inf
  let nh_nz := (1 - hidden) * (1 - z)
  let fin_nz := fin * (1 - z)
  let Ebias := fin_nz * (st.e + 126 + hidden) + (st.nan + st.inf) * 255
  let mant := fin * low + st.nan * 0x7FFFFF
  let word := st.neg * 2 ^ 31 + Ebias * 2 ^ 23 + mant
  let hi := word / 65536
  let lo := word % 65536
  let b16 := hi % 2
  let t := lo + 0x7FFF + b16
  let carry := t / 65536
  let rounded := (hi + carry) % 65536
  { hidden := hidden, z := z, nh_nz := nh_nz, fin_nz := fin_nz, Ebias := Ebias, mant := mant, word := word,
    hi := hi, lo := lo, b16 := b16, hi2 := hi / 2, carry := carry, rem := t % 65536, rounded := rounded,
    wrap := (hi + carry) / 65536, bf16 := rounded + st.nan * (0x7FFF - rounded) }

theorem bool_iff (x : Int) (h : x = 0 ∨ x = 1) : BOOL x := by
  unfold BOOL MUL; rcases h with rfl | rfl <;> decide

theorem cvt_complete (st : State) (h0 : 0 ≤ (honestPack st).word) (h1 : (honestPack st).word < 2 ^ 32) :
    cvtHolds st.nan (honestPack st) := by
  generalize hc : honestPack st = c at *
  have ehi : c.hi = c.word / 65536 := by rw [← hc]; rfl
  have elo : c.lo = c.word % 65536 := by rw [← hc]; rfl
  have eb : c.b16 = c.hi % 2 := by rw [← hc]; rfl
  have eh2 : c.hi2 = c.hi / 2 := by rw [← hc]; rfl
  have ec : c.carry = (c.lo + 0x7FFF + c.b16) / 65536 := by rw [← hc]; rfl
  have er : c.rem = (c.lo + 0x7FFF + c.b16) % 65536 := by rw [← hc]; rfl
  have ero : c.rounded = (c.hi + c.carry) % 65536 := by rw [← hc]; rfl
  have ew : c.wrap = (c.hi + c.carry) / 65536 := by rw [← hc]; rfl
  have ebf : c.bf16 = c.rounded + st.nan * (0x7FFF - c.rounded) := by rw [← hc]; rfl
  refine ⟨?_, ?_, ?_, bool_iff _ (by omega), ?_, ?_, bool_iff _ (by omega), ?_, ?_, bool_iff _ (by omega), ?_, ?_, ?_⟩
  all_goals (try simp only [RANGE, LIN, MUL])
  all_goals omega

/-- Some columns satisfy `pack_relation` for every admissible value. -/
theorem pack_complete (Z : Int) (V : Value) (hV : Admissible V) : packHolds (stateOf Z V) (honestPack (stateOf Z V)) := by
  have hword : packWordHolds (stateOf Z V) (honestPack (stateOf Z V)) := by
    have hM : 0 ≤ (stateOf Z V).M ∧ (stateOf Z V).M < 2 ^ 24 ∧
        ((stateOf Z V).M ≠ 0 → (stateOf Z V).M < 2 ^ 23 → (stateOf Z V).e = -126) := by
      cases V with
      | nan => simp [stateOf]
      | inf n => simp [stateOf]
      | fin t =>
        simp only [Admissible] at hV
        simp only [stateOf]
        split
        · simp
        · rename_i h0
          obtain ⟨he1, he2, hn⟩ := hV.2 h0
          have h24 := hV.1
          show 0 ≤ (t.magnitude : Int) ∧ (t.magnitude : Int) < 2 ^ 24 ∧
            ((t.magnitude : Int) ≠ 0 → (t.magnitude : Int) < 2 ^ 23 → t.exponent = -126)
          refine ⟨by omega, by omega, fun _ hlt => ?_⟩
          rcases hn with hn | hn
          · omega
          · exact hn
    generalize hst : stateOf Z V = st at *
    obtain ⟨hM0, hM1, hsub⟩ := hM
    unfold packWordHolds honestPack
    simp only []
    refine ⟨bool_iff _ (by split <;> simp), ?_, ⟨?_, rfl⟩, rfl, ?_, rfl, by simp only [MUL]; omega,
      by simp only [MUL]; omega, ?_⟩
    · simp only [RANGE]; split <;> omega
    · simp only [RANGE]; omega
    · simp only [MUL, FP32_MIN_E]
      by_cases h0 : st.M = 0
      · simp [h0]
      · by_cases hlt : st.M < 2 ^ 23
        · have := hsub h0 hlt
          simp [h0, this]
        · have : 8388608 ≤ st.M := by omega
          simp [this]
    · simp only [LIN]; omega
  refine ⟨hword, ?_⟩
  have hw := pack_word Z V hV _ hword
  have hlt := pack_lt V hV
  exact cvt_complete _ (by rw [hw]; omega) (by rw [hw]; exact_mod_cast hlt)

end Verity.TC.Proofs
