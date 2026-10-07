import Proofs.Core.TC.Proofs.SemGroup

/-!
# The FP32 word at the boundary

`decode_eq`: the relation's `decode_state` of an accumulator word is the canonical state of `fp32_value`'s value, which
is admissible. `decode_pack`: `decode_state` of `pack_value` is the canonical state, for an admissible value.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

theorem and_mask (x n : Nat) : x &&& (2 ^ n - 1) = x % 2 ^ n := Nat.and_two_pow_sub_one_eq_mod x n

/-- The three fields of an FP32 word below `2^32`, as arithmetic. -/
theorem fields (w : Nat) (hw : w < 2 ^ 32) :
    w >>> 31 = w / 2 ^ 31 ∧ (w >>> 23) &&& 0xFF = w / 2 ^ 23 % 2 ^ 8 ∧ w &&& 0x7FFFFF = w % 2 ^ 23 ∧
      w >>> 31 < 2 := by
  refine ⟨Nat.shiftRight_eq_div_pow _ _, ?_, ?_, ?_⟩
  · rw [Nat.shiftRight_eq_div_pow, show (0xFF : Nat) = 2 ^ 8 - 1 by decide, and_mask]
  · rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by decide, and_mask]
  · rw [Nat.shiftRight_eq_div_pow]; omega

theorem hidden_or (m : Nat) (hm : m < 2 ^ 23) : (1 <<< 23) ||| m = 2 ^ 23 + m := by
  rw [← Nat.shiftLeft_add_eq_or_of_lt hm 1, Nat.one_shiftLeft]

theorem bit23 (m : Nat) (hm : m < 2 ^ 24) : m &&& (1 <<< 23) = 0 ↔ m < 2 ^ 23 := by
  rw [Nat.one_shiftLeft]
  constructor
  · intro h
    refine Nat.lt_of_not_le fun hc => ?_
    have hb : m.testBit 23 = true := by
      rw [Nat.testBit_eq_decide_div_mod_eq]; have : m / 2 ^ 23 = 1 := by omega
      simp [this]
    have := Nat.testBit_and m (2 ^ 23) 23
    rw [h, Nat.testBit_two_pow_self, hb] at this
    simp at this
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.testBit_and, Nat.testBit_two_pow, Nat.zero_testBit]
    by_cases hi : 23 = i
    · subst hi; rw [Nat.testBit_lt_two_pow h]; rfl
    · simp [hi]

theorem decode_eq (w : Nat) (hw : w < 2 ^ 32) :
    decodeState w HOPPER_BF16_WGMMA_K16 = stateOf (-133) (fp32Value w) ∧ Admissible (fp32Value w) := by
  obtain ⟨hs, hE, hm, hs2⟩ := fields w hw
  unfold decodeState fp32Value fp32Class fp32Term
  simp only [HOPPER_BF16_WGMMA_K16]
  have hsb : ((w >>> 31 : Nat) : Int) = if w >>> 31 = 1 then 1 else 0 := by split <;> omega
  generalize w >>> 31 = s at *
  generalize (w >>> 23) &&& 0xFF = E at *
  generalize w &&& 0x7FFFFF = m at *
  have hE8 : E < 256 := by rw [hE]; exact Nat.mod_lt _ (by decide)
  have hm23 : m < 2 ^ 23 := by rw [hm]; exact Nat.mod_lt _ (by decide)
  by_cases h1 : E = 0xFF
  · by_cases h2 : m = 0 <;> simp [h1, h2, stateOf, Admissible, hsb]
  · by_cases h3 : E = 0
    · by_cases h2 : m = 0
      · simp [h1, h3, h2, stateOf, Admissible, ZERO]
      · simp only [h1, h3, h2, ite_false, ite_true, and_true, ne_eq, not_false_eq_true, and_false, stateOf,
          Admissible, FP32_MIN_NORMAL_EXPONENT, FP32_MIN_E]
        refine ⟨by simp [hsb, h2], ?_⟩
        simp [Admissible, h2]
        omega
    · have hor := hidden_or m hm23
      simp only [h1, h3, ite_false, and_false, stateOf, Admissible, hor]
      have hne : 2 ^ 23 + m ≠ 0 := by omega
      refine ⟨by simp [hsb, hne], ?_⟩
      simp [Admissible, h1, h3]
      omega

/-- Three fields packed as an FP32 word. -/
theorem pack_fields (s E x : Nat) (hs : s < 2) (hE : E < 2 ^ 8) (hx : x < 2 ^ 23) :
    (s <<< 31) ||| (E <<< 23) ||| x = s * 2 ^ 31 + E * 2 ^ 23 + x := by
  rw [Nat.or_assoc, ← Nat.shiftLeft_add_eq_or_of_lt hx E]
  have : E <<< 23 + x < 2 ^ 31 := by rw [Nat.shiftLeft_eq]; omega
  rw [← Nat.shiftLeft_add_eq_or_of_lt this s, Nat.shiftLeft_eq, Nat.shiftLeft_eq]
  omega

theorem decode_pack (V : Value) (hV : Admissible V) (hfin : ∀ t, V = .fin t → t.magnitude ≠ 0 → t.exponent ≤ 127) :
    decodeState (packValue V) HOPPER_BF16_WGMMA_K16 = stateOf (-133) V := by
  cases V with
  | nan => rfl
  | inf n => cases n <;> rfl
  | fin t =>
    simp only [Admissible] at hV
    obtain ⟨hlt, hnz⟩ := hV
    by_cases h0 : t.magnitude = 0
    · simp only [packValue, packFp32, h0, ite_true, stateOf]; rfl
    · obtain ⟨he1, he2, hn⟩ := hnz h0
      have hle := hfin t rfl h0
      have hbit := bit23 t.magnitude hlt
      have hx : t.magnitude &&& 0x7FFFFF = t.magnitude % 2 ^ 23 := by
        rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by decide, and_mask]
      have hsg : (if t.negative then 1 else 0 : Nat) < 2 := by split <;> decide
      have hng : ((if t.negative then 1 else 0 : Nat) : Int) = if t.negative then 1 else 0 := by split <;> rfl
      simp only [packValue, packFp32, h0, ite_false, FP32_MAX_EXPONENT, Int.not_lt.mpr hle, hx]
      by_cases hnorm : 2 ^ 23 ≤ t.magnitude
      · have hb : ¬ t.magnitude &&& (1 <<< 23) = 0 := by rw [hbit]; omega
        simp only [hb, ite_false]
        have hE : (t.exponent + 127).toNat < 2 ^ 8 := by omega
        rw [pack_fields _ _ _ hsg hE (Nat.mod_lt _ (by decide))]
        generalize hW : (if t.negative then 1 else 0) * 2 ^ 31 + (t.exponent + 127).toNat * 2 ^ 23 +
          t.magnitude % 2 ^ 23 = W
        have hWlt : W < 2 ^ 32 := by rw [← hW]; omega
        obtain ⟨f1, f2, f3, _⟩ := fields W hWlt
        have e1 : W / 2 ^ 31 = (if t.negative then 1 else 0) := by omega
        have e2 : W / 2 ^ 23 % 2 ^ 8 = (t.exponent + 127).toNat := by omega
        have e3 : W % 2 ^ 23 = t.magnitude % 2 ^ 23 := by omega
        have n1 : ¬ (t.exponent + 127).toNat = 0xFF := by omega
        have n2 : ¬ (t.exponent + 127).toNat = 0 := by omega
        have hh := hidden_or (t.magnitude % 2 ^ 23) (Nat.mod_lt _ (by decide))
        simp only [decodeState, f1, f2, f3, e1, e2, e3, n1, n2, ite_false, hh, stateOf, h0, State.mk.injEq]
        refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> first | trivial | exact hng | omega
      · have hsub : t.exponent = -126 := by omega
        have hb : t.magnitude &&& (1 <<< 23) = 0 := by rw [hbit]; omega
        simp only [hb, ite_true, hsub]
        have hE : ((-126 : Int) + 127 - 1).toNat = 0 := by decide
        rw [hE, Nat.zero_shiftLeft, Nat.or_zero]
        have hmod : t.magnitude % 2 ^ 23 = t.magnitude := Nat.mod_eq_of_lt (by omega)
        rw [hmod]
        have hor : ((if t.negative then 1 else 0) <<< 31) ||| t.magnitude =
            (if t.negative then 1 else 0) * 2 ^ 31 + t.magnitude := by
          rw [← Nat.shiftLeft_add_eq_or_of_lt (by omega : t.magnitude < 2 ^ 31), Nat.shiftLeft_eq]
        rw [hor]
        generalize hW : (if t.negative then 1 else 0) * 2 ^ 31 + t.magnitude = W
        have hWlt : W < 2 ^ 32 := by rw [← hW]; omega
        obtain ⟨f1, f2, f3, _⟩ := fields W hWlt
        have e1 : W / 2 ^ 31 = (if t.negative then 1 else 0) := by omega
        have e2 : W / 2 ^ 23 % 2 ^ 8 = 0 := by omega
        have e3 : W % 2 ^ 23 = t.magnitude := by omega
        simp only [decodeState, f1, f2, f3, e1, e2, e3, h0, stateOf, hsub, State.mk.injEq]
        simp [hng, FP32_MIN_E, h0]

/-- A group's output packs to a word below `2^32`. -/
theorem pack_lt (V : Value) (hV : Admissible V) : packValue V < 2 ^ 32 := by
  cases V with
  | nan => decide
  | inf n => cases n <;> decide
  | fin t =>
    simp only [Admissible] at hV
    obtain ⟨hlt, hnz⟩ := hV
    by_cases h0 : t.magnitude = 0
    · simp only [packValue, packFp32, h0, ite_true]; decide
    · obtain ⟨he1, he2, hn⟩ := hnz h0
      have hsg : (if t.negative then 1 else 0 : Nat) < 2 := by split <;> decide
      have hx : t.magnitude &&& 0x7FFFFF < 2 ^ 23 := by
        rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by decide, and_mask]; exact Nat.mod_lt _ (by decide)
      simp only [packValue, packFp32, h0, ite_false, FP32_MAX_EXPONENT, Int.not_lt.mpr he2]
      have hE : (if t.magnitude &&& 1 <<< 23 = 0 then t.exponent + 127 - 1 else t.exponent + 127).toNat < 2 ^ 8 := by
        split <;> omega
      rw [pack_fields _ _ _ hsg hE hx]
      omega

end Verity.TC.Proofs
