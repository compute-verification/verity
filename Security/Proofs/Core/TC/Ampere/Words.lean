import Proofs.Core.TC.Ampere.Step
import Proofs.Core.TC.Proofs.Words

/-!
# The FP32 word at the boundary, on the Ampere pipeline (floor `-132`)
-/

namespace Verity.TC.Proofs.Ampere

open Verity.TC Verity.TC.Spec Verity.TC.Relation

theorem decode_eq (w : Nat) (hw : w < 2 ^ 32) :
    decodeState w AMPERE_BF16_M16N8K16 = stateOf (-132) (fp32Value w) ∧ Admissible (fp32Value w) := by
  obtain ⟨hs, hE, hm, hs2⟩ := fields w hw
  unfold decodeState fp32Value fp32Class fp32Term
  simp only [AMPERE_BF16_M16N8K16]
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

theorem decode_pack (V : Value) (hV : Admissible V) (hfin : ∀ t, V = .fin t → t.magnitude ≠ 0 → t.exponent ≤ 127) :
    decodeState (packValue V) AMPERE_BF16_M16N8K16 = stateOf (-132) V := by
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

end Verity.TC.Proofs.Ampere
