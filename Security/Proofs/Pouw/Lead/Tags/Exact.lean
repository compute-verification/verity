import Definitions.Pouw.Lead.Layout
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-!
# The lead's committed words fit a 32-bit register

The model's words (`Pouw.Lead.word`) are integers; the served kernel holds each in a 32-bit register. With data in
range (`DataInRange`: a step's dot product of at most 28 int8 lanes), a step moves a word by at most
`28 · 127² + 16,479`: every step's coin lanes are at most `127 · (127 + 1 + 1) = 16,383` under `-rot`'s weights
(`laneTerm_abs`), and the coin term adds the tag lane's `2` at a later step and `72 + 24` at step 0 (`coinTerm_abs`).
The column bias adds at most `2S + 14` (`bias_abs`). So every word is within `468,093 · S + 14` of zero (`word_abs`),
inside `[−2^31, 2^31)` for at most 4,587 steps (`word_int32`): at the lead, for an inner dimension of at most 128,429
(`lead_word_int32`).
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Finset Pouw.Lead

variable {L : Pouw.Lead.Layout}

theorem abs_mul_le_of {x y a b : ℤ} (hx : |x| ≤ a) (hy : |y| ≤ b) : |x * y| ≤ a * b := by
  rw [abs_mul]
  exact mul_le_mul hx hy (abs_nonneg _) ((abs_nonneg _).trans hx)

theorem bal13_abs (z : ℤ) (q : Fin 4) : |bal13 z q| ≤ 6 := by
  unfold bal13
  have h0 := Int.emod_nonneg ((z + 14280) / 13 ^ (q : ℕ)) (by norm_num : (13 : ℤ) ≠ 0)
  have h1 := Int.emod_lt_of_pos ((z + 14280) / 13 ^ (q : ℕ)) (by norm_num : (0 : ℤ) < 13)
  rw [abs_le]; constructor <;> linarith

theorem G_abs (ω : Coins L) (q : Fin 4) (j : Fin L.n) : |ω.G q j| ≤ 3 := by
  unfold Coins.G
  have := (ω.2.2.1 q j).isLt
  rw [abs_le]; constructor <;> omega

theorem s_abs (he : L.e ≤ 7) (ω : Coins L) (t : Fin L.S) (k : Fin L.tiles) (ℓ : Fin 3) (j : Fin L.n) :
    |ω.s t k ℓ j| ≤ 127 := by
  unfold Coins.s
  have hu := (ω.2.2.2 t k ℓ j).isLt
  have hp : (2 : ℤ) ^ L.e ≤ 128 := by
    have : (2 : ℕ) ^ L.e ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) he
    exact_mod_cast this
  have hp' : (1 : ℤ) ≤ 2 ^ L.e := one_le_pow₀ (by norm_num)
  have hu' : ((ω.2.2.2 t k ℓ j : ℕ) : ℤ) < 2 ^ L.e := by exact_mod_cast hu
  have hu0 : (0 : ℤ) ≤ ((ω.2.2.2 t k ℓ j : ℕ) : ℤ) := Int.natCast_nonneg _
  rw [abs_le]; constructor <;> linarith

theorem rotWeight_abs_sum (i : ℕ) : ∑ ℓ : Fin 3, |rotWeight i ℓ| = 129 := by
  rw [Fin.sum_univ_three]
  unfold rotWeight
  rcases Nat.mod_two_eq_zero_or_one i with h | h <;> simp [h]

theorem laneTerm_abs (he : L.e ≤ 7) (ω : Coins L) (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    |laneTerm L ω i τ j| ≤ 16383 := by
  unfold laneTerm
  refine (abs_sum_le_sum_abs _ _).trans ?_
  calc ∑ ℓ : Fin 3, |rotWeight i ℓ *
        (if h : (i : ℕ) / L.tile < L.tiles then ω.s τ ⟨(i : ℕ) / L.tile, h⟩ ℓ j else 0)|
      ≤ ∑ ℓ : Fin 3, |rotWeight i ℓ| * 127 := sum_le_sum fun ℓ _ => by
        rw [abs_mul]
        refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
        split_ifs
        · exact s_abs he ω τ _ ℓ j
        · simp
    _ = 16383 := by rw [← sum_mul, rotWeight_abs_sum]; norm_num

theorem coinTerm_abs (he : L.e ≤ 7) (ω : Coins L) (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    |coinTerm L ω i τ j| ≤ 16479 := by
  unfold coinTerm
  have hD := laneTerm_abs he ω i τ j
  refine (abs_add_le _ _).trans ?_
  by_cases h0 : (τ : ℕ) = 0
  · rw [ite_eq_left h0]
    have hA : |∑ q : Fin 4, rowTag i q * ω.G q j| ≤ 72 :=
      (abs_sum_le_sum_abs _ _).trans <| (sum_le_card_nsmul _ _ 18 fun q _ =>
        abs_mul_le_of (bal13_abs _ q) (G_abs ω q j)).trans (by simp)
    have hB : |∑ q : Fin 4, ((ω.2.1 i q : ℕ) : ℤ) * colCode j q| ≤ 24 := by
      refine (abs_sum_le_sum_abs _ _).trans <| (sum_le_card_nsmul _ _ 6 fun q _ => ?_).trans (by simp)
      have hb : |((ω.2.1 i q : ℕ) : ℤ)| ≤ 1 := by
        have := (ω.2.1 i q).isLt
        rw [abs_le]; constructor <;> omega
      have h := abs_mul_le_of hb (bal13_abs (((j : ℕ) + 1 : ℕ) : ℤ) q)
      rw [one_mul] at h
      exact h
    have := abs_add_le (∑ q : Fin 4, rowTag i q * ω.G q j) (∑ q : Fin 4, ((ω.2.1 i q : ℕ) : ℤ) * colCode j q)
    linarith
  · rw [ite_eq_right h0]
    have hb : |1 + ((ω.1 i τ : ℕ) : ℤ)| ≤ 2 := by
      have := (ω.1 i τ).isLt
      rw [abs_le]; constructor <;> omega
    linarith

theorem bias_abs (S j : ℕ) : |bias S j| ≤ 2 * S + 14 := by
  unfold bias
  have hC : |∑ q : Fin 4, colCode j q| ≤ 24 :=
    (abs_sum_le_sum_abs _ _).trans <| (sum_le_card_nsmul _ _ 6 fun q _ => bal13_abs _ q).trans (by simp)
  rw [abs_le] at hC ⊢
  constructor <;> omega

/-- **Every word is within `468,093 · S + 14` of zero**, with data in range and coin lanes of at most 7 bits. -/
theorem word_abs (he : L.e ≤ 7) {κ : Fin L.m → Fin L.S → Fin L.n → ℤ} (hκ : DataInRange L κ) (ω : Coins L)
    (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) : |word L κ ω i t j| ≤ 468093 * L.S + 14 := by
  unfold word
  have hsum : |∑ τ : Fin L.S, if τ ≤ t then κ i τ j + coinTerm L ω i τ j else 0| ≤ 468091 * L.S := by
    refine (abs_sum_le_sum_abs _ _).trans ((sum_le_card_nsmul _ _ 468091 fun τ _ => ?_).trans ?_)
    · split_ifs
      · have := coinTerm_abs he ω i τ j
        have hk := hκ i τ j
        refine (abs_add_le _ _).trans ?_
        have : |κ i τ j| ≤ 28 * 127 ^ 2 := abs_le.2 hk
        linarith
      · simp
    · simp [mul_comm]
  have hb := bias_abs L.S j
  refine (abs_add_le _ _).trans ?_
  rw [abs_neg]
  linarith

/-- **Every word fits a 32-bit register** for at most 4,587 steps. -/
theorem word_int32 (he : L.e ≤ 7) (hS : L.S ≤ 4587) {κ : Fin L.m → Fin L.S → Fin L.n → ℤ}
    (hκ : DataInRange L κ) (ω : Coins L) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    -2 ^ 31 ≤ word L κ ω i t j ∧ word L κ ω i t j < 2 ^ 31 := by
  have h := abs_le.1 (word_abs he hκ ω i t j)
  have hS' : (L.S : ℤ) ≤ 4587 := by exact_mod_cast hS
  constructor <;> linarith [h.1, h.2]

/-- **At the lead every word fits a 32-bit register** for an inner dimension of at most 128,429. -/
theorem lead_word_int32 {T m n K : ℕ} (hK : K ≤ 128429)
    {κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ} (hκ : DataInRange _ κ)
    (ω : Coins (lead T m n K)) (i : Fin (lead T m n K).m) (t : Fin (lead T m n K).S) (j : Fin (lead T m n K).n) :
    -2 ^ 31 ≤ word _ κ ω i t j ∧ word _ κ ω i t j < 2 ^ 31 :=
  word_int32 (by simp [lead]) (by simp only [lead, stepsFor]; omega) hκ ω i t j

end Pouw.SecurityProofs.Lead.Tags
