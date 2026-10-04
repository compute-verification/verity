import Mathlib.Analysis.Complex.ExponentialBounds
import Pous.SecurityProofs.SecureErasure.Band
import Pous.SecurityProofs.SecureErasure.PagesAudit
import Pous.SecurityProofs.SecureErasure.ReductionDraw

/-!
# Secure erasure of the keyed page fill

`Pous.Guarantees.SecureErasurePagesFamily` and `SecureErasurePages96112`: `page_audit_secure` at `w = w₁·P` wrong bytes
and `secure_erasure_meets_draw` on top of it, at `S = ⌊(18/19)·|C|⌋ ≤ 7947103·P` bits.

**The union term** is at most `2^-130`: `C(B, w) ≤ (e·B/w)^w = (e·2^20/w₁)^(w₁·P) ≤ (a/b)^(w₁·P)`
(`choose_mul_pow_le`), and the per-page certificate `a^w₁ · 2^(7947103 + 8·w₁ + m) ≤ b^w₁ · 2^(2^23)`, raised to the
`P`-th power, leaves `s · 2^S · C(B, w) · 2^-(8·(B − w)) ≤ s · 2^(-m·P) ≤ 2^-130` (`bad_term_le`).
- the family: `w₁ = 30324`, `a/b = 94`, `m = 142`, so it holds at every `P ≥ 1`;
- `P = 96112`: `w₁ = 30335`, `a/b = 9397/100`, `m = 2`, and `2·96112 ≥ 142`.

**The sampling term** is at most `2^-129`: `(B − w)^s · 2^129 ≤ (B + 1 − s)^s ≤ B·(B − 1)⋯(B − s + 1)`. For the
family, `P·(2^20 + 1 − s) ≤ P·2^20 + 1 − s` reduces it to `P = 1`: `(2^20 − 30324)^3430 · 2^129 ≤ (2^20 − 3429)^3430`.

**The foreign content's term** is at most `s · 2^-256 ≤ 2^-130` (`s ≤ 2^12`), as on the band.

The certificates are inequalities between natural numbers of up to about 8.4 million bits, which the kernel decides by
GMP arithmetic (`Nat.ble … = true` by `rfl`).
-/

namespace PousErase

open Pous PousThm1 Pous.PageFill
open scoped ENNReal

/-! ## The binomial bound -/

/-- `C(P·pb, w₁·P) · b^(w₁·P) ≤ a^(w₁·P)` when `e·pb/w₁ ≤ a/b`: `C(B, w) ≤ B^w / w!` and `w^w / w! ≤ e^w`. -/
theorem choose_mul_pow_le (pb P w₁ a b : ℕ) (hP : 0 < P) (hw₁ : 0 < w₁)
    (he : Real.exp 1 * pb * b ≤ a * w₁) :
    (P * pb).choose (w₁ * P) * b ^ (w₁ * P) ≤ a ^ (w₁ * P) := by
  set w := w₁ * P with hw_def
  have hw : (0 : ℝ) < w := by rw [hw_def]; push_cast; positivity
  have h1 : ((P * pb).choose w : ℝ) ≤ ((P * pb : ℕ) : ℝ) ^ w / (w.factorial : ℝ) := Nat.choose_le_pow_div w (P * pb)
  have h2 : (w : ℝ) ^ w / (w.factorial : ℝ) ≤ Real.exp w := Real.pow_div_factorial_le_exp _ hw.le w
  have h3 : ((P * pb).choose w : ℝ) * (w : ℝ) ^ w ≤ ((P * pb : ℕ) : ℝ) ^ w * Real.exp 1 ^ w :=
    calc ((P * pb).choose w : ℝ) * (w : ℝ) ^ w
        ≤ ((P * pb : ℕ) : ℝ) ^ w / (w.factorial : ℝ) * (w : ℝ) ^ w :=
          mul_le_mul_of_nonneg_right h1 (by positivity)
      _ = ((P * pb : ℕ) : ℝ) ^ w * ((w : ℝ) ^ w / (w.factorial : ℝ)) := by ring
      _ ≤ ((P * pb : ℕ) : ℝ) ^ w * Real.exp w := mul_le_mul_of_nonneg_left h2 (by positivity)
      _ = ((P * pb : ℕ) : ℝ) ^ w * Real.exp 1 ^ w := by rw [Real.exp_one_pow]
  have h4 : ((P * pb : ℕ) : ℝ) * Real.exp 1 * b ≤ a * w := by
    have := mul_le_mul_of_nonneg_left he (Nat.cast_nonneg P : (0 : ℝ) ≤ P)
    rw [hw_def]
    push_cast
    nlinarith
  have h5 : ((P * pb).choose w : ℝ) * (b : ℝ) ^ w * (w : ℝ) ^ w ≤ (a : ℝ) ^ w * (w : ℝ) ^ w :=
    calc ((P * pb).choose w : ℝ) * (b : ℝ) ^ w * (w : ℝ) ^ w
        = (((P * pb).choose w : ℝ) * (w : ℝ) ^ w) * (b : ℝ) ^ w := by ring
      _ ≤ (((P * pb : ℕ) : ℝ) ^ w * Real.exp 1 ^ w) * (b : ℝ) ^ w :=
          mul_le_mul_of_nonneg_right h3 (by positivity)
      _ = (((P * pb : ℕ) : ℝ) * Real.exp 1 * b) ^ w := by rw [mul_pow, mul_pow]
      _ ≤ ((a : ℝ) * w) ^ w := pow_le_pow_left₀ (by positivity) h4 w
      _ = (a : ℝ) ^ w * (w : ℝ) ^ w := mul_pow _ _ _
  have h6 : ((P * pb).choose w : ℝ) * (b : ℝ) ^ w ≤ (a : ℝ) ^ w := le_of_mul_le_mul_right h5 (by positivity)
  exact_mod_cast h6

/-! ## From counts to probabilities -/

theorem div_le_two_inv_pow (x y n : ℕ) (h : x * 2 ^ n ≤ y) : (x : ℝ≥0∞) / y ≤ (2⁻¹ : ℝ≥0∞) ^ n := by
  apply ENNReal.div_le_of_le_mul
  have hy : ((2 ^ n * x : ℕ) : ℝ≥0∞) ≤ y := by rw [mul_comm]; exact_mod_cast h
  calc (x : ℝ≥0∞) = (2⁻¹ : ℝ≥0∞) ^ n * (2 ^ n * x) := by
        rw [← mul_assoc, ← mul_pow, ENNReal.inv_mul_cancel two_ne_zero ENNReal.ofNat_ne_top, one_pow, one_mul]
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ n * y := by gcongr; exact_mod_cast hy

/-- **The union term** at `w = w₁·P` from the per-page certificate. -/
theorem bad_term_le (pb P w₁ a b m s S Sp : ℕ) (hP : 0 < P) (hw₁ : 0 < w₁) (hw₁pb : w₁ ≤ pb) (hb : 0 < b)
    (he : Real.exp 1 * pb * b ≤ a * w₁) (hcert : a ^ w₁ * 2 ^ (Sp + 8 * w₁ + m) ≤ b ^ w₁ * 2 ^ (8 * pb))
    (hS : S ≤ Sp * P) (hs : s ≤ 2 ^ 12) (hm : 142 ≤ m * P) :
    (s : ℝ≥0∞) * 2 ^ S * ((P * pb).choose (w₁ * P) : ℝ≥0∞) / 2 ^ (8 * (P * pb - w₁ * P)) ≤ (2⁻¹ : ℝ≥0∞) ^ 130 := by
  set C := (P * pb).choose (w₁ * P)
  have hle : w₁ * P ≤ P * pb := by rw [mul_comm]; exact Nat.mul_le_mul_left P hw₁pb
  have hC : C * b ^ (w₁ * P) ≤ a ^ (w₁ * P) := choose_mul_pow_le pb P w₁ a b hP hw₁ he
  have hcertP : a ^ (w₁ * P) * 2 ^ ((Sp + 8 * w₁ + m) * P) ≤ b ^ (w₁ * P) * 2 ^ (8 * pb * P) := by
    have := Nat.pow_le_pow_left hcert P
    rwa [mul_pow, mul_pow, ← pow_mul, ← pow_mul, ← pow_mul, ← pow_mul] at this
  have hsplit : 2 ^ ((Sp + 8 * w₁ + m) * P) = 2 ^ (Sp * P) * 2 ^ (m * P) * 2 ^ (8 * (w₁ * P)) := by
    rw [← pow_add, ← pow_add]
    congr 1
    ring
  have htot : 2 ^ (8 * pb * P) = 2 ^ (8 * (P * pb - w₁ * P)) * 2 ^ (8 * (w₁ * P)) := by
    rw [← pow_add, ← mul_add, Nat.sub_add_cancel hle]
    congr 1
    ring
  have hmain : C * (2 ^ (Sp * P) * 2 ^ (m * P)) ≤ 2 ^ (8 * (P * pb - w₁ * P)) := by
    have h1 : b ^ (w₁ * P) * (C * (2 ^ (Sp * P) * 2 ^ (m * P)) * 2 ^ (8 * (w₁ * P))) ≤
        b ^ (w₁ * P) * (2 ^ (8 * (P * pb - w₁ * P)) * 2 ^ (8 * (w₁ * P))) := by
      calc b ^ (w₁ * P) * (C * (2 ^ (Sp * P) * 2 ^ (m * P)) * 2 ^ (8 * (w₁ * P)))
          = C * b ^ (w₁ * P) * 2 ^ ((Sp + 8 * w₁ + m) * P) := by rw [hsplit]; ring
        _ ≤ a ^ (w₁ * P) * 2 ^ ((Sp + 8 * w₁ + m) * P) := Nat.mul_le_mul_right _ hC
        _ ≤ b ^ (w₁ * P) * 2 ^ (8 * pb * P) := hcertP
        _ = b ^ (w₁ * P) * (2 ^ (8 * (P * pb - w₁ * P)) * 2 ^ (8 * (w₁ * P))) := by rw [htot]
    exact Nat.le_of_mul_le_mul_right (Nat.le_of_mul_le_mul_left h1 (by positivity)) (by positivity)
  have hs' : s * 2 ^ 130 ≤ 2 ^ (m * P) :=
    calc s * 2 ^ 130 ≤ 2 ^ 12 * 2 ^ 130 := Nat.mul_le_mul_right _ hs
      _ = 2 ^ 142 := by norm_num
      _ ≤ 2 ^ (m * P) := Nat.pow_le_pow_right (by norm_num) hm
  have hnat : s * 2 ^ S * C * 2 ^ 130 ≤ 2 ^ (8 * (P * pb - w₁ * P)) :=
    calc s * 2 ^ S * C * 2 ^ 130 = (s * 2 ^ 130) * 2 ^ S * C := by ring
      _ ≤ 2 ^ (m * P) * 2 ^ (Sp * P) * C :=
          Nat.mul_le_mul_right _ (Nat.mul_le_mul hs' (Nat.pow_le_pow_right (by norm_num) hS))
      _ = C * (2 ^ (Sp * P) * 2 ^ (m * P)) := by ring
      _ ≤ _ := hmain
  have := div_le_two_inv_pow (s * 2 ^ S * C) (2 ^ (8 * (P * pb - w₁ * P))) 130 hnat
  push_cast at this
  exact this

/-- **The sampling term** from a count. -/
theorem good_term_le (B w s : ℕ) (h : (B - w) ^ s * 2 ^ 129 ≤ B.descFactorial s) :
    ((B - w : ℕ) : ℝ≥0∞) ^ s / (B.descFactorial s : ℝ≥0∞) ≤ (2⁻¹ : ℝ≥0∞) ^ 129 := by
  have := div_le_two_inv_pow ((B - w) ^ s) (B.descFactorial s) 129 h
  rwa [Nat.cast_pow] at this

/-- The sampling count for `P` pages from the one at `P = 1`. -/
theorem good_fam (P w₁ s : ℕ) (hP : 1 ≤ P) (hs1 : 1 ≤ s) (hs : s ≤ 2 ^ 20)
    (hcert : (2 ^ 20 - w₁) ^ s * 2 ^ 129 ≤ (2 ^ 20 + 1 - s) ^ s) :
    (P * 2 ^ 20 - w₁ * P) ^ s * 2 ^ 129 ≤ (P * 2 ^ 20).descFactorial s := by
  have hsub : P * 2 ^ 20 - w₁ * P = P * (2 ^ 20 - w₁) := by rw [mul_tsub, mul_comm w₁]
  have hden : P * (2 ^ 20 + 1 - s) ≤ P * 2 ^ 20 + 1 - s := by
    have h2 : (2 : ℕ) ^ 20 = (2 ^ 20 + 1 - s) + (s - 1) := by omega
    have h3 : s - 1 ≤ P * (s - 1) := Nat.le_mul_of_pos_left _ hP
    have h4 : P * 2 ^ 20 = P * (2 ^ 20 + 1 - s) + P * (s - 1) := by
      conv_lhs => rw [h2]
      rw [mul_add]
    rw [h4]
    generalize P * (2 ^ 20 + 1 - s) = A
    generalize P * (s - 1) = A' at h3 ⊢
    omega
  refine le_trans ?_ (Nat.pow_sub_le_descFactorial (P * 2 ^ 20) s)
  calc (P * 2 ^ 20 - w₁ * P) ^ s * 2 ^ 129 = P ^ s * ((2 ^ 20 - w₁) ^ s * 2 ^ 129) := by rw [hsub, mul_pow]; ring
    _ ≤ P ^ s * (2 ^ 20 + 1 - s) ^ s := Nat.mul_le_mul_left _ hcert
    _ = (P * (2 ^ 20 + 1 - s)) ^ s := (mul_pow _ _ _).symm
    _ ≤ (P * 2 ^ 20 + 1 - s) ^ s := Nat.pow_le_pow_left hden s

/-- The foreign content's term is at most `2^-130` once `S' + S_φ ≥ L + 256` and `s ≤ 2^12`. -/
theorem foreign_le (s L S' S_φ : ℕ) (hs : s ≤ 2 ^ 12) (h : L + 256 ≤ S' + S_φ) :
    (s : ℝ≥0∞) * 2 ^ L / (2 ^ S' * 2 ^ S_φ) ≤ (2⁻¹ : ℝ≥0∞) ^ 130 :=
  calc (s : ℝ≥0∞) * 2 ^ L / (2 ^ S' * 2 ^ S_φ) ≤ (2 : ℝ≥0∞) ^ 12 * 2 ^ L / 2 ^ (L + 256) := by
        rw [← pow_add]
        gcongr
        · exact_mod_cast hs
        · exact one_le_two
    _ = (2⁻¹ : ℝ≥0∞) ^ 244 := by
        rw [← pow_add, div_eq_mul_inv, ENNReal.inv_pow, PousDenseChain.two_pow_mul_inv_pow _ _ (by omega)]
        congr 1
        omega
    _ ≤ (2⁻¹ : ℝ≥0∞) ^ 130 := pow_le_pow_right_of_le_one' (by norm_num) (by norm_num)

/-- The memory bound `⌊(18/19)·|C|⌋` is at most `7947103` bits per page. -/
theorem pages_state_le (P : ℕ) : ⌊(1 - (1 - Params.ρ)) * ((P * 2 ^ 20 * 8 : ℕ) : ℚ)⌋₊ ≤ 7947103 * P := by
  rw [sub_sub_cancel]
  apply Nat.floor_le_of_le
  have hP : (0 : ℚ) ≤ P := Nat.cast_nonneg P
  unfold Params.ρ
  push_cast
  linarith

/-! ## The requirement -/

/-- **The page fill meets `EraseMeetsDistinct`** at `f = 1/19`, `D = 0`, `s` distinct reads and `ε = 2^-128`, from
the union certificate at `w = w₁·P` and the sampling count. -/
theorem erase_meets_pages (P : ℕ) [NeZero P] (hP : P ≤ 2 ^ 64) (K : Bits 256) (Q s w₁ a b m : ℕ)
    (hs : s ≤ 2 ^ 12) (hsB : s ≤ P * 2 ^ 20) (hw₁ : 0 < w₁) (hw₁pb : w₁ ≤ 2 ^ 20) (hb : 0 < b)
    (he : Real.exp 1 * (2 ^ 20 : ℕ) * b ≤ a * w₁)
    (hcert : a ^ w₁ * 2 ^ (7947103 + 8 * w₁ + m) ≤ b ^ w₁ * 2 ^ (8 * 2 ^ 20)) (hm : 142 ≤ m * P)
    (hgood : (P * 2 ^ 20 - w₁ * P) ^ s * 2 ^ 129 ≤ (P * 2 ^ 20).descFactorial s) :
    EraseMeetsDistinct (pageScheme (2 ^ 20) P K) (1 - Params.ρ) 0 Q s ((2⁻¹ : ℝ≥0∞) ^ 128) := by
  set sch := pageScheme (2 ^ 20) P K
  have hP0 : 0 < P := Nat.pos_of_ne_zero (NeZero.ne P)
  set S := ⌊(1 - (1 - Params.ρ)) * ((sch.B * sch.ℓ : ℕ) : ℚ)⌋₊ with hS_def
  have hS : S ≤ 7947103 * P := pages_state_le P
  have hw : w₁ * P ≤ P * 2 ^ 20 := by rw [Nat.mul_comm w₁ P]; exact Nat.mul_le_mul_left P hw₁pb
  have haudit := PousErase.page_audit_secure (2 ^ 20) P hP K S Q s (w₁ * P) hw hsB
  have hbad := bad_term_le (2 ^ 20) P w₁ a b m s S 7947103 hP0 hw₁ hw₁pb hb he hcert hS hs hm
  have hgood' := good_term_le (P * 2 ^ 20) (w₁ * P) s hgood
  have haudit' : AuditSecureDraw sch (distinctReads sch s) S 0 Q 0 ((2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 129) :=
    fun W A₁ A₂ => (haudit W A₁ A₂).trans (by rw [zero_add, zero_add]; exact add_le_add hbad hgood')
  have hsum : (2⁻¹ : ℝ≥0∞) ^ 130 + (2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130 = (2⁻¹ : ℝ≥0∞) ^ 128 := by
    rw [add_comm _ ((2⁻¹ : ℝ≥0∞) ^ 129), add_assoc, two_inv_pow_succ_add_self 129, two_inv_pow_succ_add_self 128]
  refine ⟨le_rfl, ?_, ?_⟩
  · intro W A₁ A₂
    refine (haudit' W A₁ A₂).trans ?_
    simp only [zero_add]
    rw [← hsum]
    exact le_add_right le_rfl
  · intro S_φ hS_φ K' V _ _ dig W P'
    have : Nonempty (Fin s ↪ Fin sch.B) :=
      Function.Embedding.nonempty_of_card_le (by simpa [sch, pageScheme] using hsB)
    refine (secure_erasure_meets_draw sch K' V dig (Fin s ↪ Fin sch.B) (distinctReads sch s) (sch.B * sch.ℓ) S
      S_φ 0 Q 0 _ haudit' W P').trans ?_
    rw [zero_add, ← hsum]
    have hfl : sch.B * sch.ℓ + 256 ≤ S + S_φ := by
      rw [hS_def, sub_sub_cancel]
      exact floor_add_ge _ _ hS_φ
    exact add_le_add_right (foreign_le s _ _ _ hs hfl) _

/-- `e·2^20 ≤ 94·30324`. -/
theorem exp_fam : Real.exp 1 * (2 ^ 20 : ℕ) * (1 : ℕ) ≤ (94 : ℕ) * (30324 : ℕ) := by
  have := Real.exp_one_lt_d9
  push_cast
  nlinarith

/-- `e·2^20·100 ≤ 9397·30335`. -/
theorem exp_96112 : Real.exp 1 * (2 ^ 20 : ℕ) * (100 : ℕ) ≤ (9397 : ℕ) * (30335 : ℕ) := by
  have := Real.exp_one_lt_d9
  push_cast
  nlinarith

set_option exponentiation.threshold 10000000 in
theorem cert_fam : 94 ^ 30324 * 2 ^ (7947103 + 8 * 30324 + 142) ≤ 1 ^ 30324 * 2 ^ (8 * 2 ^ 20) :=
  Nat.le_of_ble_eq_true rfl

set_option exponentiation.threshold 10000000 in
theorem sampling_fam : (2 ^ 20 - 30324) ^ 3430 * 2 ^ 129 ≤ (2 ^ 20 + 1 - 3430) ^ 3430 :=
  Nat.le_of_ble_eq_true rfl

set_option exponentiation.threshold 10000000 in
theorem cert_96112 : 9397 ^ 30335 * 2 ^ (7947103 + 8 * 30335 + 2) ≤ 100 ^ 30335 * 2 ^ (8 * 2 ^ 20) :=
  Nat.le_of_ble_eq_true rfl

set_option exponentiation.threshold 10000000 in
theorem sampling_96112 : (96112 * 2 ^ 20 - 30335 * 96112) ^ 3046 * 2 ^ 129 ≤ (96112 * 2 ^ 20 + 1 - 3046) ^ 3046 :=
  Nat.le_of_ble_eq_true rfl

theorem secure_erasure_pages_family : Pous.Guarantees.SecureErasurePagesFamily := by
  intro P _ hP K Q
  have hP1 : 1 ≤ P := Nat.pos_of_ne_zero (NeZero.ne P)
  exact erase_meets_pages P hP K Q 3430 30324 94 1 142 (by norm_num)
    (le_trans (by norm_num) (Nat.mul_le_mul_right _ hP1)) (by norm_num) (by norm_num) (by norm_num) exp_fam cert_fam
    (by omega) (good_fam P 30324 3430 hP1 (by norm_num) (by norm_num) sampling_fam)

set_option exponentiation.threshold 10000000 in
theorem secure_erasure_pages_96112 : Pous.Guarantees.SecureErasurePages96112 := fun K Q =>
  erase_meets_pages 96112 (by norm_num) K Q 3046 30335 9397 100 2 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) exp_96112 cert_96112 (by norm_num)
    (sampling_96112.trans (Nat.pow_sub_le_descFactorial _ _))

end PousErase
