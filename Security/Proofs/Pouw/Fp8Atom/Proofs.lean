import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.Fp8Atom.Pinned
import Proofs.Pouw.Fp8Atom.Checks

/-!
# Proofs of `Pouw.Fp8Atom`'s pinned statements

`Conformance` and `AdaCaptures` assemble the kernel-evaluated chunks of `Pouw.Fp8Atom.Checks`. `CodeClasses` and the
integer form of `DecodeInjective` are decided over the 256 codes. `ProductExact` and `AlignedMono` are proved.
-/

namespace Pouw.Fp8Atom.Proofs

open Pouw.Fp8Atom

/-! ## Conformance -/

theorem all_holds {step : Nat → (Fin 32 → Code) → (Fin 32 → Code) → Nat} {l : List Vec}
    (h : l.all (Vec.holds step) = true) : ∀ v ∈ l, step v.acc (codes v.a) (codes v.b) = v.d := by
  intro v hv
  simpa [Vec.holds] using List.all_eq_true.mp h v hv

theorem vectors_all : vectors.all (Vec.holds hopperStep) = true := by
  simp only [vectors, List.all_append, Checks.c00, Checks.c01, Bool.and_self]

theorem vectors_length : vectors.length = 631 := by decide +kernel

/-- **Conformance with `verity.ml.tc`** on the 631 vectors. -/
theorem conformance : Conformance := ⟨vectors_length.ge, all_holds vectors_all⟩

/-- **The 360 RTX 4090 captures** hold for `adaStep`. -/
theorem adaCaptures : AdaCaptures := ⟨by decide +kernel, all_holds Checks.ada⟩

/-! ## The codec -/

theorem isNaN_iff : ∀ c : Code, c.isNaN = true ↔ c = 0x7F ∨ c = 0xFF := by decide +kernel

theorem sig_eq_zero_iff : ∀ c : Code, c.sig = 0 ↔ c = 0x00 ∨ c = 0x80 := by decide +kernel

theorem isZero_iff_sig : ∀ c : Code, c.isZero = true ↔ c.sig = 0 := by decide +kernel

theorem expOr1_pos (c : Code) : 1 ≤ c.expOr1 := le_max_right _ _

theorem decode_eq_none_iff (c : Code) : decode c = none ↔ c.isNaN = true := by
  unfold decode; split <;> simp_all

/-- A finite code's value, times `2^9`, as an integer. -/
def intVal (c : Code) : Int := (if c.sign then -1 else 1) * ((c.sig * 2 ^ (c.expOr1 - 1) : Nat) : Int)

theorem two_zpow_lsbExp (c : Code) : (2 : ℚ) ^ c.lsbExp = 2 ^ (c.expOr1 - 1) / 512 := by
  have h1 := expOr1_pos c
  have : c.lsbExp = ((c.expOr1 - 1 : Nat) : Int) - 9 := by unfold Code.lsbExp; omega
  rw [this, zpow_sub₀ two_ne_zero, zpow_natCast]; norm_num

theorem decode_eq (c : Code) (h : c.isNaN = false) : decode c = some ((intVal c : ℚ) / 512) := by
  unfold decode intVal
  rw [ite_eq_right (by simp [h]), two_zpow_lsbExp]
  congr 1
  split <;> push_cast <;> ring

theorem intVal_inj : ∀ a b : Code, a.isNaN = false → b.isNaN = false → intVal a = intVal b →
    a = b ∨ (a.isZero = true ∧ b.isZero = true) := by decide +kernel

theorem finite_card : (Finset.univ.filter fun c : Code => c.isNaN = false).card = 254 := by decide +kernel

/-- **The code classes.** -/
theorem codeClasses : CodeClasses := by
  refine ⟨isNaN_iff, decode_eq_none_iff, fun c => ?_, finite_card⟩
  cases hn : c.isNaN
  · rw [decode_eq c hn, Option.some.injEq, div_eq_zero_iff, ← sig_eq_zero_iff]
    simp only [intVal, Int.cast_mul, Int.cast_natCast, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
    constructor
    · rintro (h | h)
      · split at h <;> simp_all
      · norm_num at h
    · intro h; left; simp [h]
  · have : decode c = none := (decode_eq_none_iff c).mpr hn
    rw [this]
    rcases (isNaN_iff c).mp hn with rfl | rfl <;> decide

/-- **`decode` is injective on the finite codes**, but for the two zeros. -/
theorem decodeInjective : DecodeInjective := by
  intro a b ha hab
  have hb : b.isNaN = false := by
    cases h : b.isNaN
    · rfl
    · rw [(decode_eq_none_iff b).mpr h, decode_eq a ha] at hab; cases hab
  rw [decode_eq a ha, decode_eq b hb, Option.some.injEq] at hab
  have : intVal a = intVal b := by
    have h2 : ((intVal a : ℚ)) = intVal b := by linarith [hab]
    exact_mod_cast h2
  exact intVal_inj a b ha hb this

/-! ## The product -/

theorem decode_some_isNaN {c : Code} {x : ℚ} (h : decode c = some x) : c.isNaN = false := by
  cases hc : c.isNaN
  · rfl
  · rw [(decode_eq_none_iff c).mpr hc] at h; cases h

/-- **The product is exact.** -/
theorem productExact : ProductExact := by
  intro a b x y ha hb
  have hna := decode_some_isNaN ha
  have hnb := decode_some_isNaN hb
  rw [decode_eq a hna, Option.some.injEq] at ha
  rw [decode_eq b hnb, Option.some.injEq] at hb
  subst ha hb
  unfold product Term.value
  rw [Nat.shiftLeft_eq]
  by_cases h0 : a.sig * b.sig * 2 ^ 17 = 0
  · simp only [h0, ite_true, Term.zero]
    have : a.sig * b.sig = 0 := by simpa using h0
    rcases Nat.mul_eq_zero.mp this with h | h <;> simp [intVal, h]
  · simp only [h0, ite_false]
    have h1 := expOr1_pos a
    have h2 := expOr1_pos b
    have e : (2 : ℚ) ^ ((a.expOr1 : Int) + b.expOr1 - 14 - 23) =
        2 ^ (a.expOr1 - 1) * 2 ^ (b.expOr1 - 1) / 2 ^ 35 := by
      have : ((a.expOr1 : Int) + b.expOr1 - 14 - 23) = ((a.expOr1 - 1 + (b.expOr1 - 1) : Nat) : Int) - 35 := by
        omega
      rw [this, zpow_sub₀ two_ne_zero, zpow_natCast, pow_add]
      norm_num
    rw [e]
    unfold intVal
    cases a.sign <;> cases b.sign <;> simp <;> ring

/-! ## The first truncation -/

theorem toAdder_of_le {w : Nat} (hw : w ≤ 24) (m : Nat) : toAdder w m = m / 2 ^ (24 - w) := by
  unfold toAdder
  split
  · have : w = 24 := by omega
    subst this; simp
  · rw [Nat.shiftRight_eq_div_pow]

/-- The total right shift of a term aligned to `E` on a `w`-bit adder. -/
def shiftOf (w : Nat) (E : Int) (t : Term) : Nat := 24 - w + (E - t.exp).toNat

/-- The aligned magnitude is the magnitude divided by `2^shiftOf`. -/
theorem aligned_eq {w : Nat} (hw : w ≤ 24) (E : Int) (t : Term) :
    aligned w E t = (if t.neg then -1 else 1) * ((t.mag / 2 ^ shiftOf w E t : Nat) : Int) := by
  unfold aligned shiftOf
  split
  · simp_all
  · rw [toAdder_of_le hw, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul, ← pow_add]
    split <;> simp

/-- A term's value is its magnitude over `2^shiftOf`, on the grid `2^(E − w + 1)`. -/
theorem value_eq {w : Nat} (hw : w ≤ 24) {E : Int} {t : Term} (hE : t.exp ≤ E) :
    t.value = (if t.neg then -1 else 1) * ((t.mag : ℚ) / 2 ^ shiftOf w E t) * 2 ^ (E - w + 1) := by
  unfold Term.value shiftOf
  have : t.exp - 23 = (E - w + 1) - ((24 - w + (E - t.exp).toNat : Nat) : Int) := by omega
  rw [this, zpow_sub₀ two_ne_zero, zpow_natCast]
  ring

/-- Truncation toward zero of a rational to an integer. -/
def truncZ (x : ℚ) : Int := if 0 ≤ x then (⌊x⌋₊ : Int) else -(⌊-x⌋₊ : Int)

theorem truncZ_mono {x y : ℚ} (h : x ≤ y) : truncZ x ≤ truncZ y := by
  unfold truncZ
  split <;> split
  · exact_mod_cast Nat.floor_mono h
  · linarith
  · have : (0 : Int) ≤ (⌊y⌋₊ : Int) := Int.natCast_nonneg _
    linarith [Int.natCast_nonneg ⌊-x⌋₊]
  · have : ⌊-y⌋₊ ≤ ⌊-x⌋₊ := Nat.floor_mono (by linarith)
    have : (⌊-y⌋₊ : Int) ≤ (⌊-x⌋₊ : Int) := by exact_mod_cast this
    linarith

theorem aligned_eq_truncZ {w : Nat} (hw : w ≤ 24) (E : Int) (t : Term) :
    aligned w E t = truncZ ((if t.neg then -1 else 1) * ((t.mag : ℚ) / 2 ^ shiftOf w E t)) := by
  rw [aligned_eq hw]
  have hq : ((t.mag / 2 ^ shiftOf w E t : Nat) : Int) = (⌊(t.mag : ℚ) / 2 ^ shiftOf w E t⌋₊ : Int) := by
    congr 1
    rw [← Nat.floor_div_eq_div (K := ℚ)]
    simp
  have hnn : 0 ≤ (t.mag : ℚ) / 2 ^ shiftOf w E t := by positivity
  unfold truncZ
  cases t.neg
  · simp [hq, hnn]
  · by_cases h0 : (t.mag : ℚ) / 2 ^ shiftOf w E t = 0
    · simp [h0, hq]
    · have : ¬ (0 ≤ -((t.mag : ℚ) / 2 ^ shiftOf w E t)) := by
        intro h; exact h0 (by linarith)
      simp [hq, this]

/-- **The first truncation is monotone.** -/
theorem alignedMono : AlignedMono := by
  intro w E s t hw hs ht hst
  rw [aligned_eq_truncZ hw, aligned_eq_truncZ hw]
  apply truncZ_mono
  rw [value_eq hw hs, value_eq hw ht] at hst
  have hG : (0 : ℚ) < 2 ^ (E - w + 1) := zpow_pos (by norm_num) _
  exact le_of_mul_le_mul_right hst hG

end Pouw.Fp8Atom.Proofs
