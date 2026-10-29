import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.Fp8Atom.H1TPinned
import Proofs.Pouw.Fp8Atom.H1TChecks
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# Proofs of H-1T's pinned statements

The conformance statements assemble the kernel-evaluated parts of `Pouw.Fp8Atom.H1TChecks`; the chain's running words
follow by induction from its local checks. The structural statements unfold the definitions. `RnErr` is
`roundEven`'s half-unit error scaled by the ulp. `RnAddMovesHalf` follows from it: an absorbed summand below one ulp
of `r` leaves `r + i` in `r`'s binade or below, so `|i| ≤ ulp (r + i) / 2 ≤ ulp r / 2`; a summand of an ulp or more
is `RnAddMoves`.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## Conformance -/

theorem h1tFormTable : H1TFormTable := by
  refine ⟨by decide +kernel, fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.formTable_all v hv
  simpa [FormVec.holds] using h

theorem h1tTagTable : H1TTagTable := by
  refine ⟨by decide +kernel, fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.tagTable_all v hv
  simpa [TagVec.holds, and_assoc] using h

/-! ## The salt fields -/

theorem h1tReadsFields : H1TReadsFields := by
  intro U g g' h c
  have e : U.rowSalt g c.1 = U.rowSalt g' c.1 := by
    funext τ
    unfold Unit.rowSalt
    split_ifs
    · exact h _ _
    · rfl
  unfold Unit.word
  rw [e]

theorem shiftRight_small {w b : ℕ} (hw : w < 128) (hb : 7 ≤ b) : w >>> b = 0 := by
  rw [Nat.shiftRight_eq_div_pow]
  apply Nat.div_eq_of_lt
  calc w < 2 ^ 7 := by norm_num; exact hw
    _ ≤ 2 ^ b := Nat.pow_le_pow_right (by norm_num) hb

/-- Raw words below `2^7` have the fields of the zero salt: every bit `extract` reads is bit 7 or above. -/
theorem extract_small {r : RawSlice} (h : ∀ ℓ, (r ℓ).val < 128) : extract r = extract rawZero := by
  have hb : ∀ ℓ b, 7 ≤ b → (r ℓ).val >>> b = 0 := fun ℓ b hb => shiftRight_small (h ℓ) hb
  have hz : ∀ ℓ b, (rawZero ℓ).val >>> b = 0 := fun _ _ => by simp [rawZero]
  unfold extract
  congr 1
  · funext l
    simp only [bitAt, hz, hb _ _ (show 7 ≤ (if l.val % 2 = 0 then 15 else 31) by split <;> omega)]
  · funext i
    simp only [pick3, bitAt, hz, hb _ 31 (by norm_num), hb _ 15 (by norm_num)]
  · funext i
    simp only [pick3, bits3, hz, hb _ 23 (by norm_num), hb _ 7 (by norm_num)]

theorem h1tNotRawWord : H1TNotRawWord := by
  intro U c i τ ℓ e
  let one : RawSlice := Function.update rawZero ℓ ⟨1, by norm_num⟩
  let g0 : U.Salt := fun _ _ => rawZero
  let g1 : U.Salt := fun i' τ' => if i' = i ∧ τ' = τ then one else rawZero
  have hone : ∀ ℓ', (one ℓ').val < 128 := by
    intro ℓ'
    by_cases hℓ : ℓ' = ℓ
    · subst hℓ; simp [one]
    · simp [one, Function.update_of_ne hℓ, rawZero]
  have hf : ∀ i' τ', extract (g0 i' τ') = extract (g1 i' τ') := by
    intro i' τ'
    simp only [g0, g1]
    split_ifs
    · exact (extract_small hone).symm
    · rfl
  have hw := h1tReadsFields U g0 g1 hf c
  have e0 := congrFun e g0
  have e1 := congrFun e g1
  rw [hw, e1] at e0
  simp [g0, g1, one, rawZero] at e0

/-! ## The tag tuples -/

theorem h1tTupleInjective : H1TTupleInjective := by
  intro j j' hj hj' e
  have e0 := congrFun e 0
  have e1 := congrFun e 1
  have e2 := congrFun e 2
  simp only [tuple, pick3, A0, A1, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two] at e0 e1 e2
  norm_num at e0 e1 e2
  split_ifs at e1 e2 <;> omega

/-! ## Counting -/

theorem halfOfInvolution : HalfOfInvolution := by
  intro α _ ψ P hψ hP
  classical
  set A := Finset.univ.filter fun a => P a = true
  have hdisj : Disjoint A (A.image ψ) := by
    rw [Finset.disjoint_left]
    intro a ha hb'
    obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hb'
    simp only [A, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    rw [hP b hb] at ha
    exact Bool.false_ne_true ha
  have hcard : (A.image ψ).card = A.card := Finset.card_image_of_injective _ hψ.injective
  have hu := Finset.card_union_of_disjoint hdisj
  have hle : (A ∪ A.image ψ).card ≤ Fintype.card α := Finset.card_le_univ _
  omega

/-! ## The FP32 add -/

theorem roundEven_err (m : ℚ) : |(roundEven m : ℚ) - m| ≤ 1 / 2 := by
  have h1 := Int.floor_le m
  have h2 := Int.lt_floor_add_one m
  unfold roundEven
  dsimp only
  split_ifs with a b c <;> (try simp only [not_lt] at *) <;> push_cast <;> rw [abs_le] <;> constructor <;> linarith

theorem rp_err (t : ℚ) : |rp t - t| ≤ ulp t / 2 := by
  have hu := ulp_pos t
  have e : rp t - t = ((roundEven (t / ulp t) : ℚ) - t / ulp t) * ulp t := by
    unfold rp; rw [sub_mul, div_mul_cancel₀ _ hu.ne']
  rw [e, abs_mul, abs_of_pos hu]
  have := roundEven_err (t / ulp t)
  nlinarith

theorem rn_err (y : ℚ) : |rn y - y| ≤ ulp y / 2 := by
  rw [rn_eq]
  rcases lt_or_ge y 0 with h | h
  · rw [ite_eq_left h, abs_of_neg h]
    have := rp_err (-y)
    rw [ulp_neg] at this
    calc |(-1) * rp (-y) - y| = |rp (-y) - -y| := by rw [← abs_neg]; congr 1; ring
      _ ≤ _ := this
  · rw [ite_eq_right (not_lt.2 h), abs_of_nonneg h, one_mul]
    exact rp_err y

end Pouw.Fp8Atom.H1T.Proofs
