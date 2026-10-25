import Proofs.Accounting.Tag.Basic
import Definitions.Accounting.Guarantees.Tag

/-!
# A partially read block's words

`PartialReadGuess`: under `partialRead known U`, uniform on the blocks that agree with `known` outside `U`, the class of a
possible block along an unread word holds all `2^32` of its variants, each as likely as the block, so the best guess of
that word is `2^-32`; along a read word the class is the block alone, so the best guess is `1`.
-/

namespace Accounting.TagProofs

open Accounting.Tag Accounting.Guarantees.Tag
open scoped ENNReal

variable {W : ℕ}

theorem mem_consistent {known x : Block W} {U : Finset (Fin W)} :
    x ∈ consistent known U ↔ ∀ i, i ∉ U → x i = known i := by
  simp [consistent]

theorem partialRead_apply (known : Block W) (U : Finset (Fin W)) (x : Block W) :
    partialRead known U x = if x ∈ consistent known U then ((consistent known U).card : ℝ≥0∞)⁻¹ else 0 := by
  unfold partialRead
  rw [PMF.uniformOfFinset_apply]
  congr

/-- Along an unread word, every variant of a possible block is possible. -/
theorem update_mem_consistent {known x : Block W} {U : Finset (Fin W)} {j : Fin W} (hj : j ∈ U)
    (hx : x ∈ consistent known U) (a : Word) : Function.update x j a ∈ consistent known U := by
  rw [mem_consistent] at hx ⊢
  intro i hi
  rw [Function.update_of_ne (fun h => hi (by rw [h]; exact hj))]
  exact hx i hi

/-- Along a read word, the only possible variant of a possible block is the block. -/
theorem update_mem_consistent_iff {known x : Block W} {U : Finset (Fin W)} {j : Fin W} (hj : j ∉ U)
    (hx : x ∈ consistent known U) (a : Word) : Function.update x j a ∈ consistent known U ↔ a = x j := by
  rw [mem_consistent] at hx
  constructor
  · intro h
    have := (mem_consistent.1 h) j hj
    rw [Function.update_self] at this
    rw [this, hx j hj]
  · rintro rfl
    rw [Function.update_eq_self]
    exact mem_consistent.2 hx

theorem card_consistent_ne_zero (known : Block W) (U : Finset (Fin W)) : ((consistent known U).card : ℝ≥0∞) ≠ 0 := by
  have : known ∈ consistent known U := mem_consistent.2 fun _ _ => rfl
  exact_mod_cast (Finset.card_pos.2 ⟨known, this⟩).ne'

theorem card_word : Fintype.card Word = 2 ^ 32 := Fintype.card_fin _

/-- `a / (b · a) = 1 / b` for a finite nonzero `a`. -/
theorem div_mul_self {a b : ℝ≥0∞} (ha : a ≠ 0) (ha' : a ≠ ⊤) (hb : b ≠ 0) (hb' : b ≠ ⊤) : a / (b * a) = b⁻¹ := by
  rw [div_eq_mul_inv, ENNReal.mul_inv (Or.inl hb) (Or.inl hb'), mul_comm b⁻¹, ← mul_assoc,
    ENNReal.mul_inv_cancel ha ha', one_mul]

theorem partialReadGuess : PartialReadGuess := by
  intro W known U j
  classical
  set S := consistent known U
  set c : ℝ≥0∞ := (S.card : ℝ≥0∞)⁻¹
  have hc0 : c ≠ 0 := ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _)
  have hctop : c ≠ ⊤ := ENNReal.inv_ne_top.2 (card_consistent_ne_zero known U)
  have hknown : known ∈ S := mem_consistent.2 fun _ _ => rfl
  have hμ : ∀ x, partialRead known U x = if x ∈ S then c else 0 := partialRead_apply known U
  by_cases hj : j ∈ U
  · rw [ite_eq_left hj]
    have hclass : ∀ x ∈ S, ∑ a : Word, partialRead known U (Function.update x j a) = (2 ^ 32 : ℝ≥0∞) * c := by
      intro x hx
      rw [Finset.sum_congr rfl fun a _ => by rw [hμ, ite_eq_left (update_mem_consistent hj hx a)],
        Finset.sum_const, Finset.card_univ, card_word, nsmul_eq_mul]
      norm_num
    have hterm : ∀ x ∈ S, partialRead known U x / ∑ a : Word, partialRead known U (Function.update x j a) =
        (2 ^ 32 : ℝ≥0∞)⁻¹ := by
      intro x hx
      rw [hclass x hx, hμ, ite_eq_left hx]
      exact div_mul_self hc0 hctop (by norm_num) (by norm_num)
    refine le_antisymm (iSup_le fun x => ?_) (le_of_eq_of_le (hterm known hknown).symm (le_iSup
      (fun x => partialRead known U x / ∑ a : Word, partialRead known U (Function.update x j a)) known))
    by_cases hx : x ∈ S
    · exact (hterm x hx).le
    · rw [hμ, ite_eq_right hx, ENNReal.zero_div]
      exact zero_le
  · rw [ite_eq_right hj]
    have hterm : ∀ x ∈ S, partialRead known U x / ∑ a : Word, partialRead known U (Function.update x j a) = 1 := by
      intro x hx
      have hsum : ∑ a : Word, partialRead known U (Function.update x j a) = c := by
        rw [Finset.sum_eq_single (x j)]
        · rw [Function.update_eq_self, hμ, ite_eq_left hx]
        · intro a _ ha
          rw [hμ, ite_eq_right (fun h => ha ((update_mem_consistent_iff hj hx a).1 h))]
        · simp
      rw [hsum, hμ, ite_eq_left hx]
      exact ENNReal.div_self hc0 hctop
    refine le_antisymm (iSup_le fun x => ?_) (le_of_eq_of_le (hterm known hknown).symm (le_iSup
      (fun x => partialRead known U x / ∑ a : Word, partialRead known U (Function.update x j a)) known))
    by_cases hx : x ∈ S
    · exact (hterm x hx).le
    · rw [hμ, ite_eq_right hx, ENNReal.zero_div]
      exact zero_le

end Accounting.TagProofs
