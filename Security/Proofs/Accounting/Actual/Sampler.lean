import Mathlib
import Definitions.Accounting.Actual.Sampler
import Definitions.Pous.Model.StreamedWipe

/-!
# The deployed uniform sampler on one word: its counts

The sampler (`Accounting.Actual.wordIndex`) accepts `2^w − (2^w mod N)` words (`card_accepted`), and gives each index
at most `⌊2^w / N⌋` of them (`card_fiber_le`), a uniform share. Beside them, the probability facts the remote audit's
opening draw is proved with: a count bounds a probability and back (`pr_le_ofReal`, `ofReal_le_pr`), a probability is
the same over the coins in another order (`pr_comp_equiv`), and the real game is the ideal one plus the PRF's advantage
(`pr_le_add_prfAdvantage`).
-/

namespace Accounting.SecurityProofs.Actual

open Finset
open scoped ENNReal

/-- A label's value fixes the label. -/
theorem bitsVal_injective (w : ℕ) : Function.Injective (Pous.bitsVal (w := w)) :=
  fun _ _ h => (Pous.bitsEquiv w).injective (Fin.ext h)

theorem wordIndex_isSome {N w : ℕ} (u : Pous.Bits w) :
    (Actual.wordIndex N u).isSome ↔ Pous.bitsVal u < 2 ^ w - 2 ^ w % N := by
  unfold Actual.wordIndex
  split_ifs with h <;> simp [h]

theorem wordIndex_eq_some {N w : ℕ} {u : Pous.Bits w} {i : Fin N} (h : Actual.wordIndex N u = some i) :
    Pous.bitsVal u < 2 ^ w - 2 ^ w % N ∧ Pous.bitsVal u % N = i := by
  unfold Actual.wordIndex at h
  split_ifs at h with h'
  · exact ⟨h', congrArg Fin.val (Option.some_injective _ h)⟩

theorem limit_eq (N w : ℕ) : 2 ^ w - 2 ^ w % N = N * (2 ^ w / N) := by
  have h := Nat.div_add_mod (2 ^ w) N
  omega

open Classical in
/-- **The sampler accepts `2^w − (2^w mod N)` words.** -/
theorem card_accepted (N w : ℕ) :
    (univ.filter fun u : Pous.Bits w => (Actual.wordIndex N u).isSome).card = 2 ^ w - 2 ^ w % N := by
  have : (univ.filter fun u : Pous.Bits w => (Actual.wordIndex N u).isSome) =
      (univ.filter fun i : Fin (2 ^ w) => (i : ℕ) < 2 ^ w - 2 ^ w % N).map (Pous.bitsEquiv w).symm.toEmbedding := by
    ext u
    simp [Finset.mem_map_equiv, wordIndex_isSome, Pous.bitsVal]
  rw [this, card_map, Fin.card_filter_val_lt]
  exact min_eq_right (Nat.sub_le _ _)

open Classical in
/-- **Each index has at most `⌊2^w / N⌋` accepted words**: an accepted word is fixed by its quotient by `N`, which is
below `⌊2^w / N⌋`. -/
theorem card_fiber_le {N w : ℕ} (i : Fin N) :
    (univ.filter fun u : Pous.Bits w => Actual.wordIndex N u = some i).card ≤ 2 ^ w / N := by
  have hN : 0 < N := i.pos
  calc _ ≤ (range (2 ^ w / N)).card := by
        refine card_le_card_of_injOn (fun u => Pous.bitsVal u / N) (fun u hu => ?_) (fun u hu v hv huv => ?_)
        · obtain ⟨h1, -⟩ := wordIndex_eq_some (mem_filter.1 hu).2
          rw [limit_eq] at h1
          rw [coe_range, Set.mem_Iio, Nat.div_lt_iff_lt_mul hN]
          linarith [Nat.mul_comm N (2 ^ w / N)]
        · obtain ⟨-, hu2⟩ := wordIndex_eq_some (mem_filter.1 hu).2
          obtain ⟨-, hv2⟩ := wordIndex_eq_some (mem_filter.1 hv).2
          apply bitsVal_injective w
          rw [← Nat.div_add_mod (Pous.bitsVal u) N, ← Nat.div_add_mod (Pous.bitsVal v) N, hu2, hv2]
          exact congrArg (fun q => N * q + (i : ℕ)) huv
    _ = 2 ^ w / N := card_range _

open Classical in
theorem pr_le_ofReal {α : Type} [Fintype α] (P : α → Prop) {β : ℝ} (hβ : 0 ≤ β)
    (h : ((univ.filter P).card : ℝ) ≤ β * Fintype.card α) : Pous.pr P ≤ ENNReal.ofReal β := by
  unfold Pous.pr
  apply ENNReal.div_le_of_le_mul
  rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul hβ]
  exact ENNReal.ofReal_le_ofReal h

open Classical in
theorem ofReal_le_pr {α : Type} [Fintype α] (P : α → Prop) :
    ENNReal.ofReal (((univ.filter P).card : ℝ) / Fintype.card α) ≤ Pous.pr P := by
  unfold Pous.pr
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h]
  · rw [ENNReal.ofReal_div_of_pos (by exact_mod_cast h), ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

open Classical in
/-- A probability is the same over the coins in another order. -/
theorem pr_comp_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (P : β → Prop) :
    Pous.pr (fun x => P (e x)) = Pous.pr P := by
  unfold Pous.pr
  rw [Fintype.card_congr e, card_equiv e (t := univ.filter P) fun x => by simp]

/-- **The real game is the ideal one plus the PRF's advantage** against the distinguisher, for any bound on the ideal
one. -/
theorem pr_le_add_prfAdvantage {N b κ : ℕ} {ω : Type} [Fintype ω] (prf : Pous.Bits κ → Fin N → Pous.Bits b)
    (D : ω × (Fin N → Pous.Bits b) → Prop) {β : ℝ≥0∞} (h : Pous.pr D ≤ β) :
    Pous.pr (fun x : ω × Pous.Bits κ => D (x.1, prf x.2)) ≤ β + Pous.StreamedWipe.prfAdvantage prf D :=
  le_add_tsub.trans (add_le_add h le_rfl)

end Accounting.SecurityProofs.Actual
