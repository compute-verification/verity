import Mathlib.Combinatorics.SetFamily.LYM
import Proofs.Accounting.Tag.Signs

/-!
# Erdős's Littlewood–Offord bound

With positive weights, the bit vectors with one given weighted sum, read as the sets of their `1`s, form an antichain:
a strictly larger set has a strictly larger sum. Sperner's theorem (`IsAntichain.sperner`) bounds an antichain of subsets
of a `u`-set by `C(u, ⌊u/2⌋)` (`card_level_pos_le`). Nonzero weights reduce to positive ones by flipping the bits whose
weight is negative, which shifts every sum by the same constant (`card_level_le_choose`). `littlewoodOfford` is
`Accounting.Guarantees.Sketch.LittlewoodOfford`.
-/

namespace Accounting.TagProofs

open Accounting.Tag Finset
open scoped ENNReal

/-- With positive weights, at most `C(u, ⌊u/2⌋)` bit vectors share a weighted sum. -/
theorem card_level_pos_le {u : ℕ} (w : Fin u → ℤ) (hw : ∀ j, 0 < w j) (t : ℤ) :
    (univ.filter fun ε : Fin u → Bool => unreadSum w ε = t).card ≤ u.choose (u / 2) := by
  classical
  set L := univ.filter fun ε : Fin u → Bool => unreadSum w ε = t
  set ones : (Fin u → Bool) → Finset (Fin u) := fun ε => univ.filter fun j => ε j = true with hones
  have hinj : Function.Injective ones := by
    intro ε ε' h
    funext j
    have := congrArg (j ∈ ·) h
    simp only [hones, mem_filter, mem_univ, true_and, eq_iff_iff] at this
    cases hj : ε j <;> cases hj' : ε' j <;> simp_all
  have hsum : ∀ ε, unreadSum w ε = ∑ j ∈ ones ε, w j := by
    intro ε
    rw [hones, sum_filter]
    rfl
  have hanti : IsAntichain (· ⊆ ·) (SetLike.coe (L.image ones)) := by
    intro A hA B hB hne hAB
    simp only [coe_image, Set.mem_image, mem_coe] at hA hB
    obtain ⟨ε, hε, rfl⟩ := hA
    obtain ⟨ε', hε', rfl⟩ := hB
    simp only [L, mem_filter, mem_univ, true_and] at hε hε'
    obtain ⟨i, hiB, hiA⟩ := exists_of_ssubset (Finset.ssubset_iff_subset_ne.2 ⟨hAB, hne⟩)
    have hlt := sum_lt_sum_of_subset hAB hiB hiA (hw i) fun j _ _ => (hw j).le
    rw [← hsum, ← hsum, hε, hε'] at hlt
    exact lt_irrefl _ hlt
  have := hanti.sperner
  rw [card_image_of_injective _ hinj, Fintype.card_fin] at this
  exact this

/-- With nonzero weights, at most `C(u, ⌊u/2⌋)` bit vectors share a weighted sum: flipping the bits of the negative
weights maps them one to one onto the bit vectors with one sum under the weights' absolute values. -/
theorem card_level_le_choose {u : ℕ} (v : Fin u → ℤ) (hv : ∀ j, v j ≠ 0) (t : ℤ) :
    (univ.filter fun ε : Fin u → Bool => unreadSum v ε = t).card ≤ u.choose (u / 2) := by
  classical
  set φ : (Fin u → Bool) → (Fin u → Bool) := fun ε j => if v j < 0 then !ε j else ε j with hφ
  have hinj : Function.Injective φ := by
    intro ε ε' h
    funext j
    have := congrFun h j
    simp only [hφ] at this
    split_ifs at this
    · simpa using this
    · exact this
  set C := ∑ j, if v j < 0 then v j else 0 with hC
  have hshift : ∀ ε, unreadSum v ε = C + unreadSum (fun j => |v j|) (φ ε) := by
    intro ε
    simp only [unreadSum, hC, hφ, ← sum_add_distrib]
    refine sum_congr rfl fun j _ => ?_
    by_cases hj : v j < 0
    · rw [abs_of_neg hj]
      cases ε j <;> simp [hj]
    · rw [abs_of_nonneg (not_lt.1 hj)]
      simp [hj]
  have hsub : (univ.filter fun ε : Fin u → Bool => unreadSum v ε = t).image φ ⊆
      univ.filter fun ε : Fin u → Bool => unreadSum (fun j => |v j|) ε = t - C := by
    intro ε' hε'
    obtain ⟨ε, hε, rfl⟩ := mem_image.1 hε'
    simp only [mem_filter, mem_univ, true_and] at hε ⊢
    rw [hshift] at hε
    omega
  calc (univ.filter fun ε : Fin u → Bool => unreadSum v ε = t).card
      = ((univ.filter fun ε : Fin u → Bool => unreadSum v ε = t).image φ).card :=
        (card_image_of_injective _ hinj).symm
    _ ≤ _ := card_le_card hsub
    _ ≤ u.choose (u / 2) := card_level_pos_le _ (fun j => abs_pos.2 (hv j)) _

theorem littlewoodOfford : Accounting.Guarantees.Sketch.LittlewoodOfford := by
  intro u v t
  classical
  have hcard : Fintype.card (Fin u → Bool) = 2 ^ u := by simp
  by_cases hv : ∀ j, v j ≠ 0
  · refine (prob_uniform_le _ (u.choose (u / 2)) ?_).trans (le_of_eq (by rw [hcard]; push_cast; rfl))
    have e : (univ.filter fun ε : Fin u → Bool => (∀ j, v j ≠ 0) ∧ unreadSum v ε = t) =
        univ.filter fun ε : Fin u → Bool => unreadSum v ε = t := by
      ext ε
      simp only [mem_filter, mem_univ, true_and, and_iff_right_iff_imp]
      exact fun _ => hv
    rw [e]
    exact card_level_le_choose v hv t
  · refine (prob_uniform_le _ 0 ?_).trans (by simp)
    rw [Nat.le_zero, card_eq_zero, filter_eq_empty_iff]
    exact fun ε _ h => hv h.1

end Accounting.TagProofs
