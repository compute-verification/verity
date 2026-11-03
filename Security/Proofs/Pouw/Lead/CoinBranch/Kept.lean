import Proofs.Pouw.Lead.CoinBranch.PointwiseProofs
import Proofs.Pouw.Lead.CoinBranch.Revisit
import Mathlib.Tactic.LinearCombination

/-!
# Coin branch: what a run keeps (proofs)

At one coin value a run must hold a family of values `R i`, `i ∈ I` (the call's committed tiles). Under whole-slot
writes (`Step A V 1`: model W, a slot one tile) it writes at least `|I|` less the ordered pairs of `I` with equal values
and the indices whose value is already in its initial memory (`held_card_le`): those are the only tiles a program that
branches on the coins can skip, whatever it predicts. Counted by distinct values, it writes at least the distinct values
less the indices its initial memory already holds (`image_card_le`).

Summed over the coins, each pair or hit counts once per coin value it occurs on (`sum_card_filter_comm`), so a bound on
the coins of each pair bounds the sum (`sum_card_filter_le`).

* **Step 0** (`two_rows_fiber_le`): a step-0 tile reads a column's four tag weights `G[:, j] ∈ [−3, 3]^4` only through
  its rows' tags. Two rows whose tags have a nonzero 2 × 2 minor leave at most 49 of the 2,401 weights.
-/

namespace Pouw.SecurityProofs.Lead.CoinBranch

open Finset Pouw.Dimension.Pointwise

open Classical in
/-- **What one run keeps**: a run that holds every `R i`, `i ∈ I`, in some slot at some point writes at least `|I|`
less the ordered pairs of `I` with equal values and the indices whose value an initial slot already holds. -/
theorem held_card_le {A V ι : Type} [DecidableEq V] [Inhabited V] [LinearOrder ι]
    (m₀ : A → Fin 1 → V) (P : List (Step A V 1)) (I : Finset ι) (R : ι → V)
    (hR : ∀ i ∈ I, ∃ k a, run m₀ (P.take k) a 0 = R i) :
    I.card ≤ writes P + ((I ×ˢ I).filter fun p => p.1 < p.2 ∧ R p.1 = R p.2).card +
      (I.filter fun i => ∃ a, m₀ a 0 = R i).card := by
  have h1 : ((I.filter fun i => ¬ ∃ a, m₀ a 0 = R i).image R).card ≤ writes P := by
    refine run_values_le m₀ P _ (fun v hv => ?_) fun v hv a ha => ?_
    · obtain ⟨i, hi, rfl⟩ := mem_image.1 hv
      exact hR i (mem_filter.1 hi).1
    · obtain ⟨i, hi, rfl⟩ := mem_image.1 hv
      exact (mem_filter.1 hi).2 ⟨a, ha⟩
  have h2 := card_le_image_add_pairs (I.filter fun i => ¬ ∃ a, m₀ a 0 = R i) R
  have h3 : (((I.filter fun i => ¬ ∃ a, m₀ a 0 = R i) ×ˢ (I.filter fun i => ¬ ∃ a, m₀ a 0 = R i)).filter
      fun p => p.1 < p.2 ∧ R p.1 = R p.2).card ≤ ((I ×ˢ I).filter fun p => p.1 < p.2 ∧ R p.1 = R p.2).card :=
    card_le_card (filter_subset_filter _ (product_subset_product (filter_subset _ _) (filter_subset _ _)))
  have h4 : (I.filter fun i => ∃ a, m₀ a 0 = R i).card + (I.filter fun i => ¬ ∃ a, m₀ a 0 = R i).card = I.card := by
    rw [card_filter, card_filter, ← sum_add_distrib, card_eq_sum_ones]
    exact sum_congr rfl fun i _ => by by_cases h : ∃ a, m₀ a 0 = R i <;> simp [h]
  omega

open Classical in
/-- **Distinct values cost writes**: a run that holds every `R i`, `i ∈ I`, in some slot at some point writes at least
the distinct values of `R` on `I` less the indices whose value an initial slot already holds. -/
theorem image_card_le {A V ι : Type} [DecidableEq V] [Inhabited V]
    (m₀ : A → Fin 1 → V) (P : List (Step A V 1)) (I : Finset ι) (R : ι → V)
    (hR : ∀ i ∈ I, ∃ k a, run m₀ (P.take k) a 0 = R i) :
    (I.image R).card ≤ writes P + (I.filter fun i => ∃ a, m₀ a 0 = R i).card := by
  have h1 : ((I.image R).filter fun v => ¬ ∃ a, m₀ a 0 = v).card ≤ writes P := by
    refine run_values_le m₀ P _ (fun v hv => ?_) fun v hv a ha => ?_
    · obtain ⟨i, hi, rfl⟩ := mem_image.1 (mem_filter.1 hv).1
      exact hR i hi
    · exact (mem_filter.1 hv).2 ⟨a, ha⟩
  have h2 : ((I.image R).filter fun v => ∃ a, m₀ a 0 = v).card ≤
      (I.filter fun i => ∃ a, m₀ a 0 = R i).card := by
    rw [filter_image]
    exact card_image_le
  have h3 := card_filter_add_card_filter_not (s := I.image R) fun v => ∃ a, m₀ a 0 = v
  omega

/-- **Counting over the coins**: summed over coin values, a family of events counts each event once per coin value it
occurs on. -/
theorem sum_card_filter_comm {Ω Q : Type} (O : Finset Ω) (E : Finset Q) (occ : Q → Ω → Prop)
    [∀ e ω, Decidable (occ e ω)] :
    ∑ ω ∈ O, (E.filter fun e => occ e ω).card = ∑ e ∈ E, (O.filter fun ω => occ e ω).card := by
  simp only [card_filter]
  exact sum_comm

/-- **Per-event bounds bound the sum**: if each event occurs on at most `N` coin values, the events summed over the
coins number at most `|E| · N`. -/
theorem sum_card_filter_le {Ω Q : Type} (O : Finset Ω) (E : Finset Q) (occ : Q → Ω → Prop)
    [∀ e ω, Decidable (occ e ω)] {N : ℕ} (hN : ∀ e ∈ E, (O.filter fun ω => occ e ω).card ≤ N) :
    ∑ ω ∈ O, (E.filter fun e => occ e ω).card ≤ E.card * N := by
  rw [sum_card_filter_comm]
  calc ∑ e ∈ E, (O.filter fun ω => occ e ω).card ≤ ∑ _e ∈ E, N := sum_le_sum hN
    _ = E.card * N := by rw [sum_const, nsmul_eq_mul, Nat.cast_id]

/-- **Two rows fix two of a column's tag weights**: if two rows' tags `Ta` and `Tb` have a nonzero minor on
coordinates `q ≠ q'`, the weights `g ∈ [−3, 3]^4` with given `Ta · g` and `Tb · g` number at most 49. The other two
coordinates fix `g q` and `g q'`. -/
theorem two_rows_fiber_le (Ta Tb : Fin 4 → ℤ) {q q' : Fin 4} (hq : q ≠ q')
    (hdet : Ta q * Tb q' - Ta q' * Tb q ≠ 0) (ca cb : ℤ) :
    ((Fintype.piFinset fun _ : Fin 4 => Icc (-3 : ℤ) 3).filter
      fun g => ∑ k, Ta k * g k = ca ∧ ∑ k, Tb k * g k = cb).card ≤ 49 := by
  set rest : (Fin 4 → ℤ) → Fin 4 → ℤ := fun g k => if k = q ∨ k = q' then 0 else g k with hrest
  have hcard : (Fintype.piFinset fun k : Fin 4 => if k = q ∨ k = q' then ({0} : Finset ℤ) else Icc (-3 : ℤ) 3).card =
      49 := by
    rw [Fintype.card_piFinset]
    have h : ∀ k : Fin 4, (if k = q ∨ k = q' then ({0} : Finset ℤ) else Icc (-3 : ℤ) 3).card =
        if k = q ∨ k = q' then 1 else 7 := fun k => by
      split_ifs <;> simp
    simp only [h]
    fin_cases q <;> fin_cases q' <;> simp_all [Fin.prod_univ_four]
  refine le_trans (card_le_card_of_injOn rest (fun g hg => ?_) fun g hg g' hg' he => ?_) hcard.le
  · simp only [mem_coe, mem_filter, Fintype.mem_piFinset] at hg ⊢
    intro k
    simp only [hrest]
    split_ifs
    · exact mem_singleton_self 0
    · exact hg.1 k
  · simp only [mem_coe, mem_filter, Fintype.mem_piFinset] at hg hg'
    have hoff : ∀ k, k ≠ q → k ≠ q' → g k = g' k := fun k h1 h2 => by
      have := congrFun he k
      simpa [hrest, h1, h2] using this
    have hsum : ∀ T : Fin 4 → ℤ, ∑ k, T k * g k - ∑ k, T k * g' k = T q * (g q - g' q) + T q' * (g q' - g' q') :=
      fun T => by
        rw [← sum_sub_distrib]
        rw [sum_eq_add q q' hq (fun k _ hk => by simp [hoff k hk.1 hk.2]) (fun h => absurd (mem_univ q) h)
          (fun h => absurd (mem_univ q') h)]
        ring
    have ha := hsum Ta
    have hb := hsum Tb
    rw [hg.2.1, hg'.2.1, sub_self] at ha
    rw [hg.2.2, hg'.2.2, sub_self] at hb
    have e1 : (Ta q * Tb q' - Ta q' * Tb q) * (g q - g' q) = 0 := by linear_combination Tb q' * ha.symm - Ta q' * hb.symm
    have e2 : (Ta q * Tb q' - Ta q' * Tb q) * (g q' - g' q') = 0 := by
      linear_combination Ta q * hb.symm - Tb q * ha.symm
    have d1 := sub_eq_zero.1 ((mul_eq_zero.1 e1).resolve_left hdet)
    have d2 := sub_eq_zero.1 ((mul_eq_zero.1 e2).resolve_left hdet)
    funext k
    by_cases h1 : k = q
    · exact h1 ▸ d1
    · by_cases h2 : k = q'
      · exact h2 ▸ d2
      · exact hoff k h1 h2

end Pouw.SecurityProofs.Lead.CoinBranch
