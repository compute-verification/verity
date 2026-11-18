module

public import Proofs.Properties.Poub.Remote.SharedLink
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
import Mathlib.Algebra.BigOperators.Intervals

/-!
# A sample's draws number `min(s, U)`, on an accepted record (`Properties.Poub.Subprotocol.PublishedQBytes`, step 2)

`certify` reads each sample's draws as the sum of its windows' `drawn` counts (`windowsOf`, `samplesOf`). On an accepted
record, each window's draws are its share of the one array the sampler drew for the sample (`runSamplesRecords`), those
shares partition the array (`sum_shares`), and the array has `min(s, U)` entries (`Properties.Poub.sampleIsSubset`,
`sampled.size = min(samples, U)`). So a sample's draws number `min(s, U)` (`windowsOf_snd`).
-/

@[expose] public section

namespace Properties.Poub.Subprotocol.SecurityProofs

open Poub Poub.Session Properties.Poub Properties.Poub.Shared

/-- `filterMap` of a function that gives `some` exactly on `p` has `filter p`'s length. -/
theorem length_filterMap_eq {α β : Type} (f : α → Option β) (p : α → Bool)
    (hp : ∀ a, (f a).isSome = p a) (l : List α) : (l.filterMap f).length = (l.filter p).length := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    cases hfa : f a with
    | none =>
      have hpa : p a = false := by rw [← hp a, hfa]; rfl
      simp [List.filterMap_cons, List.filter_cons, hfa, hpa, ih]
    | some b =>
      have hpa : p a = true := by rw [← hp a, hfa]; rfl
      simp [List.filterMap_cons, List.filter_cons, hfa, hpa, ih]

/-- `shareOf sampled base n` has as many entries as `sampled` has in pass-indices `[base, base + n)`. -/
theorem shareOf_size (sampled : Array (Nat × Nat)) (base n : Nat) :
    (shareOf sampled base n).size =
      (sampled.toList.filter (fun x => base ≤ x.1 && x.1 < base + n)).length := by
  unfold shareOf
  rw [Array.size_eq_length_toList, Array.toList_filterMap]
  refine length_filterMap_eq _ _ (fun x => ?_) sampled.toList
  rcases x with ⟨i, p⟩
  cases h : (base ≤ i && i < base + n) with
  | false => simp [h]
  | true => simp [h]

/-- Over disjoint ranges, the entries counted sum to the entries that any range counts. -/
theorem sum_filter_disjoint {α : Type} (P : Nat → α → Bool) (m : Nat)
    (hdisj : ∀ (x : α) k₁ k₂, k₁ < m → k₂ < m → P k₁ x = true → P k₂ x = true → k₁ = k₂)
    (l : List α) : (∑ k ∈ Finset.range m, (l.filter (P k)).length) =
      (l.filter (fun x => decide (∃ k, k < m ∧ P k x = true))).length := by
  induction l with
  | nil => simp
  | cons a l ih =>
    have hcount : (∑ k ∈ Finset.range m, (if P k a then 1 else 0))
        = (if (decide (∃ k, k < m ∧ P k a = true)) then 1 else 0) := by
      by_cases h : ∃ k, k < m ∧ P k a = true
      · obtain ⟨k₀, hk₀, hPk₀⟩ := h
        rw [if_pos (by simpa using ⟨k₀, hk₀, hPk₀⟩)]
        rw [Finset.sum_eq_single k₀]
        · rw [if_pos hPk₀]
        · intro k hk hne
          rw [if_neg]
          intro hPk
          exact hne (hdisj a k k₀ (Finset.mem_range.1 hk) hk₀ hPk hPk₀)
        · intro h'; exact absurd (Finset.mem_range.2 hk₀) h'
      · rw [if_neg (by simpa using h)]
        apply Finset.sum_eq_zero
        intro k hk
        rw [if_neg]
        intro hPk
        exact h ⟨k, Finset.mem_range.1 hk, hPk⟩
    calc (∑ k ∈ Finset.range m, ((a :: l).filter (P k)).length)
        = (∑ k ∈ Finset.range m, ((if P k a then 1 else 0) + (l.filter (P k)).length)) := by
          apply Finset.sum_congr rfl; intro k _
          rw [List.filter_cons]; split <;> simp [List.length_cons] <;> omega
      _ = (∑ k ∈ Finset.range m, (if P k a then 1 else 0))
          + (∑ k ∈ Finset.range m, (l.filter (P k)).length) := by rw [Finset.sum_add_distrib]
      _ = (if (decide (∃ k, k < m ∧ P k a = true)) then 1 else 0)
          + (l.filter (fun x => decide (∃ k, k < m ∧ P k x = true))).length := by
          rw [hcount, ih]
      _ = ((a :: l).filter (fun x => decide (∃ k, k < m ∧ P k x = true))).length := by
          rw [List.filter_cons]; split <;> simp [List.length_cons] <;> omega

/-- The pass-index ranges `[b k, b k + n k)`, with `b k` the partial sums of `n`, are disjoint. -/
theorem ranges_disjoint (n : Nat → Nat) (b : Nat → Nat) (hb : ∀ k, b (k + 1) = b k + n k) (x : Nat × Nat) :
    ∀ k₁ k₂ : Nat, (b k₁ ≤ x.1 && x.1 < b k₁ + n k₁) = true →
      (b k₂ ≤ x.1 && x.1 < b k₂ + n k₂) = true → k₁ = k₂ := by
  have hmono : ∀ i j, i ≤ j → b i ≤ b j := by
    intro i j hij
    induction j with
    | zero => obtain rfl : i = 0 := Nat.le_zero.1 hij; exact Nat.le_refl _
    | succ m ih =>
      rcases Nat.lt_or_ge i (m + 1) with h | h
      · have hb' := ih (Nat.lt_succ_iff.1 h)
        rw [hb m]; omega
      · obtain rfl : i = m + 1 := Nat.le_antisymm hij h
        exact Nat.le_refl _
  intro k₁ k₂ h₁ h₂
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h₁ h₂
  by_contra hne
  rcases Nat.lt_or_ge k₁ k₂ with h | h
  · have := hmono (k₁ + 1) k₂ (by omega); rw [hb k₁] at this; omega
  · have : k₂ < k₁ := by omega
    have := hmono (k₂ + 1) k₁ (by omega); rw [hb k₂] at this; omega

/-- Every pass-index below `b m` falls in one of the `m` ranges, when `b 0 = 0` and the ranges are consecutive. -/
theorem range_covers (n : Nat → Nat) (b : Nat → Nat) (hb0 : b 0 = 0) (hb : ∀ k, b (k + 1) = b k + n k)
    (m : Nat) (i : Nat) (hi : i < b m) : ∃ k, k < m ∧ (b k ≤ i && i < b k + n k) = true := by
  induction m with
  | zero => omega
  | succ m ih =>
    rw [hb m] at hi
    rcases Nat.lt_or_ge i (b m) with h | h
    · obtain ⟨k, hk, hP⟩ := ih h
      exact ⟨k, by omega, hP⟩
    · exact ⟨m, by omega, by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega⟩

/-- **The shares partition the sampler's draws**: when the pass-index ranges `[b k, b k + n k)` are consecutive from
`0` and cover every draw's pass-index, the shares' sizes sum to the number of draws. -/
theorem sum_shares (sampled : Array (Nat × Nat)) (n b : Nat → Nat) (m : Nat)
    (hb0 : b 0 = 0) (hb : ∀ k, b (k + 1) = b k + n k)
    (hcover : ∀ j (hj : j < sampled.size), sampled[j].1 < b m) :
    (∑ k ∈ Finset.range m, (shareOf sampled (b k) (n k)).size) = sampled.size := by
  have heq : (∑ k ∈ Finset.range m, (shareOf sampled (b k) (n k)).size)
      = (sampled.toList.filter (fun x => decide (∃ k, k < m ∧ (b k ≤ x.1 && x.1 < b k + n k) = true))).length := by
    rw [← sum_filter_disjoint (fun k x => b k ≤ x.1 && x.1 < b k + n k) m
      (fun x k₁ k₂ _ _ => ranges_disjoint n b hb x k₁ k₂)]
    apply Finset.sum_congr rfl
    intro k _
    rw [shareOf_size]
  rw [heq]
  have hall : sampled.toList.filter (fun x => decide (∃ k, k < m ∧ (b k ≤ x.1 && x.1 < b k + n k) = true))
      = sampled.toList := by
    apply List.filter_eq_self.2
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem hx
    rw [Array.getElem_toList] at *
    have : sampled[j].1 < b m := hcover j (by rwa [← Array.length_toList])
    obtain ⟨k, hk, hP⟩ := range_covers n b hb0 hb m sampled[j].1 this
    exact decide_eq_true_eq.mpr ⟨k, hk, hP⟩
  rw [hall, Array.length_toList]

end Properties.Poub.Subprotocol.SecurityProofs
