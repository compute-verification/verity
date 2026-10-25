import Definitions.Pouw.PearlC.Rows

/-!
# A layout's global rows: the arithmetic

Unit `u`'s row `i` is global row `rowIdx L u i`, and back: a global row `v < rowsOf L` is row `rowOfRow L v` of unit
`unitOfRow L v`, a bijection between the declared rows `{(u, i) | u < N, i < m_u}` and `range (rowsOf L)`
(`unitOfRow_rowIdx`, `rowOfRow_rowIdx`, `rowIdx_lt_rowsOf`; `rowIdx_unitOfRow`, `rowOfRow_lt`, `unitOfRow_lt`). Units
with no rows are skipped, so none of this assumes `0 < m_u`.
-/

namespace Pouw.PearlC

open Finset

theorem rowOffset_zero (L : Layout) : rowOffset L 0 = 0 := by
  simp [rowOffset]

theorem rowOffset_succ (L : Layout) (u : ℕ) : rowOffset L (u + 1) = rowOffset L u + (L.shape u).m := by
  unfold rowOffset; exact sum_range_succ _ _

theorem rowOffset_mono (L : Layout) {u w : ℕ} (h : u ≤ w) : rowOffset L u ≤ rowOffset L w := by
  unfold rowOffset; exact sum_le_sum_of_subset (range_subset_range.mpr h)

theorem unitOfRow_rowIdx {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    unitOfRow L (rowIdx L u i) = u := by
  unfold unitOfRow
  rw [Nat.findGreatest_eq_iff]
  refine ⟨by omega, fun _ => ?_, fun w hw _ => ?_⟩
  · show rowOffset L u ≤ rowIdx L u i
    unfold rowIdx; omega
  · have h1 := rowOffset_mono L (Nat.succ_le_of_lt hw)
    rw [rowOffset_succ] at h1
    show ¬ rowOffset L w ≤ rowIdx L u i
    unfold rowIdx; omega

theorem rowOfRow_rowIdx {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    rowOfRow L (rowIdx L u i) = i := by
  unfold rowOfRow; rw [unitOfRow_rowIdx hu hi]; unfold rowIdx; omega

theorem rowIdx_lt_rowsOf {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    rowIdx L u i < rowsOf L := by
  have h1 := rowOffset_mono L (Nat.succ_le_of_lt hu)
  rw [rowOffset_succ] at h1
  unfold rowIdx rowsOf; omega

theorem rowIdx_inj {L : Layout} {u i u' i' : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) (hu' : u' < L.N)
    (hi' : i' < (L.shape u').m) (h : rowIdx L u i = rowIdx L u' i') : u = u' ∧ i = i' := by
  have h1 := unitOfRow_rowIdx hu hi
  have h2 := unitOfRow_rowIdx hu' hi'
  rw [h] at h1
  have e : u = u' := h1.symm.trans h2
  subst e
  refine ⟨rfl, ?_⟩
  unfold rowIdx at h; omega

theorem unitOfRow_lt {L : Layout} {v : ℕ} (hv : v < rowsOf L) : unitOfRow L v < L.N := by
  have hN : L.N ≠ 0 := by
    intro h
    unfold rowsOf rowOffset at hv
    rw [h] at hv
    simp at hv
  have := Nat.findGreatest_le (P := fun u => rowOffset L u ≤ v) (L.N - 1)
  unfold unitOfRow; omega

/-- On a one-shape layout the declared rows are `N·m`. -/
theorem rowsOf_const {L : Layout} {s : Shape} (h : ∀ u < L.N, L.shape u = s) : rowsOf L = L.N * s.m := by
  unfold rowsOf rowOffset
  rw [sum_congr rfl fun u hu => by rw [h u (mem_range.mp hu)], sum_const, card_range, smul_eq_mul]

theorem rowOffset_unitOfRow_le (L : Layout) (v : ℕ) : rowOffset L (unitOfRow L v) ≤ v := by
  unfold unitOfRow
  exact Nat.findGreatest_spec (P := fun u => rowOffset L u ≤ v) (Nat.zero_le _)
    (by rw [rowOffset_zero]; exact Nat.zero_le _)

/-- Every global row is its unit's row at its offset. -/
theorem rowIdx_unitOfRow (L : Layout) (v : ℕ) : rowIdx L (unitOfRow L v) (rowOfRow L v) = v := by
  have := rowOffset_unitOfRow_le L v
  unfold rowIdx rowOfRow; omega

/-- A declared global row is a row of its unit's shape. -/
theorem rowOfRow_lt {L : Layout} {v : ℕ} (hv : v < rowsOf L) : rowOfRow L v < (L.shape (unitOfRow L v)).m := by
  have hN := unitOfRow_lt hv
  have hle := rowOffset_unitOfRow_le L v
  have hlt : v < rowOffset L (unitOfRow L v + 1) := by
    by_cases h : unitOfRow L v + 1 = L.N
    · rw [h]; exact hv
    · have h2 : ¬ rowOffset L (unitOfRow L v + 1) ≤ v :=
        Nat.findGreatest_is_greatest (P := fun u => rowOffset L u ≤ v) (n := L.N - 1)
          (k := unitOfRow L v + 1) (Nat.lt_succ_self _) (by omega)
      omega
  rw [rowOffset_succ] at hlt
  unfold rowOfRow; omega

end Pouw.PearlC
