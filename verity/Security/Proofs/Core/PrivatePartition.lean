import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Specs.Core.Guarantees.PrivatePartition

/-! # Proofs about private partitions: the per-Call check, the leakage weight, and the registration -/

namespace Verity.Partition

open Finset

variable {C : Circuit} {n m : ℕ} (P : Partition C n)

theorem width_mono {S T : Finset (Fin C.N)} (h : S ⊆ T) : C.width S ≤ C.width T :=
  Finset.sum_le_sum_of_subset h

theorem widthRule_mono {X W : ℕ} {S T : Finset (Fin C.N)} (h : S ⊆ T) (hT : WidthRule (C := C) X W T) :
    WidthRule (C := C) X W S := by
  rcases hT with hT | ⟨hc, hw⟩
  · exact Or.inl ((width_mono h).trans hT)
  · rcases Nat.lt_or_ge S.card 1 with h0 | h1
    · left
      rw [Finset.card_eq_zero.mp (show S.card = 0 by omega)]
      simp [Circuit.width]
    · exact Or.inr ⟨by have := Finset.card_le_card h; omega, (width_mono h).trans hw⟩

theorem outs_subset_localOuts (K : Calls C m) (u : Fin n) : P.outs u ⊆ P.localOuts K u := by
  intro g hg
  simp only [outs, Circuit.exits, gates, Finset.mem_filter, Finset.mem_univ, true_and] at hg
  obtain ⟨hgu, hout⟩ := hg
  have hin : C.isInput g = false := by
    cases h : C.isInput g
    · rfl
    · exact absurd ((P.input_iff g).mpr h) (by simp [hgu])
  simp only [localOuts, gates, Finset.mem_filter, Finset.mem_univ, true_and]
  refine ⟨hgu, ?_⟩
  rcases hout with ho | ⟨h, hh, hr⟩
  · rcases K.out g ho with hi | hr
    · simp [hin] at hi
    · exact Or.inl hr
  · by_cases hk : K.unit h = K.unit g
    · exact Or.inr ⟨h, hr, hk, by rw [hgu]; exact hh⟩
    · rcases K.cross g h hr hk with hi | hr'
      · simp [hin] at hi
      · exact Or.inl hr'

theorem fits_of_local (K : Calls C m) {X W : ℕ} (h : P.FitsLocally K X W) : P.Fits X W :=
  fun u => widthRule_mono (P.outs_subset_localOuts K u) (h u)

theorem outBits_le {X W : ℕ} (h : P.Fits X W) (u : Fin n) : P.outBits u ≤ max X W := by
  rcases h u with h | ⟨_, h⟩
  · exact h.trans (le_max_left _ _)
  · exact h.trans (le_max_right _ _)

theorem sum_outBits_le {X W : ℕ} (h : P.Fits X W) (T : Finset (Fin n)) :
    ∑ u ∈ T, P.outBits u ≤ T.card * max X W := by
  simpa using Finset.sum_le_card_nsmul T P.outBits (max X W) fun u _ => P.outBits_le h u

end Verity.Partition

namespace Verity.SecurityProofs

theorem PrivateFitsOfLocal : Guarantees.PrivateFitsOfLocal :=
  fun _ _ _ K P _ _ h => P.fits_of_local K h

theorem PrivateOutBitsLe : Guarantees.PrivateOutBitsLe :=
  fun _ _ P _ _ h T => P.sum_outBits_le h T

theorem RegisteredMeets : Guarantees.RegisteredMeets := by
  intro R C advice Q root ⟨n, P, ⟨_, hf, hn, _⟩, hr⟩
  exact ⟨n, P, hr, hn, hf, P.sum_outBits_le hf⟩

end Verity.SecurityProofs
