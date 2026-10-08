import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Models.Core.Guarantees.PrivatePartition

/-! # Proofs about private partitions: the per-Call check, the leakage weight, and the registration -/

namespace Verity.Partition

open Finset

variable {C : Circuit} {n m : ℕ} (P : Partition C n)

theorem width_mono {S T : Finset (Fin C.N)} (h : S ⊆ T) : C.width S ≤ C.width T :=
  Finset.sum_le_sum_of_subset h

theorem widthRule_mono {X : ℕ} {S T : Finset (Fin C.N)} (h : S ⊆ T) (hT : WidthRule (C := C) X T) :
    WidthRule (C := C) X S :=
  (width_mono h).trans hT

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

theorem fits_of_local (K : Calls C m) {X : ℕ} (h : P.FitsLocally K X) : P.Fits X :=
  fun u => widthRule_mono (P.outs_subset_localOuts K u) (h u)

theorem outBits_le {X : ℕ} (h : P.Fits X) (u : Fin n) : P.outBits u ≤ X := h u

theorem sum_outBits_le {X : ℕ} (h : P.Fits X) (T : Finset (Fin n)) :
    ∑ u ∈ T, P.outBits u ≤ T.card * X := by
  simpa using Finset.sum_le_card_nsmul T P.outBits X fun u _ => P.outBits_le h u

end Verity.Partition

namespace Verity.SecurityProofs

theorem PrivateFitsOfLocal : Guarantees.PrivateFitsOfLocal :=
  fun _ _ _ K P _ h => P.fits_of_local K h

theorem PrivateOutBitsLe : Guarantees.PrivateOutBitsLe :=
  fun _ _ P _ h T => P.sum_outBits_le h T

theorem RegisteredMeets : Guarantees.RegisteredMeets := by
  intro R C advice hbind Q root ⟨n₀, P₀, ⟨_, hf, hn, _⟩, hr₀⟩ n P hr
  have h : (⟨n, P⟩ : (n : ℕ) × Partition C n) = ⟨n₀, P₀⟩ := hbind (hr.trans hr₀.symm)
  cases h
  exact ⟨hn, hf, P₀.sum_outBits_le hf⟩

end Verity.SecurityProofs
