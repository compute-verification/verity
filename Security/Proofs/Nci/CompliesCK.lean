import Definitions.Nci.Guarantees.CompliesCK
import Proofs.Core.Padding

/-!
# Proofs of NCI's policy as a `(c, k)`-partition

A Boolean declaration's novel reads are its unregistered input gates, so a complying IU of `|I| ≤ Γ` gates reads at
most `α |I| ≤ α Γ` of them, and `Padding` makes the division a `(Γ, ⌊αΓ⌋)`-partition. Through the bit-level expansion,
an IU's size becomes its width, at least `|I|` and at most `W |I|`, and its novel reads do not grow (`ExpandSize`).
-/

namespace Verity

open Finset

theorem Circuit.width_eq_card {C : Circuit} (hC : C.Boolean) (S : Finset (Fin C.N)) : C.width S = S.card := by
  simp [Circuit.width, hC.one_bit]

theorem Circuit.card_le_width (C : Circuit) (S : Finset (Fin C.N)) (h : ∀ g ∈ S, 1 ≤ C.w g) :
    S.card ≤ C.width S := by
  rw [card_eq_sum_ones]
  exact sum_le_sum h

theorem Circuit.width_le (C : Circuit) (S : Finset (Fin C.N)) (W : ℕ) (h : ∀ g, C.w g ≤ W) :
    C.width S ≤ S.card * W := by
  calc ∑ g ∈ S, C.w g ≤ ∑ _g ∈ S, W := sum_le_sum fun g _ => h g
    _ = S.card * W := by rw [sum_const, smul_eq_mul]

end Verity

namespace Nci.SecurityProofs

open Verity Finset

theorem CompliesCK : Guarantees.CompliesCK := by
  intro Γ α d hB ⟨B, P, hP⟩
  have hk : ∀ u, (d.unregistered (P.gates u)).card ≤ ⌊α * Γ⌋₊ := by
    intro u
    obtain ⟨hs, hr⟩ := hP u
    have hr' : ((d.unregistered (P.gates u)).card : ℝ) ≤ α * (P.gates u).card := by
      rw [← Circuit.width_eq_card hB]
      exact hr
    rcases Nat.eq_zero_or_pos (d.unregistered (P.gates u)).card with h0 | hpos
    · rw [h0]
      exact Nat.zero_le _
    · apply Nat.le_floor
      have hs' : ((P.gates u).card : ℝ) ≤ Γ := by exact_mod_cast hs
      have hpos' : (0 : ℝ) < (d.unregistered (P.gates u)).card := by exact_mod_cast hpos
      have hα : 0 < α := by
        by_contra hneg
        have : α * (P.gates u).card ≤ 0 :=
          mul_nonpos_of_nonpos_of_nonneg (not_lt.1 hneg) (Nat.cast_nonneg _)
        linarith
      calc ((d.unregistered (P.gates u)).card : ℝ) ≤ α * (P.gates u).card := hr'
        _ ≤ α * Γ := mul_le_mul_of_nonneg_left hs' hα.le
  obtain ⟨d', P', hpad, hB', hcard, hun⟩ := Verity.SecurityProofs.Padding d B Γ P univ hB fun u _ => (hP u).1
  exact ⟨B, P, d', P', hpad, ⟨hB', fun u => (hcard u).1 (mem_univ u), fun u => (hun u).symm ▸ hk u⟩⟩

theorem ExpandComplies : Guarantees.ExpandComplies := by
  intro Γ W α d hα hw hW ⟨B, P, hP⟩
  refine ⟨B, P.expand, fun u => ?_⟩
  obtain ⟨hsize, hunreg⟩ := (Verity.SecurityProofs.ExpandSize d B P).2 u
  obtain ⟨hs, hr⟩ := hP u
  have hcard : (P.gates u).card ≤ d.D.width (P.gates u) :=
    Circuit.card_le_width _ _ fun g hg => hw g (P.isInput_of_mem_gates hg)
  have hsize' : d.expand.D.size (P.expand.gates u) = d.D.width (P.gates u) := hsize
  have hnovel : novelReads d.expand (P.expand.gates u) ≤ novelReads d (P.gates u) := by
    have : novelReads d.expand (P.expand.gates u) = (d.expand.unregistered (P.expand.gates u)).card := by
      simp [novelReads, Circuit.width, Declaration.unregistered]
    rw [this]
    exact hunreg
  refine ⟨?_, ?_⟩
  · rw [hsize']
    calc d.D.width (P.gates u) ≤ (P.gates u).card * W := Circuit.width_le _ _ W hW
      _ ≤ Γ * W := Nat.mul_le_mul_right _ hs
  · rw [hsize']
    calc (novelReads d.expand (P.expand.gates u) : ℝ) ≤ novelReads d (P.gates u) := by exact_mod_cast hnovel
      _ ≤ α * (P.gates u).card := hr
      _ ≤ α * d.D.width (P.gates u) := mul_le_mul_of_nonneg_left (by exact_mod_cast hcard) hα

end Nci.SecurityProofs
