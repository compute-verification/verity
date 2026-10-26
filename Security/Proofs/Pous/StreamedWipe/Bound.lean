import Proofs.Pous.StreamedWipe.Count
import Definitions.Pous.Guarantees.StreamedWipe

/-!
# The streamed-label wipe: the bound

`win_count` divided by the number of runs `2^q · N^k · 2^(b·N)` is `bound` (`win_ideal_le`); the real game is the
ideal one plus the PRF's advantage against the game's distinguisher (`streamed_wipe`).
-/

namespace PousStreamedWipe

open Finset Pous Pous.StreamedWipe
open scoped ENNReal

theorem card_runs (N k b q : ℕ) :
    Fintype.card ((Bits q × (Fin k → Fin N)) × (Fin N → Bits b)) = 2 ^ q * N ^ k * (2 ^ b) ^ N := by
  simp [Fintype.card_prod, Fintype.card_bool, Fintype.card_fin]

/-- **The ideal game.** With uniform labels, `Pr[accepted ∧ keeps z] ≤ bound N k b q M E g` for every `g`. -/
theorem win_ideal_le {N k b q M E : ℕ} (P : Prover N k b q M E) (g : ℕ) :
    winIdeal P ≤ bound N k b q M E g := by
  classical
  unfold winIdeal pr bound
  rw [card_runs]
  by_cases hN : N ^ k = 0
  · have h0 : (univ.filter (wipeDistinguisher P)).card = 0 := by
      have h := card_filter_le (univ : Finset ((Bits q × (Fin k → Fin N)) × (Fin N → Bits b))) (wipeDistinguisher P)
      rw [card_univ, card_runs, hN] at h
      simpa using h
    simp [h0]
  have hNk : ((N : ℝ≥0∞) ^ k) ≠ 0 := by exact_mod_cast hN
  have hNk' : ((N : ℝ≥0∞) ^ k) ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.natCast_ne_top N)
  apply ENNReal.div_le_of_le_mul
  calc ((univ.filter (wipeDistinguisher P)).card : ℝ≥0∞)
      ≤ ((2 ^ q * (2 ^ b) ^ N * g ^ k +
          k * (N ^ k * (N.choose (g + 1) * (2 ^ M * 2 ^ E * (2 ^ b) ^ (N - (g + 1))))) : ℕ) : ℝ≥0∞) := by
        exact_mod_cast win_count g P
    _ ≤ _ := by
      rw [add_mul]
      push_cast
      gcongr ?_ + ?_
      · have hpow : ((g : ℝ≥0∞) / N) ^ k = (g : ℝ≥0∞) ^ k / (N : ℝ≥0∞) ^ k := by
          rw [div_eq_mul_inv, mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv]
        rw [hpow, show (g : ℝ≥0∞) ^ k / (N : ℝ≥0∞) ^ k * (2 ^ q * (N : ℝ≥0∞) ^ k * (2 ^ b) ^ N)
            = ((g : ℝ≥0∞) ^ k / (N : ℝ≥0∞) ^ k * (N : ℝ≥0∞) ^ k) * (2 ^ q * (2 ^ b) ^ N) by ring,
          ENNReal.div_mul_cancel hNk hNk']
        exact le_of_eq (by ring)
      · by_cases hg : g + 1 ≤ N
        · have hD : (2 : ℝ≥0∞) ^ (q + (g + 1) * b) ≠ 0 := pow_ne_zero _ two_ne_zero
          have hD' : (2 : ℝ≥0∞) ^ (q + (g + 1) * b) ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofNat_ne_top
          have hsplit : (2 : ℝ≥0∞) ^ q * (N : ℝ≥0∞) ^ k * (2 ^ b) ^ N
              = 2 ^ (q + (g + 1) * b) * ((N : ℝ≥0∞) ^ k * (2 ^ b) ^ (N - (g + 1))) := by
            have hN : ((2 : ℝ≥0∞) ^ b) ^ N = (2 ^ b) ^ (g + 1) * (2 ^ b) ^ (N - (g + 1)) := by
              rw [← pow_add, Nat.add_sub_cancel' hg]
            rw [hN, pow_add (2 : ℝ≥0∞) q, mul_comm (g + 1) b, pow_mul]
            ring
          have hcancel : ∀ X Y : ℝ≥0∞, X / 2 ^ (q + (g + 1) * b) * (2 ^ (q + (g + 1) * b) * Y) = X * Y :=
            fun X Y => by rw [← mul_assoc, ENNReal.div_mul_cancel hD hD']
          rw [hsplit, hcancel, pow_add]
          exact le_of_eq (by ring)
        · rw [Nat.choose_eq_zero_of_lt (by omega)]
          simp

/-- **`StreamedWipe`**: the ideal bound, plus the PRF's advantage against the game's distinguisher. -/
theorem streamed_wipe : Pous.Guarantees.StreamedWipe := by
  intro N k b q M E κ g prf P
  calc winReal prf P ≤ winIdeal P + (winReal prf P - winIdeal P) := le_add_tsub
    _ ≤ bound N k b q M E g + prfAdvantage prf (wipeDistinguisher P) := add_le_add (win_ideal_le P g) le_rfl

end PousStreamedWipe
