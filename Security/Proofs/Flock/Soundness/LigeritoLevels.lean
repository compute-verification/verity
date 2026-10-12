import Proofs.Flock.Soundness.LigeritoMCA

/-!
# Ligerito's levels meet the Johnson-range MCA bound (PROTOCOL.md §13, §15)

Each level's folds happen over `K`, in the level's Reed–Solomon code embedded from `F`
(`codeK`: `2^logCols` coefficients on the `2^logLen` points `ω_q`). For every level of a
`fast100` schedule (`Level.Ok`), `prCoin_mca_le` applies at the level's radius, and its
bound is the accounting's `Level.mcaA / qK` (`prCoin_level_mca_le`):
* `1 < 2^logCols`, since `logCols ≥ 1`;
* the radius `1 − √ρ − η` is positive, since `ρ ≤ 1/2`;
* the radius is in the Johnson range `< 1 − √((k − 1)/n)`, since `(k − 1)/n < ρ`;
* the bound's numerator, at `n = 2^logLen`, `k = 2^logCols` and `δ` the radius, is `mcaA`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A level's evaluation points `ω_q`, in `K`. -/
def domainK (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) : Fin (2 ^ d) ↪ K :=
  (domainEmb A hA d hd).trans ⟨A.embed, A.embed.injective⟩

/-- A level's Reed–Solomon code over `K`: the code its folds live in. -/
noncomputable def codeK (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) :
    Submodule K (Fin (2 ^ l.logLen) → K) :=
  ReedSolomon.code (domainK A hA l.logLen hf.2) (2 ^ l.logCols)

theorem rate_le_half (l : Level) (hl : l.Ok) : l.rate ≤ 1 / 2 := by
  unfold Level.rate
  calc (2⁻¹ : ℝ) ^ l.logInvRate ≤ (2⁻¹ : ℝ) ^ 1 :=
        pow_le_pow_of_le_one (by norm_num) (by norm_num) hl.1
    _ = 1 / 2 := by norm_num

theorem radius_pos (l : Level) (hl : l.Ok) : 0 < l.radius := by
  unfold Level.radius eta
  have h1 : Real.sqrt l.rate ≤ Real.sqrt (1 / 2) := Real.sqrt_le_sqrt (rate_le_half l hl)
  have h2 : Real.sqrt (1 / 2) < 3 / 4 := by
    rw [Real.sqrt_lt' (by norm_num)]; norm_num
  linarith

theorem radius_lt_johnson (l : Level) (hl : l.Ok) :
    l.radius < 1 - ((((2 ^ l.logCols - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen) ^ ((1 : ℝ) / 2)) := by
  have hrate : l.rate = (2 : ℝ) ^ l.logCols / (2 : ℝ) ^ l.logLen := by
    unfold Level.rate Level.logLen
    rw [pow_add, inv_pow]
    have : (2 : ℝ) ^ l.logCols ≠ 0 := pow_ne_zero _ two_ne_zero
    field_simp
  have hlt : ((2 ^ l.logCols - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen < l.rate := by
    rw [hrate]
    apply div_lt_div_of_pos_right _ (by positivity)
    have : 1 ≤ 2 ^ l.logCols := Nat.one_le_two_pow
    rw [Nat.cast_sub this]
    push_cast
    linarith
  have hsq : (((2 ^ l.logCols - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen) ^ ((1 : ℝ) / 2) <
      Real.sqrt l.rate := by
    rw [Real.sqrt_eq_rpow]
    exact Real.rpow_lt_rpow (by positivity) hlt (by norm_num)
  unfold Level.radius eta
  linarith

/-- **The MCA bound at a level.** For every level of a `fast100` schedule, whatever two
words the prover fixes, the MCA event of the level's code over `K` at the level's radius has
probability at most `mcaA / 2^256`. -/
theorem prCoin_level_mca_le (A : Arith F K) (hA : A.Correct)
    (l : Level) (hl : l.Ok) (hf : l.Fits) (U : Fin 2 → Fin (2 ^ l.logLen) → K) :
    prCoin (fun γ : K => IsMCA (AffineLineGenerator K) (codeK A hA l hf) γ U l.radius) ≤
      ENNReal.ofReal (l.mcaA / qK) := by
  have hr := radius_pos l hl
  have hk : 1 < 2 ^ l.logCols := Nat.one_lt_two_pow (by have := hl.2.2; omega)
  have hJ := radius_lt_johnson l hl
  have h := prCoin_mca_le (domainK A hA l.logLen hf.2) (2 ^ l.logCols) l.radius.toNNReal hk
    (by rw [Fintype.card_fin, Real.coe_toNNReal _ hr.le]; push_cast; exact hJ) U
  rw [Real.coe_toNNReal _ hr.le] at h
  refine h.trans (le_of_eq ?_)
  rw [Fintype.card_fin, hA.card_K]
  unfold Level.mcaA qK
  push_cast
  congr 3
  norm_num

end FlockSoundness
