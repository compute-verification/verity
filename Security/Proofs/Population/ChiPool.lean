import Definitions.Population.ChiPool
import Proofs.Population.DrawPressure

/-!
# The cheating pool with the verdict lag: harm through the halt

The game is `Definitions.Population.ChiPool`'s. The deployment halts `Δ` windows after the first caught window's
commitment, so a cheater does harm in those `Δ` windows as well. `tail_cum` alone stops at the caught window.

* `cumLag_le`, **the lag costs at most `Δ ℓ`**: what leaves through the halt is at most `cum + Δ ℓ`, path by path, when
  every window leaks at most `ℓ`. With `Δ = 0` it is `cum` (`cumLag_zero`).
* `tail_cumLag`: under detection pressure `κ` the probability that what leaves through the halt reaches `x` is at most
  `exp (-κ (x - (1 + Δ) ℓ))`, for every adaptive strategy and every horizon.
* `draw_tail_cumLag`, **the cheating pool under the deployed draw**: `k` of the `n` units drawn without replacement
  (`drawPressure`, `κ = k/n`). With probability at least `1 - δ` the harm through the halt is below
  `(1 + Δ) ℓ + (n/k) ln (1/δ)`.
* `ex_cumUpToLag`, `draw_ex_cumUpToLag`, **the rate form**: through the halt the `j`-th detection triggers, at most
  `j (ℓ + 1/κ) + Δ ℓ` leaves on average, which is `j (ℓ + n/k) + Δ ℓ` under the deployed draw.
-/

namespace Population.Exfil

open Finset

variable {Ω : Type}

theorem prob_mono [Fintype Ω] {μ : Ω → ℝ} (hμ : ∀ c, 0 ≤ μ c) {N : ℕ} {E E' : (Fin N → Ω) → Prop}
    (h : ∀ cs, E cs → E' cs) : prob μ N E ≤ prob μ N E' := by
  unfold prob
  refine ex_mono hμ fun cs => ?_
  by_cases h1 : E cs
  · simp [h1, h cs h1]
  · simp only [h1, ↓reduceIte]
    split_ifs <;> norm_num

theorem cumUpTo_zero [DecidableEq Ω] (A : Strategy Ω) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) :
    A.cumUpTo n 0 h cs = 0 := by
  cases n <;> rfl

/-- Through the first detection, `cumUpTo` is `cum`. -/
theorem cumUpTo_one [DecidableEq Ω] (A : Strategy Ω) :
    ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω), A.cumUpTo n 1 h cs = A.cum n h cs
  | 0, _, _ => rfl
  | n + 1, h, cs => by
    rw [Strategy.cumUpTo, Strategy.cum, cumUpTo_one A n, cumUpTo_zero]

theorem lagLeak_le {A : Strategy Ω} {ℓ : ℝ} (hL : LeakBounded ℓ A) :
    ∀ (n Δ : ℕ) (h : List Ω) (cs : Fin n → Ω), A.lagLeak n Δ h cs ≤ Δ * ℓ
  | 0, Δ, _, _ => by
    rw [Strategy.lagLeak]
    exact mul_nonneg (Nat.cast_nonneg Δ) ((hL []).1.trans (hL []).2)
  | _ + 1, 0, _, _ => by simp [Strategy.lagLeak]
  | n + 1, Δ + 1, h, cs => by
    rw [Strategy.lagLeak]
    have := lagLeak_le hL n Δ (cs 0 :: h) (Fin.tail cs)
    push_cast
    linarith [(hL h).2]

theorem lagLeak_zero (A : Strategy Ω) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) : A.lagLeak n 0 h cs = 0 := by
  cases n <;> rfl

/-- **The lag costs at most `Δ ℓ`**: through the halt the `j`-th detection triggers, at most what leaves before it plus
`Δ ℓ`, path by path. -/
theorem cumUpToLag_le [DecidableEq Ω] {A : Strategy Ω} {ℓ : ℝ} (hL : LeakBounded ℓ A) (Δ : ℕ) :
    ∀ (n j : ℕ) (h : List Ω) (cs : Fin n → Ω), A.cumUpToLag Δ n j h cs ≤ A.cumUpTo n j h cs + Δ * ℓ
  | 0, _, _, _ => by
    rw [Strategy.cumUpToLag, Strategy.cumUpTo, zero_add]
    exact mul_nonneg (Nat.cast_nonneg Δ) ((hL []).1.trans (hL []).2)
  | n + 1, 0, h, cs => by
    rw [Strategy.cumUpToLag, cumUpTo_zero, zero_add]
    exact lagLeak_le hL _ _ h cs
  | n + 1, j + 1, h, cs => by
    rw [Strategy.cumUpToLag, Strategy.cumUpTo]
    split_ifs
    · linarith [cumUpToLag_le hL Δ n (j + 1) (cs 0 :: h) (Fin.tail cs)]
    · linarith [cumUpToLag_le hL Δ n j (cs 0 :: h) (Fin.tail cs)]

theorem cumUpToLag_zero [DecidableEq Ω] (A : Strategy Ω) :
    ∀ (n j : ℕ) (h : List Ω) (cs : Fin n → Ω), A.cumUpToLag 0 n j h cs = A.cumUpTo n j h cs
  | 0, _, _, _ => rfl
  | n + 1, 0, h, cs => by rw [Strategy.cumUpToLag, lagLeak_zero, cumUpTo_zero]
  | n + 1, j + 1, h, cs => by
    rw [Strategy.cumUpToLag, Strategy.cumUpTo, cumUpToLag_zero A n (j + 1), cumUpToLag_zero A n j]

/-- With no lag, what leaves through the halt is what leaves through the first detection. -/
theorem cumLag_zero [DecidableEq Ω] (A : Strategy Ω) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) :
    A.cumLag 0 n h cs = A.cum n h cs := by
  rw [Strategy.cumLag, cumUpToLag_zero, cumUpTo_one]

/-- **What leaves through the halt** is at most what leaves through the first detection plus `Δ ℓ`. -/
theorem cumLag_le [DecidableEq Ω] {A : Strategy Ω} {ℓ : ℝ} (hL : LeakBounded ℓ A) (Δ n : ℕ) (h : List Ω)
    (cs : Fin n → Ω) : A.cumLag Δ n h cs ≤ A.cum n h cs + Δ * ℓ := by
  rw [Strategy.cumLag, ← cumUpTo_one]
  exact cumUpToLag_le hL Δ n 1 h cs

/-- **Harm through the halt**: under detection pressure `κ`, with every window leaking at most `ℓ` and the halt `Δ`
windows after the first caught window's commitment, what leaves through the halt reaches `x` with probability at most
`exp (-κ (x - (1 + Δ) ℓ))`, for every adaptive strategy and every horizon. -/
theorem tail_cumLag [Fintype Ω] [DecidableEq Ω] {μ : Ω → ℝ} (hμ : IsDist μ) {κ ℓ : ℝ} (hκ : 0 ≤ κ)
    {A : Strategy Ω} (hP : DetectionPressure μ κ A) (hL : LeakBounded ℓ A) (Δ n : ℕ) (h : List Ω) (x : ℝ) :
    prob μ n (fun cs => x ≤ A.cumLag Δ n h cs) ≤ Real.exp (-(κ * (x - (1 + Δ) * ℓ))) := by
  calc prob μ n (fun cs => x ≤ A.cumLag Δ n h cs) ≤ prob μ n (fun cs => x - Δ * ℓ ≤ A.cum n h cs) :=
        prob_mono hμ.1 fun cs hx => by linarith [cumLag_le hL Δ n h cs]
    _ ≤ Real.exp (-(κ * (x - Δ * ℓ - ℓ))) := tail_cum hμ hκ hP hL n h _
    _ = Real.exp (-(κ * (x - (1 + Δ) * ℓ))) := by rw [show x - Δ * ℓ - ℓ = x - (1 + Δ) * ℓ by ring]

/-- **The rate form with the lag**: through the halt the `j`-th detection triggers, `Δ` windows after it, at most
`j (ℓ + 1/κ) + Δ ℓ` leaves on average. -/
theorem ex_cumUpToLag [Fintype Ω] [DecidableEq Ω] {μ : Ω → ℝ} (hμ : IsDist μ) {κ ℓ : ℝ} (hκ : 0 < κ)
    {A : Strategy Ω} (hP : DetectionPressure μ κ A) (hL : LeakBounded ℓ A) (Δ n j : ℕ) (h : List Ω) :
    ex μ n (A.cumUpToLag Δ n j h) ≤ j * (ℓ + 1 / κ) + Δ * ℓ := by
  calc ex μ n (A.cumUpToLag Δ n j h) ≤ ex μ n (fun cs => Δ * ℓ + A.cumUpTo n j h cs) :=
        ex_mono hμ.1 fun cs => by linarith [cumUpToLag_le hL Δ n j h cs]
    _ = Δ * ℓ + ex μ n (A.cumUpTo n j h) := ex_add_const hμ _ _
    _ ≤ j * (ℓ + 1 / κ) + Δ * ℓ := by linarith [ex_cumUpTo hμ hκ hP hL n j h]

/-- **The cheating pool under the deployed draw**: `0 < k ≤ n`, each window as in `drawPressure` and leaking at most
`ℓ`, and the halt `Δ` windows after the first caught window's commitment. With probability at least `1 - δ` the harm
through the halt is below `(1 + Δ) ℓ + (n/k) ln (1/δ)`, for every adaptive strategy, every horizon `N` and every
history `h`. -/
theorem draw_tail_cumLag {n k : ℕ} (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ}
    (hL : LeakBounded ℓ A) {δ : ℝ} (hδ : 0 < δ) (Δ N : ℕ) (h : List (Verity.Draw.Subset n k)) :
    1 - δ ≤ prob (uniformDraw n k) N (fun cs => A.cumLag Δ N h cs < (1 + Δ) * ℓ + n / k * Real.log (1 / δ)) := by
  have hμ := uniformDraw_isDist hkn
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  set x := (1 + (Δ : ℝ)) * ℓ + n / k * Real.log (1 / δ)
  have ht := tail_cumLag hμ (by positivity) (drawPressure B w hw hesc hleak) hL Δ N h x
  have e : Real.exp (-((k : ℝ) / n * (x - (1 + Δ) * ℓ))) = δ := by
    rw [show (k : ℝ) / n * (x - (1 + Δ) * ℓ) = Real.log (1 / δ) by simp only [x]; field_simp; ring, one_div,
      Real.log_inv, neg_neg, Real.exp_log hδ]
  rw [prob_congr (E' := fun cs => ¬x ≤ A.cumLag Δ N h cs) fun cs => not_le.symm, prob_not hμ]
  linarith

/-- **The rate form under the deployed draw**: through the halt the `j`-th detection triggers, `Δ` windows after it, at
most `j (ℓ + n/k) + Δ ℓ` leaves on average. -/
theorem draw_ex_cumUpToLag {n k : ℕ} (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ}
    (hL : LeakBounded ℓ A) (Δ N j : ℕ) (h : List (Verity.Draw.Subset n k)) :
    ex (uniformDraw n k) N (A.cumUpToLag Δ N j h) ≤ j * (ℓ + n / k) + Δ * ℓ := by
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  have := ex_cumUpToLag (uniformDraw_isDist hkn) (by positivity : (0 : ℝ) < k / n) (drawPressure B w hw hesc hleak)
    hL Δ N j h
  rwa [one_div_div] at this

end Population.Exfil
