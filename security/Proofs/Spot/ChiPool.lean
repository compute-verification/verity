module

public import Definitions.Spot.ChiPool
public import Definitions.Spot.SamplePressure
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Proofs.Spot.SamplePressure
import Proofs.Spot.Exfiltration

/-! The cheating pool with the verdict lag: harm through the halt. -/

@[expose] public section

namespace Population.Exfil

open Finset

variable {Ω : Type}

theorem prob_mono [Fintype Ω] {μ : PMF Ω} {N : ℕ} {E E' : (Fin N → Ω) → Prop}
    (h : ∀ cs, E cs → E' cs) : prob μ N E ≤ prob μ N E' :=
  Verity.Game.prPMFReal_mono _ h

theorem cumUpTo_zero [DecidableEq Ω] (A : Strategy Ω) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) :
    A.cumUpTo n 0 h cs = 0 := by
  cases n <;> rfl

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

theorem cumLag_zero [DecidableEq Ω] (A : Strategy Ω) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) :
    A.cumLag 0 n h cs = A.cum n h cs := by
  rw [Strategy.cumLag, cumUpToLag_zero, cumUpTo_one]

theorem cumLag_le [DecidableEq Ω] {A : Strategy Ω} {ℓ : ℝ} (hL : LeakBounded ℓ A) (Δ n : ℕ) (h : List Ω)
    (cs : Fin n → Ω) : A.cumLag Δ n h cs ≤ A.cum n h cs + Δ * ℓ := by
  rw [Strategy.cumLag, ← cumUpTo_one]
  exact cumUpToLag_le hL Δ n 1 h cs

theorem tail_cumLag [Fintype Ω] [DecidableEq Ω] {μ : PMF Ω} {κ ℓ : ℝ} (hκ : 0 ≤ κ)
    {A : Strategy Ω} (hP : DetectionPressure μ κ A) (hL : LeakBounded ℓ A) (Δ n : ℕ) (h : List Ω) (x : ℝ) :
    prob μ n (fun cs => x ≤ A.cumLag Δ n h cs) ≤ Real.exp (-(κ * (x - (1 + Δ) * ℓ))) := by
  calc prob μ n (fun cs => x ≤ A.cumLag Δ n h cs) ≤ prob μ n (fun cs => x - Δ * ℓ ≤ A.cum n h cs) :=
        prob_mono fun cs hx => by linarith [cumLag_le hL Δ n h cs]
    _ ≤ Real.exp (-(κ * (x - Δ * ℓ - ℓ))) := tail_cum hκ hP hL n h _
    _ = Real.exp (-(κ * (x - (1 + Δ) * ℓ))) := by rw [show x - Δ * ℓ - ℓ = x - (1 + Δ) * ℓ by ring]

theorem ex_cumUpToLag [Fintype Ω] [DecidableEq Ω] {μ : PMF Ω} {κ ℓ : ℝ} (hκ : 0 < κ)
    {A : Strategy Ω} (hP : DetectionPressure μ κ A) (hL : LeakBounded ℓ A) (Δ n j : ℕ) (h : List Ω) :
    ex μ n (A.cumUpToLag Δ n j h) ≤ j * (ℓ + 1 / κ) + Δ * ℓ := by
  calc ex μ n (A.cumUpToLag Δ n j h) ≤ ex μ n (fun cs => Δ * ℓ + A.cumUpTo n j h cs) :=
        ex_mono fun cs => by linarith [cumUpToLag_le hL Δ n j h cs]
    _ = Δ * ℓ + ex μ n (A.cumUpTo n j h) := ex_add_const μ _ _
    _ ≤ j * (ℓ + 1 / κ) + Δ * ℓ := by linarith [ex_cumUpTo hκ hP hL n j h]

theorem sample_tail_cumLag {n k : ℕ} (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ}
    (hL : LeakBounded ℓ A) {δ : ℝ} (hδ : 0 < δ) (Δ N : ℕ) (h : List (Verity.Draw.Subset n k)) :
    1 - δ ≤
      prob (uniformSample n k hkn) N (fun cs => A.cumLag Δ N h cs < (1 + Δ) * ℓ + n / k * Real.log (1 / δ)) := by
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  set x := (1 + (Δ : ℝ)) * ℓ + n / k * Real.log (1 / δ)
  have ht := tail_cumLag (by positivity) (samplePressure hkn B w hw hesc hleak) hL Δ N h x
  have e : Real.exp (-((k : ℝ) / n * (x - (1 + Δ) * ℓ))) = δ := by
    rw [show (k : ℝ) / n * (x - (1 + Δ) * ℓ) = Real.log (1 / δ) by simp only [x]; field_simp; ring, one_div,
      Real.log_inv, neg_neg, Real.exp_log hδ]
  rw [prob_congr (E' := fun cs => ¬x ≤ A.cumLag Δ N h cs) fun cs => not_le.symm, prob_not]
  linarith

theorem sample_ex_cumUpToLag {n k : ℕ} (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ}
    (hL : LeakBounded ℓ A) (Δ N j : ℕ) (h : List (Verity.Draw.Subset n k)) :
    ex (uniformSample n k hkn) N (A.cumUpToLag Δ N j h) ≤ j * (ℓ + n / k) + Δ * ℓ := by
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  have := ex_cumUpToLag (by positivity : (0 : ℝ) < k / n) (samplePressure hkn B w hw hesc hleak) hL Δ N j h
  rwa [one_div_div] at this

end Population.Exfil
