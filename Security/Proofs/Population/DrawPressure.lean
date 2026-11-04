import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Definitions.Population.DrawPressure
import Proofs.Population.Exfiltration

/-!
# Detection pressure of the deployed draw

The audit draws `k` of the `n` units uniformly without replacement (`Definitions.Population.DrawPressure`), so a window
whose wrong set has `K` units escapes with probability `C(n - K, k) / C(n, k)`.

* `choose_ratio_le_pow`, `choose_ratio_le_exp`: for `K ≤ n` and `k ≤ n`,
  `C(n - K, k) / C(n, k) ≤ (1 - k/n)^K ≤ exp (-k K / n)`. A draw without replacement escapes no more often than `K`
  independent draws at rate `k/n`.
* `uniformDraw_isDist`, `sum_missing_uniformDraw`: the uniform draw is a distribution, and it misses a wrong set `B`
  with probability `C(n - |B|, k) / C(n, k)`.
* `drawPressure`: a window that has the wrong set `B h`, escapes only on the draws that miss it and leaks at most
  `Σ_{u ∈ B h} w u`, each `w u ≤ 1`, is under detection pressure `k/n`.
* `draw_tail_cum`: so through `tail_cum`, with probability at least `1 - δ` what leaves up to and including the first
  detection is below `ℓ + (n/k) ln (1/δ)`, for every adaptive strategy and every horizon.
-/

namespace Population.Exfil

open Finset

variable {n k : ℕ}

theorem cast_div_le_one (hk : k ≤ n) : (k : ℝ) / n ≤ 1 :=
  div_le_one_of_le₀ (by exact_mod_cast hk) (Nat.cast_nonneg n)

/-- **Without replacement escapes no more often than independently**: for `K ≤ n` and `k ≤ n`,
`C(n - K, k) / C(n, k) ≤ (1 - k/n)^K`. -/
theorem choose_ratio_le_pow {K : ℕ} (hK : K ≤ n) (hk : k ≤ n) :
    ((n - K).choose k : ℝ) / n.choose k ≤ (1 - k / n) ^ K := by
  have hC : (0 : ℝ) < n.choose k := by exact_mod_cast Nat.choose_pos hk
  have hd := cast_div_le_one hk
  induction K with
  | zero => rw [Nat.sub_zero, pow_zero, div_self hC.ne']
  | succ K ih =>
    obtain ⟨a, ha⟩ : ∃ a, n - K = a + 1 := ⟨n - (K + 1), by omega⟩
    rw [show n - (K + 1) = a by omega, pow_succ]
    have han : (a : ℝ) + 1 ≤ n := by exact_mod_cast (show a + 1 ≤ n by omega)
    have hc : (a.choose k : ℝ) * (a + 1) = (a + 1).choose k * ((a + 1 - k : ℕ) : ℝ) := by
      exact_mod_cast Nat.choose_mul_succ_eq a k
    set r : ℝ := ((a + 1 - k : ℕ) : ℝ) / (a + 1)
    have hr0 : 0 ≤ r := by positivity
    have hr : r ≤ 1 - k / n := by
      rcases le_or_gt k (a + 1) with hka | hka
      · have hkR : (k : ℝ) / n ≤ k / (a + 1) := div_le_div_of_nonneg_left (Nat.cast_nonneg k) (by positivity) han
        have : r = 1 - k / (a + 1) := by
          simp only [r]
          rw [Nat.cast_sub hka]
          field_simp
          push_cast
          ring
        linarith
      · simp only [r, Nat.sub_eq_zero_of_le hka.le, Nat.cast_zero, zero_div]
        linarith
    have e : (a.choose k : ℝ) / n.choose k = (a + 1).choose k / n.choose k * r := by
      simp only [r]
      field_simp
      linarith
    calc (a.choose k : ℝ) / n.choose k = (a + 1).choose k / n.choose k * r := e
      _ ≤ (1 - k / n) ^ K * r := mul_le_mul_of_nonneg_right (ha ▸ ih (by omega)) hr0
      _ ≤ (1 - k / n) ^ K * (1 - k / n) := mul_le_mul_of_nonneg_left hr (pow_nonneg (by linarith) K)

/-- `K` independent draws at rate `k/n` all miss with probability `(1 - k/n)^K ≤ exp (-k K / n)`. -/
theorem one_sub_div_pow_le_exp (hk : k ≤ n) (K : ℕ) : (1 - k / n : ℝ) ^ K ≤ Real.exp (-(k * K / n)) := by
  have := pow_escape_le_exp (cast_div_le_one hk) K
  rwa [show (K : ℝ) * (k / n) = k * K / n by ring] at this

/-- **The deployed draw's escape bound**: for `K ≤ n` and `k ≤ n`, `C(n - K, k) / C(n, k) ≤ exp (-k K / n)`. -/
theorem choose_ratio_le_exp {K : ℕ} (hK : K ≤ n) (hk : k ≤ n) :
    ((n - K).choose k : ℝ) / n.choose k ≤ Real.exp (-(k * K / n)) :=
  (choose_ratio_le_pow hK hk).trans (one_sub_div_pow_le_exp hk K)

/-- The draws that miss `B` are the `k`-subsets of its complement: `C(n - |B|, k)` of them. -/
theorem card_missing (B : Finset (Fin n)) : (missing n k B).card = (n - B.card).choose k := by
  have e : (missing n k B).map (Function.Embedding.subtype _) = powersetCard k Bᶜ := by
    ext t
    simp only [missing, mem_map, mem_filter, mem_univ, true_and, Function.Embedding.coe_subtype, mem_powersetCard,
      subset_compl_iff_disjoint_right]
    constructor
    · rintro ⟨s, hs, rfl⟩
      exact ⟨hs, s.2⟩
    · rintro ⟨hd, hc⟩
      exact ⟨⟨t, hc⟩, hd, rfl⟩
  rw [← card_map (Function.Embedding.subtype _), e, card_powersetCard, card_compl, Fintype.card_fin]

/-- The uniform draw misses the wrong set `B` with probability `C(n - |B|, k) / C(n, k)`. -/
theorem sum_missing_uniformDraw (B : Finset (Fin n)) :
    ∑ s ∈ missing n k B, uniformDraw n k s = ((n - B.card).choose k : ℝ) / n.choose k := by
  simp only [uniformDraw, sum_const, card_missing, nsmul_eq_mul, mul_one_div]

theorem uniformDraw_nonneg (s : Verity.Draw.Subset n k) : 0 ≤ uniformDraw n k s := by
  unfold uniformDraw
  positivity

/-- The uniform draw is a distribution when there are `k` units to draw. -/
theorem uniformDraw_isDist (hk : k ≤ n) : IsDist (uniformDraw n k) := by
  have hC : (n.choose k : ℝ) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
  refine ⟨uniformDraw_nonneg, ?_⟩
  simp only [uniformDraw, sum_const, card_univ, nsmul_eq_mul]
  rw [Fintype.card_finset_len, Fintype.card_fin, mul_one_div, div_self hC]

/-- **The deployed draw gives detection pressure `k/n`.** At the window after the history `h` the commitment fixes the
wrong set `B h`; the window escapes only on the draws that miss `B h` and leaks at most `Σ_{u ∈ B h} w u`, with each
unit's weight `w u ≤ 1`. Then a window that leaks `ℓ` escapes with probability at most `exp (-(k/n) ℓ)`. With `k > n`
there is no draw, and the bound holds vacuously. -/
theorem drawPressure {A : Strategy (Verity.Draw.Subset n k)} (B : List (Verity.Draw.Subset n k) → Finset (Fin n))
    (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1) (hesc : ∀ h, A.esc h ⊆ missing n k (B h))
    (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) :
    DetectionPressure (uniformDraw n k) (k / n) A := by
  intro h
  rcases le_or_gt k n with hk | hk
  · have hW : ∑ u ∈ B h, w u ≤ (B h).card := by
      simpa using sum_le_card_nsmul (B h) w 1 fun u _ => hw u
    have hB : (B h).card ≤ n := (card_le_univ (B h)).trans_eq (Fintype.card_fin n)
    calc ∑ c ∈ A.esc h, uniformDraw n k c ≤ ∑ c ∈ missing n k (B h), uniformDraw n k c :=
          sum_le_sum_of_subset_of_nonneg (hesc h) fun c _ _ => uniformDraw_nonneg c
      _ = ((n - (B h).card).choose k : ℝ) / n.choose k := sum_missing_uniformDraw (B h)
      _ ≤ Real.exp (-(k * (B h).card / n)) := choose_ratio_le_exp hB hk
      _ ≤ Real.exp (-(k / n * A.leak h)) := by
        refine Real.exp_le_exp.2 (neg_le_neg ?_)
        calc (k : ℝ) / n * A.leak h ≤ k / n * (B h).card :=
              mul_le_mul_of_nonneg_left ((hleak h).trans hW) (by positivity)
          _ = k * (B h).card / n := by ring
  · have h0 : ∀ s, uniformDraw n k s = 0 := fun _ => by simp [uniformDraw, Nat.choose_eq_zero_of_lt hk]
    simp only [h0, sum_const_zero]
    exact (Real.exp_pos _).le

theorem prob_not {Ω : Type} [Fintype Ω] {μ : Ω → ℝ} (hμ : IsDist μ) (N : ℕ) (E : (Fin N → Ω) → Prop) :
    prob μ N (fun cs => ¬E cs) = 1 - prob μ N E := by
  have h : prob μ N E + prob μ N (fun cs => ¬E cs) = ex μ N fun _ => 1 := by
    simp only [prob, ex, ← sum_add_distrib, ← mul_add]
    refine sum_congr rfl fun cs _ => ?_
    split_ifs <;> simp_all
  rw [ex_const hμ] at h
  linarith

/-- **Harm through the first detection under the deployed draw**: with `0 < k ≤ n`, each window as in `drawPressure`
and leaking at most `ℓ`, with probability at least `1 - δ` what leaves up to and including the first detection is below
`ℓ + (n/k) ln (1/δ)`, for every adaptive strategy, every horizon `N` and every history `h`. -/
theorem draw_tail_cum (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ} (hL : LeakBounded ℓ A)
    {δ : ℝ} (hδ : 0 < δ) (N : ℕ) (h : List (Verity.Draw.Subset n k)) :
    1 - δ ≤ prob (uniformDraw n k) N (fun cs => A.cum N h cs < ℓ + n / k * Real.log (1 / δ)) := by
  have hμ := uniformDraw_isDist hkn
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  set x := ℓ + n / k * Real.log (1 / δ)
  have ht := tail_cum hμ (by positivity) (drawPressure B w hw hesc hleak) hL N h x
  have e : Real.exp (-((k : ℝ) / n * (x - ℓ))) = δ := by
    rw [show (k : ℝ) / n * (x - ℓ) = Real.log (1 / δ) by simp only [x]; field_simp; ring, one_div, Real.log_inv,
      neg_neg, Real.exp_log hδ]
  rw [prob_congr (E' := fun cs => ¬x ≤ A.cum N h cs) fun cs => not_le.symm, prob_not hμ]
  linarith

end Population.Exfil
