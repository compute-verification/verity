module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Definitions.Spot.SamplePressure
public import Definitions.Spot.Exfiltration
import Proofs.Spot.Exfiltration

/-!
# Detection pressure of the deployed sample

The audit samples `k` of the `n` units uniformly without replacement (`Definitions.Spot.SamplePressure`), so a window
whose wrong set has `K` units escapes with probability `C(n - K, k) / C(n, k)`.

* `choose_ratio_le_pow`, `choose_ratio_le_exp`: for `K ≤ n` and `k ≤ n`,
  `C(n - K, k) / C(n, k) ≤ (1 - k/n)^K ≤ exp (-k K / n)`. A sample without replacement escapes no more often than `K`
  independent samples at rate `k/n`.
* `uniformSample_toReal`, `sum_missing_uniformSample`: the uniform sample gives each sample probability `1 / C(n, k)`,
  and it misses a wrong set `B` with probability `C(n - |B|, k) / C(n, k)`.
* `samplePressure`: a window that has the wrong set `B h`, escapes only on the samples that miss it and leaks at most
  `Σ_{u ∈ B h} w u`, each `w u ≤ 1`, is under detection pressure `k/n`.
* `sample_tail_cum`: so through `tail_cum`, with probability at least `1 - δ` what leaves up to and including the first
  detection is below `ℓ + (n/k) ln (1/δ)`, for every adaptive strategy and every horizon.
-/

@[expose] public section

namespace Population.Exfil

open Finset
open Verity.Game (iid)

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

/-- `K` independent samples at rate `k/n` all miss with probability `(1 - k/n)^K ≤ exp (-k K / n)`. -/
theorem one_sub_div_pow_le_exp (hk : k ≤ n) (K : ℕ) : (1 - k / n : ℝ) ^ K ≤ Real.exp (-(k * K / n)) := by
  have := pow_acceptProb_le_exp (cast_div_le_one hk) K
  rwa [show (K : ℝ) * (k / n) = k * K / n by ring] at this

/-- **The deployed sample's escape bound**: for `K ≤ n` and `k ≤ n`, `C(n - K, k) / C(n, k) ≤ exp (-k K / n)`. -/
theorem choose_ratio_le_exp {K : ℕ} (hK : K ≤ n) (hk : k ≤ n) :
    ((n - K).choose k : ℝ) / n.choose k ≤ Real.exp (-(k * K / n)) :=
  (choose_ratio_le_pow hK hk).trans (one_sub_div_pow_le_exp hk K)

/-- The samples that miss `B` are the `k`-subsets of its complement: `C(n - |B|, k)` of them. -/
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

/-- The uniform sample gives each sample probability `1 / C(n, k)`. -/
theorem uniformSample_toReal (hk : k ≤ n) (s : Verity.Draw.Subset n k) :
    (uniformSample n k hk s).toReal = 1 / n.choose k := by
  haveI : Nonempty (Verity.Draw.Subset n k) := ⟨s⟩
  rw [uniformSample, PMF.uniformOfFintype_apply, Fintype.card_finset_len, Fintype.card_fin, ENNReal.toReal_inv,
    ENNReal.toReal_natCast, one_div]

/-- The uniform sample misses the wrong set `B` with probability `C(n - |B|, k) / C(n, k)`. -/
theorem sum_missing_uniformSample (hk : k ≤ n) (B : Finset (Fin n)) :
    Verity.Game.prPMFReal (uniformSample n k hk) (· ∈ missing n k B) = ((n - B.card).choose k : ℝ) / n.choose k := by
  rw [Verity.Game.prPMFReal_mem]
  simp only [uniformSample_toReal, sum_const, card_missing, nsmul_eq_mul, mul_one_div]

/-- **The deployed sample gives detection pressure `k/n`.** At the window after the history `h` the commitment fixes the
wrong set `B h`; the window escapes only on the samples that miss `B h` and leaks at most `Σ_{u ∈ B h} w u`, with each
unit's weight `w u ≤ 1`. Then a window that leaks `ℓ` escapes with probability at most `exp (-(k/n) ℓ)`. -/
theorem samplePressure (hk : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) :
    DetectionPressure (uniformSample n k hk) (k / n) A := by
  intro h
  have hW : ∑ u ∈ B h, w u ≤ (B h).card := by
    simpa using sum_le_card_nsmul (B h) w 1 fun u _ => hw u
  have hB : (B h).card ≤ n := (card_le_univ (B h)).trans_eq (Fintype.card_fin n)
  calc Verity.Game.prPMFReal (uniformSample n k hk) (· ∈ A.esc h)
        ≤ Verity.Game.prPMFReal (uniformSample n k hk) (· ∈ missing n k (B h)) :=
        Verity.Game.prPMFReal_mono _ fun c hc => hesc h hc
    _ = ((n - (B h).card).choose k : ℝ) / n.choose k := sum_missing_uniformSample hk (B h)
    _ ≤ Real.exp (-(k * (B h).card / n)) := choose_ratio_le_exp hB hk
    _ ≤ Real.exp (-(k / n * A.leak h)) := by
      refine Real.exp_le_exp.2 (neg_le_neg ?_)
      calc (k : ℝ) / n * A.leak h ≤ k / n * (B h).card :=
            mul_le_mul_of_nonneg_left ((hleak h).trans hW) (by positivity)
        _ = k * (B h).card / n := by ring

theorem prob_not {Ω : Type} [Fintype Ω] {μ : PMF Ω} (N : ℕ) (E : (Fin N → Ω) → Prop) :
    prob μ N (fun cs => ¬E cs) = 1 - prob μ N E :=
  eq_sub_of_add_eq' (Verity.Game.prPMFReal_add_not (iid μ N) E)

/-- **Harm through the first detection under the deployed sample**: with `0 < k ≤ n`, each window as in `samplePressure`
and leaking at most `ℓ`, with probability at least `1 - δ` what leaves up to and including the first detection is below
`ℓ + (n/k) ln (1/δ)`, for every adaptive strategy, every horizon `N` and every history `h`. -/
theorem sample_tail_cum (hk : 0 < k) (hkn : k ≤ n) {A : Strategy (Verity.Draw.Subset n k)}
    (B : List (Verity.Draw.Subset n k) → Finset (Fin n)) (w : Fin n → ℝ) (hw : ∀ u, w u ≤ 1)
    (hesc : ∀ h, A.esc h ⊆ missing n k (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, w u) {ℓ : ℝ} (hL : LeakBounded ℓ A)
    {δ : ℝ} (hδ : 0 < δ) (N : ℕ) (h : List (Verity.Draw.Subset n k)) :
    1 - δ ≤ prob (uniformSample n k hkn) N (fun cs => A.cum N h cs < ℓ + n / k * Real.log (1 / δ)) := by
  have hkR : (0 : ℝ) < k := by exact_mod_cast hk
  have hnR : (0 : ℝ) < n := by exact_mod_cast hk.trans_le hkn
  set x := ℓ + n / k * Real.log (1 / δ)
  have ht := tail_cum (by positivity) (samplePressure hkn B w hw hesc hleak) hL N h x
  have e : Real.exp (-((k : ℝ) / n * (x - ℓ))) = δ := by
    rw [show (k : ℝ) / n * (x - ℓ) = Real.log (1 / δ) by simp only [x]; field_simp; ring, one_div, Real.log_inv,
      neg_neg, Real.exp_log hδ]
  rw [prob_congr (E' := fun cs => ¬x ≤ A.cum N h cs) fun cs => not_le.symm, prob_not]
  linarith

end Population.Exfil
