module

public import Proofs.Spot.Sampling.HarmBudget
public import Definitions.Core.Scheme
import Proofs.Spot.Sampling.Stratified

/-!
# A rate per unit of amount

An accounting protocol supplies an amount `a u` per proof block; a stratified sampling design whose strata each meet the
integer rate condition `n_s · c_s · NUM ≤ k_s · DEN` (or are sampled whole), `c_s` a cap on the amounts in stratum `s`,
misses every set `B` fixed before the coins with probability at most `exp(−(∑_{u ∈ B} a u) · NUM/DEN)`
(`stratified_design_rate`, in the design condition's form `SamplingDesign.Design`). A scheme under such a design has
missable cost at most `⌊ln(1/δ) · DEN/NUM⌋` (`Scheme.missable_le_rate`), so what it gets back reads the rate
(`Scheme.getsBack_rate`). Over adaptive rounds each meeting the design condition at the same amounts, the cumulative
amount the sample missed reaches `M` with probability at most `exp(−M · NUM/DEN)` plus the rounds' drawn-unit terms
(`Adaptive.rate_budget`).
-/

@[expose] public section

namespace Spot.Sampling

open Finset
open scoped ENNReal

namespace SamplingDesign

variable {n m : ℕ}

/-- **S1, the rate of a stratified sampling design.** With amounts `a`, capped per stratum by `c`, a stratum sampled
whole or meeting `n_s · c_s · NUM ≤ k_s · DEN` in each stratum, every set `B` fixed before the coins escapes with
probability at most `exp(−∑_{u ∈ B} a_u · NUM/DEN)`. -/
theorem stratified_design_rate (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (a : Fin n → ℕ) (c : Fin m → ℕ) (hc : ∀ u, a u ≤ c (σ u)) (num den : ℕ) (hden : 0 < den)
    (hrate : ∀ s, k s = (stratum σ s).card ∨ (stratum σ s).card * c s * num ≤ k s * den) :
    (stratified σ k hk).Design fun u => (a u : ℝ) * num / den := by
  intro B
  by_cases hfull : ∃ u ∈ B, k (σ u) = (stratum σ (σ u)).card
  · obtain ⟨u, hu, hku⟩ := hfull
    have hzero : (stratified σ k hk).missProb B = 0 := by
      rw [stratified_missProb]
      refine Finset.prod_eq_zero (mem_univ (σ u)) ?_
      have hpos : 0 < (B.filter (σ · = σ u)).card := Finset.card_pos.2 ⟨u, Finset.mem_filter.2 ⟨hu, rfl⟩⟩
      have hm : (B.filter (σ · = σ u)).card ≤ (stratum σ (σ u)).card :=
        card_le_card (filter_subset_filter _ (subset_univ B))
      rw [Nat.choose_eq_zero_of_lt (by omega)]
      simp
    rw [hzero]
    exact zero_le
  · simp only [not_exists, not_and] at hfull
    refine (stratified_design σ k hk B).trans (ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 (neg_le_neg ?_)))
    refine Finset.sum_le_sum fun u hu => ?_
    have hr : (stratum σ (σ u)).card * c (σ u) * num ≤ k (σ u) * den :=
      (hrate (σ u)).resolve_left (hfull u hu)
    have hN : 0 < (stratum σ (σ u)).card := Finset.card_pos.2 ⟨u, mem_stratum.2 rfl⟩
    have hle : (stratum σ (σ u)).card * a u * num ≤ k (σ u) * den :=
      le_trans (Nat.mul_le_mul_right num (Nat.mul_le_mul_left _ (hc u))) hr
    have hN' : (0 : ℝ) < (stratum σ (σ u)).card := by exact_mod_cast hN
    have hden' : (0 : ℝ) < den := by exact_mod_cast hden
    rw [div_le_div_iff₀ hden' hN']
    have := (Nat.cast_le (α := ℝ)).2 hle
    push_cast at this
    nlinarith [this]

end SamplingDesign

end Spot.Sampling

namespace Verity.Scheme

open Finset
open scoped ENNReal

/-- **S2, the missable cost at a rate.** Under a sampling design that misses every block set `T` with probability at
most `exp(−∑_{b ∈ T} cost_b · NUM/DEN)` (`SamplingDesign.Design` at those weights, which `stratified_design_rate`
gives), the missable cost at `δ > 0` is at most `⌊ln(1/δ) · DEN/NUM⌋`. -/
theorem missable_le_rate (S : Scheme) (L : SamplingDesign S.Block) (num den : ℕ) (hnum : 0 < num) (hden : 0 < den)
    (hL : ∀ T : Finset S.Block, L.missProb T ≤ ENNReal.ofReal (Real.exp (-∑ b ∈ T, (S.cost b : ℝ) * num / den)))
    {δ : ℝ≥0∞} (hδ : 0 < δ) :
    S.missable L δ ≤ ⌊Real.log δ.toReal⁻¹ * den / num⌋₊ := by
  classical
  unfold Scheme.missable
  refine Finset.sup_le fun T hT => ?_
  have hlt : δ < L.missProb T := (Finset.mem_filter.1 hT).2
  have hlt' := hlt.trans_le (hL T)
  have hδtop : δ ≠ ⊤ := ne_top_of_lt hlt'
  have hδr : 0 < δ.toReal := ENNReal.toReal_pos hδ.ne' hδtop
  have hreal : δ.toReal < Real.exp (-∑ b ∈ T, (S.cost b : ℝ) * num / den) :=
    (ENNReal.toReal_lt_of_lt_ofReal hlt')
  have hlog : Real.log δ.toReal < -∑ b ∈ T, (S.cost b : ℝ) * num / den := by
    rwa [Real.log_lt_iff_lt_exp hδr]
  have hsum : ∑ b ∈ T, (S.cost b : ℝ) * num / den = (∑ b ∈ T, (S.cost b : ℝ)) * num / den := by
    rw [← Finset.sum_div, ← Finset.sum_mul]
  rw [hsum] at hlog
  have hnum' : (0 : ℝ) < num := by exact_mod_cast hnum
  have hden' : (0 : ℝ) < den := by exact_mod_cast hden
  refine Nat.le_floor ?_
  push_cast
  rw [Real.log_inv, le_div_iff₀ hnum']
  have h1 : (∑ b ∈ T, (S.cost b : ℝ)) * num / den < -Real.log δ.toReal := by linarith
  rw [div_lt_iff₀ hden'] at h1
  linarith

/-- **S2, what a scheme gets back at a rate.** Where `S` gets back at cutoff `δ > 0` under designs that each meet the
rate, it gets back with the missable cost read as `⌊ln(1/δ) · DEN/NUM⌋`, for a monotone probability. -/
theorem getsBack_rate (S : Scheme) (L : S.Run → SamplingDesign S.Block) (pr : (S.Run → Prop) → ℝ≥0∞)
    (hmono : ∀ p q : S.Run → Prop, (∀ r, p r → q r) → pr p ≤ pr q) (num den : ℕ) (hnum : 0 < num) (hden : 0 < den)
    (hL : ∀ r (T : Finset S.Block),
      (L r).missProb T ≤ ENNReal.ofReal (Real.exp (-∑ b ∈ T, (S.cost b : ℝ) * num / den)))
    {δ ε : ℝ≥0∞} (hδ : 0 < δ) (h : S.GetsBack L pr δ ε) :
    pr (fun r => S.accepts r ∧ S.logOK r ∧ S.under r + ⌊Real.log δ.toReal⁻¹ * den / num⌋₊ + S.charges < S.over r) ≤
      δ + ε + S.δscheme :=
  (hmono _ _ fun r ⟨ha, hl, hlt⟩ =>
    ⟨ha, hl, lt_of_le_of_lt (by have := missable_le_rate S (L r) num den hnum hden (hL r) hδ; omega) hlt⟩).trans h

end Verity.Scheme

namespace Spot.Sampling.Adaptive

open Finset
open scoped ENNReal

variable {n : ℕ}

/-- **S3, the rate over adaptive rounds.** In a play whose every round's sampling design meets the design condition at
amounts `a` and rate `NUM/DEN` (`stratified_design_rate`), each round's incorrect blocks and design fixed by the history
before its coin, the session accepts with cumulative missed amount at least `M` with probability at most
`exp(−M · NUM/DEN)` plus the rounds' drawn-unit terms. -/
theorem rate_budget (a : Fin n → ℕ) (num den : ℕ) {err : ℕ → ℝ≥0∞} (M : ℝ) {N : ℕ} (P : Play n N)
    (hL : P.SamplingDesigns fun L => L.Design fun u => (a u : ℝ) * num / den) (hD : P.Drawn err) :
    P.prEv (fun u => (a u : ℝ)) (fun acc x => acc = true ∧ M ≤ x) 0 ≤
      ENNReal.ofReal (Real.exp (-((num : ℝ) / den * M))) + ∑ t ∈ range N, err t :=
  harm_budget (by positivity) (fun u => le_of_eq (by ring)) M P hL hD

end Spot.Sampling.Adaptive
