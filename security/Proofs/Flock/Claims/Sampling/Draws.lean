module

public import Definitions.Core.Interaction
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Soundness.Audit.Work
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Game
import Std.Tactic.BVDecide.Normalize.Prop

/-! The registered sampling designs' miss bounds, stated in the claims' `Claims.Pr`. -/

@[expose] public section

namespace Claims.Sampling

open FlockSoundness.Audit FlockSoundness.Game Finset
open scoped ENNReal

theorem Pr_eq_toReal_prCoin {α : Type} [Fintype α] (p : α → Prop) : Claims.Pr p = (prCoin p).toReal := by
  classical
  rw [prCoin_eq_card, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  unfold Claims.Pr Claims.prUniform
  congr 2

theorem Pr_miss_eq {n : ℕ} (L : SamplingDesign n) (B : Finset (Fin n)) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) = (L.missProb B).toReal :=
  Pr_eq_toReal_prCoin _

theorem Pr_miss_le {n : ℕ} (L : SamplingDesign n) (B : Finset (Fin n)) {e : ℝ≥0∞} (he : e ≠ ⊤) (h : L.missProb B ≤ e) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) ≤ e.toReal := by
  rw [Pr_miss_eq]
  exact ENNReal.toReal_mono he h

theorem miss_le_of {p b δ : ℝ} (h : p ≤ b) (hb : b ≤ δ) : p ≤ δ := h.trans hb

theorem subset_miss_eq {n k : ℕ} (hk : k ≤ n) (B : Finset (Fin n)) :
    Claims.Pr (fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B) =
      ((n - B.card).choose k : ℝ) / (n.choose k : ℝ) := by
  have := Pr_miss_eq (SamplingDesign.subset n k hk) B
  rw [SamplingDesign.subset_escape, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast] at this
  exact this

theorem subset_miss_le {n k t : ℕ} (hk : k ≤ n) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun S : {S : Finset (Fin n) // S.card = k} => Disjoint S.1 B) ≤
      ((n - t).choose k : ℝ) / (n.choose k : ℝ) := by
  rw [subset_miss_eq hk]
  gcongr

theorem bernFactor_zero (num den : ℕ) [NeZero den] :
    (SamplingDesign.bernFactor num den 0).toReal = ((den - num : ℕ) : ℝ) / den := by
  unfold SamplingDesign.bernFactor
  rw [mul_zero, zero_add, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem bernoulli_miss_eq {n num den : ℕ} [NeZero den] (h : num ≤ den) (B : Finset (Fin n)) :
    Claims.Pr (fun ω : Fin n → Fin den => Disjoint (univ.filter fun u => (ω u).val < num) B) =
      (((den - num : ℕ) : ℝ) / den) ^ B.card := by
  have := Pr_miss_eq (SamplingDesign.bernoulli n num den) B
  rw [SamplingDesign.bernoulli_escape h, ENNReal.toReal_pow, bernFactor_zero num den] at this
  exact this

theorem bernoulli_miss_le {n num den t : ℕ} [NeZero den] (h : num ≤ den) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : Fin n → Fin den => Disjoint (univ.filter fun u => (ω u).val < num) B) ≤
      (((den - num : ℕ) : ℝ) / den) ^ t := by
  rw [bernoulli_miss_eq h]
  have hd : (0 : ℝ) < den := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne den)
  refine pow_le_pow_of_le_one (by positivity) ?_ ht
  rw [div_le_one hd]
  exact_mod_cast Nat.sub_le den num

theorem stratified_miss_eq {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (SamplingDesign.stratum σ s).card)
    (B : Finset (Fin n)) :
    Claims.Pr (fun ω : (SamplingDesign.stratified σ k hk).Ω => Disjoint ((SamplingDesign.stratified σ k hk).draw ω) B) =
      ∏ s, (((SamplingDesign.stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ) /
        ((SamplingDesign.stratum σ s).card.choose (k s) : ℝ) := by
  rw [Pr_miss_eq, SamplingDesign.stratified_escape, ENNReal.toReal_prod]
  refine Finset.prod_congr rfl fun s _ => ?_
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem choose_ratio_le_pow {N k b : ℕ} {π : ℝ} (hk : k ≤ N) (hb : b ≤ N)
    (hπ : 0 < N → π * N ≤ k) : ((N - b).choose k : ℝ) / (N.choose k : ℝ) ≤ (1 - π) ^ b := by
  rcases Nat.eq_zero_or_pos N with h0 | hN
  · obtain rfl : b = 0 := by omega
    obtain rfl : k = 0 := by omega
    simp [h0]
  have hC : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have key := SamplingDesign.choose_sub_mul_pow_le N k b hb
  have h1 : ((N - b).choose k : ℝ) / (N.choose k : ℝ) ≤ (((N - k : ℕ) : ℝ) / N) ^ b := by
    rw [div_pow, div_le_div_iff₀ hC (by positivity)]
    exact_mod_cast key.trans_eq (mul_comm _ _)
  refine h1.trans (pow_le_pow_left₀ (by positivity) ?_ b)
  rw [div_le_iff₀ hNr, Nat.cast_sub hk]
  nlinarith [hπ hN]

theorem stratified_miss_le {n m t : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (SamplingDesign.stratum σ s).card)
    {π : ℝ} (hπ0 : 0 ≤ π) (hπ1 : π ≤ 1) (hπ : ∀ s, 0 < (SamplingDesign.stratum σ s).card → π * (SamplingDesign.stratum σ s).card ≤ k s)
    {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : (SamplingDesign.stratified σ k hk).Ω => Disjoint ((SamplingDesign.stratified σ k hk).draw ω) B) ≤ (1 - π) ^ t := by
  classical
  rw [stratified_miss_eq]
  have hb : ∀ s, (B.filter (σ · = s)).card ≤ (SamplingDesign.stratum σ s).card := fun s =>
    card_le_card (filter_subset_filter _ (subset_univ B))
  have hsum : ∑ s, (B.filter (σ · = s)).card = B.card := by
    rw [← card_biUnion]
    · congr 1
      ext u
      simp
    · intro s _ s' _ hss'
      exact disjoint_filter.2 fun u _ h h' => hss' (h.symm.trans h')
  calc ∏ s, (((SamplingDesign.stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ) /
        ((SamplingDesign.stratum σ s).card.choose (k s) : ℝ)
      ≤ ∏ s, (1 - π) ^ (B.filter (σ · = s)).card :=
        prod_le_prod₀ (fun s _ => by positivity) fun s _ => choose_ratio_le_pow (hk s) (hb s) (hπ s)
    _ = (1 - π) ^ B.card := by rw [prod_pow_eq_pow_sum, hsum]
    _ ≤ (1 - π) ^ t := pow_le_pow_of_le_one (by linarith) (by linarith) ht

theorem work_miss_le {n m : ℕ} (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < SamplingDesign.totalWork σ w) {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ SamplingDesign.workOf σ w B) :
    Claims.Pr (fun ω : (SamplingDesign.work σ w f K).Ω => Disjoint ((SamplingDesign.work σ w f K).draw ω) B) ≤
      (((SamplingDesign.totalWork σ w - T : ℕ) : ℝ) / (SamplingDesign.totalWork σ w : ℝ)) ^ K := by
  have hW0 : (SamplingDesign.totalWork σ w : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  refine (Pr_miss_le _ B (ENNReal.pow_ne_top (ENNReal.div_ne_top (ENNReal.natCast_ne_top _) hW0))
    (SamplingDesign.work_escape_le_of_le σ w f K hf hW hT)).trans_eq ?_
  rw [ENNReal.toReal_pow, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem miss_le_count {n t : ℕ} (L : SamplingDesign n) {B : Finset (Fin n)} (ht : t ≤ B.card) :
    Claims.Pr (fun ω : L.Ω => Disjoint (L.draw ω) B) ≤ (L.miss t).toReal :=
  Pr_miss_le L B (ne_top_of_le_ne_top ENNReal.one_ne_top (L.miss_le_one t)) (L.escape_le_miss ht)

end Claims.Sampling
