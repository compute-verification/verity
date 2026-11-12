module

public import Definitions.NetworkCertifier.Guarantees.Pooled
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Soundness.Audit.Periods
import Proofs.Flock.Soundness.Game
import Proofs.NetworkCertifier.PeriodSum
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The incorrect proof blocks over a monitoring period: the escapes multiply

A uniform `k`-subset of `n` blocks misses a set of `f` blocks with probability `C(n - f, k) / C(n, k)`, at most
`(1 - k/n)^f` (`choose_sub_le`, `pr_disjoint_le`). Over windows on independent coins, each with its incorrect
set fixed before its own sample, every sample missing its window's set while the sets hold `F` blocks in total has
probability at most `r^F` when each window's miss has probability at most `r` to the power of its set's size
(`pr_adaptive_le`, the monitoring-period step `prCoin_forall_sum_le_pow` read in `Pr`). With
each window's soundness error added by a union bound, that is `pooledIncorrectBlocks`.
-/

@[expose] public section

namespace NetTiming

open Claims Finset

/-! ### The union bound in `Pr` -/

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  unfold Pr Claims.prUniform
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  exact_mod_cast (Finset.card_le_card fun a ha => by
    simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at ha ⊢; exact ha).trans
    (Finset.card_union_le _ _)

/-! ### A uniform sample's miss -/

theorem choose_pred_le (m k n : ℕ) (hm : m ≤ n) (hk : k ≤ n) :
    ((m - 1).choose k : ℝ) ≤ m.choose k * (1 - k / n) := by
  rcases m with _ | m
  · rcases k with _ | k
    · simp
    · simp
  have hn : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have h := Nat.choose_mul_succ_eq m k
  have hpos : (0 : ℝ) < m + 1 := by positivity
  have hkn : (k : ℝ) / n ≤ 1 := (div_le_one hn).2 (by exact_mod_cast hk)
  have hsub : (((m + 1 - k : ℕ) : ℝ)) ≤ (m + 1) * (1 - k / n) := by
    rcases le_or_gt k (m + 1) with hkm | hkm
    · rw [Nat.cast_sub hkm]
      push_cast
      have h2 : (m + 1 : ℝ) * (1 - k / n) = (m + 1 - k) + k * (1 - (m + 1) / n) := by ring
      have h3 : ((m : ℝ) + 1) / n ≤ 1 := (div_le_one hn).2 (by exact_mod_cast hm)
      have h4 : 0 ≤ (k : ℝ) * (1 - (m + 1) / n) := mul_nonneg (by positivity) (by linarith)
      linarith
    · rw [Nat.sub_eq_zero_of_le hkm.le]
      push_cast
      exact mul_nonneg hpos.le (by linarith)
  have h' : ((m.choose k : ℕ) : ℝ) * (m + 1) = ((m + 1).choose k : ℕ) * ((m + 1 - k : ℕ) : ℝ) := by
    exact_mod_cast h
  simp only [Nat.add_sub_cancel]
  have hc : (0 : ℝ) ≤ ((m + 1).choose k : ℕ) := by positivity
  have := mul_le_mul_of_nonneg_left hsub hc
  nlinarith [h', this, hpos]

theorem choose_sub_le (n k : ℕ) (hk : k ≤ n) (f : ℕ) :
    ((n - f).choose k : ℝ) ≤ n.choose k * (1 - k / n) ^ f := by
  induction f with
  | zero => simp
  | succ f ih =>
    have h0 : 0 ≤ 1 - (k : ℝ) / n := by
      rcases Nat.eq_zero_or_pos n with hn | hn
      · subst hn; simp
      · have : (k : ℝ) / n ≤ 1 := (div_le_one (by exact_mod_cast hn)).2 (by exact_mod_cast hk)
        linarith
    have e : n - (f + 1) = n - f - 1 := by omega
    rw [e, pow_succ, ← mul_assoc]
    exact (choose_pred_le (n - f) k n (Nat.sub_le n f) hk).trans (mul_le_mul_of_nonneg_right ih h0)

theorem pr_disjoint_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {n : ℕ} (sample : Ω → Finset (Fin n)) (k : ℕ)
    (hcard : ∀ x, (sample x).card = k)
    (hfib : ∀ S : Finset (Fin n), S.card = k → (univ.filter fun x => sample x = S).card * n.choose k = Fintype.card Ω)
    (W : Finset (Fin n)) {q : ℝ} (hq : q * n ≤ k) :
    Pr (fun x => Disjoint (sample x) W) ≤ (1 - q) ^ W.card := by
  classical
  have hΩ : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hk : k ≤ n := by
    have := Finset.card_le_univ (sample (Classical.arbitrary Ω))
    rwa [hcard, Fintype.card_fin] at this
  have hC : (0 : ℝ) < n.choose k := by exact_mod_cast Nat.choose_pos hk
  have hsum : (univ.filter fun x => Disjoint (sample x) W).card =
      ∑ S ∈ powersetCard k (univ \ W), (univ.filter fun x => sample x = S).card := by
    rw [Finset.card_eq_sum_card_fiberwise (f := sample) (t := powersetCard k (univ \ W))]
    · refine Finset.sum_congr rfl fun S hS => ?_
      rw [Finset.filter_filter]
      congr 1
      refine Finset.filter_congr fun x _ => ⟨fun h => h.2, fun h => ⟨?_, h⟩⟩
      rw [h]
      obtain ⟨hsub, -⟩ := Finset.mem_powersetCard.1 hS
      exact Finset.disjoint_left.2 fun u hu huW => (Finset.mem_sdiff.1 (hsub hu)).2 huW
    · intro x hx
      rw [Finset.mem_coe, Finset.mem_filter] at hx
      rw [Finset.mem_coe, Finset.mem_powersetCard]
      exact ⟨fun u hu => Finset.mem_sdiff.2 ⟨Finset.mem_univ u, Finset.disjoint_left.1 hx.2 hu⟩, hcard x⟩
  have hnat : (univ.filter fun x => Disjoint (sample x) W).card * n.choose k =
      (n - W.card).choose k * Fintype.card Ω := by
    rw [hsum, Finset.sum_mul, Finset.sum_congr rfl fun S hS => hfib S (Finset.mem_powersetCard.1 hS).2,
      Finset.sum_const, Finset.card_powersetCard, Finset.card_sdiff_of_subset (Finset.subset_univ W), Finset.card_univ,
      Fintype.card_fin, smul_eq_mul]
  have hpr : Pr (fun x => Disjoint (sample x) W) = ((n - W.card).choose k : ℝ) / n.choose k := by
    rw [Pr_eq_sum, Finset.sum_boole, div_eq_div_iff hΩ.ne' hC.ne']
    exact_mod_cast hnat
  rw [hpr, div_le_iff₀ hC]
  refine (choose_sub_le n k hk W.card).trans ?_
  rw [mul_comm]
  refine mul_le_mul_of_nonneg_right ?_ hC.le
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    have : W.card = 0 := by simpa using Finset.card_le_univ W
    rw [this, pow_zero, pow_zero]
  · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hkn : (k : ℝ) / n ≤ 1 := (div_le_one hn').2 (by exact_mod_cast hk)
    have hqk : q ≤ k / n := (le_div_iff₀ hn').2 hq
    exact pow_le_pow_left₀ (by linarith) (by linarith) _

/-- The sample's hypotheses are satisfiable: the coins that are the `k`-subsets themselves. -/
theorem sampleUniform_val (n k : ℕ) :
    (∀ x : {S : Finset (Fin n) // S.card = k}, x.val.card = k) ∧
      ∀ S : Finset (Fin n), S.card = k →
        (univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.val = S).card * n.choose k =
          Fintype.card {S : Finset (Fin n) // S.card = k} := by
  classical
  refine ⟨fun S => S.2, fun S hS => ?_⟩
  have h1 : (univ.filter fun x : {S : Finset (Fin n) // S.card = k} => x.val = S).card = 1 := by
    rw [Finset.card_eq_one]
    exact ⟨⟨S, hS⟩, by ext x; simp [Subtype.ext_iff]⟩
  rw [h1, one_mul, Fintype.card_finset_len, Fintype.card_fin]

/-! ### The escapes multiply over windows on independent coins -/

open FlockSoundness.Game in
/-- **The escapes multiply**: execution phases whose audits, given the earlier phases' coins, accept with probability
at most `r^(f i)`, `f i` fixed before phase `i`'s coins, all accept with `Σ_i f i ≥ F` with probability at most `r^F`:
the monitoring-period step at predictable weights. -/
theorem pr_adaptive_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    ∀ (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ),
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω') →
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') →
      (∀ i ω, Pr (A i ω) ≤ r ^ f i ω) → ∀ F : ℕ,
      Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F := by
  intro m A f hA hf hesc F
  have hle1 : ∀ p : Ω → Prop, prCoin p ≤ 1 := fun p =>
    (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const])
  have hstep : ∀ (t : Fin m) (ω : Fin m → Ω), prCoin (fun a : Ω => A t (Function.update ω t a)
      (Function.update ω t a t)) ≤ ENNReal.ofReal r ^ f t ω := fun t ω => by
    have hupd : (fun a : Ω => A t (Function.update ω t a) (Function.update ω t a t)) = A t ω := funext fun a => by
      rw [Function.update_self, hA t (Function.update ω t a) ω fun j hj => Function.update_of_ne (ne_of_lt hj) _ _]
    rw [hupd, ← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (hle1 _)),
      ← Claims.Sampling.Pr_eq_toReal_prCoin, ← ENNReal.ofReal_pow hr0]
    exact ENNReal.ofReal_le_ofReal (hesc t ω)
  have key := FlockSoundness.Audit.Periods.prCoin_forall_sum_le_pow (Ω := fun _ : Fin m => Ω)
    (fun t ω => A t ω (ω t)) f
    (fun t ω ω' h => by
      show A t ω (ω t) = A t ω' (ω' t)
      rw [hA t ω ω' fun j hj => h j hj.le, h t le_rfl])
    (fun t ω ω' h => hf t ω ω' h) (ENNReal.ofReal_le_one.2 hr1) hstep F
  rw [Claims.Sampling.Pr_eq_toReal_prCoin]
  refine (ENNReal.toReal_mono (ENNReal.pow_ne_top ENNReal.ofReal_ne_top) key).trans_eq ?_
  rw [ENNReal.toReal_pow, ENNReal.toReal_ofReal hr0]

/-! ### The monitoring period -/

/-- **The incorrect proof blocks over a monitoring period**: `Guarantees.PooledIncorrectBlocksLe`. -/
theorem pooledIncorrectBlocks : Guarantees.PooledIncorrectBlocksLe := by
  intro m Ω _ _ n k sample hcard hunif W hW acc ε hsound q hq0 hq1 hq F
  refine ⟨?_, fun δ hδ hlog => ?_⟩
  · classical
    have hmiss := pr_adaptive_le (Ω := Ω) (r := 1 - q) (by linarith) (by linarith) m
      (fun i ω x => Disjoint (sample i x) (W i ω)) (fun i ω => (W i ω).card)
      (fun i ω ω' h => by simp only [hW i ω ω' h]) (fun i ω ω' h => by simp only [hW i ω ω' h])
      (fun i ω => pr_disjoint_le (sample i) (k i) (hcard i) (hunif i) (W i ω) (hq i)) F
    calc Pr (fun ω => (∀ i, acc i ω) ∧ F ≤ ∑ i, (W i ω).card)
        ≤ Pr (fun ω => ((∀ i, Disjoint (sample i (ω i)) (W i ω)) ∧ F ≤ ∑ i, (W i ω).card) ∨
            ∃ i ∈ (univ : Finset (Fin m)), acc i ω ∧ ¬Disjoint (sample i (ω i)) (W i ω)) := by
          refine Pr_mono fun ω h => ?_
          obtain ⟨hacc, hF⟩ := h
          by_cases hd : ∀ i, Disjoint (sample i (ω i)) (W i ω)
          · exact Or.inl ⟨hd, hF⟩
          · push Not at hd
            obtain ⟨i, hi⟩ := hd
            exact Or.inr ⟨i, Finset.mem_univ i, hacc i, hi⟩
      _ ≤ _ := Pr_or_le _ _
      _ ≤ (1 - q) ^ F + ∑ i, ε i :=
          add_le_add hmiss ((Pr_exists_mem_le_sum univ _).trans (Finset.sum_le_sum fun i _ => hsound i))
  · have h1 : 1 - q ≤ Real.exp (-q) := by linarith [Real.add_one_le_exp (-q)]
    calc (1 - q) ^ F ≤ Real.exp (-q) ^ F := pow_le_pow_left₀ (by linarith) h1 F
      _ = Real.exp (F * (-q)) := (Real.exp_nat_mul _ _).symm
      _ ≤ Real.exp (-Real.log (1 / δ)) := Real.exp_le_exp.2 (by nlinarith)
      _ = δ := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

end NetTiming
