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
import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Soundness.Audit.PeriodsCharge
import Proofs.NetworkCertifier.PeriodSum
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# The incorrect proof blocks over a monitoring period: the escapes multiply

A uniform `k`-subset of `n` units misses a set of `f` units with probability `C(n - f, k) / C(n, k)`, at most
`(1 - k/n)^f` (`choose_sub_le`, `pr_disjoint_le`). Over windows on independent coins, each with its wrong
set fixed before its own sample, every sample missing its window's set while the sets hold `F` units in total has
probability at most `r^F` when each window's miss has probability at most `r` to the power of its set's size
(`pr_adaptive_le`). With each window's soundness error added by a union bound, that is `pooledIncorrectBlocks`.

The area has one step over a monitoring period's execution phases, `pr_adaptive_add_ex_le`: the monitoring-period
step with a charge (`FlockSoundness.Audit.Periods.ex_prod_add_le`) at the phases' accepting events, each phase's
bound an escape `r^(f i)`, an error `e i` fixed before the period and the average over the phase's own coins of a
charge. The errors add, and the period pays the charges once, at the average over its coins of their sum.
`pr_adaptive_add_le` is that step at the charge zero, and `pr_adaptive_le` at the charge and the errors zero.
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

/-! ### The escapes multiply over execution phases on independent coins -/

/-- Bounds capped at `1`, each an escape `a i ≤ 1` plus an error `e i`, multiply to at most the escapes' product plus
the errors' sum. -/
theorem prod_min_add_le : ∀ {m : ℕ} (a e : Fin m → ℝ), (∀ i, 0 ≤ a i) → (∀ i, a i ≤ 1) → (∀ i, 0 ≤ e i) →
    ∏ i, min 1 (a i + e i) ≤ ∏ i, a i + ∑ i, e i
  | 0, _, _, _, _, _ => by simp
  | m + 1, a, e, ha0, ha1, he => by
    have ih : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ ∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ :=
      prod_min_add_le _ _ (fun i => ha0 _) (fun i => ha1 _) (fun i => he _)
    have hm0 : ∀ i : Fin m, 0 ≤ min 1 (a i.succ + e i.succ) := fun i => le_min zero_le_one (add_nonneg (ha0 _) (he _))
    have hX0 : 0 ≤ ∏ i : Fin m, min 1 (a i.succ + e i.succ) := Finset.prod_nonneg fun i _ => hm0 i
    have hX1 : ∏ i : Fin m, min 1 (a i.succ + e i.succ) ≤ 1 :=
      Finset.prod_le_one₀ (fun i _ => hm0 i) fun i _ => min_le_left _ _
    have hE0 : 0 ≤ ∑ i : Fin m, e i.succ := Finset.sum_nonneg fun i _ => he _
    rw [Fin.prod_univ_succ, Fin.prod_univ_succ, Fin.sum_univ_succ]
    calc min 1 (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ)
        ≤ (a 0 + e 0) * ∏ i : Fin m, min 1 (a i.succ + e i.succ) := mul_le_mul_of_nonneg_right (min_le_right _ _) hX0
      _ = a 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) + e 0 * ∏ i : Fin m, min 1 (a i.succ + e i.succ) :=
          add_mul _ _ _
      _ ≤ a 0 * (∏ i : Fin m, a i.succ + ∑ i : Fin m, e i.succ) + e 0 * 1 :=
          add_le_add (mul_le_mul_of_nonneg_left ih (ha0 0)) (mul_le_mul_of_nonneg_left hX1 (he 0))
      _ ≤ a 0 * ∏ i : Fin m, a i.succ + (e 0 + ∑ i : Fin m, e i.succ) := by
          rw [mul_add, mul_one]
          linarith [mul_le_of_le_one_left hE0 (ha1 0)]

open FlockSoundness.Audit.Periods in
/-- **The escapes multiply, with errors that add and a charge paid once**: execution phases whose audits, given the
earlier phases' coins `ω`, accept with probability at most `r^(f i ω) + e i` plus the average over the phase's own
coins `x` of a charge `w i ω x` that is not negative, `f i` and `w i` fixed before phase `i`'s coins, all accept with
`Σ_i f i ≥ F` with probability at most `r^F + Σ_i e i` plus the average, over the period's coins, of the charges'
sum. It is the monitoring-period step with a charge (`ex_prod_add_le`) at the accepting events' indicators, each
phase's bound capped at `1`. -/
theorem pr_adaptive_add_ex_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ) (e : Fin m → ℝ)
    (w : Fin m → (Fin m → Ω) → Ω → ℝ)
    (hA : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω')
    (hf : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω')
    (hw : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → w i ω = w i ω') (he : ∀ i, 0 ≤ e i)
    (hw0 : ∀ i ω x, 0 ≤ w i ω x) (hesc : ∀ i ω, Pr (A i ω) ≤ r ^ f i ω + e i + Ex (w i ω)) (F : ℕ) :
    Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F + ∑ i, e i + Ex fun ω => ∑ i, w i ω (ω i) := by
  classical
  have hind0 : ∀ (t : Fin m) (ω : Fin m → Ω), 0 ≤ (if A t ω (ω t) then (1 : ℝ) else 0) := fun t ω => by
    split_ifs
    · exact zero_le_one
    · exact le_rfl
  have hind1 : ∀ (t : Fin m) (ω : Fin m → Ω), (if A t ω (ω t) then (1 : ℝ) else 0) ≤ 1 := fun t ω => by
    split_ifs
    · exact le_rfl
    · exact zero_le_one
  have key := ex_prod_add_le (Ω := fun _ : Fin m => Ω)
    (fun t ω => if A t ω (ω t) then (1 : ℝ) else 0) (fun t ω => min 1 (r ^ f t ω + e t))
    (fun t ω => w t ω (ω t))
    (fun t ω ω' h => by
      show (if A t ω (ω t) then (1 : ℝ) else 0) = if A t ω' (ω' t) then (1 : ℝ) else 0
      rw [hA t ω ω' fun j hj => h j hj.le, h t le_rfl])
    (fun t ω ω' h => by simp only [hf t ω ω' h])
    (fun t ω ω' h => by
      show w t ω (ω t) = w t ω' (ω' t)
      rw [hw t ω ω' fun j hj => h j hj.le, h t le_rfl])
    hind0 hind1 (fun t ω => le_min zero_le_one (add_nonneg (pow_nonneg hr0 _) (he t))) (fun t ω => hw0 _ _ _)
    (fun t ω => by
      have hupd : ∀ a : Ω, A t (Function.update ω t a) (Function.update ω t a t) = A t ω a := fun a => by
        rw [Function.update_self,
          hA t (Function.update ω t a) ω fun j hj => Function.update_of_ne (ne_of_lt hj) _ _]
      have hwupd : ∀ a : Ω, w t (Function.update ω t a) (Function.update ω t a t) = w t ω a := fun a => by
        rw [Function.update_self,
          hw t (Function.update ω t a) ω fun j hj => Function.update_of_ne (ne_of_lt hj) _ _]
      have hE : Ex (fun a : Ω => if A t (Function.update ω t a) (Function.update ω t a t) then (1 : ℝ) else 0) =
          Pr (A t ω) := by
        simp only [hupd]
        exact (Pr_eq_sum (A t ω)).symm
      have hW : Ex (fun a : Ω => w t (Function.update ω t a) (Function.update ω t a t)) = Ex (w t ω) :=
        congrArg Ex (funext hwupd)
      have h1 : Pr (A t ω) ≤ min 1 (r ^ f t ω + e t) + Ex (w t ω) := by
        have hw1 : 0 ≤ Ex (w t ω) := Ex_nonneg fun x => hw0 t ω x
        rcases le_total 1 (r ^ f t ω + e t) with h2 | h2
        · rw [min_eq_left h2]
          have h3 := Pr_le_one (A t ω)
          linarith
        · rw [min_eq_right h2]
          exact hesc t ω
      exact (hE.trans_le h1).trans_eq (congrArg (fun y => min 1 (r ^ f t ω + e t) + y) hW.symm))
    (r ^ F + ∑ i, e i) (add_nonneg (pow_nonneg hr0 F) (Finset.sum_nonneg fun i _ => he i))
  have hprod : ∀ ω : Fin m → Ω, F ≤ ∑ i, f i ω → ∏ t, min 1 (r ^ f t ω + e t) ≤ r ^ F + ∑ i, e i :=
    fun ω hF => by
      refine (prod_min_add_le (fun i => r ^ f i ω) e (fun i => pow_nonneg hr0 _)
        (fun i => pow_le_one₀ hr0 hr1) he).trans ?_
      rw [Finset.prod_pow_eq_pow_sum]
      exact add_le_add (pow_le_pow_of_le_one hr0 hr1 hF) le_rfl
  have hfin : Pr (fun ω : Fin m → Ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤
      Ex fun ω : Fin m → Ω => (∏ t, if A t ω (ω t) then (1 : ℝ) else 0) *
        if ∏ t, min 1 (r ^ f t ω + e t) ≤ r ^ F + ∑ i, e i then 1 else 0 := by
    refine Pr_le_Ex _ _ (fun ω => mul_nonneg (Finset.prod_nonneg fun t _ => hind0 t ω) ?_) fun ω hω => ?_
    · split_ifs
      · exact zero_le_one
      · exact le_rfl
    · have hone : (∏ t, if A t ω (ω t) then (1 : ℝ) else 0) = 1 := Finset.prod_eq_one fun t _ => if_pos (hω.1 t)
      refine le_of_eq ?_
      rw [if_pos (hprod ω hω.2), mul_one, hone]
  exact hfin.trans key

/-- **The escapes multiply with additive errors**: execution phases whose audits, given the earlier phases' coins,
accept with probability at most `r^(f i) + e i`, `f i` fixed before phase `i`'s coins, all accept with `Σ_i f i ≥ F`
with probability at most `r^F + Σ_i e i`: `pr_adaptive_add_ex_le` at the charge zero. -/
theorem pr_adaptive_add_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1)
    (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ) (e : Fin m → ℝ)
    (hA : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω')
    (hf : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') (he : ∀ i, 0 ≤ e i)
    (hesc : ∀ i ω, Pr (A i ω) ≤ r ^ f i ω + e i) (F : ℕ) :
    Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F + ∑ i, e i := by
  have h := pr_adaptive_add_ex_le hr0 hr1 m A f e (fun _ _ _ => 0) hA hf (fun _ _ _ _ => rfl) he
    (fun _ _ _ => le_rfl)
    (fun i ω => by
      simp only [Ex_const, add_zero]
      exact hesc i ω)
    F
  simp only [Finset.sum_const_zero, Ex_const, add_zero] at h
  exact h

/-- **The escapes multiply**: execution phases whose audits, given the earlier phases' coins, accept with probability
at most `r^(f i)`, `f i` fixed before phase `i`'s coins, all accept with `Σ_i f i ≥ F` with probability at most `r^F`:
`pr_adaptive_add_le` at the errors zero. -/
theorem pr_adaptive_le {Ω : Type} [Fintype Ω] [Nonempty Ω] {r : ℝ} (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    ∀ (m : ℕ) (A : Fin m → (Fin m → Ω) → Ω → Prop) (f : Fin m → (Fin m → Ω) → ℕ),
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → A i ω = A i ω') →
      (∀ i ω ω', (∀ j, j < i → ω j = ω' j) → f i ω = f i ω') →
      (∀ i ω, Pr (A i ω) ≤ r ^ f i ω) → ∀ F : ℕ,
      Pr (fun ω => (∀ i, A i ω (ω i)) ∧ F ≤ ∑ i, f i ω) ≤ r ^ F := by
  intro m A f hA hf hesc F
  have h := pr_adaptive_add_le hr0 hr1 m A f (fun _ => 0) hA hf (fun _ => le_rfl)
    (fun i ω => by
      simp only [add_zero]
      exact hesc i ω)
    F
  simp only [Finset.sum_const_zero, add_zero] at h
  exact h

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
