import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Indexed
import Definitions.Pous.Game.Prob
import Proofs.Pous.Sanity

/-!
# The counting step of the continuous audit

Pure probability, about no scheme. Round `j` draws `I j` uniformly from a finite `α`, independently of the rounds
before; `X h a` says that round `|h|` misses when the history is `h` and the draw is `a`, and `q h` is a lower bound on
that round's conditional miss probability, `q h ≤ Pr_a[X h a]`, fixed by the history. Then for every `θ > 0`

  `Pr[#misses ≤ f ∧ s ≤ Σ_j q_j] ≤ exp(θ·f − (1 − e^−θ)·s)`   (`pr_misses_le`),

and at `s` just above `G(f) = inf_{θ > 0} (θ·f + ln(1/β)) / (1 − e^−θ)` the right side is `β`
(`pr_misses_gt_le`). Per challenge, `g(f) = G(f)/N` at the realized number `N` of challenges, never `G(f)/(λT)`
(`Security/Definitions/Pous/Model/Continuous.lean`).

The proof is the exponential supermartingale `exp(−θ·#misses + (1 − e^−θ)·Σ q)`: one round multiplies its mean by
`e^{(1 − e^−θ) q}·(1 − (1 − e^−θ)·p) ≤ 1` when `q ≤ p` (`sum_exp_le`), and Markov's inequality does the rest. A history
`h` is the list of earlier draws, oldest first, so round `j`'s is `(List.ofFn I).take j`. `q` may take any real values,
and `q h` and `X h` may be any functions of `h`: the bound holds for every adapted choice.
-/

namespace PousContinuous

open Pous
open scoped ENNReal

variable {α : Type} [Fintype α]

theorem pr_toReal (P : α → Prop) [DecidablePred P] :
    (pr P).toReal = ((Finset.univ.filter P).card : ℝ) / Fintype.card α := by
  rw [pr_def, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem pr_ne_top [Nonempty α] (P : α → Prop) : pr P ≠ ⊤ := by
  unfold pr
  exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by exact_mod_cast Fintype.card_ne_zero)

open Classical in
/-- One round's factor: `Σ_a e^{−θ·[P a]} = |α| − (1 − e^−θ)·|P|`. -/
theorem sum_exp_ind (θ : ℝ) (P : α → Prop) :
    ∑ a, Real.exp (-θ * (if P a then 1 else 0)) =
      Fintype.card α - (1 - Real.exp (-θ)) * (Finset.univ.filter P).card := by
  have h : ∀ a, Real.exp (-θ * (if P a then 1 else 0)) = 1 - (1 - Real.exp (-θ)) * (if P a then 1 else 0) := by
    intro a
    split_ifs <;> simp
  simp_rw [h]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_boole]
  simp

/-- One round keeps the supermartingale's mean: `e^{c q}·(N − c·p·N) ≤ N` when `q ≤ p`, `c ≥ 0`, `N ≥ 0`. -/
theorem step_le (c q p N : ℝ) (hc : 0 ≤ c) (hqp : q ≤ p) (hN : 0 ≤ N) :
    Real.exp (c * q) * (N - c * (p * N)) ≤ N := by
  have h1 : N - c * (p * N) ≤ N * (1 - c * q) := by nlinarith [mul_le_mul_of_nonneg_left hqp hc]
  have h2 : 1 - c * q ≤ Real.exp (-(c * q)) := by linarith [Real.add_one_le_exp (-(c * q))]
  calc Real.exp (c * q) * (N - c * (p * N))
      ≤ Real.exp (c * q) * (N * Real.exp (-(c * q))) :=
        mul_le_mul_of_nonneg_left (h1.trans (mul_le_mul_of_nonneg_left h2 hN)) (Real.exp_pos _).le
    _ = N := by rw [mul_left_comm, ← Real.exp_add, add_neg_cancel, Real.exp_zero, mul_one]

open Classical in
/-- **The supermartingale's mean.** `Σ_I exp(−θ·#misses(I) + (1 − e^−θ)·Σ_j q_j(I)) ≤ |α|^n`. -/
theorem sum_exp_le [Nonempty α] (θ : ℝ) (hθ : 0 < θ) :
    ∀ (n : ℕ) (X : List α → α → Prop) (q : List α → ℝ),
      (∀ h : List α, h.length < n → q h ≤ (pr (X h)).toReal) →
      ∑ I : Fin n → α, Real.exp (-θ * (∑ j : Fin n, if X ((List.ofFn I).take j) (I j) then 1 else 0) +
          (1 - Real.exp (-θ)) * ∑ j : Fin n, q ((List.ofFn I).take j)) ≤ (Fintype.card α : ℝ) ^ n
  | 0, X, q, _ => by simp
  | n + 1, X, q, hq => by
      have hc : 0 ≤ 1 - Real.exp (-θ) := by
        have := Real.exp_le_one_iff.2 (neg_nonpos.2 hθ.le)
        linarith
      have ih : ∀ a, ∑ I : Fin n → α,
          Real.exp (-θ * (∑ j : Fin n, if X (a :: (List.ofFn I).take j) (I j) then 1 else 0) +
            (1 - Real.exp (-θ)) * ∑ j : Fin n, q (a :: (List.ofFn I).take j)) ≤ (Fintype.card α : ℝ) ^ n :=
        fun a => sum_exp_le θ hθ n (fun h => X (a :: h)) (fun h => q (a :: h))
          fun h hh => hq (a :: h) (by simp only [List.length_cons]; omega)
      rw [← (Fin.consEquiv fun _ : Fin (n + 1) => α).sum_comp, Fintype.sum_prod_type]
      simp only [Fin.consEquiv_apply, List.ofFn_succ, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ,
        Fin.val_zero, Fin.val_succ, List.take_zero, List.take_succ_cons]
      have hsplit : ∀ (a : α) (I : Fin n → α),
          Real.exp (-θ * ((if X [] a then 1 else 0) +
              ∑ j : Fin n, if X (a :: (List.ofFn fun i => I i).take j) (I j) then 1 else 0) +
            (1 - Real.exp (-θ)) * (q [] + ∑ j : Fin n, q (a :: (List.ofFn fun i => I i).take j))) =
          Real.exp (-θ * (if X [] a then 1 else 0)) * Real.exp ((1 - Real.exp (-θ)) * q []) *
            Real.exp (-θ * (∑ j : Fin n, if X (a :: (List.ofFn I).take j) (I j) then 1 else 0) +
              (1 - Real.exp (-θ)) * ∑ j : Fin n, q (a :: (List.ofFn I).take j)) := by
        intro a I
        rw [← Real.exp_add, ← Real.exp_add]
        congr 1
        ring
      simp_rw [hsplit, ← Finset.mul_sum]
      calc ∑ a, Real.exp (-θ * (if X [] a then 1 else 0)) * Real.exp ((1 - Real.exp (-θ)) * q []) *
            ∑ I : Fin n → α, Real.exp (-θ * (∑ j : Fin n, if X (a :: (List.ofFn I).take j) (I j) then 1 else 0) +
              (1 - Real.exp (-θ)) * ∑ j : Fin n, q (a :: (List.ofFn I).take j))
          ≤ ∑ a, Real.exp (-θ * (if X [] a then 1 else 0)) * Real.exp ((1 - Real.exp (-θ)) * q []) *
              (Fintype.card α : ℝ) ^ n :=
            Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (ih a) (by positivity)
        _ = Real.exp ((1 - Real.exp (-θ)) * q []) * ∑ a, Real.exp (-θ * (if X [] a then 1 else 0)) *
              (Fintype.card α : ℝ) ^ n := by
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun a _ => ?_
            ring
        _ = Real.exp ((1 - Real.exp (-θ)) * q []) *
              ((Fintype.card α : ℝ) - (1 - Real.exp (-θ)) *
                ((pr (X [])).toReal * Fintype.card α)) * (Fintype.card α : ℝ) ^ n := by
            rw [← Finset.sum_mul, sum_exp_ind, pr_toReal,
              div_mul_cancel₀ _ (by exact_mod_cast Fintype.card_ne_zero)]
            ring
        _ ≤ (Fintype.card α : ℝ) * (Fintype.card α : ℝ) ^ n := by
            gcongr
            exact step_le _ _ _ _ hc (hq [] (by simp)) (Nat.cast_nonneg _)
        _ = (Fintype.card α : ℝ) ^ (n + 1) := by ring

open Classical in
/-- **The counting step, at one `θ`.** Indicators `X` of misses, each round's conditional miss probability at least
`q` given the history: for every `θ > 0`, `f` and `s`,
`Pr[#misses ≤ f ∧ s ≤ Σ_j q_j] ≤ exp(θ·f − (1 − e^−θ)·s)`. -/
theorem pr_misses_le [Nonempty α] (n : ℕ) (X : List α → α → Prop) (q : List α → ℝ)
    (hq : ∀ h : List α, h.length < n → q h ≤ (pr (X h)).toReal) (θ f s : ℝ) (hθ : 0 < θ) :
    pr (fun I : Fin n → α =>
        ((Finset.univ.filter fun j : Fin n => X ((List.ofFn I).take j) (I j)).card : ℝ) ≤ f ∧
          s ≤ ∑ j : Fin n, q ((List.ofFn I).take j)) ≤
      ENNReal.ofReal (Real.exp (θ * f - (1 - Real.exp (-θ)) * s)) := by
  have hc : 0 ≤ 1 - Real.exp (-θ) := by
    have := Real.exp_le_one_iff.2 (neg_nonpos.2 hθ.le)
    linarith
  set g : (Fin n → α) → ℝ := fun I =>
    -θ * (∑ j : Fin n, if X ((List.ofFn I).take j) (I j) then 1 else 0) +
      (1 - Real.exp (-θ)) * ∑ j : Fin n, q ((List.ofFn I).take j) with hg
  set E : (Fin n → α) → Prop := fun I =>
    ((Finset.univ.filter fun j : Fin n => X ((List.ofFn I).take j) (I j)).card : ℝ) ≤ f ∧
      s ≤ ∑ j : Fin n, q ((List.ofFn I).take j) with hE
  have hlow : ∀ I, E I → -θ * f + (1 - Real.exp (-θ)) * s ≤ g I := by
    rintro I ⟨h1, h2⟩
    rw [hg]
    dsimp only
    rw [← Finset.natCast_card_filter]
    nlinarith [mul_le_mul_of_nonneg_left h1 hθ.le, mul_le_mul_of_nonneg_left h2 hc]
  have hcount : ((Finset.univ.filter E).card : ℝ) * Real.exp (-θ * f + (1 - Real.exp (-θ)) * s) ≤
      (Fintype.card α : ℝ) ^ n :=
    calc ((Finset.univ.filter E).card : ℝ) * Real.exp (-θ * f + (1 - Real.exp (-θ)) * s)
        = ∑ _I ∈ Finset.univ.filter E, Real.exp (-θ * f + (1 - Real.exp (-θ)) * s) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ I ∈ Finset.univ.filter E, Real.exp (g I) :=
          Finset.sum_le_sum fun I hI => Real.exp_le_exp.2 (hlow I (Finset.mem_filter.1 hI).2)
      _ ≤ ∑ I, Real.exp (g I) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => (Real.exp_pos _).le
      _ ≤ (Fintype.card α : ℝ) ^ n := sum_exp_le θ hθ n X q hq
  rw [← ENNReal.ofReal_toReal (pr_ne_top E)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hpos : (0 : ℝ) < (Fintype.card α : ℝ) ^ n := pow_pos (by exact_mod_cast Fintype.card_pos) _
  rw [pr_toReal, Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, div_le_iff₀ hpos]
  have he : Real.exp (θ * f - (1 - Real.exp (-θ)) * s) * Real.exp (-θ * f + (1 - Real.exp (-θ)) * s) = 1 := by
    rw [← Real.exp_add]; ring_nf; exact Real.exp_zero
  calc ((Finset.univ.filter E).card : ℝ)
      = ((Finset.univ.filter E).card : ℝ) * Real.exp (-θ * f + (1 - Real.exp (-θ)) * s) *
          Real.exp (θ * f - (1 - Real.exp (-θ)) * s) := by rw [mul_assoc, mul_comm (Real.exp _), he, mul_one]
    _ ≤ (Fintype.card α : ℝ) ^ n * Real.exp (θ * f - (1 - Real.exp (-θ)) * s) :=
        mul_le_mul_of_nonneg_right hcount (Real.exp_pos _).le
    _ = Real.exp (θ * f - (1 - Real.exp (-θ)) * s) * (Fintype.card α : ℝ) ^ n := mul_comm _ _

open Classical in
/-- **The counting step.** Indicators `X` of misses, each round's conditional miss
probability at least `q` given the history, and any `β > 0`: the misses are at most `f` and `Σ_j q_j` exceeds
`G(f) = inf_{θ > 0} (θ·f + ln(1/β)) / (1 − e^−θ)` with probability at most `β`. With `n` rounds,
`G(f) = n·g(f)` for `g(f) = min_{θ > 0} (θ·f + ln(1/β)) / ((1 − e^−θ)·n)`. -/
theorem pr_misses_gt_le [Nonempty α] (n : ℕ) (X : List α → α → Prop) (q : List α → ℝ)
    (hq : ∀ h : List α, h.length < n → q h ≤ (pr (X h)).toReal) (f β : ℝ) (hβ : 0 < β) :
    pr (fun I : Fin n → α =>
        ((Finset.univ.filter fun j : Fin n => X ((List.ofFn I).take j) (I j)).card : ℝ) ≤ f ∧
          (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / β)) / (1 - Real.exp (-(θ : ℝ)))) <
            ∑ j : Fin n, q ((List.ofFn I).take j)) ≤ ENNReal.ofReal β := by
  set G : Set.Ioi (0 : ℝ) → ℝ := fun θ => ((θ : ℝ) * f + Real.log (1 / β)) / (1 - Real.exp (-(θ : ℝ))) with hG
  set E : (Fin n → α) → Prop := fun I =>
    ((Finset.univ.filter fun j : Fin n => X ((List.ofFn I).take j) (I j)).card : ℝ) ≤ f ∧
      iInf G < ∑ j : Fin n, q ((List.ofFn I).take j) with hE
  by_cases hS : (Finset.univ.filter E).Nonempty
  · obtain ⟨I₀, hI₀, hm⟩ := Finset.exists_mem_eq_inf' hS fun I => ∑ j : Fin n, q ((List.ofFn I).take j)
    have : Nonempty (Set.Ioi (0 : ℝ)) := ⟨⟨1, Set.mem_Ioi.2 one_pos⟩⟩
    obtain ⟨θ, hθ⟩ := exists_lt_of_ciInf_lt ((Finset.mem_filter.1 hI₀).2.2.trans_eq hm.symm)
    have hθ0 : (0 : ℝ) < θ := θ.2
    have hc : 0 < 1 - Real.exp (-(θ : ℝ)) := by
      have : Real.exp (-(θ : ℝ)) < 1 := by simpa using Real.exp_lt_exp.2 (neg_lt_zero.2 hθ0)
      linarith
    have hsub : ∀ I, E I →
        ((Finset.univ.filter fun j : Fin n => X ((List.ofFn I).take j) (I j)).card : ℝ) ≤ f ∧
          G θ ≤ ∑ j : Fin n, q ((List.ofFn I).take j) := fun I hI =>
      ⟨hI.1, hθ.le.trans (Finset.inf'_le _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hI⟩))⟩
    have hβeq : Real.exp ((θ : ℝ) * f - (1 - Real.exp (-(θ : ℝ))) * G θ) = β := by
      rw [hG]
      dsimp only
      rw [mul_div_cancel₀ _ hc.ne', sub_add_cancel_left, Real.exp_neg, Real.exp_log (by positivity), one_div,
        inv_inv]
    calc pr E ≤ _ := pr_mono hsub
      _ ≤ _ := pr_misses_le n X q hq θ f (G θ) hθ0
      _ = ENNReal.ofReal β := by rw [hβeq]
  · rw [Finset.not_nonempty_iff_eq_empty] at hS
    rw [pr_def, hS, Finset.card_empty, Nat.cast_zero, ENNReal.zero_div]
    exact zero_le

end PousContinuous
