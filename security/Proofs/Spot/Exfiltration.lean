module

public import Definitions.Spot.Exfiltration
public import Proofs.Core.Coins
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-! Exfiltration before detection: the stopping-time bounds. -/

@[expose] public section

namespace Population.Exfil

open Finset
open Verity.Game (iid_toReal iid_cons_toReal)

variable {Ω : Type} [Fintype Ω] {μ : PMF Ω}

theorem ex_zero (μ : PMF Ω) (F : (Fin 0 → Ω) → ℝ) (cs₀ : Fin 0 → Ω) : ex μ 0 F = F cs₀ := by
  rw [ex, Fintype.sum_unique, Subsingleton.elim default cs₀, iid_toReal]
  simp

theorem ex_succ (μ : PMF Ω) {n : ℕ} (F : (Fin (n + 1) → Ω) → ℝ) :
    ex μ (n + 1) F = ∑ c, (μ c).toReal * ex μ n fun cs => F (Fin.cons c cs) := by
  have e : ∀ p : Ω × (Fin n → Ω), (Fin.consEquiv fun _ : Fin (n + 1) => Ω) p = Fin.cons p.1 p.2 :=
    fun _ => rfl
  simp only [ex]
  rw [← (Fin.consEquiv fun _ : Fin (n + 1) => Ω).sum_comp, Fintype.sum_prod_type]
  simp only [e, iid_cons_toReal, mul_sum, mul_assoc]

theorem ex_mono {n : ℕ} {F G : (Fin n → Ω) → ℝ} (h : ∀ cs, F cs ≤ G cs) : ex μ n F ≤ ex μ n G :=
  sum_le_sum fun cs _ => mul_le_mul_of_nonneg_left (h cs) ENNReal.toReal_nonneg

theorem ex_const (μ : PMF Ω) (n : ℕ) (a : ℝ) : ex μ n (fun _ => a) = a := by
  induction n with
  | zero => exact ex_zero μ _ Fin.elim0
  | succ n ih => rw [ex_succ]; simp [ih, ← sum_mul, Verity.Game.sum_toReal_pmf]

theorem ex_add_const (μ : PMF Ω) {n : ℕ} (a : ℝ) (F : (Fin n → Ω) → ℝ) :
    ex μ n (fun cs => a + F cs) = a + ex μ n F := by
  have h := ex_const μ n a
  simp only [ex, mul_add, sum_add_distrib] at h ⊢
  rw [h]

open Classical in
theorem prob_eq_ex (μ : PMF Ω) (n : ℕ) (E : (Fin n → Ω) → Prop) :
    prob μ n E = ex μ n fun cs => if E cs then 1 else 0 :=
  Verity.Game.prPMFReal_eq_sum_mul _ E

theorem prob_congr {n : ℕ} {E E' : (Fin n → Ω) → Prop} (h : ∀ cs, E cs ↔ E' cs) : prob μ n E = prob μ n E' :=
  Verity.Game.prPMFReal_congr _ h

theorem prob_nonneg (n : ℕ) (E : (Fin n → Ω) → Prop) : 0 ≤ prob μ n E :=
  Verity.Game.prPMFReal_nonneg _ _

theorem prob_le_one (n : ℕ) (E : (Fin n → Ω) → Prop) : prob μ n E ≤ 1 :=
  Verity.Game.prPMFReal_le_one _ _

theorem prob_false (μ : PMF Ω) (n : ℕ) : prob μ n (fun _ => False) = 0 :=
  Verity.Game.prPMFReal_false _

open Classical in
theorem prob_zero (μ : PMF Ω) (E : (Fin 0 → Ω) → Prop) (cs₀ : Fin 0 → Ω) :
    prob μ 0 E = if E cs₀ then 1 else 0 := by
  rw [prob_eq_ex, ex_zero μ _ cs₀]

theorem prob_succ (μ : PMF Ω) {n : ℕ} (E : (Fin (n + 1) → Ω) → Prop) :
    prob μ (n + 1) E = ∑ c, (μ c).toReal * prob μ n fun cs => E (Fin.cons c cs) := by
  rw [prob_eq_ex, ex_succ]
  simp only [prob_eq_ex]

theorem sum_ite_esc (s : Finset Ω) [DecidablePred (· ∈ s)] (a : ℝ) :
    ∑ c, (μ c).toReal * (if c ∈ s then a else 0) = Verity.Game.prPMFReal μ (· ∈ s) * a := by
  rw [Verity.Game.prPMFReal_mem]
  simp [mul_ite, sum_mul]

variable [DecidableEq Ω]

omit [Fintype Ω] in
theorem cum_cons (A : Strategy Ω) (n : ℕ) (h : List Ω) (c : Ω) (cs : Fin n → Ω) :
    A.cum (n + 1) h (Fin.cons c cs) = A.leak h + if c ∈ A.esc h then A.cum n (c :: h) cs else 0 := by
  rw [Strategy.cum]
  simp only [Fin.cons_zero, Fin.tail_cons]

omit [Fintype Ω] in
theorem cumPrev_cons (A : Strategy Ω) (n : ℕ) (h : List Ω) (c : Ω) (cs : Fin n → Ω) :
    A.cumPrev (n + 1) h (Fin.cons c cs) = if c ∈ A.esc h then A.leak h + A.cumPrev n (c :: h) cs else 0 := by
  rw [Strategy.cumPrev]
  simp only [Fin.cons_zero, Fin.tail_cons]

omit [Fintype Ω] in
theorem cumUpTo_cons (A : Strategy Ω) (n j : ℕ) (h : List Ω) (c : Ω) (cs : Fin n → Ω) :
    A.cumUpTo (n + 1) (j + 1) h (Fin.cons c cs) = A.leak h +
      if c ∈ A.esc h then A.cumUpTo n (j + 1) (c :: h) cs else A.cumUpTo n j (c :: h) cs := by
  rw [Strategy.cumUpTo]
  simp only [Fin.cons_zero, Fin.tail_cons]

omit [Fintype Ω] in
theorem cum_nonneg {A : Strategy Ω} {ℓ : ℝ} (hL : LeakBounded ℓ A) :
    ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω), 0 ≤ A.cum n h cs
  | 0, _, _ => le_refl 0
  | n + 1, h, cs => by
    rw [← Fin.cons_self_tail cs, cum_cons]
    have := (hL h).1
    split_ifs
    · linarith [cum_nonneg hL n (cs 0 :: h) (Fin.tail cs)]
    · linarith

theorem tail_cum {κ ℓ : ℝ} (hκ : 0 ≤ κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
    (hL : LeakBounded ℓ A) (n : ℕ) (h : List Ω) (x : ℝ) :
    prob μ n (fun cs => x ≤ A.cum n h cs) ≤ Real.exp (-(κ * (x - ℓ))) := by
  have hℓ : 0 ≤ ℓ := (hL []).1.trans (hL []).2
  have one_le : ∀ y, y ≤ ℓ → 1 ≤ Real.exp (-(κ * (y - ℓ))) := fun y hy =>
    Real.one_le_exp (by nlinarith)
  induction n generalizing h x with
  | zero =>
    rw [prob_zero μ _ Fin.elim0]
    by_cases hx : x ≤ A.cum 0 h Fin.elim0
    · rw [ite_eq_left hx]; exact one_le x (hx.trans hℓ)
    · rw [ite_eq_right hx]; exact (Real.exp_pos _).le
  | succ n ih =>
    by_cases hx : x ≤ A.leak h
    · exact (prob_le_one _ _).trans (one_le x (hx.trans (hL h).2))
    · replace hx := not_le.1 hx
      rw [prob_succ]
      simp only [cum_cons]
      calc ∑ c, (μ c).toReal * prob μ n (fun cs => x ≤ A.leak h + if c ∈ A.esc h then A.cum n (c :: h) cs else 0)
          ≤ ∑ c, (μ c).toReal * (if c ∈ A.esc h then Real.exp (-(κ * (x - A.leak h - ℓ))) else 0) := by
            refine sum_le_sum fun c _ => mul_le_mul_of_nonneg_left ?_ ENNReal.toReal_nonneg
            split_ifs with hc
            · refine le_of_eq_of_le (prob_congr fun cs => ?_) (ih (c :: h) (x - A.leak h))
              constructor <;> intro <;> linarith
            · rw [prob_congr (E' := fun _ => False) fun cs => by simp only [add_zero, iff_false, not_le]; exact hx,
                prob_false]
        _ = Verity.Game.prPMFReal μ (· ∈ A.esc h) * Real.exp (-(κ * (x - A.leak h - ℓ))) := sum_ite_esc _ _
        _ ≤ Real.exp (-(κ * A.leak h)) * Real.exp (-(κ * (x - A.leak h - ℓ))) :=
            mul_le_mul_of_nonneg_right (hP h) (Real.exp_pos _).le
        _ = Real.exp (-(κ * (x - ℓ))) := by rw [← Real.exp_add]; ring_nf

theorem tail_cumPrev {κ : ℝ} (hκ : 0 ≤ κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
    (n : ℕ) (h : List Ω) (x : ℝ) :
    prob μ n (fun cs => x ≤ A.cumPrev n h cs) ≤ Real.exp (-(κ * x)) := by
  have one_le : ∀ y : ℝ, y ≤ 0 → 1 ≤ Real.exp (-(κ * y)) := fun y hy => Real.one_le_exp (by nlinarith)
  induction n generalizing h x with
  | zero =>
    rw [prob_zero μ _ Fin.elim0]
    by_cases hx : x ≤ A.cumPrev 0 h Fin.elim0
    · rw [ite_eq_left hx]; exact one_le x hx
    · rw [ite_eq_right hx]; exact (Real.exp_pos _).le
  | succ n ih =>
    by_cases hx : x ≤ 0
    · exact (prob_le_one _ _).trans (one_le x hx)
    · replace hx := not_le.1 hx
      rw [prob_succ]
      simp only [cumPrev_cons]
      calc ∑ c, (μ c).toReal * prob μ n (fun cs => x ≤ if c ∈ A.esc h then A.leak h + A.cumPrev n (c :: h) cs else 0)
          ≤ ∑ c, (μ c).toReal * (if c ∈ A.esc h then Real.exp (-(κ * (x - A.leak h))) else 0) := by
            refine sum_le_sum fun c _ => mul_le_mul_of_nonneg_left ?_ ENNReal.toReal_nonneg
            split_ifs with hc
            · refine le_of_eq_of_le (prob_congr fun cs => ?_) (ih (c :: h) (x - A.leak h))
              constructor <;> intro <;> linarith
            · rw [prob_congr (E' := fun _ => False) fun cs => by simp only [iff_false, not_le]; exact hx,
                prob_false]
        _ = Verity.Game.prPMFReal μ (· ∈ A.esc h) * Real.exp (-(κ * (x - A.leak h))) := sum_ite_esc _ _
        _ ≤ Real.exp (-(κ * A.leak h)) * Real.exp (-(κ * (x - A.leak h))) :=
            mul_le_mul_of_nonneg_right (hP h) (Real.exp_pos _).le
        _ = Real.exp (-(κ * x)) := by rw [← Real.exp_add]; ring_nf

theorem leak_add_esc_le {κ ℓ a e : ℝ} (hκ : 0 < κ) (ha : 0 ≤ a) (haℓ : a ≤ ℓ) (he : e ≤ Real.exp (-(κ * a))) :
    a + e * (ℓ + 1 / κ) ≤ ℓ + 1 / κ := by
  have hK : 0 < ℓ + 1 / κ := by have := one_div_pos.2 hκ; linarith
  have hexp : Real.exp (-(κ * a)) * (1 + κ * a) ≤ 1 := by
    have h1 := Real.add_one_le_exp (κ * a)
    have h2 : Real.exp (-(κ * a)) * Real.exp (κ * a) = 1 := by rw [← Real.exp_add]; simp
    nlinarith [Real.exp_pos (-(κ * a))]
  have hcore : a * (1 + κ * a) + (ℓ + 1 / κ) ≤ (ℓ + 1 / κ) * (1 + κ * a) := by
    have : (ℓ + 1 / κ) * (1 + κ * a) = ℓ + 1 / κ + κ * ℓ * a + a := by field_simp; ring
    rw [this]
    nlinarith [mul_le_mul_of_nonneg_left haℓ (mul_nonneg hκ.le ha)]
  have hpos : 0 < 1 + κ * a := by nlinarith
  have heK : e * (ℓ + 1 / κ) * (1 + κ * a) ≤ ℓ + 1 / κ := by
    calc e * (ℓ + 1 / κ) * (1 + κ * a) ≤ Real.exp (-(κ * a)) * (ℓ + 1 / κ) * (1 + κ * a) := by
          gcongr
      _ = (Real.exp (-(κ * a)) * (1 + κ * a)) * (ℓ + 1 / κ) := by ring
      _ ≤ 1 * (ℓ + 1 / κ) := by gcongr
      _ = ℓ + 1 / κ := one_mul _
  nlinarith

theorem ex_cumUpTo {κ ℓ : ℝ} (hκ : 0 < κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
    (hL : LeakBounded ℓ A) (n j : ℕ) (h : List Ω) :
    ex μ n (A.cumUpTo n j h) ≤ j * (ℓ + 1 / κ) := by
  have hℓ : 0 ≤ ℓ := (hL []).1.trans (hL []).2
  have hK : 0 ≤ ℓ + 1 / κ := by have := one_div_pos.2 hκ; linarith
  induction n generalizing j h with
  | zero =>
    rw [ex_zero μ _ Fin.elim0, Strategy.cumUpTo]
    positivity
  | succ n ih =>
    cases j with
    | zero =>
      simp only [Strategy.cumUpTo, Nat.cast_zero, zero_mul]
      rw [ex_const μ]
    | succ j =>
      rw [ex_succ]
      simp only [cumUpTo_cons]
      have hstep : ∀ c, ex μ n (fun cs => A.leak h + if c ∈ A.esc h then A.cumUpTo n (j + 1) (c :: h) cs
          else A.cumUpTo n j (c :: h) cs) ≤ A.leak h + j * (ℓ + 1 / κ) +
            (if c ∈ A.esc h then ℓ + 1 / κ else 0) := by
        intro c
        rw [ex_add_const μ]
        split_ifs with hc
        · have := ih (j + 1) (c :: h); push_cast at this; linarith
        · have := ih j (c :: h); linarith
      calc ∑ c, (μ c).toReal * ex μ n (fun cs => A.leak h + if c ∈ A.esc h then A.cumUpTo n (j + 1) (c :: h) cs
            else A.cumUpTo n j (c :: h) cs)
          ≤ ∑ c, (μ c).toReal * (A.leak h + j * (ℓ + 1 / κ) + (if c ∈ A.esc h then ℓ + 1 / κ else 0)) :=
            sum_le_sum fun c _ => mul_le_mul_of_nonneg_left (hstep c) ENNReal.toReal_nonneg
        _ = A.leak h + j * (ℓ + 1 / κ) + Verity.Game.prPMFReal μ (· ∈ A.esc h) * (ℓ + 1 / κ) := by
            rw [← sum_ite_esc]
            simp only [mul_add, sum_add_distrib, ← sum_mul, Verity.Game.sum_toReal_pmf, one_mul]
        _ ≤ ((j + 1 : ℕ) : ℝ) * (ℓ + 1 / κ) := by
            have := leak_add_esc_le hκ (hL h).1 (hL h).2 (hP h)
            push_cast
            linarith

theorem acceptProb_le_exp {U : Type} (B : Finset U) (a : U → ℝ) (ha : ∀ u ∈ B, 0 ≤ a u ∧ a u ≤ 1) :
    ∏ u ∈ B, (1 - a u) ≤ Real.exp (-∑ u ∈ B, a u) := by
  rw [← sum_neg_distrib, Real.exp_sum]
  refine prod_le_prod₀ (fun u hu => by linarith [(ha u hu).2]) fun u _ => ?_
  linarith [Real.add_one_le_exp (-a u)]

theorem pow_acceptProb_le_exp {d : ℝ} (hd1 : d ≤ 1) (k : ℕ) : (1 - d) ^ k ≤ Real.exp (-(k * d)) := by
  rw [show -(k * d) = k * (-d) by ring, Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-d)]) k

end Population.Exfil
