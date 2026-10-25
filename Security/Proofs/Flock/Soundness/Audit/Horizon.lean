import Proofs.Flock.Soundness.Audit.Law
import Proofs.Flock.Claims.Sampling.Draws
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The horizon escape: wrong units that escape every period's draw

A horizon is `T` periods, each audited by a fresh uniform `kd`-subset of the same `N` units (`Law.subset N kd`),
independent of the past. An adaptive adversary (`Horizon`) fixes period `t`'s wrong set `W_t` from the draws before
`t`, before draw `t` is made. It escapes when no period's draw hits its wrong set (`Horizon.Escapes`), and its total is
`Σ_t |W_t|` (`Horizon.total`), the wrong units that went unaudited over the horizon.

* `horizon_escape_prod`: when period `t`'s set has at least `w_t` units whatever the history, the adversary escapes
  with at most `∏_t C(N − w_t, kd)/C(N, kd)`, each period's own `subset_escape`.
* `choose_div_le_one_sub_pow`, `prod_choose_div_le_pow`: `C(N − w, kd)/C(N, kd) ≤ (1 − kd/N)^w`, so the product is
  at most `(1 − kd/N)^M` once `Σ_t w_t ≥ M`.
* `horizon_escape`: with the set sizes adaptive too, `Pr[escape ∧ Σ_t |W_t| ≥ M] ≤ (1 − kd/N)^M`.
* `horizon_escape_Pr`, `horizon_escape_log`: the same in `Claims.Pr`, and at `δ`: with `q = kd/N`, the total reaches
  `ln(1/δ)/q` on an escape with probability at most `δ`.
* `horizon_escape_adapted`, `horizon_escape_log_adapted`: the same for a wrong-set map on whole runs that doesn't look
  ahead (`NonAnticipating`), which is a `Horizon` (`Horizon.ofAdapted`).

Proof: sequential conditioning on the first period's draw. Given draw 0, the remaining periods are a `Horizon` of
`T − 1` periods (`Horizon.tail`), so the bound is an induction on `T`, the tower rule over the history; the adversary
may be adaptive because `W_t` is fixed before draw `t`.

network-accounting composes `horizon_escape_log` with the per-period D1 into
`K_horizon = (ln(1/δ)/q)·(16 + log2 unit index) + Σ_windows (ε_ks + δ_link)`: each window's error adds once per
window, so the second term grows with the number of windows in the horizon. That composition is `HorizonLeakAccepted`'s, not this
file's.
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

variable {N kd : ℕ}

/-- The draws before period `t`. -/
def past {α : Type} {T : ℕ} (ω : Fin T → α) (t : Fin T) : Fin t → α :=
  fun i => ω (Fin.castLE t.isLt.le i)

theorem past_cons_succ {α : Type} {T : ℕ} (a : α) (ρ : Fin T → α) (s : Fin T) :
    past (Fin.cons a ρ : Fin (T + 1) → α) s.succ = (Fin.cons a (past ρ s) : Fin (s.val + 1) → α) := by
  funext i
  refine Fin.cases ?_ (fun j => ?_) i
  · have h0 : Fin.castLE (Fin.succ s).isLt.le (0 : Fin (s.val + 1)) = 0 := Fin.ext (by simp)
    simp only [past, h0, Fin.cons_zero]
  · have hj : Fin.castLE (Fin.succ s).isLt.le j.succ = (Fin.castLE s.isLt.le j).succ := Fin.ext (by simp)
    simp only [past, hj, Fin.cons_succ]

/-- **An adaptive horizon adversary**: before period `t`'s draw, from the draws before `t`, period `t`'s wrong set. -/
abbrev Horizon (N kd : ℕ) (hk : kd ≤ N) (T : ℕ) : Type :=
  (t : Fin T) → (Fin t → (subset N kd hk).Ω) → Finset (Fin N)

namespace Horizon

variable {hk : kd ≤ N} {T : ℕ}

/-- Period `t`'s wrong set on the run `ω`. -/
def wrong (A : Horizon N kd hk T) (ω : Fin T → (subset N kd hk).Ω) (t : Fin T) : Finset (Fin N) :=
  A t (past ω t)

/-- Every period's draw misses its wrong set. -/
def Escapes (A : Horizon N kd hk T) (ω : Fin T → (subset N kd hk).Ω) : Prop :=
  ∀ t, Disjoint (ω t).1 (A.wrong ω t)

/-- The wrong units over the horizon, `Σ_t |W_t|`. -/
def total (A : Horizon N kd hk T) (ω : Fin T → (subset N kd hk).Ω) : ℕ :=
  ∑ t, (A.wrong ω t).card

/-- Period 0's wrong set, fixed before any draw. -/
def first (A : Horizon N kd hk (T + 1)) : Finset (Fin N) :=
  A 0 fun i => absurd i.isLt (by simp)

/-- The adversary once draw 0 is `a`: a horizon of the remaining `T` periods. -/
def tail (A : Horizon N kd hk (T + 1)) (a : (subset N kd hk).Ω) : Horizon N kd hk T :=
  fun s h => A s.succ (Fin.cons a h)

theorem wrong_cons_zero (A : Horizon N kd hk (T + 1)) (a : (subset N kd hk).Ω)
    (ρ : Fin T → (subset N kd hk).Ω) : A.wrong (Fin.cons a ρ) 0 = A.first := by
  show A 0 (past (Fin.cons a ρ) 0) = A 0 _
  exact congrArg (A 0) (funext fun i => absurd i.isLt (by simp))

theorem wrong_cons_succ (A : Horizon N kd hk (T + 1)) (a : (subset N kd hk).Ω)
    (ρ : Fin T → (subset N kd hk).Ω) (s : Fin T) : A.wrong (Fin.cons a ρ) s.succ = (A.tail a).wrong ρ s := by
  show A s.succ (past (Fin.cons a ρ) s.succ) = A s.succ (Fin.cons a (past ρ s))
  exact congrArg (A s.succ) (past_cons_succ a ρ s)

theorem escapes_cons (A : Horizon N kd hk (T + 1)) (a : (subset N kd hk).Ω) (ρ : Fin T → (subset N kd hk).Ω) :
    A.Escapes (Fin.cons a ρ) ↔ Disjoint a.1 A.first ∧ (A.tail a).Escapes ρ := by
  unfold Escapes
  rw [Fin.forall_fin_succ]
  simp only [Fin.cons_zero, Fin.cons_succ, wrong_cons_zero, wrong_cons_succ]

theorem total_cons (A : Horizon N kd hk (T + 1)) (a : (subset N kd hk).Ω) (ρ : Fin T → (subset N kd hk).Ω) :
    A.total (Fin.cons a ρ) = A.first.card + (A.tail a).total ρ := by
  unfold total
  rw [Fin.sum_univ_succ]
  simp only [wrong_cons_zero, wrong_cons_succ]

end Horizon

/-- A wrong-set map on whole runs that doesn't look ahead: period `t`'s set depends only on the draws before `t`. -/
def NonAnticipating {T : ℕ} {Ω : Type} (W : Fin T → (Fin T → Ω) → Finset (Fin N)) : Prop :=
  ∀ t ω ω', (∀ i, i < t → ω i = ω' i) → W t ω = W t ω'

/-- A wrong-set map that doesn't look ahead, as a horizon adversary: the history is padded with an arbitrary draw,
which `W t` doesn't read. -/
noncomputable def Horizon.ofAdapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)) : Horizon N kd hk T :=
  fun t h => W t fun i => if hi : i.val < t.val then h ⟨i.val, hi⟩ else Classical.arbitrary _

theorem Horizon.wrong_ofAdapted {hk : kd ≤ N} {T : ℕ}
    {W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)} (hW : NonAnticipating W)
    (ω : Fin T → (subset N kd hk).Ω) (t : Fin T) : (Horizon.ofAdapted W).wrong ω t = W t ω := by
  refine hW t _ _ fun i hi => ?_
  have hi' : i.val < t.val := hi
  show (if h : i.val < t.val then past ω t ⟨i.val, h⟩ else _) = ω i
  rw [dite_eq_left hi']
  rfl

private theorem prCoin_le_one₀ {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) : prCoin p ≤ 1 := by
  classical
  unfold prCoin
  refine (avg_mono fun c => ?_).trans (avg_const (C := C) 1).le
  split_ifs <;> simp

/-- A uniform tuple on `Fin (T + 1)` is a uniform head and an independent uniform tail. -/
theorem prCoin_fin_cons {Ω : Type} [Fintype Ω] [Nonempty Ω] {T : ℕ} (P : (Fin (T + 1) → Ω) → Prop) :
    prCoin P = avg fun a : Ω => prCoin fun ρ : Fin T → Ω => P (Fin.cons a ρ) := by
  classical
  simp only [prCoin, avg]
  rw [← (Fin.consEquiv (fun _ => Ω)).sum_comp, Fintype.sum_prod_type]
  simp only [Finset.mul_sum, ← mul_assoc]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun v _ => ?_
  congr 1
  simp only [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow]
  rw [pow_succ', ENNReal.mul_inv (Or.inr (by simp)) (Or.inl (by simp))]

/-- **One period's escape against the Bernoulli rate**: `C(N − w, kd)/C(N, kd) ≤ (1 − kd/N)^w`. -/
theorem choose_div_le_one_sub_pow (hk : kd ≤ N) {w : ℕ} (hw : w ≤ N) :
    ((N - w).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) ≤ (1 - (kd : ℝ≥0∞) / N) ^ w := by
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · obtain rfl : w = 0 := by omega
    obtain rfl : kd = 0 := by omega
    simp
  have hN0 : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  have hq : (1 - (kd : ℝ≥0∞) / N) = ((N - kd : ℕ) : ℝ≥0∞) / (N : ℝ≥0∞) := by
    rw [Nat.cast_sub hk, ENNReal.sub_div (fun _ _ => hN0), ENNReal.div_self hN0 (ENNReal.natCast_ne_top _)]
  rw [hq, div_eq_mul_inv ((N - kd : ℕ) : ℝ≥0∞), mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv]
  have hc : (N.choose kd : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
  have hp : ((N : ℝ≥0∞)) ^ w ≠ 0 := pow_ne_zero _ hN0
  rw [ENNReal.div_le_iff hc (ENNReal.natCast_ne_top _), mul_comm, ← mul_div_assoc,
    ENNReal.le_div_iff_mul_le (Or.inl hp) (Or.inl (ENNReal.pow_ne_top (ENNReal.natCast_ne_top _)))]
  exact_mod_cast choose_sub_mul_pow_le N kd w hw

theorem one_sub_div_le_one : (1 - (kd : ℝ≥0∞) / N) ≤ 1 := tsub_le_self

/-- The product of the periods' escapes is at most `(1 − kd/N)^M` once `Σ_t w_t ≥ M`. -/
theorem prod_choose_div_le_pow (hk : kd ≤ N) {T : ℕ} (w : Fin T → ℕ) (hw : ∀ t, w t ≤ N) {M : ℕ}
    (hM : M ≤ ∑ t, w t) :
    ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) ≤ (1 - (kd : ℝ≥0∞) / N) ^ M :=
  calc ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞)
      ≤ ∏ t, (1 - (kd : ℝ≥0∞) / N) ^ w t := Finset.prod_le_prod fun t _ => choose_div_le_one_sub_pow hk (hw t)
    _ = (1 - (kd : ℝ≥0∞) / N) ^ ∑ t, w t := Finset.prod_pow_eq_pow_sum _ _ _
    _ ≤ (1 - (kd : ℝ≥0∞) / N) ^ M := pow_le_pow_of_le_one zero_le one_sub_div_le_one hM

/-- **The product bound.** If period `t`'s wrong set has at least `w t` units whatever the history, the adversary
escapes every period with at most `∏_t C(N − w_t, kd)/C(N, kd)`. -/
theorem horizon_escape_prod {hk : kd ≤ N} {T : ℕ} (A : Horizon N kd hk T) (w : Fin T → ℕ)
    (hw : ∀ t h, w t ≤ (A t h).card) :
    prCoin (fun ω => A.Escapes ω) ≤ ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) := by
  induction T with
  | zero =>
      simp only [Finset.univ_eq_empty, Finset.prod_empty]
      exact prCoin_le_one₀ _
  | succ T ih =>
      rw [prCoin_fin_cons, Fin.prod_univ_succ]
      set c : ℝ≥0∞ := ∏ s : Fin T, ((N - w s.succ).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞)
      have hstep : ∀ a : (subset N kd hk).Ω,
          prCoin (fun ρ : Fin T → (subset N kd hk).Ω => A.Escapes (Fin.cons a ρ)) ≤
            (if Disjoint a.1 A.first then 1 else 0) * c := by
        intro a
        by_cases ha : Disjoint a.1 A.first
        · rw [ite_eq_left ha, one_mul]
          exact (prCoin_mono fun ρ h => ((Horizon.escapes_cons A a ρ).1 h).2).trans
            (ih (A.tail a) (fun s => w s.succ) fun s h => hw s.succ _)
        · rw [ite_eq_right ha, zero_mul]
          exact (prCoin_mono (q := fun _ => False) fun ρ h => ha ((Horizon.escapes_cons A a ρ).1 h).1).trans_eq
            prCoin_false
      have h0 : w 0 ≤ A.first.card := hw 0 _
      calc avg (fun a => prCoin fun ρ : Fin T → (subset N kd hk).Ω => A.Escapes (Fin.cons a ρ))
          ≤ avg (fun a : (subset N kd hk).Ω => (if Disjoint a.1 A.first then 1 else 0) * c) := avg_mono hstep
        _ = (subset N kd hk).escape A.first * c := by
            rw [avg_mul_const, avg_ite_eq_prCoin]
            rfl
        _ ≤ ((N - w 0).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) * c := by
            rw [subset_escape]
            exact mul_le_mul_of_nonneg_right (ENNReal.div_le_div_right
              (Nat.cast_le.2 (Nat.choose_le_choose kd (Nat.sub_le_sub_left h0 N))) _) zero_le

/-- **The horizon escape (HorizonEscape).** An adaptive adversary escapes every period's draw with at least `M` wrong
units over the horizon with probability at most `(1 − kd/N)^M`. -/
theorem horizon_escape {hk : kd ≤ N} {T : ℕ} (A : Horizon N kd hk T) (M : ℕ) :
    prCoin (fun ω => A.Escapes ω ∧ M ≤ A.total ω) ≤ (1 - (kd : ℝ≥0∞) / N) ^ M := by
  induction T generalizing M with
  | zero =>
      rcases Nat.eq_zero_or_pos M with rfl | hM
      · rw [pow_zero]
        exact prCoin_le_one₀ _
      · refine ((prCoin_mono (q := fun _ => False) fun ω h => ?_).trans_eq prCoin_false).trans zero_le
        have h2 : M ≤ A.total ω := h.2
        have : A.total ω = 0 := by simp [Horizon.total]
        omega
  | succ T ih =>
      rw [prCoin_fin_cons]
      set ρq : ℝ≥0∞ := 1 - (kd : ℝ≥0∞) / N
      have hstep : ∀ a : (subset N kd hk).Ω,
          prCoin (fun ρ : Fin T → (subset N kd hk).Ω =>
              A.Escapes (Fin.cons a ρ) ∧ M ≤ A.total (Fin.cons a ρ)) ≤
            (if Disjoint a.1 A.first then 1 else 0) * ρq ^ (M - A.first.card) := by
        intro a
        by_cases ha : Disjoint a.1 A.first
        · rw [ite_eq_left ha, one_mul]
          refine (prCoin_mono fun ρ h => ?_).trans (ih (A.tail a) (M - A.first.card))
          simp only [Horizon.escapes_cons, Horizon.total_cons] at h
          have h2 := h.2
          exact ⟨h.1.2, by omega⟩
        · rw [ite_eq_right ha, zero_mul]
          refine (prCoin_mono (q := fun _ => False) fun ρ h => ?_).trans_eq prCoin_false
          simp only [Horizon.escapes_cons] at h
          exact ha h.1.1
      have hfirst : A.first.card ≤ N := by simpa using Finset.card_le_univ A.first
      calc avg (fun a => prCoin fun ρ : Fin T → (subset N kd hk).Ω =>
              A.Escapes (Fin.cons a ρ) ∧ M ≤ A.total (Fin.cons a ρ))
          ≤ avg (fun a : (subset N kd hk).Ω => (if Disjoint a.1 A.first then 1 else 0) * ρq ^ (M - A.first.card)) :=
            avg_mono hstep
        _ = (subset N kd hk).escape A.first * ρq ^ (M - A.first.card) := by
            rw [avg_mul_const, avg_ite_eq_prCoin]
            rfl
        _ ≤ ρq ^ A.first.card * ρq ^ (M - A.first.card) := by
            rw [subset_escape]
            exact mul_le_mul_of_nonneg_right (choose_div_le_one_sub_pow hk hfirst) zero_le
        _ = ρq ^ (A.first.card + (M - A.first.card)) := (pow_add _ _ _).symm
        _ ≤ ρq ^ M := pow_le_pow_of_le_one zero_le one_sub_div_le_one (by omega)

/-- `horizon_escape` in `Claims.Pr`. -/
theorem horizon_escape_Pr {hk : kd ≤ N} {T : ℕ} (A : Horizon N kd hk T) (M : ℕ) :
    Claims.Pr (fun ω => A.Escapes ω ∧ M ≤ A.total ω) ≤ (1 - (kd : ℝ) / N) ^ M := by
  have hle : (kd : ℝ≥0∞) / N ≤ 1 := ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast hk)
  have hne : (1 - (kd : ℝ≥0∞) / N) ^ M ≠ ⊤ :=
    ENNReal.pow_ne_top (ne_top_of_le_ne_top ENNReal.one_ne_top one_sub_div_le_one)
  rw [Claims.Sampling.Pr_eq_toReal_prCoin]
  refine (ENNReal.toReal_mono hne (horizon_escape A M)).trans_eq ?_
  rw [ENNReal.toReal_pow, ENNReal.toReal_sub_of_le hle ENNReal.one_ne_top, ENNReal.toReal_one,
    ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

/-- **The horizon escape at `δ`.** With `q = kd/N`, an adaptive adversary escapes every period's draw with at least
`ln(1/δ)/q` wrong units over the horizon with probability at most `δ`. -/
theorem horizon_escape_log {hk : kd ≤ N} {T : ℕ} (A : Horizon N kd hk T) (hq : 0 < kd) {δ : ℝ} (hδ : 0 < δ) :
    Claims.Pr (fun ω => A.Escapes ω ∧ Real.log (1 / δ) / ((kd : ℝ) / N) ≤ (A.total ω : ℝ)) ≤ δ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast hq.trans_le hk
  have hq0 : (0 : ℝ) < (kd : ℝ) / N := div_pos (by exact_mod_cast hq) hN
  have hq1 : (kd : ℝ) / N ≤ 1 := (div_le_one hN).2 (by exact_mod_cast hk)
  set q : ℝ := (kd : ℝ) / N
  set x : ℝ := Real.log (1 / δ) / q with hx
  calc Claims.Pr (fun ω => A.Escapes ω ∧ x ≤ (A.total ω : ℝ))
      ≤ Claims.Pr (fun ω => A.Escapes ω ∧ ⌈x⌉₊ ≤ A.total ω) :=
        Claims.Pr_mono fun ω h => ⟨h.1, Nat.ceil_le.2 h.2⟩
    _ ≤ (1 - q) ^ ⌈x⌉₊ := horizon_escape_Pr A _
    _ ≤ Real.exp (-q) ^ ⌈x⌉₊ :=
        pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-q)]) _
    _ = Real.exp (-q * ⌈x⌉₊) := by
        rw [← Real.exp_nat_mul, mul_comm]
    _ ≤ Real.exp (-q * x) := Real.exp_le_exp.2 (mul_le_mul_of_nonpos_left (Nat.le_ceil x) (by linarith))
    _ = δ := by
        rw [hx, neg_mul, mul_div_cancel₀ _ hq0.ne', one_div, Real.log_inv, neg_neg, Real.exp_log hδ]

/-- `horizon_escape` for a wrong-set map on whole runs that doesn't look ahead. -/
theorem horizon_escape_adapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)) (hW : NonAnticipating W) (M : ℕ) :
    prCoin (fun ω => (∀ t, Disjoint (ω t).1 (W t ω)) ∧ M ≤ ∑ t, (W t ω).card) ≤
      (1 - (kd : ℝ≥0∞) / N) ^ M := by
  have h := horizon_escape (Horizon.ofAdapted W) M
  simp only [Horizon.Escapes, Horizon.total, Horizon.wrong_ofAdapted hW] at h
  exact h

/-- `horizon_escape_log` for a wrong-set map on whole runs that doesn't look ahead. -/
theorem horizon_escape_log_adapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)) (hW : NonAnticipating W) (hq : 0 < kd) {δ : ℝ}
    (hδ : 0 < δ) :
    Claims.Pr (fun ω => (∀ t, Disjoint (ω t).1 (W t ω)) ∧
      Real.log (1 / δ) / ((kd : ℝ) / N) ≤ ((∑ t, (W t ω).card : ℕ) : ℝ)) ≤ δ := by
  have h := horizon_escape_log (Horizon.ofAdapted W) hq hδ
  simp only [Horizon.Escapes, Horizon.total, Horizon.wrong_ofAdapted hW] at h
  exact h

end Law

end FlockSoundness.Audit
