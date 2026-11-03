import Proofs.Flock.Soundness.Game.Prob
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Fintype.Powerset
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The monitoring period: adaptive per-period bounds over `T` periods

A monitoring period is `T` periods, period `t` with its own uniform coins in `Ω t`, independent of the others: a run is
`ω : (t : Fin T) → Ω t`, uniform. What happens in period `t` may depend on everything before it, so an event or a factor
of period `t` is a function of the whole run that reads only the periods up to `t` (`Adapted`), and a bound fixed before
period `t`'s coins reads only the periods before `t` (`Predictable`). "Given any history, period `t`'s event has
probability at most `p` over its own coins" is `∀ ω, prCoin (fun a => E t (Function.update ω t a)) ≤ p`.

* `avg_prod_le`: the one induction. Adapted factors `f t` whose average over period `t`'s coins is at most a
  predictable `g t` have `E[∏_t f_t · [∏_t g_t ≤ c]] ≤ c`: the tower rule over the history, peeling period 0
  (`avg_fin_cons`).
* `prCoin_forall_prod_le`, `prCoin_forall_le_prod`, `prCoin_forall_mem_le_prod`: events (`f` an indicator). Every
  period of `S` has its event with probability at most `∏_{t ∈ S} p_t`.
* `prCoin_count_le`: the event holds in at least `M` of the `T` periods with probability at most
  `Σ_{|S| = M} ∏_{t ∈ S} p_t`; at one `p`, `C(T, M) · p^M` (`prCoin_count_le_choose`). `prCoin_count_mul_pow_le` is
  the Chernoff form, `Pr · z^M ≤ (1 + (z − 1)·p)^T` for `z ≥ 1` (`f = z^[E]`), and `prCoin_count_le_of_log` its
  value at `δ`: `M ≥ ((z − 1)·T·p + ln(1/δ))/ln z` gives at most `δ`.
* `prCoin_forall_sum_le_pow`: when period `t`'s event has probability at most `ρ^(w t)` with `w t` predictable, every
  period's event holds with `Σ_t w_t ≥ M` with probability at most `ρ^M`. `prCoin_forall_count_le_pow` is its count:
  when `p` bounds "accepted and short" in each period given the history and shortness is fixed before the period's
  coins, every period is accepted with at least `M` short with probability at most `p^M`. Shortness that the period's
  own coins decide gets only the count bounds: an adversary short on a `p`-share of each period's coins, where it is
  accepted, is accepted in every period and short in `Binomial(T, p)` of them.

`Audit.Law`'s pooled escape (`Pooled.lean`) is the instance at uniform `kd`-subset draws.
-/

namespace FlockSoundness.Audit.Periods

open Game
open scoped ENNReal

/-- `f t` reads only the periods up to `t`. -/
def Adapted {T : ℕ} {Ω : Fin T → Type} {β : Fin T → Type} (f : (t : Fin T) → ((i : Fin T) → Ω i) → β t) : Prop :=
  ∀ t ω ω', (∀ i, i ≤ t → ω i = ω' i) → f t ω = f t ω'

/-- `f t` reads only the periods before `t`: it is fixed before period `t`'s coins. -/
def Predictable {T : ℕ} {Ω : Fin T → Type} {β : Fin T → Type} (f : (t : Fin T) → ((i : Fin T) → Ω i) → β t) :
    Prop :=
  ∀ t ω ω', (∀ i, i < t → ω i = ω' i) → f t ω = f t ω'

theorem Predictable.adapted {T : ℕ} {Ω β : Fin T → Type} {f : (t : Fin T) → ((i : Fin T) → Ω i) → β t}
    (hf : Predictable f) : Adapted f :=
  fun t ω ω' h => hf t ω ω' fun i hi => h i hi.le

section Tail

variable {T : ℕ} {Ω β : Fin (T + 1) → Type}

theorem Adapted.tail {f : (t : Fin (T + 1)) → ((i : Fin (T + 1)) → Ω i) → β t} (hf : Adapted f) (a : Ω 0) :
    Adapted (Ω := fun i => Ω i.succ) (β := fun s => β s.succ) fun s ρ => f s.succ (Fin.cons a ρ) := by
  intro s ρ ρ' h
  refine hf s.succ _ _ fun i => Fin.cases (fun _ => by simp only [Fin.cons_zero]) (fun j hj => ?_) i
  simp only [Fin.cons_succ]
  exact h j (Fin.succ_le_succ_iff.1 hj)

theorem Predictable.tail {f : (t : Fin (T + 1)) → ((i : Fin (T + 1)) → Ω i) → β t} (hf : Predictable f)
    (a : Ω 0) :
    Predictable (Ω := fun i => Ω i.succ) (β := fun s => β s.succ) fun s ρ => f s.succ (Fin.cons a ρ) := by
  intro s ρ ρ' h
  refine hf s.succ _ _ fun i => Fin.cases (fun _ => by simp only [Fin.cons_zero]) (fun j hj => ?_) i
  simp only [Fin.cons_succ]
  exact h j (Fin.succ_lt_succ_iff.1 hj)

theorem Adapted.zero {f : (t : Fin (T + 1)) → ((i : Fin (T + 1)) → Ω i) → β t} (hf : Adapted f)
    (ω ω' : (i : Fin (T + 1)) → Ω i) (h : ω 0 = ω' 0) : f 0 ω = f 0 ω' :=
  hf 0 ω ω' fun i hi => by
    obtain rfl := Fin.le_zero_iff.1 hi
    exact h

theorem Predictable.zero {f : (t : Fin (T + 1)) → ((i : Fin (T + 1)) → Ω i) → β t} (hf : Predictable f)
    (ω ω' : (i : Fin (T + 1)) → Ω i) : f 0 ω = f 0 ω' :=
  hf 0 ω ω' fun i hi => absurd hi (Fin.not_lt_zero i)

end Tail

/-- Independent periods: a uniform run of `T + 1` periods is a uniform period 0 and an independent uniform rest. -/
theorem avg_fin_cons {T : ℕ} {Ω : Fin (T + 1) → Type} [∀ t, Fintype (Ω t)]
    (F : ((i : Fin (T + 1)) → Ω i) → ℝ≥0∞) :
    avg F = avg fun a : Ω 0 => avg fun ρ : (i : Fin T) → Ω i.succ => F (Fin.cons a ρ) := by
  simp only [avg]
  rw [← (Fin.consEquiv Ω).sum_comp, Fintype.sum_prod_type]
  simp only [Finset.mul_sum, ← mul_assoc]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun v _ => ?_
  congr 1
  simp only [Fintype.card_pi, Fin.prod_univ_succ, Nat.cast_mul]
  rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]

theorem prCoin_fin_cons {T : ℕ} {Ω : Fin (T + 1) → Type} [∀ t, Fintype (Ω t)]
    (P : ((i : Fin (T + 1)) → Ω i) → Prop) :
    prCoin P = avg fun a : Ω 0 => prCoin fun ρ : (i : Fin T) → Ω i.succ => P (Fin.cons a ρ) := by
  unfold prCoin
  exact avg_fin_cons _

theorem eq_zero_of_avg_eq_zero {C : Type} [Fintype C] {u : C → ℝ≥0∞} (h : avg u = 0) (a : C) : u a = 0 := by
  rcases mul_eq_zero.1 (Finset.sum_eq_zero_iff.1 h a (Finset.mem_univ a)) with h' | h'
  · exact absurd h' (ENNReal.inv_ne_zero.2 (ENNReal.natCast_ne_top _))
  · exact h'

/-- **The monitoring-period step.** Adapted factors `f t` whose average over period `t`'s coins, at any history, is
at most a predictable `g t`: the run's product of factors, on the runs whose bounds multiply to at most `c`, averages
at most `c`. -/
theorem avg_prod_le : ∀ {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (f g : (t : Fin T) → ((i : Fin T) → Ω i) → ℝ≥0∞), Adapted f → Predictable g →
    (∀ t ω, avg (fun a : Ω t => f t (Function.update ω t a)) ≤ g t ω) → ∀ c : ℝ≥0∞,
    avg (fun ω => (∏ t, f t ω) * if ∏ t, g t ω ≤ c then 1 else 0) ≤ c
  | 0, Ω, _, _, f, g, _, _, _, c => by
      simp only [Finset.univ_eq_empty, Finset.prod_empty, one_mul]
      rw [avg_const]
      split_ifs with h1
      · exact h1
      · exact zero_le
  | T + 1, Ω, _, _, f, g, hf, hg, h, c => by
      rcases eq_or_ne c ⊤ with rfl | hc
      · exact le_top
      let ω₀ : (i : Fin (T + 1)) → Ω i := fun i => Classical.arbitrary (Ω i)
      have hga : ∀ ω, g 0 ω = g 0 ω₀ := fun ω => hg.zero ω ω₀
      rcases eq_or_ne (g 0 ω₀) 0 with h0 | h0
      · have hz : ∀ ω, f 0 ω = 0 := fun ω => by
          have h1 : avg (fun a => f 0 (Function.update ω 0 a)) = 0 :=
            le_antisymm ((h 0 ω).trans_eq ((hga ω).trans h0)) zero_le
          have := eq_zero_of_avg_eq_zero h1 (ω 0)
          rwa [Function.update_eq_self] at this
        refine le_trans (avg_mono (g := fun _ => 0) fun ω => ?_) ?_
        · simp only [Fin.prod_univ_succ, hz ω, zero_mul, le_refl]
        · rw [avg_const]
          exact zero_le
      · have hiff : ∀ x : ℝ≥0∞, g 0 ω₀ * x ≤ c ↔ x ≤ c / g 0 ω₀ := fun x => by
          rw [ENNReal.le_div_iff_mul_le (Or.inl h0) (Or.inr hc), mul_comm]
        have hfa : ∀ a ρ, f 0 (Fin.cons a ρ : (i : Fin (T + 1)) → Ω i) = f 0 (Function.update ω₀ 0 a) :=
          fun a ρ => hf.zero _ _ (by simp)
        have ih : ∀ a : Ω 0, avg (fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, f s.succ (Fin.cons a ρ)) *
              if ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c / g 0 ω₀ then 1 else 0) ≤ c / g 0 ω₀ := fun a =>
          avg_prod_le (Ω := fun i => Ω i.succ) _ _ (hf.tail a) (hg.tail a)
            (fun s ρ => by simpa only [Fin.cons_update] using h s.succ (Fin.cons a ρ)) _
        rw [avg_fin_cons]
        calc avg (fun a : Ω 0 => avg fun ρ : (i : Fin T) → Ω i.succ =>
                (∏ t, f t (Fin.cons a ρ)) * if ∏ t, g t (Fin.cons a ρ) ≤ c then 1 else 0)
            = avg (fun a : Ω 0 => avg (fun ρ : (i : Fin T) → Ω i.succ =>
                (∏ s : Fin T, f s.succ (Fin.cons a ρ)) *
                  if ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c / g 0 ω₀ then 1 else 0) *
                f 0 (Function.update ω₀ 0 a)) := by
              refine congrArg avg (funext fun a => ?_)
              rw [← avg_mul_const]
              refine congrArg avg (funext fun ρ => ?_)
              rw [Fin.prod_univ_succ, Fin.prod_univ_succ, hfa, hga]
              simp only [hiff]
              ring
          _ ≤ avg (fun a : Ω 0 => f 0 (Function.update ω₀ 0 a) * (c / g 0 ω₀)) :=
              avg_mono fun a => by
                rw [mul_comm]
                exact mul_le_mul_of_nonneg_left (ih a) zero_le
          _ = avg (fun a : Ω 0 => f 0 (Function.update ω₀ 0 a)) * (c / g 0 ω₀) := avg_mul_const _ _
          _ ≤ g 0 ω₀ * (c / g 0 ω₀) := mul_le_mul_of_nonneg_right (h 0 ω₀) zero_le
          _ ≤ c := ENNReal.mul_div_le

variable {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]

/-- **Events.** Period `t`'s adapted event has probability at most a predictable `g t` given the history: every
period's event holds, with the bounds multiplying to at most `c`, with probability at most `c`. -/
theorem prCoin_forall_prod_le (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop)
    (g : (t : Fin T) → ((i : Fin T) → Ω i) → ℝ≥0∞) (hE : Adapted E) (hg : Predictable g)
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ g t ω) (c : ℝ≥0∞) :
    prCoin (fun ω => (∀ t, E t ω) ∧ ∏ t, g t ω ≤ c) ≤ c := by
  classical
  unfold prCoin
  refine le_trans (avg_mono fun ω => ?_)
    (avg_prod_le (fun t ω => if E t ω then 1 else 0) g (fun t ω ω' hω => by simp only [hE t ω ω' hω]) hg
      (fun t ω => (avg_ite_eq_prCoin _).trans_le (h t ω)) c)
  split_ifs with h1 h2
  · simp [h1.1]
  · exact absurd h1.2 h2
  · exact zero_le
  · exact zero_le

/-- Every period's event, at constant bounds `p t`: at most `∏_t p_t`. -/
theorem prCoin_forall_le_prod (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E) (p : Fin T → ℝ≥0∞)
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ p t) :
    prCoin (fun ω => ∀ t, E t ω) ≤ ∏ t, p t :=
  (prCoin_mono fun _ hω => ⟨hω, le_rfl⟩).trans
    (prCoin_forall_prod_le E (fun t _ => p t) hE (fun _ _ _ _ => rfl) h _)

/-- Every period of `S` has its event: at most `∏_{t ∈ S} p_t`. -/
theorem prCoin_forall_mem_le_prod (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E)
    (p : Fin T → ℝ≥0∞) (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ p t)
    (S : Finset (Fin T)) :
    prCoin (fun ω => ∀ t ∈ S, E t ω) ≤ ∏ t ∈ S, p t := by
  classical
  have key := prCoin_forall_le_prod (fun t ω => t ∈ S → E t ω) (fun t ω ω' hω => by simp only [hE t ω ω' hω])
    (fun t => if t ∈ S then p t else 1) fun t ω => by
      by_cases ht : t ∈ S
      · simpa only [ht, ↓reduceIte, forall_const] using h t ω
      · simp only [ht, ↓reduceIte, false_imp_iff]
        exact (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const])
  rwa [Finset.prod_ite_mem, Finset.univ_inter] at key

open Classical in
/-- **The count.** The adapted event holds in at least `M` of the `T` periods with probability at most
`Σ_{S ⊆ [T], |S| = M} ∏_{t ∈ S} p_t`. -/
theorem prCoin_count_le (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E) (p : Fin T → ℝ≥0∞)
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ p t) (M : ℕ) :
    prCoin (fun ω => M ≤ (Finset.univ.filter fun t => E t ω).card) ≤
      ∑ S ∈ Finset.univ.powersetCard M, ∏ t ∈ S, p t := by
  refine (prCoin_mono fun ω hω => ?_).trans ((prCoin_exists_le (Finset.univ.powersetCard M)
    (fun S ω => ∀ t ∈ S, E t ω)).trans (Finset.sum_le_sum fun S _ => prCoin_forall_mem_le_prod E hE p h S))
  obtain ⟨S, hS, hSc⟩ := Finset.exists_subset_card_eq hω
  exact ⟨S, Finset.mem_powersetCard.2 ⟨Finset.subset_univ _, hSc⟩, fun t ht => (Finset.mem_filter.1 (hS ht)).2⟩

open Classical in
/-- The count at one bound `p`: at most `C(T, M) · p^M`. -/
theorem prCoin_count_le_choose (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E) {p : ℝ≥0∞}
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ p) (M : ℕ) :
    prCoin (fun ω => M ≤ (Finset.univ.filter fun t => E t ω).card) ≤ T.choose M * p ^ M := by
  refine (prCoin_count_le E hE (fun _ => p) h M).trans_eq ?_
  calc ∑ S ∈ Finset.univ.powersetCard M, ∏ _t ∈ S, p
      = ∑ _S ∈ Finset.univ.powersetCard M, p ^ M := Finset.sum_congr rfl fun S hS => by
        rw [Finset.prod_const, (Finset.mem_powersetCard.1 hS).2]
    _ = T.choose M * p ^ M := by
        rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

open Classical in
/-- **The count, Chernoff form.** For `z ≥ 1`, `Pr[at least M periods] · z^M ≤ (1 + (z − 1)·p)^T`. -/
theorem prCoin_count_mul_pow_le (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E) {p z : ℝ≥0∞}
    (hz : 1 ≤ z) (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ p) (M : ℕ) :
    prCoin (fun ω => M ≤ (Finset.univ.filter fun t => E t ω).card) * z ^ M ≤ (1 + (z - 1) * p) ^ T := by
  have hf : ∀ t ω, avg (fun a : Ω t => if E t (Function.update ω t a) then z else 1) ≤ 1 + (z - 1) * p := by
    intro t ω
    have : (fun a : Ω t => if E t (Function.update ω t a) then z else 1) =
        fun a => 1 + (if E t (Function.update ω t a) then 1 else 0) * (z - 1) := by
      funext a
      split_ifs <;> simp [add_tsub_cancel_of_le hz]
    rw [this, avg_add, avg_const, avg_mul_const, avg_ite_eq_prCoin, mul_comm]
    gcongr
    exact h t ω
  have key := avg_prod_le (fun t ω => if E t ω then z else 1) (fun _ _ => 1 + (z - 1) * p)
    (fun t ω ω' hω => by simp only [hE t ω ω' hω]) (fun _ _ _ _ => rfl) hf ((1 + (z - 1) * p) ^ T)
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin, le_refl, ↓reduceIte, mul_one] at key
  refine le_trans ?_ key
  unfold prCoin
  rw [← avg_mul_const]
  refine avg_mono fun ω => ?_
  split_ifs with hM
  · rw [one_mul, Finset.prod_ite, Finset.prod_const, Finset.prod_const_one, mul_one]
    exact pow_le_pow_right₀ hz hM
  · rw [zero_mul]
    exact zero_le

open Classical in
/-- **The count at `δ`.** At a real bound `p` per period and any `z > 1`, the event holds in at least
`((z − 1)·T·p + ln(1/δ))/ln z` periods with probability at most `δ`; at `z = e`, `(e − 1)·T·p + ln(1/δ)`. -/
theorem prCoin_count_le_of_log (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hE : Adapted E) {p : ℝ} (hp : 0 ≤ p)
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ ENNReal.ofReal p) {z δ : ℝ} (hz : 1 < z)
    (hδ : 0 < δ) {M : ℕ} (hM : ((z - 1) * T * p + Real.log (1 / δ)) / Real.log z ≤ M) :
    prCoin (fun ω => M ≤ (Finset.univ.filter fun t => E t ω).card) ≤ ENNReal.ofReal δ := by
  have hz0 : 0 < z := by linarith
  have hzp : 0 ≤ (z - 1) * p := mul_nonneg (by linarith) hp
  have hreal : (1 + (z - 1) * p) ^ T ≤ δ * z ^ M := by
    have hM' : (z - 1) * T * p + Real.log (1 / δ) ≤ M * Real.log z := (div_le_iff₀ (Real.log_pos hz)).1 hM
    rw [one_div, Real.log_inv] at hM'
    calc (1 + (z - 1) * p) ^ T ≤ Real.exp ((z - 1) * p) ^ T :=
          pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp ((z - 1) * p)]) T
      _ = Real.exp ((z - 1) * T * p) := by
          rw [← Real.exp_nat_mul]
          congr 1
          ring
      _ ≤ Real.exp (Real.log δ + Real.log (z ^ M)) := Real.exp_le_exp.2 (by rw [Real.log_pow]; linarith)
      _ = δ * z ^ M := by rw [Real.exp_add, Real.exp_log hδ, Real.exp_log (pow_pos hz0 M)]
  have hzM0 : ENNReal.ofReal z ^ M ≠ 0 := pow_ne_zero _ (ENNReal.ofReal_pos.2 hz0).ne'
  have hzMt : ENNReal.ofReal z ^ M ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofReal_ne_top
  have key := prCoin_count_mul_pow_le E hE (z := ENNReal.ofReal z) (by simpa using hz.le) h M
  rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_sub _ zero_le_one, ← ENNReal.ofReal_mul (p := z - 1) (by linarith),
    ← ENNReal.ofReal_add (p := 1) zero_le_one hzp, ← ENNReal.ofReal_pow (p := 1 + (z - 1) * p) (by linarith)] at key
  refine (ENNReal.mul_le_mul_iff_left hzM0 hzMt).1 (key.trans ?_)
  rw [← ENNReal.ofReal_pow hz0.le, ← ENNReal.ofReal_mul hδ.le]
  exact ENNReal.ofReal_le_ofReal hreal

/-- **Predictable weights.** Period `t`'s adapted event has probability at most `ρ^(w t)` given the history, with `w t`
fixed before period `t`'s coins: every period's event holds with `Σ_t w_t ≥ M` with probability at most `ρ^M`. -/
theorem prCoin_forall_sum_le_pow (E : (t : Fin T) → ((i : Fin T) → Ω i) → Prop)
    (w : (t : Fin T) → ((i : Fin T) → Ω i) → ℕ) (hE : Adapted E) (hw : Predictable w) {ρ : ℝ≥0∞} (hρ : ρ ≤ 1)
    (h : ∀ t ω, prCoin (fun a : Ω t => E t (Function.update ω t a)) ≤ ρ ^ w t ω) (M : ℕ) :
    prCoin (fun ω => (∀ t, E t ω) ∧ M ≤ ∑ t, w t ω) ≤ ρ ^ M := by
  refine (prCoin_mono fun ω hω => ⟨hω.1, ?_⟩).trans
    (prCoin_forall_prod_le E (fun t ω => ρ ^ w t ω) hE (fun t ω ω' hω => by simp only [hw t ω ω' hω]) h (ρ ^ M))
  rw [Finset.prod_pow_eq_pow_sum]
  exact pow_le_pow_of_le_one zero_le hρ hω.2

open Classical in
/-- **Accepted and short, shortness predictable.** `p` bounds "accepted and short" in each period given the history,
and whether period `t` is short is fixed before its coins: every period is accepted with at least `M` short with
probability at most `p^M`. -/
theorem prCoin_forall_count_le_pow (A S : (t : Fin T) → ((i : Fin T) → Ω i) → Prop) (hA : Adapted A)
    (hS : Predictable S) {p : ℝ≥0∞}
    (h : ∀ t ω, prCoin (fun a : Ω t => A t (Function.update ω t a) ∧ S t (Function.update ω t a)) ≤ p) (M : ℕ) :
    prCoin (fun ω => (∀ t, A t ω) ∧ M ≤ (Finset.univ.filter fun t => S t ω).card) ≤ p ^ M := by
  rcases le_or_gt p 1 with hp | hp
  swap
  · exact (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans
      ((by simp [prCoin, avg_const] : prCoin (fun _ : (i : Fin T) → Ω i => True) = 1).trans_le (one_le_pow₀ hp.le))
  have key := prCoin_forall_sum_le_pow A (fun t ω => if S t ω then 1 else 0) hA
    (fun t ω ω' hω => by simp only [hS t ω ω' hω]) hp (fun t ω => ?_) M
  · simpa only [Finset.card_filter] using key
  · have hSu : ∀ a, S t (Function.update ω t a) = S t ω := fun a =>
      hS t _ _ fun i hi => Function.update_of_ne (ne_of_lt hi) _ _
    by_cases hs : S t ω
    · simp only [hs, ↓reduceIte, pow_one]
      refine (prCoin_mono fun a ha => ⟨ha, ?_⟩).trans (h t ω)
      rw [hSu a]
      exact hs
    · simp only [hs, ↓reduceIte, pow_zero]
      exact (prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const])

end FlockSoundness.Audit.Periods
