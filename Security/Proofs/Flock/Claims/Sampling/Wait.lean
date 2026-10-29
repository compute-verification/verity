import Proofs.Flock.Claims.Sampling.Binding

/-!
# Waiting for accepted reruns: the truncation and the union over wrong sets

The sampling extractor waits for `k` accepted reruns among `K` fresh draws (`waitAccC`) and runs the session's
extraction at each, on coins of its own. At a tape where a run accepts with probability `a`, this file bounds the
probability that the wait comes up short, or that for some set `B` among `S` every kept rerun is bad at its own coins
(`bad B`), by `2k/K + e^{−k/8} + |S|^{1/k}·y`, when each `bad B` has `Pr[acc ∧ bad B] ≤ y`.

* `waitAccC_eval`, `waitAccC_eval_nil`: the wait keeps the first `k` accepted draws (`accList`), each with what it
  showed.
* `W`, `W_eq`, `Pr_allBad`: given the draws, the kept draws are all bad (`AllBad`) with probability the product of
  their own probabilities, `0` when fewer than `k` were kept.
* `EW`, `EW_succ`, `EW_le`: over the draws, that averages at most `(g/a)^k`, `g = Pr[acc ∧ bad]`; one draw at a time,
  `EW(k+1, K+1) = g·EW(k, K) + (1 − a)·EW(k+1, K)`.
* `short_pow`, `short_le`: the wait is short with probability at most `2^k·(1 − a/2)^K`, and `a` times that is at most
  `2k/K + e^{−k/8}` (`log 2 < 7/8`).
* `wait_bad_le`: **the bound**, through a union over `S` capped at `1`: `a·min(1, |S|·(y/a)^k) ≤ |S|^{1/k}·y`.
-/

namespace Claims.Sampling

open Claims Finset

/-! ## Uniform averages -/

theorem Pr_prod_eq_Ex {α β : Type} [Fintype α] [Fintype β] (p : α × β → Prop) :
    Pr p = Ex fun a => Pr fun b => p (a, b) := by
  classical
  rw [← Ex_ind p, Ex_prod]
  congr 1
  funext a
  exact Ex_ind _

theorem Pr_not {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : Pr (fun a => ¬ p a) = 1 - Pr p := by
  classical
  rw [← Ex_ind (fun a => ¬ p a)]
  have : (fun a => if ¬ p a then (1 : ℝ) else 0) = fun a => if p a then 0 else 1 :=
    funext fun a => by by_cases h : p a <;> simp [h]
  rw [this, Ex_ite]
  ring

theorem Pr_true {α : Type} [Fintype α] [Nonempty α] : Pr (fun _ : α => True) = 1 := by
  have h := Pr_not (α := α) fun _ => False
  rw [Pr_false, sub_zero] at h
  exact (Pr_congr fun _ => ⟨fun _ => not_false, fun _ => trivial⟩).trans h

/-- **Independent coordinates, one event each**: `Pr[∀ i, q_i(ρ_i)] = ∏ Pr[q_i]`. -/
theorem Pr_pi_dep {δ : Type} [Fintype δ] (m : ℕ) (q : Fin m → δ → Prop) :
    Pr (fun ρ : Fin m → δ => ∀ i, q i (ρ i)) = ∏ i, Pr (q i) := by
  classical
  rw [Pr_eq_sum (fun ρ : Fin m → δ => ∀ i, q i (ρ i)), Fintype.card_fun, Fintype.card_fin]
  simp only [Pr_eq_sum (q _)]
  rw [Finset.prod_div_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  push_cast
  congr 1
  rw [Finset.prod_univ_sum]
  refine Finset.sum_congr (by simp) fun ρ _ => ?_
  rw [Finset.prod_boole]
  simp

/-- **The union bound** over a finite set of events. -/
theorem Pr_exists_le {α ι : Type} [Fintype α] (S : Finset ι) (p : ι → α → Prop) :
    Pr (fun a => ∃ i ∈ S, p i a) ≤ ∑ i ∈ S, Pr (p i) := by
  classical
  induction S using Finset.induction_on with
  | empty => simp [Pr_false]
  | insert i S hi ih =>
    rw [Finset.sum_insert hi]
    calc Pr (fun a => ∃ j ∈ insert i S, p j a)
        ≤ Pr (fun a => p i a ∨ ∃ j ∈ S, p j a) := Pr_mono fun a h => by
          obtain ⟨j, hj, hp⟩ := h
          rcases Finset.mem_insert.1 hj with rfl | hj
          · exact Or.inl hp
          · exact Or.inr ⟨j, hj, hp⟩
      _ ≤ Pr (p i) + Pr (fun a => ∃ j ∈ S, p j a) := Pr_union_le _ _
      _ ≤ _ := add_le_add le_rfl ih

/-- `∏_{j < k} f(l_j)`, `1` past `l`'s end, is the product over `l` when `l` has at most `k` entries. -/
theorem prod_getElem? {R : Type} (b : R → ℝ) :
    ∀ (k : ℕ) (l : List R), l.length ≤ k → ∏ j : Fin k, (l[j.val]?).elim 1 b = (l.map b).prod
  | 0, l, h => by simp [List.length_eq_zero_iff.1 (Nat.le_zero.1 h)]
  | k + 1, [], _ => by simp
  | k + 1, r :: l, h => by
    rw [Fin.prod_univ_succ]
    simp only [Fin.val_zero, List.getElem?_cons_zero, Option.elim_some, Fin.val_succ, List.getElem?_cons_succ,
      List.map_cons, List.prod_cons]
    rw [prod_getElem? b k l (by simp only [List.length_cons] at h; omega)]

/-! ## The wait keeps the first `k` accepted draws -/

/-- The first `k` accepted draws among `rs`. -/
def accList {R : Type} (acc : R → Prop) [DecidablePred acc] (k : ℕ) (rs : List R) : List R :=
  (rs.filter fun r => decide (acc r)).take k

theorem accList_length_le {R : Type} (acc : R → Prop) [DecidablePred acc] (k : ℕ) (rs : List R) :
    (accList acc k rs).length ≤ k :=
  List.length_take_le _ _

theorem waitAccC_eval {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List (R × Seen)), acc.length ≤ k →
      ((waitAccC k rs acc).eval H run).1 =
        acc ++ ((rs.map fun r => (r, (run r).seen)).filter fun p => decide (p.2.out = .acc)).take (k - acc.length) := by
  intro rs
  induction rs with
  | nil => intro acc _; simp [waitAccC, RP.eval]
  | cons r rs ih =>
    intro acc hlen
    by_cases hk : k ≤ acc.length
    · have : k - acc.length = 0 := by omega
      simp [waitAccC, hk, RP.eval, this]
    · simp only [waitAccC, hk, ↓reduceIte, RP.eval, List.map_cons, List.filter_cons]
      by_cases ha : (run r).seen.out = .acc
      · simp only [ha, ↓reduceIte, decide_true]
        rw [ih _ (by rw [List.length_append, List.length_singleton]; omega)]
        have : k - acc.length = (k - (acc ++ [(r, (run r).seen)]).length) + 1 := by
          rw [List.length_append, List.length_singleton]; omega
        rw [this, List.take_succ_cons, List.append_assoc, List.singleton_append]
      · simp only [ha, ↓reduceIte, decide_false, Bool.false_eq_true]
        exact ih _ hlen

/-- **The wait keeps the first `k` accepted draws**, each with what it showed. -/
theorem waitAccC_eval_nil {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (rs : List R) :
    ((waitAccC k rs []).eval H run).1 =
      (accList (fun r => (run r).out = .acc) k rs).map fun r => (r, (run r).seen) := by
  rw [waitAccC_eval H run k rs [] (Nat.zero_le _), List.nil_append, List.length_nil, Nat.sub_zero, accList,
    List.filter_map, List.map_take]
  rfl

/-! ## Every kept draw bad: one wait's weight -/

section Weight

variable {R : Type} (acc : R → Prop) [DecidablePred acc]

/-- The kept draws' weights `b`, multiplied: `0` when fewer than `k` were kept. -/
noncomputable def W (b : R → ℝ) : ℕ → List R → ℝ
  | 0, _ => 1
  | _ + 1, [] => 0
  | k + 1, r :: rs => if acc r then b r * W b k rs else W b (k + 1) rs

theorem W_eq (b : R → ℝ) (rs : List R) : ∀ k : ℕ,
    W acc b k rs = if (accList acc k rs).length = k then ((accList acc k rs).map b).prod else 0 := by
  induction rs with
  | nil => intro k; cases k <;> simp [W, accList]
  | cons r rs ih =>
    intro k
    cases k with
    | zero => simp [W, accList]
    | succ k =>
      by_cases h : acc r
      · simp only [W, h, ↓reduceIte, accList, List.filter_cons, decide_true, List.take_succ_cons,
          List.length_cons, List.map_cons, List.prod_cons, Nat.add_right_cancel_iff]
        rw [ih k]
        simp only [accList]
        by_cases hc : (List.take k (List.filter (fun r => decide (acc r)) rs)).length = k <;> simp [hc]
      · simp only [W, h, ↓reduceIte, accList, List.filter_cons, decide_false, Bool.false_eq_true]
        exact ih (k + 1)

variable {E : Type} [Fintype E] [Nonempty E]

/-- Every kept draw is bad at its own coins, and `k` were kept. -/
def AllBad (k : ℕ) (bad : R → E → Prop) (rs : List R) (c : Fin k → E) : Prop :=
  (accList acc k rs).length = k ∧ ∀ (j : Fin k) (r : R), (accList acc k rs)[j.val]? = some r → bad r (c j)

/-- **Given the draws**, the kept draws are all bad with probability the product of their own probabilities. -/
theorem Pr_allBad (k : ℕ) (bad : R → E → Prop) (rs : List R) :
    Pr (fun c : Fin k → E => AllBad acc k bad rs c) = W acc (fun r => Pr (bad r)) k rs := by
  rw [W_eq]
  split_ifs with h
  · have e : ∀ c : Fin k → E, AllBad acc k bad rs c ↔
        ∀ j, (fun (j : Fin k) (e : E) => ∀ r, (accList acc k rs)[j.val]? = some r → bad r e) j (c j) :=
      fun c => ⟨fun h' => h'.2, fun h' => ⟨h, h'⟩⟩
    rw [Pr_congr e]
    refine (Pr_pi_dep k fun (j : Fin k) (e : E) => ∀ r, (accList acc k rs)[j.val]? = some r → bad r e).trans ?_
    rw [← prod_getElem? (fun r => Pr (bad r)) k _ (accList_length_le acc k rs)]
    refine Finset.prod_congr rfl fun j _ => ?_
    cases hj : (accList acc k rs)[j.val]? with
    | none => exact (Pr_congr fun e => ⟨fun _ => trivial, fun _ r hr => by simp at hr⟩).trans Pr_true
    | some r =>
      exact Pr_congr fun e => ⟨fun h' => h' r rfl, fun h' r' hr' => by cases hr'; exact h'⟩
  · exact (Pr_congr fun c => ⟨fun h' => h h'.1, False.elim⟩).trans Pr_false

variable [Fintype R] [Nonempty R]

/-- One wait's weight over `K` uniform draws. -/
noncomputable def EW (b : R → ℝ) (k K : ℕ) : ℝ := Ex fun σ : Fin K → R => W acc b k ((List.finRange K).map σ)

theorem EW_zero (b : R → ℝ) (K : ℕ) : EW acc b 0 K = 1 := by
  simp only [EW, W]
  exact Ex_const 1

theorem EW_succ_zero (b : R → ℝ) (k : ℕ) : EW acc b (k + 1) 0 = 0 := by
  simp only [EW, List.finRange_zero, List.map_nil, W]
  exact Ex_const 0

/-- **One draw at a time**: `EW(k+1, K+1) = g·EW(k, K) + (1 − a)·EW(k+1, K)`. -/
theorem EW_succ (b : R → ℝ) (k K : ℕ) : EW acc b (k + 1) (K + 1) =
    Ex (fun d => if acc d then b d else 0) * EW acc b k K + (1 - Pr acc) * EW acc b (k + 1) K := by
  let F : R → (Fin K → R) → ℝ := fun d σ =>
    if acc d then b d * W acc b k ((List.finRange K).map σ) else W acc b (k + 1) ((List.finRange K).map σ)
  have hstep : ∀ ρ : Fin (K + 1) → R,
      W acc b (k + 1) ((List.finRange (K + 1)).map ρ) = F (ρ 0) fun i => ρ i.succ := by
    intro ρ
    rw [List.finRange_succ, List.map_cons, List.map_map]
    rfl
  have h2 : ∀ d, (Ex fun σ : Fin K → R => F d σ) =
      (if acc d then b d else 0) * EW acc b k K + (if acc d then 0 else 1) * EW acc b (k + 1) K := by
    intro d
    by_cases h : acc d
    · simp only [F, h, ↓reduceIte, zero_mul, add_zero, EW]
      exact Ex_mul_left _ _
    · simp only [F, h, ↓reduceIte, zero_mul, zero_add, one_mul, EW]
  unfold EW
  rw [congrArg Ex (funext hstep), Ex_cons]
  rw [congrArg Ex (funext h2), Ex_add, Ex_mul_right, Ex_mul_right, Ex_ite]
  simp only [EW]
  ring

/-- **One wait's weight averages at most `(g/a)^k`**, `g = 𝔼[1[acc]·b]`, `a = Pr[acc] > 0`. -/
theorem EW_le {b : R → ℝ} (hb : ∀ r, 0 ≤ b r) (ha : 0 < Pr acc) :
    ∀ K k, EW acc b k K ≤ (Ex (fun d => if acc d then b d else 0) / Pr acc) ^ k := by
  have hg : 0 ≤ Ex (fun d => if acc d then b d else 0) := Ex_nonneg fun d => by split_ifs; exacts [hb d, le_rfl]
  have ha1 : Pr acc ≤ 1 := Pr_le_one _
  intro K
  induction K with
  | zero =>
    intro k
    cases k with
    | zero => rw [EW_zero, pow_zero]
    | succ k => rw [EW_succ_zero]; positivity
  | succ K ih =>
    intro k
    cases k with
    | zero => rw [EW_zero, pow_zero]
    | succ k =>
      rw [EW_succ]
      set g := Ex (fun d => if acc d then b d else 0)
      set a := Pr acc
      set u := g / a with hu
      have hga : g = a * u := by rw [hu]; field_simp
      calc g * EW acc b k K + (1 - a) * EW acc b (k + 1) K
          ≤ g * u ^ k + (1 - a) * u ^ (k + 1) :=
            add_le_add (mul_le_mul_of_nonneg_left (ih k) hg) (mul_le_mul_of_nonneg_left (ih (k + 1)) (by linarith))
        _ = u ^ (k + 1) := by rw [hga]; ring

/-- **The wait is short** with probability at most `2^k·(1 − a/2)^K`. -/
theorem short_pow : ∀ K k, 1 - EW acc (fun _ => 1) k K ≤ 2 ^ k * (1 - Pr acc / 2) ^ K := by
  have ha0 : 0 ≤ Pr acc := Pr_nonneg _
  have ha1 : Pr acc ≤ 1 := Pr_le_one _
  have hg : Ex (fun d => if acc d then (1 : ℝ) else 0) = Pr acc := Ex_ind acc
  intro K
  induction K with
  | zero =>
    intro k
    cases k with
    | zero => rw [EW_zero]; norm_num
    | succ k =>
      rw [EW_succ_zero, pow_zero, mul_one, sub_zero]
      exact one_le_pow₀ (by norm_num)
  | succ K ih =>
    intro k
    cases k with
    | zero =>
      rw [EW_zero, sub_self]
      exact mul_nonneg (by positivity) (pow_nonneg (by linarith) _)
    | succ k =>
      rw [EW_succ, hg]
      set a := Pr acc
      have h1 := ih k
      have h2 := ih (k + 1)
      have hp : 0 ≤ (1 - a / 2) ^ K := pow_nonneg (by linarith) _
      calc 1 - (a * EW acc (fun _ => 1) k K + (1 - a) * EW acc (fun _ => 1) (k + 1) K)
          = a * (1 - EW acc (fun _ => 1) k K) + (1 - a) * (1 - EW acc (fun _ => 1) (k + 1) K) := by ring
        _ ≤ a * (2 ^ k * (1 - a / 2) ^ K) + (1 - a) * (2 ^ (k + 1) * (1 - a / 2) ^ K) :=
            add_le_add (mul_le_mul_of_nonneg_left h1 ha0) (mul_le_mul_of_nonneg_left h2 (by linarith))
        _ = 2 ^ (k + 1) * (1 - a / 2) ^ (K + 1) := by ring

end Weight

/-- **The truncation term**: an acceptance probability `a` times a short wait's probability is at most
`2k/K + e^{−k/8}`. -/
theorem short_le {a S : ℝ} {k K : ℕ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hK : 0 < K) (hS1 : S ≤ 1)
    (hS : S ≤ 2 ^ k * (1 - a / 2) ^ K) : a * S ≤ 2 * k / K + Real.exp (-(k : ℝ) / 8) := by
  have hKr : (0 : ℝ) < K := by exact_mod_cast hK
  have he := (Real.exp_pos (-(k : ℝ) / 8)).le
  by_cases hcase : a ≤ 2 * k / K
  · have : a * S ≤ a := mul_le_of_le_one_right ha0 hS1
    linarith
  · push Not at hcase
    have haK : 2 * (k : ℝ) < a * K := by rwa [div_lt_iff₀ hKr] at hcase
    have hb : 2 ^ k * (1 - a / 2) ^ K ≤ Real.exp (-(k : ℝ) / 8) := by
      have h1 : (1 - a / 2) ^ K ≤ Real.exp (-a / 2) ^ K :=
        pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-a / 2)]) K
      have h2 : (2 : ℝ) ^ k = Real.exp (k * Real.log 2) := by
        rw [Real.exp_nat_mul, Real.exp_log (by norm_num)]
      have h3 : Real.exp (-a / 2) ^ K = Real.exp (K * (-a / 2)) := (Real.exp_nat_mul _ _).symm
      rw [h2]
      refine (mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le).trans ?_
      rw [h3, ← Real.exp_add]
      refine Real.exp_le_exp.2 ?_
      have hl : Real.log 2 < 7 / 8 := by have := Real.log_two_lt_d9; norm_num at this ⊢; linarith
      have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
      nlinarith
    have hb0 : 0 ≤ 2 ^ k * (1 - a / 2) ^ K := by
      have : 0 ≤ 1 - a / 2 := by linarith
      positivity
    have hk0 : (0 : ℝ) ≤ 2 * k / K := by positivity
    calc a * S ≤ a * (2 ^ k * (1 - a / 2) ^ K) := mul_le_mul_of_nonneg_left hS ha0
      _ ≤ 2 ^ k * (1 - a / 2) ^ K := mul_le_of_le_one_left hb0 ha1
      _ ≤ _ := by linarith

/-- `a·min(1, u^k) ≤ a·u` for `k ≥ 1`. -/
theorem mul_min_pow_le {a u : ℝ} {k : ℕ} (hk : 1 ≤ k) (ha : 0 ≤ a) (hu : 0 ≤ u) : a * min 1 (u ^ k) ≤ a * u := by
  refine mul_le_mul_of_nonneg_left ?_ ha
  rcases le_total u 1 with h | h
  · exact (min_le_right _ _).trans (pow_le_of_le_one hu h (by omega))
  · exact (min_le_left _ _).trans h

/-- `(C^{1/k}·y/a)^k = C·(y/a)^k` for `C ≥ 0` and `k ≥ 1`. -/
theorem rpow_inv_pow {C y a : ℝ} {k : ℕ} (hk : 1 ≤ k) (hC : 0 ≤ C) :
    (C ^ (1 / (k : ℝ)) * y / a) ^ k = C * (y / a) ^ k := by
  rw [mul_div_assoc, mul_pow, ← Real.rpow_natCast (C ^ (1 / (k : ℝ))), ← Real.rpow_mul hC,
    one_div_mul_cancel (by exact_mod_cast (show k ≠ 0 by omega)), Real.rpow_one]

/-! ## The bound -/

section Bound

variable {R E : Type} [Fintype R] [Nonempty R] [Fintype E] [Nonempty E] (acc : R → Prop) [DecidablePred acc]

omit [Nonempty R] [Nonempty E] in
/-- `g = 𝔼[1[acc]·Pr[bad]]` is `Pr[acc ∧ bad]` over a draw and its own coins. -/
theorem Ex_acc_Pr (bad : R → E → Prop) :
    Ex (fun d => if acc d then Pr (bad d) else 0) = Pr (fun p : R × E => acc p.1 ∧ bad p.1 p.2) := by
  rw [Pr_prod_eq_Ex]
  congr 1
  funext d
  by_cases h : acc d
  · simp only [h, ↓reduceIte, true_and]
  · simp only [h, ↓reduceIte, false_and]
    exact Pr_false.symm

/-- **Waiting for `k` accepted draws among `K`**: at acceptance probability `a`, `a` times the probability that the
wait is short or that, for some `B ∈ S`, every kept draw is bad at its own coins (`bad B`) is at most
`2k/K + e^{−k/8} + |S|^{1/k}·y`, when each `Pr[acc ∧ bad B] ≤ y`. -/
theorem wait_bad_le {ι : Type} (k K : ℕ) (hK : 0 < K) (hk : 1 ≤ k) (S : Finset ι) (bad : ι → R → E → Prop) {y : ℝ}
    (hy0 : 0 ≤ y) (hy : ∀ B ∈ S, Pr (fun p : R × E => acc p.1 ∧ bad B p.1 p.2) ≤ y) :
    Pr acc * Pr (fun c : (Fin K → R) × (Fin k → E) =>
      (accList acc k ((List.finRange K).map c.1)).length < k ∨
        ∃ B ∈ S, AllBad acc k (bad B) ((List.finRange K).map c.1) c.2) ≤
      2 * k / K + Real.exp (-(k : ℝ) / 8) + (S.card : ℝ) ^ (1 / (k : ℝ)) * y := by
  set a := Pr acc with ha
  have ha0 : 0 ≤ a := Pr_nonneg _
  have ha1 : a ≤ 1 := Pr_le_one _
  have hC0 : (0 : ℝ) ≤ (S.card : ℝ) ^ (1 / (k : ℝ)) := by positivity
  have hrest : 0 ≤ 2 * (k : ℝ) / K + Real.exp (-(k : ℝ) / 8) + (S.card : ℝ) ^ (1 / (k : ℝ)) * y := by
    have := Real.exp_pos (-(k : ℝ) / 8)
    positivity
  rcases ha0.eq_or_lt with h0 | hpos
  · rw [← h0, zero_mul]; exact hrest
  set short := Pr (fun σ : Fin K → R => (accList acc k ((List.finRange K).map σ)).length < k)
  have hshort : short = 1 - EW acc (fun _ => 1) k K := by
    classical
    have e : ∀ σ : Fin K → R, (accList acc k ((List.finRange K).map σ)).length < k ↔
        ¬ (accList acc k ((List.finRange K).map σ)).length = k := fun σ => by
      have := accList_length_le acc k ((List.finRange K).map σ); omega
    show Pr (fun σ : Fin K → R => (accList acc k ((List.finRange K).map σ)).length < k) = _
    rw [Pr_congr e, Pr_not]
    congr 1
    rw [← Ex_ind]
    unfold EW
    congr 1
    funext σ
    rw [W_eq]
    split_ifs <;> simp
  have hS1 : short ≤ 1 := Pr_le_one _
  have hSt := short_le ha0 ha1 hK hS1 (hshort ▸ short_pow acc K k)
  have hun : Pr (fun c : (Fin K → R) × (Fin k → E) => ∃ B ∈ S, AllBad acc k (bad B) ((List.finRange K).map c.1) c.2) ≤
      min 1 ((S.card : ℝ) * (y / a) ^ k) := by
    refine le_min (Pr_le_one _) ((Pr_exists_le S _).trans ?_)
    rw [← nsmul_eq_mul, ← Finset.sum_const]
    refine Finset.sum_le_sum fun B hB => ?_
    rw [Pr_prod_eq_Ex]
    simp only [Pr_allBad]
    refine (EW_le acc (fun r => Pr_nonneg _) hpos K k).trans ?_
    refine pow_le_pow_left₀ (div_nonneg (Ex_nonneg fun d => by split_ifs; exacts [Pr_nonneg _, le_rfl]) ha0) ?_ k
    rw [Ex_acc_Pr]
    exact div_le_div_of_nonneg_right (hy B hB) ha0
  have hmin : a * min 1 ((S.card : ℝ) * (y / a) ^ k) ≤ (S.card : ℝ) ^ (1 / (k : ℝ)) * y := by
    rw [← rpow_inv_pow hk (Nat.cast_nonneg S.card)]
    refine (mul_min_pow_le hk ha0 (by positivity)).trans_eq ?_
    field_simp
  calc a * Pr (fun c : (Fin K → R) × (Fin k → E) =>
        (accList acc k ((List.finRange K).map c.1)).length < k ∨
          ∃ B ∈ S, AllBad acc k (bad B) ((List.finRange K).map c.1) c.2)
      ≤ a * (short + min 1 ((S.card : ℝ) * (y / a) ^ k)) := by
        refine mul_le_mul_of_nonneg_left ((Pr_union_le _ _).trans (add_le_add ?_ hun)) ha0
        exact le_of_eq (Pr_fst (β := Fin k → E) fun σ : Fin K → R =>
          (accList acc k ((List.finRange K).map σ)).length < k)
    _ = a * short + a * min 1 ((S.card : ℝ) * (y / a) ^ k) := by ring
    _ ≤ _ := by linarith

end Bound

end Claims.Sampling
