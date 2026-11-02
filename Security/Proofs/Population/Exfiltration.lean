import Definitions.Population.Exfiltration

/-!
# Exfiltration before detection: the stopping-time bounds

The game is `Definitions.Population.Exfiltration`'s: an adaptive prover, windows audited with live coins, each window
leaking what its incorrect units' cones carry, and the deployment halting at the first failure.

* `tail_cum`, **leakage through the first detection**: under detection pressure `κ`, with every window leaking at most
  `ℓ`, the probability that what leaves up to and including the first detection reaches `x` is at most
  `exp (-κ (x - ℓ))`, for every adaptive strategy and every horizon. So with probability at least `1 - δ` at most
  `ℓ + ln (1/δ) / κ` leaves.
* `tail_cumPrev`, **when a caught window's egress is withheld**: at most `exp (-κ x)`, and no per-window term.
* `ex_cumUpTo`, **the rate form**: before the `j`-th detection at most `j (ℓ + 1/κ)` leaves on average, so each failure a
  deployment tolerates costs at most `ℓ + 1/κ`.
* `acceptProb_le_exp`: a draw that includes the units independently, unit `u` caught with probability `a u`, lets an
  incorrect set `B` escape with probability at most `exp (-Σ_{u ∈ B} a u)`; with `a u ≥ κ E(u)` that is detection
  pressure for the leak `Σ_{u ∈ B} E(u)`.
-/

namespace Population.Exfil

open Finset

variable {Ω : Type} [Fintype Ω] {μ : Ω → ℝ}

/-! ## Expectations over coin paths -/

omit [Fintype Ω] in
theorem pathWeight_cons (μ : Ω → ℝ) {n : ℕ} (c : Ω) (cs : Fin n → Ω) :
    pathWeight μ (Fin.cons c cs : Fin (n + 1) → Ω) = μ c * pathWeight μ cs := by
  simp [pathWeight, Fin.prod_univ_succ]

omit [Fintype Ω] in
theorem pathWeight_nonneg (hμ : ∀ c, 0 ≤ μ c) {n : ℕ} (cs : Fin n → Ω) : 0 ≤ pathWeight μ cs :=
  prod_nonneg fun i _ => hμ (cs i)

theorem ex_zero (μ : Ω → ℝ) (F : (Fin 0 → Ω) → ℝ) (cs₀ : Fin 0 → Ω) : ex μ 0 F = F cs₀ := by
  rw [ex, Fintype.sum_unique, Subsingleton.elim default cs₀]
  simp [pathWeight]

/-- The first window's coin splits off: an expectation over `n + 1` windows is the first coin's average of the
expectation over the remaining `n`. -/
theorem ex_succ (μ : Ω → ℝ) {n : ℕ} (F : (Fin (n + 1) → Ω) → ℝ) :
    ex μ (n + 1) F = ∑ c, μ c * ex μ n fun cs => F (Fin.cons c cs) := by
  have e : ∀ p : Ω × (Fin n → Ω), (Fin.consEquiv fun _ : Fin (n + 1) => Ω) p = Fin.cons p.1 p.2 :=
    fun _ => rfl
  simp only [ex]
  rw [← (Fin.consEquiv fun _ : Fin (n + 1) => Ω).sum_comp, Fintype.sum_prod_type]
  simp only [e, pathWeight_cons, mul_sum, mul_assoc]

theorem ex_mono (hμ : ∀ c, 0 ≤ μ c) {n : ℕ} {F G : (Fin n → Ω) → ℝ} (h : ∀ cs, F cs ≤ G cs) :
    ex μ n F ≤ ex μ n G :=
  sum_le_sum fun cs _ => mul_le_mul_of_nonneg_left (h cs) (pathWeight_nonneg hμ cs)

theorem ex_const (hμ : IsDist μ) (n : ℕ) (a : ℝ) : ex μ n (fun _ => a) = a := by
  induction n with
  | zero => exact ex_zero μ _ Fin.elim0
  | succ n ih => rw [ex_succ]; simp [ih, ← sum_mul, hμ.2]

theorem ex_add_const (hμ : IsDist μ) {n : ℕ} (a : ℝ) (F : (Fin n → Ω) → ℝ) :
    ex μ n (fun cs => a + F cs) = a + ex μ n F := by
  have h := ex_const hμ n a
  simp only [ex, mul_add, sum_add_distrib] at h ⊢
  rw [h]

theorem prob_congr {n : ℕ} {E E' : (Fin n → Ω) → Prop} (h : ∀ cs, E cs ↔ E' cs) : prob μ n E = prob μ n E' := by
  rw [show E = E' from funext fun cs => propext (h cs)]

theorem prob_nonneg (hμ : ∀ c, 0 ≤ μ c) (n : ℕ) (E : (Fin n → Ω) → Prop) : 0 ≤ prob μ n E :=
  sum_nonneg fun cs _ => mul_nonneg (pathWeight_nonneg hμ cs) (by dsimp only; split_ifs <;> norm_num)

theorem prob_le_one (hμ : IsDist μ) (n : ℕ) (E : (Fin n → Ω) → Prop) : prob μ n E ≤ 1 := by
  unfold prob
  refine (ex_mono hμ.1 (G := fun _ => 1) fun cs => ?_).trans (ex_const hμ n 1).le
  split_ifs <;> norm_num

theorem prob_false (μ : Ω → ℝ) (n : ℕ) : prob μ n (fun _ => False) = 0 := by
  simp [prob, ex]

open Classical in
theorem prob_zero (μ : Ω → ℝ) (E : (Fin 0 → Ω) → Prop) (cs₀ : Fin 0 → Ω) :
    prob μ 0 E = if E cs₀ then 1 else 0 := by
  rw [prob, ex_zero μ _ cs₀]

theorem prob_succ (μ : Ω → ℝ) {n : ℕ} (E : (Fin (n + 1) → Ω) → Prop) :
    prob μ (n + 1) E = ∑ c, μ c * prob μ n fun cs => E (Fin.cons c cs) := by
  rw [prob, ex_succ]
  rfl

/-- The coins of a window that escape, by weight: the sum over the escaping coins. -/
theorem sum_ite_esc (s : Finset Ω) [DecidablePred (· ∈ s)] (a : ℝ) :
    ∑ c, μ c * (if c ∈ s then a else 0) = (∑ c ∈ s, μ c) * a := by
  simp [mul_ite, sum_mul]

/-! ## The stopping-time bounds -/

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

/-- **Leakage through the first detection is geometric**: under detection pressure `κ`, with every window leaking at
most `ℓ`, what leaves up to and including the first detection reaches `x` with probability at most
`exp (-κ (x - ℓ))`, whatever the prover does adaptively and however long the deployment runs. -/
theorem tail_cum (hμ : IsDist μ) {κ ℓ : ℝ} (hκ : 0 ≤ κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
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
    · exact (prob_le_one hμ _ _).trans (one_le x (hx.trans (hL h).2))
    · replace hx := not_le.1 hx
      rw [prob_succ]
      simp only [cum_cons]
      calc ∑ c, μ c * prob μ n (fun cs => x ≤ A.leak h + if c ∈ A.esc h then A.cum n (c :: h) cs else 0)
          ≤ ∑ c, μ c * (if c ∈ A.esc h then Real.exp (-(κ * (x - A.leak h - ℓ))) else 0) := by
            refine sum_le_sum fun c _ => mul_le_mul_of_nonneg_left ?_ (hμ.1 c)
            split_ifs with hc
            · refine le_of_eq_of_le (prob_congr fun cs => ?_) (ih (c :: h) (x - A.leak h))
              constructor <;> intro <;> linarith
            · rw [prob_congr (E' := fun _ => False) fun cs => by simp only [add_zero, iff_false, not_le]; exact hx,
                prob_false]
        _ = (∑ c ∈ A.esc h, μ c) * Real.exp (-(κ * (x - A.leak h - ℓ))) := sum_ite_esc _ _
        _ ≤ Real.exp (-(κ * A.leak h)) * Real.exp (-(κ * (x - A.leak h - ℓ))) :=
            mul_le_mul_of_nonneg_right (hP h) (Real.exp_pos _).le
        _ = Real.exp (-(κ * (x - ℓ))) := by rw [← Real.exp_add]; ring_nf

/-- **When a caught window's egress is withheld**, what leaves before the first detection reaches `x` with probability
at most `exp (-κ x)`: the per-window term goes. -/
theorem tail_cumPrev (hμ : IsDist μ) {κ : ℝ} (hκ : 0 ≤ κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
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
    · exact (prob_le_one hμ _ _).trans (one_le x hx)
    · replace hx := not_le.1 hx
      rw [prob_succ]
      simp only [cumPrev_cons]
      calc ∑ c, μ c * prob μ n (fun cs => x ≤ if c ∈ A.esc h then A.leak h + A.cumPrev n (c :: h) cs else 0)
          ≤ ∑ c, μ c * (if c ∈ A.esc h then Real.exp (-(κ * (x - A.leak h))) else 0) := by
            refine sum_le_sum fun c _ => mul_le_mul_of_nonneg_left ?_ (hμ.1 c)
            split_ifs with hc
            · refine le_of_eq_of_le (prob_congr fun cs => ?_) (ih (c :: h) (x - A.leak h))
              constructor <;> intro <;> linarith
            · rw [prob_congr (E' := fun _ => False) fun cs => by simp only [iff_false, not_le]; exact hx,
                prob_false]
        _ = (∑ c ∈ A.esc h, μ c) * Real.exp (-(κ * (x - A.leak h))) := sum_ite_esc _ _
        _ ≤ Real.exp (-(κ * A.leak h)) * Real.exp (-(κ * (x - A.leak h))) :=
            mul_le_mul_of_nonneg_right (hP h) (Real.exp_pos _).le
        _ = Real.exp (-(κ * x)) := by rw [← Real.exp_add]; ring_nf

/-- The scalar step of the rate form: a window that leaks `a ≤ ℓ` and escapes with probability `e ≤ exp (-κ a)` costs
at most what it saves, `a + e K ≤ K` for `K = ℓ + 1/κ`. -/
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

/-- **The rate form**: under detection pressure `κ > 0`, with every window leaking at most `ℓ`, what leaves before the
`j`-th detection is at most `j (ℓ + 1/κ)` on average, for every adaptive strategy and horizon. A deployment that
halts at the first failure (`j = 1`) gives up at most `ℓ + 1/κ`; one that tolerates failures pays that much for each. -/
theorem ex_cumUpTo (hμ : IsDist μ) {κ ℓ : ℝ} (hκ : 0 < κ) {A : Strategy Ω} (hP : DetectionPressure μ κ A)
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
      rw [ex_const hμ]
    | succ j =>
      rw [ex_succ]
      simp only [cumUpTo_cons]
      have hstep : ∀ c, ex μ n (fun cs => A.leak h + if c ∈ A.esc h then A.cumUpTo n (j + 1) (c :: h) cs
          else A.cumUpTo n j (c :: h) cs) ≤ A.leak h + j * (ℓ + 1 / κ) +
            (if c ∈ A.esc h then ℓ + 1 / κ else 0) := by
        intro c
        rw [ex_add_const hμ]
        split_ifs with hc
        · have := ih (j + 1) (c :: h); push_cast at this; linarith
        · have := ih j (c :: h); linarith
      calc ∑ c, μ c * ex μ n (fun cs => A.leak h + if c ∈ A.esc h then A.cumUpTo n (j + 1) (c :: h) cs
            else A.cumUpTo n j (c :: h) cs)
          ≤ ∑ c, μ c * (A.leak h + j * (ℓ + 1 / κ) + (if c ∈ A.esc h then ℓ + 1 / κ else 0)) :=
            sum_le_sum fun c _ => mul_le_mul_of_nonneg_left (hstep c) (hμ.1 c)
        _ = A.leak h + j * (ℓ + 1 / κ) + (∑ c ∈ A.esc h, μ c) * (ℓ + 1 / κ) := by
            rw [← sum_ite_esc]
            simp only [mul_add, sum_add_distrib, ← sum_mul, hμ.2, one_mul]
        _ ≤ ((j + 1 : ℕ) : ℝ) * (ℓ + 1 / κ) := by
            have := leak_add_esc_le hκ (hL h).1 (hL h).2 (hP h)
            push_cast
            linarith

/-! ## Where detection pressure comes from -/

/-- **A draw that includes units independently**: if unit `u` is caught with probability `a u ∈ [0, 1]`, independently,
the incorrect set `B` escapes with probability `Π_{u ∈ B} (1 - a u) ≤ exp (-Σ_{u ∈ B} a u)`. With
`a u ≥ κ E(u)` that is detection pressure `κ` for the leak `Σ_{u ∈ B} E(u)`. -/
theorem acceptProb_le_exp {U : Type} (B : Finset U) (a : U → ℝ) (ha : ∀ u ∈ B, 0 ≤ a u ∧ a u ≤ 1) :
    ∏ u ∈ B, (1 - a u) ≤ Real.exp (-∑ u ∈ B, a u) := by
  rw [← sum_neg_distrib, Real.exp_sum]
  refine prod_le_prod₀ (fun u hu => by linarith [(ha u hu).2]) fun u _ => ?_
  linarith [Real.add_one_le_exp (-a u)]

/-- **`k` independent draws**: an incorrect set each draw catches with probability `d` escapes `k` draws with
probability `(1 - d) ^ k ≤ exp (-k d)`. -/
theorem pow_acceptProb_le_exp {d : ℝ} (hd1 : d ≤ 1) (k : ℕ) : (1 - d) ^ k ≤ Real.exp (-(k * d)) := by
  rw [show -(k * d) = k * (-d) by ring, Real.exp_nat_mul]
  exact pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-d)]) k

end Population.Exfil
