import Proofs.Flock.Soundness.Audit.Work
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The cumulative harm budget over adaptive rounds

Rounds run `t = 1..T`, where `T` is a stopping time the prover chooses. In round `t` the prover
commits `X_t`, whose wrong units `B_t` are fixed before the round's fresh coins `ω_t` select the
draw `D_t`. Every drawn unit must get an accepting proof, and a failure rejects the session with no
redraw.

**The design condition** (`Law.Design L π`): `Pr[D ∩ B = ∅] ≤ exp(−Σ_{u ∈ B} π_u)` for every `B`
fixed before the coins. The Bernoulli (`bernoulli_design`, exact at rate `bernRate`:
`bernoulli_design_exact`), uniform-subset (`subset_design`), stratified (`stratified_design`) and
work (`work_design`) laws meet it.

**The model** (`Adaptive.Play`) is the run tree of one prover strategy, its own choices resolved: a
`round L B k` node is a round whose wrong units `B` and law `L` are fixed by the history before its
coin, `coin` nodes are the session's other coins, and `stop acc` ends the session. `Play n N` has at
most `N` rounds, and the prover may stop after any of them. `Drawn err` is the per-round drawn-unit
bound, `Pr[D_t ∩ B_t ≠ ∅ ∧ round t passes] ≤ err_t`, which `audit_drawn` gives with
`err_t = ε_ks,t + δ_link,t` (`HarmBudgetLife`).

**The theorem** (`harm_budget`): for harms `h ≥ 0` and any `λ ≥ 0` with `λ h_u ≤ π_u`, in
particular `λ = min_{h_u > 0} π_u / h_u` (`harmRate`, `harm_budget_rate`),
$$\Pr\Big[\text{accept} \wedge \sum_{t \le T} h(B_t) \ge H\Big] \le e^{-\lambda H} + \sum_t \mathrm{err}_t.$$
The proof is the supermartingale `exp(λ (Σ_{s<t} h(B_s) − H))` plus the error spent so far, stopped
at `T`, by induction on the tree: in each round the design condition pays for the harm the draw
missed, and `Drawn` for the harm it hit (`prEv_le_exp`). Markov is the base case.

**The count corollary** (`harm_count`, `harm_count_kstar`): under a Bernoulli law of rate `q`, the
number of wrong steps reaches `K` with probability at most `(1 − q)^K + Σ err`, and
`k*(p) = ln(1/(1−p)) / (−ln(1−q))` wrong steps with at most `1 − p + Σ err`. It is tight: one round
that accepts exactly when its draw misses `B` meets `(1 − q)^{|B|}` (`harm_count_tight`).
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n : ℕ}

/-- **The design condition** with unit weights `π`: every set `B` fixed before the coins escapes
with probability at most `exp(−Σ_{u ∈ B} π_u)`. -/
def Design (L : Law n) (π : Fin n → ℝ) : Prop :=
  ∀ B : Finset (Fin n), L.escape B ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, π u))

/-- The design condition pays for `exp(λ h(B))` when `λ h ≤ π`. -/
theorem Design.escape_mul_le {L : Law n} {π h : Fin n → ℝ} {lam : ℝ} (hL : L.Design π)
    (hlam : ∀ u, lam * h u ≤ π u) (B : Finset (Fin n)) :
    L.escape B * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u)) ≤ 1 := by
  have hsum : lam * ∑ u ∈ B, h u ≤ ∑ u ∈ B, π u := by
    rw [Finset.mul_sum]; exact Finset.sum_le_sum fun u _ => hlam u
  calc L.escape B * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u))
      ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, π u)) * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u)) :=
        mul_le_mul_left (hL B) _
    _ = ENNReal.ofReal (Real.exp (-∑ u ∈ B, π u + lam * ∑ u ∈ B, h u)) := by
        rw [Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
    _ ≤ ENNReal.ofReal (Real.exp 0) := ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 (by linarith))
    _ = 1 := by rw [Real.exp_zero, ENNReal.ofReal_one]

/-- `(b − a)/b ≤ exp(−a/b)`. -/
theorem natFrac_le_exp {a b : ℕ} (hab : a ≤ b) (hb : 0 < b) :
    ((b - a : ℕ) : ℝ≥0∞) / (b : ℝ≥0∞) ≤ ENNReal.ofReal (Real.exp (-((a : ℝ) / b))) := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  rw [← ENNReal.ofReal_natCast (b - a), ← ENNReal.ofReal_natCast b, ← ENNReal.ofReal_div_of_pos hb']
  refine ENNReal.ofReal_le_ofReal ?_
  rw [Nat.cast_sub hab, sub_div, div_self hb'.ne']
  exact Real.one_sub_le_exp_neg _

theorem bernFactor_zero_div (num den : ℕ) :
    bernFactor num den 0 = ((den - num : ℕ) : ℝ≥0∞) / den := by
  rw [bernFactor, mul_zero, zero_add]

/-- A Bernoulli draw of rate `q = num/den` meets the design condition with `π_u = q`. -/
theorem bernoulli_design {num den : ℕ} [NeZero den] (h : num ≤ den) :
    (bernoulli n num den).Design fun _ => (num : ℝ) / den := by
  intro B
  have hden : 0 < den := Nat.pos_of_ne_zero (NeZero.ne den)
  show (bernoulli n num den).escape B ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, (num : ℝ) / den))
  calc (bernoulli n num den).escape B = (((den - num : ℕ) : ℝ≥0∞) / den) ^ B.card := by
        rw [bernoulli_escape h, bernFactor_zero_div]
    _ ≤ ENNReal.ofReal (Real.exp (-((num : ℝ) / den))) ^ B.card :=
        pow_le_pow_left₀ zero_le (natFrac_le_exp h hden) _
    _ = ENNReal.ofReal (Real.exp (-∑ u ∈ B, (num : ℝ) / den)) := by
        rw [← ENNReal.ofReal_pow (Real.exp_pos _).le, ← Real.exp_nat_mul, Finset.sum_const, nsmul_eq_mul,
          mul_neg]

/-- A uniform `k`-subset of `n` meets the design condition with `π_u = k/n`. -/
theorem subset_design {k : ℕ} (hk : k ≤ n) (hn : 0 < n) : (subset n k hk).Design fun _ => (k : ℝ) / n := by
  intro B
  show (subset n k hk).escape B ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, (k : ℝ) / n))
  calc (subset n k hk).escape B = ((n - B.card).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := subset_escape hk B
    _ ≤ (((n - B.card : ℕ) : ℝ≥0∞) / (n : ℝ≥0∞)) ^ k := choose_div_le_pow' hk hn
    _ ≤ ENNReal.ofReal (Real.exp (-((B.card : ℝ) / n))) ^ k :=
        pow_le_pow_left₀ zero_le (natFrac_le_exp (by simpa using Finset.card_le_univ B) hn) _
    _ = ENNReal.ofReal (Real.exp (-∑ u ∈ B, (k : ℝ) / n)) := by
        rw [← ENNReal.ofReal_pow (Real.exp_pos _).le, ← Real.exp_nat_mul, Finset.sum_const, nsmul_eq_mul]
        congr 2; ring

/-- A stratified law meets the design condition with `π_u = k_s / n_s` for `u` in stratum `s`. -/
theorem stratified_design {m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) :
    (stratified σ k hk).Design fun u => (k (σ u) : ℝ) / (stratum σ (σ u)).card := by
  intro B
  show (stratified σ k hk).escape B ≤
    ENNReal.ofReal (Real.exp (-∑ u ∈ B, (k (σ u) : ℝ) / (stratum σ (σ u)).card))
  have hfac : ∀ s, (((stratum σ s).card - (B.filter (σ · = s)).card).choose (k s) : ℝ≥0∞) /
      ((stratum σ s).card.choose (k s) : ℝ≥0∞) ≤
      ENNReal.ofReal (Real.exp (-((B.filter (σ · = s)).card * ((k s : ℝ) / (stratum σ s).card)))) := by
    intro s
    have hm : (B.filter (σ · = s)).card ≤ (stratum σ s).card :=
      card_le_card (filter_subset_filter _ (subset_univ B))
    rcases Nat.eq_zero_or_pos (stratum σ s).card with h0 | hpos
    · rw [h0, Nat.cast_zero, div_zero, mul_zero, neg_zero, Real.exp_zero, ENNReal.ofReal_one]
      exact choose_div_le_one _ _ _
    · calc _ ≤ ((((stratum σ s).card - (B.filter (σ · = s)).card : ℕ) : ℝ≥0∞) /
              ((stratum σ s).card : ℝ≥0∞)) ^ k s :=
            choose_div_le_pow' (hk s) hpos
        _ ≤ ENNReal.ofReal (Real.exp (-(((B.filter (σ · = s)).card : ℝ) / (stratum σ s).card))) ^ k s :=
            pow_le_pow_left₀ zero_le (natFrac_le_exp hm hpos) _
        _ = _ := by
            rw [← ENNReal.ofReal_pow (Real.exp_pos _).le, ← Real.exp_nat_mul]
            congr 2; ring
  have hsum : ∑ u ∈ B, (k (σ u) : ℝ) / (stratum σ (σ u)).card =
      ∑ s, ((B.filter (σ · = s)).card * ((k s : ℝ) / (stratum σ s).card)) := by
    rw [← Finset.sum_fiberwise' B σ (fun s => (k s : ℝ) / (stratum σ s).card)]
    refine Finset.sum_congr rfl fun s _ => ?_
    rw [Finset.sum_const, nsmul_eq_mul]
  rw [stratified_escape, hsum, ← Finset.sum_neg_distrib, Real.exp_sum, ENNReal.ofReal_prod_of_nonneg]
  · exact Finset.prod_le_prod fun s _ => hfac s
  · exact fun s _ => (Real.exp_pos _).le

/-- The work law meets the design condition with `π_u = K w_u / W`. -/
theorem work_design {m : ℕ} (σ : Fin n → Fin m) (w f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < totalWork σ w) : (work σ w f K).Design fun u => (K : ℝ) * w (σ u) / totalWork σ w := by
  intro B
  show (work σ w f K).escape B ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, (K : ℝ) * w (σ u) / totalWork σ w))
  have hB : workOf σ w B ≤ totalWork σ w := Finset.sum_le_sum_of_subset (subset_univ B)
  have hsum : ∑ u ∈ B, (K : ℝ) * w (σ u) / totalWork σ w = K * ((workOf σ w B : ℝ) / totalWork σ w) := by
    rw [workOf, Nat.cast_sum, Finset.sum_div, Finset.mul_sum]
    exact Finset.sum_congr rfl fun u _ => mul_div_assoc _ _ _
  calc (work σ w f K).escape B
      ≤ (((totalWork σ w - workOf σ w B : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K :=
        work_escape_le σ w f K hf hW B
    _ ≤ ENNReal.ofReal (Real.exp (-((workOf σ w B : ℝ) / totalWork σ w))) ^ K :=
        pow_le_pow_left₀ zero_le (natFrac_le_exp hB hW) _
    _ = _ := by rw [← ENNReal.ofReal_pow (Real.exp_pos _).le, ← Real.exp_nat_mul, hsum, mul_neg]

/-- The Bernoulli escape rate per unit, `−ln(1 − q)` for `q = num/den`. -/
noncomputable def bernRate (num den : ℕ) : ℝ := -Real.log (((den - num : ℕ) : ℝ) / den)

theorem bernRate_nonneg {num den : ℕ} (h : num < den) : 0 ≤ bernRate num den := by
  have hden : (0 : ℝ) < den := Nat.cast_pos.2 ((Nat.zero_le num).trans_lt h)
  rw [bernRate, neg_nonneg]
  exact Real.log_nonpos (div_nonneg (Nat.cast_nonneg _) hden.le)
    ((div_le_one hden).2 (Nat.cast_le.2 (Nat.sub_le den num)))

theorem bernRate_pos {num den : ℕ} (h0 : 0 < num) (h : num < den) : 0 < bernRate num den := by
  have hden : (0 : ℝ) < den := Nat.cast_pos.2 ((Nat.zero_le num).trans_lt h)
  have hq0 : (0 : ℝ) < ((den - num : ℕ) : ℝ) / den := div_pos (Nat.cast_pos.2 (Nat.sub_pos_of_lt h)) hden
  have hq1 : ((den - num : ℕ) : ℝ) / den < 1 :=
    (div_lt_one hden).2 (Nat.cast_lt.2 (Nat.sub_lt (h0.trans h) h0))
  rw [bernRate]
  exact neg_pos.2 (Real.log_neg hq0 hq1)

theorem ofReal_exp_bernRate {num den : ℕ} (h : num < den) (k : ℕ) :
    ENNReal.ofReal (Real.exp (-((k : ℝ) * bernRate num den))) = bernFactor num den 0 ^ k := by
  have hden : (0 : ℝ) < den := Nat.cast_pos.2 ((Nat.zero_le num).trans_lt h)
  have hq : (0 : ℝ) < ((den - num : ℕ) : ℝ) / den := div_pos (Nat.cast_pos.2 (Nat.sub_pos_of_lt h)) hden
  rw [bernRate, mul_neg, neg_neg, Real.exp_nat_mul, Real.exp_log hq, ENNReal.ofReal_pow hq.le,
    ENNReal.ofReal_div_of_pos hden, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast, bernFactor_zero_div]

/-- A Bernoulli draw meets the design condition with equality at `π_u = −ln(1 − q)`. -/
theorem bernoulli_design_exact {num den : ℕ} [NeZero den] (h : num < den) :
    (bernoulli n num den).Design fun _ => bernRate num den := by
  intro B
  show (bernoulli n num den).escape B ≤ ENNReal.ofReal (Real.exp (-∑ u ∈ B, bernRate num den))
  refine le_of_eq ?_
  rw [bernoulli_escape h.le, Finset.sum_const, nsmul_eq_mul, ofReal_exp_bernRate h]

end Law

namespace Adaptive

/-- **The run tree of one prover strategy over adaptive rounds**, at most `N` of them. `round L B k`
is a round whose law `L` and wrong units `B` the history fixes before its coin; `coin` is any other
verifier coin; `stop acc` ends the session, accepting when `acc`. -/
inductive Play (n : ℕ) : ℕ → Type 1 where
  | stop {N : ℕ} (acc : Bool) : Play n N
  | coin {N : ℕ} (C : Type) [Fintype C] [Nonempty C] (k : C → Play n N) : Play n N
  | round {N : ℕ} (L : Law n) (B : Finset (Fin n)) (k : L.Ω → Play n N) : Play n (N + 1)

namespace Play

variable {n : ℕ}

open Classical in
/-- The probability that the session ends with `Ev acc (a + Σ_t h(B_t))`, the harm summed over
the rounds played. -/
noncomputable def prEv (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) : {N : ℕ} → Play n N → ℝ → ℝ≥0∞
  | _, .stop acc, a => if Ev acc a then 1 else 0
  | _, @Play.coin _ _ _ _ _ k, a => avg fun c => prEv h Ev (k c) a
  | _, .round _ B k, a => avg fun ω => prEv h Ev (k ω) (a + ∑ u ∈ B, h u)

/-- The probability that the current round passes: the session accepts or reaches the next round. -/
noncomputable def pass : {N : ℕ} → Play n N → ℝ≥0∞
  | _, .stop acc => if acc = true then 1 else 0
  | _, @Play.coin _ _ _ _ _ k => avg fun c => pass (k c)
  | _, .round _ _ _ => 1

/-- Every round's law satisfies `S`. -/
def Laws (S : Law n → Prop) : {N : ℕ} → Play n N → Prop
  | _, .stop _ => True
  | _, @Play.coin _ _ _ _ _ k => ∀ c, Laws S (k c)
  | _, .round L _ k => S L ∧ ∀ ω, Laws S (k ω)

/-- **The drawn-unit bound** of every round: its draw hits its wrong units and the round passes
with probability at most `err` of the number of rounds left after it. -/
def Drawn (err : ℕ → ℝ≥0∞) : {N : ℕ} → Play n N → Prop
  | _, .stop _ => True
  | _, @Play.coin _ _ _ _ _ k => ∀ c, Drawn err (k c)
  | _, @Play.round _ N L B k =>
      avg (fun ω => if Disjoint (L.draw ω) B then 0 else (k ω).pass) ≤ err N ∧ ∀ ω, Drawn err (k ω)

open Classical in
theorem prEv_stop (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) {N : ℕ} (acc : Bool) (a : ℝ) :
    (Play.stop acc : Play n N).prEv h Ev a = if Ev acc a then 1 else 0 := rfl

theorem prEv_coin (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) {N : ℕ} (C : Type) [Fintype C] [Nonempty C]
    (k : C → Play n N) (a : ℝ) : (Play.coin C k).prEv h Ev a = avg fun c => (k c).prEv h Ev a := rfl

theorem prEv_round (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) {N : ℕ} (L : Law n) (B : Finset (Fin n))
    (k : L.Ω → Play n N) (a : ℝ) :
    (Play.round L B k).prEv h Ev a = avg fun ω => (k ω).prEv h Ev (a + ∑ u ∈ B, h u) := rfl

theorem pass_stop {N : ℕ} (acc : Bool) : (Play.stop acc : Play n N).pass = if acc = true then 1 else 0 := rfl

theorem pass_coin {N : ℕ} (C : Type) [Fintype C] [Nonempty C] (k : C → Play n N) :
    (Play.coin C k).pass = avg fun c => (k c).pass := rfl

theorem pass_round {N : ℕ} (L : Law n) (B : Finset (Fin n)) (k : L.Ω → Play n N) :
    (Play.round L B k : Play n (N + 1)).pass = 1 := rfl

theorem Laws.mono {S S' : Law n → Prop} (hS : ∀ L, S L → S' L) :
    ∀ {N : ℕ} (P : Play n N), P.Laws S → P.Laws S'
  | _, .stop _, _ => trivial
  | _, @Play.coin _ _ _ _ _ k, hP => fun c => Laws.mono hS (k c) (hP c)
  | _, .round L _ k, hP => ⟨hS L hP.1, fun ω => Laws.mono hS (k ω) (hP.2 ω)⟩

theorem prEv_le_one (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) :
    ∀ {N : ℕ} (P : Play n N) (a : ℝ), P.prEv h Ev a ≤ 1
  | _, .stop _, _ => by rw [prEv_stop]; split_ifs <;> simp
  | _, @Play.coin _ _ _ _ _ k, a => by
      rw [prEv_coin]
      exact (avg_mono fun c => prEv_le_one h Ev (k c) a).trans (avg_const 1).le
  | _, .round _ _ k, _ => by
      rw [prEv_round]
      exact (avg_mono fun ω => prEv_le_one h Ev (k ω) _).trans (avg_const 1).le

/-- An event that needs acceptance needs the current round to pass. -/
theorem prEv_le_pass (h : Fin n → ℝ) (Ev : Bool → ℝ → Prop) (hEv : ∀ acc x, Ev acc x → acc = true) :
    ∀ {N : ℕ} (P : Play n N) (a : ℝ), P.prEv h Ev a ≤ P.pass
  | _, .stop acc, a => by
      rw [prEv_stop, pass_stop]
      by_cases hE : Ev acc a
      · exact (ite_eq_left hE).trans_le (ite_eq_left (hEv acc a hE)).symm.le
      · rw [ite_eq_right hE]
        exact zero_le
  | _, @Play.coin _ _ _ _ _ k, a => by
      rw [prEv_coin, pass_coin]
      exact avg_mono fun c => prEv_le_pass h Ev hEv (k c) a
  | _, .round _ _ _, a => by
      rw [pass_round]
      exact prEv_le_one h Ev _ a

end Play

open Play

theorem one_le_ofReal_exp {x : ℝ} (hx : 0 ≤ x) : 1 ≤ ENNReal.ofReal (Real.exp x) := by
  rw [← ENNReal.ofReal_one]
  exact ENNReal.ofReal_le_ofReal (Real.one_le_exp hx)

theorem avg_ite_zero {C : Type} [Fintype C] (D : C → Prop) [DecidablePred D] (x : ℝ≥0∞) :
    avg (fun c => if D c then x else 0) = prCoin D * x := by
  rw [← avg_ite_eq_prCoin, ← avg_mul_const]
  refine congrArg avg (funext fun c => ?_)
  by_cases hc : D c <;> simp [hc]

variable {n : ℕ}

/-- **The supermartingale step, stopped**: from accumulated harm `a`, the session accepts with harm
at least `H` with probability at most `exp(λ (a − H))` plus the error of the rounds left. -/
theorem prEv_le_exp {π h : Fin n → ℝ} {lam : ℝ} {err : ℕ → ℝ≥0∞} (H : ℝ) (hlam0 : 0 ≤ lam)
    (hlam : ∀ u, lam * h u ≤ π u) :
    ∀ {N : ℕ} (P : Play n N) (a : ℝ), P.Laws (fun L => L.Design π) → P.Drawn err →
      P.prEv h (fun acc x => acc = true ∧ H ≤ x) a ≤
        ENNReal.ofReal (Real.exp (lam * (a - H))) + ∑ m ∈ range N, err m
  | _, .stop acc, a, _, _ => by
      rw [prEv_stop]
      split_ifs with hE
      · exact le_add_right (one_le_ofReal_exp (mul_nonneg hlam0 (sub_nonneg.2 hE.2)))
      · exact zero_le
  | _, @Play.coin _ _ _ _ _ k, a, hL, hD => by
      rw [prEv_coin]
      exact (avg_mono fun c => prEv_le_exp H hlam0 hlam (k c) a (hL c) (hD c)).trans (avg_const _).le
  | _, @Play.round _ N L B k, a, hL, hD => by
      obtain ⟨hLd, hLk⟩ := hL
      obtain ⟨hDd, hDk⟩ := hD
      have step : ∀ ω, (k ω).prEv h (fun acc x => acc = true ∧ H ≤ x) (a + ∑ u ∈ B, h u) ≤
          ((if Disjoint (L.draw ω) B then
              ENNReal.ofReal (Real.exp (lam * (a - H))) * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u))
            else 0) +
            (if Disjoint (L.draw ω) B then 0 else (k ω).pass)) + ∑ m ∈ range N, err m := by
        intro ω
        by_cases hd : Disjoint (L.draw ω) B
        · rw [ite_eq_left hd, ite_eq_left hd, add_zero, ← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add,
            show lam * (a - H) + lam * ∑ u ∈ B, h u = lam * (a + ∑ u ∈ B, h u - H) by ring]
          exact prEv_le_exp H hlam0 hlam (k ω) _ (hLk ω) (hDk ω)
        · rw [ite_eq_right hd, ite_eq_right hd, zero_add]
          exact (prEv_le_pass h (fun acc x => acc = true ∧ H ≤ x) (fun _ _ hE => hE.1) (k ω) _).trans
            le_self_add
      rw [prEv_round, Finset.sum_range_succ]
      calc avg (fun ω => (k ω).prEv h (fun acc x => acc = true ∧ H ≤ x) (a + ∑ u ∈ B, h u))
          ≤ avg (fun ω => ((if Disjoint (L.draw ω) B then
                ENNReal.ofReal (Real.exp (lam * (a - H))) * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u))
              else 0) + (if Disjoint (L.draw ω) B then 0 else (k ω).pass)) + ∑ m ∈ range N, err m) :=
            avg_mono step
        _ = prCoin (fun ω => Disjoint (L.draw ω) B) *
                (ENNReal.ofReal (Real.exp (lam * (a - H))) * ENNReal.ofReal (Real.exp (lam * ∑ u ∈ B, h u))) +
              avg (fun ω => if Disjoint (L.draw ω) B then 0 else (k ω).pass) + ∑ m ∈ range N, err m := by
            rw [avg_add₃, avg_ite_zero, avg_const]
        _ ≤ ENNReal.ofReal (Real.exp (lam * (a - H))) + err N + ∑ m ∈ range N, err m := by
            refine add_le_add (add_le_add ?_ hDd) le_rfl
            rw [mul_left_comm]
            exact mul_le_of_le_one_right' (Law.Design.escape_mul_le hLd hlam B)
        _ = ENNReal.ofReal (Real.exp (lam * (a - H))) + (∑ m ∈ range N, err m + err N) := by
            rw [add_assoc, add_comm (err N)]

/-- **The cumulative harm budget.** Over at most `N` adaptive rounds whose laws meet the design
condition with weights `π` and whose drawn units fail with at most `err`, for harms with
`λ h ≤ π` and `λ ≥ 0`, the session accepts with total harm at least `H` with probability at most
`e^{−λH} + Σ_t err_t`. -/
theorem harm_budget {π h : Fin n → ℝ} {lam : ℝ} {err : ℕ → ℝ≥0∞} (hlam0 : 0 ≤ lam)
    (hlam : ∀ u, lam * h u ≤ π u) (H : ℝ) {N : ℕ} (P : Play n N) (hL : P.Laws fun L => L.Design π)
    (hD : P.Drawn err) :
    P.prEv h (fun acc x => acc = true ∧ H ≤ x) 0 ≤
      ENNReal.ofReal (Real.exp (-(lam * H))) + ∑ m ∈ range N, err m := by
  have key := prEv_le_exp H hlam0 hlam P 0 hL hD
  rwa [zero_sub, mul_neg] at key

open Classical in
/-- The budget's rate, `λ = min_{h_u > 0} π_u / h_u` (zero when no unit does harm). -/
noncomputable def harmRate (π h : Fin n → ℝ) : ℝ :=
  if hne : (univ.filter fun u => 0 < h u).Nonempty then
    (univ.filter fun u => 0 < h u).inf' hne fun u => π u / h u
  else 0

theorem harmRate_nonneg {π h : Fin n → ℝ} (hπ : ∀ u, 0 ≤ π u) (hh : ∀ u, 0 ≤ h u) :
    0 ≤ harmRate π h := by
  unfold harmRate
  split_ifs with hne
  · exact Finset.le_inf' hne _ fun u _ => div_nonneg (hπ u) (hh u)
  · exact le_rfl

theorem harmRate_mul_le {π h : Fin n → ℝ} (hπ : ∀ u, 0 ≤ π u) (hh : ∀ u, 0 ≤ h u) (u : Fin n) :
    harmRate π h * h u ≤ π u := by
  classical
  rcases (hh u).lt_or_eq with hu | hu
  · have hmem : u ∈ univ.filter fun u => 0 < h u := Finset.mem_filter.2 ⟨Finset.mem_univ u, hu⟩
    have hne : (univ.filter fun u => 0 < h u).Nonempty := ⟨u, hmem⟩
    rw [harmRate, dite_eq_left hne]
    exact (le_div_iff₀ hu).1 (Finset.inf'_le _ hmem)
  · rw [← hu, mul_zero]
    exact hπ u

/-- **The cumulative harm budget at `λ = min_{h_u > 0} π_u / h_u`.** -/
theorem harm_budget_rate {π h : Fin n → ℝ} {err : ℕ → ℝ≥0∞} (hπ : ∀ u, 0 ≤ π u) (hh : ∀ u, 0 ≤ h u)
    (H : ℝ) {N : ℕ} (P : Play n N) (hL : P.Laws fun L => L.Design π) (hD : P.Drawn err) :
    P.prEv h (fun acc x => acc = true ∧ H ≤ x) 0 ≤
      ENNReal.ofReal (Real.exp (-(harmRate π h * H))) + ∑ m ∈ range N, err m :=
  harm_budget (harmRate_nonneg hπ hh) (fun u => harmRate_mul_le hπ hh u) H P hL hD

/-- **The count budget under Bernoulli draws.** With every round drawing each unit independently at
rate `q = num/den`, the session accepts with at least `K` wrong steps in total with probability at
most `(1 − q)^K + Σ_t err_t`. -/
theorem harm_count {num den : ℕ} [NeZero den] (hq : num < den) {err : ℕ → ℝ≥0∞} (K : ℕ) {N : ℕ}
    (P : Play n N) (hL : P.Laws fun L => L = Law.bernoulli n num den) (hD : P.Drawn err) :
    P.prEv (fun _ => 1) (fun acc x => acc = true ∧ (K : ℝ) ≤ x) 0 ≤
      Law.bernFactor num den 0 ^ K + ∑ m ∈ range N, err m := by
  have key := harm_budget (π := fun _ => Law.bernRate num den) (h := fun _ => (1 : ℝ))
    (Law.bernRate_nonneg hq) (fun _ => (mul_one _).le) (K : ℝ) P
    (Play.Laws.mono (fun L (hL : L = Law.bernoulli n num den) => by
      subst hL; exact Law.bernoulli_design_exact hq) P hL) hD
  rwa [mul_comm (Law.bernRate num den) (K : ℝ), Law.ofReal_exp_bernRate hq] at key

/-- `k*(p) = ln(1/(1−p)) / (−ln(1−q))`: the number of wrong steps that a session accepted with
probability above `1 − p` (less the drawn-unit error) stays under. -/
noncomputable def kStar (num den : ℕ) (p : ℝ) : ℝ := Real.log (1 / (1 - p)) / Law.bernRate num den

/-- **The count corollary.** Under Bernoulli draws of rate `q > 0`, the session accepts with at least
`k*(p)` wrong steps with probability at most `1 − p + Σ_t err_t`. -/
theorem harm_count_kstar {num den : ℕ} [NeZero den] (h0 : 0 < num) (hq : num < den) {p : ℝ} (hp : p < 1)
    {err : ℕ → ℝ≥0∞} {H : ℝ} (hH : kStar num den p ≤ H) {N : ℕ} (P : Play n N)
    (hL : P.Laws fun L => L = Law.bernoulli n num den) (hD : P.Drawn err) :
    P.prEv (fun _ => 1) (fun acc x => acc = true ∧ H ≤ x) 0 ≤
      ENNReal.ofReal (1 - p) + ∑ m ∈ range N, err m := by
  have hr := Law.bernRate_pos h0 hq
  have key := harm_budget (π := fun _ => Law.bernRate num den) (h := fun _ => (1 : ℝ)) hr.le
    (fun _ => (mul_one _).le) H P
    (Play.Laws.mono (fun L (hL : L = Law.bernoulli n num den) => by
      subst hL; exact Law.bernoulli_design_exact hq) P hL) hD
  refine key.trans (add_le_add (ENNReal.ofReal_le_ofReal ?_) le_rfl)
  have h1p : 0 < 1 - p := sub_pos.2 hp
  calc Real.exp (-(Law.bernRate num den * H)) ≤ Real.exp (-Real.log (1 / (1 - p))) := by
        refine Real.exp_le_exp.2 (neg_le_neg ?_)
        rw [kStar, div_le_iff₀ hr, mul_comm] at hH
        exact hH
    _ = 1 - p := by rw [one_div, Real.log_inv, neg_neg, Real.exp_log h1p]

/-- One round over `B` that accepts exactly when its draw misses `B`. -/
def oneShot (L : Law n) (B : Finset (Fin n)) : Play n 1 :=
  .round L B fun ω => .stop (decide (Disjoint (L.draw ω) B))

theorem oneShot_laws {S : Law n → Prop} {L : Law n} (hS : S L) (B : Finset (Fin n)) :
    (oneShot L B).Laws S :=
  ⟨hS, fun _ => trivial⟩

theorem oneShot_drawn (L : Law n) (B : Finset (Fin n)) : (oneShot L B).Drawn (fun _ => 0) := by
  refine ⟨le_of_eq ?_, fun _ => trivial⟩
  refine Eq.trans ?_ (avg_const (C := L.Ω) (0 : ℝ≥0∞))
  refine congrArg avg (funext fun ω => ?_)
  by_cases hd : Disjoint (L.draw ω) B <;> simp [hd, pass_stop]

theorem oneShot_prEv (L : Law n) (B : Finset (Fin n)) :
    (oneShot L B).prEv (fun _ => 1) (fun acc x => acc = true ∧ (B.card : ℝ) ≤ x) 0 = L.escape B := by
  rw [oneShot, prEv_round, Verity.Law.escape, ← avg_ite_eq_prCoin]
  refine congrArg avg (funext fun ω => ?_)
  by_cases hd : Disjoint (L.draw ω) B
  · have hE : decide (Disjoint (L.draw ω) B) = true ∧ (B.card : ℝ) ≤ 0 + ∑ u ∈ B, (1 : ℝ) :=
      ⟨decide_eq_true hd, by simp⟩
    rw [prEv_stop, ite_eq_left hE, ite_eq_left hd]
  · have hE : ¬ (decide (Disjoint (L.draw ω) B) = true ∧ (B.card : ℝ) ≤ 0 + ∑ u ∈ B, (1 : ℝ)) :=
      fun hE => hd (of_decide_eq_true hE.1)
    rw [prEv_stop, ite_eq_right hE, ite_eq_right hd]

/-- **The count budget is tight**: a single Bernoulli round over `B` with no drawn-unit error
accepts with `|B|` wrong steps with probability exactly `(1 − q)^{|B|}`. -/
theorem harm_count_tight {num den : ℕ} [NeZero den] (h : num ≤ den) (B : Finset (Fin n)) :
    (oneShot (Law.bernoulli n num den) B).prEv (fun _ => 1)
        (fun acc x => acc = true ∧ (B.card : ℝ) ≤ x) 0 =
      Law.bernFactor num den 0 ^ B.card := by
  rw [oneShot_prEv, Law.bernoulli_escape h]

end Adaptive

end FlockSoundness.Audit
