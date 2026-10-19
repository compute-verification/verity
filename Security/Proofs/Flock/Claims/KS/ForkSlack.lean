import Proofs.Flock.Claims.KS.Rounds

/-!
# The fork bound's slack and one more escape per round

A simulated fork at round `j` falls short of the deployed one only where the fork's coins agree with its first run's
on what the simulator reads, round after round: a chance `δ_t` at round `t`, so `Δ_{j,i} = Π_{j ≤ t < i} δ_t` through
round `i` (`slackΔ`). There the fork moves on to round `i`, and past the last round it is the first run itself, so the
simulated fork's bound is the deployed fork's, at round `j`, and the slack
`Σ_{j<i<K} Δ_{j,i}·Pr[F_i] + Δ_{j,K}·Pr[F_K]` (`forkSlack`), with `F_i` the accepting fork at round `i` with a
collision (`forkEv`) and `F_K` the accepting run with a collision (`oneEv`, `forkPrAt`).

* **The coupling.** `ForkCoinCoupling` is the step on the fork's coin law: the simulated fork at round `t` is at most
  the deployed one there and `δ_t` times the simulated fork at `t + 1`, and past the last round at most `F_K`;
  `sim_le_slack` unrolls it into the slack, so the coupling gives the slack form (`slack_of_coupling`). At a
  simulator's coins `Φ` the simulated fork is `simForkPr`, the two runs at the tape and the live coins of a live
  fork's two branches.
* **One more escape per round.** The slack folds into one fork event per round `i`, `F_i` or, at the last round, the
  one-run event (`slackG`), of weight `C_i = (Σ_{r,ℓ} √(2^{d_{ℓ+1}}·Q_{ℓ+1}·w_{r,ℓ,i}))²` (`slackC`) at the slack's
  weights `w` (`slackW`), since `√(c·(p + Σ_i w_i·q_i)) ≤ √(c·p) + Σ_i √(c·w_i)·√q_i` (`sum_sqrt_slack_le`). The old
  escapes with those (`slackEsc`) bound the old fork terms at the slack (`slackEsc_term`, `forkSlack_le`).
* **The fit.** With the plain fork rounds past round 0 (`plainJ_ne_zero`), the old escapes, the fork of weight `k` and
  one more event a round are still two a round, so they fit the plain weight when `Σ_i C_i ≤ 2·2^{d₀}·Q₀`
  (`slackEsc_fitsFork_plain`, from `plain_fork_room`).
-/

namespace Claims.Flock

/-- `Δ_{j,i} = Π_{j ≤ t < i} δ_t`. -/
noncomputable def slackΔ (δ : ℕ → ℝ) (j i : ℕ) : ℝ := ∏ t ∈ Finset.Ico j i, δ t

theorem slackΔ_nonneg {δ : ℕ → ℝ} (hδ : ∀ t, 0 ≤ δ t) (j i : ℕ) : 0 ≤ slackΔ δ j i :=
  Finset.prod_nonneg fun t _ => hδ t

/-- **The slack's weight at round `i`** of a fork at round `j`, of `K` rounds: `Δ_{j,i}` past `j`, and at the last
round `Δ_{j,K}` more, the one-run event's. -/
noncomputable def slackW (K : ℕ) (δ : ℕ → ℝ) (j i : ℕ) : ℝ :=
  (if j < i then slackΔ δ j i else 0) + (if i + 1 = K then slackΔ δ j K else 0)

theorem slackW_nonneg {δ : ℕ → ℝ} (hδ : ∀ t, 0 ≤ δ t) (K j i : ℕ) : 0 ≤ slackW K δ j i := by
  unfold slackW
  refine add_nonneg ?_ ?_ <;> split_ifs <;> first | exact slackΔ_nonneg hδ _ _ | exact le_rfl

/-- **The fork's slack, unrolled**: where the simulated fork at each round `t < K` is at most the deployed one there
and `δ_t` times the simulated fork at `t + 1`, and past the last round at most `dep K`, the simulated fork at `j < K`
is at most `dep j + Σ_{j<i<K} Δ_{j,i}·dep i + Δ_{j,K}·dep K`. -/
theorem sim_le_slack (K : ℕ) (δ sim dep : ℕ → ℝ) (hδ : ∀ t, 0 ≤ δ t)
    (hstep : ∀ t < K, sim t ≤ dep t + δ t * sim (t + 1)) (hlast : sim K ≤ dep K) :
    ∀ j < K, sim j ≤ dep j + (∑ i ∈ Finset.Ico (j + 1) K, slackΔ δ j i * dep i) + slackΔ δ j K * dep K := by
  have key : ∀ n j, j + n + 1 = K →
      sim j ≤ dep j + (∑ i ∈ Finset.Ico (j + 1) K, slackΔ δ j i * dep i) + slackΔ δ j K * dep K := by
    intro n
    induction n with
    | zero =>
      intro j hj
      have e : Finset.Ico (j + 1) K = ∅ := Finset.Ico_eq_empty (by omega)
      have hΔ : slackΔ δ j K = δ j := by
        unfold slackΔ
        rw [← hj, show j + 0 + 1 = j + 1 by omega, Nat.Ico_succ_singleton, Finset.prod_singleton]
      rw [e, Finset.sum_empty, add_zero, hΔ]
      have := hstep j (by omega)
      rw [show j + 1 = K by omega] at this
      nlinarith [hδ j]
    | succ n ih =>
      intro j hj
      have h1 := ih (j + 1) (by omega)
      have hs := hstep j (by omega)
      have hΔ : ∀ i, j + 1 ≤ i → slackΔ δ j i = δ j * slackΔ δ (j + 1) i := fun i hi => by
        unfold slackΔ
        exact Finset.prod_eq_prod_Ico_succ_bot (by omega) _
      have hsum : ∑ i ∈ Finset.Ico (j + 1) K, slackΔ δ j i * dep i =
          δ j * dep (j + 1) + δ j * ∑ i ∈ Finset.Ico (j + 1 + 1) K, slackΔ δ (j + 1) i * dep i := by
        rw [Finset.sum_eq_sum_Ico_succ_bot (by omega), hΔ (j + 1) le_rfl, Finset.mul_sum]
        congr 1
        · unfold slackΔ
          rw [Finset.Ico_self, Finset.prod_empty, mul_one]
        · refine Finset.sum_congr rfl fun i hi => ?_
          rw [hΔ i (by simp only [Finset.mem_Ico] at hi; omega), mul_assoc]
      rw [hsum, hΔ K (by omega)]
      have hδj := hδ j
      have := mul_le_mul_of_nonneg_left h1 hδj
      nlinarith [this, hs]
  intro j hj
  exact key (K - j - 1) j (by omega)

section Events

variable {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)

/-- **Two accepting runs with a collision** among the verifier's calls in the two. -/
def pairEv (ω : Ω) (s s' : Fin (V.k x) → Coin) : Prop :=
  (run H P V x ω s).out = .acc ∧ (run H P V x ω s').out = .acc ∧
    TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω s').vLog)

/-- **An accepting fork at round `i` with a collision** among the verifier's calls in its two runs. -/
def forkEv (i : Fin (V.k x)) (ω : Ω) (s s' : Fin (V.k x) → Coin) : Prop :=
  pairEv H V P x ω s (forkCoins i s s')

/-- **An accepting run with a collision** among the verifier's calls: the fork past the last round. -/
def oneEv (ω : Ω) (s : Fin (V.k x) → Coin) : Prop :=
  (run H P V x ω s).out = .acc ∧ TiedColl H (run H P V x ω s).vLog

/-- **The fork events' probabilities by round**: `F_i`'s for `i < K`, and the one-run event's past the last round. -/
noncomputable def forkPrAt [Fintype Coin] (ω : Ω) (i : ℕ) : ℝ :=
  if h : i < V.k x then Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => forkEv H V P x ⟨i, h⟩ ω s.1 s.2
  else Pr fun s : Fin (V.k x) → Coin => oneEv H V P x ω s

/-- **The fork's slack** at round `j` and chances `δ`: `Σ_{j<i<K} Δ_{j,i}·Pr[F_i] + Δ_{j,K}·Pr[F_K]`. -/
noncomputable def forkSlack [Fintype Coin] (δ : ℕ → ℝ) (j : ℕ) (ω : Ω) : ℝ :=
  (∑ i ∈ Finset.Ico (j + 1) (V.k x), slackΔ δ j i * forkPrAt H V P x ω i) +
    slackΔ δ j (V.k x) * forkPrAt H V P x ω (V.k x)

/-- **The slack's events, one a round**: `F_i`, or at the last round the one-run event. -/
def slackG (i : Fin (V.k x)) (ω : Ω) (s s' : Fin (V.k x) → Coin) : Prop :=
  forkEv H V P x i ω s s' ∨ (i.val + 1 = V.k x ∧ oneEv H V P x ω s)

theorem slackG_tied (i : Fin (V.k x)) (ω : Ω) (s s' : Fin (V.k x) → Coin) (h : slackG H V P x i ω s s') :
    TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins i s s')).vLog) := by
  rcases h with h | h
  · exact h.2.2
  · exact h.2.2.append_right _

/-- **The slack is at most the slack's events at their weights**: `F_i` is in round `i`'s event and the one-run event
in the last round's, whose probability over a fork's pairs is its own (`Pr_fst`). -/
theorem forkSlack_le [Fintype Coin] [Nonempty Coin] {δ : ℕ → ℝ} (hδ : ∀ t, 0 ≤ δ t) (j₀ : Fin (V.k x)) (ω : Ω) :
    forkSlack H V P x δ j₀.val ω ≤ ∑ i : Fin (V.k x), slackW (V.k x) δ j₀.val i.val *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => slackG H V P x i ω s.1 s.2 := by
  classical
  have hK0 : 0 < V.k x := j₀.pos
  set j := j₀.val
  set K := V.k x
  have hG : ∀ i : Fin K, forkPrAt H V P x ω i.val ≤
      Pr fun s : (Fin K → Coin) × (Fin K → Coin) => slackG H V P x i ω s.1 s.2 := fun i => by
    unfold forkPrAt
    rw [dite_eq_left_of_eq_true (eq_true i.isLt)]
    exact Pr_mono fun s hs => Or.inl hs
  have hG₁ : ∀ i : Fin K, i.val + 1 = K → forkPrAt H V P x ω K ≤
      Pr fun s : (Fin K → Coin) × (Fin K → Coin) => slackG H V P x i ω s.1 s.2 := fun i hi => by
    unfold forkPrAt
    rw [dite_eq_right_of_eq_false (eq_false (lt_irrefl _)), ← Pr_fst (β := Fin K → Coin) fun s => oneEv H V P x ω s]
    exact Pr_mono fun s hs => Or.inr ⟨hi, hs⟩
  have hper : ∀ i : Fin K,
      (if j < i.val then slackΔ δ j i.val * forkPrAt H V P x ω i.val else 0) +
        (if i.val + 1 = K then slackΔ δ j K * forkPrAt H V P x ω K else 0) ≤
      slackW K δ j i.val * Pr fun s : (Fin K → Coin) × (Fin K → Coin) => slackG H V P x i ω s.1 s.2 := fun i => by
    unfold slackW
    rw [add_mul]
    refine add_le_add ?_ ?_
    · split_ifs
      · exact mul_le_mul_of_nonneg_left (hG i) (slackΔ_nonneg hδ _ _)
      · simp
    · split_ifs with hi
      · exact mul_le_mul_of_nonneg_left (hG₁ i hi) (slackΔ_nonneg hδ _ _)
      · simp
  refine le_trans (le_of_eq ?_) (Finset.sum_le_sum fun i _ => hper i)
  rw [Finset.sum_add_distrib]
  unfold forkSlack
  congr 1
  · rw [Fin.sum_univ_eq_sum_range (fun i => if j < i then slackΔ δ j i * forkPrAt H V P x ω i else 0) K,
      ← Finset.sum_filter]
    refine Finset.sum_congr ?_ fun _ _ => rfl
    ext i
    simp only [Finset.mem_Ico, Finset.mem_filter, Finset.mem_range]
    omega
  · rw [Fin.sum_univ_eq_sum_range (fun i => if i + 1 = K then slackΔ δ j K * forkPrAt H V P x ω K else 0) K]
    rw [Finset.sum_eq_single (K - 1) (fun b _ hb => ite_eq_right_of_eq_false _ _ (eq_false (by omega))) (fun h => absurd
      (Finset.mem_range.2 (by omega)) h), ite_eq_left_of_eq_true _ _ (eq_true (by omega))]

/-- **A simulated fork at deployed round `t`**: the two runs at a simulator's coins `Φ`, at its tape and the live
coins of each branch of a live fork (`lf t`, the second branch's live coins at the first's and fresh ones) accept with
a collision. -/
noncomputable def simForkPr [Fintype Coin] {τ LC : Type} [Fintype τ] [Fintype LC] {n : ℕ}
    (Φ : τ × (Fin n → LC) → Fin (V.k x) → Coin) (lf : ℕ → (Fin n → LC) → (Fin n → LC) → Fin n → LC) (ω : Ω)
    (t : ℕ) : ℝ :=
  Pr fun p : τ × (Fin n → LC) × (Fin n → LC) => pairEv H V P x ω (Φ (p.1, p.2.1)) (Φ (p.1, lf t p.2.1 p.2.2))

/-- **The coupling of the fork's coin law**: at chances `δ`, the simulated forks `sim` are at most the deployed fork
at each round `t` and `δ_t` times the simulated fork at `t + 1`, and past the last round at most the one-run event.
At a simulator whose tape holds uniform functions of its live coins' history (`fnC`), against the deployed verifier
revealing whole coins, `δ_t` is the chance that a fresh live coin agrees with the first branch's on the function's
key. -/
def ForkCoinCoupling [Fintype Coin] (δ sim : ℕ → ℝ) (ω : Ω) : Prop :=
  (∀ t < V.k x, sim t ≤ forkPrAt H V P x ω t + δ t * sim (t + 1)) ∧ sim (V.k x) ≤ forkPrAt H V P x ω (V.k x)

/-- **The coupling gives the slack**: a simulated fork at round `j` is at most the deployed one there and the slack
(`sim_le_slack`). -/
theorem slack_of_coupling [Fintype Coin] {δ sim : ℕ → ℝ} (hδ : ∀ t, 0 ≤ δ t) {ω : Ω}
    (hc : ForkCoinCoupling H V P x δ sim ω) (j : Fin (V.k x)) :
    sim j ≤ (Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => forkEv H V P x j ω s.1 s.2) +
      forkSlack H V P x δ j ω := by
  have h := sim_le_slack (V.k x) δ sim (forkPrAt H V P x ω) hδ hc.1 hc.2 j j.isLt
  have e : forkPrAt H V P x ω j.val =
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => forkEv H V P x j ω s.1 s.2 := by
    unfold forkPrAt
    rw [dite_eq_left_of_eq_true (eq_true j.isLt)]
  unfold forkSlack
  rw [← e]
  linarith

end Events

/-! ## One more escape per round -/

theorem sqrt_add_le_add {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) : Real.sqrt (a + b) ≤ Real.sqrt a + Real.sqrt b := by
  rw [show Real.sqrt a + Real.sqrt b = Real.sqrt ((Real.sqrt a + Real.sqrt b) ^ 2) from
    (Real.sqrt_sq (add_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b))).symm]
  refine Real.sqrt_le_sqrt ?_
  nlinarith [Real.sq_sqrt ha, Real.sq_sqrt hb, mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b)]

theorem sqrt_sum_le {ι : Type} (s : Finset ι) (f : ι → ℝ) (hf : ∀ i ∈ s, 0 ≤ f i) :
    Real.sqrt (∑ i ∈ s, f i) ≤ ∑ i ∈ s, Real.sqrt (f i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    exact (sqrt_add_le_add (hf a (Finset.mem_insert_self a s))
      (Finset.sum_nonneg fun i hi => hf i (Finset.mem_insert_of_mem hi))).trans
      (add_le_add le_rfl (ih fun i hi => hf i (Finset.mem_insert_of_mem hi)))

/-- **The slack folds into one root per round**: `Σ_a √(c_a·(p_a + Σ_i w_{a,i}·q_i))` is at most `Σ_a √(c_a·p_a)` and
`Σ_i √(C_i·q_i)`, `C_i = (Σ_a √(c_a·w_{a,i}))²`. -/
theorem sum_sqrt_slack_le {A I : Type} [Fintype A] [Fintype I] (c p : A → ℝ) (w : A → I → ℝ) (q : I → ℝ)
    (hc : ∀ a, 0 ≤ c a) (hp : ∀ a, 0 ≤ p a) (hw : ∀ a i, 0 ≤ w a i) (hq : ∀ i, 0 ≤ q i) :
    ∑ a, Real.sqrt (c a * (p a + ∑ i, w a i * q i)) ≤
      ∑ a, Real.sqrt (c a * p a) + ∑ i, Real.sqrt ((∑ a, Real.sqrt (c a * w a i)) ^ 2 * q i) := by
  have h1 : ∀ a, Real.sqrt (c a * (p a + ∑ i, w a i * q i)) ≤
      Real.sqrt (c a * p a) + ∑ i, Real.sqrt (c a * w a i) * Real.sqrt (q i) := by
    intro a
    have hn : ∀ i ∈ (Finset.univ : Finset I), 0 ≤ c a * (w a i * q i) := fun i _ =>
      mul_nonneg (hc a) (mul_nonneg (hw a i) (hq i))
    rw [mul_add, Finset.mul_sum]
    refine (sqrt_add_le_add (mul_nonneg (hc a) (hp a)) (Finset.sum_nonneg hn)).trans
      (add_le_add le_rfl ((sqrt_sum_le _ _ hn).trans (le_of_eq (Finset.sum_congr rfl fun i _ => ?_))))
    rw [← mul_assoc, Real.sqrt_mul (mul_nonneg (hc a) (hw a i))]
  refine (Finset.sum_le_sum fun a _ => h1 a).trans (le_of_eq ?_)
  rw [Finset.sum_add_distrib, Finset.sum_comm]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_mul, Real.sqrt_mul (sq_nonneg _),
    Real.sqrt_sq (Finset.sum_nonneg fun a _ => Real.sqrt_nonneg _)]

/-- **Round `i`'s weight in the slack's escapes**: `(Σ_{r,ℓ} √(2^{d_{ℓ+1}}·Q_{ℓ+1}·w r ℓ i))²`. -/
noncomputable def slackC {n : ℕ} (sch : FlockSoundness.Accounting.Schedule)
    (w : Fin 2 → Fin sch.levels.length → Fin n → ℝ) (i : Fin n) : ℝ :=
  (∑ a : Fin 2 × Fin sch.levels.length, Real.sqrt (cOf sch (a.2.val + 1) * w a.1 a.2 i)) ^ 2

/-- **The old escapes and one more fork event a round**: `oldEsc`'s, and at each round `i` the event `G i` with
weight `slackC sch w i`. -/
noncomputable def slackEsc {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (sch : FlockSoundness.Accounting.Schedule) (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog)
    (w : Fin 2 → Fin sch.levels.length → Fin (V.k x) → ℝ)
    (G : Fin (V.k x) → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (tiedG : ∀ i ω s s', G i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins i s s')).vLog)) : Escapes H V P x where
  I := (Fin 2 × Fin sch.levels.length) ⊕ Fin (V.k x)
  c := Sum.elim (fun a => cOf sch (a.2.val + 1)) (slackC sch w)
  j := Sum.elim (fun a => j a.1 a.2) id
  ev := Sum.elim (fun a => ev a.1 a.2) G
  L := Unit
  β _ := 1
  ev₁ _ := ev₁
  c_nonneg i := match i with
    | .inl _ => cOf_nonneg sch _
    | .inr _ => sq_nonneg _
  β_nonneg _ := zero_le_one
  tied i := match i with
    | .inl a => tied a.1 a.2
    | .inr i => tiedG i
  tied₁ _ := tied₁

/-- **The old fork terms at the slack are at most the slack's escapes' term**: where each fork term's `Adv` is at most
its fork event's probability and the slack's events at weights `w`, and the self-clash at most the one-run event's
(`sum_sqrt_slack_le`). -/
theorem slackEsc_term {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) (sch : FlockSoundness.Accounting.Schedule)
    (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog)
    (w : Fin 2 → Fin sch.levels.length → Fin (V.k x) → ℝ) (hw : ∀ r ℓ i, 0 ≤ w r ℓ i)
    (G : Fin (V.k x) → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (tiedG : ∀ i ω s s', G i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins i s s')).vLog))
    (ω : Ω) (adv : ℕ → ℕ → ℝ) (sc : ℝ)
    (hadv : ∀ r ℓ, adv r.val (ℓ.val + 1) ≤
      (Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev r ℓ ω s.1 s.2) +
        ∑ i, w r ℓ i * Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => G i ω s.1 s.2)
    (hsc : sc ≤ Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        Real.sqrt (2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries * adv r ℓ)) + sc ≤
      (slackEsc H V P x sch j ev ev₁ tied tied₁ w G tiedG).term ω := by
  set p : Fin 2 × Fin sch.levels.length → ℝ := fun a =>
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev a.1 a.2 ω s.1 s.2
  set q : Fin (V.k x) → ℝ := fun i =>
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => G i ω s.1 s.2
  have hterm : (slackEsc H V P x sch j ev ev₁ tied tied₁ w G tiedG).term ω =
      (∑ a, Real.sqrt (cOf sch (a.2.val + 1) * p a)) + (∑ i, Real.sqrt (slackC sch w i * q i)) +
        Pr fun s : Fin (V.k x) → Coin => ev₁ ω s := by
    show (∑ i : (Fin 2 × Fin sch.levels.length) ⊕ Fin (V.k x), Real.sqrt
        (Sum.elim (fun a => cOf sch (a.2.val + 1)) (slackC sch w) i *
          Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
            Sum.elim (fun a => ev a.1 a.2) G i ω s.1 s.2)) +
        ∑ _l : Unit, (1 : ℝ) * (Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) = _
    rw [Fintype.sum_sum_type]
    simp [p, q]
  rw [hterm, Finset.sum_range]
  refine add_le_add ?_ hsc
  have hold : (∑ r : Fin 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
      Real.sqrt (2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries * adv r ℓ)) ≤
      ∑ a : Fin 2 × Fin sch.levels.length, Real.sqrt (cOf sch (a.2.val + 1) * (p a + ∑ i, w a.1 a.2 i * q i)) := by
    rw [Fintype.sum_prod_type]
    refine Finset.sum_le_sum fun r _ => ?_
    rw [sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ =>
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hadv r ℓ) (cOf_nonneg sch _))
  exact hold.trans (sum_sqrt_slack_le _ p (fun a i => w a.1 a.2 i) q (fun a => cOf_nonneg sch _)
    (fun _ => Pr_nonneg _) (fun a i => hw a.1 a.2 i) fun _ => Pr_nonneg _)

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-- **The plain fork rounds are past round 0**: each is after the link message. -/
theorem plainJ_ne_zero {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables false x).head? = some (spec, st)) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some sch) (r : Fin 2) (ℓ : Fin sch.levels.length) :
    (plainJ h0 hsch r ℓ).val ≠ 0 := by
  simp only [plainJ, forkRound]
  omega

/-- **The slack's escapes fit the plain weight with the fork event of weight `k`**, when the slack's weights total at
most level 0's `2·2^{d₀}·Q₀`: the plain fork rounds are distinct and past round 0, so with one more event a round and
the fork at round 0 there are two a round, and `plain_fork_room` leaves that much of the weight. -/
theorem slackEsc_fitsFork_plain (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin ((deployPlain rel).V.k x) → Coin) →
      (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployPlain rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployPlain rel).V x ω (Claims.forkCoins (plainJ h0 hsch r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployPlain rel).V x ω s).vLog)
    (w : Fin 2 → Fin sch.levels.length → Fin ((deployPlain rel).V.k x) → ℝ)
    (G : Fin ((deployPlain rel).V.k x) → Ω → (Fin ((deployPlain rel).V.k x) → Coin) →
      (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (tiedG : ∀ i ω s s', G i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployPlain rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployPlain rel).V x ω (Claims.forkCoins i s s')).vLog))
    (hfit : ∑ i, Claims.Flock.slackC sch w i ≤
      2 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries)) :
    (Claims.Flock.slackEsc sha512 (deployPlain rel).V P x sch (plainJ h0 hsch) ev ev₁ tied tied₁ w G tiedG).FitsFork
      ((deployPlain rel).weight x) ((deployPlain rel).k x) := by
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · show (Finset.univ.filter fun i : (Fin 2 × Fin sch.levels.length) ⊕ Fin ((deployPlain rel).V.k x) =>
        (Sum.elim (fun a : Fin 2 × Fin sch.levels.length => plainJ h0 hsch a.1 a.2) id i :
          Fin ((deployPlain rel).V.k x)) = r).card + (if r.val = 0 then 1 else 0) ≤ 2
    rw [Finset.card_filter, Fintype.sum_sum_type]
    have h1 : (∑ a : Fin 2 × Fin sch.levels.length,
        if id (α := Fin ((deployPlain rel).V.k x)) (plainJ h0 hsch a.1 a.2) = r then 1 else 0) ≤
          if r.val = 0 then 0 else 1 := by
      rw [← Finset.card_filter]
      split_ifs with hr
      · rw [Nat.le_zero, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro a _ he
        exact plainJ_ne_zero h0 hsch a.1 a.2 ((congrArg Fin.val he).trans hr)
      · exact Finset.card_le_one.2 fun a ha b hb => by
          simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
          obtain ⟨e1, e2⟩ := plainJ_inj h0 hsch _ _ _ _ (ha.trans hb.symm)
          exact Prod.ext e1 e2
    have h2 : (∑ i : Fin ((deployPlain rel).V.k x), if i = r then 1 else 0) = 1 := by
      rw [Finset.sum_ite_eq']
      simp
    show (∑ a : Fin 2 × Fin sch.levels.length,
        if id (α := Fin ((deployPlain rel).V.k x)) (plainJ h0 hsch a.1 a.2) = r then 1 else 0) +
      (∑ i : Fin ((deployPlain rel).V.k x), if i = r then 1 else 0) + (if r.val = 0 then 1 else 0) ≤ 2
    rw [h2]
    split_ifs at h1 ⊢ <;> omega
  · have hroom := plain_fork_room rel x h0 hsch
    have hs : ∑ i : (Fin 2 × Fin sch.levels.length) ⊕ Fin ((deployPlain rel).V.k x),
        Sum.elim (fun a => Claims.Flock.cOf sch (a.2.val + 1)) (Claims.Flock.slackC sch w) i =
        ∑ a : Fin 2 × Fin sch.levels.length, Claims.Flock.cOf sch (a.2.val + 1) +
          ∑ i, Claims.Flock.slackC sch w i := by
      rw [Fintype.sum_sum_type]
      rfl
    show ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : (Fin 2 × Fin sch.levels.length) ⊕
      Fin ((deployPlain rel).V.k x), Sum.elim (fun a => Claims.Flock.cOf sch (a.2.val + 1))
        (Claims.Flock.slackC sch w) i) ≤ 2 * (deployPlain rel).weight x
    rw [hs]
    push_cast at hroom ⊢
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ (V false).k x
    rw [Fintype.sum_unique]
    exact_mod_cast V_k_pos false x

open FlockSoundness.Discharge.Exec in
/-- **The salted session's mean `ε_c⁻` at one table, at the slack**: at most `ε_table` and the slack's escapes' term,
when each fork term's mean `Adv` is at most its fork event's probability and the slack's events at weights `w`, and
the mean self-clash at most the one-run event's (`mean_epsSL_le` at `slackEsc_term`). -/
theorem mean_epsSL_le_slack {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K) (Hc : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List F) D) (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D Hc) {x : Claims.Msg}
    {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : FlockSoundness.Layout st)
    (m k0 : ℕ) (l₀ l₁ : FlockSoundness.Accounting.Level) (rest : List FlockSoundness.Accounting.Level) (ext : ℕ)
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (FlockSoundness.Refine.regionsOf st)).m) (ptLocal : ℕ)
    (pre post : List ((Fin 2 → Fin spec.mPts → F) → FlockSoundness.Game (FlockSoundness.Model.OutCL F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → FlockSoundness.Game.Strategy (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts pre post cap₀))
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (j : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Fin (V.k x))
    (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog)
    (w : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Fin (V.k x) → ℝ) (hw : ∀ r ℓ i, 0 ≤ w r ℓ i)
    (G : Fin (V.k x) → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (tiedG : ∀ i ω s s', G i ω s s' →
      Claims.TiedColl H ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins i s s')).vLog))
    (ω : Ω)
    (hadv : ∀ r ℓ, (∑ t, advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
        ptLocal spec.mPts pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev r ℓ ω s.1 s.2) +
        ∑ i, w r ℓ i * Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => G i ω s.1 s.2)
    (hsc : (∑ t, FlockSoundness.Game.expect
        (FlockSoundness.Game.ind fun outs : List (FlockSoundness.Model.OutCL F D Sl) =>
          SelfClashL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
        (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm
          ptLocal spec.mPts pre post cap₀) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ t, epsSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal
        spec.mPts pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable false x +
        (Claims.Flock.slackEsc H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁ w G tiedG).term ω := by
  have key := Claims.Flock.slackEsc_term H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁ w hw G tiedG ω
    (fun r ℓ => (∑ t, advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
      ptLocal spec.mPts pre post cap₀ r ℓ (σ t)) / Fintype.card τ) _ hadv hsc
  unfold epsSL
  rw [Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_const, εtable_one_plain h hsch L, add_assoc]
  refine add_le_add le_rfl (le_trans (add_le_add ?_ le_rfl) key)
  rw [Finset.sum_comm, Finset.sum_div]
  exact Finset.sum_le_sum fun r _ => Claims.Flock.mean_sum_sqrt_mul_le _ _
    (fun ℓ => Claims.Flock.cOf_nonneg _ ℓ) _ fun ℓ t => advRSL_nonneg A Hc E Lf _ m k0 l₀ l₁ rest ext hk0 hm ptLocal
      spec.mPts pre post cap₀ r ℓ (σ t)

end FlockDeploy.KS
