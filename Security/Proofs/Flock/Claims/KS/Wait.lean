import Proofs.Flock.Claims.ERepCost

/-!
# One wait, truncated: from `k` accepted runs to `E^rep`'s `K` draws

`E^rep`'s wait stops at `k` accepted reruns among `K` fresh draws (`waitAcc`). A per-tape knowledge bound speaks of `k`
independent accepted runs: at a tape with acceptance `ε`, `ε · Q/ε^k ≤ a + rate·ε`, where `Q` is the probability that
`k` independent runs all accept and their openings fail (`accFail`), so `Q/ε^k` is the failure probability under the
law of `k` accepted runs (the old soundness's `expectAcc`).

* **Coupling** (`wait_couple`): the first `k` accepted runs among the draws have the law of `k` independent accepted
  runs, so `ε^k · Pr[the wait collects k runs and fails] ≤ Q`, by induction on the draws.
* **Chernoff** (`wait_short_le`, `short_le_exp`): the wait collects fewer than `k` with probability at most
  `2^k·(1 − ε/2)^K` (Markov on `2^{−#accepted}`), at most `2^k·e^{−k}` once `εK ≥ 2k`.
* **Truncation** (`ERep_trunc`): `Pr[acc ∧ ¬R(x, E^rep)] ≤ 4·𝔼[a] + 2^{−m} + 2k/K + e^{−k/8}` once `rate ≤ 1/4`,
  `K > 0` and `k ≥ 8`, by cases on each tape (`tape_bound`): `ε ≤ 2k/K`; or `a > ε/4`, so `ε < 4a`; or else one wait
  fails with probability at most `1/2 + 2^k e^{−k}`, and `m` of them with at most `2^{−m} + 4·2^k e^{−k}`, which is at
  most `2^{−m} + e^{−k/8}`.
-/

namespace Claims

/-! ## Probability lemmas -/

theorem Pr_const_of {α : Type} [Fintype α] [Nonempty α] {p : Prop} (hp : p) : Pr (fun _ : α => p) = 1 := by
  unfold Pr
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [div_eq_one_iff_eq this.ne']
  norm_cast
  rw [← Finset.card_univ]
  congr 1
  classical
  exact Finset.filter_true_of_mem fun _ _ => hp

theorem Pr_const_of_not {α : Type} [Fintype α] {p : Prop} (hp : ¬ p) : Pr (fun _ : α => p) = 0 := by
  rw [Pr_congr (q := fun _ : α => False) fun _ => ⟨fun h => hp h, False.elim⟩, Pr_false]

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  unfold Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  norm_cast
  refine (Finset.card_le_card ?_).trans (Finset.card_union_le _ _)
  intro a ha
  simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and] at ha ⊢
  exact ha

/-- Markov: an event on which `g ≥ 1`, with `g ≥ 0`, has probability at most `𝔼[g]`. -/
theorem Pr_le_Ex {α : Type} [Fintype α] (p : α → Prop) (g : α → ℝ) (hg : ∀ a, 0 ≤ g a)
    (h : ∀ a, p a → 1 ≤ g a) : Pr p ≤ Ex g := by
  classical
  rw [← Ex_ind p]
  exact Ex_mono fun a => by split_ifs with hp; exacts [h a hp, hg a]

/-- `K + 1` uniform coins are a uniform first coin and `K` uniform others, for an event. -/
theorem Pr_cons {D : Type} [Fintype D] (K : ℕ) (p : D → (Fin K → D) → Prop) :
    Pr (fun ρ : Fin (K + 1) → D => p (ρ 0) fun i => ρ i.succ) = Ex fun d => Pr fun σ : Fin K → D => p d σ := by
  classical
  rw [← Ex_ind (fun ρ : Fin (K + 1) → D => p (ρ 0) fun i => ρ i.succ)]
  have e : ∀ d, Pr (fun σ : Fin K → D => p d σ) = Ex fun σ => if p d σ then (1 : ℝ) else 0 :=
    fun d => (Ex_ind _).symm
  simp_rw [e]
  exact Ex_cons K fun d σ => if p d σ then (1 : ℝ) else 0

/-- A product over independent coordinates averages to the product of the averages. -/
theorem Ex_pi_prod {D : Type} [Fintype D] [Nonempty D] (f : D → ℝ) :
    ∀ K : ℕ, Ex (fun ρ : Fin K → D => ∏ i, f (ρ i)) = Ex f ^ K
  | 0 => by simp only [Finset.univ_eq_empty, Finset.prod_empty, pow_zero]; exact Ex_const 1
  | K + 1 => by
    calc Ex (fun ρ : Fin (K + 1) → D => ∏ i, f (ρ i))
        = Ex (fun ρ : Fin (K + 1) → D =>
            (fun d (σ : Fin K → D) => f d * ∏ i, f (σ i)) (ρ 0) fun i => ρ i.succ) := by
          congr 1; funext ρ; exact Fin.prod_univ_succ _
      _ = Ex fun d => Ex fun σ : Fin K → D => f d * ∏ i, f (σ i) :=
          Ex_cons K fun d (σ : Fin K → D) => f d * ∏ i, f (σ i)
      _ = Ex fun d => f d * Ex f ^ K := by congr 1; funext d; rw [Ex_mul_left, Ex_pi_prod f K]
      _ = Ex f ^ (K + 1) := by rw [Ex_mul_right, pow_succ, mul_comm]

/-! ## What a wait collects -/

/-- What one wait collects over the coin draws `rs`, from `acc` so far: `waitAcc`'s result. -/
def waitRes {R : Type} (run : R → Trace) (k : ℕ) : List R → List Seen → List Seen
  | [], acc => acc
  | r :: rs, acc =>
    if k ≤ acc.length then acc
    else waitRes run k rs (if (run r).out = .acc then acc ++ [(run r).seen] else acc)

theorem waitAcc_res {R : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List Seen), ((waitAcc k rs acc).eval H run).1 = waitRes run k rs acc := by
  intro rs
  induction rs with
  | nil => intro acc; simp [waitAcc, waitRes, RP.eval]
  | cons r rs ih =>
    intro acc
    simp only [waitAcc, waitRes]
    split_ifs with h hacc
    · simp [RP.eval]
    all_goals simp [RP.eval, ih, Trace.seen, hacc]

/-- **The coupling**: the wait collects `k` accepted runs and they satisfy `G` with probability at most that of `k`
independent runs all accepting and satisfying `G`, over `ε^k`; from `acc` collected so far, with `n` still to come. -/
theorem wait_couple {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (k : ℕ) (G : List Seen → Prop) :
    ∀ (K n : ℕ) (acc : List Seen), acc.length + n = k →
      Pr (fun d => (run d).out = .acc) ^ n *
          Pr (fun ρ : Fin K → D => k ≤ (waitRes run k ((List.finRange K).map ρ) acc).length ∧
            G (waitRes run k ((List.finRange K).map ρ) acc)) ≤
        Pr (fun σ : Fin n → D => (∀ i, (run (σ i)).out = .acc) ∧
          G (acc ++ (List.finRange n).map fun i => (run (σ i)).seen)) := by
  have base : ∀ (acc : List Seen) (L : ℕ), Pr (fun _ : Fin L → D => G acc) ≤
      Pr (fun σ : Fin 0 → D => (∀ i, (run (σ i)).out = .acc) ∧
        G (acc ++ (List.finRange 0).map fun i => (run (σ i)).seen)) := by
    intro acc L
    simp only [List.finRange_zero, List.map_nil, List.append_nil, IsEmpty.forall_iff, true_and]
    by_cases hG : G acc
    · rw [Pr_const_of hG, Pr_const_of hG]
    · rw [Pr_const_of_not hG, Pr_const_of_not hG]
  intro K
  induction K with
  | zero =>
    intro n acc hn
    simp only [List.finRange_zero, List.map_nil, waitRes]
    rcases n with _ | n
    · have hk : k ≤ acc.length := by omega
      simp only [hk, true_and, pow_zero, one_mul]
      exact base acc 0
    · rw [Pr_const_of_not fun h => absurd h.1 (by omega), mul_zero]
      exact Pr_nonneg _
  | succ K ih =>
    intro n acc hn
    have hcons : ∀ ρ : Fin (K + 1) → D, (List.finRange (K + 1)).map ρ =
        ρ 0 :: (List.finRange K).map fun i => ρ i.succ := by
      intro ρ; rw [List.finRange_succ, List.map_cons, List.map_map]; rfl
    simp only [hcons]
    rcases n with _ | n
    · have hk : k ≤ acc.length := by omega
      simp only [waitRes, hk, ↓reduceIte, true_and, pow_zero, one_mul]
      exact base acc (K + 1)
    · have hlt : ¬ k ≤ acc.length := by omega
      simp only [waitRes, hlt, ↓reduceIte]
      set ε := Pr fun d => (run d).out = .acc with hε
      set RHS := Pr (fun σ : Fin (n + 1) → D => (∀ i, (run (σ i)).out = .acc) ∧
        G (acc ++ (List.finRange (n + 1)).map fun i => (run (σ i)).seen)) with hRHS
      set Q' : D → ℝ := fun d => Pr fun τ : Fin n → D => (∀ i, (run (τ i)).out = .acc) ∧
        G ((acc ++ [(run d).seen]) ++ (List.finRange n).map fun i => (run (τ i)).seen) with hQ'
      have hR : RHS = Ex fun d => if (run d).out = .acc then Q' d else 0 := by
        have e1 : RHS = Pr (fun σ : Fin (n + 1) → D =>
            (fun d (τ : Fin n → D) => (run d).out = .acc ∧ (∀ i, (run (τ i)).out = .acc) ∧
              G ((acc ++ [(run d).seen]) ++ (List.finRange n).map fun i => (run (τ i)).seen))
              (σ 0) fun i => σ i.succ) := by
          refine Pr_congr fun σ => ?_
          rw [Fin.forall_fin_succ, List.finRange_succ, List.map_cons, List.map_map, List.append_cons]
          simp only [Function.comp_def, and_assoc]
        rw [e1]
        refine (Pr_cons n fun d (τ : Fin n → D) => (run d).out = .acc ∧ (∀ i, (run (τ i)).out = .acc) ∧
          G ((acc ++ [(run d).seen]) ++ (List.finRange n).map fun i => (run (τ i)).seen)).trans ?_
        congr 1; funext d
        by_cases hd : (run d).out = .acc
        · simp only [hd, true_and, ↓reduceIte, hQ']
        · simp only [hd, false_and, ↓reduceIte, Pr_false]
      have hε0 : 0 ≤ ε := Pr_nonneg _
      have hpt : ∀ d, ε ^ (n + 1) * Pr (fun σ : Fin K → D =>
            k ≤ (waitRes run k ((List.finRange K).map σ)
              (if (run d).out = .acc then acc ++ [(run d).seen] else acc)).length ∧
            G (waitRes run k ((List.finRange K).map σ)
              (if (run d).out = .acc then acc ++ [(run d).seen] else acc))) ≤
          ε * (if (run d).out = .acc then Q' d else 0) + (if (run d).out = .acc then 0 else RHS) := by
        intro d
        by_cases hd : (run d).out = .acc
        · simp only [hd, ↓reduceIte, add_zero]
          have h := ih n (acc ++ [(run d).seen]) (by simp; omega)
          rw [pow_succ, mul_comm (ε ^ n) ε, mul_assoc]
          exact mul_le_mul_of_nonneg_left h hε0
        · simp only [hd, ↓reduceIte, mul_zero, zero_add]
          exact ih (n + 1) acc hn
      calc ε ^ (n + 1) * Pr (fun ρ : Fin (K + 1) → D =>
            k ≤ (waitRes run k ((List.finRange K).map fun i => ρ i.succ)
              (if (run (ρ 0)).out = .acc then acc ++ [(run (ρ 0)).seen] else acc)).length ∧
            G (waitRes run k ((List.finRange K).map fun i => ρ i.succ)
              (if (run (ρ 0)).out = .acc then acc ++ [(run (ρ 0)).seen] else acc)))
          = Ex fun d => ε ^ (n + 1) * Pr (fun σ : Fin K → D =>
              k ≤ (waitRes run k ((List.finRange K).map σ)
                (if (run d).out = .acc then acc ++ [(run d).seen] else acc)).length ∧
              G (waitRes run k ((List.finRange K).map σ)
                (if (run d).out = .acc then acc ++ [(run d).seen] else acc))) := by
            rw [Ex_mul_left]
            congr 1
            exact Pr_cons K fun d (σ : Fin K → D) =>
              k ≤ (waitRes run k ((List.finRange K).map σ)
                (if (run d).out = .acc then acc ++ [(run d).seen] else acc)).length ∧
              G (waitRes run k ((List.finRange K).map σ)
                (if (run d).out = .acc then acc ++ [(run d).seen] else acc))
        _ ≤ Ex fun d => ε * (if (run d).out = .acc then Q' d else 0) +
              (if (run d).out = .acc then 0 else RHS) := Ex_mono hpt
        _ = ε * RHS + (1 - ε) * RHS := by
            rw [Ex_add, Ex_mul_left, ← hR, Ex_ite]
            ring
        _ = RHS := by ring

/-- A wait that collects fewer than `k` accepted runs saw fewer than `k` accepting draws:
`(1/2)^k ≤ (1/2)^{#collected} · ∏_r (1/2 if r accepts, else 1)`. -/
theorem waitRes_short {R : Type} (run : R → Trace) (k : ℕ) :
    ∀ (rs : List R) (acc : List Seen), acc.length ≤ k → (waitRes run k rs acc).length < k →
      (1 / 2 : ℝ) ^ k ≤
        (1 / 2) ^ acc.length * (rs.map fun r => if (run r).out = .acc then (1 / 2 : ℝ) else 1).prod
  | [], acc, h, _ => by
    simp only [List.map_nil, List.prod_nil, mul_one]
    exact pow_le_pow_of_le_one (by norm_num) (by norm_num) h
  | r :: rs, acc, h, hs => by
    simp only [waitRes] at hs
    split_ifs at hs with hk hr
    · omega
    · have ih := waitRes_short run k rs (acc ++ [(run r).seen]) (by simp; omega) hs
      simp only [List.length_append, List.length_singleton, pow_succ] at ih
      simp only [List.map_cons, List.prod_cons, hr, ↓reduceIte]
      linarith [show (1 / 2 : ℝ) ^ acc.length * (1 / 2) *
          (rs.map fun r => if (run r).out = .acc then (1 / 2 : ℝ) else 1).prod =
        (1 / 2) ^ acc.length * ((1 / 2) *
          (rs.map fun r => if (run r).out = .acc then (1 / 2 : ℝ) else 1).prod) by ring]
    · have ih := waitRes_short run k rs acc h hs
      simp only [List.map_cons, List.prod_cons, hr, ↓reduceIte, one_mul]
      exact ih

/-- **Chernoff, by Markov on `2^{−#accepted}`**: a wait over `K` draws collects fewer than `k` accepted runs with
probability at most `2^k·(1 − ε/2)^K`. -/
theorem wait_short_le {D : Type} [Fintype D] [Nonempty D] (run : D → Trace) (k K : ℕ) :
    Pr (fun ρ : Fin K → D => (waitRes run k ((List.finRange K).map ρ) []).length < k) ≤
      2 ^ k * (1 - Pr (fun d => (run d).out = .acc) / 2) ^ K := by
  classical
  set f : D → ℝ := fun d => if (run d).out = .acc then 1 / 2 else 1 with hf
  have hf0 : ∀ d, 0 ≤ f d := fun d => by simp only [hf]; split_ifs <;> norm_num
  have hEf : Ex f = 1 - Pr (fun d => (run d).out = .acc) / 2 := by
    rw [hf, Ex_ite]; ring
  rw [← hEf, ← Ex_pi_prod f K, ← Ex_mul_left]
  refine Pr_le_Ex _ _ (fun ρ => mul_nonneg (by positivity) (Finset.prod_nonneg fun i _ => hf0 _)) fun ρ hρ => ?_
  have h := waitRes_short run k ((List.finRange K).map ρ) [] (Nat.zero_le _) hρ
  rw [List.length_nil, pow_zero, one_mul, List.map_map] at h
  have hprod : (((List.finRange K).map ((fun r => if (run r).out = .acc then (1 / 2 : ℝ) else 1) ∘ ρ))).prod =
      ∏ i, f (ρ i) := by
    rw [Fin.prod_univ_def]; rfl
  rw [hprod] at h
  have h2 : (2 : ℝ) ^ k * (1 / 2) ^ k = 1 := by rw [← mul_pow]; norm_num
  calc (1 : ℝ) = 2 ^ k * (1 / 2) ^ k := h2.symm
    _ ≤ 2 ^ k * ∏ i, f (ρ i) := mul_le_mul_of_nonneg_left h (by positivity)

theorem short_le_exp {ε : ℝ} (hε1 : ε ≤ 1) (k K : ℕ) (hK : 2 * (k : ℝ) ≤ ε * K) :
    (2 : ℝ) ^ k * (1 - ε / 2) ^ K ≤ 2 ^ k * Real.exp (-(k : ℝ)) := by
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  have h1 : 1 - ε / 2 ≤ Real.exp (-ε / 2) := by
    have := Real.add_one_le_exp (-ε / 2); linarith
  calc (1 - ε / 2) ^ K ≤ Real.exp (-ε / 2) ^ K := pow_le_pow_left₀ (by linarith) h1 K
    _ = Real.exp (K * (-ε / 2)) := (Real.exp_nat_mul _ K).symm
    _ ≤ Real.exp (-(k : ℝ)) := Real.exp_le_exp.2 (by nlinarith)

/-! ## Numbers -/

theorem four_two_pow_exp_le (k : ℕ) (hk : 8 ≤ k) :
    4 * ((2 : ℝ) ^ k * Real.exp (-(k : ℝ))) ≤ Real.exp (-(k : ℝ) / 8) := by
  have h2 : (2 : ℝ) = Real.exp (Real.log 2) := (Real.exp_log two_pos).symm
  have hpow : (2 : ℝ) ^ (k + 2) = Real.exp ((k + 2 : ℕ) * Real.log 2) := by
    rw [Real.exp_nat_mul, ← h2]
  have e : 4 * ((2 : ℝ) ^ k * Real.exp (-(k : ℝ))) = Real.exp ((k + 2 : ℕ) * Real.log 2 + -(k : ℝ)) := by
    rw [Real.exp_add, ← hpow, pow_add]; ring
  rw [e]
  refine Real.exp_le_exp.2 ?_
  have hl := Real.log_two_lt_d9
  have hk' : (8 : ℝ) ≤ k := by exact_mod_cast hk
  push_cast
  nlinarith

theorem pow_sub_pow_le_four {b c : ℝ} (hb : 0 ≤ b) (hbc : b ≤ c) (hc : c ≤ 3 / 4) :
    ∀ m : ℕ, c ^ m - b ^ m ≤ 4 * (c - b)
  | 0 => by simp only [pow_zero, sub_self]; linarith
  | m + 1 => by
    have ih := pow_sub_pow_le_four hb hbc hc m
    have hbm : b ^ m ≤ 1 := pow_le_one₀ hb (by linarith)
    have hc0 : 0 ≤ c := hb.trans hbc
    rw [pow_succ, pow_succ]
    nlinarith [mul_le_mul_of_nonneg_left ih hc0, mul_le_mul_of_nonneg_right hbm (sub_nonneg.2 hbc),
      mul_nonneg (sub_nonneg.2 hc) (sub_nonneg.2 hbc), pow_nonneg hb m]

/-- **One tape**: acceptance `ε`, one wait's failure `f ≤ t + Q` with `t` the short-wait probability and `Q` the
failure on `k` accepted runs, `ε·Q ≤ a + ε/4`: then `ε·f^m ≤ 4a + 2^{−m} + 2k/K + e^{−k/8}`. -/
theorem tape_bound {ε f a t Q : ℝ} {m k K : ℕ} (hε0 : 0 ≤ ε) (hε1 : ε ≤ 1) (hf0 : 0 ≤ f) (hf1 : f ≤ 1)
    (ha : 0 ≤ a) (ht0 : 0 ≤ t) (hK : 0 < K) (hk : 8 ≤ k) (hf : 0 < ε → f ≤ t + Q)
    (hQ : 0 < ε → ε * Q ≤ a + ε / 4) (ht : 2 * (k : ℝ) ≤ ε * K → t ≤ 2 ^ k * Real.exp (-(k : ℝ))) :
    ε * f ^ m ≤ 4 * a + (2 : ℝ) ^ (-(m : ℝ)) + 2 * k / K + Real.exp (-(k : ℝ) / 8) := by
  have hfm : f ^ m ≤ 1 := pow_le_one₀ hf0 hf1
  have hfm0 : 0 ≤ f ^ m := pow_nonneg hf0 m
  have hεf : ε * f ^ m ≤ ε := mul_le_of_le_one_right hε0 hfm
  have hr1 : 0 ≤ (2 : ℝ) ^ (-(m : ℝ)) := Real.rpow_nonneg (by norm_num) _
  have hK' : (0 : ℝ) < K := by exact_mod_cast hK
  have hr2 : 0 ≤ 2 * (k : ℝ) / K := by positivity
  have hr3 : 0 ≤ Real.exp (-(k : ℝ) / 8) := (Real.exp_pos _).le
  by_cases h1 : ε ≤ 2 * k / K
  · linarith
  push Not at h1
  have hεpos : 0 < ε := lt_of_le_of_lt hr2 h1
  have hεK : 2 * (k : ℝ) ≤ ε * K := by rw [div_lt_iff₀ hK'] at h1; linarith
  by_cases h2 : ε / 4 < a
  · linarith
  push Not at h2
  have hQ' : Q ≤ 1 / 2 := by
    have := hQ hεpos
    by_contra hc; push Not at hc
    nlinarith
  have h4t : 4 * t ≤ Real.exp (-(k : ℝ) / 8) :=
    (mul_le_mul_of_nonneg_left (ht hεK) (by norm_num)).trans (four_two_pow_exp_le k hk)
  have hexp1 : Real.exp (-(k : ℝ) / 8) ≤ 1 := Real.exp_le_one_iff.2 (by
    have : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    linarith)
  have hfc : f ≤ 1 / 2 + t := by linarith [hf hεpos]
  have hpow : f ^ m ≤ (1 / 2 + t) ^ m := pow_le_pow_left₀ hf0 hfc m
  have hsub := pow_sub_pow_le_four (b := 1 / 2) (c := 1 / 2 + t) (by norm_num) (by linarith) (by linarith) m
  have h2m : (2 : ℝ) ^ (-(m : ℝ)) = (1 / 2) ^ m := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, one_div, inv_pow]
  rw [h2m] at hr1 ⊢
  have hεf' : ε * f ^ m ≤ f ^ m := mul_le_of_le_one_left hfm0 hε1
  linarith

/-! ## `E^rep`, truncated -/

/-- The accepted runs `vs` assemble a candidate that `check` accepts. -/
def Assembles {W Op : Type} (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool)
    (vs : List Seen) : Prop :=
  ∃ w, assemble (vs.flatMap opens) = some w ∧ check w = true

section ERep

variable {Coin W Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Q`: `k` independent runs at tape `ω`, all accepted, whose views fail `G`. Over `ε^k`, the probability that `k`
accepted runs fail `G`. -/
noncomputable def accFail (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k : ℕ)
    (G : List Seen → Prop) : ℝ :=
  Claims.Pr fun σ : Fin k → Fin (V.k x) → Coin =>
    (∀ i, (run H P V x ω (σ i)).out = .acc) ∧ ¬ G ((List.finRange k).map fun i => (run H P V x ω (σ i)).seen)

/-- **One wait's failure**: at most the short-wait probability plus the failure on `k` accepted runs. -/
theorem waitFail_le (H : Msg → Msg) (V : Verifier Coin) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω) (k K : ℕ)
    (opens : Seen → List Op) (assemble : List Op → Option W) (check : W → Bool) (hε : 0 < accPr H V P x ω) :
    waitFail H V P x ω k K opens assemble check ≤
      Pr (fun ρ : Fin K → Fin (V.k x) → Coin =>
          (waitRes (run H P V x ω) k ((List.finRange K).map ρ) []).length < k) +
        accFail H V P x ω k (Assembles opens assemble check) / accPr H V P x ω ^ k := by
  unfold waitFail
  set res := fun ρ : Fin K → Fin (V.k x) → Coin => waitRes (run H P V x ω) k ((List.finRange K).map ρ) []
  have hok : ∀ ρ, WaitOk H (run H P V x ω) k opens assemble check ((List.finRange K).map ρ) ↔
      Assembles opens assemble check (res ρ) := by
    intro ρ; unfold WaitOk Assembles; rw [waitAcc_res]
  refine (Pr_mono (q := fun ρ => (res ρ).length < k ∨ (k ≤ (res ρ).length ∧
      ¬ Assembles opens assemble check (res ρ))) fun ρ h => ?_).trans ((Pr_or_le _ _).trans ?_)
  · rw [hok] at h
    by_cases hl : (res ρ).length < k
    · exact Or.inl hl
    · exact Or.inr ⟨by omega, h⟩
  · refine add_le_add le_rfl ?_
    rw [le_div_iff₀ (pow_pos hε k), mul_comm]
    have h := wait_couple (run H P V x ω) k (fun vs => ¬ Assembles opens assemble check vs) K k [] (by simp)
    simp only [List.nil_append] at h
    exact h

/-- **`E^rep`, truncated**: when `check` is sound for `R`, `K > 0`, `k ≥ 8`, and each tape with acceptance `ε > 0`
has one wait's knowledge bound on `k` accepted runs, `ε · Q/ε^k ≤ a + rate·ε` with `rate ≤ 1/4`,
`Pr[acc ∧ ¬R(x, E^rep)] ≤ 4·𝔼[a] + 2^{−m} + 2k/K + e^{−k/8}`. -/
theorem ERep_trunc (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (hcheck : ∀ x w, check x w = true → R x w)
    (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (hK : 0 < K x) (hk : 8 ≤ k x)
    (a : Ω → ℝ) (ha : ∀ ω, 0 ≤ a ω) {rate : ℝ} (hrate : rate ≤ 1 / 4)
    (hwait : ∀ ω, 0 < accPr H V P x ω →
      accPr H V P x ω * (accFail H V P x ω (k x) (Assembles opens (assemble x) (check x)) /
        accPr H V P x ω ^ k x) ≤ a ω + rate * accPr H V P x ω) :
    PrBad H V R (ERep V m k K opens assemble check) x P ≤
      4 * Ex a + 2 ^ (-(m x : ℝ)) + 2 * k x / K x + Real.exp (-(k x : ℝ) / 8) := by
  refine (ERep_bad_le H V R m k K opens assemble check hcheck x P).trans ?_
  have hpt : ∀ ω, accPr H V P x ω * waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ m x ≤
      4 * a ω + (2 ^ (-(m x : ℝ)) + 2 * k x / K x + Real.exp (-(k x : ℝ) / 8)) := by
    intro ω
    have hε0 : 0 ≤ accPr H V P x ω := Pr_nonneg _
    have hrε : rate * accPr H V P x ω ≤ accPr H V P x ω / 4 := by
      have := mul_le_mul_of_nonneg_right hrate hε0; linarith
    have h := tape_bound (ε := accPr H V P x ω)
      (f := waitFail H V P x ω (k x) (K x) opens (assemble x) (check x)) (a := a ω) (m := m x)
      hε0 (Pr_le_one _) (Pr_nonneg _) (Pr_le_one _) (ha ω) (Pr_nonneg _) hK hk
      (waitFail_le H V P x ω (k x) (K x) opens (assemble x) (check x))
      (fun hε => (hwait ω hε).trans (by linarith))
      (fun hεK => (wait_short_le (run H P V x ω) (k x) (K x)).trans (short_le_exp (Pr_le_one _) _ _ hεK))
    linarith
  calc Ex (fun ω => accPr H V P x ω * waitFail H V P x ω (k x) (K x) opens (assemble x) (check x) ^ m x)
      ≤ Ex (fun ω => 4 * a ω + (2 ^ (-(m x : ℝ)) + 2 * k x / K x + Real.exp (-(k x : ℝ) / 8))) := Ex_mono hpt
    _ = 4 * Ex a + 2 ^ (-(m x : ℝ)) + 2 * k x / K x + Real.exp (-(k x : ℝ) / 8) := by
        rw [Ex_add, Ex_mul_left, Ex_const]; ring

end ERep

end Claims
