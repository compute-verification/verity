import Proofs.Flock.Claims.Finder

/-!
# Many finders to one `B`: one square root

Today's bounds are sums `Σ_i √(c_i · a_i)` over finders `i`, each forking the prover's run at a fixed round `j_i`, so
`a_i ≤ f(j_i)`, where `f(j)` is the probability that `B`'s two runs forked at round `j` collide. `B` guesses its round
uniformly, so `Pr[B] = (Σ_j f(j))/k` (`PrB_eq_avg`), and by Cauchy–Schwarz over the finders and the fibers of the
rounds (at most `m` finders at one round)
`Σ_i √(c_i a_i) ≤ √((m · k · Σ_i c_i) · Pr[B])` (`collapse`, `collapse_B`). Linear terms fold in through
`√(C p) + β p ≤ √((√C + β)² p)` for `0 ≤ p ≤ 1` (`sqrt_add_lin`), and an average of roots is at most the root of the
average (`Ex_sqrt_le`).
-/

namespace Claims

/-- **The collapse**, over a `Finset` of finders: weights `c i`, advantages `a i ≤ f (j i)`, at most `m` finders at one
round: `Σ_i √(c_i a_i) ≤ √((m · k · Σ_i c_i) · ((Σ_r f r)/k))`. -/
theorem collapseS {I : Type} (s : Finset I) {k : ℕ} (hk : 0 < k) (c a : I → ℝ) (hc : ∀ i ∈ s, 0 ≤ c i)
    (ha : ∀ i ∈ s, 0 ≤ a i) (j : I → Fin k) (f : Fin k → ℝ) (hf : ∀ r, 0 ≤ f r)
    (haf : ∀ i ∈ s, a i ≤ f (j i)) (m : ℕ) (hm : ∀ r, (s.filter fun i => j i = r).card ≤ m) :
    ∑ i ∈ s, Real.sqrt (c i * a i) ≤ Real.sqrt ((m * k * ∑ i ∈ s, c i) * ((∑ r, f r) / k)) := by
  have hcs : (∑ i ∈ s, Real.sqrt (c i) * Real.sqrt (a i)) ^ 2 ≤
      (∑ i ∈ s, Real.sqrt (c i) ^ 2) * ∑ i ∈ s, Real.sqrt (a i) ^ 2 :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  rw [Finset.sum_congr rfl fun i hi => Real.sq_sqrt (hc i hi),
    Finset.sum_congr rfl fun i hi => Real.sq_sqrt (ha i hi)] at hcs
  have hsa : ∑ i ∈ s, a i ≤ m * ∑ r, f r := by
    calc ∑ i ∈ s, a i ≤ ∑ i ∈ s, f (j i) := Finset.sum_le_sum fun i hi => haf i hi
      _ = ∑ r, ∑ i ∈ s.filter (fun i => j i = r), f (j i) := (Finset.sum_fiberwise s j _).symm
      _ = ∑ r, ((s.filter fun i => j i = r).card : ℝ) * f r := by
          refine Finset.sum_congr rfl fun r _ => ?_
          rw [Finset.sum_congr rfl (g := fun _ => f r) (fun i hi => by rw [(Finset.mem_filter.1 hi).2]),
            Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ r, (m : ℝ) * f r := Finset.sum_le_sum fun r _ =>
          mul_le_mul_of_nonneg_right (by exact_mod_cast hm r) (hf r)
      _ = m * ∑ r, f r := (Finset.mul_sum _ _ _).symm
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  have hrhs : (m * k * ∑ i ∈ s, c i) * ((∑ r, f r) / k) = (∑ i ∈ s, c i) * (m * ∑ r, f r) := by
    field_simp
  rw [hrhs, Finset.sum_congr rfl fun i hi => Real.sqrt_mul (hc i hi) (a i)]
  have h0 : 0 ≤ ∑ i ∈ s, Real.sqrt (c i) * Real.sqrt (a i) :=
    Finset.sum_nonneg fun i _ => mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  rw [← Real.sqrt_sq h0]
  exact Real.sqrt_le_sqrt (hcs.trans (mul_le_mul_of_nonneg_left hsa (Finset.sum_nonneg hc)))

/-- **The collapse**, over every finder of a finite type. -/
theorem collapse {I : Type} [Fintype I] {k : ℕ} (hk : 0 < k) (c a : I → ℝ) (hc : ∀ i, 0 ≤ c i)
    (ha : ∀ i, 0 ≤ a i) (j : I → Fin k) (f : Fin k → ℝ) (hf : ∀ r, 0 ≤ f r) (haf : ∀ i, a i ≤ f (j i)) (m : ℕ)
    (hm : ∀ r, (Finset.univ.filter fun i => j i = r).card ≤ m) :
    ∑ i, Real.sqrt (c i * a i) ≤ Real.sqrt ((m * k * ∑ i, c i) * ((∑ r, f r) / k)) :=
  collapseS Finset.univ hk c a (fun i _ => hc i) (fun i _ => ha i) j f hf (fun i _ => haf i) m hm

/-- **Linear terms into the one square root**: `√(C·p) + β·p ≤ √((√C + β)²·p)` for `0 ≤ p ≤ 1`. -/
theorem sqrt_add_lin {C β p : ℝ} (hC : 0 ≤ C) (hβ : 0 ≤ β) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    Real.sqrt (C * p) + β * p ≤ Real.sqrt ((Real.sqrt C + β) ^ 2 * p) := by
  have hpp : p ≤ Real.sqrt p := by
    have := Real.sqrt_le_sqrt (show p * p ≤ p by nlinarith)
    rwa [Real.sqrt_mul_self hp] at this
  rw [Real.sqrt_mul hC, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (add_nonneg (Real.sqrt_nonneg _) hβ)]
  nlinarith [mul_le_mul_of_nonneg_left hpp hβ, Real.sqrt_nonneg p, Real.sqrt_nonneg C]

/-- Each finder at most `M`: `Σ_i √(c_i a_i) ≤ (Σ_i √c_i) · √M`. -/
theorem sum_sqrt_le {I : Type} (s : Finset I) (c a : I → ℝ) (hc : ∀ i ∈ s, 0 ≤ c i) {M : ℝ}
    (ha : ∀ i ∈ s, a i ≤ M) : ∑ i ∈ s, Real.sqrt (c i * a i) ≤ (∑ i ∈ s, Real.sqrt (c i)) * Real.sqrt M := by
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun i hi => ?_
  rw [← Real.sqrt_mul (hc i hi)]
  exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (ha i hi) (hc i hi))

/-- **An average of roots is at most the root of the average** (Jensen for `√`, by Cauchy–Schwarz). -/
theorem Ex_sqrt_le {α : Type} [Fintype α] (f : α → ℝ) (hf : ∀ a, 0 ≤ f a) :
    Ex (fun a => Real.sqrt (f a)) ≤ Real.sqrt (Ex f) := by
  unfold Ex
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h0 | hpos
  · simp [h0]
  have hn : (0 : ℝ) < Fintype.card α := by exact_mod_cast hpos
  have hcs : (∑ a, (1 : ℝ) * Real.sqrt (f a)) ^ 2 ≤ (∑ _a : α, (1 : ℝ) ^ 2) * ∑ a, Real.sqrt (f a) ^ 2 :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  simp only [one_mul, one_pow, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one,
    Real.sq_sqrt (hf _)] at hcs
  have h0 : 0 ≤ (∑ a, Real.sqrt (f a)) / Fintype.card α :=
    div_nonneg (Finset.sum_nonneg fun a _ => Real.sqrt_nonneg _) hn.le
  rw [← Real.sqrt_sq h0]
  refine Real.sqrt_le_sqrt ?_
  rw [div_pow, div_le_div_iff₀ (by positivity) hn]
  nlinarith [hcs]

section B

variable {Ω Coin : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- The probability that `B`'s two runs, forked at round `j`, collide. -/
noncomputable def forkPr (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x)
    (j : Fin (V.k x)) : ℝ :=
  Claims.Pr fun s : Ω × (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ((B H V P x hk).out (j, s)).isSome

/-- **`Pr[B]` is the average over the round guess** of the forks' collision probabilities. -/
theorem PrB_eq_avg (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg) (hk : 0 < V.k x) :
    PrB H V P x = (∑ j, forkPr H V P x hk j) / V.k x := by
  simp only [PrB, hk, ↓reduceDIte]
  rw [B_Pr_eq]
  exact Pr_guess (fun s : (B H V P x hk).S => ((B H V P x hk).out s).isSome)

/-- **Many finders, one root**: finders `i ∈ s` with weights `c i`, each with advantage at most the collision
probability of `B`'s fork at its round `j i`, and at most `m` finders at one round, give
`Σ_i √(c_i a_i) ≤ √((m · k(x) · Σ_i c_i) · Pr[B])`. -/
theorem collapse_B {I : Type} (s : Finset I) (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (hk : 0 < V.k x) (c a : I → ℝ) (hc : ∀ i ∈ s, 0 ≤ c i) (ha : ∀ i ∈ s, 0 ≤ a i) (j : I → Fin (V.k x))
    (haf : ∀ i ∈ s, a i ≤ forkPr H V P x hk (j i)) (m : ℕ)
    (hm : ∀ r, (s.filter fun i => j i = r).card ≤ m) :
    ∑ i ∈ s, Real.sqrt (c i * a i) ≤ Real.sqrt ((m * V.k x * ∑ i ∈ s, c i) * PrB H V P x) := by
  rw [PrB_eq_avg H V P x hk]
  exact collapseS s hk c a hc ha j (forkPr H V P x hk) (fun _ => Pr_nonneg _) haf m hm

end B

end Claims
