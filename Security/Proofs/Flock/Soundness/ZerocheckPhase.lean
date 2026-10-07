import Proofs.Flock.Soundness.LincheckPhase
import Proofs.Flock.Soundness.ListSize
import Proofs.Flock.Soundness.OpeningPhase

/-!
# The zerocheck phase (PROTOCOL.md §9.2)

For one codeword near the committed table whose witness makes the region claims true, the doomed
start says the block R1CS or the pin fails. The zerocheck then ends with the `a`, `b`, `c` claims
and the pin's extension all true with small probability:

* **R1CS.** A nonzero error bit at skip index `s₀` stays visible after the eq-reduction over the
  other address bits (`m − 13` random coordinates: the pinned coordinates' weights are independent
  over GF(2)). `Pcomb` vanishes on the skip nodes by construction, so either the prover's `P_c` is
  wrong at `z` or its `P_ab` is (degree 127). The residual rounds keep a false claim false
  (`2/|F|` each, given every `r_eq ≠ 1`, one event for all codewords), and a false final claim
  makes the `a` or the `b` claim false.
* **Pin.** A block whose pin bit is 0 makes the pin's extension at `ρ_out` a nonzero polynomial of
  degree `m − k_log` in the rounds' coins.

`zerocheck_phase_of_card` takes the union over a list of at most `L` codewords.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## Averages and coins -/

theorem avg_sum {C ι : Type} [Fintype C] (s : Finset ι) (f : ι → C → ℝ≥0∞) :
    avg (fun c => ∑ i ∈ s, f i c) = ∑ i ∈ s, avg (f i) := by
  simp only [avg, Finset.mul_sum]
  exact Finset.sum_comm

theorem value_coin_le_avg {α : Type} (p : α → Prop) (C : Type) [Fintype C] [Nonempty C]
    (k : C → Game α) (f : C → ℝ≥0∞) (h : ∀ c, value p (k c) ≤ f c) :
    value p (.coin C k) ≤ avg f := by
  rw [value_coin]; exact avg_mono h

theorem value_coin_le_of_forall {α : Type} (p : α → Prop) (C : Type) [Fintype C] [Nonempty C]
    (k : C → Game α) (ε : ℝ≥0∞) (h : ∀ c, value p (k c) ≤ ε) : value p (.coin C k) ≤ ε :=
  (value_coin_le_avg p C k _ h).trans (avg_const ε).le

omit [DecidableEq F] in
/-- A uniform vector on `Fin (n+1)` is a uniform head and an independent uniform tail. -/
theorem prCoin_cons {n : ℕ} (P : (Fin (n + 1) → F) → Prop) :
    prCoin P = avg (fun a : F => prCoin (fun v : Fin n → F => P (Fin.cons a v))) := by
  classical
  simp only [prCoin, avg]
  rw [← (Fin.consEquiv (fun _ => F)).sum_comp, Fintype.sum_prod_type]
  simp only [Finset.mul_sum, ← mul_assoc]
  refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun v _ => ?_
  congr 1
  simp only [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow]
  rw [pow_succ', ENNReal.mul_inv (Or.inr (by simp)) (Or.inl (by simp))]

omit [DecidableEq F] in
/-- A single coordinate of a uniform vector hits a given value with probability `1/|F|`. -/
theorem prCoin_coord_le {n : ℕ} (j : Fin n) (c : F) :
    prCoin (fun v : Fin n → F => v j = c) ≤ 1 / Fintype.card F := by
  classical
  let Q : MvPolynomial (Fin n) F := MvPolynomial.X j - MvPolynomial.C c
  have hQ : Q ≠ 0 := by
    intro h0
    have := congrArg (MvPolynomial.eval (fun _ => c + 1)) h0
    simp [Q] at this
  have hdeg : Q.totalDegree ≤ 1 := by
    refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ ?_)
    · rw [MvPolynomial.totalDegree_X]
    · rw [MvPolynomial.totalDegree_C]; omega
  calc prCoin (fun v : Fin n → F => v j = c)
      = prCoin (fun v : Fin n → F => MvPolynomial.eval v Q = 0) := by
        congr 1; funext v; simp [Q, sub_eq_zero]
    _ ≤ (Q.totalDegree : ℝ≥0∞) / Fintype.card F := prCoin_mvEval_eq_zero_le _ hQ
    _ ≤ 1 / Fintype.card F := by gcongr; exact_mod_cast hdeg

/-! ## Folding the lowest bit, and the equality weights -/

/-- Splitting off the lowest coordinate of an equality weight. -/
theorem eqAt_two_mul_add {R : Type} [CommRing R] {n : ℕ} (r : Fin (n + 1) → R) (y b : ℕ)
    (hb : b < 2) :
    eqAt r (2 * y + b) = (if b = 1 then r 0 else 1 + r 0) * eqAt (Fin.tail r) y := by
  unfold eqAt
  rw [Fin.prod_univ_succ]
  congr 1
  · have hb' : (2 * y + b).testBit 0 = decide (b = 1) := by
      rw [Nat.testBit_zero]; congr 1; apply propext; omega
    simp only [Fin.val_zero, hb']
    by_cases hb1 : b = 1 <;> simp [hb1]
  · refine Finset.prod_congr rfl fun j _ => ?_
    have htb : (2 * y + b).testBit (j.val + 1) = y.testBit j.val := by
      rw [Nat.testBit_succ]; congr 1; omega
    simp only [Fin.val_succ, htb, Fin.tail]

/-- Splitting off the lowest coordinate at an even index. -/
theorem eqAt_two_mul {R : Type} [CommRing R] {n : ℕ} (r : Fin (n + 1) → R) (y : ℕ) :
    eqAt r (2 * y) = (1 + r 0) * eqAt (Fin.tail r) y := by
  have := eqAt_two_mul_add r y 0 (by norm_num)
  simpa using this

/-- Splitting off the lowest coordinate at an odd index. -/
theorem eqAt_two_mul_add_one {R : Type} [CommRing R] {n : ℕ} (r : Fin (n + 1) → R) (y : ℕ) :
    eqAt r (2 * y + 1) = r 0 * eqAt (Fin.tail r) y := by
  have := eqAt_two_mul_add r y 1 (by norm_num)
  simpa using this

/-- Sums over `2^(n+1)` indices by pairs `(2y, 2y+1)`. -/
theorem sum_range_pairs {M : Type} [AddCommMonoid M] (n : ℕ) (f : ℕ → M) :
    ∑ x ∈ Finset.range (2 ^ (n + 1)), f x =
      ∑ y ∈ Finset.range (2 ^ n), (f (2 * y) + f (2 * y + 1)) := by
  rw [← Fin.sum_univ_eq_sum_range, ← Fin.sum_univ_eq_sum_range (fun y => f (2 * y) + f (2 * y + 1))]
  have h : 2 ^ n * 2 = 2 ^ (n + 1) := (pow_succ 2 n).symm
  rw [← (finCongr h).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Fin.sum_univ_two]
  simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, Fin.val_zero, Fin.val_one]
  congr 2 <;> ring

/-- Fold the lowest bit at `t`: `u'(y) = u(2y) + t·(u(2y+1) + u(2y))`. -/
def foldLo (t : F) (u : ℕ → F) : ℕ → F := fun y => u (2 * y) + t * (u (2 * y + 1) + u (2 * y))

/-- Fold every bit, lowest first, by the coins `ρ`. -/
def foldLoAll : {n : ℕ} → (Fin n → F) → (ℕ → F) → (ℕ → F)
  | 0, _, u => u
  | _ + 1, ρ, u => foldLoAll (Fin.tail ρ) (foldLo (ρ 0) u)

omit [Fintype F] [DecidableEq F] in
/-- Folding every bit evaluates the multilinear extension at the coins. -/
theorem foldLoAll_eq [CharP F 2] : ∀ (n : ℕ) (ρ : Fin n → F) (u : ℕ → F),
    foldLoAll ρ u 0 = ∑ x ∈ Finset.range (2 ^ n), eqAt ρ x * u x
  | 0, ρ, u => by simp [foldLoAll, eqAt]
  | n + 1, ρ, u => by
      rw [foldLoAll, foldLoAll_eq n, sum_range_pairs]
      refine Finset.sum_congr rfl fun y _ => ?_
      rw [eqAt_two_mul, eqAt_two_mul_add_one]
      simp only [foldLo]
      ring

/-! ## The residual rounds (Z6) -/

/-- The true eq-stripped quadratic of a residual round over `Σ_x eq(req, x)·u(x)·v(x)`:
`G(X) = Σ_y eq(req[1…], y)·(u(2y) + X·(u(2y+1) + u(2y)))·(v(2y) + X·(v(2y+1) + v(2y)))`. -/
noncomputable def resPoly {n : ℕ} (req : Fin (n + 1) → F) (u v : ℕ → F) : F[X] :=
  ∑ y ∈ Finset.range (2 ^ n), C (eqAt (Fin.tail req) y) *
    ((C (u (2 * y)) + C (u (2 * y + 1) + u (2 * y)) * Polynomial.X) *
      (C (v (2 * y)) + C (v (2 * y + 1) + v (2 * y)) * Polynomial.X))

omit [Fintype F] [DecidableEq F] in
theorem resPoly_natDegree_le {n : ℕ} (req : Fin (n + 1) → F) (u v : ℕ → F) :
    (resPoly req u v).natDegree ≤ 2 := by
  unfold resPoly
  refine natDegree_sum_le_of_forall_le _ _ fun y _ => ?_
  have hlin : ∀ a b : F, (C a + C b * Polynomial.X : F[X]).natDegree ≤ 1 := fun a b =>
    (natDegree_add_le _ _).trans (max_le (by simp) ((natDegree_C_mul_le _ _).trans (by simp)))
  refine (natDegree_C_mul_le _ _).trans ?_
  have h1 := hlin (u (2 * y)) (u (2 * y + 1) + u (2 * y))
  have h2 := hlin (v (2 * y)) (v (2 * y + 1) + v (2 * y))
  exact natDegree_mul_le.trans (by omega)

omit [Fintype F] [DecidableEq F] in
theorem resPoly_eval {n : ℕ} (req : Fin (n + 1) → F) (u v : ℕ → F) (t : F) :
    (resPoly req u v).eval t =
      ∑ y ∈ Finset.range (2 ^ n), eqAt (Fin.tail req) y * foldLo t u y * foldLo t v y := by
  simp only [resPoly, eval_finset_sum, eval_mul, eval_add, eval_C, eval_X, foldLo]
  exact Finset.sum_congr rfl fun y _ => by ring

omit [Fintype F] [DecidableEq F] in
theorem resPoly_sum [CharP F 2] {n : ℕ} (req : Fin (n + 1) → F) (u v : ℕ → F) :
    (1 + req 0) * (resPoly req u v).eval 0 + req 0 * (resPoly req u v).eval 1 =
      ∑ x ∈ Finset.range (2 ^ (n + 1)), eqAt req x * u x * v x := by
  rw [resPoly_eval, resPoly_eval, sum_range_pairs, Finset.mul_sum, Finset.mul_sum,
    ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [eqAt_two_mul, eqAt_two_mul_add_one]
  have h1 : ∀ w : ℕ → F, foldLo 1 w y = w (2 * y + 1) := fun w => by
    simp only [foldLo, one_mul]
    rw [add_comm (w (2 * y + 1)), ← add_assoc, Rounds.add_self, zero_add]
  have h0 : ∀ w : ℕ → F, foldLo 0 w y = w (2 * y) := fun w => by simp [foldLo]
  rw [h1 u, h1 v, h0 u, h0 v]
  ring

theorem residualRounds_succ (n : ℕ) (req : Fin (n + 1) → F) (claim : F) :
    residualRounds (n + 1) req claim = Game.send (F × F) fun g => Game.coin F fun ρ =>
      (residualRounds n (Fin.tail req) (residualStep (req 0) claim g.1 g.2 ρ)).bind
        fun r => Game.ret (Fin.cons ρ r.1, r.2) := by
  rw [residualRounds]
  rfl

/-- **The residual rounds** (Z6). From a false claim about `Σ_x eq(r, x)·u(x)·v(x)`, with every
`r_eq ≠ 1`, the rounds end with a true claim about the folded values with probability at most
`2n/|F|`, whatever the prover sends. -/
theorem value_residualRounds_le [CharP F 2] : ∀ (n : ℕ) (req : Fin n → F), (∀ j, req j ≠ 1) →
    ∀ (u v : ℕ → F) (claim : F), claim ≠ ∑ x ∈ Finset.range (2 ^ n), eqAt req x * u x * v x →
    value (fun r : (Fin n → F) × F => r.2 = foldLoAll r.1 u 0 * foldLoAll r.1 v 0)
      (residualRounds n req claim) ≤ ((2 * n : ℕ) : ℝ≥0∞) / Fintype.card F
  | 0, req, _, u, v, claim, hc => by
      show value _ (Game.ret (Fin.elim0, claim)) ≤ _
      rw [value_ret, if_neg]
      · exact zero_le
      · simpa [foldLoAll, eqAt] using hc
  | n + 1, req, hreq, u, v, claim, hc => by
      rw [residualRounds_succ, value_send]
      refine iSup_le fun g => ?_
      refine (value_coin_le _ F _ (fun ρ => residualStep (req 0) claim g.1 g.2 ρ =
          (resPoly req u v).eval ρ) (2 / Fintype.card F)
          (((2 * n : ℕ) : ℝ≥0∞) / Fintype.card F) ?_ fun ρ hρ => ?_).trans ?_
      · refine Rounds.residualRound_sound (req 0) claim (hreq 0) _ (resPoly_natDegree_le req u v)
          ?_ g.1 g.2
        rwa [resPoly_sum]
      · rw [value_bind_ret]
        refine (value_mono (fun r hr => ?_) _).trans (value_residualRounds_le n (Fin.tail req)
          (fun j => hreq j.succ) (foldLo ρ u) (foldLo ρ v) _ ?_)
        · simpa only [foldLoAll, Fin.tail_cons, Fin.cons_zero] using hr
        · rwa [resPoly_eval] at hρ
      · rw [ENNReal.div_add_div_same]
        gcongr
        push_cast
        ring_nf
        exact le_rfl

/-- An event on the rounds' coins alone has at most its probability over uniform coins. -/
theorem value_residualRounds_coin_le : ∀ (n : ℕ) (req : Fin n → F) (claim : F)
    (P : (Fin n → F) → Prop), value (fun r => P r.1) (residualRounds n req claim) ≤ prCoin P
  | 0, req, claim, P => by
      classical
      show value _ (Game.ret (Fin.elim0, claim)) ≤ _
      rw [value_ret]
      simp only [prCoin, avg, Fintype.card_fun, Fintype.card_fin, pow_zero, Nat.cast_one,
        inv_one, one_mul]
      rw [Fintype.sum_unique]
      have : (default : Fin 0 → F) = Fin.elim0 := Subsingleton.elim _ _
      rw [this]
  | n + 1, req, claim, P => by
      rw [residualRounds_succ, value_send, prCoin_cons]
      refine iSup_le fun g => ?_
      refine value_coin_le_avg _ F _ _ fun ρ => ?_
      rw [value_bind_ret]
      exact value_residualRounds_coin_le n _ _ (fun v => P (Fin.cons ρ v))

/-! ## The nodes, and polynomials through them -/

/-- The skip nodes `S` (§9). -/
noncomputable def nodeS (A : Arith F K) (hA : A.Correct) : Fin 64 → F := hA.nodes.choose

/-- The output coset `Λ` (§9.2). -/
noncomputable def nodeΛ (A : Arith F K) (hA : A.Correct) : Fin 64 → F :=
  hA.nodes.choose_spec.choose

theorem nodes_spec (A : Arith F K) (hA : A.Correct) :
    Function.Injective (nodeS A hA) ∧ Function.Injective (nodeΛ A hA) ∧
    (∀ s i, nodeS A hA s ≠ nodeΛ A hA i) ∧
    (∀ p : F[X], p.natDegree < 64 → ∀ ζ, ∑ s, A.lagS ζ s * p.eval (nodeS A hA s) = p.eval ζ) ∧
    (∀ p : F[X], p.natDegree < 64 → ∀ ζ, ∑ i, A.lagΛ ζ i * p.eval (nodeΛ A hA i) = p.eval ζ) ∧
    (∀ p : F[X], p.natDegree < 128 → (∀ s, p.eval (nodeS A hA s) = 0) →
      ∀ ζ, ∑ i, A.combW ζ i * p.eval (nodeΛ A hA i) = p.eval ζ) :=
  hA.nodes.choose_spec.choose_spec

/-- The polynomial of degree below 64 with the given values on the skip nodes. -/
noncomputable def polyS (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) : F[X] :=
  Lagrange.interpolate Finset.univ (nodeS A hA) vals

/-- The polynomial of degree below 64 with the given values on `Λ`. -/
noncomputable def polyΛ (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) : F[X] :=
  Lagrange.interpolate Finset.univ (nodeΛ A hA) vals

/-- `Pcomb` (§9.2): the polynomial of degree below 128 that is 0 on `S` and `vals` on `Λ`. -/
noncomputable def polyComb (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) : F[X] :=
  Lagrange.interpolate (Finset.univ : Finset (Fin 64 ⊕ Fin 64))
    (Sum.elim (nodeS A hA) (nodeΛ A hA)) (Sum.elim (fun _ => 0) vals)

theorem nodes_inj (A : Arith F K) (hA : A.Correct) :
    Function.Injective (Sum.elim (nodeS A hA) (nodeΛ A hA)) :=
  (nodes_spec A hA).1.sumElim (nodes_spec A hA).2.1 (nodes_spec A hA).2.2.1

theorem natDegree_interpolate_lt {ι : Type} [Fintype ι] [DecidableEq ι] (v : ι → F)
    (hv : Function.Injective v) (vals : ι → F) (hι : 0 < Fintype.card ι) :
    (Lagrange.interpolate Finset.univ v vals).natDegree < Fintype.card ι := by
  by_cases h : Lagrange.interpolate Finset.univ v vals = 0
  · rw [h, natDegree_zero]; exact hι
  · have hdeg := Lagrange.degree_interpolate_lt (s := Finset.univ) (r := vals) hv.injOn
    rw [Finset.card_univ] at hdeg
    exact (natDegree_lt_iff_degree_lt h).2 hdeg

theorem polyS_natDegree (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) :
    (polyS A hA vals).natDegree < 64 := by
  have := natDegree_interpolate_lt (nodeS A hA) (nodes_spec A hA).1 vals (by norm_num)
  rwa [Fintype.card_fin] at this

theorem polyΛ_natDegree (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) :
    (polyΛ A hA vals).natDegree < 64 := by
  have := natDegree_interpolate_lt (nodeΛ A hA) (nodes_spec A hA).2.1 vals (by norm_num)
  rwa [Fintype.card_fin] at this

theorem polyComb_natDegree (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) :
    (polyComb A hA vals).natDegree < 128 := by
  have := natDegree_interpolate_lt _ (nodes_inj A hA) (Sum.elim (fun _ => 0) vals)
    (by rw [Fintype.card_sum, Fintype.card_fin]; norm_num)
  rw [Fintype.card_sum, Fintype.card_fin] at this
  exact this

theorem polyS_node (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (s : Fin 64) :
    (polyS A hA vals).eval (nodeS A hA s) = vals s :=
  Lagrange.eval_interpolate_at_node vals (nodes_spec A hA).1.injOn (Finset.mem_univ s)

theorem polyΛ_node (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (i : Fin 64) :
    (polyΛ A hA vals).eval (nodeΛ A hA i) = vals i :=
  Lagrange.eval_interpolate_at_node vals (nodes_spec A hA).2.1.injOn (Finset.mem_univ i)

theorem polyComb_nodeS (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (s : Fin 64) :
    (polyComb A hA vals).eval (nodeS A hA s) = 0 := by
  have := Lagrange.eval_interpolate_at_node (Sum.elim (fun _ => (0 : F)) vals)
    (nodes_inj A hA).injOn (Finset.mem_univ (Sum.inl s))
  simp only [Sum.elim_inl] at this
  exact this

theorem polyComb_nodeΛ (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (i : Fin 64) :
    (polyComb A hA vals).eval (nodeΛ A hA i) = vals i := by
  have := Lagrange.eval_interpolate_at_node (Sum.elim (fun _ => (0 : F)) vals)
    (nodes_inj A hA).injOn (Finset.mem_univ (Sum.inr i))
  simp only [Sum.elim_inr] at this
  exact this

theorem polyS_eval (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (ζ : F) :
    (polyS A hA vals).eval ζ = ∑ s, A.lagS ζ s * vals s := by
  rw [← (nodes_spec A hA).2.2.2.1 _ (polyS_natDegree A hA vals) ζ]
  simp only [polyS_node]

theorem polyΛ_eval (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (ζ : F) :
    (polyΛ A hA vals).eval ζ = ∑ i, A.lagΛ ζ i * vals i := by
  rw [← (nodes_spec A hA).2.2.2.2.1 _ (polyΛ_natDegree A hA vals) ζ]
  simp only [polyΛ_node]

theorem polyComb_eval (A : Arith F K) (hA : A.Correct) (vals : Fin 64 → F) (ζ : F) :
    (polyComb A hA vals).eval ζ = ∑ i, A.combW ζ i * vals i := by
  rw [← (nodes_spec A hA).2.2.2.2.2 _ (polyComb_natDegree A hA vals)
    (polyComb_nodeS A hA vals) ζ]
  simp only [polyComb_nodeΛ]

/-! ## The witness by skip index -/

theorem two_pow_six_mul {m : ℕ} (hm : 6 ≤ m) : 2 ^ (m - 6) * 64 = 2 ^ m := by
  rw [show (64 : ℕ) = 2 ^ 6 by norm_num, ← pow_add, Nat.sub_add_cancel hm]

/-- The address `s + 64·x` of skip index `s` at address bits 6… equal to `x`. -/
def addr64 {m : ℕ} (hm : 6 ≤ m) (x : Fin (2 ^ (m - 6))) (s : Fin 64) : Fin (2 ^ m) :=
  ⟨s.val + 64 * x.val, by
    rw [← two_pow_six_mul hm]; have := x.isLt; have := s.isLt; omega⟩

/-- A claim evaluates by skip index: `Σ_x eq(x_claim, x)·Σ_s skip[s]·ω[s + 64x]`. -/
theorem claim_eval_skip {m : ℕ} (hm : 6 ≤ m) (c : Claim F m) (ω : Witness m) :
    c.eval ω = ∑ x : Fin (2 ^ (m - 6)), eqAt c.x x.val *
      ∑ s : Fin 64, c.skip s * ((ω (addr64 hm x s)).val : F) := by
  unfold Claim.eval
  rw [← (finCongr (two_pow_six_mul hm)).sum_comp, sum_fin_mul]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  have ha : (finCongr (two_pow_six_mul hm)) (finProdFinEquiv (x, s)) = addr64 hm x s := by
    ext; simp only [finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val, addr64]
  rw [ha]
  have hmod : (addr64 hm x s).val % 64 = s.val := by simp only [addr64]; omega
  have hdiv : (addr64 hm x s).val / 64 = x.val := by simp only [addr64]; omega
  have hskip : c.skip ⟨(addr64 hm x s).val % 64, Nat.mod_lt _ (by norm_num)⟩ = c.skip s :=
    congrArg c.skip (Fin.ext hmod)
  rw [hskip, hdiv]
  ring

/-- `ω̂_x(Y)`: the polynomial through the bits `ω[s + 64x]` on the skip nodes. -/
noncomputable def hatP (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (ω : Witness m)
    (x : Fin (2 ^ (m - 6))) : F[X] :=
  polyS A hA fun s => ((ω (addr64 hm x s)).val : F)

/-- A claim with the skip weights `L_s(ζ)` evaluates to `Σ_x eq(x_claim, x)·ω̂_x(ζ)`. -/
theorem claim_eval_hat (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (v ζ : F)
    (x : Fin (m - 6) → F) (ω : Witness m) :
    (⟨v, A.lagS ζ, x⟩ : Claim F m).eval ω =
      ∑ x' : Fin (2 ^ (m - 6)), eqAt x x'.val * (hatP A hA hm ω x').eval ζ := by
  rw [claim_eval_skip hm]
  simp only [hatP, polyS_eval]

/-- The R1CS error bit at an address: `(Az)·(Bz) + z`, zero exactly where the constraint holds. -/
def errBit (S : Statement) (w : Witness S.m) (a : Fin (2 ^ S.m)) : ZMod 2 :=
  aVec S w a * bVec S w a + w a

/-- The block R1CS fails at some address. -/
def R1CSFails (S : Statement) (w : Witness S.m) : Prop := ∃ a, errBit S w a ≠ 0

theorem r1cs_of_not_fails (S : Statement) (w : Witness S.m) (h : ¬ R1CSFails S w) :
    ∀ o i, (S.A₀.mulVec (S.block w o)) i * (S.B₀.mulVec (S.block w o)) i = S.block w o i := by
  intro o i
  have h0 : errBit S w (S.addr o i) = 0 := by
    by_contra hne; exact h ⟨_, hne⟩
  simp only [errBit, aVec_addr, bVec_addr] at h0
  have key : ∀ x y : ZMod 2, x + y = 0 → x = y := by decide
  exact key _ _ h0

/-! ## The true polynomials of the zerocheck -/

/-- The true `P_c(Y) = Σ_x eq(r, x)·ẑ_x(Y)` (degree below 64). -/
noncomputable def PcT (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (w : Witness m)
    (r : Fin (m - 6) → F) : F[X] :=
  ∑ x : Fin (2 ^ (m - 6)), C (eqAt r x.val) * hatP A hA hm w x

/-- The true `P_ab(Y) = Σ_x eq(r, x)·Â_x(Y)·B̂_x(Y)` (degree at most 126). -/
noncomputable def PabT (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 6 ≤ S.m)
    (w : Witness S.m) (r : Fin (S.m - 6) → F) : F[X] :=
  ∑ x : Fin (2 ^ (S.m - 6)), C (eqAt r x.val) *
    (hatP A hA hm (aVec S w) x * hatP A hA hm (bVec S w) x)

theorem PcT_natDegree (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (w : Witness m)
    (r : Fin (m - 6) → F) : (PcT A hA hm w r).natDegree ≤ 63 := by
  unfold PcT
  refine natDegree_sum_le_of_forall_le _ _ fun x _ => (natDegree_C_mul_le _ _).trans ?_
  have := polyS_natDegree A hA fun s => ((w (addr64 hm x s)).val : F)
  simp only [hatP]; omega

theorem PabT_natDegree (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 6 ≤ S.m)
    (w : Witness S.m) (r : Fin (S.m - 6) → F) : (PabT A hA S hm w r).natDegree ≤ 126 := by
  unfold PabT
  refine natDegree_sum_le_of_forall_le _ _ fun x _ => (natDegree_C_mul_le _ _).trans ?_
  refine natDegree_mul_le.trans ?_
  have h1 := polyS_natDegree A hA fun s => ((aVec S w (addr64 hm x s)).val : F)
  have h2 := polyS_natDegree A hA fun s => ((bVec S w (addr64 hm x s)).val : F)
  simp only [hatP]; omega

theorem PcT_eval (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (w : Witness m)
    (r : Fin (m - 6) → F) (ζ : F) :
    (PcT A hA hm w r).eval ζ = ∑ x : Fin (2 ^ (m - 6)), eqAt r x.val * (hatP A hA hm w x).eval ζ := by
  simp only [PcT, eval_finset_sum, eval_mul, eval_C]

theorem PabT_eval (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 6 ≤ S.m)
    (w : Witness S.m) (r : Fin (S.m - 6) → F) (ζ : F) :
    (PabT A hA S hm w r).eval ζ = ∑ x : Fin (2 ^ (S.m - 6)), eqAt r x.val *
      (hatP A hA hm (aVec S w) x).eval ζ * (hatP A hA hm (bVec S w) x).eval ζ := by
  simp only [PabT, eval_finset_sum, eval_mul, eval_C, mul_assoc]

/-- At a skip node, `P_ab + P_c` is the eq-reduction of the error bits there. -/
theorem errSum_eq [CharP F 2] (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 6 ≤ S.m)
    (w : Witness S.m) (r : Fin (S.m - 6) → F) (s : Fin 64) :
    (PabT A hA S hm w r + PcT A hA hm w r).eval (nodeS A hA s) =
      ∑ x : Fin (2 ^ (S.m - 6)), eqAt r x.val * ((errBit S w (addr64 hm x s)).val : F) := by
  rw [eval_add, PabT_eval, PcT_eval, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [hatP, polyS_node, errBit, val_cast_eq, map_add, map_mul]
  ring

/-! ## The three coin bounds -/

omit [Fintype F] [DecidableEq F] in
/-- In characteristic 2 the equality weights sum to 1. -/
theorem sum_eqAt [CharP F 2] : ∀ (n : ℕ) (r : Fin n → F),
    ∑ x ∈ Finset.range (2 ^ n), eqAt r x = 1
  | 0, r => by simp [eqAt]
  | n + 1, r => by
      rw [sum_range_pairs]
      simp only [eqAt_two_mul, eqAt_two_mul_add_one, ← add_mul]
      rw [add_assoc, Rounds.add_self, add_zero]
      simp only [one_mul]
      exact sum_eqAt n _

/-- **Z3.** A nonzero error bit survives the eq-reduction over address bits 6… at some skip node,
except with probability `(m − 13)/|F|`: the pinned coordinates' 128 weights are independent over
GF(2), and what remains is a nonzero multilinear polynomial in `r[13…]`. -/
theorem prCoin_zc3_le (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 13 ≤ S.m)
    (w : Witness S.m) (hfail : R1CSFails S w) :
    prCoin (fun rs : Fin (S.m - 13) → F => ∀ s : Fin 64,
      ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val *
        ((errBit S w (addr64 (le_trans (by norm_num) hm) x s)).val : F) = 0) ≤
      ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  have := hA.charTwo
  have h6 : 6 ≤ S.m := le_trans (by norm_num) hm
  obtain ⟨a₀, ha₀⟩ := hfail
  have hpow6 := two_pow_six_mul h6
  let s₀ : Fin 64 := ⟨a₀.val % 64, Nat.mod_lt _ (by norm_num)⟩
  let x₀ : Fin (2 ^ (S.m - 6)) := ⟨a₀.val / 64, by
    have := lt_of_lt_of_eq a₀.isLt hpow6.symm; omega⟩
  have hax : addr64 h6 x₀ s₀ = a₀ := Fin.ext (by simp only [addr64, x₀, s₀]; omega)
  have hpow : 2 ^ (S.m - 13) * 128 = 2 ^ (S.m - 6) := by
    rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add]; congr 1; omega
  let e : Fin (2 ^ (S.m - 6)) → F := fun x => ((errBit S w (addr64 h6 x s₀)).val : F)
  let ix : Fin (2 ^ (S.m - 13)) → Fin 128 → Fin (2 ^ (S.m - 6)) := fun x' b =>
    (finCongr hpow) (finProdFinEquiv (x', b))
  have hix : ∀ x' b, (ix x' b).val = b.val + 2 ^ 7 * x'.val := fun x' b => by
    simp only [ix, finCongr_apply, Fin.val_cast, finProdFinEquiv_apply_val]; norm_num
  let d : Fin (2 ^ (S.m - 13)) → F := fun x' => ∑ b : Fin 128, eqAt A.pinned b.val * e (ix x' b)
  have hsplit : ∀ rs : Fin (S.m - 13) → F,
      ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val * e x =
        ∑ x' : Fin (2 ^ (S.m - 13)), d x' * eqAt rs x'.val := by
    intro rs
    rw [← (finCongr hpow).sum_comp, sum_fin_mul]
    refine Finset.sum_congr rfl fun x' _ => ?_
    simp only [d, Finset.sum_mul]
    refine Finset.sum_congr rfl fun b _ => ?_
    have hb : b.val < 2 ^ 7 := by have := b.isLt; norm_num
    show eqAt (zcVector A rs) (ix x' b).val * e (ix x' b) = _
    rw [hix, eqAt_split_at 7 (S.m - 13) (by omega) (zcVector A rs) b.val x'.val hb]
    have hlo : (fun j : Fin 7 => zcVector A rs ⟨j.val, by omega⟩) = A.pinned :=
      funext fun j => by simp [zcVector, j.isLt]
    have hhi : (fun j : Fin (S.m - 13) => zcVector A rs ⟨7 + j.val, by omega⟩) = rs :=
      funext fun j => by simp [zcVector]
    rw [hlo, hhi]
    ring
  let x₀' : Fin (2 ^ (S.m - 13)) := ⟨x₀.val / 128, by
    have := lt_of_lt_of_eq x₀.isLt hpow.symm; omega⟩
  let b₀ : Fin 128 := ⟨x₀.val % 128, Nat.mod_lt _ (by norm_num)⟩
  have hx₀ : ix x₀' b₀ = x₀ := Fin.ext (by rw [hix]; simp only [x₀', b₀]; omega)
  have hd : d x₀' ≠ 0 := by
    intro h0
    have hc := hA.pinned_indep (fun b => errBit S w (addr64 h6 (ix x₀' b) s₀)) (by
      rw [← h0]
      exact Finset.sum_congr rfl fun b _ => by simp only [e]; ring)
    have hb := congrFun hc b₀
    simp only [hx₀, hax, Pi.zero_apply] at hb
    exact ha₀ hb
  calc prCoin (fun rs : Fin (S.m - 13) → F => ∀ s : Fin 64,
        ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val *
          ((errBit S w (addr64 h6 x s)).val : F) = 0)
      ≤ prCoin (fun rs : Fin (S.m - 13) → F =>
          ∑ x' : Fin (2 ^ (S.m - 13)), d x' * eqAt rs x'.val = 0) :=
        prCoin_mono fun rs h => by rw [← hsplit]; exact h s₀
    _ ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := prCoin_eqSum_le d x₀' hd

/-- `Pc(z)` as the verifier computes it from `round1_c` (Z5). -/
def pcOf (A : Arith F K) (msg : (Fin 64 → F) × (Fin 64 → F)) (z : F) : F :=
  ∑ i, A.lagΛ z i * msg.2 i

/-- The zerocheck's starting claim `Pcomb(z) + Pc(z)` (Z5). -/
def claim0 (A : Arith F K) (msg : (Fin 64 → F) × (Fin 64 → F)) (z : F) : F :=
  ∑ i, A.combW z i * (msg.1 i + msg.2 i) + pcOf A msg z

/-- **Z5, the skip point.** If the error does not vanish at some skip node, then at a live `z`
either the prover's `P_c` is off the true one or its `P_ab = Pcomb + P_c` is, since `Pcomb`
vanishes on the skip nodes by construction: probability at most `127/|F|` of neither. -/
theorem prCoin_zc5_le (A : Arith F K) (hA : A.Correct) (S : Statement) (h6 : 6 ≤ S.m)
    (w : Witness S.m) (r : Fin (S.m - 6) → F) (msg : (Fin 64 → F) × (Fin 64 → F)) (s₀ : Fin 64)
    (hE : ∑ x : Fin (2 ^ (S.m - 6)), eqAt r x.val * ((errBit S w (addr64 h6 x s₀)).val : F) ≠ 0) :
    prCoin (fun z : F => (PcT A hA h6 w r).eval z = pcOf A msg z ∧
      claim0 A msg z = (PabT A hA S h6 w r).eval z) ≤ 127 / Fintype.card F := by
  have := hA.charTwo
  have hpc : ∀ z, pcOf A msg z = (polyΛ A hA msg.2).eval z := fun z =>
    (polyΛ_eval A hA msg.2 z).symm
  have hcomb : ∀ z, ∑ i, A.combW z i * (msg.1 i + msg.2 i) =
      (polyComb A hA (fun i => msg.1 i + msg.2 i)).eval z := fun z =>
    (polyComb_eval A hA _ z).symm
  by_cases hP : polyΛ A hA msg.2 = PcT A hA h6 w r
  · have hne : polyComb A hA (fun i => msg.1 i + msg.2 i) ≠
        PabT A hA S h6 w r + PcT A hA h6 w r := by
      intro h
      have := congrArg (fun p => p.eval (nodeS A hA s₀)) h
      simp only [polyComb_nodeS] at this
      rw [errSum_eq] at this
      exact hE this.symm
    have hdeg1 := polyComb_natDegree A hA (fun i => msg.1 i + msg.2 i)
    have hdeg2 : (PabT A hA S h6 w r + PcT A hA h6 w r).natDegree ≤ 127 :=
      (natDegree_add_le _ _).trans (max_le ((PabT_natDegree A hA S h6 w r).trans (by norm_num))
        ((PcT_natDegree A hA h6 w r).trans (by norm_num)))
    refine (prCoin_mono fun z hz => ?_).trans
      ((prCoin_eval_eq_eval_le _ _ hne 127 (by omega) hdeg2).trans (by norm_num))
    obtain ⟨_, h2⟩ := hz
    rw [claim0, hcomb, hpc, hP] at h2
    rw [eval_add, ← h2, add_assoc, Rounds.add_self, add_zero]
  · refine (prCoin_mono fun z hz => ?_).trans
      ((prCoin_eval_eq_eval_le _ _ (Ne.symm hP) 63 (PcT_natDegree A hA h6 w r)
        (by have := polyΛ_natDegree A hA msg.2; omega)).trans ?_)
    · rw [hpc] at hz; exact hz.1
    · gcongr; norm_num

/-- `ρ_out` from the residual challenges: address bits `k_log …` (as `ZcOut.ρOut` reads them). -/
def ρOutOf (S : Statement) (ρ : Fin (S.m - 6) → F) : Fin S.nbl → F := fun j =>
  if h : S.kLog - 6 + j.val < S.m - 6 then ρ ⟨S.kLog - 6 + j.val, h⟩ else 0

/-- **The pin.** A block whose pin bit is 0 makes the pin's extension at `ρ_out` differ from 1,
except with probability `(m − k_log)/|F|` over the residual challenges. -/
theorem prCoin_pin_le (A : Arith F K) (hA : A.Correct) (S : Statement) (w : Witness S.m)
    (o₀ : Fin (2 ^ S.nbl)) (ho₀ : S.block w o₀ S.pin ≠ 1) :
    prCoin (fun ρ : Fin (S.m - 6) → F => pinMLE S w (ρOutOf S ρ) = 1) ≤
      ((S.nbl : ℕ) : ℝ≥0∞) / Fintype.card F := by
  have := hA.charTwo
  let d : Fin (2 ^ S.nbl) → F := fun o => ((S.block w o S.pin).val : F) - 1
  have hd : d o₀ ≠ 0 := by
    have key : ∀ x : ZMod 2, x ≠ 1 → x = 0 := by decide
    simp only [d, key _ ho₀, ZMod.val_zero, Nat.cast_zero, zero_sub, neg_ne_zero]
    exact one_ne_zero
  let f : Fin S.nbl → Fin (S.m - 6) := fun j => ⟨S.kLog - 6 + j.val, by
    have := S.hk; have := S.hk6; have hj : j.val < S.m - S.kLog := j.isLt; omega⟩
  have hf : Function.Injective f := fun j j' h => by
    simp only [f, Fin.mk.injEq] at h; exact Fin.ext (by omega)
  let R := MvPolynomial.rename f (eqPoly d)
  have hR : R ≠ 0 := fun h0 => eqPoly_ne_zero d o₀ hd
    ((MvPolynomial.rename_injective f hf) (by rw [map_zero]; exact h0))
  have hdeg : R.totalDegree ≤ S.nbl :=
    (MvPolynomial.totalDegree_rename_le f _).trans (totalDegree_eqPoly_le d)
  have hρ : ∀ ρ : Fin (S.m - 6) → F, ρOutOf S ρ = ρ ∘ f := fun ρ => funext fun j => by
    simp only [ρOutOf, Function.comp, f]
    rw [dif_pos]
  have hev : ∀ ρ : Fin (S.m - 6) → F, pinMLE S w (ρOutOf S ρ) = 1 ↔
      MvPolynomial.eval ρ R = 0 := by
    intro ρ
    rw [MvPolynomial.eval_rename, eval_eqPoly, ← hρ]
    have hsum1 : ∑ o : Fin (2 ^ S.nbl), eqAt (ρOutOf S ρ) o.val = 1 := by
      rw [Fin.sum_univ_eq_sum_range (fun o => eqAt (ρOutOf S ρ) o)]
      exact sum_eqAt _ _
    have hdsum : ∑ o, d o * eqAt (ρOutOf S ρ) o.val = pinMLE S w (ρOutOf S ρ) - 1 := by
      simp only [d, sub_mul, one_mul, Finset.sum_sub_distrib, hsum1, pinMLE]
      congr 1
      exact Finset.sum_congr rfl fun o _ => by ring
    rw [hdsum, sub_eq_zero]
  calc prCoin (fun ρ : Fin (S.m - 6) → F => pinMLE S w (ρOutOf S ρ) = 1)
      = prCoin (fun ρ : Fin (S.m - 6) → F => MvPolynomial.eval ρ R = 0) := by
        congr 1; funext ρ; exact propext (hev ρ)
    _ ≤ (R.totalDegree : ℝ≥0∞) / Fintype.card F := prCoin_mvEval_eq_zero_le R hR
    _ ≤ ((S.nbl : ℕ) : ℝ≥0∞) / Fintype.card F := by gcongr

/-! ## The game -/

/-- The zerocheck's verdict (Z7). -/
def zcResult (A : Arith F K) (m : ℕ) (rs : Fin (m - 13) → F)
    (msg : (Fin 64 → F) × (Fin 64 → F)) (z : F) (ρ : Fin (m - 6) → F) (claim : F) (v : F × F) :
    Option (ZcOut F m) :=
  if claim = v.1 * v.2 then some ⟨z, ρ, zcVector A rs, v.1, v.2, pcOf A msg z⟩ else none

/-- The zerocheck after `r[13…]` (Z4–Z7). -/
def zcRest (A : Arith F K) (m : ℕ) (rs : Fin (m - 13) → F) : Game (Option (ZcOut F m)) :=
  Game.send ((Fin 64 → F) × (Fin 64 → F)) fun msg => Game.coin F fun z =>
    (residualRounds (m - 6) (zcVector A rs) (claim0 A msg z)).bind fun r =>
      Game.send (F × F) fun v => Game.ret (zcResult A m rs msg z r.1 r.2 v)

theorem zerocheck_eq (A : Arith F K) (m : ℕ) :
    zerocheck A m = Game.coin (Fin 6 → F) fun _ => Game.coin (Fin (m - 13) → F) fun rs =>
      zcRest A m rs := rfl

/-- The `a`, `b` and `c` claims all hold for a witness at the zerocheck's output. -/
def ZcHolds (A : Arith F K) (S : Statement) (w : Witness S.m) (zc : ZcOut F S.m) : Prop :=
  (⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S w) ∧
    (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S w) ∧ (cClaim A zc).Holds w

/-- The values `ω̂_x(ζ)`, as a vector for the rounds. -/
noncomputable def hatVals (A : Arith F K) (hA : A.Correct) {m : ℕ} (hm : 6 ≤ m) (ω : Witness m)
    (ζ : F) : ℕ → F := fun x =>
  if hx : x < 2 ^ (m - 6) then (hatP A hA hm ω ⟨x, hx⟩).eval ζ else 0

theorem sum_hatVals {m : ℕ} (A : Arith F K) (hA : A.Correct) (hm : 6 ≤ m) (ω : Witness m)
    (ζ : F) (g : ℕ → F) :
    ∑ x ∈ Finset.range (2 ^ (m - 6)), g x * hatVals A hA hm ω ζ x =
      ∑ x : Fin (2 ^ (m - 6)), g x.val * (hatP A hA hm ω x).eval ζ := by
  rw [← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [hatVals, dif_pos x.isLt]

/-- **Z5–Z7, the R1CS path.** With the error visible at a skip node and every `r_eq ≠ 1`, the rest
of the zerocheck ends with the `a`, `b` and `c` claims all true with probability at most
`(127 + 2(m − 6))/|F|`. -/
theorem value_zcRest_r1cs_le (A : Arith F K) (hA : A.Correct) (S : Statement) (hm : 13 ≤ S.m)
    (w : Witness S.m) (rs : Fin (S.m - 13) → F) (hreq : ∀ j, zcVector A rs j ≠ 1) (s₀ : Fin 64)
    (hE : ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val *
      ((errBit S w (addr64 (le_trans (by norm_num) hm) x s₀)).val : F) ≠ 0) :
    value (fun o => ∃ zc, o = some zc ∧ ZcHolds A S w zc) (zcRest A S.m rs) ≤
      ((127 + 2 * (S.m - 6) : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  have := hA.charTwo
  have h6 : 6 ≤ S.m := le_trans (by norm_num) hm
  set r := zcVector A rs with hr_def
  unfold zcRest
  rw [value_send]
  refine iSup_le fun msg => ?_
  refine (value_coin_le _ F _ (fun z => (PcT A hA h6 w r).eval z = pcOf A msg z ∧
      claim0 A msg z = (PabT A hA S h6 w r).eval z) (127 / Fintype.card F)
      (((2 * (S.m - 6) : ℕ) : ℝ≥0∞) / Fintype.card F) (prCoin_zc5_le A hA S h6 w r msg s₀ hE)
      fun z hz => ?_).trans ?_
  · set uA := hatVals A hA h6 (aVec S w) z
    set uB := hatVals A hA h6 (bVec S w) z
    have hevalA : ∀ (va : F) (ρ : Fin (S.m - 6) → F),
        (⟨va, A.lagS z, ρ⟩ : Claim F S.m).eval (aVec S w) = foldLoAll ρ uA 0 := fun va ρ => by
      rw [claim_eval_hat A hA h6, foldLoAll_eq, sum_hatVals]
    have hevalB : ∀ (vb : F) (ρ : Fin (S.m - 6) → F),
        (⟨vb, A.lagS z, ρ⟩ : Claim F S.m).eval (bVec S w) = foldLoAll ρ uB 0 := fun vb ρ => by
      rw [claim_eval_hat A hA h6, foldLoAll_eq, sum_hatVals]
    by_cases hc : (PcT A hA h6 w r).eval z = pcOf A msg z
    · -- the start claim is false: the rounds, then the final check
      have hclaim : claim0 A msg z ≠ ∑ x ∈ Finset.range (2 ^ (S.m - 6)), eqAt r x * uA x * uB x := by
        intro h
        apply hz ⟨hc, ?_⟩
        rw [h, PabT_eval]
        rw [← Fin.sum_univ_eq_sum_range]
        refine Finset.sum_congr rfl fun x _ => ?_
        simp only [uA, uB, hatVals, dif_pos x.isLt]
      refine (value_bind_le _ _ _ (fun r' : (Fin (S.m - 6) → F) × F =>
          r'.2 ≠ foldLoAll r'.1 uA 0 * foldLoAll r'.1 uB 0) 0 (fun r' hr' => ?_)).trans ?_
      · rw [value_send]
        refine iSup_le fun v => ?_
        rw [value_ret, if_neg]
        rintro ⟨zc, hzc, hA', hB', _⟩
        simp only [zcResult] at hzc
        split_ifs at hzc with hchk
        simp only [Option.some.injEq] at hzc
        subst hzc
        unfold Claim.Holds at hA' hB'
        rw [hevalA] at hA'
        rw [hevalB] at hB'
        simp only at hA' hB'
        exact hr' (by rw [hchk, ← hA', ← hB'])
      · rw [add_zero]
        refine (value_mono (fun r' h => not_not.1 h) _).trans ?_
        exact value_residualRounds_le (S.m - 6) r hreq uA uB (claim0 A msg z) hclaim
    · -- the `c` claim is false at every output
      refine (value_bind_le _ _ _ (fun _ => True) 0 (fun r' _ => ?_)).trans ?_
      · rw [value_send]
        refine iSup_le fun v => ?_
        rw [value_ret, if_neg]
        rintro ⟨zc, hzc, _, _, hC⟩
        simp only [zcResult] at hzc
        split_ifs at hzc with hchk
        simp only [Option.some.injEq] at hzc
        subst hzc
        apply hc
        unfold Claim.Holds cClaim at hC
        rw [claim_eval_hat A hA h6, ← PcT_eval] at hC
        exact hC
      · rw [add_zero]
        calc value (fun a => ¬ True) _ ≤ value (fun _ => False) _ :=
              value_mono (fun _ h => h trivial) _
          _ = 0 := value_false _
          _ ≤ _ := zero_le
  · rw [ENNReal.div_add_div_same]
    gcongr
    push_cast
    exact le_rfl

/-- **The pin path.** A block whose pin bit is 0: the rest of the zerocheck ends with the pin's
extension at `ρ_out` equal to 1 with probability at most `(m − k_log)/|F|`. -/
theorem value_zcRest_pin_le (A : Arith F K) (hA : A.Correct) (S : Statement) (w : Witness S.m)
    (rs : Fin (S.m - 13) → F) (o₀ : Fin (2 ^ S.nbl)) (ho₀ : S.block w o₀ S.pin ≠ 1) :
    value (fun o => ∃ zc, o = some zc ∧ pinMLE S w (zc.ρOut S) = 1) (zcRest A S.m rs) ≤
      ((S.nbl : ℕ) : ℝ≥0∞) / Fintype.card F := by
  classical
  unfold zcRest
  rw [value_send]
  refine iSup_le fun msg => ?_
  refine value_coin_le_of_forall _ F _ _ fun z => ?_
  refine (value_bind_le _ _ _ (fun r' : (Fin (S.m - 6) → F) × F =>
      ¬ pinMLE S w (ρOutOf S r'.1) = 1) 0 (fun r' hr' => ?_)).trans ?_
  · rw [value_send]
    refine iSup_le fun v => ?_
    rw [value_ret, if_neg]
    rintro ⟨zc, hzc, hpin⟩
    simp only [zcResult] at hzc
    split_ifs at hzc with hchk
    simp only [Option.some.injEq] at hzc
    subst hzc
    exact hr' hpin
  · rw [add_zero]
    refine (value_mono (fun r' h => not_not.1 h) _).trans ?_
    exact (value_residualRounds_coin_le _ _ _ (fun ρ => pinMLE S w (ρOutOf S ρ) = 1)).trans
      (prCoin_pin_le A hA S w o₀ ho₀)

/-! ## The phase -/

/-- Some coordinate of `r[13…]` is 1: the one event charged once for every codeword. -/
theorem prCoin_req_one_le (m : ℕ) :
    prCoin (fun rs : Fin (m - 13) → F => ∃ j, rs j = 1) ≤ ((m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := by
  calc prCoin (fun rs : Fin (m - 13) → F => ∃ j, rs j = 1)
      ≤ prCoin (fun rs : Fin (m - 13) → F => ∃ j ∈ Finset.univ, rs j = 1) :=
        prCoin_mono fun rs ⟨j, hj⟩ => ⟨j, Finset.mem_univ j, hj⟩
    _ ≤ ∑ j ∈ (Finset.univ : Finset (Fin (m - 13))), prCoin (fun rs : Fin (m - 13) → F => rs j = 1) :=
        prCoin_exists_le _ _
    _ ≤ ∑ _j ∈ (Finset.univ : Finset (Fin (m - 13))), 1 / (Fintype.card F : ℝ≥0∞) :=
        Finset.sum_le_sum fun j _ => prCoin_coord_le j 1
    _ = ((m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one_div]

open Classical in
/-- **The zerocheck phase** (§9.2), for a list `W` of at most `L` messages: from a doomed start over
`W`, the zerocheck ends undoomed with probability at most
`(m − 13)/|F| + L·(m − 13 + 127 + 2(m − 6) + (m − k_log))/|F|`. -/
theorem zerocheck_phase_of_list (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hm : 13 ≤ S.m) (W : Set (Message F sch.k0 sch.level₀.logCols)) (extra : List (Claim F S.m))
    (hdoom : RepDoomedW A S sch W extra) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ W).card ≤ L) :
    value (fun o => ¬ DZcW A S sch W extra o) (zerocheck A S.m) ≤
      ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
        L * (((S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  classical
  have := hA.charTwo
  have h6 : 6 ≤ S.m := le_trans (by norm_num) hm
  set s := Finset.univ.filter fun M => M ∈ W
  let E : Message F sch.k0 sch.level₀.logCols → Option (ZcOut F S.m) → Prop := fun M o =>
    ∃ zc, o = some zc ∧ ((⟨zc.va, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (aVec S (wit A S sch M))
      ∧ (⟨zc.vb, A.lagS zc.z, zc.ρ⟩ : Claim F S.m).Holds (bVec S (wit A S sch M))
      ∧ (cClaim A zc).Holds (wit A S sch M) ∧ pinMLE S (wit A S sch M) (zc.ρOut S) = 1
      ∧ ∀ e ∈ extra, e.Holds (wit A S sch M))
  let G : (Fin (S.m - 13) → F) → Prop := fun rs => ∃ j, rs j = 1
  let B3 : Message F sch.k0 sch.level₀.logCols → (Fin (S.m - 13) → F) → Prop := fun M rs =>
    R1CSFails S (wit A S sch M) ∧ (∀ e ∈ extra, e.Holds (wit A S sch M)) ∧
      ∀ s₀ : Fin 64, ∑ x : Fin (2 ^ (S.m - 6)), eqAt (zcVector A rs) x.val *
        ((errBit S (wit A S sch M) (addr64 h6 x s₀)).val : F) = 0
  set εr : ℝ≥0∞ := ((127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) / Fintype.card F
  have hB3 : ∀ M, prCoin (B3 M) ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F := by
    intro M
    by_cases hr : R1CSFails S (wit A S sch M)
    · exact (prCoin_mono fun rs h => h.2.2).trans (prCoin_zc3_le A hA S hm _ hr)
    · calc prCoin (B3 M) ≤ prCoin (fun _ : Fin (S.m - 13) → F => False) :=
            prCoin_mono fun rs h => hr h.1
        _ = 0 := by simp [prCoin, avg]
        _ ≤ _ := zero_le
  calc value (fun o => ¬ DZcW A S sch W extra o) (zerocheck A S.m)
      ≤ value (fun o => ∃ M ∈ s, E M o) (zerocheck A S.m) := by
        refine value_mono (fun o h => ?_) _
        cases o with
        | none => exact absurd trivial h
        | some zc =>
          simp only [DZcW, not_forall, not_not] at h
          obtain ⟨M, hM, hall⟩ := h
          exact ⟨M, by simp [s, hM], zc, rfl, hall⟩
    _ ≤ avg (fun rs : Fin (S.m - 13) → F =>
          (if G rs then 1 else 0) + ∑ M ∈ s, ((if B3 M rs then 1 else 0) + εr)) := by
        rw [zerocheck_eq]
        refine value_coin_le_of_forall _ _ _ _ fun _ => ?_
        refine value_coin_le_avg _ _ _ _ fun rs => ?_
        by_cases hG : G rs
        · simp only [hG, if_true]; exact (value_le_one _ _).trans le_self_add
        · simp only [hG, if_false, zero_add]
          refine (value_exists_finset_le _ _ _).trans (Finset.sum_le_sum fun M hM => ?_)
          by_cases hB : B3 M rs
          · simp only [hB, if_true]; exact (value_le_one _ _).trans le_self_add
          · simp only [hB, if_false, zero_add]
            by_cases hext : ∀ e ∈ extra, e.Holds (wit A S sch M)
            · have hMc : M ∈ W := by simpa [s] using hM
              have hdM := hdoom M hMc
              by_cases hr : R1CSFails S (wit A S sch M)
              · obtain ⟨s₀, hs₀⟩ : ∃ s₀ : Fin 64, ∑ x : Fin (2 ^ (S.m - 6)),
                    eqAt (zcVector A rs) x.val *
                      ((errBit S (wit A S sch M) (addr64 h6 x s₀)).val : F) ≠ 0 := by
                  by_contra hno
                  push Not at hno
                  exact hB ⟨hr, hext, hno⟩
                have hreq : ∀ j, zcVector A rs j ≠ 1 := by
                  intro j
                  simp only [zcVector]
                  split_ifs with hj
                  · exact hA.pinned_ne_one _
                  · exact fun h1 => hG ⟨_, h1⟩
                refine (value_mono (fun o ⟨zc, ho, hA', hB', hC', _, _⟩ =>
                    ⟨zc, ho, hA', hB', hC'⟩) _).trans
                  ((value_zcRest_r1cs_le A hA S hm _ rs hreq s₀ hs₀).trans ?_)
                exact ENNReal.div_le_div_right (by exact_mod_cast (by omega :
                  127 + 2 * (S.m - 6) ≤ 127 + 2 * (S.m - 6) + (S.m - S.kLog))) _
              · obtain ⟨o₀, ho₀⟩ : ∃ o₀, S.block (wit A S sch M) o₀ S.pin ≠ 1 := by
                  by_contra hno
                  push Not at hno
                  exact hdM ⟨r1cs_of_not_fails S _ hr, hno, hext⟩
                refine (value_mono (fun o ⟨zc, ho, _, _, _, hp, _⟩ => ⟨zc, ho, hp⟩) _).trans
                  ((value_zcRest_pin_le A hA S _ rs o₀ ho₀).trans ?_)
                exact ENNReal.div_le_div_right (by exact_mod_cast (by simp only [Statement.nbl]; omega :
                  S.nbl ≤ 127 + 2 * (S.m - 6) + (S.m - S.kLog))) _
            · calc value (E M) _ ≤ value (fun _ => False) _ :=
                  value_mono (fun o ⟨_, _, _, _, _, _, he⟩ => hext he) _
                _ = 0 := value_false _
                _ ≤ εr := zero_le
    _ = prCoin G + ∑ M ∈ s, (prCoin (B3 M) + εr) := by
        rw [avg_add, avg_sum]
        have h1 : avg (fun rs : Fin (S.m - 13) → F => if G rs then (1 : ℝ≥0∞) else 0) =
            prCoin G := by unfold prCoin; congr; funext rs; congr
        have h2 : ∀ M, avg (fun rs : Fin (S.m - 13) → F =>
            (if B3 M rs then (1 : ℝ≥0∞) else 0) + εr) = prCoin (B3 M) + εr := fun M => by
          rw [avg_add, avg_const]; unfold prCoin; congr; funext rs; congr
        rw [h1]
        simp only [h2]
    _ ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
          ∑ _M ∈ s, (((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F + εr) := by
        gcongr with M hM
        · exact prCoin_req_one_le S.m
        · exact hB3 M
    _ ≤ ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
          L * (((S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) /
            Fintype.card F) := by
        rw [Finset.sum_const, nsmul_eq_mul]
        gcongr
        rw [ENNReal.div_add_div_same]
        push_cast
        ring_nf
        exact le_rfl

open Classical in
/-- **The zerocheck phase** for the codewords near the committed table: `zerocheck_phase_of_list`
at `closeMsgs`. -/
theorem zerocheck_phase_of_card (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hm : 13 ≤ S.m) (C : Oracle F) (extra : List (Claim F S.m))
    (hdoom : RepDoomed A S sch C extra) (L : ℕ)
    (hL : (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ L) :
    value (fun o => ¬ DZc A S sch C extra o) (zerocheck A S.m) ≤
      ((S.m - 13 : ℕ) : ℝ≥0∞) / Fintype.card F +
        L * (((S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog) : ℕ) : ℝ≥0∞) / Fintype.card F) := by
  simp only [dZc_eq]
  exact zerocheck_phase_of_list A hA S sch hm _ extra ((repDoomed_iff A S sch C extra).1 hdoom) L hL

end FlockSoundness
