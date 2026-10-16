import Mathlib

/-!
# Relative security for matmul PoUW: the definition, the masking algebra, the reduction's shape

Neekon Vafa's definition (`note:20261015T1810Z-draft-relative-security-def`) as RS-DEF froze it
(`note:20261015T1815Z-draft-rs-definition-frozen`, v1), in an abstract machine model, so that the construction and
proof workers' results land here as theorems or as named hypotheses.

* The machine model (`Machine`, `MatMachine`): algorithms from inputs to outputs, their semantics with the coins as
  part of the input, a per-run `cost` and its worst case `time`, with axioms: nonnegativity, monotonicity (running `f`
  first, or beside another algorithm, costs at least `f`), subadditivity of sequential and side-by-side composition,
  free routing, and the cost of a linear map (an `a × b` by `b × c` product costs at most `κ a b c`). `costed` shows
  the axioms are satisfiable with nontrivial prices.
* The definition (`Protocol`, with a commitment before the live coins, `CommitsFirst`; `Complete`; `Secure`, Neekon's
  `∃ M'` form; `SecureInf`, RS-DEF's frozen form against `Tuse`; `Core`, the per-run proof target; `Efficient`;
  `RelativelySecure`) with `ε`, `δ`, `γ`, `η`, `α`, `κ`; usefulness, completeness and the floors quantify over the
  protocol's `promise` (RS-DEF v2 §1.4, `A, B ∈ G_use`). Probabilities are uniform counts over the finite σ and coins
  (`prob`, `expect`). Core, the floors and Lemma R charge a run the program's size and its cost,
  `max(|M|, cost)` (`Machine.charge`, RS-DEF v2 after RS-RT's V1). `secureInf_of_core` (Core ⇒ Security for `δ > κ`),
  `rate_of_core` (RS-DEF's rate form) and `core_of_floor` (RS-DEF's Lemma R) are proved.
* The masking algebra, proved over any ring (`mask_peel`, `lowRank_peel`), exact over `ZMod (2^32)` and over any
  field, and recovering the integer product from its s32 residue when the entries are below `2^31`
  (`int8_product_exact`); a saturating int8 accumulator breaks the peel (`sat8_breaks_peel`).
* The reduction's shape (`reduction`, `M' = M ++ Q`): run the extractor (M's run and its committed blocks), keep the
  prover messages, peel and correct; `time_reduction_le` is `time M' ≤ time E + time peel + time correction`.
  `secure_of_extraction_floor` gives security from two named hypotheses, `Extraction` (RS-EXT's) and `Floor` (the floor
  agenda's), and `secure_fullRank` and `secure_lowRank` instantiate it with the explicit peels of `MatMachine`. With
  products typed by grid (`WideGridMatMachine`: grid products at `κG`, wide-by-grid products at `κW`), the full-rank
  peel costs `2κG n³` and the low-rank peel `(2κG + 2κW) n² r` (`secure_fullRank_grid`, `secure_lowRank_grid`,
  `secure_lowRank_advTime_grid`; RS-MODEL §5's `10 n² r` u at 1 u and 4 u).
* Route (c), edited instances (Daniel's rulings of 18:20Z and 18:28Z): correctness reads σ, an `Edit` adds σ-indexed
  noise to both operands, unpeeled; `edited` is the protocol (promised on the instances `use`, served by a
  product algorithm `m`), `edited_complete` its completeness, `honestEdit` costs `time m + 2κadd n²` beyond the proof
  layer and the noise expansion, and `relativelySecure_of_editFloor` gives relative security from the named hypothesis
  `EditFloor` (RS-D1E's floor), with `m` the worst-case product (`_wide`) or a grid product (`GridMatMachine`, `_grid`). The bit-pair projection's algebra
  (`bitPair_coefficient`) and the abstract core of RS-D1E's homogenization (`bitProj_rank_le`, `products_ge_half_rank`;
  `editedEntry_products_ge`: an edited prefix-sum entry with `r` live terms needs at least `r / 2` plain products,
  whatever the committed instance; `editedEntries_products_ge`: entries at distinct rows and columns, sharing their
  products, add up, Theorem B's accounting without merges) are proved.
* The Model B bridge (RS-MODEL): typed circuits of priced gates with free routing; `ModelB.machine` is a `Machine`
  whose time is `timeB`, `ModelB.matMachine` a `MatMachine` from priced circuits (`MatGates`, realized by a GEMM gate
  family), `matMachine120` the sm_120 instance at `price120` with the worst-case `κ = 16` u (circuits named `Gemm120`),
  and `gridMatMachine120` its width-typed refinement, the int8 tile GEMM at `κ = 1` u on `G_op = int8` (circuits named
  `Int8Gemm120`). `relativelySecure_edited120` is route (c) on the int8 grid at 1 u, `relativelySecure_edited120_wide`
  on unrestricted operands at 16 u. Machines take any additive `Price`. The service range-checks the committed instance
  into `G_use = [−(127 − s), 127 − s]` (`GridService`, RS-RT's V6), and with noise bounded by `s` the edits are int8
  (`edit_int8`; on the entries σ opens, `accepted_edit_int8`).
* B_θ (RS-MODEL §6.2, θ = 1/5): `price120θ` over instructions with their shared-memory words; B's circuits lift at
  `4/5` and every B_θ adversary has a B shadow, so B's floor carries at `4/5` (`serviceFloor_θ`). Route (c) there at
  `N = 0.8 n³` u takes any B floor `F_B ≤ n³` (`relativelySecure_edited120θ_of_floor`, `ε = U / (0.8 F_B) − 1`):
  Conjecture A's `n³` (`ε = γ`), Theorem E's form `(c / m_max)(φ n³ − sl)` over ℚ (`_rat`) and its proved case
  `c = ½`, `m_max = 2` (`_half`). The certified fraction (`certifiedFraction120θ`, `_half`): accepted runs are charged at
  least `(c φ / m_max) N` less the slack, except with probability `κ`. Model A (RS-MODEL §3) is ported over
  ℚ, and `timeθ_le_timeA` is `B_θ ≤ A` as the combination `(4/5)·ys + (1/5)·R6` of A's rows.
-/

namespace Pouw.RelativeSecurity

open Finset

section Probability

/-- The probability of an event under the uniform distribution on a finite type. -/
noncomputable def prob {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : ℝ := by
  classical exact ((univ.filter p).card : ℝ) / (Fintype.card Ω : ℝ)

theorem prob_nonneg {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : 0 ≤ prob p := by
  unfold prob
  positivity

theorem prob_le_one {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : prob p ≤ 1 := by
  classical
  unfold prob
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast card_filter_le _ _

theorem prob_mono {Ω : Type*} [Fintype Ω] {p q : Ω → Prop} (h : ∀ ω, p ω → q ω) : prob p ≤ prob q := by
  classical
  unfold prob
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast card_le_card fun ω hω => by
    simp only [mem_filter, mem_univ, true_and] at hω ⊢
    exact h ω hω

theorem mem_filter_univ_iff {Ω : Type*} [Fintype Ω] {p : Ω → Prop} (inst : DecidablePred p) (ω : Ω) :
    ω ∈ @Finset.filter _ p inst univ ↔ p ω := by simp

theorem prob_or_le {Ω : Type*} [Fintype Ω] (p q : Ω → Prop) : prob (fun ω => p ω ∨ q ω) ≤ prob p + prob q := by
  classical
  unfold prob
  rw [← add_div]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have key : ∀ s t u : Finset Ω, (∀ ω ∈ s, ω ∈ t ∨ ω ∈ u) → (s.card : ℝ) ≤ t.card + u.card := fun s t u h => by
    exact_mod_cast (card_le_card fun ω hω => mem_union.mpr (h ω hω)).trans (card_union_le t u)
  exact key _ _ _ fun ω hω => ((mem_filter_univ_iff _ ω).mp hω).imp ((mem_filter_univ_iff _ ω).mpr)
    ((mem_filter_univ_iff _ ω).mpr)

/-- The expectation of a real function under the uniform distribution on a finite type. -/
noncomputable def expect {Ω : Type*} [Fintype Ω] (f : Ω → ℝ) : ℝ := (∑ ω, f ω) / (Fintype.card Ω : ℝ)

/-- Markov's inequality for a nonnegative function. -/
theorem markov {Ω : Type*} [Fintype Ω] {f : Ω → ℝ} (hf : ∀ ω, 0 ≤ f ω) (X : ℝ) :
    X * prob (fun ω => X ≤ f ω) ≤ expect f := by
  classical
  unfold prob expect
  rw [mul_div_assoc']
  apply div_le_div_of_nonneg_right _ (by positivity)
  have key : ∀ s : Finset Ω, (∀ ω ∈ s, X ≤ f ω) → X * (s.card : ℝ) ≤ ∑ ω, f ω := fun s hs => by
    calc X * (s.card : ℝ) = ∑ _ω ∈ s, X := by rw [sum_const, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ ω ∈ s, f ω := sum_le_sum hs
      _ ≤ ∑ ω, f ω := sum_le_sum_of_subset_of_nonneg (subset_univ _) fun ω _ _ => hf ω
  exact key _ fun ω hω => (mem_filter_univ_iff _ ω).mp hω

theorem expect_mono {Ω : Type*} [Fintype Ω] {f g : Ω → ℝ} (h : ∀ ω, f ω ≤ g ω) : expect f ≤ expect g :=
  div_le_div_of_nonneg_right (sum_le_sum fun ω _ => h ω) (by positivity)

theorem expect_mul_left {Ω : Type*} [Fintype Ω] (k : ℝ) (f : Ω → ℝ) :
    expect (fun ω => k * f ω) = k * expect f := by
  unfold expect
  rw [← mul_sum, mul_div_assoc]

theorem expect_sum {Ω ι : Type*} [Fintype Ω] (s : Finset ι) (f : ι → Ω → ℝ) :
    expect (fun ω => ∑ l ∈ s, f l ω) = ∑ l ∈ s, expect (f l) := by
  unfold expect
  rw [Finset.sum_comm, Finset.sum_div]

theorem expect_le_of_le {Ω : Type*} [Fintype Ω] {f : Ω → ℝ} {k : ℝ} (hk : 0 ≤ k) (h : ∀ ω, f ω ≤ k) :
    expect f ≤ k := by
  unfold expect
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · simp [h0, hk]
  · rw [div_le_iff₀ (by exact_mod_cast h0)]
    calc ∑ ω, f ω ≤ ∑ _ω : Ω, k := sum_le_sum fun ω _ => h ω
      _ = k * (Fintype.card Ω : ℝ) := by rw [sum_const, card_univ, nsmul_eq_mul, mul_comm]

/-- The expectation over a product is the iterated expectation (fresh coins). -/
theorem expect_prod {α β : Type*} [Fintype α] [Fintype β] (f : α × β → ℝ) :
    expect f = expect fun a => expect fun b => f (a, b) := by
  unfold expect
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul, ← sum_div, div_div, mul_comm (Fintype.card β : ℝ)]

theorem expect_prod_swap {α β : Type*} [Fintype α] [Fintype β] (f : α × β → ℝ) :
    expect f = expect fun b => expect fun a => f (a, b) := by
  unfold expect
  rw [Fintype.sum_prod_type, Finset.sum_comm, Fintype.card_prod, Nat.cast_mul, ← sum_div, div_div]

/-- A probability is the expectation of the event's indicator. -/
theorem prob_eq_expect {Ω : Type*} [Fintype Ω] (p : Ω → Prop) [DecidablePred p] :
    prob p = expect fun ω => if p ω then 1 else 0 := by
  classical
  unfold prob expect
  rw [Finset.card_filter, Nat.cast_sum]
  congr 1
  refine Finset.sum_congr rfl fun ω _ => ?_
  by_cases h : p ω <;> simp [h]

/-- An event whose every slice at the second coordinate has probability at most `κ ≥ 0` has probability at most
`κ`. -/
theorem prob_le_of_slices {α β : Type*} [Fintype α] [Fintype β] (E : α × β → Prop) {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ b, prob (fun a => E (a, b)) ≤ κ) : prob E ≤ κ := by
  classical
  rw [prob_eq_expect, expect_prod_swap]
  exact expect_le_of_le hκ fun b => by rw [← prob_eq_expect (fun a => E (a, b))]; exact h b

/-- Probabilities are invariant under a relabelling of the sample space. -/
theorem prob_equiv {α β : Type*} [Fintype α] [Fintype β] (e : α ≃ β) (p : α → Prop) :
    prob p = prob fun b => p (e.symm b) := by
  classical
  unfold prob
  rw [Fintype.card_congr e]
  congr 2
  exact Finset.card_equiv e fun a => by simp

/-- The union bound over a finite family of events. -/
theorem prob_exists_le {Ω ι : Type*} [Fintype Ω] (s : Finset ι) (B : ι → Ω → Prop) :
    prob (fun ω => ∃ w ∈ s, B w ω) ≤ ∑ w ∈ s, prob (B w) := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    simp only [Finset.notMem_empty, false_and, exists_false, sum_empty]
    unfold prob
    simp
  | insert a s ha ih =>
    rw [Finset.sum_insert ha]
    refine (prob_mono fun ω hω => ?_).trans ((prob_or_le (B a) _).trans (add_le_add le_rfl ih))
    obtain ⟨w, hw, hB⟩ := hω
    rcases Finset.mem_insert.mp hw with rfl | hw
    · exact Or.inl hB
    · exact Or.inr ⟨w, hw, hB⟩

theorem one_sub_mul_exp_le (u : ℝ) : (1 - u) * Real.exp u ≤ 1 := by
  have h := Real.add_one_le_exp (-u)
  have h2 : Real.exp (-u) * Real.exp u = 1 := by rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  nlinarith [Real.exp_pos u]

end Probability

section Rounds

/-! ### Theorem S: one spot check per round, fresh coins (luck-loss round 4 §1; guarded, round 5's Theorem S_G)

A window is `R` rounds. Round `w` draws fresh coins, uniform on `Ω`, and runs one check; the window accepts iff every
check passes. The prover's play is a `RoundGame`, a tree over the earlier rounds' coins, so each round's deficiency
and check may depend on the whole past. The deficiency is fixed before the round's coins (straight-line rounds,
`PassBound`) or read off the round's guard path (`PathBound`). The proof is the supermartingale
`Π_w 1[pass_w] e^{(1−p) d_w}` as a finite product over rounds (`expect_weight_le`), then Markov. -/

/-- The coins of `n` rounds, each fresh and uniform on `Ω`. -/
@[reducible] def Paths (Ω : Type) : ℕ → Type
  | 0 => Unit
  | n + 1 => Ω × Paths Ω n

instance Paths.instFintype (Ω : Type) [Fintype Ω] : ∀ n, Fintype (Paths Ω n)
  | 0 => inferInstanceAs (Fintype Unit)
  | n + 1 => @instFintypeProd Ω (Paths Ω n) _ (Paths.instFintype Ω n)

/-- A prover's play over `n` rounds of coins on `Ω`: per round, its deficiency `d` and its check `pass` as
functions of the round's coins, then the rest of the play, which may depend on them. -/
inductive RoundGame (Ω : Type) : ℕ → Type
  | done : RoundGame Ω 0
  | round {n : ℕ} (d : Ω → ℝ) (pass : Ω → Prop) (next : Ω → RoundGame Ω n) : RoundGame Ω (n + 1)

namespace RoundGame

variable {Ω : Type}

/-- The window accepts: every round's check passes. -/
def accepts : {n : ℕ} → RoundGame Ω n → Paths Ω n → Prop
  | _, .done, _ => True
  | _, .round _ pass next, x => pass x.1 ∧ (next x.1).accepts x.2

/-- The hidden deficiency `Σ_w d_w` along the realized coins. -/
def deficiency : {n : ℕ} → RoundGame Ω n → Paths Ω n → ℝ
  | _, .done, _ => 0
  | _, .round d _ next, x => d x.1 + (next x.1).deficiency x.2

open Classical in
/-- The supermartingale's last value, `Π_w 1[pass_w] e^{c d_w}`. -/
noncomputable def weight (c : ℝ) : {n : ℕ} → RoundGame Ω n → Paths Ω n → ℝ
  | _, .done, _ => 1
  | _, .round d pass next, x => (if pass x.1 then Real.exp (c * d x.1) else 0) * (next x.1).weight c x.2

open Classical in
/-- The exponential form of the per-round bound (Part III.1): given the past, `E[1[pass_w] e^{c d_w}] ≤ 1`. -/
def ExpBound [Fintype Ω] (c : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    expect (fun a => if pass a then Real.exp (c * d a) else 0) ≤ 1 ∧ ∀ a, (next a).ExpBound c

/-- **Theorem S's per-round hypothesis**, named: the round's deficiency is fixed before its coins (straight-line
rounds), and given the past the check passes with probability at most `1 − (1 − p) d_w`. -/
def PassBound [Fintype Ω] (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ d₀, (∀ a, d a = d₀) ∧ prob pass ≤ 1 - (1 - p) * d₀) ∧ ∀ a, (next a).PassBound p

/-- **Theorem S_G's per-round hypothesis**, named: the round's coins split into guard paths `path a ∈ Λ`, the
round's deficiency is its path's implied deficiency `dt`, and given the past and the path the check passes with
probability at most `1 − (1 − p) dt`. -/
def PathBound [Fintype Ω] (p : ℝ) : {n : ℕ} → RoundGame Ω n → Prop
  | _, .done => True
  | _, .round d pass next =>
    (∃ (Λ : Type) (_ : Fintype Λ) (path : Ω → Λ) (dt : Λ → ℝ), (∀ a, d a = dt (path a)) ∧
      ∀ l, prob (fun a => pass a ∧ path a = l) ≤ (1 - (1 - p) * dt l) * prob fun a => path a = l) ∧
      ∀ a, (next a).PathBound p

theorem weight_nonneg (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n) (x : Paths Ω n), 0 ≤ G.weight c x
  | _, .done, _ => by simp [weight]
  | _, .round d pass next, x => by
    simp only [weight]
    exact mul_nonneg (by split_ifs <;> positivity) (weight_nonneg c (next x.1) x.2)

theorem weight_of_accepts (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n) (x : Paths Ω n), G.accepts x →
    G.weight c x = Real.exp (c * G.deficiency x)
  | _, .done, _, _ => by simp [weight, deficiency]
  | _, .round d pass next, x, h => by
    obtain ⟨h1, h2⟩ := h
    simp only [weight, deficiency, h1, ↓reduceIte, weight_of_accepts c (next x.1) x.2 h2, mul_add, Real.exp_add]

/-- **The supermartingale bound**, as a finite product over rounds: `E[Π_w 1[pass_w] e^{c d_w}] ≤ 1`. -/
theorem expect_weight_le [Fintype Ω] (c : ℝ) : ∀ {n : ℕ} (G : RoundGame Ω n), G.ExpBound c →
    expect (G.weight c) ≤ 1
  | _, .done, _ => by simp [expect, weight]
  | _, .round d pass next, h => by
    classical
    obtain ⟨h1, h2⟩ := h
    rw [expect_prod]
    calc expect (fun a => expect fun y => (round d pass next).weight c (a, y))
        = expect (fun a => (if pass a then Real.exp (c * d a) else 0) * expect ((next a).weight c)) := by
          congr 1
          funext a
          simp only [weight]
          exact expect_mul_left _ _
      _ ≤ expect (fun a => if pass a then Real.exp (c * d a) else 0) :=
          expect_mono fun a => mul_le_of_le_one_right (by split_ifs <;> positivity)
            (expect_weight_le c (next a) (h2 a))
      _ ≤ 1 := h1

/-- The per-round pass bound gives the exponential form at `c = 1 − p`. -/
theorem expBound_of_passBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PassBound p →
    G.ExpBound (1 - p)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨d₀, hd, hpass⟩, h2⟩ := h
    refine ⟨?_, fun a => expBound_of_passBound (next a) (h2 a)⟩
    have e : (fun a => if pass a then Real.exp ((1 - p) * d a) else 0) =
        fun a => Real.exp ((1 - p) * d₀) * if pass a then 1 else 0 := by
      funext a
      rw [hd a]
      split_ifs <;> simp
    rw [e, expect_mul_left, ← prob_eq_expect]
    calc Real.exp ((1 - p) * d₀) * prob pass ≤ Real.exp ((1 - p) * d₀) * (1 - (1 - p) * d₀) :=
          mul_le_mul_of_nonneg_left hpass (Real.exp_pos _).le
      _ ≤ 1 := by rw [mul_comm]; exact one_sub_mul_exp_le _

/-- The per-path pass bound gives the exponential form at `c = 1 − p` (round 5 §1(a)). -/
theorem expBound_of_pathBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PathBound p →
    G.ExpBound (1 - p)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨⟨Λ, _, path, dt, hd, hl⟩, h2⟩ := h
    refine ⟨?_, fun a => expBound_of_pathBound (next a) (h2 a)⟩
    have e : (fun a => if pass a then Real.exp ((1 - p) * d a) else 0) =
        fun a => ∑ l, Real.exp ((1 - p) * dt l) * if pass a ∧ path a = l then 1 else 0 := by
      funext a
      rw [Finset.sum_eq_single_of_mem (path a) (mem_univ _) fun l _ hl => by
        simp [Ne.symm hl]]
      rw [hd a]
      by_cases hp : pass a <;> simp [hp]
    have e2 : (fun _ : Ω => (1 : ℝ)) = fun a => ∑ l, if path a = l then 1 else 0 := by
      funext a
      rw [Finset.sum_eq_single_of_mem (path a) (mem_univ _) fun l _ hl => by simp [Ne.symm hl]]
      simp
    rw [e, expect_sum]
    calc ∑ l, expect (fun a => Real.exp ((1 - p) * dt l) * if pass a ∧ path a = l then 1 else 0)
        = ∑ l, Real.exp ((1 - p) * dt l) * prob (fun a => pass a ∧ path a = l) := by
          refine Finset.sum_congr rfl fun l _ => ?_
          rw [expect_mul_left, prob_eq_expect]
      _ ≤ ∑ l, Real.exp ((1 - p) * dt l) * ((1 - (1 - p) * dt l) * prob fun a => path a = l) :=
          Finset.sum_le_sum fun l _ => mul_le_mul_of_nonneg_left (hl l) (Real.exp_pos _).le
      _ ≤ ∑ l, prob fun a => path a = l := Finset.sum_le_sum fun l _ => by
          rw [← mul_assoc, mul_comm (Real.exp _)]
          exact mul_le_of_le_one_left (prob_nonneg _) (one_sub_mul_exp_le _)
      _ = expect fun _ : Ω => (1 : ℝ) := by
          rw [e2, expect_sum]
          exact Finset.sum_congr rfl fun l _ => prob_eq_expect _
      _ ≤ 1 := expect_le_of_le zero_le_one fun _ => le_rfl

/-- **Theorem S from the exponential form**: for `c ≥ 0`, `Pr[accept ∧ Σ_w d_w ≥ D] ≤ e^{−c D}` (Markov on the
supermartingale's last value, which is `e^{c Σ d}` on acceptance). -/
theorem prob_accepts_le [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) {n : ℕ} (G : RoundGame Ω n) (hG : G.ExpBound c) (D : ℝ) :
    prob (fun x => G.accepts x ∧ D ≤ G.deficiency x) ≤ Real.exp (-(c * D)) := by
  have hm := markov (G.weight_nonneg c) (Real.exp (c * D))
  have hsub : prob (fun x => G.accepts x ∧ D ≤ G.deficiency x) ≤
      prob fun x => Real.exp (c * D) ≤ G.weight c x :=
    prob_mono fun x hx => by
      rw [G.weight_of_accepts c x hx.1]
      exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hx.2 hc)
  have hpos := Real.exp_pos (c * D)
  calc prob (fun x => G.accepts x ∧ D ≤ G.deficiency x)
      ≤ prob fun x => Real.exp (c * D) ≤ G.weight c x := hsub
    _ = (Real.exp (c * D))⁻¹ * (Real.exp (c * D) * prob fun x => Real.exp (c * D) ≤ G.weight c x) :=
        (inv_mul_cancel_left₀ hpos.ne' _).symm
    _ ≤ (Real.exp (c * D))⁻¹ * 1 :=
        mul_le_mul_of_nonneg_left (hm.trans (G.expect_weight_le c hG)) (inv_nonneg.mpr hpos.le)
    _ = Real.exp (-(c * D)) := by rw [mul_one, Real.exp_neg]

/-- Theorem S with binding and zero knowledge: if every accepted run is accepted by the game or breaks binding or
zero knowledge in some round, the union bound adds `R (δb + δz)`. -/
theorem prob_accept_le [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) {R : ℕ} (G : RoundGame Ω R) (hG : G.ExpBound c)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-(c * D)) + R * (δb + δz) := by
  have hbad : prob (fun x => ∃ w ∈ (univ : Finset (Fin R)), Bind w x ∨ ZK w x) ≤ R * (δb + δz) := by
    refine (prob_exists_le _ _).trans ?_
    calc ∑ w : Fin R, prob (fun x => Bind w x ∨ ZK w x) ≤ ∑ _w : Fin R, (δb + δz) :=
          Finset.sum_le_sum fun w _ => (prob_or_le _ _).trans (add_le_add (hBind w) (hZK w))
      _ = R * (δb + δz) := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  refine (prob_mono fun x hx => ?_).trans ((prob_or_le _ _).trans (add_le_add (G.prob_accepts_le hc hG D) hbad))
  rcases hacc x hx.1 with h | ⟨w, hw⟩
  · exact Or.inl ⟨h, hx.2⟩
  · exact Or.inr ⟨w, mem_univ _, hw⟩

/-- **Theorem S** (luck-loss round 4 §1, refereed in Part III): over `R` rounds with fresh coins, one check a round,
accepting iff every check passes, given the per-round pass bound `1 − (1 − p) d_w` (`PassBound`, named) and
binding and zero-knowledge failures of probability at most `δb` and `δz` per round,
`Pr[accept ∧ Σ_w d_w ≥ D] ≤ e^{−(1−p) D} + R (δb + δz)`. -/
theorem theoremS [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PassBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-((1 - p) * D)) + R * (δb + δz) :=
  G.prob_accept_le (by linarith) (G.expBound_of_passBound hG) acc Bind ZK hBind hZK hacc D

/-- **Theorem S_G** (round 5 §1(a)): the same bound with the deficiency read off each round's guard path
(`PathBound`, named), so `Σ_w d̃_w` is the implied deficiency along the realized paths. -/
theorem theoremS_G [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PathBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (D : ℝ) :
    prob (fun x => acc x ∧ D ≤ G.deficiency x) ≤ Real.exp (-((1 - p) * D)) + R * (δb + δz) :=
  G.prob_accept_le (by linarith) (G.expBound_of_pathBound hG) acc Bind ZK hBind hZK hacc D

/-- **The window form** (Theorem S's cost line, and S_G's with Lemma RD): if every run's cost is at least
`(R − Σ_w d_w) F₀ − V − R sl` (Theorem E½'s per-round `x_w F₀ − L(M + K)` summed, with `V` the noise reads priced
on the realized paths, `0` for straight-line rounds), then accepted runs costing below `(R − D) F₀ − V − R sl` have
probability at most `e^{−(1−p) D} + R (δb + δz)`. -/
theorem window_le [Fintype Ω] {p : ℝ} (hp : p ≤ 1) {R : ℕ} (G : RoundGame Ω R) (hG : G.PathBound p)
    (acc : Paths Ω R → Prop) (Bind ZK : Fin R → Paths Ω R → Prop) {δb δz : ℝ}
    (hBind : ∀ w, prob (Bind w) ≤ δb) (hZK : ∀ w, prob (ZK w) ≤ δz)
    (hacc : ∀ x, acc x → G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (cost V : Paths Ω R → ℝ) {F₀ sl : ℝ}
    (hF₀ : 0 < F₀) (hcost : ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ cost x) (D : ℝ) :
    prob (fun x => acc x ∧ cost x < (R - D) * F₀ - V x - R * sl) ≤
      Real.exp (-((1 - p) * D)) + R * (δb + δz) := by
  refine (prob_mono fun x hx => ⟨hx.1, ?_⟩).trans (G.theoremS_G hp hG acc Bind ZK hBind hZK hacc D)
  have h := (hcost x).trans_lt hx.2
  have h' : (R - G.deficiency x) * F₀ < (R - D) * F₀ := by linarith
  have := (mul_lt_mul_iff_of_pos_right hF₀).mp h'
  linarith

/-- Theorem S's straight-line rounds are guarded rounds with one path. -/
theorem pathBound_of_passBound [Fintype Ω] {p : ℝ} : ∀ {n : ℕ} (G : RoundGame Ω n), G.PassBound p →
    G.PathBound p
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    obtain ⟨⟨d₀, hd, hpass⟩, h2⟩ := h
    refine ⟨⟨Unit, inferInstance, fun _ => (), fun _ => d₀, hd, fun l => ?_⟩,
      fun a => pathBound_of_passBound (next a) (h2 a)⟩
    have e1 : prob (fun a => pass a ∧ () = l) = prob pass := by congr 1; funext a; simp
    have e2 : prob (fun _ : Ω => () = l) = prob fun _ : Ω => True := by
      congr 1
      funext a
      exact eq_iff_iff.mpr ⟨fun _ => trivial, fun _ => Subsingleton.elim _ _⟩
    rw [e1, e2]
    rcases isEmpty_or_nonempty Ω with hΩ | hΩ
    · have h0 : ∀ q : Ω → Prop, prob q = 0 := fun q => by
        unfold prob
        simp
      rw [h0, h0, mul_zero]
    · have h1 : prob (fun _ : Ω => True) = 1 := by
        unfold prob
        simp
      rw [h1, mul_one]
      exact hpass

end RoundGame

end Rounds

section Machine

/-- A machine model: algorithms from inputs `I` to outputs `O` (coins are part of the input), what they compute, and
their cost. The axioms are the properties of `time` the reduction uses. -/
structure Machine where
  /-- The algorithms from `I` to `O`. -/
  Alg : Type → Type → Type
  /-- What an algorithm computes. -/
  eval : ∀ {I O : Type}, Alg I O → I → O
  /-- Its cost (an abstract `time`: priced work, worst case over inputs and coins). -/
  time : ∀ {I O : Type}, Alg I O → ℝ
  /-- One run's cost (RS-DEF §1.2's `cost_P(x; σ, ρ)`), which `time` bounds. -/
  cost : ∀ {I O : Type}, Alg I O → I → ℝ
  /-- The program's size `|M|` (RS-DEF's T4), charged with every run (`charge`, RS-RT's V1). -/
  size : ∀ {I O : Type}, Alg I O → ℝ
  /-- Run `f`, then `g` on its output. -/
  seq : ∀ {I O P : Type}, Alg I O → Alg O P → Alg I P
  /-- Run `f` and `g` on the same input. -/
  pair : ∀ {I O P : Type}, Alg I O → Alg I P → Alg I (O × P)
  /-- The projections (routing). -/
  fst : ∀ {I J : Type}, Alg (I × J) I
  snd : ∀ {I J : Type}, Alg (I × J) J
  eval_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P) (x : I), eval (seq f g) x = eval g (eval f x)
  eval_pair : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P) (x : I), eval (pair f g) x = (eval f x, eval g x)
  eval_fst : ∀ {I J : Type} (x : I × J), eval (fst (I := I) (J := J)) x = x.1
  eval_snd : ∀ {I J : Type} (x : I × J), eval (snd (I := I) (J := J)) x = x.2
  /-- Costs are nonnegative. -/
  time_nonneg : ∀ {I O : Type} (f : Alg I O), 0 ≤ time f
  /-- `time` is the worst case: every run costs at most `time`. -/
  cost_le_time : ∀ {I O : Type} (f : Alg I O) (x : I), cost f x ≤ time f
  /-- T4: a program's time is at least its size. -/
  size_le_time : ∀ {I O : Type} (f : Alg I O), size f ≤ time f
  size_nonneg : ∀ {I O : Type} (f : Alg I O), 0 ≤ size f
  /-- Subadditivity of sequential composition. -/
  time_seq_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time (seq f g) ≤ time f + time g
  /-- Monotonicity: running `f` first costs at least `f`. (Not `g`: `g`'s worst case may lie outside `f`'s range.) -/
  le_time_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time f ≤ time (seq f g)
  /-- Subadditivity of side-by-side composition. -/
  time_pair_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time (pair f g) ≤ time f + time g
  /-- Monotonicity: running `f` beside `g` costs at least each. -/
  le_time_pair_left : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time f ≤ time (pair f g)
  le_time_pair_right : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time g ≤ time (pair f g)
  /-- Routing is free. -/
  time_fst : ∀ {I J : Type}, time (fst (I := I) (J := J)) = 0
  time_snd : ∀ {I J : Type}, time (snd (I := I) (J := J)) = 0

/-- `n × m` matrices over `R`, indexed by `Fin`. -/
abbrev Mat (R : Type) (a b : ℕ) : Type := Matrix (Fin a) (Fin b) R

/-- A machine model with matrix arithmetic over a ring `R`: the cost of a linear map, `X ↦ X Y` for an `a × b` by
`b × c` product, is at most `κ a b c` (`κ` the price of a multiply-accumulate), and an entrywise addition or
subtraction of `a × b` matrices costs at most `κadd a b`. -/
structure MatMachine (R : Type) [Ring R] extends Machine where
  κ : ℝ
  κadd : ℝ
  mul : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  add : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  sub : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  eval_mul : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), eval (mul a b c) x = x.1 * x.2
  eval_add : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (add a b) x = x.1 + x.2
  eval_sub : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (sub a b) x = x.1 - x.2
  /-- The cost of a linear map. -/
  time_mul_le : ∀ a b c : ℕ, time (mul a b c) ≤ κ * ((a : ℝ) * b * c)
  time_add_le : ∀ a b : ℕ, time (add a b) ≤ κadd * ((a : ℝ) * b)
  time_sub_le : ∀ a b : ℕ, time (sub a b) ≤ κadd * ((a : ℝ) * b)

namespace Machine

variable (Mc : Machine)

/-- Run `f` and `g` on the same input and combine their outputs with `h`. -/
def lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) : Mc.Alg I Q :=
  Mc.seq (Mc.pair f g) h

theorem eval_lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) (x : I) :
    Mc.eval (Mc.lift2 f g h) x = Mc.eval h (Mc.eval f x, Mc.eval g x) := by
  rw [lift2, Mc.eval_seq, Mc.eval_pair]

theorem time_lift2_le {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) :
    Mc.time (Mc.lift2 f g h) ≤ Mc.time f + Mc.time g + Mc.time h :=
  (Mc.time_seq_le _ _).trans (by linarith [Mc.time_pair_le f g])

theorem time_seq_eq_zero {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} (hf : Mc.time f = 0)
    (hg : Mc.time g = 0) : Mc.time (Mc.seq f g) = 0 :=
  le_antisymm ((Mc.time_seq_le f g).trans (by rw [hf, hg, add_zero])) (Mc.time_nonneg _)

/-! Cost bounds in goal-directed form, for `apply_rules`. -/

theorem time_seq_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.seq f g) ≤ a + b :=
  (Mc.time_seq_le f g).trans (add_le_add hf hg)

theorem time_pair_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg I P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.pair f g) ≤ a + b :=
  (Mc.time_pair_le f g).trans (add_le_add hf hg)

theorem time_fst_le {I J : Type} : Mc.time (Mc.fst (I := I) (J := J)) ≤ 0 := Mc.time_fst.le

theorem time_snd_le {I J : Type} : Mc.time (Mc.snd (I := I) (J := J)) ≤ 0 := Mc.time_snd.le

/-- **A run's charge** (RS-DEF v2's `max(|M|, cost_M(σ, ρ))`, RS-RT's V1): the program's size and the run's cost.
Charging the run alone lets a coin-indexed table of accepted answers pass at `O(λ)` per run. -/
def charge {I O : Type} (f : Mc.Alg I O) (x : I) : ℝ := max (Mc.size f) (Mc.cost f x)

theorem charge_le_time {I O : Type} (f : Mc.Alg I O) (x : I) : Mc.charge f x ≤ Mc.time f :=
  max_le (Mc.size_le_time f) (Mc.cost_le_time f x)

theorem charge_nonneg {I O : Type} (f : Mc.Alg I O) (x : I) : 0 ≤ Mc.charge f x :=
  le_max_of_le_left (Mc.size_nonneg f)

theorem size_le_charge {I O : Type} (f : Mc.Alg I O) (x : I) : Mc.size f ≤ Mc.charge f x := le_max_left _ _

end Machine

/-- The axioms are satisfiable with nontrivial prices: algorithms are functions annotated with a cost, composition
adds costs, routing is free, and an `a × b` by `b × c` product costs exactly `a b c`. (This model has no floor: any
function is an algorithm. A floor needs a syntactic model, RS-MODEL's.) -/
def costed : Machine where
  Alg I O := (I → O) × ℕ
  eval f := f.1
  time f := f.2
  cost f _ := f.2
  size f := f.2
  seq f g := (g.1 ∘ f.1, f.2 + g.2)
  pair f g := (fun x => (f.1 x, g.1 x), f.2 + g.2)
  fst := (Prod.fst, 0)
  snd := (Prod.snd, 0)
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := Nat.cast_nonneg _
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := Nat.cast_nonneg _
  time_seq_le _ _ := by push_cast; rfl
  le_time_seq _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  time_pair_le _ _ := by push_cast; rfl
  le_time_pair_left _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  le_time_pair_right f _ := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) f.2]
  time_fst := by simp
  time_snd := by simp

/-- `costed` with matrix arithmetic over a ring: `κ = κadd = 1`. -/
def costedMat (R : Type) [Ring R] : MatMachine R where
  toMachine := costed
  κ := 1
  κadd := 1
  mul a b c := (fun x => x.1 * x.2, a * b * c)
  add a b := (fun x => x.1 + x.2, a * b)
  sub a b := (fun x => x.1 - x.2, a * b)
  eval_mul _ _ _ _ := rfl
  eval_add _ _ _ := rfl
  eval_sub _ _ _ := rfl
  time_mul_le a b c := by simp [costed]
  time_add_le a b := by simp [costed]
  time_sub_le a b := by simp [costed]

end Machine

section Definition

/-- A proof-of-useful-work protocol in a machine model, for instances `Inst` with answers `Out` (RS-DEF §1.3, one
round): the prover sends a commitment to its instance before any coin, the verifier sends live coins σ, the prover
answers, and the verifier decides, deterministically in its view. `Solve` also outputs `C`, kept by the prover. The
`|π| ≤ 10λ` bound of Neekon's draft is dropped, as RS-DEF does (security never uses it). -/
structure Protocol (Mc : Machine) (Inst Out : Type) where
  /-- The verifier's live coins σ, `λ` bits. -/
  Seed : Type
  [instSeed : Fintype Seed]
  lam : ℕ
  card_seed : Fintype.card Seed = 2 ^ lam
  /-- The prover's commitment `c`, sent before σ. -/
  Com : Type
  /-- The prover's answer `a` to σ. -/
  Ans : Type
  /-- Solve's coins `ρ`. -/
  Coins : Type
  [instCoins : Fintype Coins]
  /-- `C` answers `x` against σ (for matmul: `A · B = C` on the exact grid; for an edited instance, route (c), the
  edited product `(A + E_A(σ)) (B + E_B(σ)) = C`). -/
  correct : Seed → Inst → Out → Prop
  /-- The promise (RS-DEF v2 §1.4, `A, B ∈ G_use`): usefulness and completeness quantify over the promised instances
  only, so a verifier may range-check what is committed. -/
  promise : Inst → Prop := fun _ => True
  /-- `Solve(x; ρ)` against σ: `((c, a), C)`. -/
  solve : Mc.Alg (Seed × Inst × Coins) ((Com × Ans) × Out)
  /-- `V(c, σ, a) → b`, deterministic and not priced. -/
  verify : Com → Seed → Ans → Bool

attribute [instance] Protocol.instSeed Protocol.instCoins

/-- A transcript's prover messages `(c, a)`. -/
abbrev Protocol.Proof {Mc : Machine} {Inst Out : Type} (P : Protocol Mc Inst Out) : Type := P.Com × P.Ans

/-- The verifier accepts the prover messages `π = (c, a)` against σ. -/
def Protocol.accepts {Mc : Machine} {Inst Out : Type} (P : Protocol Mc Inst Out) (σ : P.Seed) (π : P.Proof) : Bool :=
  P.verify π.1 σ π.2

variable {Mc : Machine} {Inst Out : Type}

/-- **Live coins** (RS-DEF's T6): the commitment a prover sends doesn't depend on σ. -/
def CommitsFirst (P : Protocol Mc Inst Out) {X : Type} (f : P.Seed × X → P.Proof) : Prop :=
  ∀ σ σ' x, (f (σ, x)).1 = (f (σ', x)).1

/-- A useful prover `M'(x; r)` against σ answers `x` correctly and is accepted, with probability at least `1 - η` over
σ and its coins, for every promised instance (RS-DEF §1.4's event `b = 1 ∧ C = AB`). -/
def SolvesAll (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  ∀ x : Inst, P.promise x → 1 - η ≤ prob fun ω : P.Seed × Ω =>
    P.correct ω.1 x (Mc.eval M' (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval M' (ω.1, x, ω.2)).1 = true

/-- A useful prover: it commits first and solves every instance. -/
def Useful (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  CommitsFirst P (fun y => (Mc.eval M' y).1) ∧ SolvesAll P M' η

/-- `p(M)`: the probability that an adversary `M(σ; r)`, with no input, is accepted. -/
noncomputable def acceptance (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof) :
    ℝ :=
  prob fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true

/-- **Completeness** (RS-DEF §1.4): Solve commits first and answers every instance with a proof that verifies, with
probability at least `1 - η`. -/
def Complete (P : Protocol Mc Inst Out) (η : ℝ) : Prop :=
  Useful P P.solve η

/-- **Security, Neekon's form**: every adversary `M` that commits first and is accepted with probability at least `δ`
gives a useful `M'` with `time M' ≤ (1 + ε) time M + α`. -/
def Secure (P : Protocol Mc Inst Out) (ε δ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    δ ≤ acceptance P M →
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + ε) * Mc.time M + α ∧ Useful P M' η

/-- `T_use(η)` (RS-DEF §1.5): the least time of a useful prover. -/
noncomputable def Tuse (P : Protocol Mc Inst Out) (η : ℝ) : ℝ :=
  sInf {t | ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
    Useful P M' η ∧ Mc.time M' = t}

theorem tuse_le (P : Protocol Mc Inst Out) {η : ℝ} {Ω' : Type} [Fintype Ω']
    {M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)} (h : Useful P M' η) : Tuse P η ≤ Mc.time M' :=
  csInf_le ⟨0, fun _ ⟨_, _, M'', _, ht⟩ => ht ▸ Mc.time_nonneg M''⟩ ⟨Ω', inferInstance, M', h, rfl⟩

/-- **Security, RS-DEF's frozen form** (§1.5): every adversary that commits first and is accepted with probability at
least `δ` has `(1 + ε) time M + α ≥ T_use(η)`. -/
def SecureInf (P : Protocol Mc Inst Out) (ε δ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    δ ≤ acceptance P M → Tuse P η ≤ (1 + ε) * Mc.time M + α

theorem secureInf_of_secure (P : Protocol Mc Inst Out) {ε δ η α : ℝ} (h : Secure P ε δ η α) :
    SecureInf P ε δ η α := fun Ω _ M hc hM => by
  obtain ⟨_, _, M', ht, hu⟩ := h Ω M hc hM
  exact (tuse_le P hu).trans ht

/-- **The per-run core** (RS-DEF v2 §1.6, the proof target): for every adversary that commits first, the runs that
are accepted and whose charge `max(|M|, cost)` is below `(T_use(η) − α)/(1 + ε)` have probability at most `κ`. -/
def Core (P : Protocol Mc Inst Out) (ε κ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧
      Mc.charge M ω < (Tuse P η - α) / (1 + ε)) ≤ κ

/-- **Core ⇒ Security** for every `δ > κ`: some accepted run costs at least the bound, and `time` bounds every run. -/
theorem secureInf_of_core (P : Protocol Mc Inst Out) {ε κ δ η α : ℝ} (h : Core P ε κ η α) (hκδ : κ < δ)
    (hε : 0 < 1 + ε) : SecureInf P ε δ η α := by
  intro Ω _ M hc hM
  by_contra hlt
  replace hlt := not_le.mp hlt
  have hbound : Mc.time M < (Tuse P η - α) / (1 + ε) := by
    rw [lt_div_iff₀ hε]
    linarith
  have hsub := prob_mono (Ω := P.Seed × Ω) (p := fun ω => P.accepts ω.1 (Mc.eval M ω) = true)
    (q := fun ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.charge M ω < (Tuse P η - α) / (1 + ε))
    fun ω hω => ⟨hω, (Mc.charge_le_time M ω).trans_lt hbound⟩
  have := h Ω M hc
  unfold acceptance at hM
  linarith

/-- **The rate form** (RS-DEF v2 §1.6): under Core, an adversary that commits first is charged in expectation at least
`(p(M) − κ) (T_use(η) − α)/(1 + ε)` per run (Markov on the charge). -/
theorem rate_of_core (P : Protocol Mc Inst Out) {ε κ η α : ℝ} (h : Core P ε κ η α)
    (hX : 0 ≤ (Tuse P η - α) / (1 + ε)) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hc : CommitsFirst P (Mc.eval M)) :
    (acceptance P M - κ) * ((Tuse P η - α) / (1 + ε)) ≤ expect (Mc.charge M) := by
  have hcore := h Ω M hc
  set X := (Tuse P η - α) / (1 + ε)
  have hsplit : acceptance P M ≤ prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧
      Mc.charge M ω < X) + prob fun ω => X ≤ Mc.charge M ω := by
    unfold acceptance
    exact (prob_mono fun ω hω => (lt_or_ge (Mc.charge M ω) X).imp (fun hl => ⟨hω, hl⟩) id).trans
      (prob_or_le _ _)
  calc (acceptance P M - κ) * X ≤ prob (fun ω => X ≤ Mc.charge M ω) * X :=
        mul_le_mul_of_nonneg_right (by linarith) hX
    _ = X * prob (fun ω => X ≤ Mc.charge M ω) := mul_comm _ _
    _ ≤ expect (Mc.charge M) := markov (fun ω => Mc.charge_nonneg M ω) X

/-- **The adversary floor** (a named hypothesis, the lane's floor results with Theorem F's luck term in `κ`): for every
adversary that commits first, the runs that are accepted and whose charge `max(|M|, cost)` is below `F` have
probability at most `κ` (RS-DEF v2's Lemma R premise). -/
def AdvFloor (P : Protocol Mc Inst Out) (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.charge M ω < F) ≤ κ

/-- **Lemma R** (RS-DEF §1.6: a floor plus an honest upper bound give the core): under the adversary floor `F`, if some
useful prover costs at most `U`, then Core holds with `1 + ε = U / F` and `α = 0`. -/
theorem core_of_floor (P : Protocol Mc Inst Out) {κ η F U : ℝ} (hF : 0 < F) (hFU : F ≤ U)
    (hfloor : AdvFloor P F κ) {Ω₀ : Type} [Fintype Ω₀] {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)}
    (hM₀ : Useful P M₀ η) (hU : Mc.time M₀ ≤ U) : Core P (U / F - 1) κ η 0 := by
  intro Ω _ M hc
  have hU0 : 0 < U := hF.trans_le hFU
  have hb : (Tuse P η - 0) / (1 + (U / F - 1)) ≤ F := by
    rw [sub_zero, add_sub_cancel, div_div_eq_mul_div, div_le_iff₀ hU0]
    rw [mul_comm F]
    exact mul_le_mul_of_nonneg_right ((tuse_le P hM₀).trans hU) hF.le
  exact (prob_mono fun ω hω => ⟨hω.1, hω.2.trans_le hb⟩).trans (hfloor Ω M hc)

/-- **Efficiency** (RS-DEF §1.7): Solve costs at most `(1 + γ) N + α`, for `N` the reference matmul's time. -/
def Efficient (P : Protocol Mc Inst Out) (γ N α : ℝ) : Prop :=
  Mc.time P.solve ≤ (1 + γ) * N + α

/-- **Relative security** with parameters `ε, δ, γ, η, α` against the reference time `N` (RS-DEF's frozen form). -/
def RelativelySecure (P : Protocol Mc Inst Out) (ε δ γ η α N : ℝ) : Prop :=
  Complete P η ∧ SecureInf P ε δ η α ∧ Efficient P γ N α

/-- **Relative security from the adversary floor** (RS-DEF's route (a), end to end in the frozen definition): if Solve
is complete, costs at least the floor `F > 0` and is efficient, and every adversary's accepted runs below `F` have
probability at most `κ < δ`, the protocol is relatively secure with `ε = time(Solve) / F − 1` and `α = 0`. -/
theorem relativelySecure_of_advFloor (P : Protocol Mc Inst Out) {κ δ γ η F N : ℝ} (hc : Complete P η) (hF : 0 < F)
    (hFS : F ≤ Mc.time P.solve) (hfloor : AdvFloor P F κ) (hκδ : κ < δ) (heff : Efficient P γ N 0) :
    RelativelySecure P (Mc.time P.solve / F - 1) δ γ η 0 N := by
  refine ⟨hc, secureInf_of_core P (core_of_floor P hF hFS hfloor hc le_rfl) hκδ ?_, heff⟩
  rw [add_sub_cancel]
  exact div_pos (hF.trans_le hFS) hF

/-- The matmul problem on instances that embed into pairs of `n × n` matrices over `R` (for example RS-DEF's useful
grid `G_use` as a subtype): `x` is answered by `C = A · B` for `(A, B) = emb x`. -/
def IsMatmulOn {R : Type} [Ring R] {n : ℕ} {Mc : Machine} {Inst : Type} (P : Protocol Mc Inst (Mat R n n))
    (emb : Inst → Mat R n n × Mat R n n) : Prop :=
  ∀ σ x C, P.correct σ x C ↔ (emb x).1 * (emb x).2 = C

/-- The matmul problem on all pairs of `n × n` matrices over `R`. -/
def IsMatmul {R : Type} [Ring R] {n : ℕ} {Mc : Machine} (P : Protocol Mc (Mat R n n × Mat R n n) (Mat R n n)) :
    Prop :=
  IsMatmulOn P id

/-- Under completeness, `T_use(η)` is at most Solve's time. -/
theorem tuse_le_solve (P : Protocol Mc Inst Out) {η : ℝ} (h : Complete P η) : Tuse P η ≤ Mc.time P.solve :=
  tuse_le P h

theorem secure_mono (P : Protocol Mc Inst Out) {ε ε' δ δ' η η' α α' : ℝ} (h : Secure P ε δ η α) (hε : ε ≤ ε')
    (hδ : δ ≤ δ') (hη : η ≤ η') (hα : α ≤ α') : Secure P ε' δ' η' α' := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hc', hs⟩ := h Ω M hc (hδ.trans hM)
  refine ⟨Ω', inferInstance, M', ht.trans ?_, hc', fun x hx => le_trans (by linarith) (hs x hx)⟩
  have := mul_le_mul_of_nonneg_right (show 1 + ε ≤ 1 + ε' by linarith) (Mc.time_nonneg M)
  linarith

theorem complete_mono (P : Protocol Mc Inst Out) {η η' : ℝ} (h : Complete P η) (hη : η ≤ η') : Complete P η' :=
  ⟨h.1, fun x hx => le_trans (by linarith) (h.2 x hx)⟩

end Definition

section Masking

variable {α : Type*} [Ring α] {m k p r r' : Type*} [Fintype k] [Fintype r] [Fintype r']

/-- **The full-rank peel**: `A · B = (A + R)(B + S) − R (B + S) − A · S`, over any ring (in particular any commutative
ring, `ZMod (2^32)` and any field). -/
theorem mask_peel (A R : Matrix m k α) (B S : Matrix k p α) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S := by
  rw [Matrix.add_mul, Matrix.mul_add]
  abel

/-- **The low-rank peel**: with masks `U Vᵀ` (rank `r`) and `W Zᵀ` (rank `r'`), the masked product minus the two
corrections, each a product through an inner dimension `r` or `r'`, is `A · B`:
`(A + U Vᵀ)(B + W Zᵀ) − U (Vᵀ (B + W Zᵀ)) − (A W) Zᵀ = A · B`. -/
theorem lowRank_peel (A : Matrix m k α) (B : Matrix k p α) (U : Matrix m r α) (Vt : Matrix r k α)
    (W : Matrix k r' α) (Zt : Matrix r' p α) :
    (A + U * Vt) * (B + W * Zt) - U * (Vt * (B + W * Zt)) - (A * W) * Zt = A * B := by
  rw [Matrix.add_mul, Matrix.mul_add A, Matrix.mul_assoc U, Matrix.mul_assoc A]
  abel

/-- The full-rank peel in s32 arithmetic (`ZMod (2^32)`), where it is exact. -/
theorem mask_peel_zmod32 (A R : Matrix m k (ZMod (2 ^ 32))) (B S : Matrix k p (ZMod (2 ^ 32))) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

/-- The full-rank peel over a field, where it is also exact. -/
theorem mask_peel_field {F : Type*} [Field F] (A R : Matrix m k F) (B S : Matrix k p F) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

end Masking

section Exactness

/-- An integer is its s32 residue's signed representative when `-2^31 < z ≤ 2^31`. -/
theorem valMinAbs_intCast_of_bounds {z : ℤ} (hlo : -2 ^ 31 < z) (hhi : z ≤ 2 ^ 31) :
    ((z : ZMod (2 ^ 32))).valMinAbs = z := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨rfl, ?_, ?_⟩ <;> push_cast <;> omega

variable {m k p : Type*} [Fintype k]

/-- The full-rank peel of integer matrices `A`, `B` computed in s32 (`ZMod (2^32)`) with masks `R`, `S`:
`(A + R)(B + S) − R (B + S) − A · S`. -/
def peel32 (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32))) : Matrix m p (ZMod (2 ^ 32)) :=
  let A' := A.map (Int.castRingHom (ZMod (2 ^ 32)))
  let B' := B.map (Int.castRingHom (ZMod (2 ^ 32)))
  (A' + R) * (B' + S) - R * (B' + S) - A' * S

/-- **s32 recovers the integer product**: for integer matrices whose product has entries in `(-2^31, 2^31]`, the
peel computed in s32, with any masks, has the integer product as its signed representative. -/
theorem s32_peel_recovers (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32)))
    (hb : ∀ i j, -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j := by
  have h : peel32 A B R S = (A * B).map (Int.castRingHom (ZMod (2 ^ 32))) := by
    rw [Matrix.map_mul]
    exact (mask_peel _ R _ S).symm
  rw [h, Matrix.map_apply]
  exact valMinAbs_intCast_of_bounds (hb i j).1 (hb i j).2

/-- int8 by int8 products of inner dimension at most `2^17` lie in `(-2^31, 2^31]`, so s32 accumulation is exact. -/
theorem int8_product_bounds (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127) (i : m) (j : p) :
    -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31 := by
  have hterm : ∀ l, -16256 ≤ A i l * B l j ∧ A i l * B l j ≤ 16384 := by
    intro l
    obtain ⟨a1, a2⟩ := hA i l
    obtain ⟨b1, b2⟩ := hB l j
    constructor <;> nlinarith
  have hk' : (Fintype.card k : ℤ) ≤ 2 ^ 17 := by exact_mod_cast hk
  rw [Matrix.mul_apply]
  constructor
  · have : ∑ l, (-16256 : ℤ) ≤ ∑ l, A i l * B l j := Finset.sum_le_sum fun l _ => (hterm l).1
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith
  · have : ∑ l, A i l * B l j ≤ ∑ l, (16384 : ℤ) := Finset.sum_le_sum fun l _ => (hterm l).2
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith

/-- **int8 with s32 accumulation is exact** for inner dimension at most `2^17`: the s32 peel of int8 matrices, with
any masks, recovers the integer product. -/
theorem int8_product_exact (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127)
    (R : Matrix m k (ZMod (2 ^ 32))) (S : Matrix k p (ZMod (2 ^ 32))) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j :=
  s32_peel_recovers A B R S (int8_product_bounds A B hk hA hB) i j

/-- A product of integer matrices with entries at most `M` in absolute value has entries at most `|k| M²`. -/
theorem grid_product_bound (A : Matrix m k ℤ) (B : Matrix k p ℤ) {M : ℤ} (hA : ∀ i l, |A i l| ≤ M)
    (hB : ∀ l j, |B l j| ≤ M) (i : m) (j : p) : |(A * B) i j| ≤ Fintype.card k * M ^ 2 := by
  rw [Matrix.mul_apply]
  calc |∑ l, A i l * B l j| ≤ ∑ l, |A i l * B l j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _l : k, M ^ 2 := Finset.sum_le_sum fun l _ => by
        rw [abs_mul, sq]
        exact mul_le_mul (hA i l) (hB l j) (abs_nonneg _) ((abs_nonneg _).trans (hA i l))
    _ = Fintype.card k * M ^ 2 := by simp [Finset.sum_const, Finset.card_univ]

/-- **RS-DEF's exactness condition (E)** on the integer grid: if the useful operands, the noise and the masked operands
all have entries at most `M_op` and `n M_op² ≤ B_acc`, every product the peel computes, `(A + R)(B + S)`, `R (B + S)`
and `A S`, has entries at most `B_acc`, so an accumulator exact up to `B_acc` (`2^24` for E4M3 integers with FP32
accumulation, `2^31 − 1` for int8 with s32) computes each exactly, and the peel over `ℤ` is `A B` (`mask_peel`). -/
theorem grid_peel_in_range (A R : Matrix m k ℤ) (B S : Matrix k p ℤ) {Mop Bacc : ℤ}
    (hA : ∀ i l, |A i l| ≤ Mop) (hR : ∀ i l, |R i l| ≤ Mop) (hS : ∀ l j, |S l j| ≤ Mop)
    (hAR : ∀ i l, |(A + R) i l| ≤ Mop) (hBS : ∀ l j, |(B + S) l j| ≤ Mop)
    (hE : Fintype.card k * Mop ^ 2 ≤ Bacc) (i : m) (j : p) :
    |((A + R) * (B + S)) i j| ≤ Bacc ∧ |(R * (B + S)) i j| ≤ Bacc ∧ |(A * S) i j| ≤ Bacc :=
  ⟨(grid_product_bound _ _ hAR hBS i j).trans hE, (grid_product_bound _ _ hR hBS i j).trans hE,
    (grid_product_bound _ _ hA hS i j).trans hE⟩

/-- RS-DEF §1.1's instances meet (E): int8 operands in `[-126, 126]` (useful and noise in `[-63, 63]`) with s32
accumulation for `n ≤ 135,266`, the largest such `n`; E4M3 integers in `[-16, 16]` with FP32 accumulation for
`n ≤ 65,536`. -/
theorem grid_instances :
    (135266 : ℤ) * 126 ^ 2 ≤ 2 ^ 31 - 1 ∧ ¬ (135267 : ℤ) * 126 ^ 2 ≤ 2 ^ 31 - 1 ∧ (65536 : ℤ) * 16 ^ 2 ≤ 2 ^ 24 := by
  norm_num

/-- Saturating int8 arithmetic (clamped to `[-128, 127]`), the kind of non-exact accumulator the peel needs to avoid. -/
def sat8 (z : ℤ) : ℤ := max (-128) (min 127 z)

/-- **A non-exact arithmetic breaks the peel**: masking `a = 100` by `r = 100` in saturating int8 and peeling the
mask off gives `27`, not `100`. -/
theorem sat8_breaks_peel : sat8 (sat8 (100 + 100) - 100) ≠ 100 := by
  decide

end Exactness

section ZeroInstance

/-! RS-DEF's Proposition Z, the algebra: what the zero instance's transcript is. `D` is any matrix (the prefix
selector `diag(1_{k ≤ jL})` for the `j`-th prefix sum `P_j = A' D B'`). -/

variable {α : Type*} [Ring α] {m k p r r' : Type*} [Fintype k] [Fintype r] [Fintype r']

/-- **Z(a), σ before the instance**: an instance chosen after the noise as `A_M = −E`, `B_M = −F` makes every
masked prefix sum zero, for free. -/
theorem sigma_first_cancels (E : Matrix m k α) (F : Matrix k p α) (D : Matrix k k α) :
    (-E + E) * D * (-F + F) = 0 := by
  simp

/-- **Z(c)**: with `A_M = B_M = 0` the masked prefix sums are `P_j = E (D F)`, one product with the noise `E`. -/
theorem zero_instance_prefix (E : Matrix m k α) (F : Matrix k p α) (D : Matrix k k α) :
    (0 + E) * D * (0 + F) = E * (D * F) := by
  simp [Matrix.mul_assoc]

/-- **Z(d)**: with rank-`r` noise `E = E_L E_R` and `F = F_L F_R` and `A_M = B_M = 0`, the masked prefix sums are
`P_j = E_L Q_j F_R` with `Q_j = E_R D F_L` an `r × r'` matrix: products through the inner dimensions `r`, `r'`. -/
theorem zero_instance_lowRank_prefix (EL : Matrix m r α) (ER : Matrix r k α) (FL : Matrix k r' α)
    (FR : Matrix r' p α) (D : Matrix k k α) :
    (0 + EL * ER) * D * (0 + FL * FR) = EL * (ER * D * FL) * FR := by
  simp only [zero_add, Matrix.mul_assoc]

end ZeroInstance

section Reduction

namespace Machine

variable (Mc : Machine)

/-- **The reduction** `M' = M ++ Q`: run the extractor `E` (M's own program on the masked instance; its proof and the
blocks read off its wires), keep the proof, and run the peel and then the correction on the blocks. -/
def reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.Alg I (Pf × O) :=
  Mc.seq E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))

theorem eval_reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) (x : I) :
    Mc.eval (Mc.reduction E peel corr) x = ((Mc.eval E x).1, Mc.eval corr (Mc.eval peel (Mc.eval E x).2)) := by
  simp only [reduction, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]

/-- The reduction costs at most the extractor, the peel and the correction. -/
theorem time_reduction_le {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) : Mc.time (Mc.reduction E peel corr) ≤ Mc.time E + Mc.time peel + Mc.time corr := by
  have h1 := Mc.time_seq_le E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))
  have h2 := Mc.time_pair_le (Mc.fst (I := Pf) (J := Bl)) (Mc.seq Mc.snd (Mc.seq peel corr))
  have h3 := Mc.time_seq_le (Mc.snd (I := Pf) (J := Bl)) (Mc.seq peel corr)
  have h4 := Mc.time_seq_le peel corr
  have h5 := Mc.time_fst (I := Pf) (J := Bl)
  have h6 := Mc.time_snd (I := Pf) (J := Bl)
  unfold reduction
  linarith

/-- **The cost inequality**: with an extractor that costs at most `M`,
`time M' ≤ time M + time peel + time correction`. -/
theorem time_reduction_le_of_le {I Pf Bl Mid O J : Type} {E : Mc.Alg I (Pf × Bl)} {M : Mc.Alg J Pf}
    (hE : Mc.time E ≤ Mc.time M) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.time (Mc.reduction E peel corr) ≤ Mc.time M + Mc.time peel + Mc.time corr := by
  linarith [Mc.time_reduction_le E peel corr]

end Machine

variable {Mc : Machine} {Inst Out : Type}

/-- **Extraction** (a named hypothesis, RS-EXT's Lemma E and Theorem R): from an adversary `M`, an extractor `E` that
commits first, costs at most `(1 + εext) time M` (one straight-line run of M on the planted instance, reading M's
committed blocks off its hash log, plus the check, the fixes and its own proof), and, for every promised instance
`x`, with probability at least `1 - η` over σ and its coins, outputs prover messages that are accepted and blocks
valid for `x`. -/
def Extraction (P : Protocol Mc Inst Out) {Bl : Type} (valid : Inst → Bl → Prop) (εext : ℝ) {Ω : Type} [Fintype Ω]
    (M : Mc.Alg (P.Seed × Ω) P.Proof) (η : ℝ) : Prop :=
  ∃ (Ω' : Type) (_ : Fintype Ω') (E : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Bl)),
    CommitsFirst P (fun y => (Mc.eval E y).1) ∧ Mc.time E ≤ (1 + εext) * Mc.time M ∧
    ∀ x, P.promise x → 1 - η ≤ prob fun ω : P.Seed × Ω' =>
      valid x (Mc.eval E (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval E (ω.1, x, ω.2)).1 = true

/-- **The floor** (a named hypothesis, the floor agenda's: Conjecture A, Theorems B/C/W/F): every algorithm that
answers every promised instance correctly with probability at least `1 - η` over its coins costs at least `T`
(`T*_mm(n)` for matmul, RS-DEF §1.7; no proof is asked for, so `T_use(η) ≥ T`). -/
def Floor (Mc : Machine) {Inst Out : Type} (promise : Inst → Prop) (correct : Inst → Out → Prop) (T η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (N : Mc.Alg (Inst × Ω) Out),
    (∀ x, promise x → 1 - η ≤ prob fun ω : Ω => correct x (Mc.eval N (x, ω))) → T ≤ Mc.time N

/-- An algorithm that solves every instance (with a proof) is above the floor of any problem its answers solve
(`hcorr`; for an unedited protocol, its own): drop σ into its coins and forget the proof, which routing does for
free. -/
theorem Floor.le_time (P : Protocol Mc Inst Out) {correct : Inst → Out → Prop} {T η : ℝ}
    (h : Floor Mc P.promise correct T η)
    (hcorr : ∀ σ x C, P.correct σ x C → correct x C) {Ω : Type} [Fintype Ω]
    {M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out)} (hM' : SolvesAll P M' η) : T ≤ Mc.time M' := by
  let route : Mc.Alg (Inst × P.Seed × Ω) (P.Seed × Inst × Ω) :=
    Mc.pair (Mc.seq Mc.snd Mc.fst) (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.snd))
  have hroute : ∀ y, Mc.eval route y = (y.2.1, y.1, y.2.2) := fun y => by
    simp only [route, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]
  have hT := h (P.Seed × Ω) (Mc.seq (Mc.seq route M') Mc.snd) fun x hx => (hM' x hx).trans (prob_mono fun ω hω => by
    rw [Mc.eval_seq, Mc.eval_seq, hroute, Mc.eval_snd]
    exact hcorr _ _ _ hω.1)
  have hr : Mc.time route ≤ 0 := by
    apply le_trans
    · apply_rules [Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
    · norm_num
  have r1 := Mc.time_seq_le (Mc.seq route M') (Mc.snd (I := P.Proof) (J := Out))
  have r2 := Mc.time_seq_le route M'
  have z := Mc.time_snd (I := P.Proof) (J := Out)
  linarith

theorem Floor.le_tuse (P : Protocol Mc Inst Out) {correct : Inst → Out → Prop} {T η : ℝ}
    (h : Floor Mc P.promise correct T η)
    (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    (hne : ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)), Useful P M' η) :
    T ≤ Tuse P η := by
  obtain ⟨Ω', _, M', hu⟩ := hne
  exact le_csInf ⟨_, Ω', inferInstance, M', hu, rfl⟩ fun _ ⟨_, _, _, hu', ht⟩ => ht ▸ h.le_time P hcorr hu'.2

/-- With extraction, the reduction is useful and costs at most `(1 + εext) time M + time peel + time corr`, when the
peel and the correction answer `x` from blocks valid for `x`. -/
theorem exists_reduction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext η : ℝ} {Ω : Type} [Fintype Ω]
    {M : Mc.Alg (P.Seed × Ω) P.Proof} (hext : Extraction P valid εext M η) :
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + εext) * Mc.time M + Mc.time peel + Mc.time corr ∧ Useful P M' η := by
  obtain ⟨Ω', _, E, hc, hE, hsucc⟩ := hext
  refine ⟨Ω', inferInstance, Mc.reduction E peel corr, (Mc.time_reduction_le E peel corr).trans (by linarith),
    fun σ σ' y => ?_, fun x hx => (hsucc x hx).trans (prob_mono fun ω hω => ?_)⟩
  · simp only [Mc.eval_reduction]
    exact hc σ σ' y
  · rw [Mc.eval_reduction]
    exact ⟨hpeel _ x _ hω.1, hω.2⟩

/-- **Security from extraction**, given that the peel and the correction cost at most `ε time M` for every adversary
`M` accepted with probability at least `δ` (extraction overhead `εext`). -/
theorem secure_of_extraction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext ε δ η : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Mc.time peel + Mc.time corr ≤ ε * Mc.time M) :
    Secure P (εext + ε) δ η 0 := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hu⟩ := exists_reduction P peel corr hpeel (hext Ω M hc hM)
  exact ⟨Ω', inferInstance, M', by linarith [hcost Ω M hc hM], hu⟩

/-- **Security from extraction and the floor**: if the peel and the correction cost at most `q < T`, where `T` is the
floor, then `ε = εext + (1 + εext) q / (T − q)`; with `εext = 0` this is RS-MODEL's `q / (T − q)`, eq. (2).
(`M'` is useful, so `T ≤ time M' ≤ (1 + εext) time M + q`.) -/
theorem secure_of_extraction_floor (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext δ η T q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η) {correct : Inst → Out → Prop}
    (hfloor : Floor Mc P.promise correct T η) (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    (hq : Mc.time peel + Mc.time corr ≤ q) (hqT : q < T) :
    Secure P (εext + (1 + εext) * (q / (T - q))) δ η 0 := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hu⟩ := exists_reduction P peel corr hpeel (hext Ω M hc hM)
  refine ⟨Ω', inferInstance, M', ?_, hu⟩
  have hTM := hfloor.le_time P hcorr hu.2
  have hTq : 0 < T - q := sub_pos.2 hqT
  have hq0 : 0 ≤ q := (add_nonneg (Mc.time_nonneg _) (Mc.time_nonneg _)).trans hq
  set θ := q / (T - q) with hθ
  have hθ0 : 0 ≤ θ := div_nonneg hq0 hTq.le
  have hθq : θ * (T - q) = q := div_mul_cancel₀ q hTq.ne'
  have hM : θ * (T - q) ≤ θ * ((1 + εext) * Mc.time M) := mul_le_mul_of_nonneg_left (by linarith) hθ0
  nlinarith

/-- **Security from extraction and M's own floor** (RS-MODEL's route 2, `ε ≤ time(Q) / (c F_M)`, which needs no
`T_mm`): if every adversary accepted with probability at least `δ` costs at least `F > 0` (a named hypothesis, the
floor on M's committed targets) and the peel and the correction cost at most `q`, then `ε = εext + q / F`. -/
theorem secure_of_extraction_advTime (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ σ x b, valid x b → P.correct σ x (Mc.eval corr (Mc.eval peel b))) {εext δ η F q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M)
    (hF : 0 < F) (hq : Mc.time peel + Mc.time corr ≤ q) : Secure P (εext + q / F) δ η 0 := by
  refine secure_of_extraction P peel corr hpeel hext fun Ω _ M hc hδ => hq.trans ?_
  have hq0 : 0 ≤ q := (add_nonneg (Mc.time_nonneg _) (Mc.time_nonneg _)).trans hq
  calc q = q / F * F := (div_mul_cancel₀ q hF.ne').symm
    _ ≤ q / F * Mc.time M := mul_le_mul_of_nonneg_left (hM Ω M hc hδ) (div_nonneg hq0 hF.le)

/-- **Security is a relative floor** (RS-DEF §1.5), one direction: if some useful `M₀` costs at most
`(1 + ε) time M + α` for every adversary `M` accepted with probability at least `δ`, the protocol is secure. No
reduction from `M` is needed. -/
theorem secure_of_solver (P : Protocol Mc Inst Out) {ε δ η α : ℝ} {Ω₀ : Type} [Fintype Ω₀]
    {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)} (hM₀ : Useful P M₀ η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Mc.time M₀ ≤ (1 + ε) * Mc.time M + α) :
    Secure P ε δ η α :=
  fun Ω _ M hc hM => ⟨Ω₀, inferInstance, M₀, hcost Ω M hc hM, hM₀⟩

/-- **Security is a relative floor**, the other direction: under security, every adversary accepted with probability
at least `δ` has `T ≤ (1 + ε) time M + α`, for `T` the floor. -/
theorem floor_le_of_secure (P : Protocol Mc Inst Out) {ε δ η α T : ℝ} (h : Secure P ε δ η α)
    {correct : Inst → Out → Prop} (hfloor : Floor Mc P.promise correct T η) (hcorr : ∀ σ x C, P.correct σ x C → correct x C)
    {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hc : CommitsFirst P (Mc.eval M)) (hM : δ ≤ acceptance P M) : T ≤ (1 + ε) * Mc.time M + α := by
  obtain ⟨Ω', _, M', ht, hu⟩ := h Ω M hc hM
  exact (hfloor.le_time P hcorr hu.2).trans ht

end Reduction

section MatmulPeel

namespace MatMachine

variable {R : Type} [Ring R] (Mc : MatMachine R)

/-- Multiply the outputs of `f` and `g`. -/
def mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mul a b c)

/-- Subtract the output of `g` from that of `f`. -/
def sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) : Mc.Alg I (Mat R a b) :=
  Mc.lift2 f g (Mc.sub a b)

theorem eval_mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) (x : I) :
    Mc.eval (Mc.mul₂ f g) x = Mc.eval f x * Mc.eval g x := by
  rw [mul₂, Machine.eval_lift2, Mc.eval_mul]

theorem eval_sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) (x : I) :
    Mc.eval (Mc.sub₂ f g) x = Mc.eval f x - Mc.eval g x := by
  rw [sub₂, Machine.eval_lift2, Mc.eval_sub]

theorem time_mul₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mul₂ f g) ≤ u + v + Mc.κ * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mul_le a b c])

theorem time_sub₂_le_of {I : Type} {a b : ℕ} {f g : Mc.Alg I (Mat R a b)} {u v : ℝ} (hf : Mc.time f ≤ u)
    (hg : Mc.time g ≤ v) : Mc.time (Mc.sub₂ f g) ≤ u + v + Mc.κadd * ((a : ℝ) * b) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_sub_le a b])

/-- Add the outputs of `f` and `g`. -/
def add₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) : Mc.Alg I (Mat R a b) :=
  Mc.lift2 f g (Mc.add a b)

theorem eval_add₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) (x : I) :
    Mc.eval (Mc.add₂ f g) x = Mc.eval f x + Mc.eval g x := by
  rw [add₂, Machine.eval_lift2, Mc.eval_add]

theorem time_add₂_le_of {I : Type} {a b : ℕ} {f g : Mc.Alg I (Mat R a b)} {u v : ℝ} (hf : Mc.time f ≤ u)
    (hg : Mc.time g ≤ v) : Mc.time (Mc.add₂ f g) ≤ u + v + Mc.κadd * ((a : ℝ) * b) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_add_le a b])

/-- The correction `(C', X, Y) ↦ C' − X − Y`. -/
def corr (n : ℕ) : Mc.Alg (Mat R n n × Mat R n n × Mat R n n) (Mat R n n) :=
  Mc.sub₂ (Mc.sub₂ Mc.fst (Mc.seq Mc.snd Mc.fst)) (Mc.seq Mc.snd Mc.snd)

theorem eval_corr (n : ℕ) (b : Mat R n n × Mat R n n × Mat R n n) :
    Mc.eval (Mc.corr n) b = b.1 - b.2.1 - b.2.2 := by
  simp only [corr, eval_sub₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The correction costs two subtractions. -/
theorem time_corr_le (n : ℕ) : Mc.time (Mc.corr n) ≤ 2 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold corr
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The full-rank peel's blocks `(C', A, Y, R, S)`: the committed product, the instance's `A`, M's masked
`Y = B + S`, and the masks. -/
abbrev FullBlocks (R : Type) (n : ℕ) : Type := Mat R n n × Mat R n n × Mat R n n × Mat R n n × Mat R n n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + S`, and `C' = (A + R) Y`. -/
def ValidFull {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1) * b.2.2.1

/-- **The full-rank peel**: `(C', A, Y, R, S) ↦ (C', R Y, A S)`, two `n × n × n` products. -/
def peelFull (n : ℕ) : Mc.Alg (FullBlocks R n) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
      (Mc.mul₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_peelFull (n : ℕ) (b : FullBlocks R n) :
    Mc.eval (Mc.peelFull n) b = (b.1, b.2.2.2.1 * b.2.2.1, b.2.1 * b.2.2.2.2) := by
  simp only [peelFull, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The full-rank peel and correction answer the instance from valid blocks: `C' − R Y − A S = A B`. -/
theorem peelFull_correct {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) (hb : ValidFull x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelFull n) b) := by
  obtain ⟨C', A, Y, Rm, S⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelFull, eval_corr]
  exact mask_peel x.1 Rm x.2 S

/-- The full-rank peel costs two `n × n × n` products. -/
theorem time_peelFull_le (n : ℕ) : Mc.time (Mc.peelFull n) ≤ 2 * Mc.κ * (n : ℝ) ^ 3 := by
  apply le_trans
  · unfold peelFull
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The low-rank peel's blocks `(C', A, Y, U, Vᵀ, W, Zᵀ)`: the committed product, the instance's `A`, M's masked
`Y = B + W Zᵀ`, and the factors of the rank-`r` masks `U Vᵀ` and `W Zᵀ`. -/
abbrev LowBlocks (R : Type) (n r : ℕ) : Type :=
  Mat R n n × Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + W Zᵀ`, and `C' = (A + U Vᵀ) Y`. -/
def ValidLow {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2.2.1 * b.2.2.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1 * b.2.2.2.2.1) * b.2.2.1

/-- **The low-rank peel**: `(C', A, Y, U, Vᵀ, W, Zᵀ) ↦ (C', U (Vᵀ Y), (A W) Zᵀ)`, four products through the inner
dimension `r`. -/
def peelLow (n r : ℕ) : Mc.Alg (LowBlocks R n r) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
          (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
      (Mc.mul₂
        (Mc.mul₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))))
        (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))

theorem eval_peelLow (n r : ℕ) (b : LowBlocks R n r) :
    Mc.eval (Mc.peelLow n r) b =
      (b.1, b.2.2.2.1 * (b.2.2.2.2.1 * b.2.2.1), (b.2.1 * b.2.2.2.2.2.1) * b.2.2.2.2.2.2) := by
  simp only [peelLow, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The low-rank peel and correction answer the instance from valid blocks:
`C' − U (Vᵀ Y) − (A W) Zᵀ = A B`. -/
theorem peelLow_correct {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) (hb : ValidLow x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelLow n r) b) := by
  obtain ⟨C', A, Y, U, Vt, W, Zt⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelLow, eval_corr]
  exact (lowRank_peel x.1 x.2 U Vt W Zt).symm

/-- The low-rank peel costs four products through the inner dimension `r`: `4 κ n² r`. -/
theorem time_peelLow_le (n r : ℕ) : Mc.time (Mc.peelLow n r) ≤ 4 * Mc.κ * (n : ℝ) ^ 2 * r := by
  apply le_trans
  · unfold peelLow
    apply_rules (maxDepth := 200) [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- **The honest useful computation, full-rank masks**: `(A, B, R, S) ↦ (A + R)(B + S) − R (B + S) − A S`, the masked
product and its peel (RS-DEF §1.7's Solve without its commitments and proof). -/
def honestFull (n : ℕ) : Mc.Alg (Mat R n n × Mat R n n × Mat R n n × Mat R n n) (Mat R n n) :=
  Mc.sub₂
    (Mc.sub₂
      (Mc.mul₂ (Mc.add₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd))))
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))
    (Mc.mul₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))

theorem eval_honestFull (n : ℕ) (x : Mat R n n × Mat R n n × Mat R n n × Mat R n n) :
    Mc.eval (Mc.honestFull n) x = x.1 * x.2.1 := by
  simp only [honestFull, eval_sub₂, eval_mul₂, eval_add₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  exact (mask_peel x.1 x.2.2.1 x.2.1 x.2.2.2).symm

/-- The full-rank honest computation costs three `n × n × n` products: `γ ≤ 2 + 5 κadd / (κ n)` against the
reference `κ n³` (RS-DEF's Proposition Z(e): full-rank noise buys nothing). -/
theorem time_honestFull_le (n : ℕ) :
    Mc.time (Mc.honestFull n) ≤ 3 * Mc.κ * (n : ℝ) ^ 3 + 5 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold honestFull
    apply_rules (maxDepth := 200) [time_mul₂_le_of, time_sub₂_le_of, time_add₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- **The honest useful computation, rank-`r` masks**:
`(A, B, U, Vᵀ, W, Zᵀ) ↦ (A + U Vᵀ)(B + W Zᵀ) − U (Vᵀ (B + W Zᵀ)) − (A W) Zᵀ`. -/
def honestLow (n r : ℕ) :
    Mc.Alg (Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n) (Mat R n n) :=
  Mc.sub₂
    (Mc.sub₂
      (Mc.mul₂
        (Mc.add₂ Mc.fst
          (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))))
        (Mc.add₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
            (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))
        (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
          (Mc.add₂ (Mc.seq Mc.snd Mc.fst)
            (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
              (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))))
    (Mc.mul₂ (Mc.mul₂ Mc.fst (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))))
      (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_honestLow (n r : ℕ) (x : Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n) :
    Mc.eval (Mc.honestLow n r) x = x.1 * x.2.1 := by
  simp only [honestLow, eval_sub₂, eval_mul₂, eval_add₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  exact lowRank_peel x.1 x.2.1 x.2.2.1 x.2.2.2.1 x.2.2.2.2.1 x.2.2.2.2.2

/-- The rank-`r` honest computation costs one `n × n × n` product and seven products through the inner dimension `r`
(the mask `W Zᵀ` is formed twice): `γ ≤ 7 r / n + 5 κadd / (κ n)` against the reference `κ n³`. -/
theorem time_honestLow_le (n r : ℕ) :
    Mc.time (Mc.honestLow n r) ≤ Mc.κ * (n : ℝ) ^ 3 + 7 * Mc.κ * (n : ℝ) ^ 2 * r + 5 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold honestLow
    apply_rules (maxDepth := 300) [time_mul₂_le_of, time_sub₂_le_of, time_add₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

variable {n : ℕ} {Inst : Type} {P : Protocol Mc.toMachine Inst (Mat R n n)} {emb : Inst → Mat R n n × Mat R n n}

/-- **Relative security of matmul PoUW with full-rank masks**, from extraction (named hypothesis `Extraction`, overhead
`εext`) and the floor (named hypothesis `Floor`) above the peel's and the correction's cost
`q = 2κ n³ + 2κadd n²`: `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_fullRank (hmat : IsMatmulOn P emb) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidFull (emb x)) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η) (hT : 2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelFull n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n]) hT

/-- **Relative security of matmul PoUW with rank-`r` masks**, from extraction and the floor above
`q = 4κ n² r + 2κadd n²`: `ε = εext + (1 + εext) q / (T − q)`, and `q / (T − q)` is `O(r / n)` against a floor of order
`n³`. (RS-EXT's Lemma U: planting rank-`r` masks is detectable, so for RS-D2 this `Extraction` is expected to fail and
security comes from the floor itself, `core_of_floor`.) -/
theorem secure_lowRank (hmat : IsMatmulOn P emb) (r : ℕ) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidLow (r := r) (emb x)) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelLow n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n]) hT

/-- **Full-rank masks by route 2**: with every accepted adversary costing at least `F`,
`ε = εext + (2κ n³ + 2κadd n²) / F`, about `2/c` for `F = c n³` (RS-MODEL §5: says nothing useful). -/
theorem secure_fullRank_advTime (hmat : IsMatmulOn P emb) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidFull (emb x)) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) / F) δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelFull n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hM hF
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n])

/-- **Rank-`r` masks by route 2**: with every accepted adversary costing at least `F`,
`ε = εext + (4κ n² r + 2κadd n²) / F`, i.e. `(4κ r + 2κadd) / (c n)` for `F = c n³` (RS-MODEL §5's table, operands on
the grid). -/
theorem secure_lowRank_advTime (hmat : IsMatmulOn P emb) (r : ℕ) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidLow (r := r) (emb x)) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) / F) δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelLow n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hM hF
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n])

end MatMachine

end MatmulPeel

section EditedInstances

/-! ## Route (c): edited instances (Daniel, 18:20Z and 18:28Z)

The protocol edits both operands, unpeeled: the useful output is the edited product `(A + E_A(σ)) (B + E_B(σ))`,
and the edited product is what is served. Dynamic × dynamic only: both operands are the instance, adversarial, and
both are edited fresh per instance. With no peel there are no correction products, so γ is the proof layer's cost
(commitments, proving), the noise expansion and two additions. The floor is RS-D1E's (named `EditFloor`). -/

/-- The bit-pair projection's algebra (RS-D1E): the mixed second difference of `(A + S) D (B + T)` in the two edits
is `S D T`. The committed parts project to 0: `A D B`, `A D T` and `S D B` each lack one of the two edits. -/
theorem edit_mixed_difference {R : Type} [Ring R] {a b c : ℕ} (A S : Mat R a b) (D : Mat R b b) (B T : Mat R b c) :
    (A + S) * D * (B + T) - (A + S) * D * B - A * D * (B + T) + A * D * B = S * D * T := by
  simp only [Matrix.add_mul, Matrix.mul_add]
  abel

/-- **The bit-pair coefficient** (RS-D1E): editing `A` at `(i, k)` by `d β` and `B` at `(k, l)` by `d' β'`, the
`β β'` coefficient of the prefix-sum target `A' diag(p) B'` (with `p` the prefix's indicator) is `d p_k d'` at
`(i, l)` and 0 elsewhere, whatever the committed `A` and `B`. -/
theorem bitPair_coefficient {R : Type} [CommRing R] {a b c : ℕ} (A : Mat R a b) (B : Mat R b c) (p : Fin b → R)
    (i : Fin a) (k : Fin b) (l : Fin c) (d d' : R) :
    (A + Matrix.single i k d) * Matrix.diagonal p * (B + Matrix.single k l d') -
        (A + Matrix.single i k d) * Matrix.diagonal p * B - A * Matrix.diagonal p * (B + Matrix.single k l d') +
        A * Matrix.diagonal p * B =
      Matrix.single i l (d * p k * d') := by
  rw [edit_mixed_difference, Matrix.single_mul_mul_single, Matrix.diagonal_apply_eq]

/-! ### The abstract core of RS-D1E's homogenization (Lemma 3 and linearity, for one word)

Words are functions of the noise bits `z : X ⊕ Y → K` (x-bits and y-bits). The bit-pair projection of a word `f` is
RS-D1E's mixed second difference, `P(f)[x, y] = f(e_x + e_y) − f(e_x) − f(e_y) + f(0)`. It is linear, kills affine
words, and sends a product of two affine (plain) operands to two rank-one terms; so a word computed with `m` plain
products and linear steps projects to rank at most `2m`, and a word whose projection has rank `r` needs at least
`r / 2` products: the products needed for a word are at least half the rank of its degree-(1, 1) part. Over
`Z/2³²` the rank is too coarse and Theorem W's `r_k` replaces it. Many target words add up when their live terms are
disjoint (`editedEntries_products_ge`); Theorem B's accounting with merges (`F(G)`) is still on paper. -/

/-- An affine (plain) operand in the noise bits: `c + Σ α_x z_x + Σ γ_y z_y`. -/
structure Aff (X Y K : Type) where
  c : K
  α : X → K
  γ : Y → K

/-- Its value at the noise `z`. -/
def Aff.eval {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] (u : Aff X Y K) (z : X ⊕ Y → K) : K :=
  u.c + ∑ x, u.α x * z (.inl x) + ∑ y, u.γ y * z (.inr y)

/-- The noise `s e_x + t e_y`. -/
def bitPt {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (x : X) (y : Y) (s t : K) : X ⊕ Y → K :=
  Pi.single (.inl x) s + Pi.single (.inr y) t

/-- **The bit-pair projection** (RS-D1E): `P(f)[x, y] = f(e_x + e_y) − f(e_x) − f(e_y) + f(0)`. -/
def bitProj {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (f : (X ⊕ Y → K) → K) : Matrix X Y K :=
  fun x y => f (bitPt x y 1 1) - f (bitPt x y 1 0) - f (bitPt x y 0 1) + f (bitPt x y 0 0)

theorem Aff.eval_bitPt {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u : Aff X Y K) (x : X) (y : Y) (s t : K) : u.eval (bitPt x y s t) = u.c + u.α x * s + u.γ y * t := by
  simp [Aff.eval, bitPt, Pi.single_apply]

/-- **RS-D1E's Lemma 3** over a commutative ring: a product of two plain operands projects to `α_u γ_vᵀ + α_v γ_uᵀ`,
and its constants (the instance, the pair offsets, anything computed before the coins) cancel. -/
theorem bitProj_mul {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u v : Aff X Y K) :
    bitProj (fun z => u.eval z * v.eval z) = Matrix.vecMulVec u.α v.γ + Matrix.vecMulVec v.α u.γ := by
  ext x y
  simp only [bitProj, Aff.eval_bitPt, Matrix.add_apply, Matrix.vecMulVec_apply]
  ring

/-- Plain words project to 0. -/
theorem bitProj_aff {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    (u : Aff X Y K) : bitProj (fun z => u.eval z) = 0 := by
  ext x y
  simp only [bitProj, Aff.eval_bitPt, Matrix.zero_apply]
  ring

/-- The projection is additive. -/
theorem bitProj_add {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] (f g : (X ⊕ Y → K) → K) :
    bitProj (fun z => f z + g z) = bitProj f + bitProj g := by
  ext x y
  simp only [bitProj, Matrix.add_apply]
  ring

theorem bitProj_sum {X Y K : Type} [DecidableEq X] [DecidableEq Y] [CommRing K] {m : ℕ}
    (f : Fin m → (X ⊕ Y → K) → K) : bitProj (fun z => ∑ j, f j z) = ∑ j, bitProj (f j) := by
  ext x y
  simp only [bitProj, Matrix.sum_apply, Finset.sum_sub_distrib, Finset.sum_add_distrib]

/-- A sum of `m` pairs of rank-one terms has rank at most `2m`: it factors through `Fin m ⊕ Fin m`. -/
theorem rank_sum_vecMulVec_le {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] [StrongRankCondition K] {m : ℕ}
    (a c : Fin m → X → K) (b d : Fin m → Y → K) :
    (∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j))).rank ≤ 2 * m := by
  let U : Matrix X (Fin m ⊕ Fin m) K := fun x s => Sum.elim (fun j => a j x) (fun j => c j x) s
  let V : Matrix (Fin m ⊕ Fin m) Y K := fun s y => Sum.elim (fun j => b j y) (fun j => d j y) s
  have hUV : ∑ j, (Matrix.vecMulVec (a j) (b j) + Matrix.vecMulVec (c j) (d j)) = U * V := by
    ext x y
    rw [Matrix.mul_apply, Fintype.sum_sum_type]
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.vecMulVec_apply, Finset.sum_add_distrib, U, V,
      Sum.elim_inl, Sum.elim_inr]
  rw [hUV]
  refine (Matrix.rank_mul_le_left U V).trans ((Matrix.rank_le_card_width U).trans ?_)
  simp [two_mul]

/-- **The core of RS-D1E's homogenization, for one word**: a word computed with `m` products of plain operands and
linear steps (`Σ_j u_j v_j + r`, scalars absorbed into the operands) projects to rank at most `2m`. -/
theorem bitProj_rank_le {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    [StrongRankCondition K] {m : ℕ} (u v : Fin m → Aff X Y K) (r : Aff X Y K) :
    (bitProj fun z => ∑ j, (u j).eval z * (v j).eval z + r.eval z).rank ≤ 2 * m := by
  rw [bitProj_add (fun z => ∑ j, (u j).eval z * (v j).eval z) (fun z => r.eval z), bitProj_aff, add_zero,
    bitProj_sum (fun j z => (u j).eval z * (v j).eval z)]
  simp only [bitProj_mul]
  exact rank_sum_vecMulVec_le _ _ _ _

/-- **Products needed ≥ half the rank of the degree-(1, 1) part**: if a word computed with `m` plain products agrees
with a target `T` at the four points of every bit pair, then `rank P(T) ≤ 2m`. -/
theorem products_ge_half_rank {X Y K : Type} [Fintype X] [Fintype Y] [DecidableEq X] [DecidableEq Y] [CommRing K]
    [StrongRankCondition K] {m : ℕ} (u v : Fin m → Aff X Y K) (r : Aff X Y K) (T : (X ⊕ Y → K) → K)
    (hT : ∀ x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      T (bitPt x y s t) = ∑ j, (u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t) + r.eval (bitPt x y s t)) :
    (bitProj T).rank ≤ 2 * m := by
  have h : bitProj T = bitProj fun z => ∑ j, (u j).eval z * (v j).eval z + r.eval z := by
    ext x y
    simp only [bitProj]
    rw [hT x y 1 1 (.inr rfl) (.inr rfl), hT x y 1 0 (.inr rfl) (.inl rfl), hT x y 0 1 (.inl rfl) (.inr rfl),
      hT x y 0 0 (.inl rfl) (.inl rfl)]
  rw [h]
  exact bitProj_rank_le u v r

/-- The edited prefix-sum entry `(i, l)` of `A′ diag(p) B′` as a word in the noise bits (RS-D1E's Lemma 1 form):
`A′_ik = c_ik + d_ik z_(i,k)` and `B′_kl = c′_kl + d′_kl z′_(k,l)`, the constants `c` adversarial. -/
def editedEntry {K : Type} [CommRing K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c) (p : Fin b → K)
    (i : Fin a) (l : Fin c) (z : (Fin a × Fin b) ⊕ (Fin b × Fin c) → K) : K :=
  ∑ k, (C i k + D i k * z (.inl (i, k))) * p k * (C' k l + D' k l * z (.inr (k, l)))

/-- The bit-pair projection of an edited prefix-sum entry: `Σ_k e_(i,k) e_(k,l)ᵀ · d_ik p_k d′_kl`, the committed
constants gone. -/
theorem bitProj_editedEntry {K : Type} [CommRing K] {a b c : ℕ} (C D : Mat K a b) (C' D' : Mat K b c) (p : Fin b → K)
    (i : Fin a) (l : Fin c) :
    bitProj (editedEntry C D C' D' p i l) =
      ∑ k, Matrix.vecMulVec (Pi.single (i, k) (D i k * p k)) (Pi.single (k, l) (D' k l)) := by
  let U : Fin b → Aff (Fin a × Fin b) (Fin b × Fin c) K := fun k => ⟨C i k * p k, Pi.single (i, k) (D i k * p k), 0⟩
  let V : Fin b → Aff (Fin a × Fin b) (Fin b × Fin c) K := fun k => ⟨C' k l, 0, Pi.single (k, l) (D' k l)⟩
  have hw : editedEntry C D C' D' p i l =
      fun z => ∑ k, (U k).eval z * (V k).eval z + (⟨0, 0, 0⟩ : Aff _ _ K).eval z := by
    funext z
    simp only [editedEntry, Aff.eval, U, V, Pi.single_apply, Pi.zero_apply, zero_mul, Finset.sum_const_zero,
      add_zero, ite_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    refine Finset.sum_congr rfl fun k _ => ?_
    ring
  rw [hw, bitProj_add (fun z => ∑ k, (U k).eval z * (V k).eval z), bitProj_aff, add_zero,
    bitProj_sum (fun k z => (U k).eval z * (V k).eval z)]
  simp only [bitProj_mul, U, V, Matrix.zero_vecMulVec, add_zero]

/-- **RS-D1E's Lemma 2, counted** (over a field): a word computed with `m` plain products that agrees with an
edited prefix-sum entry at every bit pair's four points has `m ≥ #{k : d_ik p_k d′_kl ≠ 0} / 2`, whatever the
committed constants. (One entry; entries with disjoint live terms add up, `editedEntries_products_ge`.) -/
theorem editedEntry_products_ge {K : Type} [Field K] [DecidableEq K] {a b c m : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (i : Fin a) (l : Fin c)
    (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K) (r : Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p i l (bitPt x y s t) =
        ∑ j, (u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t) + r.eval (bitPt x y s t)) :
    Fintype.card {k // D i k * p k * D' k l ≠ 0} ≤ 2 * m := by
  have hproj := bitProj_editedEntry C D C' D' p i l
  have hdiag : (bitProj (editedEntry C D C' D' p i l)).submatrix (fun k => (i, k)) (fun k => (k, l)) =
      Matrix.diagonal fun k => D i k * p k * D' k l := by
    rw [hproj]
    ext k k'
    simp only [Matrix.submatrix_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Prod.mk.injEq,
      true_and, and_true, Matrix.diagonal_apply]
    by_cases h : k = k'
    · subst h
      simp [mul_assoc]
    · simp [h]
  calc Fintype.card {k // D i k * p k * D' k l ≠ 0}
      = (Matrix.diagonal fun k => D i k * p k * D' k l).rank := (Matrix.rank_diagonal _).symm
    _ ≤ (bitProj (editedEntry C D C' D' p i l)).rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _
    _ ≤ 2 * m := products_ge_half_rank u v r _ hT

/-- A scalar multiple of a plain operand. -/
def Aff.smul {X Y K : Type} [CommRing K] (s : K) (u : Aff X Y K) : Aff X Y K :=
  ⟨s * u.c, fun x => s * u.α x, fun y => s * u.γ y⟩

theorem Aff.eval_smul {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] (s : K) (u : Aff X Y K)
    (z : X ⊕ Y → K) : (u.smul s).eval z = s * u.eval z := by
  simp only [Aff.smul, Aff.eval, mul_add, Finset.mul_sum, mul_assoc]

/-- The sum of plain operands. -/
def Aff.sum {X Y K : Type} [CommRing K] {q : ℕ} (r : Fin q → Aff X Y K) : Aff X Y K :=
  ⟨∑ e, (r e).c, fun x => ∑ e, (r e).α x, fun y => ∑ e, (r e).γ y⟩

theorem Aff.eval_sum {X Y K : Type} [Fintype X] [Fintype Y] [CommRing K] {q : ℕ} (r : Fin q → Aff X Y K)
    (z : X ⊕ Y → K) : (Aff.sum r).eval z = ∑ e, (r e).eval z := by
  simp only [Aff.sum, Aff.eval, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm (s := (Finset.univ : Finset X)), Finset.sum_comm (s := (Finset.univ : Finset Y))]

/-- **Entries with disjoint live terms add up** (RS-D1E's Theorem B accounting without merges, over a field): if `m`
plain products, shared, compute `q` edited prefix-sum entries `(i_e, l_e)` at distinct rows and distinct columns (each
entry's word a linear combination of the products plus a plain word, agreeing at every bit pair's four points), then
`2m ≥ Σ_e #{k : d_(i_e k) p_k d′_(k l_e) ≠ 0}`, whatever the committed constants. -/
theorem editedEntries_products_ge {K : Type} [Field K] [DecidableEq K] {a b c m q : ℕ} (C D : Mat K a b)
    (C' D' : Mat K b c) (p : Fin b → K) (ent : Fin q → Fin a × Fin c)
    (hrow : Function.Injective fun e => (ent e).1) (hcol : Function.Injective fun e => (ent e).2)
    (lam : Fin q → Fin m → K) (u v : Fin m → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (r : Fin q → Aff (Fin a × Fin b) (Fin b × Fin c) K)
    (hT : ∀ e x y s t, (s = 0 ∨ s = 1) → (t = 0 ∨ t = 1) →
      editedEntry C D C' D' p (ent e).1 (ent e).2 (bitPt x y s t) =
        ∑ j, lam e j * ((u j).eval (bitPt x y s t) * (v j).eval (bitPt x y s t)) + (r e).eval (bitPt x y s t)) :
    ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} ≤ 2 * m := by
  have hr : ∀ e e', (ent e).1 = (ent e').1 ↔ e = e' := fun _ _ => hrow.eq_iff
  have hc : ∀ e e', (ent e).2 = (ent e').2 ↔ e = e' := fun _ _ => hcol.eq_iff
  let W : ((Fin a × Fin b) ⊕ (Fin b × Fin c) → K) → K := fun z => ∑ e, editedEntry C D C' D' p (ent e).1 (ent e).2 z
  have hW : (bitProj W).rank ≤ 2 * m := by
    refine products_ge_half_rank (fun j => (u j).smul (∑ e, lam e j)) v (Aff.sum r) W fun x y s t hs ht => ?_
    simp only [W, Aff.eval_smul, Aff.eval_sum]
    rw [Finset.sum_congr rfl fun e _ => hT e x y s t hs ht, Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Finset.sum_mul, Finset.sum_mul]
    refine Finset.sum_congr rfl fun e _ => ?_
    ring
  have hproj : bitProj W = ∑ e, ∑ k,
      Matrix.vecMulVec (Pi.single ((ent e).1, k) (D (ent e).1 k * p k)) (Pi.single (k, (ent e).2) (D' k (ent e).2)) := by
    rw [bitProj_sum (fun e => editedEntry C D C' D' p (ent e).1 (ent e).2)]
    exact Finset.sum_congr rfl fun e _ => bitProj_editedEntry C D C' D' p _ _
  have hdiag : (bitProj W).submatrix (fun ek : Fin q × Fin b => ((ent ek.1).1, ek.2))
      (fun ek : Fin q × Fin b => (ek.2, (ent ek.1).2)) =
      Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 := by
    rw [hproj]
    ext ⟨e, k⟩ ⟨e', k'⟩
    simp only [Matrix.submatrix_apply, Matrix.sum_apply, Matrix.vecMulVec_apply, Pi.single_apply, Prod.mk.injEq,
      hr, hc, Matrix.diagonal_apply]
    by_cases h : e = e' ∧ k = k'
    · obtain ⟨rfl, rfl⟩ := h
      simp [ite_and, mul_assoc]
    · simp only [h, ↓reduceIte]
      refine Finset.sum_eq_zero fun e'' _ => Finset.sum_eq_zero fun k'' _ => ?_
      by_cases h1 : e = e'' ∧ k = k''
      · have h2 : ¬(k' = k'' ∧ e' = e'') := fun h2 => h ⟨h1.1.trans h2.2.symm, h1.2.trans h2.1.symm⟩
        simp only [h2, ↓reduceIte, mul_zero]
      · simp only [h1, ↓reduceIte, zero_mul]
  have hcard : ∑ e, Fintype.card {k // D (ent e).1 k * p k * D' k (ent e).2 ≠ 0} =
      Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0} := by
    simp only [Fintype.card_subtype, Finset.card_filter]
    rw [Fintype.sum_prod_type]
  rw [hcard]
  calc Fintype.card {ek : Fin q × Fin b // D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2 ≠ 0}
      = (Matrix.diagonal fun ek : Fin q × Fin b => D (ent ek.1).1 ek.2 * p ek.2 * D' ek.2 (ent ek.1).2).rank :=
        (Matrix.rank_diagonal _).symm
    _ ≤ (bitProj W).rank := hdiag ▸ Matrix.rank_submatrix_le _ _ _
    _ ≤ 2 * m := hW

/-- The proof service's layer of a protocol (AGENTS.md's one proof service): live coins, messages and the verifier.
A protocol is a service with a problem (`correct`) and an honest Solve. -/
structure Service where
  Seed : Type
  [instSeed : Fintype Seed]
  lam : ℕ
  card_seed : Fintype.card Seed = 2 ^ lam
  Com : Type
  Ans : Type
  Coins : Type
  [instCoins : Fintype Coins]
  verify : Com → Seed → Ans → Bool

attribute [instance] Service.instSeed Service.instCoins

/-- **The service's adversary floor** in a machine: every adversary that commits first has its accepted runs charged
(`max(|M|, cost)`) below `F` with probability at most `κ`. It reads only the service, so it is `AdvFloor` of every protocol over the service
(`MatMachine.editFloor_iff`). -/
def ServiceFloor (Mc : Machine) (S : Service) (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    prob (fun ω : S.Seed × Ω => S.verify (Mc.eval M ω).1 ω.1 (Mc.eval M ω).2 = true ∧ Mc.charge M ω < F) ≤ κ

/-- **The window's rounds** (Theorem S's and S_G's setting, a named hypothesis): the service's coins are `R` rounds
of fresh coins on `Ω` (`e`), and every adversary that commits first plays, at each value `r` of its own coins, a
round game that meets the per-round pass bound along its guard paths (`PathBound p`; `pathBound_of_passBound` gives
it for straight-line rounds). Its accepted runs are accepted by the game or break binding or zero knowledge in some
round (probabilities at most `δb` and `δz`), and its charge is at least `(R − Σ_w d_w) F₀ − V − R sl` with noise
reads `V ≤ Vmax` (Theorem E½ per round, `x_w F₀ − L(M + K)` with `sl = L(M + K)`, and Lemma RD's `n v_λ`). -/
def RoundsHyp (Mc : Machine) (S : Service) (Ω : Type) [Fintype Ω] (R : ℕ) (p F₀ sl Vmax δb δz : ℝ) : Prop :=
  ∃ e : S.Seed ≃ Paths Ω R, ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
    (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
    ∃ (G : RoundGame Ω R) (Bind ZK : Fin R → Paths Ω R → Prop) (V : Paths Ω R → ℝ),
      G.PathBound p ∧ (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        G.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, (R - G.deficiency x) * F₀ - V x - R * sl ≤ Mc.charge M (e.symm x, r)

/-- **The window's service floor from Theorem S** (in place of Theorem M): under the window's rounds, every adversary
that commits first has its accepted runs charged below `(R − D) F₀ − Vmax − R sl` with probability at most
`e^{−(1−p) D} + R (δb + δz)`. -/
theorem serviceFloor_of_rounds {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp Mc S Ω R p F₀ sl Vmax δb δz) (hp : p ≤ 1) (hF₀ : 0 < F₀)
    (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor Mc S ((R - D) * F₀ - Vmax - R * sl) (Real.exp (-((1 - p) * D)) + R * (δb + δz)) := by
  obtain ⟨e, h⟩ := h
  intro Ω' _ M hc
  refine prob_le_of_slices _ (add_nonneg (Real.exp_pos _).le (mul_nonneg (Nat.cast_nonneg _) (add_nonneg hδb hδz)))
    fun r => ?_
  obtain ⟨G, Bind, ZK, V, hG, hB, hZ, hacc, hV, hcost⟩ := h Ω' M hc r
  rw [prob_equiv e]
  refine (prob_mono fun x hx => ⟨hx.1, hx.2.trans_le ?_⟩).trans
    (G.window_le hp hG _ Bind ZK hB hZ hacc (fun x => Mc.charge M (e.symm x, r)) V hF₀ hcost D)
  linarith [hV x]

/-- Both operands lie in `G_use = [−g, g]` (RS-DEF v2 §1.10). -/
def InGrid {n : ℕ} (g : ℤ) (x : Mat ℤ n n × Mat ℤ n n) : Prop := ∀ i j, |x.1 i j| ≤ g ∧ |x.2 i j| ≤ g

/-- **A range-checking service** (RS-DEF v2 §1.3 after RS-RT's V6): the commitment binds the instance `(A_M, B_M)`
(`committed`), and for each block σ opens (`opened`; every entry for a full check) the verifier's relation includes
the range check `A_M, B_M ∈ G_use = [−g, g]`, proved inside the zero-knowledge proof. -/
structure GridService (n : ℕ) (g : ℤ) extends Service where
  committed : Com → Mat ℤ n n × Mat ℤ n n
  opened : Seed → Fin n → Fin n → Prop
  range_check : ∀ c σ a, verify c σ a = true → ∀ i j, opened σ i j →
    |(committed c).1 i j| ≤ g ∧ |(committed c).2 i j| ≤ g

/-- With every entry opened, an accepted transcript's committed instance is in `G_use`. -/
theorem GridService.inGrid_of_accepts {n : ℕ} {g : ℤ} (S : GridService n g) (hall : ∀ σ i j, S.opened σ i j)
    {c : S.Com} {σ : S.Seed} {a : S.Ans} (h : S.verify c σ a = true) : InGrid g (S.committed c) :=
  fun i j => S.range_check c σ a h i j (hall σ i j)

namespace MatMachine

variable {R : Type} [Ring R] (Mc : MatMachine R)

/-- **An edit** (route (c)): a σ-indexed map on each operand, `A ↦ A + E_A(σ)` and `B ↦ B + E_B(σ)`, with the
algorithm `gen` that expands σ into the two noise matrices (its time is part of γ). -/
structure Edit (Seed : Type) (n : ℕ) where
  EA : Seed → Mat R n n
  EB : Seed → Mat R n n
  gen : Mc.Alg Seed (Mat R n n × Mat R n n)
  eval_gen : ∀ σ, Mc.eval gen σ = (EA σ, EB σ)

variable {Mc} in
/-- The edited instance `(A + E_A(σ), B + E_B(σ))`. -/
def Edit.edit {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) :
    Mat R n n × Mat R n n :=
  (x.1 + E.EA σ, x.2 + E.EB σ)

variable {Mc} in
/-- The edited product `(A + E_A(σ)) (B + E_B(σ))`, the useful output on the instance `(A, B)`. -/
def Edit.product {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) : Mat R n n :=
  (x.1 + E.EA σ) * (x.2 + E.EB σ)

/-- The noise is bounded by `s` (`|E| ≤ max|S|`, RS-DEF v2 §1.10). -/
def Edit.Bounded {Mc : MatMachine ℤ} {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (s : ℤ) : Prop :=
  ∀ σ i j, |E.EA σ i j| ≤ s ∧ |E.EB σ i j| ≤ s

/-- The edited product as an algorithm, with the product computed by `m`, `(σ, (A, B)) ↦ m (A + E_A σ, B + E_B σ)`:
expand σ once, two additions, one run of `m`, and no peel. `m` is the worst-case `mul n n n` on unrestricted
operands, or a cheaper product on a grid the edited operands stay on (`GridMatMachine`). -/
def editMul {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n)) :
    Mc.Alg (Seed × (Mat R n n × Mat R n n)) (Mat R n n) :=
  Mc.seq (Mc.pair Mc.snd (Mc.seq Mc.fst E.gen))
    (Mc.lift2 (Mc.add₂ (Mc.seq Mc.fst Mc.fst) (Mc.seq Mc.snd Mc.fst))
      (Mc.add₂ (Mc.seq Mc.fst Mc.snd) (Mc.seq Mc.snd Mc.snd)) m)

theorem eval_editMul {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (y : Seed × (Mat R n n × Mat R n n)) : Mc.eval (Mc.editMul E m) y = Mc.eval m (E.edit y.1 y.2) := by
  simp only [editMul, Machine.eval_lift2, eval_add₂, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd,
    E.eval_gen, Edit.edit]

theorem time_editMul_le {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n)) :
    Mc.time (Mc.editMul E m) ≤ Mc.time E.gen + Mc.time m + 2 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold editMul Machine.lift2
    apply_rules [time_add₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le,
      Machine.time_snd_le, le_refl]
  · linarith

/-- **Honest Solve on edited instances** (`honestEdit`): compute the edited product with `m`, then run the proof layer
`prove` on the transcript's inputs and the served product; the useful output is the edited product, unpeeled. -/
def honestEdit {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf) :
    Mc.Alg (Seed × (Mat R n n × Mat R n n) × Coins) (Pf × Mat R n n) :=
  Mc.seq (Mc.pair (Mc.pair Mc.fst Mc.snd) (Mc.seq (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.fst)) (Mc.editMul E m)))
    (Mc.pair prove Mc.snd)

theorem eval_honestEdit {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf)
    (y : Seed × (Mat R n n × Mat R n n) × Coins) :
    Mc.eval (Mc.honestEdit E m prove) y =
      (Mc.eval prove (y, Mc.eval m (E.edit y.1 y.2.1)), Mc.eval m (E.edit y.1 y.2.1)) := by
  simp only [honestEdit, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd, eval_editMul, Prod.mk.eta]

/-- `honestEdit` costs one run of `m` and two additions beyond the proof layer and the noise expansion:
`time ≤ time m + 2 κadd n² + time gen + time prove`, with no peel. -/
theorem time_honestEdit_le {Seed Coins Pf : Type} {n : ℕ} (E : Mc.Edit Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((Seed × (Mat R n n × Mat R n n) × Coins) × Mat R n n) Pf) :
    Mc.time (Mc.honestEdit E m prove) ≤
      Mc.time m + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove := by
  apply le_trans
  · unfold honestEdit
    apply_rules [Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le,
      time_editMul_le, le_refl]
  · linarith [Mc.time_add_le n n]

/-- **The edited protocol** on the instance `(A, B)` (both operands adversarial, both edited), a promise problem on
the instances `use` (all of them, or those whose edits stay on a grid): the service's coins, messages and verifier,
the edited product as the answer on every promised instance, and `honestEdit` with the product `m` as Solve. -/
def edited (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) :
    Protocol Mc.toMachine (Mat R n n × Mat R n n) (Mat R n n) where
  Seed := S.Seed
  lam := S.lam
  card_seed := S.card_seed
  Com := S.Com
  Ans := S.Ans
  Coins := S.Coins
  correct σ x C := E.product σ x = C
  promise := use
  solve := Mc.honestEdit E m prove
  verify := S.verify

/-- **Completeness of the edited protocol**: if `m` computes the edited product on every promised instance (`hm`), the
proof layer's commitment reads neither σ nor the product (`hcom`, commit-first) and the verifier accepts its proof of
the served product with probability at least `1 − η` on every promised instance (`hacc`), the edited protocol is
complete. -/
theorem edited_complete (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {η : ℝ}
    (hm : ∀ σ x, use x → Mc.eval m (E.edit σ x) = E.product σ x)
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).2 = true) :
    Complete (Mc.edited S E m prove use) η := by
  refine ⟨fun σ σ' y => ?_, fun x hx => (hacc x hx).trans (prob_mono fun ω hω => ?_)⟩
  · change ((Mc.eval (Mc.honestEdit E m prove) (σ, y)).1).1 = ((Mc.eval (Mc.honestEdit E m prove) (σ', y)).1).1
    rw [eval_honestEdit, eval_honestEdit]
    exact hcom σ σ' y _ _
  · change E.product ω.1 x = (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).2 ∧
      S.verify (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).1.1 ω.1
        (Mc.eval (Mc.honestEdit E m prove) (ω.1, x, ω.2)).1.2 = true
    rw [eval_honestEdit]
    exact ⟨(hm ω.1 x hx).symm, hω⟩

/-- **The edit floor** (a named hypothesis; RS-D1E proves it on paper by the bit-pair projection, `bitPair_coefficient`,
with `F = c n³` per MAC slot, `c = ½` unconditional through Theorem W and `c = 1` under Conjecture A): on the edited
protocol, every adversary that commits first has its accepted runs below `F` with probability at most `κ`. -/
def EditFloor (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) (F κ : ℝ) : Prop :=
  AdvFloor (Mc.edited S E m prove use) F κ

/-- The edit floor reads only the service: it is the service's floor in the machine. -/
theorem editFloor_iff (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n) (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) (F κ : ℝ) :
    Mc.EditFloor S E m prove use F κ ↔ ServiceFloor Mc.toMachine S F κ :=
  Iff.rfl

/-- **Relative security of edited instances** (route (c), end to end, for a product `m` at `κm` per
multiply-accumulate on the promised instances): with the edited protocol complete (`hm`, `hcom`, `hacc`) and the edit
floor `F` (named hypothesis `EditFloor`, with `κf < δ`), the edited protocol is relatively secure against the reference
`N = κm n³` with `ε = U / F − 1`, `U = κm n³ + 2κadd n² + time gen + time prove`, and
`γ = (2κadd n² + time gen + time prove) / (κm n³)`: the proof layer, the noise expansion and two additions, no peel. -/
theorem relativelySecure_of_editFloor (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (m : Mc.Alg (Mat R n n × Mat R n n) (Mat R n n))
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {κm κf δ η F : ℝ}
    (hm : ∀ σ x, use x → Mc.eval m (E.edit σ x) = E.product σ x) (hmt : Mc.time m ≤ κm * (n : ℝ) ^ 3)
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval m (E.edit ω.1 x))).2 = true)
    (hfloor : Mc.EditFloor S E m prove use F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < κm * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E m prove use)
      ((κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (κm * (n : ℝ) ^ 3)) η 0
      (κm * (n : ℝ) ^ 3) := by
  have hc := Mc.edited_complete S E m prove use hm hcom hacc
  have hU : Mc.time (Mc.honestEdit E m prove) ≤
      κm * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove := by
    linarith [Mc.time_honestEdit_le E m prove]
  refine ⟨hc, secureInf_of_core _ (core_of_floor _ hF hFU hfloor hc hU) hκδ ?_, ?_⟩
  · rw [add_sub_cancel]
    exact div_pos (hF.trans_le hFU) hF
  · change Mc.time (Mc.honestEdit E m prove) ≤ _
    rw [add_mul, one_mul, div_mul_cancel₀ _ hN.ne', add_zero]
    linarith

/-- The worst-case product computes the edited product on every instance. -/
theorem eval_mul_edit {Seed : Type} {n : ℕ} (E : Mc.Edit Seed n) (σ : Seed) (x : Mat R n n × Mat R n n) :
    Mc.eval (Mc.mul n n n) (E.edit σ x) = E.product σ x :=
  Mc.eval_mul n n n _

/-- **Route (c) on unrestricted operands** (`relativelySecure_of_editFloor` with the worst-case product `mul n n n`
at `κ`, and every instance promised): reference `N = κ n³`. -/
theorem relativelySecure_of_editFloor_wide (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans)) {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), E.product ω.1 x)).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), E.product ω.1 x)).2 = true)
    (hfloor : Mc.EditFloor S E (Mc.mul n n n) prove (fun _ => True) F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < Mc.κ * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E (Mc.mul n n n) prove fun _ => True)
      ((Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (Mc.κ * (n : ℝ) ^ 3)) η 0
      (Mc.κ * (n : ℝ) ^ 3) :=
  Mc.relativelySecure_of_editFloor S E _ prove _ (fun σ x _ => Mc.eval_mul_edit E σ x)
    ((Mc.time_mul_le n n n).trans_eq (by ring)) hcom (fun x _ => by simpa only [Mc.eval_mul_edit] using hacc x)
    hfloor hκδ hF hFU hN

end MatMachine

/-- **A matrix machine with a grid product** (the width-typed refinement): on the shapes `shape` and on operands
whose entries lie in the grid `G`, `mulG` computes the product at `κG` per multiply-accumulate; off the grid it may
compute anything. `mul` stays the worst-case product on all operands. -/
structure GridMatMachine (R : Type) [Ring R] extends MatMachine R where
  G : R → Prop
  shape : ℕ → ℕ → ℕ → Prop
  κG : ℝ
  mulG : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  eval_mulG : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), shape a b c → (∀ i k, G (x.1 i k)) → (∀ k j, G (x.2 k j)) →
    eval (mulG a b c) x = x.1 * x.2
  time_mulG_le : ∀ a b c : ℕ, shape a b c → time (mulG a b c) ≤ κG * ((a : ℝ) * b * c)

namespace GridMatMachine

variable {R : Type} [Ring R] (Mc : GridMatMachine R)

/-- Both operands lie on the grid. -/
def OnGrid {n : ℕ} (x : Mat R n n × Mat R n n) : Prop := (∀ i k, Mc.G (x.1 i k)) ∧ ∀ k j, Mc.G (x.2 k j)

/-- **Route (c) on the grid**: when every promised instance's edits stay on the grid (`hgrid`) and the shape is one
the grid product takes, the edited protocol with `mulG` is relatively secure against the grid reference
`N = κG n³`, with `U = κG n³ + 2κadd n² + time gen + time prove`. -/
theorem relativelySecure_of_editFloor_grid (S : Service) {n : ℕ} (E : Mc.Edit S.Seed n)
    (prove : Mc.Alg ((S.Seed × (Mat R n n × Mat R n n) × S.Coins) × Mat R n n) (S.Com × S.Ans))
    (use : Mat R n n × Mat R n n → Prop) {κf δ η F : ℝ} (hshape : Mc.shape n n n)
    (hgrid : ∀ σ x, use x → Mc.OnGrid (E.edit σ x))
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat R n n × Mat R n n) × S.Coins) (C C' : Mat R n n),
      (Mc.eval prove ((σ, y), C)).1 = (Mc.eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat R n n × Mat R n n, use x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).1 ω.1
        (Mc.eval prove ((ω.1, x, ω.2), Mc.eval (Mc.mulG n n n) (E.edit ω.1 x))).2 = true)
    (hfloor : Mc.EditFloor S E (Mc.mulG n n n) prove use F κf) (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove)
    (hN : 0 < Mc.κG * (n : ℝ) ^ 3) :
    RelativelySecure (Mc.edited S E (Mc.mulG n n n) prove use)
      ((Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / F - 1) δ
      ((2 * Mc.κadd * (n : ℝ) ^ 2 + Mc.time E.gen + Mc.time prove) / (Mc.κG * (n : ℝ) ^ 3)) η 0
      (Mc.κG * (n : ℝ) ^ 3) :=
  Mc.relativelySecure_of_editFloor S E _ prove use
    (fun σ x hx => Mc.eval_mulG n n n _ hshape (hgrid σ x hx).1 (hgrid σ x hx).2)
    ((Mc.time_mulG_le n n n hshape).trans_eq (by ring)) hcom hacc hfloor hκδ hF hFU hN

end GridMatMachine

/-- **A matrix machine with grid and wide-by-grid products** (RS-MODEL §4's width refinement for the peel routes): on
the shapes `shape`, `mulW` computes the product when one operand is on the grid (a 32-bit by small product, byte-split),
at `κW` per multiply-accumulate (4 u on sm_120). -/
structure WideGridMatMachine (R : Type) [Ring R] extends GridMatMachine R where
  κW : ℝ
  mulW : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  eval_mulW : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), shape a b c →
    ((∀ i k, G (x.1 i k)) ∨ ∀ k j, G (x.2 k j)) → eval (mulW a b c) x = x.1 * x.2
  time_mulW_le : ∀ a b c : ℕ, shape a b c → time (mulW a b c) ≤ κW * ((a : ℝ) * b * c)

namespace WideGridMatMachine

variable {R : Type} [Ring R] (Mc : WideGridMatMachine R)

/-- Multiply the outputs of `f` and `g` with the grid product. -/
def mulG₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mulG a b c)

/-- Multiply the outputs of `f` and `g` with the wide-by-grid product. -/
def mulW₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mulW a b c)

theorem time_mulG₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hs : Mc.shape a b c) (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mulG₂ f g) ≤ u + v + Mc.κG * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mulG_le a b c hs])

theorem time_mulW₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hs : Mc.shape a b c) (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mulW₂ f g) ≤ u + v + Mc.κW * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mulW_le a b c hs])

/-- A matrix on the grid. -/
def On {a b : ℕ} (X : Mat R a b) : Prop := ∀ i k, Mc.G (X i k)

/-- The full-rank blocks on the grid: `A`, `Y = B + S`, `R` and `S`. -/
def GridFull {n : ℕ} (b : MatMachine.FullBlocks R n) : Prop :=
  Mc.On b.2.1 ∧ Mc.On b.2.2.1 ∧ Mc.On b.2.2.2.1 ∧ Mc.On b.2.2.2.2

/-- **The full-rank peel on the grid**: `R Y` and `A S` are grid products. -/
def peelFullG (n : ℕ) : Mc.Alg (MatMachine.FullBlocks R n) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair (Mc.mulG₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
      (Mc.mulG₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_peelFullG {n : ℕ} (hs : Mc.shape n n n) (b : MatMachine.FullBlocks R n) (hb : Mc.GridFull b) :
    Mc.eval (Mc.peelFullG n) b = (b.1, b.2.2.2.1 * b.2.2.1, b.2.1 * b.2.2.2.2) := by
  simp only [peelFullG, mulG₂, Machine.eval_lift2, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  rw [Mc.eval_mulG n n n (b.2.2.2.1, b.2.2.1) hs hb.2.2.1 hb.2.1,
    Mc.eval_mulG n n n (b.2.1, b.2.2.2.2) hs hb.1 hb.2.2.2]

theorem peelFullG_correct {n : ℕ} (hs : Mc.shape n n n) (x : Mat R n n × Mat R n n) (b : MatMachine.FullBlocks R n)
    (hb : MatMachine.ValidFull x b) (hg : Mc.GridFull b) : x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelFullG n) b) := by
  rw [Mc.eval_peelFullG hs b hg, ← Mc.eval_peelFull n b]
  exact Mc.peelFull_correct x b hb

/-- The full-rank peel on the grid costs two grid products: `2 κG n³`. -/
theorem time_peelFullG_le {n : ℕ} (hs : Mc.shape n n n) : Mc.time (Mc.peelFullG n) ≤ 2 * Mc.κG * (n : ℝ) ^ 3 := by
  apply le_trans
  · unfold peelFullG
    apply_rules [time_mulG₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le,
      Machine.time_snd_le]
  · linarith

/-- The low-rank blocks on the grid: `A`, `Y = B + W Zᵀ` and the factors `U`, `Vᵀ`, `W`, `Zᵀ`. -/
def GridLow {n r : ℕ} (b : MatMachine.LowBlocks R n r) : Prop :=
  Mc.On b.2.1 ∧ Mc.On b.2.2.1 ∧ Mc.On b.2.2.2.1 ∧ Mc.On b.2.2.2.2.1 ∧ Mc.On b.2.2.2.2.2.1 ∧ Mc.On b.2.2.2.2.2.2

/-- The shapes the low-rank peel's products take. -/
def LowShapes (n r : ℕ) : Prop := Mc.shape r n n ∧ Mc.shape n r n ∧ Mc.shape n n r

/-- **The low-rank peel on the grid** (RS-MODEL §5): `Vᵀ Y` and `A W` are grid products, and `U (Vᵀ Y)` and
`(A W) Zᵀ`, whose `r`-wide intermediates are off the grid, are wide-by-grid products. -/
def peelLowG (n r : ℕ) : Mc.Alg (MatMachine.LowBlocks R n r) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair
      (Mc.mulW₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.mulG₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
          (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
      (Mc.mulW₂
        (Mc.mulG₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))))
        (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))

theorem eval_peelLowG {n r : ℕ} (hs : Mc.LowShapes n r) (b : MatMachine.LowBlocks R n r) (hb : Mc.GridLow b) :
    Mc.eval (Mc.peelLowG n r) b =
      (b.1, b.2.2.2.1 * (b.2.2.2.2.1 * b.2.2.1), (b.2.1 * b.2.2.2.2.2.1) * b.2.2.2.2.2.2) := by
  obtain ⟨hA, hY, hU, hV, hW, hZ⟩ := hb
  simp only [peelLowG, mulG₂, mulW₂, Machine.eval_lift2, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]
  rw [Mc.eval_mulG r n n (b.2.2.2.2.1, b.2.2.1) hs.1 hV hY,
    Mc.eval_mulG n n r (b.2.1, b.2.2.2.2.2.1) hs.2.2 hA hW,
    Mc.eval_mulW n r n (b.2.2.2.1, b.2.2.2.2.1 * b.2.2.1) hs.2.1 (Or.inl hU),
    Mc.eval_mulW n r n (b.2.1 * b.2.2.2.2.2.1, b.2.2.2.2.2.2) hs.2.1 (Or.inr hZ)]

theorem peelLowG_correct {n r : ℕ} (hs : Mc.LowShapes n r) (x : Mat R n n × Mat R n n) (b : MatMachine.LowBlocks R n r)
    (hb : MatMachine.ValidLow x b) (hg : Mc.GridLow b) : x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelLowG n r) b) := by
  rw [Mc.eval_peelLowG hs b hg, ← Mc.eval_peelLow n r b]
  exact Mc.peelLow_correct x b hb

/-- The low-rank peel on the grid costs two grid and two wide-by-grid products through `r`:
`2 κG n² r + 2 κW n² r` (RS-MODEL §5's `10 n² r` u at `κG = 1`, `κW = 4`). -/
theorem time_peelLowG_le {n r : ℕ} (hs : Mc.LowShapes n r) :
    Mc.time (Mc.peelLowG n r) ≤ 2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r := by
  obtain ⟨h1, h2, h3⟩ := hs
  apply le_trans
  · unfold peelLowG
    apply_rules (maxDepth := 200) [time_mulG₂_le_of, time_mulW₂_le_of, Machine.time_pair_le_of,
      Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
  · linarith

variable {n : ℕ} {Inst : Type} {P : Protocol Mc.toMachine Inst (Mat R n n)} {emb : Inst → Mat R n n × Mat R n n}

/-- **Full-rank masks, products typed by grid**: from extraction of on-grid valid blocks and the floor above
`q = 2κG n³ + 2κadd n²`, `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_fullRank_grid (hmat : IsMatmulOn P emb) (hs : Mc.shape n n n) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidFull (emb x) b ∧ Mc.GridFull b) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κG * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelFullG n) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelFullG_correct hs (emb x) b hb.1 hb.2)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelFullG_le hs, Mc.time_corr_le n]) hT

/-- **Rank-`r` masks, products typed by grid**: from extraction of on-grid valid blocks and the floor above
`q = (2κG + 2κW) n² r + 2κadd n²`, `ε = εext + (1 + εext) q / (T − q)`. -/
theorem secure_lowRank_grid (hmat : IsMatmulOn P emb) (r : ℕ) (hs : Mc.LowShapes n r) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidLow (r := r) (emb x) b ∧ Mc.GridLow b) εext M η)
    (hfloor : Floor Mc.toMachine P.promise (fun x C => (emb x).1 * (emb x).2 = C) T η)
    (hT : 2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r +
      2 * Mc.κadd * (n : ℝ) ^ 2) / (T - (2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r +
        2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelLowG n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLowG_correct hs (emb x) b hb.1 hb.2)) hext hfloor
    (fun σ x C h => (hmat σ x C).1 h)
    (by linarith [Mc.time_peelLowG_le hs, Mc.time_corr_le n]) hT

/-- **Rank-`r` masks by route 2, products typed by grid** (RS-MODEL §5): with every accepted adversary costing at
least `F`, `ε = εext + ((2κG + 2κW) n² r + 2κadd n²) / F`, `(10 r + 2)/(c n)` on sm_120 for `F = c n³`. -/
theorem secure_lowRank_advTime_grid (hmat : IsMatmulOn P emb) (r : ℕ) (hs : Mc.LowShapes n r) {εext δ η F : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x b => MatMachine.ValidLow (r := r) (emb x) b ∧ Mc.GridLow b) εext M η)
    (hM : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → F ≤ Mc.time M) (hF : 0 < F) :
    Secure P (εext + (2 * Mc.κG * (n : ℝ) ^ 2 * r + 2 * Mc.κW * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) / F)
      δ η 0 :=
  secure_of_extraction_advTime P (Mc.peelLowG n r) (Mc.corr n)
    (fun σ x b hb => (hmat σ x _).2 (Mc.peelLowG_correct hs (emb x) b hb.1 hb.2)) hext hM hF
    (by linarith [Mc.time_peelLowG_le hs, Mc.time_corr_le n])

end WideGridMatMachine

end EditedInstances

section ModelB

/-! ## The Model B bridge (RS-MODEL: note:20261015T1814Z-draft-rs-machine-model, `gpu-model/rs-model.lean`)

Model B prices a straight-line gate program by the sum of its gates' prices (`timeB`, additive) and never reads its
wiring (`timeB_rewire`). A Model B algorithm `I → O` is a typed circuit (`Circ`) of gates and routing: its semantics
is gate evaluation and its time is `timeB` of its gate list. `machine` is the `Machine`; `matMachine` is the
`MatMachine` over ℤ given circuits for the products, sums and differences at their prices (`MatGates`), which a gate
family with one GEMM kind per shape provides (`gemmGates`). On sm_120 (`price120`) the worst case is `κ = 16` u per
product (s32 operands as 4 × 4 int8 limb products) and `κadd = 1` u per entry; on the int8 grid and the tile shapes the
product is `κG = 1` u (`gridMatMachine120`). -/

namespace ModelB

/-- RS-MODEL's gate kind: arity, outputs, the operand range on which it is well-formed (`ok`), and its exact semantics
there. -/
structure Kind where
  arity : ℕ
  outs : ℕ
  ok : (Fin arity → ℤ) → Bool
  sem : (Fin arity → ℤ) → Fin outs → ℤ

/-- RS-MODEL's gate: a kind, the wires it reads, its round. -/
structure Gate (K : Type) where
  kind : K
  args : List ℕ
  round : ℕ := 0

/-- RS-MODEL's straight-line program. -/
abbrev Prog (K : Type) : Type := List (Gate K)

variable {K : Type}

/-- RS-MODEL's `time_B`: the sum of the gates' prices (on sm_120 in u = 1/R* SMSP-cycles). -/
def timeB (price : K → ℕ) (P : Prog K) : ℕ := (P.map fun g => price g.kind).sum

theorem timeB_append (price : K → ℕ) (P Q : Prog K) : timeB price (P ++ Q) = timeB price P + timeB price Q := by
  simp [timeB]

/-- RS-MODEL's `timeB_rewire`: rewiring never changes time. -/
theorem timeB_rewire (price : K → ℕ) (P : Prog K) (f : Gate K → List ℕ) :
    timeB price (P.map fun g => { g with args := f g }) = timeB price P := by
  simp [timeB, Function.comp_def]

/-- The program of a gate list, with its wiring left out (Model B's time doesn't read it, `timeB_rewire`). -/
def progOf (l : List K) : Prog K := l.map fun k => { kind := k, args := [] }

theorem timeB_progOf_append (price : K → ℕ) (l m : List K) :
    timeB price (progOf (l ++ m)) = timeB price (progOf l) + timeB price (progOf m) := by
  rw [progOf, List.map_append, timeB_append]
  rfl

theorem timeB_progOf (price : K → ℕ) (l : List K) : timeB price (progOf l) = (l.map price).sum := by
  simp [timeB, progOf, Function.comp_def]

/-- An additive price on gate lists, nonnegative and additive under concatenation: RS-MODEL's `timeB` at a per-kind
price in ℕ (`Price.ofNat`), or the sum of a rational per-kind price (`Price.ofRat`, for `B_θ`). -/
structure Price (K : Type) where
  t : List K → ℝ
  nonneg : ∀ l, 0 ≤ t l
  append : ∀ l m, t (l ++ m) = t l + t m

theorem Price.t_nil (p : Price K) : p.t [] = 0 := by
  have h := p.append [] []
  rw [List.nil_append] at h
  linarith

/-- RS-MODEL's `timeB` at a per-kind price in ℕ. -/
def Price.ofNat (price : K → ℕ) : Price K where
  t l := (timeB price (progOf l) : ℝ)
  nonneg _ := Nat.cast_nonneg _
  append l m := by rw [timeB_progOf_append, Nat.cast_add]

/-- The sum of a nonnegative rational per-kind price. -/
def Price.ofRat (price : K → ℚ) (h : ∀ k, 0 ≤ price k) : Price K where
  t l := (((l.map price).sum : ℚ) : ℝ)
  nonneg l := Rat.cast_nonneg.mpr (List.sum_nonneg fun x hx => by
    obtain ⟨k, -, rfl⟩ := List.mem_map.mp hx
    exact h k)
  append l m := by rw [List.map_append, List.sum_append, Rat.cast_add]

variable (kd : K → Kind)

/-- A typed Model B circuit. `gate k` is one gate of kind `k` (off its range `ok` it outputs 0, which only adds
programs); the rest is routing, which moves wires and computes nothing: `reindex` (read wires in any order, with
fan-out), `curry` and `uncurry` (reshape a block), `join` (two bundles side by side), `ofBits` (bit wires, such as σ and
the coins, read as 0/1 integer wires, as RS-MODEL's `extra` inputs are), and the combinators. A protocol in Model B
has bit bundles (`α → Bool`) for σ and the coins, and integer bundles for its matrices and messages. -/
inductive Circ : Type → Type → Type 1 where
  | gate (k : K) : Circ (Fin (kd k).arity → ℤ) (Fin (kd k).outs → ℤ)
  | reindex {α β V : Type} (f : β → α) : Circ (α → V) (β → V)
  | curry {α β V : Type} : Circ (α × β → V) (α → β → V)
  | uncurry {α β V : Type} : Circ (α → β → V) (α × β → V)
  | join {α β V : Type} : Circ ((α → V) × (β → V)) (α ⊕ β → V)
  | ofBits {α : Type} : Circ (α → Bool) (α → ℤ)
  | seq {I J O : Type} : Circ I J → Circ J O → Circ I O
  | pair {I J O : Type} : Circ I J → Circ I O → Circ I (J × O)
  | fst {I J : Type} : Circ (I × J) I
  | snd {I J : Type} : Circ (I × J) J

variable {kd}

/-- What a circuit computes. -/
def Circ.eval : {I O : Type} → Circ kd I O → I → O
  | _, _, .gate k, w => fun o => if (kd k).ok w then (kd k).sem w o else 0
  | _, _, .reindex f, w => w ∘ f
  | _, _, .curry, w => fun a b => w (a, b)
  | _, _, .uncurry, w => fun p => w p.1 p.2
  | _, _, .join, w => Sum.elim w.1 w.2
  | _, _, .ofBits, w => fun a => if w a then 1 else 0
  | _, _, .seq f g, x => g.eval (f.eval x)
  | _, _, .pair f g, x => (f.eval x, g.eval x)
  | _, _, .fst, x => x.1
  | _, _, .snd, x => x.2

/-- Its gates, in order. -/
def Circ.gates : {I O : Type} → Circ kd I O → List K
  | _, _, .gate k => [k]
  | _, _, .reindex _ => []
  | _, _, .curry => []
  | _, _, .uncurry => []
  | _, _, .join => []
  | _, _, .ofBits => []
  | _, _, .seq f g => f.gates ++ g.gates
  | _, _, .pair f g => f.gates ++ g.gates
  | _, _, .fst => []
  | _, _, .snd => []

/-- Its Model B program (wiring left out). -/
def Circ.prog {I O : Type} (c : Circ kd I O) : Prog K := progOf c.gates

variable (kd) in
/-- A Model B algorithm `I → O`: what it computes and its gate list, as some circuit's. (A subtype, so that
algorithms are small types; the circuit itself lives in `Type 1`.) -/
def Alg (I O : Type) : Type := {p : (I → O) × List K // ∃ c : Circ kd I O, c.eval = p.1 ∧ c.gates = p.2}

/-- A circuit's algorithm. -/
def Circ.alg {I O : Type} (c : Circ kd I O) : Alg kd I O := ⟨(c.eval, c.gates), c, rfl, rfl⟩

variable (kd) in
/-- **Model B as a `Machine`**, at an additive price `p` (`timeB` for RS-MODEL's B, `Price.ofNat`): time is `p` of the
gate list and a run costs its time (Model B is oblivious); composition concatenates gate lists, so time is additive
(RS-DEF's T1, `timeB_append`), and routing is free. -/
def machine (p : Price K) : Machine where
  Alg := Alg kd
  eval f := f.1.1
  time f := p.t f.1.2
  cost f _ := p.t f.1.2
  size f := p.t f.1.2
  seq f g := ⟨(g.1.1 ∘ f.1.1, f.1.2 ++ g.1.2), by
    obtain ⟨c, hc, hc'⟩ := f.2
    obtain ⟨d, hd, hd'⟩ := g.2
    exact ⟨.seq c d, by rw [← hc, ← hd]; rfl, by rw [← hc', ← hd']; rfl⟩⟩
  pair f g := ⟨(fun x => (f.1.1 x, g.1.1 x), f.1.2 ++ g.1.2), by
    obtain ⟨c, hc, hc'⟩ := f.2
    obtain ⟨d, hd, hd'⟩ := g.2
    exact ⟨.pair c d, by rw [← hc, ← hd]; rfl, by rw [← hc', ← hd']; rfl⟩⟩
  fst := Circ.fst.alg
  snd := Circ.snd.alg
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg f := p.nonneg f.1.2
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg f := p.nonneg f.1.2
  time_seq_le f g := (p.append f.1.2 g.1.2).le
  le_time_seq f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg g.1.2]
  time_pair_le f g := (p.append f.1.2 g.1.2).le
  le_time_pair_left f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg g.1.2]
  le_time_pair_right f g := by
    show _ ≤ p.t (f.1.2 ++ g.1.2)
    rw [p.append]
    linarith [p.nonneg f.1.2]
  time_fst := p.t_nil
  time_snd := p.t_nil

/-- A circuit's time in `machine` is the price of its gates. -/
theorem time_alg (p : Price K) {I O : Type} (c : Circ kd I O) : (machine kd p).time c.alg = p.t c.gates := rfl

/-- At `Price.ofNat`, it is RS-MODEL's `time_B` of its program. -/
theorem time_alg_ofNat (price : K → ℕ) {I O : Type} (c : Circ kd I O) :
    (machine kd (.ofNat price)).time c.alg = (timeB price c.prog : ℝ) := rfl

theorem eval_alg (p : Price K) {I O : Type} (c : Circ kd I O) : (machine kd p).eval c.alg = c.eval := rfl

variable (kd) in
/-- Circuits for the matrix operations over ℤ at their prices: an `a × b` by `b × c` product at `κ` per
multiply-accumulate, an entrywise sum or difference at `κadd` per entry. -/
structure MatGates (p : Price K) (κ κadd : ℝ) where
  mul : ∀ a b c : ℕ, Circ kd (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c)
  add : ∀ a b : ℕ, Circ kd (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b)
  sub : ∀ a b : ℕ, Circ kd (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b)
  eval_mul : ∀ (a b c : ℕ) (x : Mat ℤ a b × Mat ℤ b c), (mul a b c).eval x = x.1 * x.2
  eval_add : ∀ (a b : ℕ) (x : Mat ℤ a b × Mat ℤ a b), (add a b).eval x = x.1 + x.2
  eval_sub : ∀ (a b : ℕ) (x : Mat ℤ a b × Mat ℤ a b), (sub a b).eval x = x.1 - x.2
  time_mul_le : ∀ a b c : ℕ, p.t (mul a b c).gates ≤ κ * ((a : ℝ) * b * c)
  time_add_le : ∀ a b : ℕ, p.t (add a b).gates ≤ κadd * ((a : ℝ) * b)
  time_sub_le : ∀ a b : ℕ, p.t (sub a b).gates ≤ κadd * ((a : ℝ) * b)

/-- **Model B as a `MatMachine`** over ℤ, from circuits for the matrix operations. -/
def matMachine {p : Price K} {κ κadd : ℝ} (G : MatGates kd p κ κadd) : MatMachine ℤ where
  toMachine := machine kd p
  κ := κ
  κadd := κadd
  mul a b c := (G.mul a b c).alg
  add a b := (G.add a b).alg
  sub a b := (G.sub a b).alg
  eval_mul := G.eval_mul
  eval_add := G.eval_add
  eval_sub := G.eval_sub
  time_mul_le := G.time_mul_le
  time_add_le := G.time_add_le
  time_sub_le := G.time_sub_le

variable (kd) in
/-- Circuits for a grid product: exact on the shapes `shape` with operand entries in the grid `Gr`, at `κG` per
multiply-accumulate. Off the grid or the shapes they may compute anything. -/
structure GridGates (p : Price K) (Gr : ℤ → Prop) (shape : ℕ → ℕ → ℕ → Prop) (κG : ℝ) where
  mul : ∀ a b c : ℕ, Circ kd (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c)
  eval_mul : ∀ (a b c : ℕ) (x : Mat ℤ a b × Mat ℤ b c), shape a b c → (∀ i k, Gr (x.1 i k)) →
    (∀ k j, Gr (x.2 k j)) → (mul a b c).eval x = x.1 * x.2
  time_mul_le : ∀ a b c : ℕ, shape a b c → p.t (mul a b c).gates ≤ κG * ((a : ℝ) * b * c)

/-- **Model B as a `GridMatMachine`** over ℤ: `matMachine G` with the grid product from `I`. -/
def gridMatMachine {p : Price K} {κ κadd κG : ℝ} {Gr : ℤ → Prop} {shape : ℕ → ℕ → ℕ → Prop}
    (G : MatGates kd p κ κadd) (I : GridGates kd p Gr shape κG) : GridMatMachine ℤ where
  toMatMachine := matMachine G
  G := Gr
  shape := shape
  κG := κG
  mulG a b c := (I.mul a b c).alg
  eval_mulG := I.eval_mul
  time_mulG_le := I.time_mul_le

/-! ### The circuits exist: one GEMM kind per shape -/

/-- A pair of matrices on the wires `Fin (a b + c d)`, row-major (routing only). -/
def flatPair (a b c d : ℕ) : Circ kd (Mat ℤ a b × Mat ℤ c d) (Fin (a * b + c * d) → ℤ) :=
  .seq (.pair (.seq .fst (.seq (Circ.uncurry (α := Fin a) (β := Fin b) (V := ℤ)) (.reindex finProdFinEquiv.symm)))
      (.seq .snd (.seq (Circ.uncurry (α := Fin c) (β := Fin d) (V := ℤ)) (.reindex finProdFinEquiv.symm))))
    (.seq .join (.reindex finSumFinEquiv.symm))

/-- The wires `Fin (a c)` as an `a × c` matrix (routing only). -/
def unflat (a c : ℕ) : Circ kd (Fin (a * c) → ℤ) (Mat ℤ a c) :=
  .seq (.reindex finProdFinEquiv) (Circ.curry (α := Fin a) (β := Fin c) (V := ℤ))

/-- The first matrix read off `flatPair`'s wires. -/
def leftOf (a b c d : ℕ) (w : Fin (a * b + c * d) → ℤ) : Mat ℤ a b :=
  fun i k => w (finSumFinEquiv (.inl (finProdFinEquiv (i, k))))

/-- The second matrix read off `flatPair`'s wires. -/
def rightOf (a b c d : ℕ) (w : Fin (a * b + c * d) → ℤ) : Mat ℤ c d :=
  fun k j => w (finSumFinEquiv (.inr (finProdFinEquiv (k, j))))

theorem leftOf_flatPair (a b c d : ℕ) (x : Mat ℤ a b × Mat ℤ c d) :
    leftOf a b c d ((flatPair (kd := kd) a b c d).eval x) = x.1 := by
  funext i k
  simp [leftOf, flatPair, Circ.eval]

theorem rightOf_flatPair (a b c d : ℕ) (x : Mat ℤ a b × Mat ℤ c d) :
    rightOf a b c d ((flatPair (kd := kd) a b c d).eval x) = x.2 := by
  funext i k
  simp [rightOf, flatPair, Circ.eval]

/-- The GEMM gate family: one kind per shape for the product, the sum and the difference. -/
inductive GemmK
  | mm (a b c : ℕ)
  | add (a b : ℕ)
  | sub (a b : ℕ)

/-- Their exact semantics on `flatPair`'s wires, every operand in range. -/
def gemmKind : GemmK → Kind
  | .mm a b c => ⟨a * b + b * c, a * c, fun _ => true, fun w o =>
      (leftOf a b b c w * rightOf a b b c w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩
  | .add a b => ⟨a * b + a * b, a * b, fun _ => true, fun w o =>
      (leftOf a b a b w + rightOf a b a b w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩
  | .sub a b => ⟨a * b + a * b, a * b, fun _ => true, fun w o =>
      (leftOf a b a b w - rightOf a b a b w) (finProdFinEquiv.symm o).1 (finProdFinEquiv.symm o).2⟩

/-- Their prices: `κ₀` per multiply-accumulate, `κa` per entry. -/
def gemmPrice (κ₀ κa : ℕ) : GemmK → ℕ
  | .mm a b c => κ₀ * (a * b * c)
  | .add a b => κa * (a * b)
  | .sub a b => κa * (a * b)

/-- The product circuit: `flatPair`, one `mm` gate, `unflat`. -/
def gemmMul (a b c : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ b c) (Mat ℤ a c) :=
  .seq (flatPair a b b c) (.seq (Circ.gate (kd := gemmKind) (.mm a b c)) (unflat a c))

/-- The sum circuit. -/
def gemmAdd (a b : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b) :=
  .seq (flatPair a b a b) (.seq (Circ.gate (kd := gemmKind) (.add a b)) (unflat a b))

/-- The difference circuit. -/
def gemmSub (a b : ℕ) : Circ gemmKind (Mat ℤ a b × Mat ℤ a b) (Mat ℤ a b) :=
  .seq (flatPair a b a b) (.seq (Circ.gate (kd := gemmKind) (.sub a b)) (unflat a b))

/-- **The circuits exist**: the GEMM gate family realizes `MatGates` with `κ = κ₀` and `κadd = κa`. -/
def gemmGates (κ₀ κa : ℕ) : MatGates gemmKind (.ofNat (gemmPrice κ₀ κa)) κ₀ κa where
  mul := gemmMul
  add := gemmAdd
  sub := gemmSub
  eval_mul a b c x := by
    funext i j
    simp only [gemmMul, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  eval_add a b x := by
    funext i j
    simp only [gemmAdd, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  eval_sub a b x := by
    funext i j
    simp only [gemmSub, Circ.eval, unflat, gemmKind, Function.comp_apply, Equiv.symm_apply_apply,
      leftOf_flatPair, rightOf_flatPair, ite_true]
  time_mul_le a b c := by
    simp [gemmMul, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]
  time_add_le a b := by
    simp [gemmAdd, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]
  time_sub_le a b := by
    simp [gemmSub, Price.ofNat, Circ.gates, flatPair, unflat, progOf, timeB, gemmPrice]

/-! ### sm_120 (RS-MODEL §4) -/

/-- RS-MODEL's sm_120 gate kinds, per warp instruction. -/
inductive K120
  | mma | mmaTf32 | imad | dp4a | hfma2 | ffma | alu | ldst

/-- RS-MODEL's `price120`, the LP dual of Model A's census rows, in u per warp instruction. -/
def price120 : K120 → ℕ
  | .mma => 4096 | .mmaTf32 => 4096 | .imad => 128 | .dp4a => 128 | .hfma2 => 64 | .ffma => 32 | .alu => 32
  | .ldst => 32

/-- **Model B on sm_120 as a `MatMachine`**, with one worst-case constant: `κ = 16` u per product (an s32 product as
4 × 4 int8 limb products on the tensor pipe at 1 u each) and `κadd = 1` u per entry (one ALU lane). It reads the
kinds' silicon semantics `kd` and the circuits `G` over them (named hypothesis `Gemm120`). -/
def matMachine120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : MatMachine ℤ := matMachine G

theorem matMachine120_κ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : (matMachine120 kd G).κ = 16 := rfl

theorem matMachine120_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) : (matMachine120 kd G).κadd = 1 :=
  rfl

/-- **The sm_120 circuits** (a named hypothesis): over the kinds' silicon semantics `kd`, exact circuits for the
matrix operations over ℤ at 16 u per product and 1 u per entry. RS-MODEL prices sm_120's kinds and gives them no
semantics, so this is where `mma`'s exact semantics and the limb decomposition enter. -/
def Gemm120 (kd : K120 → Kind) : Prop := Nonempty (MatGates kd (.ofNat price120) 16 1)

/-- The int8 grid, `G_op`. -/
def Int8 (z : ℤ) : Prop := -128 ≤ z ∧ z ≤ 127

/-- The `m16n8k32` tile shapes, with the s32 accumulator exact (`b ≤ 2^17`, `int8_product_exact`). -/
def Tile120 (a b c : ℕ) : Prop := 16 ∣ a ∧ 32 ∣ b ∧ 8 ∣ c ∧ b ≤ 2 ^ 17

theorem tile120_square {n : ℕ} (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17) : Tile120 n n n :=
  ⟨(by norm_num : (16 : ℕ) ∣ 32).trans h32, h32, (by norm_num : (8 : ℕ) ∣ 32).trans h32, h17⟩

theorem int8_of_abs_le {a e s : ℤ} (ha : |a| ≤ 127 - s) (he : |e| ≤ s) : Int8 (a + e) := by
  have h := abs_le.mp ((abs_add_le a e).trans (by linarith : |a| + |e| ≤ 127))
  exact ⟨by linarith [h.1], h.2⟩

/-- **The edited operands are int8** (RS-RT's V6): an instance in `G_use = [−(127 − s), 127 − s]` edited by noise
bounded by `s` stays in int8. -/
theorem edit_int8 {Mc : MatMachine ℤ} {Seed : Type} {n : ℕ} {E : Mc.Edit Seed n} {s : ℤ} (hE : E.Bounded s)
    (σ : Seed) {x : Mat ℤ n n × Mat ℤ n n} (hx : InGrid (127 - s) x) :
    (∀ i k, Int8 ((E.edit σ x).1 i k)) ∧ ∀ k j, Int8 ((E.edit σ x).2 k j) :=
  ⟨fun i k => int8_of_abs_le (hx i k).1 (hE σ i k).1, fun k j => int8_of_abs_le (hx k j).2 (hE σ k j).2⟩

/-- **An accepted transcript's opened edited entries are int8**: the range check puts the committed instance's opened
entries in `G_use`, so under every σ' their edits are int8 operands, the grid the edit floor prices at 1 u. -/
theorem accepted_edit_int8 {Mc : MatMachine ℤ} {n : ℕ} {s : ℤ} (S : GridService n (127 - s))
    {E : Mc.Edit S.Seed n} (hE : E.Bounded s) {c : S.Com} {σ : S.Seed} {a : S.Ans} (h : S.verify c σ a = true)
    (σ' : S.Seed) {i j : Fin n} (hij : S.opened σ i j) :
    Int8 ((E.edit σ' (S.committed c)).1 i j) ∧ Int8 ((E.edit σ' (S.committed c)).2 i j) :=
  ⟨int8_of_abs_le (S.range_check c σ a h i j hij).1 (hE σ' i j).1,
    int8_of_abs_le (S.range_check c σ a h i j hij).2 (hE σ' i j).2⟩

/-- With every entry opened, an accepted transcript's whole edited instance is int8. -/
theorem accepted_edit_int8_all {Mc : MatMachine ℤ} {n : ℕ} {s : ℤ} (S : GridService n (127 - s))
    (hall : ∀ σ i j, S.opened σ i j) {E : Mc.Edit S.Seed n} (hE : E.Bounded s) {c : S.Com} {σ : S.Seed}
    {a : S.Ans} (h : S.verify c σ a = true) (σ' : S.Seed) :
    (∀ i k, Int8 ((E.edit σ' (S.committed c)).1 i k)) ∧ ∀ k j, Int8 ((E.edit σ' (S.committed c)).2 k j) :=
  edit_int8 hE σ' (S.inGrid_of_accepts hall h)

/-- Circuits for the int8 product on sm_120's tile shapes: exact on int8 operands at 1 u per multiply-accumulate
(RS-MODEL's `gemm120`: `a b c` u for the `m16n8k32` tile GEMM). -/
abbrev Int8Gates120 (kd : K120 → Kind) : Type 1 := GridGates kd (.ofNat price120) Int8 Tile120 1

/-- **The int8 circuits** (a named hypothesis, the width-typed refinement): over the kinds' silicon semantics `kd`,
exact int8 × int8 products on tile-multiple shapes at 1 u per multiply-accumulate, s32 exact. -/
def Int8Gemm120 (kd : K120 → Kind) : Prop := Nonempty (Int8Gates120 kd)

/-- **Model B on sm_120 with the operand grid typed**: the worst-case `matMachine120` (16 u per product on all
operands, 1 u per entry for sums) with the int8 product at `κG = 1` u on `G_op = int8` and the tile shapes. -/
def gridMatMachine120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    GridMatMachine ℤ :=
  gridMatMachine G I

theorem gridMatMachine120_κG (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120 kd G I).κG = 1 := rfl

theorem gridMatMachine120_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120 kd G I).κadd = 1 := rfl

/-- **Route (c) on sm_120, on the int8 grid** (`κ = 1` u on `G_op = int8`): on a range-checking service with
`G_use = [−(127 − s), 127 − s]` (the promise), noise bounded by `s` and `32 ∣ n ≤ 2^17`, the edited protocol served by
the int8 tile GEMM is relatively secure against `N = n³` u, with `ε = U / F − 1`, `U = n³ + 2 n² + time gen +
time prove`, and `γ = (2 n² + time gen + time prove) / n³`. The edits stay int8 by `edit_int8`. -/
theorem relativelySecure_edited120 (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120 kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120 kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120 kd G I).eval prove ((σ, y), C)).1 = ((gridMatMachine120 kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120 kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120 kd G I).eval ((gridMatMachine120 kd G I).mulG n n n) (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120 kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120 kd G I).eval ((gridMatMachine120 kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (hfloor : (gridMatMachine120 kd G I).EditFloor S.toService E ((gridMatMachine120 kd G I).mulG n n n) prove
      (InGrid (127 - s)) F κf)
    (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen +
      (gridMatMachine120 kd G I).time prove) :
    RelativelySecure ((gridMatMachine120 kd G I).edited S.toService E ((gridMatMachine120 kd G I).mulG n n n) prove
      (InGrid (127 - s)))
      (((n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen +
        (gridMatMachine120 kd G I).time prove) / F - 1) δ
      ((2 * (n : ℝ) ^ 2 + (gridMatMachine120 kd G I).time E.gen + (gridMatMachine120 kd G I).time prove) /
        (n : ℝ) ^ 3) η 0 ((n : ℝ) ^ 3) := by
  have h := (gridMatMachine120 kd G I).relativelySecure_of_editFloor_grid S.toService E prove (InGrid (127 - s))
    (tile120_square h32 h17) (fun σ _ hx => edit_int8 hE σ hx) hcom hacc hfloor hκδ hF
    (by rw [gridMatMachine120_κG, gridMatMachine120_κadd, one_mul, mul_one]; exact hFU)
    (by rw [gridMatMachine120_κG, one_mul]; exact pow_pos (Nat.cast_pos.mpr hn) 3)
  rwa [gridMatMachine120_κG, gridMatMachine120_κadd, one_mul, mul_one] at h

/-- **Route (c) on sm_120, unrestricted operands** (the worst case, `κ = 16` u): reference `N = 16 n³` u, `ε = U / F − 1`
with `U = 16 n³ + 2 n² + time gen + time prove`, and `γ = (2 n² + time gen + time prove) / (16 n³)`. -/
theorem relativelySecure_edited120_wide (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (S : Service) {n : ℕ}
    (hn : 0 < n) (E : (matMachine120 kd G).Edit S.Seed n)
    (prove : (matMachine120 kd G).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n) (S.Com × S.Ans))
    {κf δ η F : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((matMachine120 kd G).eval prove ((σ, y), C)).1 = ((matMachine120 kd G).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((matMachine120 kd G).eval prove ((ω.1, x, ω.2), E.product ω.1 x)).1 ω.1
        ((matMachine120 kd G).eval prove ((ω.1, x, ω.2), E.product ω.1 x)).2 = true)
    (hfloor : (matMachine120 kd G).EditFloor S E ((matMachine120 kd G).mul n n n) prove (fun _ => True) F κf)
    (hκδ : κf < δ) (hF : 0 < F)
    (hFU : F ≤ 16 * (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen +
      (matMachine120 kd G).time prove) :
    RelativelySecure ((matMachine120 kd G).edited S E ((matMachine120 kd G).mul n n n) prove fun _ => True)
      ((16 * (n : ℝ) ^ 3 + 2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen + (matMachine120 kd G).time prove) / F -
        1) δ
      ((2 * (n : ℝ) ^ 2 + (matMachine120 kd G).time E.gen + (matMachine120 kd G).time prove) / (16 * (n : ℝ) ^ 3)) η 0
      (16 * (n : ℝ) ^ 3) := by
  have h := (matMachine120 kd G).relativelySecure_of_editFloor_wide S E prove hcom hacc hfloor hκδ hF
    (by rw [matMachine120_κ, matMachine120_κadd, mul_one]; exact hFU)
    (by rw [matMachine120_κ]; exact mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3))
  rwa [matMachine120_κ, matMachine120_κadd, mul_one] at h

/-! ### B_θ (RS-MODEL §6.2): Model B with shared memory's row mixed in

`B_θ = (1 − θ)·B + θ·R6`, with R6 the shared-memory data path (bytes / 128 per SM per cycle), at θ = 1/5. An instruction
carries its kind and the 4-byte words it moves through shared memory or L1; a gate reading its wires freely moves none,
so a B circuit is a B_θ circuit at `4/5` of its B time (`Circ.liftθ`, `MatGates.liftθ`), and every B_θ adversary has a
B shadow at most `5/4` of its B_θ time (`Circ.projθ`), which carries B's floor to `4/5` of it (`serviceFloor_θ`). -/

/-- An sm_120 warp instruction for `B_θ`: its kind and the 4-byte words it moves through shared memory or L1 (a gather
or a scatter is an `ldst`). -/
structure Instr120 where
  kind : K120
  words : ℕ

/-- One 4-byte word on the shared-memory row R6 alone: `W = R*/8 = 37.51` u (`R* = 300.08`, `gpu-model/rs-gather.py`). -/
def wordPrice : ℚ := 3751 / 100

/-- **RS-MODEL §6.2's `B_θ` at θ = 1/5**, `(1 − θ)·price120 + θ·R6`: MMA 3276.8 u, IMAD and DP4A 102.4, HFMA2 51.2,
every lane or ld/st instruction 25.6, plus `θ W = 7.502` u per word through shared memory or L1. -/
def price120θ (i : Instr120) : ℚ := 4 / 5 * price120 i.kind + 1 / 5 * wordPrice * i.words

theorem price120θ_nonneg (i : Instr120) : 0 ≤ price120θ i := by
  unfold price120θ wordPrice
  positivity

/-- `B_θ` as an additive price. -/
def priceθ : Price Instr120 := .ofRat price120θ price120θ_nonneg

/-- An instruction's semantics is its kind's (the words it moves are routing). -/
abbrev kdθ (kd : K120 → Kind) : Instr120 → Kind := fun i => kd i.kind

/-- B_θ prices a list at least `4/5` of B's price of its kinds. -/
theorem timeB_le_priceθ (l : List Instr120) :
    4 / 5 * (timeB price120 (progOf (l.map Instr120.kind)) : ℝ) ≤ priceθ.t l := by
  show _ ≤ (((l.map price120θ).sum : ℚ) : ℝ)
  rw [timeB_progOf, List.map_map, Nat.cast_list_sum, Rat.cast_list_sum, List.map_map, List.map_map,
    ← List.sum_map_mul_left]
  refine List.sum_le_sum fun i _ => ?_
  simp only [Function.comp_apply, price120θ, wordPrice]
  push_cast
  have : (0 : ℝ) ≤ 1 / 5 * (3751 / 100) * i.words := by positivity
  linarith

/-- A B gate list read in B_θ, each gate moving no words, costs `4/5` of its B time. -/
theorem priceθ_map_lift (l : List K120) :
    priceθ.t (l.map fun k => (⟨k, 0⟩ : Instr120)) = 4 / 5 * (timeB price120 (progOf l) : ℝ) := by
  show (((l.map fun k => (⟨k, 0⟩ : Instr120)).map price120θ).sum : ℝ) = _
  have h : (l.map fun k => (⟨k, 0⟩ : Instr120)).map price120θ = l.map fun k => (4 / 5 : ℚ) * price120 k := by
    simp [price120θ, Function.comp_def]
  rw [h, List.sum_map_mul_left, timeB_progOf]
  push_cast
  simp [Function.comp_def]

/-- A B circuit as a B_θ circuit: its gates read their wires freely, so they move no words. -/
def Circ.liftθ {kd : K120 → Kind} : {I O : Type} → Circ kd I O → Circ (kdθ kd) I O
  | _, _, .gate k => Circ.gate (kd := kdθ kd) ⟨k, 0⟩
  | _, _, .reindex f => .reindex f
  | _, _, .curry => .curry
  | _, _, .uncurry => .uncurry
  | _, _, .join => .join
  | _, _, .ofBits => .ofBits
  | _, _, .seq f g => .seq f.liftθ g.liftθ
  | _, _, .pair f g => .pair f.liftθ g.liftθ
  | _, _, .fst => .fst
  | _, _, .snd => .snd

/-- A B_θ circuit's B shadow: each instruction as its kind. -/
def Circ.projθ {kd : K120 → Kind} : {I O : Type} → Circ (kdθ kd) I O → Circ kd I O
  | _, _, .gate i => Circ.gate (kd := kd) i.kind
  | _, _, .reindex f => .reindex f
  | _, _, .curry => .curry
  | _, _, .uncurry => .uncurry
  | _, _, .join => .join
  | _, _, .ofBits => .ofBits
  | _, _, .seq f g => .seq f.projθ g.projθ
  | _, _, .pair f g => .pair f.projθ g.projθ
  | _, _, .fst => .fst
  | _, _, .snd => .snd

theorem Circ.eval_liftθ {kd : K120 → Kind} {I O : Type} (c : Circ kd I O) : c.liftθ.eval = c.eval := by
  induction c with
  | seq f g ihf ihg => funext x; simp only [Circ.liftθ, Circ.eval, ihf, ihg]
  | pair f g ihf ihg => funext x; simp only [Circ.liftθ, Circ.eval, ihf, ihg]
  | _ => rfl

theorem Circ.gates_liftθ {kd : K120 → Kind} {I O : Type} (c : Circ kd I O) :
    c.liftθ.gates = c.gates.map fun k => (⟨k, 0⟩ : Instr120) := by
  induction c with
  | seq f g ihf ihg => simp only [Circ.liftθ, Circ.gates, ihf, ihg, List.map_append]
  | pair f g ihf ihg => simp only [Circ.liftθ, Circ.gates, ihf, ihg, List.map_append]
  | _ => rfl

theorem Circ.eval_projθ {kd : K120 → Kind} {I O : Type} (c : Circ (kdθ kd) I O) : c.projθ.eval = c.eval := by
  induction c with
  | seq f g ihf ihg => funext x; simp only [Circ.projθ, Circ.eval, ihf, ihg]
  | pair f g ihf ihg => funext x; simp only [Circ.projθ, Circ.eval, ihf, ihg]
  | _ => rfl

theorem Circ.gates_projθ {kd : K120 → Kind} {I O : Type} (c : Circ (kdθ kd) I O) :
    c.projθ.gates = c.gates.map Instr120.kind := by
  induction c with
  | seq f g ihf ihg => simp only [Circ.projθ, Circ.gates, ihf, ihg, List.map_append]
  | pair f g ihf ihg => simp only [Circ.projθ, Circ.gates, ihf, ihg, List.map_append]
  | _ => rfl

/-- A B_θ algorithm's B shadow: the same function, each instruction as its kind. -/
def Alg.projθ {kd : K120 → Kind} {I O : Type} (f : Alg (kdθ kd) I O) : Alg kd I O :=
  ⟨(f.1.1, f.1.2.map Instr120.kind), by
    obtain ⟨c, hc, hc'⟩ := f.2
    exact ⟨c.projθ, by rw [Circ.eval_projθ, hc], by rw [Circ.gates_projθ, hc']⟩⟩

/-- **B's floor carries to B_θ at `1 − θ = 4/5` of it**: an adversary's B shadow computes the same messages, and B_θ
prices it at least `4/5` of its B time (`timeB_le_priceθ`). -/
theorem serviceFloor_θ (kd : K120 → Kind) (S : Service) {F κ : ℝ}
    (h : ServiceFloor (machine kd (.ofNat price120)) S F κ) : ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * F) κ := by
  intro Ω _ M hc
  refine (prob_mono fun ω hω => ⟨hω.1, ?_⟩).trans (h Ω M.projθ hc)
  have h1 := timeB_le_priceθ M.1.2
  have h2 : priceθ.t M.1.2 < 4 / 5 * F := (max_lt_iff.mp hω.2).1
  have h3 : (timeB price120 (progOf (M.1.2.map Instr120.kind)) : ℝ) < F := by linarith
  exact max_lt h3 h3

/-- B's matrix circuits in B_θ, at `4/5` of their prices. -/
noncomputable def MatGates.liftθ {kd : K120 → Kind} {κ κadd : ℝ} (G : MatGates kd (.ofNat price120) κ κadd) :
    MatGates (kdθ kd) priceθ (4 / 5 * κ) (4 / 5 * κadd) where
  mul a b c := (G.mul a b c).liftθ
  add a b := (G.add a b).liftθ
  sub a b := (G.sub a b).liftθ
  eval_mul a b c x := by rw [Circ.eval_liftθ]; exact G.eval_mul a b c x
  eval_add a b x := by rw [Circ.eval_liftθ]; exact G.eval_add a b x
  eval_sub a b x := by rw [Circ.eval_liftθ]; exact G.eval_sub a b x
  time_mul_le a b c := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_mul_le a b c) (by norm_num)
  time_add_le a b := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_add_le a b) (by norm_num)
  time_sub_le a b := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (G.time_sub_le a b) (by norm_num)

/-- B's grid circuits in B_θ, at `4/5` of their price. -/
noncomputable def GridGates.liftθ {kd : K120 → Kind} {Gr : ℤ → Prop} {shape : ℕ → ℕ → ℕ → Prop} {κG : ℝ}
    (I : GridGates kd (.ofNat price120) Gr shape κG) : GridGates (kdθ kd) priceθ Gr shape (4 / 5 * κG) where
  mul a b c := (I.mul a b c).liftθ
  eval_mul a b c x hs hx hy := by rw [Circ.eval_liftθ]; exact I.eval_mul a b c x hs hx hy
  time_mul_le a b c hs := by
    rw [Circ.gates_liftθ, priceθ_map_lift, mul_assoc]
    exact mul_le_mul_of_nonneg_left (I.time_mul_le a b c hs) (by norm_num)

/-- **Model B_θ on sm_120 with the operand grid typed**: B's circuits read in B_θ, the int8 tile GEMM at
`κG = 4/5` u per multiply-accumulate (the in-model `0.8 n³`, gates reading their wires freely), sums at `κadd = 4/5`
u per entry, and the worst case at `κ = 64/5` u. -/
noncomputable def gridMatMachine120θ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    GridMatMachine ℤ :=
  gridMatMachine G.liftθ I.liftθ

theorem gridMatMachine120θ_κG (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120θ kd G I).κG = 4 / 5 := mul_one _

theorem gridMatMachine120θ_κadd (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd) :
    (gridMatMachine120θ kd G I).κadd = 4 / 5 := mul_one _

/-- **Route (c) on sm_120 in B_θ, on the int8 grid, from any B floor**: B's service floor `F_B ≤ n³` u (a named
hypothesis, `ServiceFloor`) carries to `0.8 F_B` u in B_θ (`serviceFloor_θ`), and on a range-checking service with
`G_use = [−(127 − s), 127 − s]` and noise bounded by `s` (so the edits stay int8, `edit_int8`) the edited protocol served
by the int8 tile GEMM is relatively secure against `N = 0.8 n³` u with `ε = U / (0.8 F_B) − 1`,
`U = 0.8 n³ + 1.6 n² + time gen + time prove` (no commitment cost credited), and
`γ = (1.6 n² + time gen + time prove) / (0.8 n³)`. -/
theorem relativelySecure_edited120θ_of_floor (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    {FB : ℝ} (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService FB κf) (hκδ : κf < δ) (hFB : 0 < FB)
    (hFBn : FB ≤ (n : ℝ) ^ 3) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * FB) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hN : (0 : ℝ) < 4 / 5 * (n : ℝ) ^ 3 := mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3)
  have hg := (gridMatMachine120θ kd G I).time_nonneg E.gen
  have hp := (gridMatMachine120θ kd G I).time_nonneg prove
  have h := (gridMatMachine120θ kd G I).relativelySecure_of_editFloor_grid S.toService E prove (InGrid (127 - s))
    (tile120_square h32 h17) (fun σ _ hx => edit_int8 hE σ hx) hcom hacc (serviceFloor_θ kd S.toService hfloorB) hκδ
    (by positivity) (by rw [gridMatMachine120θ_κG, gridMatMachine120θ_κadd]; nlinarith [sq_nonneg (n : ℝ)])
    (by rw [gridMatMachine120θ_κG]; exact hN)
  rwa [gridMatMachine120θ_κG, gridMatMachine120θ_κadd] at h

/-- **Route (c) in B_θ at Conjecture A's floor** (`c = 1`, `F_B = n³` u, open): `ε = γ`. -/
theorem relativelySecure_edited120θ (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService ((n : ℝ) ^ 3) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hN : (0 : ℝ) < 4 / 5 * (n : ℝ) ^ 3 := mul_pos (by norm_num) (pow_pos (Nat.cast_pos.mpr hn) 3)
  have h := relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfloorB hκδ
    (pow_pos (Nat.cast_pos.mpr hn) 3) le_rfl
  rw [div_sub_one hN.ne'] at h
  convert h using 2
  ring

/-- **Route (c) in B_θ at a floor of Theorem E's form** (RS-D1E): B's floor `F_B = (c / m_max) (φ n³ − sl)` u for
rationals `0 < c ≤ m_max` and `φ ≤ 1` and a slack `0 ≤ sl < φ n³` (`L (M + K)`, and RS-W′'s excess while it stands), so
`ε = U / (0.8 F_B) − 1`. -/
theorem relativelySecure_edited120θ_rat (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (c mmax φ : ℚ) {sl : ℝ} (hc : 0 < c) (hcm : c ≤ mmax) (hφ : φ ≤ 1) (hsl : 0 ≤ sl) (hslφ : sl < φ * (n : ℝ) ^ 3)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService
      (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl)) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl))) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hq0 : (0 : ℝ) < ((c / mmax : ℚ) : ℝ) := by exact_mod_cast div_pos hc (hc.trans_le hcm)
  have hq1 : ((c / mmax : ℚ) : ℝ) ≤ 1 := by exact_mod_cast (div_le_one (hc.trans_le hcm)).mpr hcm
  have hφ' : (φ : ℝ) ≤ 1 := by exact_mod_cast hφ
  have hn3 : (0 : ℝ) ≤ (n : ℝ) ^ 3 := by positivity
  have hφn : (φ : ℝ) * (n : ℝ) ^ 3 ≤ (n : ℝ) ^ 3 := by nlinarith
  refine relativelySecure_edited120θ_of_floor kd G I S hn h32 h17 E hE prove hcom hacc hfloorB hκδ
    (mul_pos hq0 (by linarith)) ?_
  calc ((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl) ≤ 1 * ((φ : ℝ) * (n : ℝ) ^ 3 - sl) :=
        mul_le_mul_of_nonneg_right hq1 (by linarith)
    _ ≤ (n : ℝ) ^ 3 := by linarith

/-- Theorem E's `φ = 1 − f − m / N_B` (RS-D1E). -/
def phiE (f m NB : ℚ) : ℚ := 1 - f - m / NB

/-- **Route (c) in B_θ at Theorem E's proved floor** (`c = ½`, `m_max = 2`, `φ = 1 − f − m / N_B`): B's floor
`F_B = (φ n³ − sl) / 4` u gives `ε = U / ((φ n³ − sl) / 5) − 1`, about `4 / φ − 1` for large `n`. -/
theorem relativelySecure_edited120θ_half (kd : K120 → Kind) (G : MatGates kd (.ofNat price120) 16 1) (I : Int8Gates120 kd)
    {n : ℕ} {s : ℤ} (S : GridService n (127 - s)) (hn : 0 < n) (h32 : 32 ∣ n) (h17 : n ≤ 2 ^ 17)
    (E : (gridMatMachine120θ kd G I).Edit S.Seed n) (hE : E.Bounded s)
    (prove : (gridMatMachine120θ kd G I).Alg ((S.Seed × (Mat ℤ n n × Mat ℤ n n) × S.Coins) × Mat ℤ n n)
      (S.Com × S.Ans)) {κf δ η : ℝ}
    (hcom : ∀ (σ σ' : S.Seed) (y : (Mat ℤ n n × Mat ℤ n n) × S.Coins) (C C' : Mat ℤ n n),
      ((gridMatMachine120θ kd G I).eval prove ((σ, y), C)).1 =
        ((gridMatMachine120θ kd G I).eval prove ((σ', y), C')).1)
    (hacc : ∀ x : Mat ℤ n n × Mat ℤ n n, InGrid (127 - s) x → 1 - η ≤ prob fun ω : S.Seed × S.Coins =>
      S.verify ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).1
        ω.1 ((gridMatMachine120θ kd G I).eval prove
          ((ω.1, x, ω.2), (gridMatMachine120θ kd G I).eval ((gridMatMachine120θ kd G I).mulG n n n)
            (E.edit ω.1 x))).2 = true)
    (f m NB : ℚ) {sl : ℝ} (hf : 0 ≤ f) (hm : 0 ≤ m / NB) (hsl : 0 ≤ sl)
    (hslφ : sl < (phiE f m NB : ℝ) * (n : ℝ) ^ 3)
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S.toService
      (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl)) κf) (hκδ : κf < δ) :
    RelativelySecure ((gridMatMachine120θ kd G I).edited S.toService E ((gridMatMachine120θ kd G I).mulG n n n) prove (InGrid (127 - s)))
      ((4 / 5 * (n : ℝ) ^ 3 + 2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) / (4 / 5 * (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl))) - 1) δ
      ((2 * (4 / 5) * (n : ℝ) ^ 2 + (gridMatMachine120θ kd G I).time E.gen + (gridMatMachine120θ kd G I).time prove) /
        (4 / 5 * (n : ℝ) ^ 3)) η 0 (4 / 5 * (n : ℝ) ^ 3) := by
  have hq : ((1 / 2 / 2 : ℚ) : ℝ) = 1 / 4 := by norm_num
  have h := relativelySecure_edited120θ_rat kd G I S hn h32 h17 E hE prove hcom hacc (1 / 2) 2 (phiE f m NB)
    (by norm_num) (by norm_num) (by unfold phiE; linarith) hsl hslφ (by rw [hq]; exact hfloorB) hκδ
  rwa [hq] at h

/-- **The certified fraction** (the headline): from B's floor of Theorem E's form, every adversary in B_θ that commits
first has its accepted runs charged (`max(|M|, cost)`, the program included) below `(c φ / m_max) N − 0.8 (c / m_max) sl`
with probability at most `κ`, for `N = 0.8 n³` u the reference matmul. No commitment cost is credited. -/
theorem certifiedFraction120θ (kd : K120 → Kind) (S : Service) (n : ℕ) (c mmax φ : ℚ) {sl κf : ℝ}
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S (((c / mmax : ℚ) : ℝ) * ((φ : ℝ) * (n : ℝ) ^ 3 - sl)) κf) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      (((c * φ / mmax : ℚ) : ℝ) * (4 / 5 * (n : ℝ) ^ 3) - 4 / 5 * ((c / mmax : ℚ) : ℝ) * sl) κf := by
  convert serviceFloor_θ kd S hfloorB using 1
  push_cast
  ring

/-- **The certified fraction at Theorem E's proved floor** (`c = ½`, `m_max = 2`): accepted runs are charged at least
`(φ / 4) N − sl / 5` except with probability `κ`, `N = 0.8 n³` u. -/
theorem certifiedFraction120θ_half (kd : K120 → Kind) (S : Service) (n : ℕ) (f m NB : ℚ) {sl κf : ℝ}
    (hfloorB : ServiceFloor (machine kd (.ofNat price120)) S (1 / 4 * ((phiE f m NB : ℝ) * (n : ℝ) ^ 3 - sl)) κf) :
    ServiceFloor (machine (kdθ kd) priceθ) S
      ((phiE f m NB : ℝ) / 4 * (4 / 5 * (n : ℝ) ^ 3) - sl / 5) κf := by
  convert serviceFloor_θ kd S hfloorB using 1
  ring

/-- **The certified fraction from Theorem S** (in place of Theorem M): under the window's rounds in Model B
(`RoundsHyp`: per round, Theorem E's floor `F₀` of the round's GEMMs, the slack `sl = L(M + K)`, and the pass bound
along guard paths), every adversary in B_θ that commits first has its accepted runs charged below
`0.8 ((R − D) F₀ − Vmax − R sl)` with probability at most `e^{−(1−p) D} + R (δb + δz)`. Of the window's `0.8 R F₀`
that is the share `1 − D / R − (Vmax + R sl) / (R F₀)`. -/
theorem certifiedFraction120θ_rounds (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp (machine kd (.ofNat price120)) S Ω R p F₀ sl Vmax δb δz) (hp : p ≤ 1)
    (hF₀ : 0 < F₀) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) (D : ℝ) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - D) * F₀ - Vmax - R * sl))
      (Real.exp (-((1 - p) * D)) + R * (δb + δz)) :=
  serviceFloor_θ kd S (serviceFloor_of_rounds h hp hF₀ hδb hδz D)

/-- **The certified fraction at `κ = 2⁻⁴⁰ + R (δb + δz)`**: `D₀ = 40 ln 2 / (1 − p)`, so the luck loss is
`27.73 / ((1 − p) R)` of the window, detection included. -/
theorem certifiedFraction120θ_rounds_40 (kd : K120 → Kind) (S : Service) {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb δz : ℝ} (h : RoundsHyp (machine kd (.ofNat price120)) S Ω R p F₀ sl Vmax δb δz) (hp : p < 1)
    (hF₀ : 0 < F₀) (hδb : 0 ≤ δb) (hδz : 0 ≤ δz) :
    ServiceFloor (machine (kdθ kd) priceθ) S (4 / 5 * ((R - 40 * Real.log 2 / (1 - p)) * F₀ - Vmax - R * sl))
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz)) := by
  have hc : (1 - p) * (40 * Real.log 2 / (1 - p)) = ((40 : ℕ) : ℝ) * Real.log 2 := by
    field_simp [(sub_pos.mpr hp).ne']
    norm_num
  have h40 : Real.exp (-((1 - p) * (40 * Real.log 2 / (1 - p)))) = ((2 : ℝ) ^ 40)⁻¹ := by
    rw [hc, Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos]
  rw [← h40]
  exact certifiedFraction120θ_rounds kd S h hp.le hF₀ hδb hδz _

/-! ### Model A (RS-MODEL §3, over ℚ) and `B_θ ≤ A` -/

section ModelA

variable {Res : Type} (rs : List Res) (cost : K → Res → ℚ)

/-- RS-MODEL's `use`: per resource, the sum of a gate list's cost vectors. -/
def useA (l : List K) (r : Res) : ℚ := (l.map fun k => cost k r).sum

/-- RS-MODEL's `dot`: a row's load on a use vector, over the resources `rs`. -/
def dotA (row u : Res → ℚ) : ℚ := (rs.map fun r => row r * u r).sum

/-- RS-MODEL's `time_A`: the largest certified row (rows are normalized: a real run of length `T` has
`dot row (use P) ≤ T`). -/
def timeA (rows : List (Res → ℚ)) (l : List K) : ℚ := rows.foldr (fun row m => max (dotA rs row (useA cost l)) m) 0

theorem timeA_nonneg (rows : List (Res → ℚ)) (l : List K) : 0 ≤ timeA rs cost rows l := by
  induction rows with
  | nil => exact le_refl 0
  | cons a rows ih =>
    simp only [timeA, List.foldr_cons] at *
    exact le_max_of_le_right ih

theorem dotA_le_timeA {rows : List (Res → ℚ)} {row : Res → ℚ} (h : row ∈ rows) (l : List K) :
    dotA rs row (useA cost l) ≤ timeA rs cost rows l := by
  induction rows with
  | nil => cases h
  | cons a rows ih =>
    simp only [timeA, List.foldr_cons] at *
    rcases List.mem_cons.mp h with rfl | h
    · exact le_max_left _ _
    · exact (ih h).trans (le_max_right _ _)

/-- **RS-MODEL's `timeB_le_timeA`** (B as a dual combination of A's rows), over ℚ and for a price bounded by the
combination: if `price k ≤ Σ_p y_p (row_p · cost k)` with weights `y_p ≥ 0` on rows of A, then
`Σ price ≤ (Σ y) · time_A`. -/
theorem sum_le_timeA (rows : List (Res → ℚ)) (ys : List (ℚ × (Res → ℚ))) (hys : ∀ p ∈ ys, 0 ≤ p.1 ∧ p.2 ∈ rows)
    (price : K → ℚ) (hprice : ∀ k, price k ≤ (ys.map fun p => p.1 * dotA rs p.2 (cost k)).sum) (l : List K) :
    (l.map price).sum ≤ (ys.map Prod.fst).sum * timeA rs cost rows l := by
  have key : (l.map price).sum ≤ (ys.map fun p => p.1 * dotA rs p.2 (useA cost l)).sum := by
    induction l with
    | nil => simp [useA, dotA]
    | cons k l ih =>
      have e : ∀ p : ℚ × (Res → ℚ), p.1 * dotA rs p.2 (useA cost (k :: l)) =
          p.1 * dotA rs p.2 (cost k) + p.1 * dotA rs p.2 (useA cost l) := by
        intro p
        rw [← mul_add]
        congr 1
        simp [dotA, useA, mul_add, List.sum_map_add]
      rw [List.map_cons, List.sum_cons, List.map_congr_left (fun p _ => e p), List.sum_map_add]
      linarith [hprice k]
  calc _ ≤ _ := key
    _ ≤ (ys.map fun p => p.1 * timeA rs cost rows l).sum :=
        List.sum_le_sum fun p hp => mul_le_mul_of_nonneg_left (dotA_le_timeA rs cost (hys p hp).2 l) (hys p hp).1
    _ = _ := List.sum_map_mul_right _ _ _

end ModelA

/-- **`B_θ ≤ A`** (RS-MODEL §6.2's lemma, as an instance of `timeB_le_timeA`): if B's prices are a dual combination of
A's rows with total weight at most 1 (`hB`, `hsum`) and R6 is a row of A that prices each word at `W` (`hW`), then
`B_θ = (4/5)·B + (1/5)·R6` is the combination `(4/5)·ys + (1/5)·R6`, of total weight at most 1, so B_θ's time of
every gate list is at most A's. -/
theorem timeθ_le_timeA {Res : Type} (rs : List Res) (cost : Instr120 → Res → ℚ) (rows : List (Res → ℚ))
    (ys : List (ℚ × (Res → ℚ))) (hys : ∀ p ∈ ys, 0 ≤ p.1 ∧ p.2 ∈ rows) (hsum : (ys.map Prod.fst).sum ≤ 1)
    (hB : ∀ i, (price120 i.kind : ℚ) ≤ (ys.map fun p => p.1 * dotA rs p.2 (cost i)).sum)
    {R6 : Res → ℚ} (hR6 : R6 ∈ rows) (hW : ∀ i : Instr120, wordPrice * i.words ≤ dotA rs R6 (cost i))
    (l : List Instr120) : (l.map price120θ).sum ≤ timeA rs cost rows l := by
  have h := sum_le_timeA rs cost rows (ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]) ?_ price120θ ?_ l
  · have hs : ((ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]).map Prod.fst).sum =
        4 / 5 * (ys.map Prod.fst).sum + 1 / 5 := by
      simp only [List.map_append, List.map_map, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero]
      rw [← List.sum_map_mul_left]
      rfl
    rw [hs] at h
    have hA := timeA_nonneg rs cost rows l
    nlinarith
  · intro p hp
    rcases List.mem_append.mp hp with hp | hp
    · obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hp
      have := (hys q hq).1
      exact ⟨by positivity, (hys q hq).2⟩
    · rw [List.mem_singleton] at hp
      subst hp
      exact ⟨by norm_num, hR6⟩
  · intro i
    have e : ((ys.map (fun p : ℚ × (Res → ℚ) => (4 / 5 * p.1, p.2)) ++ [(1 / 5, R6)]).map fun p : ℚ × (Res → ℚ) => p.1 * dotA rs p.2 (cost i)).sum =
        4 / 5 * (ys.map fun p => p.1 * dotA rs p.2 (cost i)).sum + 1 / 5 * dotA rs R6 (cost i) := by
      simp only [List.map_append, List.map_map, List.sum_append, List.map_cons, List.map_nil, List.sum_cons,
        List.sum_nil, add_zero]
      rw [← List.sum_map_mul_left]
      congr 2
      exact List.map_congr_left fun p _ => by simp [mul_assoc]
    rw [e]
    unfold price120θ
    linarith [hB i, hW i]

end ModelB

end ModelB

end Pouw.RelativeSecurity
