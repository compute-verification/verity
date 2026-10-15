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
  `RelativelySecure`) with `ε`, `δ`, `γ`, `η`, `α`, `κ`. Probabilities are uniform counts over the finite σ and coins
  (`prob`). `secureInf_of_core` (Core ⇒ Security for `δ > κ`) and `core_of_floor` (RS-DEF's Lemma R) are proved.
* The masking algebra, proved over any ring (`mask_peel`, `lowRank_peel`), exact over `ZMod (2^32)` and over any
  field, and recovering the integer product from its s32 residue when the entries are below `2^31`
  (`int8_product_exact`); a saturating int8 accumulator breaks the peel (`sat8_breaks_peel`).
* The reduction's shape (`reduction`, `M' = M ++ Q`): run the extractor (M's run and its committed blocks), keep the
  prover messages, peel and correct; `time_reduction_le` is `time M' ≤ time E + time peel + time correction`.
  `secure_of_extraction_floor` gives security from two named hypotheses, `Extraction` (RS-EXT's) and `Floor` (the floor
  agenda's), and `secure_fullRank` and `secure_lowRank` instantiate it with the explicit peels of `MatMachine`.
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

end Probability

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

end Machine

/-- The axioms are satisfiable with nontrivial prices: algorithms are functions annotated with a cost, composition
adds costs, routing is free, and an `a × b` by `b × c` product costs exactly `a b c`. (This model has no floor: any
function is an algorithm. A floor needs a syntactic model, RS-MODEL's.) -/
def costed : Machine where
  Alg I O := (I → O) × ℕ
  eval f := f.1
  time f := f.2
  cost f _ := f.2
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
  /-- `C` answers `x` (for matmul: `A · B = C` on the exact grid). -/
  correct : Inst → Out → Prop
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
σ and its coins, for every instance (RS-DEF §1.4's event `b = 1 ∧ C = AB`). -/
def SolvesAll (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  ∀ x : Inst, 1 - η ≤ prob fun ω : P.Seed × Ω =>
    P.correct x (Mc.eval M' (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval M' (ω.1, x, ω.2)).1 = true

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

/-- **The per-run core** (RS-DEF §1.6, the proof target): for every adversary that commits first, the runs that are
accepted and cost less than `(T_use(η) − α)/(1 + ε)` have probability at most `κ`. -/
def Core (P : Protocol Mc Inst Out) (ε κ η α : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧
      Mc.cost M ω < (Tuse P η - α) / (1 + ε)) ≤ κ

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
    (q := fun ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.cost M ω < (Tuse P η - α) / (1 + ε))
    fun ω hω => ⟨hω, (Mc.cost_le_time M ω).trans_lt hbound⟩
  have := h Ω M hc
  unfold acceptance at hM
  linarith

/-- **The adversary floor** (a named hypothesis, the lane's floor results with Theorem F's luck term in `κ`): for every
adversary that commits first, the runs that are accepted and cost less than `F` have probability at most `κ`. -/
def AdvFloor (P : Protocol Mc Inst Out) (F κ : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
    prob (fun ω : P.Seed × Ω => P.accepts ω.1 (Mc.eval M ω) = true ∧ Mc.cost M ω < F) ≤ κ

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
  ∀ x C, P.correct x C ↔ (emb x).1 * (emb x).2 = C

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
  refine ⟨Ω', inferInstance, M', ht.trans ?_, hc', fun x => le_trans (by linarith) (hs x)⟩
  have := mul_le_mul_of_nonneg_right (show 1 + ε ≤ 1 + ε' by linarith) (Mc.time_nonneg M)
  linarith

theorem complete_mono (P : Protocol Mc Inst Out) {η η' : ℝ} (h : Complete P η) (hη : η ≤ η') : Complete P η' :=
  ⟨h.1, fun x => le_trans (by linarith) (h.2 x)⟩

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
committed blocks off its hash log, plus the check, the fixes and its own proof), and, for every instance `x`, with
probability at least `1 - η` over σ and its coins, outputs prover messages that are accepted and blocks valid for
`x`. -/
def Extraction (P : Protocol Mc Inst Out) {Bl : Type} (valid : Inst → Bl → Prop) (εext : ℝ) {Ω : Type} [Fintype Ω]
    (M : Mc.Alg (P.Seed × Ω) P.Proof) (η : ℝ) : Prop :=
  ∃ (Ω' : Type) (_ : Fintype Ω') (E : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Bl)),
    CommitsFirst P (fun y => (Mc.eval E y).1) ∧ Mc.time E ≤ (1 + εext) * Mc.time M ∧
    ∀ x, 1 - η ≤ prob fun ω : P.Seed × Ω' =>
      valid x (Mc.eval E (ω.1, x, ω.2)).2 ∧ P.accepts ω.1 (Mc.eval E (ω.1, x, ω.2)).1 = true

/-- **The floor** (a named hypothesis, the floor agenda's: Conjecture A, Theorems B/C/W/F): every algorithm that
answers every instance correctly with probability at least `1 - η` over its coins costs at least `T` (`T*_mm(n)` for
matmul, RS-DEF §1.7; no proof is asked for, so `T_use(η) ≥ T`). -/
def Floor (Mc : Machine) {Inst Out : Type} (correct : Inst → Out → Prop) (T η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (N : Mc.Alg (Inst × Ω) Out),
    (∀ x, 1 - η ≤ prob fun ω : Ω => correct x (Mc.eval N (x, ω))) → T ≤ Mc.time N

/-- An algorithm that solves every instance (with a proof) is above the floor: drop σ into its coins and forget
the proof, which routing does for free. -/
theorem Floor.le_time (P : Protocol Mc Inst Out) {T η : ℝ} (h : Floor Mc P.correct T η) {Ω : Type} [Fintype Ω]
    {M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out)} (hM' : SolvesAll P M' η) : T ≤ Mc.time M' := by
  let route : Mc.Alg (Inst × P.Seed × Ω) (P.Seed × Inst × Ω) :=
    Mc.pair (Mc.seq Mc.snd Mc.fst) (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.snd))
  have hroute : ∀ y, Mc.eval route y = (y.2.1, y.1, y.2.2) := fun y => by
    simp only [route, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]
  have hT := h (P.Seed × Ω) (Mc.seq (Mc.seq route M') Mc.snd) fun x => (hM' x).trans (prob_mono fun ω hω => by
    rw [Mc.eval_seq, Mc.eval_seq, hroute, Mc.eval_snd]
    exact hω.1)
  have hr : Mc.time route ≤ 0 := by
    apply le_trans
    · apply_rules [Machine.time_pair_le_of, Machine.time_seq_le_of, Machine.time_fst_le, Machine.time_snd_le]
    · norm_num
  have r1 := Mc.time_seq_le (Mc.seq route M') (Mc.snd (I := P.Proof) (J := Out))
  have r2 := Mc.time_seq_le route M'
  have z := Mc.time_snd (I := P.Proof) (J := Out)
  linarith

theorem Floor.le_tuse (P : Protocol Mc Inst Out) {T η : ℝ} (h : Floor Mc P.correct T η)
    (hne : ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)), Useful P M' η) :
    T ≤ Tuse P η := by
  obtain ⟨Ω', _, M', hu⟩ := hne
  exact le_csInf ⟨_, Ω', inferInstance, M', hu, rfl⟩ fun _ ⟨_, _, _, hu', ht⟩ => ht ▸ h.le_time P hu'.2

/-- With extraction, the reduction is useful and costs at most `(1 + εext) time M + time peel + time corr`, when the
peel and the correction answer `x` from blocks valid for `x`. -/
theorem exists_reduction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {εext η : ℝ} {Ω : Type} [Fintype Ω]
    {M : Mc.Alg (P.Seed × Ω) P.Proof} (hext : Extraction P valid εext M η) :
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + εext) * Mc.time M + Mc.time peel + Mc.time corr ∧ Useful P M' η := by
  obtain ⟨Ω', _, E, hc, hE, hsucc⟩ := hext
  refine ⟨Ω', inferInstance, Mc.reduction E peel corr, (Mc.time_reduction_le E peel corr).trans (by linarith),
    fun σ σ' y => ?_, fun x => (hsucc x).trans (prob_mono fun ω hω => ?_)⟩
  · simp only [Mc.eval_reduction]
    exact hc σ σ' y
  · rw [Mc.eval_reduction]
    exact ⟨hpeel x _ hω.1, hω.2⟩

/-- **Security from extraction**, given that the peel and the correction cost at most `ε time M` for every adversary
`M` accepted with probability at least `δ` (extraction overhead `εext`). -/
theorem secure_of_extraction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {εext ε δ η : ℝ}
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
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {εext δ η T q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P valid εext M η)
    (hfloor : Floor Mc P.correct T η) (hq : Mc.time peel + Mc.time corr ≤ q) (hqT : q < T) :
    Secure P (εext + (1 + εext) * (q / (T - q))) δ η 0 := by
  intro Ω _ M hc hM
  obtain ⟨Ω', _, M', ht, hu⟩ := exists_reduction P peel corr hpeel (hext Ω M hc hM)
  refine ⟨Ω', inferInstance, M', ?_, hu⟩
  have hTM := hfloor.le_time P hu.2
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
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {εext δ η F q : ℝ}
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
    (hfloor : Floor Mc P.correct T η) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hc : CommitsFirst P (Mc.eval M)) (hM : δ ≤ acceptance P M) : T ≤ (1 + ε) * Mc.time M + α := by
  obtain ⟨Ω', _, M', ht, hu⟩ := h Ω M hc hM
  exact (hfloor.le_time P hu.2).trans ht

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
    (hfloor : Floor Mc.toMachine P.correct T η) (hT : 2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelFull n) (Mc.corr n)
    (fun x b hb => (hmat x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hfloor
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n]) hT

/-- **Relative security of matmul PoUW with rank-`r` masks**, from extraction and the floor above
`q = 4κ n² r + 2κadd n²`: `ε = εext + (1 + εext) q / (T − q)`, and `q / (T − q)` is `O(r / n)` against a floor of order
`n³`. (RS-EXT's Lemma U: planting rank-`r` masks is detectable, so for RS-D2 this `Extraction` is expected to fail and
security comes from the floor itself, `core_of_floor`.) -/
theorem secure_lowRank (hmat : IsMatmulOn P emb) (r : ℕ) {εext δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), CommitsFirst P (Mc.eval M) →
      δ ≤ acceptance P M → Extraction P (fun x => ValidLow (r := r) (emb x)) εext M η)
    (hfloor : Floor Mc.toMachine P.correct T η)
    (hT : 4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P (εext + (1 + εext) * ((4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2)))) δ η 0 :=
  secure_of_extraction_floor P (Mc.peelLow n r) (Mc.corr n)
    (fun x b hb => (hmat x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hfloor
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
    (fun x b hb => (hmat x _).2 (Mc.peelFull_correct (emb x) b hb)) hext hM hF
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
    (fun x b hb => (hmat x _).2 (Mc.peelLow_correct (emb x) b hb)) hext hM hF
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n])

end MatMachine

end MatmulPeel

end Pouw.RelativeSecurity
