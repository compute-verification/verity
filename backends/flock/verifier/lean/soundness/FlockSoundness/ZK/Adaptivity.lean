import Mathlib.Data.Fintype.Pi
import Mathlib.Basic.Finite.Prod
import Mathlib.Data.Finset.Card
import Mathlib.Logic.Function.Iterate
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.Abel

/-!
# Zero knowledge: the bijection lemma's backbone, and adaptivity

Two steps of the public-circuit zero-knowledge proof (`docs/zk-proof-public.md` in the Project store), in the abstract form
the proof uses them.

* **The bijection lemma** (Lemma A). At fixed coins, the proof maps the prover's randomness for one witness to its
  randomness for the other by translating each block by an amount that reads only earlier blocks. `triShift_bijective`:
  such a triangular translation is a bijection of the (finite) randomness space. `card_fiber_eq_of_triShift`: so two views
  that agree along it take every value equally often, which is Lemma A's conclusion from its construction. The
  construction itself (which blocks, which amounts) is protocol-specific: `Table.table_prefinal_translate`, and
  `Table.table_prefinal_translate_cut` where the prover refuses.
* **Adaptivity** (T5). A prover strategy's `i`-th message reads its randomness and the verifier's answers `j < i`; a
  deterministic verifier's `i`-th answer reads the messages `j ≤ i`. Answers are arbitrary values, so an abort, an invalid
  or a different valid coin opening, and the verifier's choice of which stream to serve next are all answers. `run` is the
  interaction's transcript. `run_eq_iff`: the transcript is `τ` exactly when the fixed-answer messages at the answers `τ`
  itself determines equal `τ`. `card_run_eq`: so if two provers' messages have equal distributions at every fixed answer
  vector, their transcripts against every verifier have equal distributions.
* `card_run_eq_of_triShift` chains the two: a triangular translation at every fixed answer vector gives equal transcript
  distributions against every deterministic verifier. That is the step from Lemma A to Lemma C in the ideal-leaf world.

Only exact equality lifts this way; a statistical gap at fixed coins does not, which is why the paper proof passes through
ideal leaves first.
-/

namespace FlockSoundness.ZK

open Finset

/-! ## Triangular translations -/

section Triangular

variable {K : ℕ} {G : Fin K → Type*}

/-- `δ` is triangular: coordinate `k`'s shift reads only coordinates `j < k`. -/
def Triangular (δ : ∀ k, (∀ j, G j) → G k) : Prop :=
  ∀ k (ω ω' : ∀ j, G j), (∀ j, j < k → ω j = ω' j) → δ k ω = δ k ω'

variable [∀ k, AddGroup (G k)]

/-- The triangular translation `ω ↦ (ω_k + δ_k(ω))_k`. -/
def triShift (δ : ∀ k, (∀ j, G j) → G k) (ω : ∀ j, G j) : ∀ k, G k :=
  fun k => ω k + δ k ω

theorem triShift_injective {δ : ∀ k, (∀ j, G j) → G k} (hδ : Triangular δ) :
    Function.Injective (triShift δ) := by
  intro ω ω' h
  have key : ∀ m : ℕ, ∀ k : Fin K, k.val < m → ω k = ω' k := by
    intro m
    induction m with
    | zero => intro k hk; exact absurd hk (Nat.not_lt_zero _)
    | succ m ih =>
      intro k hk
      have hprev : ∀ j, j < k → ω j = ω' j := fun j hj =>
        ih j (Nat.lt_of_lt_of_le hj (Nat.lt_succ_iff.mp hk))
      have hk' := congrFun h k
      simp only [triShift] at hk'
      rw [hδ k ω ω' hprev] at hk'
      exact add_right_cancel hk'
  funext k
  exact key (k.val + 1) k (Nat.lt_succ_self _)

theorem triShift_bijective [∀ k, Finite (G k)] {δ : ∀ k, (∀ j, G j) → G k} (hδ : Triangular δ) :
    Function.Bijective (triShift δ) :=
  Finite.injective_iff_bijective.mp (triShift_injective hδ)

/-- **The bijection lemma's backbone.** If `view` agrees with `view'` along a triangular translation of the randomness,
both take every value equally often: the map Lemma A builds preserves the uniform distribution. -/
theorem card_fiber_eq_of_triShift [∀ k, Fintype (G k)] [∀ k, DecidableEq (G k)] {V : Type*} [DecidableEq V]
    (view view' : (∀ j, G j) → V) {δ : ∀ k, (∀ j, G j) → G k} (hδ : Triangular δ)
    (h : ∀ ω, view ω = view' (triShift δ ω)) (v : V) :
    #{ω | view ω = v} = #{ω | view' ω = v} := by
  let e := Equiv.ofBijective _ (triShift_bijective hδ)
  refine Finset.card_equiv e (fun ω => ?_)
  simp [e, h]

/-- **Sequential masking.** A view whose component `k` is anything computed from the randomness blocks `j < k` plus a
surjective image of block `k` takes every value equally often, whatever that computation is. `f` carries the witness, the
coins and the earlier randomness, and it drops out: every value is taken by as many `ω` as there are vectors in the masks'
kernels. This is the engine of Lemma B's sampling order, and of Lemma A's conclusion. -/
theorem card_seq_masked {K : ℕ} {G B : Fin K → Type*} [∀ k, AddCommGroup (G k)] [∀ k, Fintype (G k)]
    [∀ k, DecidableEq (G k)] [∀ k, AddCommGroup (B k)] [∀ k, DecidableEq (B k)]
    (φ : ∀ k, G k →+ B k) (hφ : ∀ k, Function.Surjective (φ k)) (f : ∀ k, (∀ j, G j) → B k)
    (hf : ∀ k (ω ω' : ∀ j, G j), (∀ j, j < k → ω j = ω' j) → f k ω = f k ω') (v : ∀ k, B k) :
    #{ω | ∀ k, f k ω + φ k (ω k) = v k} = #{κ : ∀ k, G k | ∀ k, φ k (κ k) = 0} := by
  choose s hs using hφ
  let δ : ∀ k, (∀ j, G j) → G k := fun k ω => -(s k (v k - f k ω))
  have hδ : Triangular δ := fun k ω ω' h => by simp only [δ, hf k ω ω' h]
  let e := Equiv.ofBijective _ (triShift_bijective hδ)
  refine Finset.card_equiv e (fun ω => ?_)
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, e, Equiv.ofBijective_apply, triShift, δ, map_add, map_neg,
    hs]
  refine ⟨fun h k => ?_, fun h k => ?_⟩
  · rw [← h k]
    abel
  · have h1 := h k
    rw [← sub_eq_add_neg, sub_eq_zero] at h1
    rw [h1]
    abel

end Triangular

/-! ## Adaptive verifiers -/

section Adaptive

variable {Ω M C : Type*} {n : ℕ}

/-- An `n`-round prover strategy: message `i` from the randomness and the verifier's answers `j < i`. -/
structure Strategy (Ω M C : Type*) (n : ℕ) where
  msg : Ω → (Fin n → C) → Fin n → M
  causal : ∀ ω (c c' : Fin n → C) (i : Fin n), (∀ j, j < i → c j = c' j) → msg ω c i = msg ω c' i

/-- A deterministic verifier (its tape and auxiliary input fixed): answer `i` from the messages `j ≤ i`. -/
structure Responder (M C : Type*) (n : ℕ) where
  ans : (Fin n → M) → Fin n → C
  causal : ∀ (τ τ' : Fin n → M) (i : Fin n), (∀ j, j ≤ i → τ j = τ' j) → ans τ i = ans τ' i

/-- One pass of the interaction: every message recomputed from the verifier's answers to the current transcript. -/
def Strategy.step (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) (τ : Fin n → M) : Fin n → M :=
  P.msg ω (V.ans τ)

/-- The interaction's transcript on randomness `ω`: `n` passes from any start (message `i` is settled after pass
`i + 1`). -/
def run [Inhabited M] (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) : Fin n → M :=
  (P.step V ω)^[n] (fun _ => default)

/-- A pass extends agreement by one message: transcripts that agree before `k` step to transcripts that agree before
`k + 1`. -/
theorem Strategy.step_agree (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) {τ τ' : Fin n → M} {k : ℕ}
    (h : ∀ j : Fin n, j.val < k → τ j = τ' j) : ∀ i : Fin n, i.val < k + 1 → P.step V ω τ i = P.step V ω τ' i := by
  intro i hi
  refine P.causal ω _ _ i (fun j hj => V.causal τ τ' j (fun j' hj' => h j' ?_))
  have : j'.val ≤ j.val := hj'
  have : j.val < i.val := hj
  omega

theorem Strategy.iterate_agree (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) (τ₀ : Fin n → M) :
    ∀ k : ℕ, ∀ i : Fin n, i.val < k → ((P.step V ω)^[k] τ₀) i = ((P.step V ω)^[k + 1] τ₀) i := by
  intro k
  induction k with
  | zero => intro i hi; exact absurd hi (Nat.not_lt_zero _)
  | succ k ih =>
    intro i hi
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply' (P.step V ω) (k + 1)]
    exact P.step_agree V ω ih i hi

/-- The transcript is a fixed point of a pass. -/
theorem run_fixed [Inhabited M] (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) :
    P.step V ω (run P V ω) = run P V ω := by
  funext i
  unfold run
  rw [← Function.iterate_succ_apply' (P.step V ω)]
  exact (P.iterate_agree V ω _ n i i.isLt).symm

/-- A pass has one fixed point. -/
theorem fixed_unique (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) {σ σ' : Fin n → M}
    (hσ : P.step V ω σ = σ) (hσ' : P.step V ω σ' = σ') : σ = σ' := by
  have key : ∀ k : ℕ, ∀ i : Fin n, i.val < k → σ i = σ' i := by
    intro k
    induction k with
    | zero => intro i hi; exact absurd hi (Nat.not_lt_zero _)
    | succ k ih =>
      intro i hi
      rw [← hσ, ← hσ']
      exact P.step_agree V ω ih i hi
  funext i
  exact key n i i.isLt

/-- **The transcript is `τ` exactly when the fixed-answer messages at `τ`'s own answers are `τ`.** -/
theorem run_eq_iff [Inhabited M] (P : Strategy Ω M C n) (V : Responder M C n) (ω : Ω) (τ : Fin n → M) :
    run P V ω = τ ↔ P.msg ω (V.ans τ) = τ := by
  constructor
  · rintro rfl
    exact run_fixed P V ω
  · intro h
    exact fixed_unique P V ω (run_fixed P V ω) h

/-- **T5, adaptivity is free for exact equality.** If two provers' messages take every value equally often at every
fixed answer vector, their transcripts against any deterministic verifier take every value equally often. -/
theorem card_run_eq [Inhabited M] [Fintype Ω] [DecidableEq M] (P P' : Strategy Ω M C n) (V : Responder M C n)
    (h : ∀ (c : Fin n → C) (τ : Fin n → M), #{ω | P.msg ω c = τ} = #{ω | P'.msg ω c = τ}) (τ : Fin n → M) :
    #{ω | run P V ω = τ} = #{ω | run P' V ω = τ} := by
  have e : ∀ Q : Strategy Ω M C n, #{ω | run Q V ω = τ} = #{ω | Q.msg ω (V.ans τ) = τ} := fun Q => by
    congr 1
    ext ω
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, run_eq_iff]
  rw [e P, e P', h]

/-- **From Lemma A to Lemma C.** If at every fixed answer vector the two provers' messages agree along a triangular
translation of the randomness, their transcripts against any deterministic verifier take every value equally often. -/
theorem card_run_eq_of_triShift {K : ℕ} {G : Fin K → Type*} [∀ k, AddGroup (G k)] [∀ k, Fintype (G k)]
    [∀ k, DecidableEq (G k)] [Inhabited M] [DecidableEq M] (P P' : Strategy (∀ k, G k) M C n) (V : Responder M C n)
    (δ : (Fin n → C) → ∀ k, (∀ j, G j) → G k) (hδ : ∀ c, Triangular (δ c))
    (h : ∀ c ω, P.msg ω c = P'.msg (triShift (δ c) ω) c) (τ : Fin n → M) :
    #{ω | run P V ω = τ} = #{ω | run P' V ω = τ} :=
  card_run_eq P P' V
    (fun c τ => card_fiber_eq_of_triShift (fun ω => P.msg ω c) (fun ω => P'.msg ω c) (hδ c) (h c) τ) τ

end Adaptive

end FlockSoundness.ZK
