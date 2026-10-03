import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Indexed
import Pous.Protocol.Model.Inc

/-!
# The continuous audit (DRAFT: a new trusted game, not pinned)

**Draft.** Nothing here is pinned or graded. It is a new game, so it needs @lean's statement review and a red team
before any of it is cited. Once reviewed, the definitions move to `Pous/Protocol/Model/` and the two statements to
`Pous/Guarantees/`. `Continuous/PerAnswer.lean` proves both statements from `TimedINC`.

**The game** (the PoUS explorer note, §1), conditional on the arrivals: `n` challenges, each index uniform and
independent. At challenge `j` the responder holds a state `σ_j` of any length. It answers index `i` by a
`(D, Q)`-bounded program of `(W, pp, j, σ_j, i)`, so every answer is on time; a late answer is a wrong one here. Between
challenges it replaces its state by any function of `(W, pp, j, σ_j, i)` and the whole primitive: the time between
arrivals is unbounded, and `C` is gone after preprocessing. Answer `j` misses if it is wrong. `ε_j = 1 − |σ_j|/|C|`
is the freed fraction at challenge `j`, and the verifier accepts iff the misses are at most `f₀`.

What it leaves out:

* **The arrival times.** Conditioning on them fixes `n` and lets the responder know every arrival time in advance,
  which only helps it. `ContinuousAudit`'s event, `Σ_j (ε_j − ε_proof) > G(f)`, doesn't mention `n`, so with the
  number of challenges `N` drawn independently of the indices (Poisson arrivals) the bound holds unconditionally, with
  `E[N]·(β + 1)·π` in place of `n·(β + 1)·π`. The explorer's `ε̄ > ε_proof + g(f₀)` with `g = G/(λT)` is this event
  when `ε̄` is the sum over the challenges divided by `λT`. With `ε̄` the average over the `N` challenges, `g` must be
  `G/N`: dividing by `λT` when `N < λT` claims more than this proves.
* **Overlapping challenges.** Index `j + 1` is revealed only after answer `j`. If the verifier sends challenge `j + 1`
  while answer `j` is still within its deadline, the responder may use `i_{j+1}` for answer `j`, which this game
  doesn't model.
* **The time average.** `ε_j` is sampled at the challenges, not averaged over `[0, T]`.
-/

namespace PousContinuous

open Pous
open scoped ENNReal

variable {M : IdealModel}

/-- **Draft.** Preprocessing: any function of `(W, pp, C)` and the whole primitive, with a state of any length. -/
abbrev VarPreprocessor (sch : Scheme M) : Type :=
  Bits sch.n → Bits sch.t → sch.Code → M.Ω → List Bool

/-- **Draft.** A responder in the continuous audit. `ans W pp j σ i` answers index `i` at challenge `j` from state `σ`
within `(D, Q)`; `upd W pp j σ i ω` is the state it holds at the next challenge, any function of the same inputs and
the whole primitive. -/
structure VarResponder (sch : Scheme M) (D Q : ℕ) : Type where
  ans : Bits sch.n → Bits sch.t → ℕ → List Bool → Fin sch.B → Prog M (Bits sch.ℓ)
  bounded : ∀ W pp j σ i, (ans W pp j σ i).Bounded D Q
  upd : Bits sch.n → Bits sch.t → ℕ → List Bool → Fin sch.B → M.Ω → List Bool

/-- The state after the indices `h`, from challenge `j` in state `σ`. -/
def varState {B : ℕ} (upd : ℕ → List Bool → Fin B → List Bool) : ℕ → List (Fin B) → List Bool → List Bool
  | _, [], σ => σ
  | j, i :: is, σ => varState upd (j + 1) is (upd j σ i)

/-- **Draft.** The state at the challenge after the history `h` (the earlier indices, oldest first), at setup
`a = (ω, r)`. -/
def stateAt (sch : Scheme M) {D Q : ℕ} (W : Bits sch.n) (A₁ : VarPreprocessor sch) (R : VarResponder sch D Q)
    (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) : List Bool :=
  varState (fun j σ i => R.upd W (sch.enc W a.1 a.2).1 j σ i a.1) 0 h
    (A₁ W (sch.enc W a.1 a.2).1 (sch.enc W a.1 a.2).2 a.1)

/-- **Draft.** After the history `h`, index `i` is answered correctly. -/
def hit (sch : Scheme M) {D Q : ℕ} (W : Bits sch.n) (A₁ : VarPreprocessor sch) (R : VarResponder sch D Q)
    (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) (i : Fin sch.B) : Prop :=
  (R.ans W (sch.enc W a.1 a.2).1 h.length (stateAt sch W A₁ R a h) i).run (M.answer a.1) = (sch.enc W a.1 a.2).2 i

/-- **Draft.** The freed fraction at the challenge after `h`: `ε = 1 − |σ|/|C|`. -/
noncomputable def freed (sch : Scheme M) {D Q : ℕ} (W : Bits sch.n) (A₁ : VarPreprocessor sch)
    (R : VarResponder sch D Q) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) : ℝ :=
  1 - ((stateAt sch W A₁ R a h).length : ℝ) / (sch.B * sch.ℓ)

/-- **Draft.** The per-answer slack of a compression bound to `β` bits: `ε_proof = 1 − (β − B)/|C|`. With
`TimedINC sch D Q β π`, an answer from a state of `(1 − ε)·|C|` bits is correct with probability at most
`1 − ε + ε_proof`, off a setup event of probability at most `(β + 1)·π` (`PerAnswer`). -/
noncomputable def epsProof (sch : Scheme M) (β : ℕ) : ℝ :=
  1 - ((β : ℝ) - sch.B) / (sch.B * sch.ℓ)

/-- **Draft, step (a): the per-answer bound given the history.** From `TimedINC sch D Q β π`: for every history `h`,
off a setup event of probability at most `(β + 1)·π`, the challenge after `h` is answered correctly with probability at
most `1 − ε + ε_proof` over its uniform index, where `ε` is the freed fraction then. The `β + 1` is a union over the
state's length. -/
def PerAnswer : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q β : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (A₁ : VarPreprocessor sch) (R : VarResponder sch D Q) (h : List (Fin sch.B)),
      pr (fun a : M.Ω × sch.Coins =>
          ¬ pr (hit sch W A₁ R a h) ≤ ENNReal.ofReal (1 - freed sch W A₁ R a h + epsProof sch β)) ≤ (β + 1) * π

open Classical in
/-- **Draft: the continuous audit's guarantee** (the explorer note, §1), conditional on `n` arrivals. From
`TimedINC sch D Q β π`, for every `f` and every `βe > 0` (the explorer's `β`): the misses are at most `f` and
`Σ_j (ε_j − ε_proof)` exceeds `G(f) = inf_{θ > 0} (θ·f + ln(1/βe)) / (1 − e^−θ)` with probability at most
`βe + n·(β + 1)·π`. Dividing by `n`: `Pr[F ≤ f₀ ∧ ε̄ > ε_proof + g(f₀)] ≤ βe + n·(β + 1)·π`, `g = G/n`. -/
def ContinuousAudit : Prop :=
  ∀ {M : IdealModel} (sch : Scheme M) (D Q β n : ℕ) (π : ℝ≥0∞), TimedINC sch D Q β π →
    ∀ (W : Bits sch.n) (A₁ : VarPreprocessor sch) (R : VarResponder sch D Q) (f βe : ℝ), 0 < βe →
      pr (fun x : Outcome sch n =>
          ((Finset.univ.filter fun j : Fin n =>
              ¬ hit sch W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) (x.2.2 j)).card : ℝ) ≤ f ∧
            (⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ)))) <
              ∑ j : Fin n, (freed sch W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) - epsProof sch β))
        ≤ ENNReal.ofReal βe + n * (β + 1) * π

end PousContinuous
