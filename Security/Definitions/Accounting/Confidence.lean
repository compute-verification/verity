import Definitions.Accounting.Actual
import Mathlib.Basic.Real.Basic

/-!
# Runs, strategies and the confidence set (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1), sections 2 ("Runs"), 4 and 5. A *run*
(`Run`) is an actual circuit with its gates' values, each computed gate's value its operation on the values it reads
(faithfulness). A gate is seen as its tick, leaf, width and value (`Run.seen`), so runs of different circuits compare.
The verifier at leaves `V` sees its gates (`view`). A *strategy* maps coins `r : Ω` to a run, *locally causal*
(`LocallyCausal`): what a location `X` does before `t` (`before`) is a function of its input gates (`inputs`) and of
what its gates read from outside it before `t` (`heard`). A guarantee `E` is *sound at `δ`* against the feasible runs
`𝓕` (`Sound`) when every locally causal strategy with runs in `𝓕` whose prover's inputs (every input gate off `V`) don't
depend on the coins accepts with `¬E` on at most a `δ` fraction of the coins, uniform over a finite `Ω`. The
*confidence set* `𝓒(v)` (`confidence`) is the feasible runs with view `v` meeting every guarantee.

For a quantity `μ` of a run: `req μ 𝓕 P` is the least `μ` of a feasible run that implements `P`, `Additive` says every
run of a set that implements `P` has `μ ≥ cert + req`, and the residual is `min(cap, obs) − cert`.
-/

namespace Accounting

open Finset

/-- **A run**: an actual circuit and its gates' values, each computed gate's its operation on what it reads. -/
structure Run (L : Type*) where
  A : Placed L
  val : A.Transcript
  faithful : ∀ g, A.isInput g = false → val g = A.op g val

namespace Run

variable {L : Type*} (R : Run L)

/-- A gate as seen from outside the circuit's numbering: its tick, leaf, width and value. -/
def seen (g : Fin R.A.N) : ℕ × L × ℕ × ℕ := (R.A.t g, R.A.x g, R.A.w g, (R.val g).toNat)

/-- **The view** of the verifier at leaves `V`: its gates, their ticks and values. -/
def view (V : Finset L) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ V ∧ p = R.seen g}

/-- What the location `X` does before `t`: its gates placed before `t`, when and with what values. -/
def before (X : Finset L) (t : ℕ) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.x g ∈ X ∧ R.A.t g < t ∧ p = R.seen g}

/-- The input gates placed at leaves of `X`. -/
def inputs (X : Set L) : Set (ℕ × L × ℕ × ℕ) := {p | ∃ g, R.A.isInput g = true ∧ R.A.x g ∈ X ∧ p = R.seen g}

/-- What the gates of `X` read from outside it before `t`, each with the tick it was read at. -/
def heard (X : Finset L) (t : ℕ) : Set (ℕ × ℕ × L × ℕ × ℕ) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ R.A.x g ∈ X ∧ R.A.x a ∉ X ∧ R.A.t g < t ∧ p = (R.A.t g, R.seen a)}

end Run

/-- **Causality at every location**: in a strategy `S`, what each location `X` does before `t` is a function of its
input gates and of what it reads from outside before `t`. -/
def LocallyCausal {Ω L : Type*} (S : Ω → Run L) : Prop :=
  ∀ (X : Finset L) t r r', (S r).inputs X = (S r').inputs X → (S r).heard X t = (S r').heard X t →
    (S r).before X t = (S r').before X t

open Classical in
/-- **`E` is sound at `δ`** against the feasible runs `𝓕`, for the verifier at `V` accepting on `acc` of its view with
coins uniform over `Ω`: every locally causal strategy with runs in `𝓕` whose input gates off `V` don't depend on the
coins has `Pr_r[acc ∧ ¬E] ≤ δ`. -/
def Sound {L : Type*} (Ω : Type) [Fintype Ω] (𝓕 : Set (Run L)) (V : Finset L) (acc : Set (ℕ × L × ℕ × ℕ) → Prop)
    (E : Run L → Prop) (δ : ℝ) : Prop :=
  ∀ S : Ω → Run L, (∀ r, S r ∈ 𝓕) → LocallyCausal S → (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
    ((univ.filter fun r => acc ((S r).view V) ∧ ¬E (S r)).card : ℝ) ≤ δ * Fintype.card Ω

/-- **The confidence set** `𝓒(v)`: the feasible runs with the view `v` at `V` that meet every guarantee `E i`. -/
def confidence {L ι : Type*} (𝓕 : Set (Run L)) (V : Finset L) (E : ι → Run L → Prop) (v : Set (ℕ × L × ℕ × ℕ)) :
    Set (Run L) :=
  {R | R ∈ 𝓕 ∧ R.view V = v ∧ ∀ i, E i R}

/-- `req_μ(P)`: the least `μ` of a run of `𝓕` that implements `P` (`impl`), `⊤` when none does. -/
noncomputable def req {L : Type*} (μ : Run L → ℕ) (𝓕 : Set (Run L)) (impl : Run L → Prop) : ℕ∞ :=
  ⨅ (R : Run L) (_ : R ∈ 𝓕 ∧ impl R), (μ R : ℕ∞)

/-- **`μ` adds** over the runs `𝓒` for `P` above `cert`: each of them that implements `P` has `μ ≥ cert + req_μ(P)`. -/
def Additive {L : Type*} (𝓒 : Set (Run L)) (μ : Run L → ℕ) (cert : ℕ) (r : ℕ∞) (impl : Run L → Prop) : Prop :=
  ∀ R ∈ 𝓒, impl R → (cert : ℕ∞) + r ≤ μ R

end Accounting
