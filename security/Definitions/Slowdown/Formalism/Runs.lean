module

public import Definitions.Slowdown.Formalism.Training

/-! Runs, the loss a run reaches, and the honest cost, taken over a class of runs because a circuit alone does not say which method produced it. -/

@[expose] public section

namespace Slowdown.Formalism

open Finset
open Verity (RegisteredCircuit)
open scoped ENNReal

/-- **A run**: a declaration, an annotation of its training with `numTokens` tokens, and the inputs it runs on. -/
structure Run where
  d : RegisteredCircuit
  numTokens : ℕ
  A : Annotation d (Fin numTokens)
  inputs : d.D.Trace

namespace Run

/-- **The loss the run reaches**: the evaluator `ℓ` on the bits of the values `θ'` takes on the run's inputs. -/
def loss (ℓ : List Bool → ℝ) (r : Run) : ℝ := ℓ (r.d.D.bits (r.d.D.eval r.inputs) (r.A.θ'.sort (· ≤ ·)))

/-- **The run reaches** the target loss `ℓ*`. -/
def Reaches (ℓ : List Bool → ℝ) (ℓstar : ℝ) (r : Run) : Prop := r.loss ℓ ≤ ℓstar

/-- **The run's update stays within** `D` bits: its total update `Δ`, decoded by `M` within `c` steps, is at most `D`. -/
def Within (M : Verity.Decoder (List Bool)) (c : ℕ) (D : ℕ∞) (r : Run) : Prop := r.A.totalUpdate M c r.inputs ≤ D

/-- **The run is one of** `methods`'s runs. -/
def Among (methods : Set Run) (r : Run) : Prop := r ∈ methods

end Run

/-- **The honest cost** `H(ℓ*)`: the least compute `|C|` of a run of `methods` that reaches `ℓ*`, `⊤` when none does. -/
noncomputable def honestCost (methods : Set Run) (ℓ : List Bool → ℝ) (ℓstar : ℝ) : ℝ≥0∞ :=
  ⨅ (r : Run) (_ : r ∈ methods) (_ : r.Reaches ℓ ℓstar), (r.d.D.computed.card : ℝ≥0∞)

noncomputable def Run.inefficiency (methods : Set Run) (ℓ : List Bool → ℝ) (ℓstar : ℝ) (r : Run) : ℝ≥0∞ :=
  (r.d.D.computed.card : ℝ≥0∞) / honestCost methods ℓ ℓstar

/-- **`B` blocks fit in `N` periods** of `n` blocks each. -/
def InPeriods (B N n : ℕ) : Prop := B ≤ N * n

end Slowdown.Formalism
