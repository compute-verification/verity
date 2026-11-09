module

public import Definitions.Nci.Formalism.Training

/-!
# The training formalism v2: runs, losses and the honest cost

A **run** (`Run`) is a declaration, an annotation of its training and the inputs it runs on. Its trained model is the
values `θ'` takes on those inputs, and the **loss it reaches** (`Run.loss`) is the evaluator `ℓ` on their bits in gate
order: `ℓ` maps trained weights to the loss of the model `θ` updated by them, so `ℓ₀`, the loss of `θ`, is `ℓ` of no
update, and `ℓ*` is the target loss.

A circuit alone does not say which method produced it: any run's `θ'` can be output by gates that compute constants, a
circuit about as large as `θ'` that reaches the same loss. The least compute of any method to reach `ℓ*` is therefore
taken over a class of runs, `methods`, the runs of the methods considered (algorithms that start from `θ` and the data,
not from the answer): `H(ℓ*)` is `honestCost methods ℓ ℓ*`, and a run's inefficiency `p = |C| / H(ℓ*)` is
`Run.inefficiency`. A run's update **stays within** `D` bits (`Run.Within`) when its total update `Δ` is at most `D`:
the bins of `Δ`, the update profiles along which A2's floor is measured.
-/

@[expose] public section

namespace Nci.Formalism

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

/-- **The inefficiency** `p = |C| / H(ℓ*)` of a run. -/
noncomputable def Run.inefficiency (methods : Set Run) (ℓ : List Bool → ℝ) (ℓstar : ℝ) (r : Run) : ℝ≥0∞ :=
  (r.d.D.computed.card : ℝ≥0∞) / honestCost methods ℓ ℓstar

/-- **`B` blocks fit in `N` periods** of `n` blocks each. -/
def InPeriods (B N n : ℕ) : Prop := B ≤ N * n

end Nci.Formalism
