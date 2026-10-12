import Definitions.Core.Circuit
import Definitions.Core.Declaration
import Mathlib.Order.Interval.Set.Defs

/-!
# The actual circuit (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2), section 2. The *actual circuit* of a run is
the circuit of everything the hardware did (`Verity.Circuit`): a gate for each operation, a wire for each dependence,
and each gate placed at a leaf `x g` of the tree of locations, a cell `cell g` of that leaf and a tick `t g` (`Placed`;
the cell is v2.5's: a leaf's memory is its cells, a program reads cells, and a cell takes at most one gate a tick). A
location is a set of leaves (`Finset L`); the tree is laminar and never needed explicitly. Its input gates are the
values that existed before the run, such as stored weights and the prover's secrets (the model places them at tick 0;
nothing here needs it).

For a location `X` and a set `I` of ticks (an interval `[a, b)`, `Set.Ico a b`, or `[t, ∞)`, `Set.Ici t`):
* `region X I`: `A[X, I]`, the computed gates placed in `X` during `I`. Input gates are in no region, as they are in no
  unit of a `Verity.Partition`: a region reads the values it starts from, so `state` counts what `X` stores;
* `work π R`: `π(R)`, the work of a set of gates at a pricing `π`;
* `ingress X I`: `in_X(I)`, the bits read into `X` during `I`: the width of the gates placed outside `X` that `A[X, I]`
  reads;
* `egress X I`: `out_X(I)`, the width of the gates of `A[X, I]` that a gate placed outside `X` reads;
* `state X t`: `mem_X(t)`, the bits `X` carries across `t`: the width of the gates placed in `X` that `A[X, [t, ∞))`
  reads from outside itself, the gates `X` computed before `t` and the input gates it stores.

What a region reads from outside itself is what its location carried in plus what crossed its boundary (the split,
`Accounting.SecurityProofs.Actual.width_into_region_le`), when wires run forward in time inside the location (`Forward`):
causality at latency 0, inside `X`, for computed sources, which is all the split needs.

`cut S` is a set `S` of gates as a circuit whose inputs are what it reads: `A` with every gate outside `S` an input gate,
and the gates of `S` its outputs. `declaration X t θ` is `A[X, [0, t))` so cut, with the baseline `θ_A`, the input gates
committed before the run, as NCI's `Declaration`. The runtime `t_A(P)` reads the strategy, not one run
(`Accounting.Actual.runtime`, v2.7).
-/

namespace Accounting

open Verity Finset

/-- **An actual circuit**: a circuit, each gate `g` placed at a leaf `x g` of the tree of locations, a cell `cell g`
of that leaf and a tick `t g`, a cell taking at most one gate a tick. -/
structure Placed (L : Type*) extends Circuit where
  /-- The leaf at which gate `g` ran. -/
  x : Fin N → L
  /-- The tick at which gate `g` ran. -/
  t : Fin N → ℕ
  /-- The cell of its leaf at which gate `g` ran (v2.5). -/
  cell : Fin N → ℕ
  /-- A cell takes at most one gate a tick. -/
  cell_inj : Function.Injective fun g => (x g, cell g, t g)

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
/-- **The region** `A[X, I]`: the computed gates placed in the location `X` during the ticks `I`. -/
noncomputable def region (X : Finset L) (I : Set ℕ) : Finset (Fin A.N) :=
  A.computed.filter fun g => A.x g ∈ X ∧ A.t g ∈ I

/-- **Work** `π(R)` at a pricing `π`: the total price of the gates of `R`. -/
def work (π : Fin A.N → ℕ) (R : Finset (Fin A.N)) : ℕ := ∑ g ∈ R, π g

open Classical in
/-- **Ingress** `in_X(I)`: the width of the gates placed outside `X` that `A[X, I]` reads. -/
noncomputable def ingress (X : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.into (A.region X I)).filter fun a => A.x a ∉ X)

open Classical in
/-- **Egress** `out_X(I)`: the width of the gates of `A[X, I]` that some gate placed outside `X` reads. -/
noncomputable def egress (X : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.region X I).filter fun g => ∃ h, A.x h ∉ X ∧ g ∈ A.reads h)

open Classical in
/-- **State** `mem_X(t)`: the width of the gates placed in `X` that `A[X, [t, ∞))` reads from outside itself. -/
noncomputable def state (X : Finset L) (t : ℕ) : ℕ :=
  A.width ((A.into (A.region X (Set.Ici t))).filter fun a => A.x a ∈ X)

/-- **Wires run forward in time inside `X`**: a computed gate placed in `X` is read by gates placed in `X` only at its
tick or later. -/
def Forward (X : Finset L) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.isInput a = false → A.x a ∈ X → A.x g ∈ X → A.t a ≤ A.t g

open Classical in
/-- **A set of gates as a circuit whose inputs are what it reads**: `A` with every gate outside `S` an input gate, which
reads nothing, and the gates of `S` its outputs. -/
noncomputable def cut (S : Finset (Fin A.N)) : Circuit where
  N := A.N
  w := A.w
  isInput g := if g ∈ S then A.isInput g else true
  reads g := if g ∈ S then A.reads g else ∅
  reads_lt g a ha := by
    by_cases hg : g ∈ S
    · simp only [hg, ite_true] at ha
      exact A.reads_lt g a ha
    · simp [hg] at ha
  input_reads g hg := by
    by_cases hS : g ∈ S
    · simp only [hS, ite_true] at hg ⊢
      exact A.input_reads g hg
    · simp [hS]
  op g := if g ∈ S then A.op g else fun _ => 0
  op_congr g v v' hv := by
    by_cases hS : g ∈ S
    · simp only [hS, ite_true] at hv ⊢
      exact A.op_congr g v v' hv
    · simp [hS]
  outputs := S

/-- **The developer's declaration** over `[0, t)`: `A[X, [0, t))` as a circuit whose inputs are what it reads, with the
baseline `θ`, input gates of `A` (the model's `θ_A`, those committed before the run). Every input gate of `A` is an
input gate of the cut, so `θ` needs no restricting. -/
noncomputable def declaration (X : Finset L) (t : ℕ) (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) :
    Declaration where
  D := A.cut (A.region X (Set.Iio t))
  θ := θ
  θ_input a ha := by simp [cut, hθ a ha]

end Placed

end Accounting
