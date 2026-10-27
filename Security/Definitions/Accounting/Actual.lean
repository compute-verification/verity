import Definitions.Core.Circuit
import Definitions.Core.Declaration
import Mathlib.Order.Interval.Set.Defs

/-!
# The actual circuit (trusted)

The actual-circuit model. The *actual circuit* of a run is the circuit of everything the hardware did
(`Verity.Circuit`): a gate for each operation, a wire for each dependence, and each gate placed at a leaf `x g` of the
tree of locations, a cell `cell g` of that leaf and a tick `t g` (`Placed`; a leaf's memory is its cells, a program
reads cells, and a cell takes at most one gate a tick). A location is a set of leaves (`Finset L`); the tree is laminar
and never needed explicitly. Its input gates are the values that existed before the run, such as stored weights and the
prover's secrets (the model places them at tick 0; nothing here needs it).

For a location `X` and a set `I` of ticks (an interval `[a, b)`, `Set.Ico a b`, or `[t, ∞)`, `Set.Ici t`):
* `region X I`: `A[X, I]`, the computed gates placed in `X` during `I`. Input gates are in no region, as they are in no
  unit of a `Verity.Partition`: a region reads the values it starts from, so `state` counts what `X` stores;
* `work π R`: `π(R)`, the work of a set of gates at a pricing `π`;
* `ingress X I`: `in_X(I)`, the bits read into `X` during `I`: the width of the gates placed outside `X` that `A[X, I]`
  reads;
* `egress X I`: `out_X(I)`, the width of the gates placed in `X` that a gate placed outside `X` during `I` reads;
* `state X t`: `mem_X(t)`, the bits `X` carries across `t`: the width of the gates placed in `X` that `A[X, [t, ∞))`
  reads from outside itself, the gates `X` computed before `t` and the input gates it stores;
* `stateIn X t G`: the part of `mem_X(t)` in a set `G` of gates. `state X t` is `stateIn X t G + stateIn X t Gᶜ`
  (`Accounting.SecurityProofs.Actual.state_eq_stateIn_add_compl`).

A run's windows are cut at ticks `0 = τ 0 ≤ τ 1 ≤ … ≤ τ J = t` (`Cuts t`), window `j` being `[τ j, τ (j + 1))`; the
cuts may differ from run to run.

What a region reads from outside itself is what its location carried in plus what crossed its boundary (the split,
`Accounting.SecurityProofs.Actual.width_into_region_le`), when wires run forward in time inside the location (`Forward`):
causality at latency 0, inside `X`, for computed sources, which is all the split needs.

`cut S` is a set `S` of gates as a circuit whose inputs are what it reads: `A` with every gate outside `S` an input gate,
and the gates of `S` its outputs. `declaration X t θ` is `A[X, [0, t))` so cut, with the baseline `θ_A`, the input gates
committed before the run, as NCI's `Declaration`. The runtime `t_A(P)` reads the strategy, not one run
(`Accounting.Actual.runtime`).
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
  /-- The cell of its leaf at which gate `g` ran. -/
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
/-- **Egress** `out_X(I)` (at the reader's tick): the width of the gates placed in `X` that some gate placed
outside `X` during `I` reads, `in(A[X̄, I]) ∩ A[X]` (an input gate reads nothing). -/
noncomputable def egress (X : Finset L) (I : Set ℕ) : ℕ :=
  A.width (univ.filter fun g => A.x g ∈ X ∧ ∃ h, A.x h ∉ X ∧ A.t h ∈ I ∧ g ∈ A.reads h)

open Classical in
/-- **State** `mem_X(t)`: the width of the gates placed in `X` that `A[X, [t, ∞))` reads from outside itself. -/
noncomputable def state (X : Finset L) (t : ℕ) : ℕ :=
  A.width ((A.into (A.region X (Set.Ici t))).filter fun a => A.x a ∈ X)

open Classical in
/-- **The state in `G`**: the width of the gates of `G` placed in `X` that `A[X, [t, ∞))` reads from outside itself,
the part of `mem_X(t)` in `G`. -/
noncomputable def stateIn (X : Finset L) (t : ℕ) (G : Finset (Fin A.N)) : ℕ :=
  A.width (((A.into (A.region X (Set.Ici t))).filter fun a => A.x a ∈ X) ∩ G)

/-- **Wires run forward in time inside `X`**: a computed gate placed in `X` is read by gates placed in `X` only at its
tick or later. -/
def Forward (X : Finset L) : Prop :=
  ∀ g, ∀ a ∈ A.reads g, A.isInput a = false → A.x a ∈ X → A.x g ∈ X → A.t a ≤ A.t g

/-- **Faithfulness**: each computed gate's value is its operation on the values `val` holds. A run's field
(`Actual.Run.faithful`), named so a property over runs takes it as what it is, the physical fact that a gate computes
its operation. -/
def Faithful (val : A.Transcript) : Prop :=
  ∀ g, A.isInput g = false → val g = A.op g val

/-- **Holding**: `held y t` is the gates the leaf `y` holds at the start of tick `t`. A held gate is `y`'s own
and an input or placed before `t`; `y`'s inputs are held at tick 0; and a gate of `y` that `y` reads at a later tick is
held at `y` at the start of every tick from the one after its placement (from 0 for an input) to the reading one. -/
def Holding (held : L → ℕ → Finset (Fin A.N)) : Prop :=
  (∀ y t, ∀ a ∈ held y t, A.x a = y ∧ (A.isInput a = true ∨ A.t a < t)) ∧
  (∀ a, A.isInput a = true → a ∈ held (A.x a) 0) ∧
  ∀ g, ∀ a ∈ A.reads g, A.x a = A.x g → ∀ s, (A.isInput a = true ∨ A.t a < s) → s ≤ A.t g → a ∈ held (A.x g) s

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

/-- **Cuts of `[0, t)`**: ticks `0 = τ 0 ≤ τ 1 ≤ … ≤ τ J = t`, cutting it into the `J` windows `[τ j, τ (j + 1))`. -/
structure Cuts (t : ℕ) where
  J : ℕ
  τ : Fin (J + 1) → ℕ
  mono : Monotone τ
  zero : τ 0 = 0
  last : τ (Fin.last J) = t

end Accounting
