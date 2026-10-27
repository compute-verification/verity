import Definitions.Accounting.Actual
import Definitions.Core.Unroll
import Definitions.Core.Formalism

/-!
# Fixed machine dynamics

An actual circuit `A` (`Placed`) is **the fixed machine `H` run `T` steps** (`FixedMachine H S T A`) when its circuit is
`H.unroll S T` with its gates renumbered (`Verity.Circuit.Iso`): one step `H`, fixed before the run, applied `T` times
along its state wires `S`. Control, decoding, generation, selection and retries are all inside `H`; nothing of the run
is a circuit built for one candidate, so no smaller candidate-specific circuit is a fixed machine of `H`. Its work is a
gate count, `T · |H|` (`Accounting.Guarantees.FixedMachine.GateCount`).

A proof block of `A` is the region of one location over the whole run, `A[X, [0, ∞))`. What it starts from is its
**initial information** (`InitialBudget`): a set `a` of input gates of `A`, and every input gate placed in `X` is
registered or in `a`; its budget is `D` bits of `X`'s state at tick `0` in `a`, `stateIn X 0 a ≤ D`. A proof block's
extracted string is **described within its budget** (`BlockDescribed`) when its local description length is at most
`D` plus the bits read into `X` over the run (`ingress X [0, ∞)`), the proof block's index and the evaluator's length.
-/

namespace Accounting

open Verity Finset

variable {L : Type*}

/-- **`A` is the fixed machine `H` run `T` steps** along its state wires `S`: `A`'s circuit is `H.unroll S T` with its
gates renumbered. -/
def FixedMachine (H : Circuit) (S : H.StateWires) (T : ℕ) (A : Placed L) : Prop :=
  Nonempty (Circuit.Iso (H.unroll S T) A.toCircuit)

/-- **Initial information within a budget**: `a` is a set of input gates of `A`; every input gate placed in the location
`X` is registered (in `θ`) or in `a`; and the part of `X`'s state at tick `0` in `a` is at most `D` bits. -/
def InitialBudget (A : Placed L) (θ a : Finset (Fin A.N)) (X : Finset L) (D : ℕ) : Prop :=
  (∀ g ∈ a, A.isInput g = true) ∧ (∀ g, A.isInput g = true → A.x g ∈ X → g ∈ θ ∨ g ∈ a) ∧ A.stateIn X 0 a ≤ D

/-- **A proof block's string described within its budget**: `s` has local description length, at the decoder `M` with
registered information `r` and time `t`, at most `X`'s initial budget `D`, plus the bits read into `X` over the run,
plus the `⌈log₂ B⌉` bits of a proof block's index among `B` and the evaluator's `a_blk` bits. -/
def BlockDescribed {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (A : Placed L) (X : Finset L) (D B a_blk : ℕ)
    (s : List Bool) : Prop :=
  LDL M t r s ≤ (D + A.ingress X (Set.Ici 0) + Nat.clog 2 B + a_blk : ℕ)

end Accounting
