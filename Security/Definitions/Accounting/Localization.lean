import Definitions.Accounting.Actual
import Mathlib.Logic.Relation

/-!
# Reach (trusted)

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1), section 6, on one actual circuit. A gate is
*downstream* of a set `C` of gates (`downstream C`) when a path of wires runs to it from a gate of `C`. A challenge `C`
*reaches the prover through `X` during `W`* (`Reaches C X Y W`, with `Y` the rest of the prover and the verifier's
leaves the ones in neither) when, in `W`, the rest reads no gate of `C` and nothing downstream of `C` from the verifier.
`X` *sends the rest nothing of `C`* during `W` (`SendsIndependent C X Y W`) when no gate the rest reads from `X` in `W` is
downstream of `C`: section 6's `e = 0`, with dependence read off the wires (a gate's value is a function of the values it
reads and of nothing else). Localization is `Accounting.Guarantees.Localization`.
-/

namespace Accounting.Placed

open Finset

variable {L : Type*} (A : Placed L)

open Classical in
/-- **Downstream of `C`**: the gates that a path of wires from a gate of `C` reaches, `C`'s among them. -/
noncomputable def downstream (C : Finset (Fin A.N)) : Finset (Fin A.N) :=
  univ.filter fun g => ∃ c ∈ C, Relation.ReflTransGen (fun a g => a ∈ A.reads g) c g

/-- **`C` reaches the prover through `X` during `W`**, `Y` the rest of the prover: every gate of `A[Y, W]` reads no gate
of `C`, and of the verifier's (placed in neither `X` nor `Y`) none downstream of `C`. -/
def Reaches (C : Finset (Fin A.N)) (X Y : Finset L) (W : Set ℕ) : Prop :=
  ∀ g ∈ A.region Y W, ∀ a ∈ A.reads g, a ∉ C ∧ (A.x a ∉ X → A.x a ∉ Y → a ∉ A.downstream C)

/-- **`X` sends the rest `Y` of the prover nothing of `C` during `W`**: no gate placed in `X` that a gate of `A[Y, W]`
reads is downstream of `C`. -/
def SendsIndependent (C : Finset (Fin A.N)) (X Y : Finset L) (W : Set ℕ) : Prop :=
  ∀ g ∈ A.region Y W, ∀ a ∈ A.reads g, A.x a ∈ X → a ∉ A.downstream C

end Accounting.Placed
