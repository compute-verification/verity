import Definitions.Accounting.Actual
import Mathlib.Logic.Relation

/-!
# Reach

The actual-circuit model's reach, on one actual circuit. A gate is *downstream* of a set `C` of gates (`downstream C`)
when a path of wires runs to it from a gate of `C`. A challenge `C` *reaches the prover through `X` during `W`*
(`Reaches C X Y W`, with `Y` the rest of the prover and the verifier's leaves the ones in neither) when, in `W`, the
rest reads no gate of `C` and nothing downstream of `C` from the verifier. `SendsIndependent C X Y W`: no wire from `X`
to `Y` during `W` carries a gate downstream of `C`. That is the model's `e = 0` only when, in addition, the gates `X`
sends and their ticks don't depend on `C`, which one circuit can't say: the per-run lemma is
`Accounting.SecurityProofs.Actual.localization_wires`, and localization is `Accounting.Guarantees.Localization`, over
strategies.
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

/-- No wire from `X` to the rest `Y` of the prover during `W` carries a gate downstream of `C`: no gate placed in `X`
that a gate of `A[Y, W]` reads is downstream of `C`. -/
def SendsIndependent (C : Finset (Fin A.N)) (X Y : Finset L) (W : Set ℕ) : Prop :=
  ∀ g ∈ A.region Y W, ∀ a ∈ A.reads g, A.x a ∈ X → a ∉ A.downstream C

end Accounting.Placed
