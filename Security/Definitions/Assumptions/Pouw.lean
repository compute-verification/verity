module

public import Definitions.Pouw.Game.Defs

/-!
# PoUW's hashless cost models

A PoUW guarantee bounds its event together with no SHA-512 collision among the inputs the run hashed (`Pouw.Breaks`).
A model whose programs hash nothing never breaks, and the guarantees that drop the break from their event take it.
It is a named assumption under `Definitions.Assumptions`, in a module of its own that reads only PoUW's cost model, so the
public PoUW proofs import it without the rest of the named assumptions.
-/

@[expose] public section

namespace Pouw

/-- The model's programs hash nothing, so no run breaks. -/
def CostModel.Hashless {Q R S : Type} (CM : CostModel Q R S) : Prop :=
  ∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S), CM.hashed prog H U act σ s = []

end Pouw
