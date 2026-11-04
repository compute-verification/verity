import Definitions.Core.Circuit

/-!
# What the developer declares

A declaration `(D, θ)` is the declared circuit `D` and its registered inputs `θ`: the input gates whose values the
developer committed beforehand (its models, code and data in storage). The builder `B` is registered
(`Verity.Registration`), and the declared circuit is `D = B(a_1) ‖ … ‖ B(a_n)`, an execution block `B(a_i)` for each
declared advice `a_i`. The values of every other gate are fresh input: the remaining input gates hold data such as
prompts, and every computed gate is computed after the commitment. Per deployment, the workload's Program is the one
circuit of record.

Implementation carries over to declarations when it keeps fresh input fresh: an input gate of `t` outside `t.θ` is an
input gate of `d` outside `d.θ`. Containment carries over as it is between the circuits, and says nothing about `θ`:
that a containment keeps fresh input fresh is a premise about the protocol's rule for `θ`
(`Nci.Assumptions.ThetaRegistered`), which each guarantee that needs it takes for its containment.
-/

namespace Verity

/-- A declaration: the declared circuit `D = B(a_1) ‖ … ‖ B(a_n)`, the registered builder `B`'s execution blocks, and
its registered inputs `θ`, a set of its input gates. -/
structure Declaration where
  D : Circuit
  θ : Finset (Fin D.N)
  θ_input : ∀ g ∈ θ, D.isInput g = true

namespace Declaration

/-- `t ≼ d`: `d`'s circuit implements `t`'s, through an implementation that sends no input gate of `t` outside `t.θ`
into `d.θ`. -/
def Implements (t d : Declaration) : Prop :=
  ∃ I : Implementation t.D d.D, ∀ g, t.D.isInput g = true → I.inp g ∈ d.θ → g ∈ t.θ

/-- `d` contains `t`: `d`'s circuit contains `t`'s. -/
def Contains (t d : Declaration) : Prop := Nonempty (Containment t.D d.D)

end Declaration

end Verity
