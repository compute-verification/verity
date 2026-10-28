import Definitions.Core.Circuit

/-!
# What the developer declares

A declaration `(D, θ)` is the circuit `D` the developer registers and its `θ`: the input gates whose values it
committed beforehand (its models, code and data in storage). The values of every other gate are novel information: the
remaining input gates hold data such as prompts, and every computed gate is computed after the commitment. Per
deployment, the workload's Program is the one circuit of record.

Implementation carries over to declarations when it keeps novel information novel: an input gate of `t` outside `t`'s
θ is an input gate of `d` outside `d`'s. Containment carries over as it is between the circuits, and says nothing
about θ: that a containment keeps novel information novel is a premise about the protocol's θ rule
(`Nci.Assumptions.ThetaRegistered`), which each guarantee that needs it takes for its containment.
-/

namespace Verity

/-- A declaration: the circuit `D` and its `θ`, a set of its input gates. -/
structure Declaration where
  D : Circuit
  θ : Finset (Fin D.N)
  θ_input : ∀ g ∈ θ, D.isInput g = true

namespace Declaration

/-- `t ≼ d`: `d`'s circuit implements `t`'s, through an implementation that sends no novel input gate of `t` into
`d`'s θ. -/
def Implements (t d : Declaration) : Prop :=
  ∃ I : Implementation t.D d.D, ∀ g, t.D.isInput g = true → I.inp g ∈ d.θ → g ∈ t.θ

/-- `d` contains `t`: `d`'s circuit contains `t`'s. -/
def Contains (t d : Declaration) : Prop := Nonempty (Containment t.D d.D)

end Declaration

end Verity
