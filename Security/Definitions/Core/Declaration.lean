import Definitions.Core.Circuit

/-!
# What the developer declares

A declaration `(D, θ)` is the circuit `D` the developer registers and its baseline `θ`: the input gates whose values it
committed beforehand (its models, code and data in storage). The values of every other gate are novel information: the
remaining input gates hold data such as prompts, and every computed gate is computed after the commitment. Per
deployment, the workload's Program is the one circuit of record.

Implementation and containment carry over to declarations when they keep novel information novel: an input gate of `t`
outside `t`'s baseline is an input gate of `d` outside `d`'s.
-/

namespace Verity

/-- A declaration: the circuit `D` and its baseline `θ`, a set of its input gates. -/
structure Declaration where
  D : Circuit
  θ : Finset (Fin D.N)
  θ_input : ∀ g ∈ θ, D.isInput g = true

namespace Declaration

/-- `t ≼ d`: `d`'s circuit implements `t`'s, through an implementation that sends no novel input gate of `t` into
`d`'s baseline. -/
def Implements (t d : Declaration) : Prop :=
  ∃ I : Implementation t.D d.D, ∀ g, t.D.isInput g = true → I.inp g ∈ d.θ → g ∈ t.θ

/-- `d` contains `t`: `d`'s circuit contains `t`'s, through a containment that sends no novel input gate of `t` into
`d`'s baseline. -/
def Contains (t d : Declaration) : Prop :=
  ∃ M : Containment t.D d.D, ∀ g, t.D.isInput g = true → M.emb g ∈ d.θ → g ∈ t.θ

end Declaration

end Verity
