import Definitions.Core.Circuit

/-!
# What the developer declares

A declaration `(D, θ)` is the circuit `D` the developer registers and its baseline `θ`: the input gates whose values it
committed beforehand (its models, code and data in storage). The values of every other gate are novel information: the
remaining input gates hold data such as prompts, and every computed gate is computed after the commitment. Per
deployment, the workload's Program is the one circuit of record.

Implementation and containment carry over to declarations as they are between their circuits, and say nothing about the
baselines. That a containment keeps novel information novel, sending no novel input of the contained declaration into the
containing one's baseline, is a premise about the protocol's baseline rule (`Nci.Assumptions.BaselineRegistered`), which
each guarantee that needs it takes for its containment.
-/

namespace Verity

/-- A declaration: the circuit `D` and its baseline `θ`, a set of its input gates. -/
structure Declaration where
  D : Circuit
  θ : Finset (Fin D.N)
  θ_input : ∀ g ∈ θ, D.isInput g = true

namespace Declaration

/-- `t ≼ d`: `d`'s circuit implements `t`'s. -/
def Implements (t d : Declaration) : Prop := Nonempty (Implementation t.D d.D)

/-- `d` contains `t`: `d`'s circuit contains `t`'s. -/
def Contains (t d : Declaration) : Prop := Nonempty (Containment t.D d.D)

end Declaration

end Verity
