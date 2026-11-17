module

public import Definitions.Registration.Physical

/-! Ingress by source, `in_X(I; Y)`, and the capacity of the links from `Y`, which B3 reads. -/

@[expose] public section

namespace ResourceAccounting

open Verity Finset

namespace Placed

variable {L : Type*} (A : Placed L)

open Classical in
noncomputable def ingressFrom (X Y : Finset L) (I : Set ℕ) : ℕ :=
  A.width ((A.into (A.region X I)).filter fun a => A.x a ∉ X ∧ A.x a ∈ Y)

end Placed

section Links

variable {L E : Type*} [Fintype E]

open Classical in
noncomputable def linkCapacityFrom (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) : ℕ :=
  ∑ e ∈ (boundary ends X).filter (fun e => ((ends e).1 ∉ X ∧ (ends e).1 ∈ Y) ∨ ((ends e).2 ∉ X ∧ (ends e).2 ∈ Y)), β e

end Links

end ResourceAccounting
