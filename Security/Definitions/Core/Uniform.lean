module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Card

/-!
# Uniform probability over a finite type

`Claims.Pr p` is the fraction of a finite type on which `p` holds, a real number in `[0, 1]`; it is `0` on an empty
type. Its lemmas are in `Proofs/Flock/Claims/Prob.lean`. It imports only the reals and finite cardinalities, so an area
can state its probability as Core's without importing the interaction model (`Definitions/Core/Interaction.lean`).
-/

@[expose] public section

namespace Claims

open Classical in
noncomputable def Pr {α : Type} [Fintype α] (p : α → Prop) : ℝ :=
  ((Finset.univ.filter p).card : ℝ) / Fintype.card α

end Claims
