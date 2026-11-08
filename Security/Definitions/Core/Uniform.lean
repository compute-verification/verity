module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Card

/-!
# Uniform probability over a finite type

`Claims.prUniform p` is the fraction of a finite type, in any universe, on which `p` holds, a real number in `[0, 1]`;
it is `0` on an empty type. `Claims.Pr` is the same probability over a type in `Type`, the universe the claims state
it in, so that a statement reads `Pr` with no universe level. Its lemmas are in `Proofs/Flock/Claims/Prob.lean`. It
imports only the reals and finite cardinalities, so an area can state its probability as Core's without importing the
interaction model (`Definitions/Core/Interaction.lean`).
-/

@[expose] public section

namespace Claims

open Classical in
/-- The uniform probability of `p` over a finite type in any universe. -/
noncomputable def prUniform {α : Type*} [Fintype α] (p : α → Prop) : ℝ :=
  ((Finset.univ.filter p).card : ℝ) / Fintype.card α

/-- `prUniform` over a type in `Type`. -/
noncomputable def Pr {α : Type} [Fintype α] (p : α → Prop) : ℝ :=
  prUniform p

end Claims
