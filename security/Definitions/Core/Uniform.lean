module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Card

/-! Uniform probability over a finite type (`Claims.prUniform`, `Claims.Pr`), `0` on an empty type. -/

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
