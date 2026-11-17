module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Card
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Data.Fintype.Sigma
public import Definitions.Core.Uniform

/-! Uniform probability over a finite type, Core's `Claims.Pr` under PoUW's name, against deterministic adversaries, which covers randomized ones (`Pouw.Sanity.randomized_of_deterministic`). -/

@[expose] public section

namespace Pouw

/-- The probability that a uniform `x : α` satisfies `P`: `|{x | P x}| / |α|` (`Claims.Pr`). -/
noncomputable abbrev pr {α : Type} [Fintype α] (P : α → Prop) : ℝ :=
  Claims.Pr P

end Pouw
