module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Fintype.Card
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Data.Fintype.Sigma
public import Definitions.Core.Uniform

/-!
# Uniform probability (trusted)

Every random quantity is uniform over a finite type: the oracle `H : Q → R`, the salt, and the verifier's sample.
`pr P` is the fraction of the type on which `P` holds, a real number in `[0, 1]`; it is `0` on an empty type. It is
Core's uniform probability `Claims.Pr` under PoUW's name.

Adversaries are deterministic functions. A randomized adversary is an average of deterministic ones (fix its coins),
and each coin value gives a deterministic adversary within the same budget, so every bound proved for all
deterministic adversaries holds for randomized ones too (`Pouw.Sanity.randomized_of_deterministic`).
-/

@[expose] public section

namespace Pouw

/-- The probability that a uniform `x : α` satisfies `P`: `|{x | P x}| / |α|` (`Claims.Pr`). -/
noncomputable abbrev pr {α : Type} [Fintype α] (P : α → Prop) : ℝ :=
  Claims.Pr P

end Pouw
