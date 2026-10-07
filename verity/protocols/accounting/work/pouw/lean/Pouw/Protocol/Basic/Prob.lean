import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.Sigma
import Mathlib.Basic.Real.Basic

/-!
# Uniform probability (trusted)

Every random quantity is uniform over a finite type: the oracle `H : Q → R`, the salt, and the verifier's sample.
`pr P` is the fraction of the type on which `P` holds, a real number in `[0, 1]`; it is `0` on an empty type.

Adversaries are deterministic functions. A randomized adversary is an average of deterministic ones (fix its coins),
and each coin value gives a deterministic adversary within the same budget, so every bound proved for all
deterministic adversaries holds for randomized ones too (`Pouw.Sanity.randomized_of_deterministic`).
-/

namespace Pouw

open Classical in
/-- The probability that a uniform `x : α` satisfies `P`: `|{x | P x}| / |α|`. -/
noncomputable def pr {α : Type} [Fintype α] (P : α → Prop) : ℝ :=
  ((Finset.univ.filter P).card : ℝ) / Fintype.card α

end Pouw
