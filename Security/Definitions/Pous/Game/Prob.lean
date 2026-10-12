import Mathlib.Basic.ENNReal.Inv
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod

/-!
# Bit strings and uniform probability

Every random quantity in the trusted layer is uniform over a finite type: the ideal primitive, the encoder's
coins and the challenge indices. `pr P` is the probability that a uniform `x` satisfies `P`: the fraction of the
type on which `P` holds. Adversaries are deterministic functions. A randomized adversary is an average of
deterministic ones, so every bound that holds for all deterministic adversaries holds for it too.
-/

namespace Pous

open scoped ENNReal

/-- A string of `n` bits. Every size bound (state, compressed string, `W`, `pp`, a block) is a length in bits. -/
abbrev Bits (n : ℕ) : Type := Fin n → Bool

open Classical in
/-- The probability that a uniform `x : α` satisfies `P`: `|{x | P x}| / |α|`. -/
noncomputable def pr {α : Type} [Fintype α] (P : α → Prop) : ℝ≥0∞ :=
  ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card α

end Pous
