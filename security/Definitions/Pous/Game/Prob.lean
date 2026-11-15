module

public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod

/-!
# Bit strings and uniform probability

Every random quantity in the trusted layer is uniform over a finite type: the ideal primitive, the encoder's
coins and the challenge indices. `pr P` is the probability that a uniform `x` satisfies `P`: Core's uniform probability
`Verity.Game.prCoin`, the fraction of the type on which `P` holds. Adversaries are deterministic functions. A randomized
adversary is an average of deterministic ones, so every bound that holds for all deterministic adversaries holds for it
too.
-/

@[expose] public section

namespace Pous

open scoped ENNReal

/-- A string of `n` bits. Every size bound (state, compressed string, `W`, `pp`, a block) is a length in bits. -/
abbrev Bits (n : ℕ) : Type := Fin n → Bool

/-- The probability that a uniform `x : α` satisfies `P` (`Verity.Game.prCoin`): `|{x | P x}| / |α|`. -/
noncomputable abbrev pr {α : Type} [Fintype α] (P : α → Prop) : ℝ≥0∞ :=
  Verity.Game.prCoin P

end Pous
