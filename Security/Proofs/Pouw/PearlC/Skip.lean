import Definitions.Pouw.PearlC.Skip

/-! The proof-only part of `Definitions.Pouw.PearlC.Skip`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Skip

open Pouw.Fp8Atom Pouw.PearlC

variable {m n k : Nat}

/-- **The unit's chain debit**: the flagged atoms `t < k/32` over the unit's `m·n` words, each at `c`, the chain's
credit per atom; `c · Forming.flagCount k flag`. -/
def chainDebit (k : ℕ) (c : ℚ) (flag : Fin m → Fin n → ℕ → Bool) : ℚ :=
  c * ∑ i, ∑ j, ((Finset.range (k / 32)).filter fun t => flag i j t = true).card

/-- The needed elements of `A′`. -/
def neededSet (flag : Fin m → Fin n → ℕ → Bool) : Finset (Fin m × Fin k) :=
  Finset.univ.filter fun p => needed flag p.1 p.2

end Pouw.PearlC.Skip
