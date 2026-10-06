import Definitions.Flock.Game.Basic
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Fintype.Prod
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-! The definitions of `Proofs.Flock.Soundness.Game.Expect` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness

open scoped ENNReal

namespace Game

variable {α : Type}

/-- The expectation of `f` over the verifier's uniform coins, against the strategy `s`. -/
noncomputable def expect (f : α → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => f a
  | .send _ k, s => expect f (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, expect f (k c) (s c)) / (@Fintype.card C iF)

end Game

namespace Rewinding

open Game Finset

variable {α : Type} {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]

variable (op : α → P → Option V)

end Rewinding

end FlockSoundness
