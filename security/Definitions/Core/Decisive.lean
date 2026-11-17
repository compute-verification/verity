module

public import Definitions.Core.Game
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Decisive games and unambiguous batches; decisiveness is existential and says nothing of how often any prover is
accepted. -/

@[expose] public section

namespace Verity.Game

variable {α : Type}

def Productive : Game α → Prop
  | .ret _ => True
  | .send M k => Nonempty M ∧ ∀ m, Productive (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Productive (k c)

/-- Some run of `g`, for some messages and some coins, ends with `a`. -/
def Reaches : Game α → α → Prop
  | .ret b, a => b = a
  | .send _ k, a => ∃ m, Reaches (k m) a
  | @Game.coin _ _ _ _ k, a => ∃ c, Reaches (k c) a

def Decisive (g : Game α) : Prop :=
  Productive g ∧ ∃ a b, a ≠ b ∧ Reaches g a ∧ Reaches g b

def Unambiguous (L : List (Game α)) : Prop :=
  ∀ (pre : List (Game α)) (g : Game α) (post : List (Game α)), batch (pre ++ g :: post) = batch L →
    pre ++ g :: post = L

end Verity.Game
