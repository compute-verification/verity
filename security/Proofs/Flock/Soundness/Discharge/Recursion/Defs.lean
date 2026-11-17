module

import Proofs.Flock.Soundness.Game
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.Defs`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Game

variable {α : Type}

/-- **An outcome the strategy reaches**: `s` plays `g` to `a` at some coins. -/
def Outcome : (g : Game α) → Strategy g → α → Prop
  | .ret b, _, a => b = a
  | .send _ k, s, a => Outcome (k s.1) s.2 a
  | @Game.coin _ _ _ _ k, s, a => ∃ c, Outcome (k c) (s c) a

end FlockSoundness.Game

namespace FlockSoundness.Discharge.Recursion

open Game Audit

open scoped ENNReal

/-- **An inner protocol**: an interactive verifier with `r` rounds of public coins, messages committed before coins. In
each round the prover sends a message of type `Msg` and the verifier flips a uniform coin from `Coin`; after the last
round it decides `accepts x t` from the statement `x` and the transcript `t`, first round first. `holds x` is the
statement's truth. -/
structure Inner : Type 1 where
  Stmt : Type
  Msg : Type
  Coin : Type
  [fintype : Fintype Coin]
  [nonempty : Nonempty Coin]
  r : ℕ
  accepts : Stmt → List (Msg × Coin) → Prop
  holds : Stmt → Prop

attribute [instance] Inner.fintype Inner.nonempty

end FlockSoundness.Discharge.Recursion
