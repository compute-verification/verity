module

import Proofs.Flock.Soundness.Game
public import Proofs.Flock.Soundness.StrictCR
import Mathlib.Combinatorics.Matroid.Init

/-! # What V*'s bridge obligation reads of `Recursive.Multi`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game

open scoped ENNReal

section Seq

variable {α : Type}

/-- **`g` with the prover's messages fixed by `τ`**: only the coins remain. -/
def play : (g : Game α) → Strategy g → Game α
  | .ret a, _ => .ret a
  | .send _ k, τ => play (k τ.1) τ.2
  | @Game.coin _ C iF iN k, τ => @Game.coin _ C iF iN fun c => play (k c) (τ c)

/-- **`g` played by a strategy the prover names first**: the prover sends a name `τ`, `g` is played by `of τ`, and the
output is the name with the outcome. -/
def committed {T : Type} (g : Game α) (of : T → Strategy g) : Game (T × α) :=
  .send T fun τ => (play g (of τ)).map fun a => (τ, a)

end Seq

end FlockSoundness.Discharge.Recursion
