module

public import Definitions.Core.Game
public import Mathlib.Data.Fintype.Pi
public import Mathlib.Data.Fintype.Prod

/-!
# A play's coins as one uniform coin

At a fixed strategy a play's coins are a tree: each coin's value picks the play after it. `Tape g s` holds, at each
coin, a value and the coins of the play after every value: a finite type (`Tape.instFintype`). A finder or a forger
built from a strategy samples one tape and reads the play on it (`Tape.out`). That a uniform tape gives the play's
output the sampling design `prob` reads is proved in `Proofs.Core.Game.Tape`.
-/

@[expose] public section

namespace Verity.Game

variable {α : Type}

/-- **The coins of a play of `g` at the strategy `s`**: at each coin a value, and the coins of the play after every
value. -/
def Tape : (g : Game α) → Strategy g → Type
  | .ret _, _ => Unit
  | .send _ k, s => Tape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → Tape (k c) (s c))

/-- The play's output on the tape `t`: at each coin the tape's value, and the tape after it. -/
def Tape.out : (g : Game α) → (s : Strategy g) → Tape g s → α
  | .ret a, _, _ => a
  | .send _ k, s, t => Tape.out (k s.1) s.2 t
  | @Game.coin _ C _ _ k, s, t =>
    let t' : C × ((c : C) → Tape (k c) (s c)) := t
    Tape.out (k t'.1) (s t'.1) (t'.2 t'.1)

open Classical in
/-- **A play's tapes are finitely many**: at each coin, the coin's values and the tapes of the play after each. -/
noncomputable instance Tape.instFintype : (g : Game α) → (s : Strategy g) → Fintype (Tape g s)
  | .ret _, _ => inferInstanceAs (Fintype Unit)
  | .send _ k, s => Tape.instFintype (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
    haveI := fun c => Tape.instFintype (k c) (s c)
    inferInstanceAs (Fintype (C × ((c : C) → Tape (k c) (s c))))

end Verity.Game
