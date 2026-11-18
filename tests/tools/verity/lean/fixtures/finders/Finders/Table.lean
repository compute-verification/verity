import Finders.Assumptions

/-! Stand-ins of the rest of the table's names (`finder-walk.json`), under the same full names and with a finder at each
head's position, since the check fails on a name Lean doesn't hold or a position that is no finder's (`choosing.table`).
No guarantee here hands them a finder. -/

/-- The stand-in of `Claims.Finder.Pr`, its argument 1 a finder. -/
def Claims.Finder.Pr (_H : Nat → Nat) (_F : Claims.Finder) : Prop := True

/-- The stand-in of `Claims.StrictBirthday`, its argument 1 a finder. -/
def Claims.StrictBirthday (_H : Nat → Nat) (_F : Claims.Finder) : Prop := True

/-- The stand-in of a game, a finder type. -/
inductive Verity.Game (α : Type) : Type where
  | ret : α → Verity.Game α

/-- The stand-in of a game's strategy, a finder type. -/
def Verity.Game.Strategy {α : Type} (_ : Verity.Game α) : Type := Unit

/-- The stand-ins of the game-form collision premises and event, their arguments 3, 4 and 5 a game, its strategy and
its output map. -/
def FlockSoundness.Assumptions.SHA512CRExpected {D β : Type} (_H : List UInt8 → D) (g : Verity.Game β)
    (_s : Verity.Game.Strategy g) (_out : β → Option (List UInt8 × List UInt8)) (_cost : β → Nat) : Prop := True

def FlockSoundness.Assumptions.SHA512CRStrict {D β : Type} (_H : List UInt8 → D) (g : Verity.Game β)
    (_s : Verity.Game.Strategy g) (_out : β → Option (List UInt8 × List UInt8)) (_cost : β → Nat) (_q : Nat) : Prop := True

def FlockSoundness.StrictCR.FindsColl {D β : Type} (_H : List UInt8 → D) (g : Verity.Game β)
    (_s : Verity.Game.Strategy g) (_out : β → Option (List UInt8 × List UInt8)) : Prop := True

/-- The stand-in of an LMS forger: its coins, the key it forges under, the messages signed for it, its output and its
cost. -/
structure Hash.Assumptions.LmsForger where
  S : Type
  key : Nat
  signed : S → List Nat
  out : S → Option (Nat × Nat)
  cost : S → Nat

/-- The stand-in of `Hash.Assumptions.LmsUnforgeable`, its argument 4 a forger. -/
def Hash.Assumptions.LmsUnforgeable (_verify : Nat → Nat → Nat → Bool) (_ε : Nat → Nat → Nat) (_t _Q : Nat)
    (_F : Hash.Assumptions.LmsForger) : Prop := True
