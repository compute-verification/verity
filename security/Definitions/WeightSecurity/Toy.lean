module

public import Definitions.WeightSecurity

/-!
# Model weight security's toy instance

The instance `WeightSecurity.Guarantees.Theft`'s witnesses are stated at. The weights are two uniform bits, `θ = ω`,
and the served function answers every query with the first: honest use reveals one bit and hides the other, so one
query steals the weights (outputs both bits) with probability `1/2`. The honest datacenter answers honestly, and its
user outputs the bit it was answered and a guess. The cheating one appends the second bit to each answer, one covert
bit, and the thief outputs the bit it was answered and the bit appended to it.
-/

@[expose] public section

namespace WeightSecurity.Toy

/-- The served function: every query is answered with the first weight bit. -/
def C (θ : Bool × Bool) (_ : Unit) : Bool := θ.1

/-- The weights are the coins. -/
def gen : Bool × Bool → Bool × Bool := id

/-- Stealing is outputting the weights. -/
def Stolen (θ θ' : Bool × Bool) : Prop := θ' = θ

/-- Every honest-use party, on one coin: the class of all parties. -/
def anyone : Set (Outsider Unit Unit Bool (Bool × Bool)) := Set.univ

/-- The honest datacenter. -/
def honestDC (θ : Bool × Bool) (_ : Unit) : List (Unit × Bool) → Unit → Bool := honest C θ

/-- The channel of honest use, with one covert string: the query as asked, the answer as given. -/
def plain : Channel Unit Unit Bool Bool where
  S := Unit
  ask _ _ _ x := x
  tell _ _ _ _ y := y

/-- A user of the honest datacenter: it asks once, and outputs the answer and a guess of `false` for the other bit. -/
def user : Outsider Unit Unit Bool (Bool × Bool) where
  query _ _ := ()
  out _ h := ((h.headD ((), false)).2, false)

/-- The cheating datacenter: each answer, with the second weight bit appended. -/
def cheatDC (θ : Bool × Bool) (_ : Unit) : List (Unit × (Bool × Bool)) → Unit → Bool × Bool := fun _ x => (C θ x, θ.2)

/-- The thief: it outputs the last answer and the bit appended to it. -/
def thief : Outsider Unit Unit (Bool × Bool) (Bool × Bool) where
  query _ _ := ()
  out _ h := (h.headD ((), (false, false))).2

/-- The channel that appends nothing: each answer with `false` appended. -/
def silent : Channel Unit Unit Bool (Bool × Bool) where
  S := Unit
  ask _ _ _ x := x
  tell _ _ _ _ y := (y, false)

/-- The one-bit channel: each answer with the covert bit appended. -/
def oneBit : Channel Unit Unit Bool (Bool × Bool) where
  S := Bool
  ask _ _ _ x := x
  tell s _ _ _ y := (y, s)

end WeightSecurity.Toy
