module

public import Flock.RecVStar

/-! `live-verifier` at the inner session: the challenger's live-coin part's records of it, which nothing the verifier checks ties to the recursive game's history. -/

@[expose] public section

namespace FlockSoundness.Assumptions

open Flock

def heldRow (cs : Except String RecVStar.InnerCommits) (sparse : Option RecSparse.Shape) (value : String) (x : Nat) :
    Option (List UInt8) :=
  match cs with
  | .ok cs => match RecVStar.fixedRow cs value x sparse with
    | some (.ok (_, c)) => some c.toList
    | _ => none
  | .error _ => none

def heldCoins (cs : Except String RecVStar.InnerCommits) (s : String) (k : Nat) : Option (Array F128) :=
  match cs with
  | .ok cs => (cs.stream? s).bind (·.coins[k]?)
  | .error _ => none

/-- A stream round can hold several rounds' coins, the last of them round `i`'s: a sparse phase's ζ_c is one
round of six coins and six rounds of one. -/
structure InnerRecordMap (Cm Coin : Type) where
  sel : Nat → String → Nat → Prop
  bc : Cm → String → Nat → List UInt8
  selC : Nat → String → Nat → Prop
  coin : List Coin → String → Nat → Array F128
  points : Coin → Array (Array F128)

/-- `live-verifier` at the inner session. -/
def InnerRecordCustody {Cm Coin : Type} (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (sparse : Option RecSparse.Shape)
    (m : InnerRecordMap Cm Coin) : Prop :=
  ∀ t : List (Cm × Coin),
    (∀ i e, t[i]? = some e →
      (∀ v x, m.sel i v x → heldRow (held (wr t)) sparse v x = some (m.bc e.1 v x)) ∧
      ∀ s k, m.selC i s k → heldCoins (held (wr t)) s k = some (m.coin ((t.take (i + 1)).map Prod.snd) s k)) ∧
    ∀ e, t[0]? = some e → ∀ spec sess, Record.decode spec (wr t).2 = .ok sess → sess.points = m.points e.2

end FlockSoundness.Assumptions
