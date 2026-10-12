import Flock.RecVStar
import Flock.Record

/-!
# live-verifier at the inner session: the coin server's records of it

Assumption (`live-verifier` at the inner session): the inner transcript and the session record the verifier of record
reads (`--rec-inner-transcript`, `--rec-inner-session`) are the coin server's records of the live session. Lean reads
them only as the verifier does (`heldRow`, `heldCoins`, `Record.decode`), and nothing the verifier checks ties them to
the recursive game's history: the event constrains the records, the bound is over the history. `InnerRecordCustody`
says that at every history the records the coin server writes there (`wr t`) hold that history's commitments and
coins, where a map `InnerRecordMap` says what each round fixes in them.

The records are read through `held`, the verifier's hold of the inner transcript (`Recursion.innerCommitsAt` against
R's registration, at the guarantee), so this module reads only the executable: `Flock.Premises` abbreviates it.
-/

namespace FlockSoundness.Assumptions

open Flock

/-- **The `b ‖ c` the verifier holds at a registered value's row**: `RecVStar.fixedRow` at its hold of the inner
transcript `cs`, by the value's prefix, at the inner shape's sparse entry; none where it holds none. -/
def heldRow (cs : Except String RecVStar.InnerCommits) (sparse : Option RecSparse.Shape) (value : String) (x : Nat) :
    Option (List UInt8) :=
  match cs with
  | .ok cs => match RecVStar.fixedRow cs value x sparse with
    | some (.ok (_, c)) => some c.toList
    | _ => none
  | .error _ => none

/-- **The coins the verifier holds of a stream's round**: round `k` of stream `s` in its hold of the inner transcript
`cs` (`InnerCommits.streams`: candidate 3's every stream, the table's reps' held to the session record's coins by
`innerCommits`, and `sparse/rep<r>` and `stacked/rep0`, which check 5 reads as `RecCoins.C3.coins`); none where it holds
none. -/
def heldCoins (cs : Except String RecVStar.InnerCommits) (s : String) (k : Nat) : Option (Array F128) :=
  match cs with
  | .ok cs => (cs.stream? s).bind (·.coins[k]?)
  | .error _ => none

/-- **What a history's rounds fix in the inner session's records**: row `x` of registered value `v` holds round `i`'s
commitment (`sel i v x`) as its `b ‖ c` (`bc`), round `k` of stream `s` holds round `i`'s coin (`selC i s k`) as its
coins (`coin`), and the first round's coin is the session's link points (`points`). -/
structure InnerRecordMap (Cm Coin : Type) where
  sel : Nat → String → Nat → Prop
  bc : Cm → String → Nat → List UInt8
  selC : Nat → String → Nat → Prop
  coin : Coin → String → Nat → Array F128
  points : Coin → Array (Array F128)

/-- **The coin server's records of the inner session are the live history's** (`live-verifier` at the inner session):
at every history `t`, the inner transcript and session record the coin server writes there (`wr t`), as the verifier
holds them (`held`), hold

* at every row `x` of a value `v` that round `i` fixes (`m.sel i v x`), the `b ‖ c` of `t`'s round-`i` commitment;
* at every round `k` of a stream `s` that round `i` draws (`m.selC i s k`), `t`'s round-`i` coin's (the link points,
  `{table}/rep<r>`, `sparse/rep<r>` and `stacked/rep0`'s rounds, which check 5 and V*'s registered values read);
* in the session record, at whatever session parameters it decodes under, `t`'s first coin as its link points. -/
def InnerRecordCustody {Cm Coin : Type} (wr : List (Cm × Coin) → Lean.Json × String)
    (held : Lean.Json × String → Except String RecVStar.InnerCommits) (sparse : Option RecSparse.Shape)
    (m : InnerRecordMap Cm Coin) : Prop :=
  ∀ t : List (Cm × Coin),
    (∀ i e, t[i]? = some e →
      (∀ v x, m.sel i v x → heldRow (held (wr t)) sparse v x = some (m.bc e.1 v x)) ∧
      ∀ s k, m.selC i s k → heldCoins (held (wr t)) s k = some (m.coin e.2 s k)) ∧
    ∀ e, t[0]? = some e → ∀ spec sess, Record.decode spec (wr t).2 = .ok sess → sess.points = m.points e.2

end FlockSoundness.Assumptions
