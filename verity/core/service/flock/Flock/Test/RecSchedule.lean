/-
`RecVStar.checkSchedule` counts the coin server's rounds. At the plain session's inner shape (fixture
`rec-coins/16-16-8-n64`: m 27, k_log 22, 8 claims, whose two reps' 96 rounds each are `inner-honest`'s), a
record listing the schedule's 192 rounds is on it, and one listing 194 is refused before any round is read. 194 is
the inner verifier's `k` at that statement, `2 + rounds` (`FlockDeploy.V`): the link round 0, the reps' rounds and
the last round. So a record that lists every inner round, as the proofs' first transcript model did, sets up no V*
statement: `RecSetup.commits` runs `checkSchedule` on the staged transcript, and a premise read at that model held of
no staged statement. `RecDeploy.recTr` now lists the rounds the coin server records, the reps' alone, with round 0
under the record's link.
-/
import Flock.RecVStar

namespace Flock.Test.RecSchedule

open Lean Flock.RecVStar

def inner : Inner := { m := 27, kLog := 22, claims := 8, regions := 1 }

def hex (n : Nat) : Json := Json.str (String.ofList (List.replicate n '0'))

/-- A round of stream `s` at its round `k` with the schedule's counts `w`: `w.1` coins, tops of `w.2.1`'s node
counts, `w.2.2.1` level-0 copies, and one `c_k` where it absorbs a message. -/
def roundJ (s : String) (k : Nat) (w : SchedRound) : Json :=
  Json.mkObj [("stream", s), ("k", toJson k),
    ("tops", Json.arr (w.2.1.map fun n => Json.arr (Array.replicate n (hex 256)))), ("level0", toJson w.2.2.1),
    ("commits", Json.arr (if w.2.2.2 then #[hex 256] else #[])), ("n", toJson w.1),
    ("coins", Json.arr (Array.replicate w.1 (hex 32)))]

/-- The record of a session on `inner`'s schedule over the table `circuit`, `pre` before its rounds and `post`
after. -/
def recordJ (pre post : Array Json := #[]) : Option Json := do
  let want ← scheduleOf inner "circuit"
  let rounds := want.flatMap fun (s, _, _, rs) => rs.mapIdx fun k w => roundJ s k w
  let streams := want.toList.map fun (s, p, a, rs) =>
    (s, Json.mkObj [("parent", p), ("at", toJson a), ("rounds", toJson rs.size)])
  return Json.mkObj [("rounds", Json.arr (pre ++ rounds ++ post)),
    ("hello", Json.mkObj [("tables", toJson #["circuit"])]), ("streams", Json.mkObj streams)]

def verdict (pre post : Array Json := #[]) : Option (Except String Unit) :=
  (recordJ pre post).map (checkSchedule inner)

def first : Json := roundJ "circuit/rep0" 0 (6, #[], 1, false)
def last : Json := roundJ "circuit/rep1" 96 (1, #[], 0, true)

#guard (scheduleOf inner "circuit").map (·.map fun (s, _, _, rs) => (s, rs.size)) ==
  some #[("circuit/rep0", 96), ("circuit/rep1", 96)]

#guard verdict.any (·.toOption.isSome)

#guard match verdict #[first] #[last] with
  | some (.error e) => (e.splitOn "194 rounds, and the schedule's 192").length == 2
  | _ => false

#guard (verdict #[first]).any (·.toOption.isNone) && (verdict #[] #[last]).any (·.toOption.isNone)

end Flock.Test.RecSchedule
