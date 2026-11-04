/-
The sampled run's Hello: at the run's round 0 the challenger
answers the prover's Hello with one keyed coin tree, `coin-tree/hm96-sha512/v2`, over every coin of the run: the selection
draw's stream (or streams) first, then each session slot's streams in session order. The verifier of record checks the
run's record with C-Flock's own checks (`Flock.CoinTree`) at run scope: R7 with the prover's nonce, the record's
`coin_tree`, the run's stream layout, and that no earlier run used the key.
-/
import Flock.CoinTree
import SampledProofs.Json

namespace SampledProofs.Hello

open Lean (Json)
open Flock (V)

/-- A session slot's stream `s` in the run's tree. -/
def slotStream (i : Nat) (s : String) : String := s!"slot/{i}/{s}"

/-- The run's streams in leaf order: the selection streams, then `slots` session slots' streams, each the session's in its
order. -/
def runStreams (selection : Array String) (slots : Nat) (session : Array String) : Array String :=
  selection ++ (Array.range slots).flatMap fun i => session.map (slotStream i)

private def stage {α} (code : String) : V α → Except (String × String) α
  | .ok a => .ok a
  | .error e => .error (code, e)

/-- The run's Hello on its record: the failure's code and reason, or the run's key. `own` is the verifier's own run
Hello (its `coins` with the nonce empty), `hello` the prover's as received, `ct` the record's `coin_tree` and `seen` the
keys of the runs verified before this one. -/
def checkRun (own : Json) (hello : String) (ct : Json) (seen : Array String) (selection : Array String) (slots : Nat)
    (session : Array String) : Except (String × String) String := do
  let nonce ← stage "hello" (Flock.CoinTree.checkHello own hello)
  let some spec := Flock.CoinTree.ofHello hello
    | throw ("hello", s!"the run's Hello names no {Flock.CoinTree.SCHEME} coins")
  if selection.isEmpty then throw ("streams", "the run has no selection stream")
  if spec.streams != runStreams selection slots session then
    throw ("streams", "the Hello's streams are not the selection streams, then each session slot's in session order")
  let key ← stage "coin_tree" (Flock.CoinTree.checkRecord spec nonce ct)
  stage "fresh" (Flock.CoinTree.checkFresh seen key)
  return key

private def strings (j : Json) (what : String) : V (Array String) := do
  let .arr xs := j | throw s!"{what} is not a list of strings"
  xs.mapM fun
    | .str s => pure s
    | _ => throw s!"{what} is not a list of strings"

/-- `{"own", "hello", "coin_tree", "seen", "selection", "slots", "session"}` to a verdict `{"ok", "codes", "detail"}`: no
code, or the one failing check's (`hello`, `streams`, `coin_tree`, `fresh`) with its reason. -/
def check (j : Json) : V Json := do
  let own := getD j "own"
  let .obj _ := getD own "coins" | throw "the verifier's own Hello is not an object with an object coins"
  let hello ← Flock.J.str j "hello" "the run"
  let seen ← strings (getD j "seen") "seen"
  let selection ← strings (getD j "selection") "selection"
  let session ← strings (getD j "session") "session"
  let slots ← Flock.J.nat j "slots" "the run"
  return match checkRun own hello (getD j "coin_tree") seen selection slots session with
    | .ok _ => Json.mkObj [("ok", Json.bool true), ("codes", Json.arr #[]), ("detail", Json.mkObj [])]
    | .error (code, why) => Json.mkObj [("ok", Json.bool false), ("codes", Json.arr #[Json.str code]),
        ("detail", Json.mkObj [(code, Json.str why)])]

end SampledProofs.Hello
