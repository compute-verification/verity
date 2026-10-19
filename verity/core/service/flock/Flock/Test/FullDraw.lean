/-
`RecSetup.fullDraw` on a population of four units, under F1's inner shape (m 27, k_log 24, claims 8 over 3 regions):
the `subset` draw of all four passes, and a draw of fewer units, a stratified draw of all four and a subset draw with a
closure are refused; with no draw, or no inner shape outside a recursive session, it passes whatever the record holds.
A recursive session (`--rec-part`, `--rec-bindings` or `--rec-rows`) without an inner shape is refused, whatever its
draw. `RecSetup.innerDrawn`: an inner session's record whose `unit_draw` is absent or null, or that is not a JSON object,
is refused, from the inner statement's files or the verifier's inner coins alike; one with a draw passes, and so does a
session without the inner record.
-/
import Flock.RecSetup

namespace Flock.Test.FullDraw

open Lean (Json)
open Flock

def n (x : Nat) : Json := Lean.toJson x

def nats (xs : List Nat) : Json := Json.arr (xs.toArray.map n)

def subset (units : List Nat) : Json := Json.mkObj [("law", "subset"), ("population", n 4), ("k", n units.length),
  ("units", nats units)]

def stratified : Json := Json.mkObj [("law", "stratified"), ("population", n 4), ("k", n 4),
  ("strata", Json.arr #[Json.mkObj [("template", "t"), ("units", Json.arr #[nats [0, 4]]), ("k", n 4)]]),
  ("units", nats [0, 1, 2, 3])]

def withClosure : Json := Json.mkObj [("law", "subset"), ("population", n 4), ("k", n 3), ("units", nats [0, 1, 2]),
  ("closure", nats [3])]

def inner : RecVStar.Inner := { m := 27, kLog := 24, claims := 8, regions := 3 }

def record (d : Json) : String := (Json.mkObj [("unit_draw", d)]).compress

def why (i : Option RecVStar.Inner) (d : Option Json) (rec : Bool := true) : Option String :=
  match RecSetup.fullDraw i rec d (record (d.getD Json.null)) with
  | .ok () => none
  | .error e => some e

def tail : String := ": a V* session draws every unit of its population, the only draw the recursion's proof covers"

#guard why inner (some (subset [0, 1, 2, 3])) == none
#guard why inner (some (subset [1, 3])) == some ("refused: the draw is subset:2 of 2 units over 4, not every one of 4" ++ tail)
#guard why inner (some (subset [0, 1, 2])) == some ("refused: the draw is subset:3 of 3 units over 4, not every one of 4" ++ tail)
#guard why inner (some stratified) == some ("refused: the draw's law is not subset" ++ tail)
#guard why inner (some withClosure) == some "U2: unit_draw's keys are not law, population, k, units"
#guard why (some { inner with holo := true }) (some (subset [0, 1])) ==
  some ("refused: the draw is subset:2 of 2 units over 4, not every one of 4" ++ tail)
#guard why inner none == none
#guard why inner (some (subset [1, 3])) (rec := false) ==
  some ("refused: the draw is subset:2 of 2 units over 4, not every one of 4" ++ tail)
#guard why none (some (subset [1, 3])) (rec := false) == none
#guard why none none (rec := false) == none
#guard why none (some (subset [0, 1, 2, 3])) == some RecSetup.noInner
#guard why none (some (subset [1, 3])) == some RecSetup.noInner
#guard why none none == some RecSetup.noInner
#guard RecSetup.noInner.startsWith "refused: a recursive session (--rec-part, --rec-bindings or --rec-rows) without"

def files (r : String) : RecCoins.Source := .files { circuit := .empty, pub := .empty, partition := Json.null, record := r }

def innerWhy (src : Option RecCoins.Source) : Option String :=
  match RecSetup.innerDrawn src with
  | .ok () => none
  | .error e => some e

#guard innerWhy (some (files (record (subset [0, 1, 2, 3])))) == none
#guard innerWhy (some (.coins Json.null (record (subset [1])))) == none
#guard innerWhy (some (files (record Json.null))) == some RecSetup.noInnerDraw
#guard innerWhy (some (.coins Json.null (record Json.null))) == some RecSetup.noInnerDraw
#guard innerWhy (some (files (Json.mkObj [("verdict", Json.null)]).compress)) == some RecSetup.noInnerDraw
#guard innerWhy (some (files "not json")) == some RecSetup.noInnerDraw
#guard innerWhy (some (files "[]")) == some RecSetup.noInnerDraw
#guard innerWhy none == none
#guard RecSetup.noInnerDraw.startsWith "refused: the inner session's record (--rec-inner-session) holds no draw"

end Flock.Test.FullDraw
