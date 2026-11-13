/-
A class's place in its window's session, and the family of that session its sample reads its challenges from: both
functions of a registration record's window alone (`place`, `family`), beside the root the window's session is opened
on (`run`).

A window's session holds one family of challenges for each sample drawn in it, and a sample's evidence is read as
challenges of one family (`Challenger.drawCoins`). Which family a registration's sample reads is not its prover's to
name: it is the family of the place the record's own window names. A check that holds the family it reads to `family`
of the record, and the session's root to `run` of the record, reads the one sample the record fixes.

* **The place** of a window `w` is `((prompt · 2^32 + outputs) · 2 + stopped) · 2 + blocks`, read from four items of
  `w` and no other:
  * `prompt` and `outputs` are `w`'s natural numbers of those names, each below `2^32` (`SIZES`, `size?`). The bound
    keeps the places of two classes of different sizes apart;
  * `stopped` is 1 where `w`'s `stopped` is `true`, and 0 where it is `false` or `w` names none (`stopped?`);
  * `blocks` is 1 where `w`'s `population` is the string `stop`, as a stop population's record names it, and 0 where
    `w` names none (`blocks?`).

  A window with any other word in one of the four has no place: a size that is not a natural number below `2^32`, a
  `stopped` it names that is neither `true` nor `false`, a `population` it names that is not `stop`. An item named
  `null` is named: it is not an absent one. And there is none where `blocks` is 1 and `stopped` is 0: a stop
  population is a stop class's.
* **The family** is `draw/<place>`, the place in decimal (`FAMILY`).
* **The run** is `w`'s `run` where it is a string.
-/
module

public import Spot.Json
meta import Spot.Json
import Lean.Data.Json.Parser
meta import Lean.Data.Json.Parser

@[expose] public section

namespace Spot.Place

open Lean (Json)

/-- A class's two sizes are each below this. -/
def SIZES : Nat := 2 ^ 32

/-- The name the families of a window's samples begin with. -/
def FAMILY : String := "draw"

/-- The size the window `w` names under `k`: a natural number below `SIZES`. -/
def size? (w : Json) (k : String) : Option Nat :=
  match nat? (getD w k) with
  | some n => if n < SIZES then some n else none
  | none => none

/-- Whether the window `w` says its class is a stop class: its `stopped` where that is `true` or `false`, and `false`
where it names none. None where it names anything else, `null` included. -/
def stopped? (w : Json) : Option Bool :=
  match field? w "stopped" with
  | none => some false
  | some (.bool b) => some b
  | some _ => none

/-- Whether the window `w` is a stop population's: `true` where its `population` is the string `stop`, and `false`
where it names none. None where it names anything else, `null` included. -/
def blocks? (w : Json) : Option Bool :=
  match field? w "population" with
  | none => some false
  | some p => if jsonStrIs p "stop" then some true else none

/-- **The place the window `w` names in its session** (module docstring): none where a size is not a natural number
below `SIZES`, where the window names a `stopped` that is neither `true` nor `false` or a `population` that is not
`stop`, or where the window is a stop population's of a class that is no stop class. -/
def place (w : Json) : Option Nat :=
  match size? w "prompt", size? w "outputs", stopped? w, blocks? w with
  | some prompt, some outputs, some stopped, some blocks =>
    if blocks && !stopped then none
    else some (((prompt * SIZES + outputs) * 2 + stopped.toNat) * 2 + blocks.toNat)
  | _, _, _, _ => none

/-- **The family the window `w`'s sample reads its challenges from**: `draw/<place>`, the place in decimal; none where
the window names no place. -/
def family (w : Json) : Option String := (place w).map fun n => s!"{FAMILY}/{n}"

/-- **The root the window `w`'s session is opened on**: its `run`, where that is a string. -/
def run (w : Json) : Option String :=
  match getD w "run" with
  | .str r => some r
  | _ => none

/-- The place of the window the JSON text `s` reads as, as `Json.parse` reads it. -/
private def placeOf (s : String) : Option (Option Nat) := (Json.parse s).toOption.map place

/-! The rows the Python definition's place is held to, each a record's window as text: the windows that name a
place. -/
#guard placeOf "{\"prompt\": 4, \"outputs\": 4}" == some (some 68719476752)
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": false}" == some (some 68719476752)
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": true}" == some (some 68719476754)
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": true, \"population\": \"stop\"}" ==
  some (some 68719476755)
#guard placeOf "{\"prompt\": 4, \"outputs\": 2}" == some (some 68719476744)
#guard placeOf "{\"prompt\": 2, \"outputs\": 8}" == some (some 34359738400)
#guard placeOf "{\"prompt\": 8, \"outputs\": 2}" == some (some 137438953480)
#guard placeOf "{\"prompt\": 0, \"outputs\": 0}" == some (some 0)
#guard placeOf "{\"prompt\": 0, \"outputs\": 1}" == some (some 4)
#guard placeOf "{\"prompt\": 1, \"outputs\": 0}" == some (some 17179869184)
#guard placeOf "{\"prompt\": 4294967295, \"outputs\": 4294967295, \"stopped\": true, \"population\": \"stop\"}" ==
  some (some (2 ^ 66 - 1))
/-! The windows that name none: a size of `2^32`, below zero, a string, a number that is not whole, a boolean, `null`
or absent; a `stopped` that is neither `true` nor `false`; the stop blocks of a class that is no stop class; a
`population` that is not `stop`. -/
#guard placeOf "{\"prompt\": 4294967296, \"outputs\": 4}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4294967296}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": -1}" == some none
#guard placeOf "{\"prompt\": \"4\", \"outputs\": 4}" == some none
#guard placeOf "{\"prompt\": 4.0, \"outputs\": 4}" == some none
#guard placeOf "{\"prompt\": true, \"outputs\": 4}" == some none
#guard placeOf "{\"prompt\": null, \"outputs\": 4}" == some none
#guard placeOf "{\"outputs\": 4}" == some none
#guard placeOf "{\"prompt\": 4}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": 1}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": \"true\"}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": null}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"population\": \"stop\"}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": false, \"population\": \"stop\"}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": true, \"population\": \"spans\"}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": true, \"population\": null}" == some none
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"stopped\": true, \"population\": 1}" == some none
/-! No other item of a window moves its place, and a value that is no object names none. -/
#guard placeOf "{\"prompt\": 4, \"outputs\": 4, \"run\": \"ab\", \"row\": 3, \"verdicts\": true}" ==
  some (some 68719476752)
#guard placeOf "[4, 4]" == some none
#guard place Json.null == none
/-! A number written with an exponent is the whole number it denotes where it denotes one, as `Json.parse` reads it:
`4e0` is 4. A record's canonical text writes no number so. -/
#guard placeOf "{\"prompt\": 4e0, \"outputs\": 4}" == some (some 68719476752)
#guard placeOf "{\"prompt\": 40e-1, \"outputs\": 4}" == some none

/-- A window of the sizes `prompt` and `outputs`, with `more`. -/
private def windowOf (prompt outputs : Nat) (more : List (String × Json) := []) : Json :=
  Json.mkObj ([("prompt", Lean.toJson prompt), ("outputs", Lean.toJson outputs)] ++ more)

#guard family (windowOf 4 4) == some "draw/68719476752"
#guard family (windowOf 4 4 [("stopped", Json.bool true)]) == some "draw/68719476754"
#guard family (windowOf 4 4 [("stopped", Json.bool true), ("population", "stop")]) == some "draw/68719476755"
#guard family (windowOf 4 4 [("population", "stop")]) == none
#guard family (windowOf 4 4 [("stopped", Json.null)]) == none
#guard run (Json.mkObj [("run", "ab")]) == some "ab"
#guard run (Json.mkObj [("run", Json.null)]) == none
#guard run Json.null == none

/-- What `piece` refuses a window with. -/
def NO_PLACE : String := "the window names no place in its session"

/-- The `place` piece: a window to its place, its family and its run (null: it names none). Refused: a window that
names no place. -/
def piece (w : Json) : Except String Json :=
  match place w, family w with
  | some n, some f =>
    .ok (Json.mkObj [("place", Lean.toJson n), ("family", .str f),
      ("run", match run w with | some r => .str r | none => .null)])
  | _, _ => .error NO_PLACE

end Spot.Place
