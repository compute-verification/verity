/-
A class's place in its window's session, and the family of that session its sample reads its challenges from: both
functions of a registration record's window alone (`place`, `family`), beside the root the window's session is opened
on (`run`).

A window's session holds one family of challenges for each sample drawn in it, and a sample's evidence is read as
challenges of one family (`Challenger.drawCoins`). Which family a registration's sample reads is not its prover's to
name: it is the family of the place the record's own window names. A check that holds the family it reads to `family`
of the record, and the session's root to `run` of the record, reads the one sample the record fixes.

* **The place** of a window `w` is `((prompt · 2^32 + outputs) · 2 + stopped) · 2 + blocks`:
  * `prompt` and `outputs` are `w`'s natural numbers of those names, each below `2^32` (`SIZES`, `size?`). The bound
    keeps the places of two classes of different sizes apart;
  * `stopped` is 1 where `w`'s `stopped` is `true`, and 0 where it is anything else or absent;
  * `blocks` is 1 where `w`'s `population` is the string `stop`, as a stop population's record names it, and 0 where
    it is anything else or absent.

  There is none where a size is not a natural number below `2^32`, or where `blocks` is 1 and `stopped` is 0: a stop
  population is a stop class's.
* **The family** is `draw/<place>`, the place in decimal (`FAMILY`).
* **The run** is `w`'s `run` where it is a string.
-/
module

public import Spot.Json
meta import Spot.Json

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

/-- Whether the window `w` says its class is a stop class: its `stopped` is `true`. -/
def stopped (w : Json) : Bool :=
  match getD w "stopped" with
  | .bool b => b
  | _ => false

/-- Whether the window `w` is a stop population's: its `population` is the string `stop`. -/
def blocks (w : Json) : Bool := jsonStrIs (getD w "population") "stop"

/-- **The place the window `w` names in its session** (module docstring): none where a size is not a natural number
below `SIZES`, or where the window is a stop population's of a class that is no stop class. -/
def place (w : Json) : Option Nat :=
  match size? w "prompt", size? w "outputs" with
  | some prompt, some outputs =>
    if blocks w && !stopped w then none
    else some (((prompt * SIZES + outputs) * 2 + (stopped w).toNat) * 2 + (blocks w).toNat)
  | _, _ => none

/-- **The family the window `w`'s sample reads its challenges from**: `draw/<place>`, the place in decimal; none where
the window names no place. -/
def family (w : Json) : Option String := (place w).map fun n => s!"{FAMILY}/{n}"

/-- **The root the window `w`'s session is opened on**: its `run`, where that is a string. -/
def run (w : Json) : Option String :=
  match getD w "run" with
  | .str r => some r
  | _ => none

private def windowOf (prompt outputs : Nat) (more : List (String × Json) := []) : Json :=
  Json.mkObj ([("prompt", Lean.toJson prompt), ("outputs", Lean.toJson outputs)] ++ more)

#guard place (windowOf 4 4) == some 68719476752
#guard place (windowOf 4 4 [("stopped", Json.bool false)]) == some 68719476752
#guard place (windowOf 4 4 [("stopped", Json.bool true)]) == some 68719476754
#guard place (windowOf 4 4 [("stopped", Json.bool true), ("population", "stop")]) == some 68719476755
#guard place (windowOf 4 4 [("stopped", Json.bool true), ("population", "served")]) == some 68719476754
#guard place (windowOf 4 2) != place (windowOf 2 4)
#guard place (windowOf (2 ^ 32 - 1) (2 ^ 32 - 1)) == some (((2 ^ 32 - 1) * 2 ^ 32 + (2 ^ 32 - 1)) * 4)
/-! A stop population of a class that is no stop class, a size of `2^32`, and a size that is no natural number. -/
#guard place (windowOf 4 4 [("population", "stop")]) == none
#guard place (windowOf 4 4 [("stopped", Json.bool false), ("population", "stop")]) == none
#guard place (windowOf 4 4 [("stopped", "true"), ("population", "stop")]) == none
#guard place (windowOf (2 ^ 32) 4) == none
#guard place (windowOf 4 (2 ^ 32)) == none
#guard place (Json.mkObj [("prompt", "4"), ("outputs", Lean.toJson (4 : Nat))]) == none
#guard place (Json.mkObj [("prompt", Lean.toJson (-4 : Int)), ("outputs", Lean.toJson (4 : Nat))]) == none
#guard place (Json.mkObj [("prompt", Lean.toJson (4 : Nat))]) == none
#guard place Json.null == none
#guard family (windowOf 4 4) == some "draw/68719476752"
#guard family (windowOf 4 4 [("stopped", Json.bool true)]) == some "draw/68719476754"
#guard family (windowOf 4 4 [("stopped", Json.bool true), ("population", "stop")]) == some "draw/68719476755"
#guard family (windowOf 4 4 [("population", "stop")]) == none
#guard run (Json.mkObj [("run", "ab")]) == some "ab"
#guard run (Json.mkObj [("run", Json.null)]) == none
#guard run Json.null == none

end Spot.Place
