/-
A statement's public file's units, its header's `units.indices` (`indicesOf`), and the rule that they are exactly given
units in ascending order (`exactly`): slot `i` of the file's table is the `i`-th of those units. `exactly #[(0, N)]` is
a file of the registration's units `0, …, N − 1` in order, which v2 holds the parameters' public file to
(`OneStageZk.unitsHeld`), so the units its session proves are the registration's. O(the ranges + the file's units),
which the setup reads already.
-/
module

public import SampledProofs.Json
meta import SampledProofs.Json

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace SampledProofs.PublicUnits

open Lean (Json)
open Flock (V)

/-- A public file's `units.indices`, from its header `h`. -/
def indicesOf (h : Json) : V (Array Nat) := do
  let .arr is := getD (getD h "units") "indices"
    | throw "the public file's header names no units.indices"
  is.mapM fun i => match nat? i with
    | some u => pure u
    | none => throw "the public file's units.indices is not a list of natural numbers"

/-- Whether `indices` are exactly the units of the `[lo, hi)` ranges `units`, ascending: the ranges in order of their
first unit (then their last), each range's units in order. -/
def exactly (units : Array (Nat × Nat)) (indices : Array Nat) : Bool := Id.run do
  let mut i := 0
  for (lo, hi) in units.qsort fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2) do
    for u in [lo:hi] do
      if indices[i]? != some u then return false
      i := i + 1
  return i == indices.size

private def nats (j : Json) (what : String) : V (Array Nat) := do
  let .arr xs := j | throw s!"{what} is not a list of natural numbers"
  xs.mapM fun x => match nat? x with
    | some n => pure n
    | none => throw s!"{what} is not a list of natural numbers"

/-- `{"units", "indices"}`, a list of `[lo, hi]` with `0 ≤ lo < hi` and a list of natural numbers, to `{"exactly"}`
(`one_stage.proofs.units_exactly`); refused where Python raises. -/
def piece (j : Json) : V Json := do
  let .arr rs := getD j "units" | throw "units is not a list of [lo, hi] with 0 <= lo < hi"
  let units ← rs.mapM fun r => do
    let p ← nats r "a range"
    match p.toList with
    | [lo, hi] => if lo < hi then pure (lo, hi) else throw "units is not a list of [lo, hi] with 0 <= lo < hi"
    | _ => throw "units is not a list of [lo, hi] with 0 <= lo < hi"
  return Json.mkObj [("exactly", Json.bool (exactly units (← nats (getD j "indices") "indices")))]

#guard exactly #[(0, 2)] #[0, 1]
#guard exactly #[(5, 7), (2, 4)] #[2, 3, 5, 6]
#guard !exactly #[(0, 2)] #[1, 2]
#guard !exactly #[(0, 2)] #[1, 0]
#guard !exactly #[(0, 2)] #[0]
#guard !exactly #[(0, 2)] #[0, 1, 2]
#guard exactly #[] #[]
#guard exactly #[(0, 0)] #[]
#guard (indicesOf (Json.mkObj [("units", Json.mkObj [("indices", Json.arr #[3, 4])])])).toOption == some #[3, 4]
#guard (indicesOf (Json.mkObj [("units", Json.mkObj [("indices", Json.arr #[3, .num ⟨-1, 0⟩])])])).toOption.isNone
#guard (indicesOf (Json.mkObj [("units", Json.mkObj [("indices", Json.arr #[3, .num ⟨5, 1⟩])])])).toOption.isNone
#guard (indicesOf (Json.mkObj [("units", Json.mkObj [])])).toOption.isNone

end SampledProofs.PublicUnits
