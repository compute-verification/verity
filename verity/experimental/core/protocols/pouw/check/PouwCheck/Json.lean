import Lean.Data.Json
import Flock.Canon
import PouwWindow.Json

/-!
# JSON as `core_check.py` reads it

An object's keys sorted (Python's `sorted(dict)`), a string, a value's canonical string (`_canon`, which the check
compares registered items by), and a field of an object whose keys were already checked.
-/

namespace PouwCheck

open Lean (Json)

/-- `sorted(j)` for an object, `none` for anything else. -/
def keys : Json → Option (List String)
  | .obj kvs => some ((kvs.toArray.map (·.1)).qsort (· < ·)).toList
  | _ => none

def str? : Json → Option String
  | .str s => some s
  | _ => none

/-- `_nat`: a natural number written as an integer. -/
def nat? (j : Json) : Option Nat := PouwWindow.natOf? j

/-- A field, `null` when absent (read only after `keys` has fixed which are present). -/
def get (j : Json) (k : String) : Json := (PouwWindow.field? j k).getD .null

/-- `_canon`: the value's canonical string. Every value read here came from canonical bytes, where `Flock.canon` doesn't
fail; the failure's message stands in for a string it can't write. -/
def cs (j : Json) : String :=
  match Flock.canon j with
  | .ok s => s
  | .error e => e

end PouwCheck
