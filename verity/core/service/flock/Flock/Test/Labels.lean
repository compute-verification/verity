/-
`RecVStar.checkLabels` on a population of four units: a drawn table's statement must label its instances with the units its
draw proves, at one table and at two (each table's part of the draw, `tableDraw`, as `Stmt.setupTables` builds it).
-/
import Flock.RecVStar

namespace Flock.Test.Labels

open Lean (Json)
open Flock.RecVStar

def n (x : Nat) : Json := Lean.toJson x

def subset (units : List Nat) : Json := Json.mkObj [("law", "subset"), ("population", (n 4)), ("k", n units.length),
  ("units", Json.arr (units.toArray.map n))]

def stmt (labels : List Nat) : Stmt :=
  { (default : Stmt) with pub := { (default : Public) with
      header := Json.mkObj [("units", Json.mkObj [("indices", Json.arr (labels.toArray.map n))])] } }

def ok (draw : Option Json) (count : Nat) (labels : List (List Nat)) : Bool :=
  (checkLabels draw count (labels.toArray.map stmt)).isOk

#guard tableDraw (subset [0, 1, 2, 3]) 2 1 == subset [2, 3]
#guard tableDraw (subset [1, 3]) 1 0 == subset [1, 3]

#guard ok (some (subset [1, 3])) 1 [[1, 3]]
#guard !ok (some (subset [1, 3])) 1 [[0, 1]]
#guard !ok (some (subset [1, 3])) 1 [[3, 1]]
#guard !ok (some (subset [1, 3])) 1 [[1]]
#guard ok (some (subset [0, 1, 2, 3])) 2 [[0, 1], [2, 3]]
#guard !ok (some (subset [0, 1, 2, 3])) 2 [[0, 1], [0, 1]]
#guard !ok (some (subset [0, 1, 2, 3])) 2 [[0, 1], [3, 4]]
#guard ok none 1 [[5, 6]]

end Flock.Test.Labels
