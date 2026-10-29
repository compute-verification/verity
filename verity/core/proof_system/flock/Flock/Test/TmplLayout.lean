/-
`Tmpl.layout` on `Flock.Test.Tmpl`'s unit: the pair's two copies take two lanes at 128, so the unit's columns 128 to 133
sit at 128, 130, 132, 129, 131 and 133 (the map `benchmarks/verifier_cost/tmpl_layouts.py` gives, lenient and strict);
`posOf` agrees with the reference `posMap` on every column, which is a permutation. The layouts it refuses: an output
group whose start a call's lanes would move, and a call in the input rows to a Definition laid out apart.
-/
import Flock.TmplLayout
import Flock.Test.Tmpl

namespace Flock.Test.TmplLayout

open Flock Flock.Test.Tmpl

def laid (lines : Array String) (strict : Bool) : Except String (Tmpl.Tmpl × Array Tmpl.Lay) := do
  let t ← _root_.Flock.Tmpl.parse lines
  return (t, ← _root_.Flock.Tmpl.layout t strict)

def moved : List (Nat × Nat) := [(129, 130), (130, 132), (131, 129), (132, 131)]

/-- The map is the identity but for `moved`, `posOf` agrees with it, and it is a permutation. -/
def mapIs (lines : Array String) (strict : Bool) (moved : List (Nat × Nat)) : Bool :=
  match laid lines strict with
  | .ok (t, lays) =>
    let m := _root_.Flock.Tmpl.posMap t lays
    m.size == 400 && (List.range 400).all (fun x => m[x]! == (moved.lookup x).getD x) &&
      (List.range 400).all (fun x => _root_.Flock.Tmpl.posOf t lays x == m[x]!) &&
      (m.toList.mergeSort (· ≤ ·)) == List.range 400 && (lays.getD t.root default).size == 400
  | .error _ => false

/-- `Tmpl.parse` reads it and `layout` refuses it. -/
def refused (lines : Array String) : Bool :=
  (_root_.Flock.Tmpl.parse lines).toOption.isSome && (laid lines false).toOption.isNone

#guard mapIs tmplLines false moved
#guard mapIs tmplLines true moved
-- the pair's layout: its two ANDs in two lanes, its sum after them
#guard ((laid tmplLines false).toOption.map fun (_, l) => ((l.getD 1 default).size, (l.getD 1 default).align)) ==
  some (3, 2)
-- one copy keeps its columns
#guard mapIs (tmplLines "CALL 1 1 128 3 1 2 0 1 1 0") false []
-- two pairs at 253, writing the output rows: their lanes at 256 put the group's first column at 257
#guard refused (header :: and2 ++ pair ++ ["DEF 400 0 4 1", "0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2",
  "300 3 0 1 0 1 400", "CALL 1 2 253 3 1 2 0 1 1 0"]).toArray
-- the pair in the input rows: laid out apart, it can't keep its columns there
#guard refused (header :: and2 ++ pair ++ ["DEF 400 0 3 1", "256 1 130 1 399", "257 1 133 1 399",
  "300 3 0 1 0 1 400", "CALL 1 1 0 3 1 1 399 0 2 0"]).toArray
-- two pairs before the output group, pushing it from 256 to 384: accepted, the constant last at 527
#guard ((laid (tmplLines "CALL 1 2 250 3 1 2 0 1 1 0") false).toOption.map fun (t, l) =>
  ((l.getD t.root default).size, _root_.Flock.Tmpl.posOf t l 256, _root_.Flock.Tmpl.posOf t l 399)) ==
  some (528, 384, 527)

end Flock.Test.TmplLayout
