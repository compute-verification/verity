/-
`Tmpl.unroll` against `Net.parse`: a templated unit (two copies of a two-input AND, bound to the same input bits, and two
output bits) unrolls to the net `Net.parse` reads from its flat text, written out here by hand, and `Tmpl.flatLines`
prints text `Net.parse` reads as the same net. The forms `Tmpl.parse` and `Tmpl.unroll` refuse: overlapping copies, a
callee that is not an earlier definition, segments that don't cover the callee's inputs, a column written twice, a root
with inputs.
-/
import Flock.Tmpl

namespace Flock.Test.Tmpl

open Flock

def same (n m : Net) : Bool :=
  n.name == m.name && n.useful == m.useful && n.constPos == m.constPos && n.inWords == m.inWords &&
    n.unitLog == m.unitLog && n.inGroups.size == m.inGroups.size && n.outGroups.size == m.outGroups.size &&
    ((n.inGroups ++ n.outGroups).toList.zip (m.inGroups ++ m.outGroups).toList).all
      (fun (g, h) => g.col == h.col && g.words == h.words && g.bits == h.bits) &&
    [(n.a, m.a), (n.b, m.b)].all fun (x, y) =>
      x.rows == y.rows && x.width == y.width && x.data.data == y.data.data && x.start == y.start && x.len == y.len

def header : String := "flock-ir-tmpl/v1 t 1 1 0 1 1 2 1 2 1 1 2"

def and2 : List String := ["DEF 4 2 1 0", "2 1 0 1 1"]

def root (call : String) (extra : List String) : List String :=
  [s!"DEF 400 0 {4 + extra.length} 1", "0 1 0 1 0", "1 1 1 1 1", "256 1 130 1 399", "257 1 134 1 399"] ++ extra ++ [call]

def tmplLines (call : String := "CALL 0 2 128 4 1 2 0 0") (extra : List String := []) : Array String :=
  (header :: and2 ++ root call extra).toArray

/-- The flat unit, by hand: input bits 0 and 1, copy `j`'s binding rows at `128 + 4j` and `129 + 4j`, its AND at
`130 + 4j`, its callee constant's column `131 + 4j` empty, the outputs at 256 and 257, the constant at 399. -/
def flatText : Array String := Id.run do
  let rows : List (Nat × String) :=
    [(0, "1 0 1 0"), (1, "1 1 1 1"), (128, "1 0 1 399"), (129, "1 1 1 399"), (130, "1 128 1 129"),
     (132, "1 0 1 399"), (133, "1 1 1 399"), (134, "1 132 1 133"), (256, "1 130 1 399"), (257, "1 134 1 399"),
     (399, "1 399 1 399")]
  let mut out := #["flock-ir-unit/v2 t 400 399 1 0 1 1 2 1 2 1 1 2"]
  for i in [0:400] do out := out.push ((rows.lookup i).getD "0 0")
  return out

def unrolled (lines : Array String) : Except String Net := do
  _root_.Flock.Tmpl.unroll (← _root_.Flock.Tmpl.parse lines)

def agrees : Bool :=
  match unrolled tmplLines, Net.parse flatText with
  | .ok n, .ok m => same n m
  | _, _ => false

def roundTrip : Bool :=
  match _root_.Flock.Tmpl.parse tmplLines >>= _root_.Flock.Tmpl.flatLines, unrolled tmplLines with
  | .ok ls, .ok n => match Net.parse ls with
    | .ok m => same n m
    | .error _ => false
  | _, _ => false

def refuses (lines : Array String) : Bool := (unrolled lines).toOption.isNone

#guard (Net.parse flatText).toOption.isSome
#guard agrees
#guard roundTrip
-- copies at stride 2 overlap a callee of 4 columns
#guard refuses (tmplLines "CALL 0 2 128 2 1 2 0 0")
-- slots not aligned to the stride
#guard refuses (tmplLines "CALL 0 2 130 4 1 2 0 0")
-- a callee that is not an earlier definition
#guard refuses (tmplLines "CALL 1 2 128 4 1 2 0 0")
-- segments binding one input of two
#guard refuses (tmplLines "CALL 0 2 128 4 1 1 0 0")
-- the copies' rows meet the root's row at 130
#guard refuses (tmplLines (extra := ["130 1 0 1 1"]))
-- a copy over the root's output rows
#guard refuses (tmplLines "CALL 0 2 252 4 1 2 0 0")
-- a root with inputs
#guard refuses #[header, "DEF 4 2 1 0", "2 1 0 1 1"]

end Flock.Test.Tmpl
